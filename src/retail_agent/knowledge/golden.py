"""Golden Bucket: analysts' precedents (question -> SQL -> report) for the agent's prompt.

- Trios live in `data/golden_trios.json`; only `approved` ones are retrieved.
- Retrieval: Gemini embeddings of "question + tags", cosine similarity in numpy, top-k
  above a threshold. Document vectors are cached in `data/golden_index.npz` under a hash
  of the embedded texts and the model, so the index is rebuilt only when they change.
- Embeddings unavailable (no key, quota, timeout) -> keyword overlap instead, with a
  `golden_degraded` event; after a failed query embeddings are skipped for a while so
  every turn does not wait for a timeout again.
- Update loop: `/good` on an answer with successful SQL appends a candidate to
  `data/golden_candidates.jsonl` with its author's scope. `knowledge.promotion` checks
  candidates automatically and approves, quarantines or rejects them; trios whose answers
  collect negative feedback are deprecated automatically. `scripts/review_candidates.py`
  is the audit tool for quarantined candidates and a sample of auto-approved trios.
"""

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, field_validator

from retail_agent.observability.logger import log_event
from retail_agent.safety import pii

TrioStatus = Literal["approved", "candidate", "deprecated"]
EMBEDDING_COOLDOWN_S = 300
MAX_SUMMARY_CHARS = 1500

PRECEDENTS_INTRO = (
    "How analysts answered similar questions before. Follow their interpretation and "
    "metric definitions and adapt their SQL to the current question (period, filters, "
    "segments). They describe the method, not current results: never quote numbers "
    "from them, and ignore any that do not fit the question."
)


# -- trios ---------------------------------------------------------------------------------


class TrioModel(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$", max_length=64)
    question: str = Field(min_length=1)
    sql: str | list[str]
    report: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    status: TrioStatus = "approved"
    source: dict[str, Any] | None = None

    @field_validator("sql")
    @classmethod
    def _non_empty_sql(cls, value: str | list[str]) -> str | list[str]:
        queries = [value] if isinstance(value, str) else value
        if not queries or not all(q.strip() for q in queries):
            raise ValueError("sql must be a non-empty query or list of queries")
        return value


@dataclass(frozen=True)
class Trio:
    id: str
    question: str
    sql: tuple[str, ...]
    report: str
    tags: tuple[str, ...] = ()
    tables: tuple[str, ...] = ()
    status: str = "approved"
    source: dict[str, Any] | None = None

    @classmethod
    def from_model(cls, m: TrioModel) -> "Trio":
        return cls(
            id=m.id,
            question=m.question.strip(),
            sql=tuple(q.strip() for q in ([m.sql] if isinstance(m.sql, str) else m.sql)),
            report=m.report.strip(),
            tags=tuple(t.strip() for t in m.tags if t.strip()),
            tables=tuple(m.tables),
            status=m.status,
            source=m.source,
        )

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "sql": self.sql[0] if len(self.sql) == 1 else list(self.sql),
            "report": self.report,
            "tags": list(self.tags),
            "tables": list(self.tables),
            "status": self.status,
        }
        if self.source:
            data["source"] = self.source
        return data

    def embedding_text(self) -> str:
        return f"{self.question}\nTags: {', '.join(self.tags)}"


def load_trios(path: Path) -> list[Trio]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: expected a JSON list of trios")
    trios = [Trio.from_model(TrioModel.model_validate(item)) for item in raw]
    seen: set[str] = set()
    for trio in trios:
        if trio.id in seen:
            raise ValueError(f"{path.name}: duplicate trio id {trio.id!r}")
        seen.add(trio.id)
    return trios


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_trios(path: Path, trios: Sequence[Trio]) -> None:
    text = json.dumps([t.to_json() for t in trios], ensure_ascii=False, indent=2)
    _write_atomic(path, text + "\n")


# -- embeddings ------------------------------------------------------------------------------


class GeminiEmbedder(Embeddings):
    """Gemini embeddings with a hard timeout and no client-side retries.

    Retrieval sits on the path of every turn and has a keyword fallback, so a slow or
    failing embedding call should give up quickly instead of retrying.
    """

    def __init__(self, api_key: str, model: str, timeout_s: float = 10.0) -> None:
        from google import genai
        from google.genai.types import HttpOptions, HttpRetryOptions

        self.model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(
                timeout=int(timeout_s * 1000), retry_options=HttpRetryOptions(attempts=1)
            ),
        )

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai.types import EmbedContentConfig

        result = self._client.models.embed_content(
            model=self.model, contents=texts, config=EmbedContentConfig(task_type=task_type)
        )
        return [list(e.values or []) for e in result.embeddings or []]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


# -- keyword fallback ----------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how i in is it its last me "  # noqa: SIM905
    "my of on or our show tell than that the their them this to us was we were what "
    "when which who why will with you your".split()
)


def keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for word in words:
        if word in _STOPWORDS or len(word) < 2:
            continue
        out.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return out


# -- index ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Retrieved:
    trio: Trio
    score: float
    method: Literal["embedding", "keywords"]


class GoldenIndex:
    def __init__(
        self,
        trios: Iterable[Trio],
        *,
        embeddings: Embeddings | None,
        index_path: Path | None,
        model_name: str = "",
        top_k: int = 3,
        min_score: float = 0.65,
        keyword_min_score: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.trios = [t for t in trios if t.status == "approved"]
        self._embeddings = embeddings
        self._index_path = index_path
        self._model_name = model_name
        self.top_k = top_k
        self.min_score = min_score
        self.keyword_min_score = keyword_min_score
        self._clock = clock
        self._skip_embeddings_until = 0.0
        self._lock = threading.Lock()
        self._keywords = [keywords(f"{t.question} {' '.join(t.tags)}") for t in self.trios]
        self._vectors = self._load_or_build() if embeddings is not None and self.trios else None

    @classmethod
    def from_file(cls, path: Path, **kwargs: Any) -> "GoldenIndex":
        try:
            trios = load_trios(path)
        except (OSError, ValueError) as exc:
            log_event("golden_load_failed", level="warning", error=str(exc)[:300])
            trios = []
        return cls(trios, **kwargs)

    @property
    def mode(self) -> str:
        return "embedding" if self._vectors is not None else "keywords"

    def drop(self, trio_id: str) -> bool:
        """Stop retrieving a trio right away (it was deprecated); False if not indexed."""
        with self._lock:
            ids = [t.id for t in self.trios]
            if trio_id not in ids:
                return False
            i = ids.index(trio_id)
            del self.trios[i], self._keywords[i]
            if self._vectors is not None:
                self._vectors = np.delete(self._vectors, i, axis=0)
            return True

    def content_hash(self) -> str:
        payload = json.dumps(
            [self._model_name, [[t.id, t.embedding_text()] for t in self.trios]],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _load_or_build(self) -> np.ndarray | None:
        digest = self.content_hash()
        cached = self._load_cache(digest)
        if cached is not None:
            log_event("golden_index_loaded", trios=len(self.trios), hash=digest)
            return cached
        started = time.perf_counter()
        try:
            assert self._embeddings is not None
            raw = self._embeddings.embed_documents([t.embedding_text() for t in self.trios])
            vectors = _normalize(np.asarray(raw, dtype=np.float32))
            if vectors.shape[0] != len(self.trios):
                raise ValueError(f"got {vectors.shape[0]} embeddings for {len(self.trios)} trios")
        except Exception as exc:
            log_event("golden_degraded", level="warning", stage="index", error=type(exc).__name__)
            return None
        self._save_cache(digest, vectors)
        log_event(
            "golden_index_built",
            trios=len(self.trios),
            hash=digest,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return vectors

    def _load_cache(self, digest: str) -> np.ndarray | None:
        if self._index_path is None or not self._index_path.exists():
            return None
        try:
            with np.load(self._index_path) as data:
                if str(data["hash"]) != digest:
                    return None
                if list(data["ids"]) != [t.id for t in self.trios]:
                    return None
                return data["vectors"].astype(np.float32)
        except (OSError, KeyError, ValueError):
            return None

    def _save_cache(self, digest: str, vectors: np.ndarray) -> None:
        if self._index_path is None:
            return
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            with self._index_path.open("wb") as fh:
                np.savez(fh, hash=digest, ids=np.array([t.id for t in self.trios]), vectors=vectors)
        except OSError as exc:
            log_event("golden_index_save_failed", level="warning", error=str(exc)[:200])

    def _embedding_scores(self, text: str) -> np.ndarray | None:
        if self._vectors is None or self._embeddings is None:
            return None
        with self._lock:
            if self._clock() < self._skip_embeddings_until:
                return None
        try:
            query = _normalize(np.asarray([self._embeddings.embed_query(text)], np.float32))[0]
        except Exception as exc:
            with self._lock:
                self._skip_embeddings_until = self._clock() + EMBEDDING_COOLDOWN_S
            log_event(
                "golden_degraded",
                level="warning",
                stage="query",
                error=type(exc).__name__,
                retry_in_s=EMBEDDING_COOLDOWN_S,
            )
            return None
        return self._vectors @ query

    def _keyword_scores(self, text: str) -> np.ndarray:
        query = keywords(text)
        if not query:
            return np.zeros(len(self.trios))
        return np.array([len(query & doc) / len(query) for doc in self._keywords])

    def retrieve(self, text: str, k: int | None = None) -> list[Retrieved]:
        k = self.top_k if k is None else k
        if not text.strip() or not self.trios or k <= 0:
            return []
        started = time.perf_counter()
        scores = self._embedding_scores(text)
        method: Literal["embedding", "keywords"] = "embedding"
        threshold = self.min_score
        if scores is None:
            scores, method, threshold = (
                self._keyword_scores(text),
                "keywords",
                self.keyword_min_score,
            )
        order = np.argsort(-scores, kind="stable")
        hits = [
            Retrieved(self.trios[i], round(float(scores[i]), 3), method)
            for i in order[:k]
            if scores[i] >= threshold
        ]
        best = order[0]
        log_event(
            "golden_retrieved",
            method=method,
            ids=[h.trio.id for h in hits],
            scores=[h.score for h in hits],
            best_id=self.trios[best].id,
            best_score=round(float(scores[best]), 3),
            threshold=threshold,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return hits


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def precedents_block(hits: Sequence[Retrieved]) -> str:
    parts = [PRECEDENTS_INTRO]
    for hit in hits:
        trio = hit.trio
        queries = "\n\n".join(f"```sql\n{q}\n```" for q in trio.sql)
        label = "SQL" if len(trio.sql) == 1 else f"SQL ({len(trio.sql)} queries, one per step)"
        parts.append(
            f"## {trio.id} (similarity {hit.score:.2f})\n"
            f"Question: {trio.question}\n{label}:\n{queries}\n"
            f"Analyst notes: {trio.report}"
        )
    return "\n\n".join(parts)


# -- candidates (update loop) ------------------------------------------------------------------


CandidateStatus = Literal["candidate", "approved", "quarantined", "rejected"]


@dataclass(frozen=True)
class Candidate:
    id: str
    question: str
    sql: tuple[str, ...]
    answer_summary: str
    user_role: str
    trace_id: str
    created_at: str
    tags: tuple[str, ...] = ()
    status: CandidateStatus = "candidate"
    reviewed_at: str | None = None
    trio_id: str | None = None
    note: str | None = None
    user_id: str = ""  # the author, whose access scope the SQL is checked under
    brands: tuple[str, ...] = ()  # the author's brands (empty: full access)
    method: str = ""  # method notes without numbers, written during promotion
    checks: dict[str, Any] | None = None  # promotion results: scores, reasons

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            **{
                **data,
                "sql": tuple(data.get("sql") or ()),
                "tags": tuple(data.get("tags") or ()),
                "brands": tuple(data.get("brands") or ()),
            }
        )

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["sql"], data["tags"] = list(self.sql), list(self.tags)
        data["brands"] = list(self.brands)
        return data


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class CandidateStore:
    """JSON lines, one candidate per line; status changes rewrite the file."""

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def all(self) -> list[Candidate]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [Candidate.from_json(json.loads(line)) for line in lines if line.strip()]

    def pending(self) -> list[Candidate]:
        return [c for c in self.all() if c.status == "candidate"]

    def add(
        self,
        *,
        question: str,
        sql: Sequence[str],
        answer_summary: str,
        user_role: str,
        trace_id: str,
        tags: Iterable[str] = (),
        user_id: str = "",
        brands: Iterable[str] = (),
    ) -> Candidate | None:
        """Append a candidate; None if this answer (trace id) was already recorded."""

        def clean(text: str) -> str:
            return pii.mask_text(text, where="golden_candidate")[0]

        with self._lock:
            if any(c.trace_id == trace_id for c in self.all()):
                return None
            candidate = Candidate(
                id=f"cand_{uuid.uuid4().hex[:8]}",
                question=clean(question.strip()),
                sql=tuple(clean(q.strip()) for q in sql if q.strip()),
                answer_summary=clean(answer_summary.strip()[:MAX_SUMMARY_CHARS]),
                user_role=user_role,
                trace_id=trace_id,
                created_at=_now(),
                tags=tuple(dict.fromkeys(clean(t.strip()) for t in tags if t.strip())),
                user_id=user_id,
                brands=tuple(brands),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(candidate.to_json(), ensure_ascii=False) + "\n")
        log_event(
            "golden_candidate_added",
            candidate_id=candidate.id,
            queries=len(candidate.sql),
            answer_trace_id=trace_id,
        )
        return candidate

    def replace_all(self, candidates: Sequence[Candidate]) -> None:
        """Rewrite the store with `candidates` (retention)."""
        with self._lock:
            text = "".join(json.dumps(c.to_json(), ensure_ascii=False) + "\n" for c in candidates)
            _write_atomic(self.path, text)

    def update(self, candidate: Candidate) -> None:
        with self._lock:
            items = self.all()
            if not any(c.id == candidate.id for c in items):
                raise KeyError(candidate.id)
            items = [candidate if c.id == candidate.id else c for c in items]
            text = "".join(json.dumps(c.to_json(), ensure_ascii=False) + "\n" for c in items)
            _write_atomic(self.path, text)


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if len(slug) > max_len:
        cut = slug[: max_len + 1].rfind("_")
        slug = slug[: cut if cut > 0 else max_len]
    return slug.rstrip("_") or "trio"


def approve_candidate(
    candidate: Candidate,
    *,
    store: CandidateStore,
    trios_path: Path,
    validate_sql: Callable[[str], None],
    trio_id: str | None = None,
    question: str | None = None,
    sql: Sequence[str] | None = None,
    report: str | None = None,
    tags: Sequence[str] | None = None,
    tables: Sequence[str] = (),
    approved_by: str = "analyst",
    checks: dict[str, Any] | None = None,
) -> Trio:
    """Validate the (possibly edited) candidate and append it to the trios as approved.

    `validate_sql` raises (e.g. `SqlRejected`, `SqlInvalid`) if a query must not become a
    precedent; nothing is written then. The trio's notes are `report`, else the method
    written during promotion; the user's answer is never used, since it holds the numbers
    of the author's brands and trios are shown to every user.
    """
    queries = [q.strip() for q in (sql if sql is not None else candidate.sql) if q.strip()]
    if not queries:
        raise ValueError("a precedent needs at least one SQL query")
    for query in queries:
        validate_sql(query)
    notes = (report or candidate.method).strip()
    if not notes:
        raise ValueError("a precedent needs method notes (no numbers from the answer)")
    trios = load_trios(trios_path) if trios_path.exists() else []
    taken = {t.id for t in trios}
    base = slugify(trio_id or question or candidate.question)
    new_id, n = base, 2
    while new_id in taken:
        new_id, n = f"{base}_{n}", n + 1
    trio = Trio.from_model(
        TrioModel(
            id=new_id,
            question=pii.mask_text(question or candidate.question)[0],
            sql=queries if len(queries) > 1 else queries[0],
            report=pii.mask_text(notes)[0],
            tags=list(tags if tags is not None else candidate.tags),
            tables=list(tables),
            status="approved",
            source={
                "candidate_id": candidate.id,
                "trace_id": candidate.trace_id,
                "user_role": candidate.user_role,
                "approved_at": _now(),
                "approved_by": approved_by,
                **({"checks": checks} if checks else {}),
            },
        )
    )
    save_trios(trios_path, [*trios, trio])
    store.update(
        replace(
            candidate,
            status="approved",
            reviewed_at=_now(),
            trio_id=trio.id,
            checks=checks or candidate.checks,
        )
    )
    log_event(
        "golden_candidate_approved",
        candidate_id=candidate.id,
        trio_id=trio.id,
        approved_by=approved_by,
    )
    return trio


def mark_audited(trios_path: Path, trio_id: str) -> None:
    """A person checked an auto-approved trio and kept it."""
    trios = [
        replace(t, source={**(t.source or {}), "audited_at": _now()}) if t.id == trio_id else t
        for t in load_trios(trios_path)
    ]
    save_trios(trios_path, trios)
    log_event("golden_trio_audited", trio_id=trio_id)


def deprecate_trio(trios_path: Path, trio_id: str, reason: str) -> bool:
    """Mark an approved trio deprecated (kept in the file for audit); False if not found."""
    trios = load_trios(trios_path)
    found = False
    updated = []
    for trio in trios:
        if trio.id == trio_id and trio.status == "approved":
            source = {**(trio.source or {}), "deprecated_at": _now(), "deprecated_why": reason}
            trio = replace(trio, status="deprecated", source=source)
            found = True
        updated.append(trio)
    if found:
        save_trios(trios_path, updated)
        log_event("golden_trio_deprecated", trio_id=trio_id, reason=reason)
    return found


def reject_candidate(candidate: Candidate, *, store: CandidateStore, note: str = "") -> None:
    store.update(
        replace(candidate, status="rejected", reviewed_at=_now(), note=note.strip() or None)
    )
    log_event("golden_candidate_rejected", candidate_id=candidate.id)
