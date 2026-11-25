"""Automatic promotion of Golden Bucket candidates, and automatic deprecation of trios.

Human review does not scale, so a candidate (an answer the user rated 👍 that ran
successful SQL) is judged by a pipeline of checks instead; people only audit a sample.

1. **Rate limit**: at most `max_per_author_per_day` candidates of one author are considered
   per day, so nobody can flood the knowledge base (the rest stays queued).
2. **SQL**: every query passes the SQL Guard under the *author's* access scope (role and
   brands from the candidate) and a BigQuery dry run, then runs: an error or an empty
   result rejects the candidate.
3. **Duplicates**: a question too close to an approved trio (cosine above
   `duplicate_score`) is rejected as a duplicate.
4. **Judge**: one call to the light model with the question, the SQL and a few result
   rows (PII-masked, as everything sent to a model). It scores whether the SQL answers the
   question, whether the case is reusable, and writes the method notes: interpretation,
   metric definitions, tables, filters and steps, without numbers.
5. **Leak check** (deterministic): the method may not contain numbers from the answer or
   the result, amounts or percentages. Trios are shown to every user, while the answer
   holds the numbers of the author's brands.

Approved when the judge scores at least `approve_score`, finds the case reusable and the
method passes; quarantined (left for the audit) when the judge is unsure or the method
fails the leak check; rejected otherwise. Approved trios are marked for the audit
sample, and `record_feedback` deprecates any trio whose answers collect negative feedback
from several users. Changes reach the agent through the index (rebuilt at the next start);
in production the eval gate runs on the new trios first (step 15.2).
"""

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from retail_agent.auth import UserConfigError, UserContext, _parse_user
from retail_agent.data.storage import Storage
from retail_agent.knowledge.golden import (
    Candidate,
    CandidateStore,
    GoldenIndex,
    approve_candidate,
    deprecate_trio,
    load_trios,
    reject_candidate,
)
from retail_agent.observability.logger import log_event
from retail_agent.safety import pii

Decision = Literal["approved", "quarantined", "rejected", "deferred"]
RESULT_ROWS_TO_JUDGE = 10
MAX_METHOD_CHARS = 1200


@dataclass(frozen=True)
class PromotionPolicy:
    approve_score: int = 4  # judge score (1-5) needed to approve automatically
    quarantine_score: int = 3  # below this: rejected
    duplicate_score: float = 0.92  # cosine similarity to an approved trio
    max_per_author_per_day: int = 5
    audit_share: float = 0.2  # share of auto-approved trios sampled for the human audit
    deprecate_after_users: int = 2  # distinct users with a bad rating of answers using a trio


class CandidateReview(BaseModel):
    """What the judge returns for one candidate."""

    answers_question: int = Field(
        ge=1,
        le=5,
        description="5: the SQL answers exactly this question with correct metric definitions, "
        "filters and period; 1: it answers something else or is wrong.",
    )
    reusable: bool = Field(
        description="True if other analysts' questions could follow this approach; false for a "
        "one-off lookup (e.g. one order's status) or a trivial query."
    )
    method: str = Field(
        description="Notes for analysts on how to answer such questions: the interpretation, "
        "metric definitions, tables, joins, filters and steps. No numbers or values from the "
        "results, no brand-specific findings.",
    )
    reason: str = Field(description="One sentence explaining the score.")


JUDGE_PROMPT = """\
You review a candidate example for a knowledge base of data-analysis precedents. Analysts' \
precedents are shown to an AI analyst when it answers similar questions, so they must be \
correct and reusable, and they must never contain results (numbers) of one user's data.

Database: BigQuery `thelook_ecommerce` (orders, order_items, products, users). Revenue is \
the sum of sale_price of items not Cancelled or Returned. The asker's data access ({scope}) \
is applied automatically to every table before the query runs, so "my brands", "our \
sales" and the like need no filter in the SQL: do not mark the SQL down for that.

Question: {question}

SQL:
{sql}

First rows of the results:
{rows}

Score the candidate and write the method notes as described in the schema. The method \
describes how to answer, never what the answer was."""

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])")
_MONEY_OR_SHARE = re.compile(r"[$€£]\s?\d|\d\s?%|\d\s?(?:USD|EUR|percent)\b", re.IGNORECASE)
_YEAR = re.compile(r"^(19|20)\d\d$")


@dataclass(frozen=True)
class Outcome:
    candidate_id: str
    decision: Decision
    reason: str
    trio_id: str | None = None


def _numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUMBER.finditer(text)}


def leaked_numbers(method: str, *, answer: str, rows: str, question: str) -> list[str]:
    """Numbers of the answer or the results that the method repeats (years and numbers
    that are part of the question, like "top 10" or "2024", are allowed)."""
    data = _numbers(answer) | _numbers(rows)
    allowed = _numbers(question)
    found = sorted(
        n
        for n in _numbers(method)
        if n in data and n not in allowed and not _YEAR.match(n.split(".")[0])
    )
    if _MONEY_OR_SHARE.search(method):
        found.append("amount or percentage")
    return found


def author_of(candidate: Candidate) -> UserContext:
    """The author's access scope; a candidate without a usable scope cannot be checked."""
    return _parse_user(
        candidate.user_id or "unknown",
        {"role": candidate.user_role, "brands": list(candidate.brands)},
    )


def _today(ts: str) -> str:
    return ts[:10]


class Promoter:
    def __init__(
        self,
        *,
        store: CandidateStore,
        trios_path: Path,
        check_sql: Callable[[str, UserContext], str],
        run_sql: Callable[[str], pd.DataFrame],
        judge: Callable[[str], CandidateReview] | None,
        index: GoldenIndex | None = None,
        policy: PromotionPolicy = PromotionPolicy(),
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        rng: random.Random | None = None,
    ) -> None:
        """`check_sql(sql, author)` returns the guarded SQL after a dry run (raises if it
        must not run); `run_sql` executes it."""
        self.store = store
        self.trios_path = trios_path
        self.check_sql = check_sql
        self.run_sql = run_sql
        self.judge = judge
        self.index = index
        self.policy = policy
        self.now = now
        self.rng = rng or random.Random()

    def run(self) -> list[Outcome]:
        outcomes = [self._one(c) for c in self._admitted()]
        log_event(
            "golden_promotion_run",
            **{
                d: sum(o.decision == d for o in outcomes)
                for d in ("approved", "quarantined", "rejected", "deferred")
            },
        )
        return outcomes

    def _admitted(self) -> list[Candidate]:
        """Pending candidates within each author's daily allowance; the rest wait."""
        today = self.now().date().isoformat()
        used: dict[str, int] = {}
        for c in self.store.all():
            if c.status != "candidate" and c.reviewed_at and _today(c.reviewed_at) == today:
                used[c.user_id] = used.get(c.user_id, 0) + 1
        admitted = []
        for c in self.store.pending():
            if used.get(c.user_id, 0) >= self.policy.max_per_author_per_day:
                log_event("golden_candidate_deferred", candidate_id=c.id, reason="rate_limit")
                continue
            used[c.user_id] = used.get(c.user_id, 0) + 1
            admitted.append(c)
        return admitted

    def _settle(
        self, candidate: Candidate, decision: Decision, reason: str, checks: dict[str, Any]
    ) -> Outcome:
        checks = {**checks, "decision": decision, "reason": reason}
        if decision == "rejected":
            reject_candidate(replace(candidate, checks=checks), store=self.store, note=reason)
        else:  # quarantined
            self.store.update(
                replace(
                    candidate,
                    status="quarantined",
                    reviewed_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    note=reason,
                    checks=checks,
                    method=checks.get("method", candidate.method),
                )
            )
            log_event("golden_candidate_quarantined", candidate_id=candidate.id, reason=reason)
        return Outcome(candidate.id, decision, reason)

    def _one(self, candidate: Candidate) -> Outcome:
        checks: dict[str, Any] = {}
        try:
            author = author_of(candidate)
        except UserConfigError as exc:
            return self._settle(candidate, "rejected", f"no usable author scope: {exc}", checks)

        frames = []
        for sql in candidate.sql:
            try:
                guarded = self.check_sql(sql, author)
                df = self.run_sql(guarded)
            except Exception as exc:
                return self._settle(
                    candidate,
                    "rejected",
                    f"SQL check failed: {type(exc).__name__}: {exc}"[:300],
                    checks,
                )
            if df.empty:
                return self._settle(candidate, "rejected", "a query returns no rows", checks)
            frames.append(pii.mask_dataframe(df, where="golden_promotion")[0])

        if self.index is not None:
            hits = self.index.retrieve(candidate.question, k=1)
            if (
                hits
                and hits[0].method == "embedding"
                and hits[0].score >= self.policy.duplicate_score
            ):
                checks["duplicate_of"] = hits[0].trio.id
                return self._settle(
                    candidate, "rejected", f"duplicate of trio {hits[0].trio.id}", checks
                )

        if self.judge is None:
            return self._settle(candidate, "quarantined", "no judge model available", checks)
        rows = "\n\n".join(
            f.head(RESULT_ROWS_TO_JUDGE).to_csv(index=False, lineterminator="\n") for f in frames
        )
        scope = "all brands" if author.has_full_access else f"brands {author.scope_label}"
        prompt = JUDGE_PROMPT.format(
            question=candidate.question, sql="\n\n".join(candidate.sql), rows=rows, scope=scope
        )
        try:
            review = self.judge(pii.mask_text(prompt)[0])
        except Exception as exc:
            log_event("golden_judge_failed", level="warning", error=type(exc).__name__)
            return Outcome(candidate.id, "deferred", "judge unavailable, retried next run")

        method = pii.mask_text(" ".join(review.method.split())[:MAX_METHOD_CHARS])[0]
        checks.update(
            score=review.answers_question,
            reusable=review.reusable,
            judge_reason=review.reason,
            method=method,
        )
        leaks = leaked_numbers(
            method, answer=candidate.answer_summary, rows=rows, question=candidate.question
        )
        if review.answers_question < self.policy.quarantine_score:
            return self._settle(
                candidate, "rejected", f"judge score {review.answers_question}", checks
            )
        if not review.reusable:
            return self._settle(candidate, "rejected", "not reusable", checks)
        if leaks or pii.REDACTED in method or len(method) < 20:
            checks["leaks"] = leaks
            return self._settle(
                candidate, "quarantined", "method notes failed the leak check", checks
            )
        if review.answers_question < self.policy.approve_score:
            return self._settle(
                candidate, "quarantined", f"judge score {review.answers_question}", checks
            )

        checks["audit"] = "sampled" if self.rng.random() < self.policy.audit_share else "skipped"
        trio = approve_candidate(
            replace(candidate, method=method),
            store=self.store,
            trios_path=self.trios_path,
            validate_sql=lambda _sql: None,  # already checked and run above
            report=method,
            # The author's brands are not tags: they would point other users at this trio
            # and tell them whose scope it came from.
            tags=[
                t for t in candidate.tags if t.lower() not in {b.lower() for b in candidate.brands}
            ],
            approved_by="auto",
            checks=checks,
        )
        return Outcome(candidate.id, "approved", "all checks passed", trio.id)


# -- feedback -> deprecation --------------------------------------------------------------------


def record_feedback(
    storage: Storage,
    trios_path: Path,
    *,
    trio_ids: Sequence[str],
    user_id: str,
    rating: str,
    trace_id: str,
    index: GoldenIndex | None = None,
    policy: PromotionPolicy = PromotionPolicy(),
) -> list[str]:
    """Count a rating against the trios that were in the answer's prompt; deprecate the
    ones with bad ratings from at least `deprecate_after_users` users and more bad than
    good ratings. Returns the deprecated trio ids."""
    deprecated = []
    for trio_id in dict.fromkeys(trio_ids):
        storage.add_trio_feedback(trio_id, user_id, rating, trace_id)
        good, bad, bad_users = storage.trio_feedback_counts(trio_id)
        if bad_users >= policy.deprecate_after_users and bad > good:
            reason = f"negative feedback from {bad_users} users ({bad} bad, {good} good)"
            if trios_path.exists() and deprecate_trio(trios_path, trio_id, reason):
                deprecated.append(trio_id)
                if index is not None:
                    index.drop(trio_id)
    return deprecated


def audit_queue(store: CandidateStore, trios_path: Path) -> tuple[list[Candidate], list[Any]]:
    """What a person looks at: quarantined candidates and sampled auto-approved trios."""
    quarantined = [c for c in store.all() if c.status == "quarantined"]
    trios = load_trios(trios_path) if trios_path.exists() else []
    sampled = [
        t
        for t in trios
        if t.status == "approved"
        and (t.source or {}).get("approved_by") == "auto"
        and ((t.source or {}).get("checks") or {}).get("audit") == "sampled"
    ]
    return quarantined, sampled
