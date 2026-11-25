"""Publishing the persona through the automated quality gate (no manual approval, no redeploy).

A non-developer edits `config/persona_draft.md`; `scripts/publish_persona.py publish`:

1. **Static checks**: not empty, not too long, no prompt-injection or data-changing
   phrasing (the input guard's rules), no wording that grants permissions or tries to
   override the fixed rules ("you may show e-mails", "ignore the rules"), no personal data.
   The persona sets tone and style only; safety rules live in the immutable core and the
   code, so a persona cannot change them anyway, but a draft that tries is refused early.
2. **Eval gate**: the `persona` suite (security cases plus a few ordinary questions) runs
   with the draft; it must pass the thresholds in `evals/gate.yaml`.
3. **Publish**: the draft is saved as a version and atomically replaces the active persona,
   which the agent re-reads on the next turn. `rollback` restores an earlier published
   version (it passed the gate when it was published).

An inconclusive gate (model outage) publishes nothing; the draft can be published later.
"""

import hashlib
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from retail_agent.evaluation.gate import FAILED, INCONCLUSIVE, PASSED
from retail_agent.observability.logger import log_event
from retail_agent.prompts.context import MAX_PERSONA_CHARS, persona_text
from retail_agent.safety import input_guard, pii

FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rx, re.IGNORECASE), why)
    for rx, why in (
        (
            r"\b(ignore|disregard|override|bypass|forget|relax)\b.{0,40}"
            r"\b(rules?|instructions?|restrictions?|guardrails?|safety|polic(?:y|ies)|limits?)\b",
            "tries to override the fixed rules",
        ),
        (
            r"\b(you (?:may|can|are allowed to|are permitted to)|feel free to|it is (?:ok|fine) to)"
            r"\b.{0,60}\b(show|reveal|share|print|list|include|access|delete|query|run)\b",
            "grants a permission (the persona sets tone only)",
        ),
        (
            r"\b(personal data|PII|e-?mails?|phone numbers?|street address(?:es)?|home address)\b",
            "talks about personal data (the persona sets tone only)",
        ),
        (
            r"\b(all|other|every) (brands?|categories|users'?|managers'?)\b",
            "talks about data access (the persona sets tone only)",
        ),
        (
            r"\b(system prompt|developer mode|jailbreak|you are now)\b",
            "role or prompt manipulation",
        ),
    )
)

Status = Literal["published", "rejected", "inconclusive", "unchanged"]
ACTIVE_HEADER = """\
# Active persona: tone and style of the analyst
#
# Do not edit by hand: this file is written by `scripts/publish_persona.py` from
# config/persona_draft.md after the draft passes the quality gate. The agent re-reads it
# at the start of every turn. Lines starting with "#" are ignored.
"""


def active_file_text(raw: str) -> str:
    return f"{ACTIVE_HEADER}\n{persona_text(raw)}\n"


def _same(a: str, b: str) -> bool:
    return persona_text(a) == persona_text(b)


@dataclass
class PublishResult:
    status: Status
    version: str | None = None
    reasons: list[str] = field(default_factory=list)


def static_problems(raw: str) -> list[str]:
    """Why the draft cannot be published, before any model is asked; empty if fine."""
    text = persona_text(raw)
    problems = []
    if not text:
        problems.append("the persona is empty (lines starting with # are comments)")
    if len(raw.strip()) > MAX_PERSONA_CHARS:
        problems.append(f"the persona is longer than {MAX_PERSONA_CHARS} characters")
    ruled = input_guard.check_rules(text)
    if ruled is not None:
        problems.append(f"blocked by the input guard ({ruled.reason})")
    for pattern, why in FORBIDDEN:
        if match := pattern.search(text):
            problems.append(f"{why}: {match.group(0)!r}")
    if pii.mask_text(text)[1]:
        problems.append("contains personal data")
    return problems


def version_id(raw: str, now: datetime, number: int) -> str:
    """`0003_20261125-090255_4ad5a73a`: the number keeps the order within one second."""
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{number:04d}_{now.strftime('%Y%m%d-%H%M%S')}_{digest}"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class PersonaPublisher:
    active: Path
    draft: Path
    versions_dir: Path
    run_gate: Callable[[Path], int]  # eval gate with the draft: 0 passed, 1 failed, 3 inconclusive
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def versions(self) -> list[Path]:
        """Published versions, oldest first."""
        return sorted(self.versions_dir.glob("*.md")) if self.versions_dir.exists() else []

    def _save_version(self, raw: str) -> str:
        version = version_id(raw, self.now(), len(self.versions()) + 1)
        _write_atomic(self.versions_dir / f"{version}.md", raw)
        return version

    def publish(self) -> PublishResult:
        raw = self.draft.read_text(encoding="utf-8")
        current = self.active.read_text(encoding="utf-8") if self.active.exists() else ""
        if _same(raw, current):
            return PublishResult("unchanged", reasons=["the draft equals the active persona"])
        problems = static_problems(raw)
        if problems:
            log_event("persona_rejected", stage="static", reasons=problems)
            return PublishResult("rejected", reasons=problems)
        code = self.run_gate(self.draft)
        if code == INCONCLUSIVE:
            log_event("persona_gate_inconclusive")
            return PublishResult(
                "inconclusive", reasons=["the eval gate could not judge the draft; try later"]
            )
        if code != PASSED:
            reasons = ["the eval gate failed with the draft (see the eval output)"]
            log_event("persona_rejected", stage="eval", code=code if code != FAILED else 1)
            return PublishResult("rejected", reasons=reasons)
        if current and not self.versions():
            self._save_version(current)  # keep the persona that was active before the first publish
        version = self._save_version(raw)
        _write_atomic(self.active, active_file_text(raw))
        log_event("persona_published", version=version)
        return PublishResult("published", version=version)

    def rollback(self, to: str | None = None) -> PublishResult:
        """Restore a published version: `to`, or the one before the active persona."""
        versions = self.versions()
        current = self.active.read_text(encoding="utf-8") if self.active.exists() else ""
        if to is not None:
            chosen = [v for v in versions if v.stem == to or v.stem.endswith(to)]
            if not chosen:
                return PublishResult("rejected", reasons=[f"no published version {to!r}"])
            target = chosen[-1]
        else:
            earlier = [v for v in versions if not _same(v.read_text(encoding="utf-8"), current)]
            if not earlier:
                return PublishResult("rejected", reasons=["no earlier version to roll back to"])
            active_index = next(
                (
                    i
                    for i, v in enumerate(versions)
                    if _same(v.read_text(encoding="utf-8"), current)
                ),
                len(versions),
            )
            before = [v for v in versions[:active_index] if v in earlier]
            target = (before or earlier)[-1]
        _write_atomic(self.active, active_file_text(target.read_text(encoding="utf-8")))
        log_event("persona_rolled_back", version=target.stem)
        return PublishResult("published", version=target.stem)
