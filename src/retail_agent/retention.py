"""Data retention (D31): how long each kind of data is kept, enforced when the agent starts.

Periods (days are settings):

- chat history: the conversation row and its checkpoint thread (messages, tool calls and
  results, analysis context, pending confirmations), 90 days after the last activity
  (`CHAT_RETENTION_DAYS`);
- event log (`agent.jsonl`: masked messages, SQL, outcomes): 90 days (`CHAT_RETENTION_DAYS`);
- Golden Bucket candidates (a question and the answer a user rated): 90 days;
- full model conversations (`llm_payloads.jsonl`, debugging): 30 days
  (`PAYLOAD_RETENTION_DAYS`);
- exported report files (`exports/`; the report stays in the library): 30 days
  (`EXPORT_RETENTION_DAYS`);
- preference signals: 180 days; with a half-life of 30 days older ones weigh under 2%
  (`SIGNAL_RETENTION_DAYS`);
- inferred preferences: 1 year after the last update (`INFERRED_PREFERENCE_RETENTION_DAYS`);
- ratings of Golden Bucket trios (ids only): 1 year (`TRIO_FEEDBACK_RETENTION_DAYS`);
- audit log: 1 year (`Storage` purges it on open);
- saved reports: until the user deletes them (hard delete);
- explicit preferences: until the user changes or forgets them.

Golden Bucket trios hold no personal data and are kept as versioned knowledge (deprecated
ones for the record); eval results are synthetic test data. In production the same rules
are Firestore TTL policies, log-bucket retention and a LangSmith project retention of 14
days.
"""

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from retail_agent.data.storage import Storage
from retail_agent.knowledge.golden import CandidateStore
from retail_agent.observability import logger
from retail_agent.observability.logger import log_event
from retail_agent.settings import Settings


@dataclass(frozen=True)
class RetentionReport:
    conversations: int = 0
    log_lines: int = 0
    payload_lines: int = 0
    candidates: int = 0
    exports: int = 0
    signals: int = 0
    inferred_preferences: int = 0
    trio_ratings: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def prune_jsonl(path: Path, cutoff: datetime) -> int:
    """Drop the lines whose `ts` is older than `cutoff`; returns how many were dropped.

    Holds the logger's lock, so no event is written into the file while it is rewritten.
    Lines that cannot be parsed are kept (nothing is deleted by mistake)."""
    if not path.exists():
        return 0
    limit = _iso(cutoff)
    with logger._lock:
        kept, dropped = [], 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    ts = str(json.loads(line).get("ts") or "")
                except (ValueError, AttributeError):
                    ts = ""
                if ts and ts[:19] < limit[:19]:
                    dropped += 1
                else:
                    kept.append(line)
        if dropped:
            tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
            tmp.write_text("".join(kept), encoding="utf-8")
            os.replace(tmp, path)
    return dropped


def prune_candidates(store: CandidateStore, cutoff: datetime) -> int:
    items = store.all()
    keep = [c for c in items if c.created_at >= _iso(cutoff)]
    if len(keep) == len(items):
        return 0
    store.replace_all(keep)
    return len(items) - len(keep)


def prune_files(directory: Path, cutoff: datetime, pattern: str = "*.md") -> int:
    """Delete files in `directory` last modified before `cutoff`."""
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.glob(pattern):
        if datetime.fromtimestamp(path.stat().st_mtime, UTC) < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def purge_expired(
    settings: Settings,
    storage: Storage,
    checkpointer: Any,
    candidates: CandidateStore | None = None,
    *,
    now: datetime | None = None,
) -> RetentionReport:
    now = now or datetime.now(UTC)

    def before(days: int) -> datetime:
        return now - timedelta(days=days)

    cutoff = before(settings.chat_retention_days)
    expired = storage.conversations_inactive_since(_iso(cutoff))
    for conversation_id in expired:
        checkpointer.delete_thread(conversation_id)
        storage.delete_conversation(conversation_id)
    rows = storage.purge_older_than(
        signals=_iso(before(settings.signal_retention_days)),
        inferred=_iso(before(settings.inferred_preference_retention_days)),
        trio_feedback=_iso(before(settings.trio_feedback_retention_days)),
    )
    report = RetentionReport(
        conversations=len(expired),
        log_lines=prune_jsonl(settings.log_path, cutoff),
        payload_lines=prune_jsonl(
            settings.llm_payload_log_path, before(settings.payload_retention_days)
        ),
        candidates=prune_candidates(candidates, cutoff) if candidates is not None else 0,
        exports=prune_files(settings.exports_dir, before(settings.export_retention_days)),
        **rows,
    )
    if any(report.as_dict().values()):
        log_event("retention_purged", **report.as_dict())
    return report
