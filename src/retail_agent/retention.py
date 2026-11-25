"""Data retention (D11, D31): chat history 90 days, audit log 1 year, reports until deleted.

`purge_expired` runs when the agent starts (CLI, web app, eval) and removes everything
that holds the content of conversations older than `CHAT_RETENTION_DAYS` (by their last
activity):

- the conversation row (title, owner, turn count) and its LangGraph checkpoint thread
  (messages, tool calls and results, analysis context, pending confirmations);
- lines of the event log and of the full model payload log older than the cutoff (they
  hold masked messages, SQL and answers);
- Golden Bucket candidates created before the cutoff (they hold a question and the
  answer the user rated); an approved trio keeps only the method notes, not the answer.

Not chat history, so kept: saved reports (until the user deletes them, hard delete),
the audit log (purged after one year by `Storage`), user preferences and the weights of
preference signals, ratings of Golden Bucket trios (ids only). In production the same
rules are Firestore TTL policies and log-bucket retention.
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


def purge_expired(
    settings: Settings,
    storage: Storage,
    checkpointer: Any,
    candidates: CandidateStore | None = None,
    *,
    now: datetime | None = None,
) -> RetentionReport:
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=settings.chat_retention_days)
    expired = storage.conversations_inactive_since(_iso(cutoff))
    for conversation_id in expired:
        checkpointer.delete_thread(conversation_id)
        storage.delete_conversation(conversation_id)
    report = RetentionReport(
        conversations=len(expired),
        log_lines=prune_jsonl(settings.log_path, cutoff),
        payload_lines=prune_jsonl(settings.llm_payload_log_path, cutoff),
        candidates=prune_candidates(candidates, cutoff) if candidates is not None else 0,
    )
    if any(report.as_dict().values()):
        log_event(
            "retention_purged", retention_days=settings.chat_retention_days, **report.as_dict()
        )
    return report
