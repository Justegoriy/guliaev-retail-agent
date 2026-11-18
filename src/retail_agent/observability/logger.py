"""Structured JSON event log with per-turn correlation ids.

Every event is one JSON line in `LOG_PATH`. `trace_id`, `conversation_id` and `user_id`
come from context variables, so callers only pass event-specific fields.
All string values are PII-masked before they are written.
"""

import json
import sys
import threading
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from retail_agent.safety import pii

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
conversation_id_var: ContextVar[str | None] = ContextVar("conversation_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)

_lock = threading.Lock()
_log_path: Path | None = None
_debug = False


def configure(log_path: Path, debug: bool = False) -> None:
    global _log_path, _debug
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_path = log_path
    _debug = debug


def new_trace_id() -> str:
    trace_id = uuid.uuid4().hex[:16]
    trace_id_var.set(trace_id)
    return trace_id


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return pii.mask_text(value)
    if isinstance(value, dict):
        return {k: _mask(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_mask(v) for v in value]
    return value


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(_mask(record), ensure_ascii=False, default=str)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "level": level,
        "event": event,
        "trace_id": trace_id_var.get(),
        "conversation_id": conversation_id_var.get(),
        "user_id": user_id_var.get(),
        **fields,
    }
    if _log_path is not None:
        try:
            write_jsonl(_log_path, record)
        except OSError as exc:
            print(f"[log] failed to write {_log_path}: {exc}", file=sys.stderr)
    if _debug:
        details = " ".join(f"{k}={v}" for k, v in _mask(fields).items())
        print(f"[{level}] {event} {details}".rstrip(), file=sys.stderr)
