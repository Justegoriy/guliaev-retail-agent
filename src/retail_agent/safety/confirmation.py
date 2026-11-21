"""Confirmation of actions with side effects (deleting and exporting reports, later e-mail).

A tool calls `require(...)`: LangGraph `interrupt()` pauses the graph, the CLI shows the
preview and asks the user outside the LLM, and the answer comes back through
`Command(resume=...)`. Nothing the model writes can answer the question.

The answer is bound to what was shown: the CLI resumes with the preview's `digest`, and
the tool, re-run on resume, recomputes it from the current data. If they differ the
action is not performed (`Decision.STALE`).

Two guards keep tools honest:
- `require` works only inside a tool registered with `requires_confirmation=True`
  (the registry opens a confirmation scope around each call of such a tool);
- a side effect calls `ensure_approved(action)` first, which raises unless `require`
  for that action was approved in the same tool call.
"""

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from retail_agent.observability.logger import log_event

YES_ANSWERS = frozenset({"y", "yes", "д", "да"})

# (audit action, target ids, details) -> None
AuditSink = Callable[[str, list[str], dict[str, Any]], None]


class Decision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale"  # the objects changed between the preview and the answer


class ConfirmationError(Exception):
    """A tool asks for confirmation or performs a side effect without the required setup."""


@dataclass
class _Scope:
    tool: str
    approved: set[str] = field(default_factory=set)


_scope: ContextVar[_Scope | None] = ContextVar("confirmation_scope", default=None)


@contextmanager
def scope(tool: str) -> Iterator[None]:
    token = _scope.set(_Scope(tool))
    try:
        yield
    finally:
        _scope.reset(token)


def scoped(tool: str, func: Callable[..., Any]) -> Callable[..., Any]:
    """`func` running inside its own confirmation scope (used by the tool registry)."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with scope(tool):
            return func(*args, **kwargs)

    return wrapper


def is_yes(answer: Any) -> bool:
    if isinstance(answer, bool):
        return answer
    return isinstance(answer, str) and answer.strip().lower() in YES_ANSWERS


def _parse_reply(reply: Any) -> tuple[Any, str | None]:
    if isinstance(reply, dict):
        return reply.get("answer"), reply.get("digest")
    return reply, None


def require(
    action: str,
    *,
    question: str,
    summary: str,
    items: list[dict[str, Any]],
    digest: str,
    audit: AuditSink | None = None,
) -> Decision:
    """Pause for the user's answer; only `y`/`yes` (and Russian `д`/`да`) approve."""
    current = _scope.get()
    if current is None:
        raise ConfirmationError(
            f"confirmation of {action!r} requested outside a tool registered with "
            "requires_confirmation=True"
        )
    targets = [str(item["id"]) for item in items if "id" in item]
    payload = {
        "type": "confirm",
        "action": action,
        "question": question,
        "summary": summary,
        "items": items,
        "digest": digest,
    }
    try:
        reply = interrupt(payload)
    except GraphInterrupt:
        # Raised only when the graph actually pauses, not when it resumes with an answer.
        log_event("confirmation_requested", action=action, tool=current.tool, count=len(items))
        if audit:
            audit("confirmation_requested", targets, {"tool": current.tool, "digest": digest})
        raise

    answer, shown_digest = _parse_reply(reply)
    if shown_digest is not None and shown_digest != digest:
        decision = Decision.STALE
    elif is_yes(answer):
        decision = Decision.APPROVED
        current.approved.add(action)
    else:
        decision = Decision.REJECTED
    log_event(f"confirmation_{decision}", action=action, tool=current.tool, count=len(items))
    if audit:
        audit(f"confirmation_{decision}", targets, {"tool": current.tool, "digest": digest})
    return decision


def ensure_approved(action: str) -> None:
    """Call right before a side effect: raises unless the user approved `action` just now."""
    current = _scope.get()
    if current is None or action not in current.approved:
        raise ConfirmationError(f"{action!r} was not confirmed by the user")
