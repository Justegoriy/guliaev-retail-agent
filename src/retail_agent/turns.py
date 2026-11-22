"""One user turn as every interface runs it: trace id, logs, error handling, metrics.

A turn may span several graph runs: it pauses on a confirmation (`status="pending"`)
and continues with `Turn.resume`, possibly much later (a web page waits for a click).
`turn_end` is logged once, when the turn is finished, so `/stats` counts turns from the
CLI and the web interface the same way.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from langgraph.errors import GraphRecursionError

from retail_agent import errors
from retail_agent.graph.builder import AgentSession, StageCallback, TurnOutcome
from retail_agent.observability import callbacks, logger
from retail_agent.observability.logger import log_event

STEP_LIMIT_MESSAGE = (
    "The request turned out to be too complex to finish in one go. "
    "Please narrow it down or split it into smaller questions."
)
CANCELLED_MESSAGE = "Cancelled. Ask again or try something else."

Status = Literal["ok", "blocked", "pending", "error", "cancelled"]


@dataclass(frozen=True)
class TurnResult:
    trace_id: str
    status: Status
    outcome: TurnOutcome | None = None  # set for ok / blocked / pending
    message: str | None = None  # what to tell the user for error / cancelled
    error_kind: str | None = None

    @property
    def answer(self) -> str:
        return self.outcome.answer if self.outcome is not None else ""


class Turn:
    """Start with `start(text)`; while the result is pending, answer with `resume`."""

    def __init__(self, session: AgentSession) -> None:
        self.session = session
        self.trace_id = logger.new_trace_id()
        self._started = time.perf_counter()
        self.finished = False

    def start(self, text: str, on_stage: StageCallback | None = None) -> TurnResult:
        log_event("turn_start", length=len(text))
        return self._call(lambda: self.session.run_turn(text, self.trace_id, on_stage))

    def resume(
        self, answers: Mapping[str, str], on_stage: StageCallback | None = None
    ) -> TurnResult:
        """Answers: interrupt id -> user's text; unanswered confirmations count as "no"."""
        logger.trace_id_var.set(self.trace_id)
        return self._call(lambda: self.session.resume(answers, self.trace_id, on_stage))

    def _call(self, action: Callable[[], TurnOutcome]) -> TurnResult:
        try:
            outcome = action()
        except KeyboardInterrupt:
            self._cancel()
            return TurnResult(self.trace_id, "cancelled", message=CANCELLED_MESSAGE)
        except GraphRecursionError:
            self._end("error", "warning", error_kind="step_limit")
            message = f"{STEP_LIMIT_MESSAGE} (trace: {self.trace_id})"
            return TurnResult(self.trace_id, "error", message=message, error_kind="step_limit")
        except Exception as exc:
            kind = errors.classify(exc)
            self._end(
                "error",
                "error",
                error_kind=kind.value,
                error=type(exc).__name__,
                detail=str(exc)[:500],
            )
            message = errors.user_message(kind, self.trace_id)
            return TurnResult(self.trace_id, "error", message=message, error_kind=kind.value)
        except BaseException:
            # The host stopped the run (e.g. Streamlit's stop button): close the turn, re-raise.
            self._cancel()
            raise
        if outcome.pending:
            return TurnResult(self.trace_id, "pending", outcome)
        status: Status = "blocked" if outcome.blocked else "ok"
        self._end(status)
        return TurnResult(self.trace_id, status, outcome)

    def _cancel(self) -> None:
        try:
            self.session.cancel_turn()
        except Exception as exc:
            log_event("cancel_failed", level="error", error=type(exc).__name__)
        self._end("cancelled", "warning")

    def _end(self, status: str, level: str = "info", **extra) -> None:
        if self.finished:
            return
        self.finished = True
        usage = callbacks.pop_usage(self.trace_id)
        turn = self.session.deps.tool_ctx.turn
        log_event(
            "turn_end",
            level=level,
            status=status,
            latency_ms=int((time.perf_counter() - self._started) * 1000),
            llm_calls=usage.llm_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            sql_count=len(turn.executed),
            sql_failures=turn.sql_failures,
            bq_bytes=turn.bq_bytes,
            **extra,
        )
