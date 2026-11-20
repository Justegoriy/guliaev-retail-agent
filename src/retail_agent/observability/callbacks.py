"""LangChain callback handler that turns model and tool runs into log events.

Events: `llm_call` (model, latency, tokens), `llm_error`, `tool_call`. With
`LOG_LLM_PAYLOADS` the full request and response of every model call also go to
`logs/llm_payloads.jsonl`, keyed by `trace_id` + `call_id`, for `/trace <id> --full`.
Tool arguments and results are PII-masked before they are clipped, so a cut cannot
leave half of an e-mail address unmasked.
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult

from retail_agent.errors import classify
from retail_agent.observability import logger
from retail_agent.observability.logger import log_event
from retail_agent.safety import pii

MAX_FIELD_CHARS = 2048
MAX_ERROR_CHARS = 500


def clip(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    text = pii.mask_text(text)[0]
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [+{len(text) - limit} chars]"


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    data: dict[str, Any] = {"type": message.type, "text": message.text}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in tool_calls]
    if isinstance(message, ToolMessage):
        data["name"] = message.name
        data["tool_call_id"] = message.tool_call_id
    return data


@dataclass
class _Run:
    name: str
    started: float = field(default_factory=time.perf_counter)
    metadata: dict[str, Any] = field(default_factory=dict)
    request: list[dict[str, Any]] | None = None
    args: str = ""

    @property
    def latency_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)


class ObservabilityCallbackHandler(BaseCallbackHandler):
    def __init__(self, payload_log_path: Path | None = None) -> None:
        self._runs: dict[UUID, _Run] = {}
        self._payload_log_path = payload_log_path

    # -- models ---------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = metadata or {}
        model = metadata.get("ls_model_name") or (invocation_params or {}).get("model") or "?"
        request = None
        if self._payload_log_path is not None and messages:
            request = [message_to_dict(m) for m in messages[0]]
        self._runs[run_id] = _Run(name=str(model), metadata=metadata, request=request)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        generation = response.generations[0][0] if response.generations[0:1] else None
        message: BaseMessage | None = getattr(generation, "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        tool_calls = getattr(message, "tool_calls", None) or []
        finish_reason = (getattr(message, "response_metadata", None) or {}).get("finish_reason")
        log_event(
            "llm_call",
            call_id=_call_id(run_id),
            model=run.name,
            purpose=run.metadata.get("purpose"),
            model_role=run.metadata.get("model_role"),
            latency_ms=run.latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            tool_calls=[c["name"] for c in tool_calls],
            finish_reason=finish_reason,
        )
        if message is not None:
            self._write_payload(run_id, run, response=message_to_dict(message))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        log_event(
            "llm_error",
            level="warning",
            call_id=_call_id(run_id),
            model=run.name,
            purpose=run.metadata.get("purpose"),
            model_role=run.metadata.get("model_role"),
            latency_ms=run.latency_ms,
            kind=classify(error).value,
            error=type(error).__name__,
            detail=clip(str(error), MAX_ERROR_CHARS),
        )
        self._write_payload(run_id, run, error=f"{type(error).__name__}: {error}")

    def _write_payload(self, run_id: UUID, run: _Run, **result: Any) -> None:
        if self._payload_log_path is None or run.request is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "trace_id": logger.trace_id_var.get(),
            "conversation_id": logger.conversation_id_var.get(),
            "call_id": _call_id(run_id),
            "model": run.name,
            "purpose": run.metadata.get("purpose"),
            "latency_ms": run.latency_ms,
            "request": run.request,
            **result,
        }
        try:
            self._payload_log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.write_jsonl(self._payload_log_path, record)
        except OSError as exc:
            log_event("payload_log_failed", level="warning", error=type(exc).__name__)

    # -- tools ----------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "?"
        self._runs[run_id] = _Run(name=str(name), args=clip(input_str), metadata=metadata or {})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        text = output.content if isinstance(output, ToolMessage) else output
        log_event(
            "tool_call",
            tool=run.name,
            status="ok",
            latency_ms=run.latency_ms,
            args=run.args,
            result=clip(text if isinstance(text, str) else str(text)),
            **_tool_flags(run),
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        log_event(
            "tool_call",
            level="warning",
            tool=run.name,
            status="error",
            latency_ms=run.latency_ms,
            args=run.args,
            error=type(error).__name__,
            detail=clip(str(error), MAX_ERROR_CHARS),
            **_tool_flags(run),
        )


def _call_id(run_id: UUID) -> str:
    return run_id.hex[:12]


def _tool_flags(run: _Run) -> dict[str, Any]:
    """Registry flags (`side_effects`, `requires_confirmation`) set in the tool's metadata."""
    keys = ("side_effects", "requires_confirmation")
    return {k: run.metadata[k] for k in keys if k in run.metadata}
