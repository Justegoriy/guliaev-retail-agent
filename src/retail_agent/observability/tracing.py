"""Optional LangSmith tracing and the per-turn `RunnableConfig`.

Tracing never affects a turn: the API key is checked once at startup and tracing is
switched off (with a warning for the banner) if LangSmith is unreachable; runs are
uploaded in the background and upload failures are not reported to the user. Every
string sent to LangSmith goes through `pii.mask_text` via the client's anonymizer.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langsmith.anonymizer import create_anonymizer

from retail_agent.observability.callbacks import ObservabilityCallbackHandler
from retail_agent.observability.logger import log_event
from retail_agent.safety import pii
from retail_agent.settings import Settings

# Variables that make LangChain attach its own tracer, which would bypass the anonymizer.
_ENV_TRACING_FLAGS = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2")
LANGSMITH_TIMEOUT_MS = 5000


@dataclass(frozen=True)
class TracingStatus:
    enabled: bool
    warning: str | None = None


_client: Client | None = None
_project: str | None = None


anonymize: Callable[[Any], Any] = create_anonymizer(lambda text, _path: pii.mask_text(text)[0])


def setup_langsmith(
    settings: Settings, *, client_factory: Callable[..., Client] = Client
) -> TracingStatus:
    global _client, _project
    _client = None
    for name in _ENV_TRACING_FLAGS:
        os.environ[name] = "false"

    if not settings.langsmith_tracing:
        return TracingStatus(enabled=False)
    if settings.langsmith_api_key is None:
        return TracingStatus(
            enabled=False, warning="LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty."
        )
    try:
        client = client_factory(
            api_key=settings.langsmith_api_key.get_secret_value(),
            anonymizer=anonymize,
            timeout_ms=LANGSMITH_TIMEOUT_MS,
        )
        next(iter(client.list_projects(limit=1)), None)
    except Exception as exc:
        log_event("langsmith_disabled", level="warning", error=type(exc).__name__)
        return TracingStatus(
            enabled=False,
            warning=(
                f"LangSmith is unavailable ({type(exc).__name__}); tracing is off, "
                "local logs are still written."
            ),
        )
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)
    _client, _project = client, settings.langsmith_project
    log_event("langsmith_enabled", project=_project)
    return TracingStatus(enabled=True)


def turn_config(
    settings: Settings, *, trace_id: str, conversation_id: str, user_role: str
) -> RunnableConfig:
    """Config for one graph run: local log callbacks, LangSmith tracer, run metadata."""
    payload_path = settings.llm_payload_log_path if settings.log_llm_payloads else None
    callbacks: list[BaseCallbackHandler] = [ObservabilityCallbackHandler(payload_path)]
    if _client is not None:
        callbacks.append(LangChainTracer(project_name=_project, client=_client))
    return {
        "run_name": "turn",
        "metadata": {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "user_role": user_role,
        },
        "tags": [user_role],
        "callbacks": callbacks,
    }
