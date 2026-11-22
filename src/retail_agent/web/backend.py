"""What the web app needs from the agent, kept apart from Streamlit so it can be tested
and replaced in tests: shared resources, startup checks and small view helpers."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from retail_agent import auth, errors, health
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.graph import memory
from retail_agent.graph.builder import AgentSession, create_resources
from retail_agent.observability import inspect, logger, tracing
from retail_agent.observability.logger import log_event
from retail_agent.settings import Settings, get_settings

CONFIRM_LABELS = {"delete_reports": "Delete", "export_report": "Export"}


@dataclass
class Backend:
    """Process-wide state shared by all browser tabs (one per server)."""

    settings: Settings
    users: dict[str, auth.UserContext]
    checks: list[health.Check]
    langsmith: tracing.TracingStatus
    session_factory: Callable[[auth.UserContext], AgentSession] | None
    startup_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.session_factory is not None

    def session(self, user: auth.UserContext) -> AgentSession:
        if self.session_factory is None:
            raise RuntimeError("the agent is not available")
        logger.user_id_var.set(user.user_id)
        session = self.session_factory(user)
        bind_context(session)
        log_event(
            "session_start",
            role=user.role,
            category_groups=list(user.category_groups),
            categories=list(user.categories),
            langsmith=self.langsmith.enabled,
            resumed=False,
            client="web",
        )
        return session


def load_backend() -> Backend:
    """Configure logging and tracing, check connections and load shared resources once."""
    settings = get_settings()
    logger.configure(settings.log_path)
    langsmith = tracing.setup_langsmith(settings)
    users = auth.load_users()
    runner = BigQueryRunner.from_settings(settings)
    checks = [health.check_gemini(settings), health.check_bigquery(runner, settings)]
    backend = Backend(settings, users, checks, langsmith, session_factory=None)
    if any(c.fatal for c in checks):
        return backend
    try:
        backend.session_factory = create_resources(settings, runner=runner).session
    except Exception as exc:
        kind = errors.classify(exc)
        log_event("startup_failed", level="error", error_kind=kind.value, error=type(exc).__name__)
        detail = f" ({exc})" if isinstance(exc, errors.ConfigurationError) else ""
        backend.startup_error = errors.user_message(kind) + detail
    log_event("web_started", ready=backend.ready, langsmith=langsmith.enabled)
    return backend


def bind_context(session: AgentSession) -> None:
    """Log context for this rerun; Streamlit may run each rerun on a different thread."""
    logger.user_id_var.set(session.user.user_id)
    logger.conversation_id_var.set(session.conversation_id)


def transcript(session: AgentSession) -> list[tuple[str, str]]:
    """(question, answer) pairs of the current conversation, oldest first."""
    messages = session.state().get("messages") or []
    return memory.recent_exchanges(messages, n=len(messages))


def access_label(user: auth.UserContext) -> str:
    label = user.scope_label
    if user.categories:
        label += f" ({', '.join(user.categories)})"
    return label


def langsmith_url(settings: Settings, trace_id: str) -> str | None:
    """Link to the LangSmith run of a turn, if tracing was on for it."""
    events = inspect.trace_events(inspect.read_jsonl(settings.log_path), trace_id)
    run_id = inspect.langsmith_run_id(events)
    return tracing.run_url(run_id) if run_id else None


def confirm_label(actions: Sequence[str]) -> str:
    labels = {CONFIRM_LABELS.get(a, "Confirm") for a in actions}
    return labels.pop() if len(labels) == 1 else "Confirm"


def short_date(iso: str) -> str:
    return iso[:16].replace("T", " ")
