"""Command-line chat interface."""

import argparse
import sys
import time
from collections.abc import Callable

from langgraph.errors import GraphRecursionError
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from retail_agent import auth, errors
from retail_agent.graph.builder import AgentSession, create_session
from retail_agent.observability import logger, tracing
from retail_agent.observability.logger import log_event
from retail_agent.settings import get_settings

EXIT_COMMANDS = {"/exit", "/quit"}
STEP_LIMIT_MESSAGE = (
    "The request turned out to be too complex to finish in one go. "
    "Please narrow it down or split it into smaller questions."
)

console = Console()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="retail-agent", description="Chat with your retail data in plain language."
    )
    parser.add_argument(
        "--user", help="demo user id from config/users.yaml (e.g. ceo, bottoms_manager)"
    )
    parser.add_argument("--debug", action="store_true", help="print log events to stderr")
    return parser.parse_args(argv)


def _choose_user(users: dict[str, auth.UserContext]) -> auth.UserContext:
    table = Table(title="Demo users", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("id")
    table.add_column("name")
    table.add_column("role")
    table.add_column("data access")
    ordered = list(users.values())
    for i, user in enumerate(ordered, start=1):
        table.add_row(str(i), user.user_id, user.name, user.role, user.scope_label)
    console.print(table)
    while True:
        choice = console.input("Choose a user (number or id): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(ordered):
            return ordered[int(choice) - 1]
        if choice in users:
            return users[choice]
        console.print(f"[red]No such user: {choice!r}[/red]")


def _make_reader() -> Callable[[str], str]:
    # prompt_toolkit needs a real terminal; piped stdin (scripts, evals) falls back to input().
    if sys.stdin.isatty() and sys.stdout.isatty():
        return PromptSession[str]().prompt
    return input


def _run_turn(session: AgentSession, text: str) -> None:
    trace_id = logger.new_trace_id()
    log_event("turn_start", length=len(text))
    started = time.perf_counter()
    try:
        with console.status("Thinking..."):
            outcome = session.run_turn(text, trace_id)
    except GraphRecursionError:
        log_event("turn_end", level="warning", status="error", error_kind="step_limit")
        console.print(f"[yellow]{STEP_LIMIT_MESSAGE}[/yellow] [dim](trace: {trace_id})[/dim]")
        return
    except Exception as exc:
        kind = errors.classify(exc)
        log_event(
            "turn_end",
            level="error",
            status="error",
            error_kind=kind.value,
            error=type(exc).__name__,
            detail=str(exc)[:500],
        )
        console.print(f"[red]{errors.user_message(kind, trace_id)}[/red]")
        return
    log_event(
        "turn_end",
        status="blocked" if outcome.blocked else "ok",
        latency_ms=int((time.perf_counter() - started) * 1000),
        sql_count=len(session.deps.tool_ctx.turn.executed),
        sql_failures=session.deps.tool_ctx.turn.sql_failures,
        bq_bytes=session.deps.tool_ctx.turn.bq_bytes,
    )
    console.print(Markdown(outcome.answer))
    console.print(f"[dim]trace: {trace_id}[/dim]")


def _repl(session: AgentSession) -> None:
    user = session.user
    read = _make_reader()
    console.print(
        f"Hi {user.name} ({user.role}, access: {user.scope_label}). "
        "Ask a question about sales, or type /exit to quit."
    )
    while True:
        try:
            # Windows PowerShell prefixes piped stdin with a byte order mark.
            text = read("you> ").lstrip("\ufeff").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in EXIT_COMMANDS:
            break
        _run_turn(session, text)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    logger.configure(settings.log_path, debug=args.debug)
    tracing_status = tracing.setup_langsmith(settings)
    if tracing_status.warning:
        console.print(f"[yellow]{tracing_status.warning}[/yellow]")

    try:
        users = auth.load_users()
        user = auth.load_user(args.user) if args.user else _choose_user(users)
    except auth.UserConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    except (EOFError, KeyboardInterrupt):
        return 130

    logger.user_id_var.set(user.user_id)
    try:
        with console.status("Connecting..."):
            session = create_session(user, settings)
    except Exception as exc:
        kind = errors.classify(exc)
        log_event("startup_failed", level="error", error_kind=kind.value, error=type(exc).__name__)
        detail = f" ({exc})" if isinstance(exc, errors.ConfigurationError) else ""
        console.print(f"[red]{errors.user_message(kind)}{detail}[/red]")
        return 2
    log_event(
        "session_start",
        role=user.role,
        category_groups=list(user.category_groups),
        categories=list(user.categories),
        langsmith=tracing_status.enabled,
    )
    try:
        _repl(session)
    finally:
        log_event("session_end")
    console.print("Bye!")
    return 0
