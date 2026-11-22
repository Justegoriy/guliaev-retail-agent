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
from retail_agent.graph.builder import (
    AgentSession,
    PendingConfirmation,
    TurnOutcome,
    create_session,
)
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


def looks_like_new_message(answer: str) -> bool:
    """An answer to a y/N question that is really a new request (D12, scenario A32)."""
    return "?" in answer or len(answer.split()) > 3


def _show_confirmation(pending: PendingConfirmation) -> None:
    table = Table(title=pending.summary or None, show_lines=False)
    for column in ("id", "title", "created (UTC)", "preview"):
        table.add_column(column, overflow="fold")
    for item in pending.items:
        created = str(item.get("created_at", ""))[:16].replace("T", " ")
        table.add_row(
            str(item.get("id", "")),
            str(item.get("title", "")),
            created,
            str(item.get("snippet", "")),
        )
    console.print(table)


def _ask_confirmations(
    pending: tuple[PendingConfirmation, ...], read: Callable[[str], str]
) -> tuple[dict[str, str], str | None]:
    """Ask each question outside the LLM; returns the answers and a new message, if typed."""
    answers: dict[str, str] = {}
    for item in pending:
        _show_confirmation(item)
        try:
            answer = read(f"{item.question} [y/N] ").lstrip("\ufeff").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "no"
        if looks_like_new_message(answer):
            log_event("confirmation_replaced_by_message", action=item.action)
            return {}, answer  # unanswered confirmations count as "no"
        answers[item.interrupt_id] = answer
    return answers, None


def _call(action: Callable[[], TurnOutcome], trace_id: str) -> TurnOutcome | None:
    """Run a graph call; on failure print the user message, log `turn_end` and return None."""
    try:
        with console.status("Thinking..."):
            return action()
    except GraphRecursionError:
        log_event("turn_end", level="warning", status="error", error_kind="step_limit")
        console.print(f"[yellow]{STEP_LIMIT_MESSAGE}[/yellow] [dim](trace: {trace_id})[/dim]")
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
    return None


def _run_turn(session: AgentSession, text: str, read: Callable[[str], str]) -> None:
    trace_id = logger.new_trace_id()
    log_event("turn_start", length=len(text))
    started = time.perf_counter()
    outcome = _call(lambda: session.run_turn(text, trace_id), trace_id)
    next_message: str | None = None
    while outcome is not None and outcome.pending:
        answers, typed = _ask_confirmations(outcome.pending, read)
        next_message = next_message or typed
        outcome = _call(lambda answers=answers: session.resume(answers, trace_id), trace_id)
    if outcome is not None:
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
    if next_message:
        _run_turn(session, next_message, read)


def _good(session: AgentSession) -> None:
    try:
        result = session.mark_good()
    except Exception as exc:
        kind = errors.classify(exc)
        log_event("feedback_failed", level="error", error=type(exc).__name__, detail=str(exc)[:300])
        console.print(f"[red]{errors.user_message(kind)}[/red]")
        return
    console.print(f"[green]{result.message}[/green]")


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
        if text.lower() == "/good":
            _good(session)
            continue
        _run_turn(session, text, read)


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
