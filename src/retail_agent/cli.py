"""Command-line chat interface."""

import argparse
import sys
from collections.abc import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

from retail_agent import auth, errors, health
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.data.storage import Conversation, ConversationAccessError
from retail_agent.graph.builder import (
    STAGE_START,
    AgentSession,
    PendingConfirmation,
    create_session,
)
from retail_agent.observability import inspect, logger, tracing
from retail_agent.observability.logger import log_event
from retail_agent.personalization.profile import PreferenceRejected, display
from retail_agent.settings import Settings, get_settings
from retail_agent.turns import Turn, TurnResult

EXIT_COMMANDS = {"/exit", "/quit"}
RESUME_SHOW_TURNS = 3
RESUME_ANSWER_CHARS = 600
HELP = """\
Ask a question about sales in plain language, or use a command:

  /help                         this help
  /reports                      your saved reports
  /prefs                        your preferences; /prefs set <key> <value> | forget <key> | reset
  /good, /bad [comment]         rate the last answer
  /new                          start a new conversation
  /history                      your recent conversations
  /resume <id>                  continue a conversation from /history
  /trace [id] [--full]          what happened during a turn (default: the last one)
  /stats [all]                  usage and quality metrics (all users: full-access roles only)
  /exit                         quit (also Ctrl+D; Ctrl+C cancels a running request)
"""

console = Console()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="retail-agent", description="Chat with your retail data in plain language."
    )
    parser.add_argument(
        "--user", help="demo user id from config/users.yaml (e.g. ceo, bottoms_manager)"
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", metavar="ID", help="continue a conversation (id or prefix)")
    resume.add_argument(
        "--continue", dest="continue_last", action="store_true", help="continue the last one"
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
        return PromptSession[str](history=InMemoryHistory()).prompt
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


# -- turns ----------------------------------------------------------------------------


def _with_status(action: Callable[[Callable[[str], None]], TurnResult]) -> TurnResult:
    with Status(STAGE_START, console=console) as status:
        return action(status.update)


def _run_turn(session: AgentSession, text: str, read: Callable[[str], str]) -> None:
    turn = Turn(session)
    result = _with_status(lambda on_stage: turn.start(text, on_stage))
    next_message: str | None = None
    while result.status == "pending" and result.outcome is not None:
        answers, typed = _ask_confirmations(result.outcome.pending, read)
        next_message = next_message or typed
        result = _with_status(lambda on_stage, answers=answers: turn.resume(answers, on_stage))
    if result.status == "cancelled":
        console.print(f"[yellow]{result.message}[/yellow]")
    elif result.status == "error":
        style = "yellow" if result.error_kind == "step_limit" else "red"
        console.print(f"[{style}]{result.message}[/{style}]")
    else:
        console.print(Markdown(result.answer))
        console.print(f"[dim]trace: {result.trace_id}[/dim]")
    if next_message:
        _run_turn(session, next_message, read)


# -- commands -------------------------------------------------------------------------


def _report_failure(event: str, exc: Exception) -> None:
    kind = errors.classify(exc)
    log_event(event, level="error", error=type(exc).__name__, detail=str(exc)[:300])
    console.print(f"[red]{errors.user_message(kind)}[/red]")


def _good(session: AgentSession) -> None:
    try:
        result = session.mark_good()
    except Exception as exc:
        _report_failure("feedback_failed", exc)
        return
    console.print(f"[green]{result.message}[/green]")


def _bad(session: AgentSession, comment: str) -> None:
    try:
        result = session.mark_bad(comment)
    except Exception as exc:
        _report_failure("feedback_failed", exc)
        return
    console.print(f"[green]{result.message}[/green]")


def _show_profile(session: AgentSession) -> None:
    profile = session.profile()
    if not profile.items:
        console.print("No saved preferences yet. Example: /prefs set format table")
        return
    table = Table(title="Your preferences", show_lines=False)
    for column in ("key", "value", "source", "confidence", "updated (UTC)"):
        table.add_column(column, overflow="fold")
    for p in profile.items:
        table.add_row(
            p.key,
            display(p.value),
            p.source,
            f"{p.confidence:.2f}",
            p.updated_at[:16].replace("T", " "),
        )
    console.print(table)


def _prefs(session: AgentSession, args: str) -> None:
    """`/prefs`, `/prefs set <key> <value>`, `/prefs forget <key>`, `/prefs reset`."""
    command, _, rest = args.strip().partition(" ")
    key, _, value = rest.strip().partition(" ")
    try:
        if not command:
            _show_profile(session)
        elif command == "set" and key and value.strip():
            saved = session.set_preference(key, value.strip())
            console.print(f"[green]Saved: {saved.key} = {display(saved.value)}[/green]")
        elif command == "forget" and key and not value:
            done = session.forget_preference(key)
            console.print(f"Forgot {key}." if done else f"{key} was not set.")
        elif command == "reset" and not rest.strip():
            removed = session.reset_preferences()
            console.print(f"Preferences reset ({removed} removed).")
        else:
            allowed = session.preferences.config.describe() if session.preferences else "-"
            console.print(
                "Usage: /prefs | /prefs set <key> <value> | /prefs forget <key> | /prefs reset\n"
                f"Keys: {allowed}"
            )
    except PreferenceRejected as exc:
        console.print(f"[yellow]Not saved: {exc}[/yellow]")
    except Exception as exc:
        _report_failure("preferences_command_failed", exc)


def _date(iso: str) -> str:
    return iso[:16].replace("T", " ")


def _reports(session: AgentSession) -> None:
    reports = session.reports()
    if not reports:
        console.print("No saved reports yet. Ask for an analysis and say 'save it as a report'.")
        return
    table = Table(title="Your reports", show_lines=False)
    for column in ("id", "created (UTC)", "title", "preview"):
        table.add_column(column, overflow="fold")
    for r in reports:
        here = " (this conversation)" if r.conversation_id == session.conversation_id else ""
        table.add_row(r.id, _date(r.created_at), r.title + here, r.snippet(80))
    console.print(table)


def _history(session: AgentSession) -> None:
    conversations = session.conversations()
    if not conversations:
        console.print("No conversations yet.")
        return
    table = Table(title="Your conversations", show_lines=False)
    for column in ("id", "updated (UTC)", "title", "turns"):
        table.add_column(column, overflow="fold")
    for c in conversations:
        marker = " *" if c.id == session.conversation_id else ""
        table.add_row(c.id[:8] + marker, _date(c.updated_at), c.title, str(c.turns))
    console.print(table)
    console.print("[dim]Continue one with /resume <id>; * marks the current conversation.[/dim]")


def _show_recent(session: AgentSession) -> None:
    for question, answer in session.recent_exchanges(RESUME_SHOW_TURNS):
        console.print(f"[bold]you>[/bold] {question}")
        if len(answer) > RESUME_ANSWER_CHARS:
            # Cut on a line boundary so a table is never left half-drawn.
            cut = answer.rfind("\n", 0, RESUME_ANSWER_CHARS)
            answer = answer[: cut if cut > 0 else RESUME_ANSWER_CHARS].rstrip() + "\n\n..."
        console.print(Markdown(answer), style="dim")


def _switch(session: AgentSession, target: str) -> Conversation | None:
    """Switch to one of the user's conversations by id or prefix."""
    if not target:
        console.print("Usage: /resume <id> (see /history)")
        return None
    try:
        conversation = session.find_conversation(target)
    except ConversationAccessError:
        console.print("[red]That conversation belongs to another user.[/red]")
        return None
    if conversation is None:
        console.print(
            f"[yellow]No conversation of yours matches {target!r}; see /history.[/yellow]"
        )
        return None
    session.switch_conversation(conversation.id)
    log_event("conversation_resumed", turns=conversation.turns)
    return conversation


def _show_resumed(session: AgentSession, conversation: Conversation) -> None:
    console.print(f"Resumed [bold]{conversation.title}[/bold] ({conversation.turns} turns).")
    _show_recent(session)
    if session.pending():
        console.print("[dim]A confirmation was pending; your next message will cancel it.[/dim]")


def _resume(session: AgentSession, target: str) -> None:
    conversation = _switch(session, target)
    if conversation is not None:
        _show_resumed(session, conversation)


def _new(session: AgentSession) -> None:
    conversation_id = session.new_conversation()
    log_event("conversation_started")
    console.print(f"New conversation {conversation_id[:8]}.")


def _trace(session: AgentSession, settings: Settings, args: str) -> None:
    words = args.split()
    full = "--full" in words
    words = [w for w in words if w != "--full"]
    events = list(inspect.read_jsonl(settings.log_path))
    user_id = session.user.user_id
    if words:
        trace_id = inspect.resolve_trace_id(events, words[0], user_id)
    else:
        trace_id = inspect.last_trace_id(events, user_id)
    if trace_id is None:
        target = words[0] if words else "(last turn)"
        console.print(f"[yellow]No turn of yours matches {target}.[/yellow]")
        return
    turn = inspect.trace_events(events, trace_id)
    payloads = inspect.trace_payloads(settings.llm_payload_log_path, trace_id) if full else None
    run_id = inspect.langsmith_run_id(turn)
    url = tracing.run_url(run_id) if run_id else None
    console.print(inspect.render_trace(trace_id, turn, payloads, url))


def _stats(session: AgentSession, settings: Settings, args: str) -> None:
    everyone = args.strip() == "all"
    if everyone and not session.user.has_full_access:
        console.print("[yellow]Only full-access roles can see everyone's statistics.[/yellow]")
        return
    user_id = None if everyone else session.user.user_id
    events = inspect.read_jsonl(settings.log_path)
    title = "Statistics (all users)" if everyone else f"Statistics ({session.user.user_id})"
    console.print(inspect.render_stats(inspect.compute_stats(events, settings, user_id), title))


def _dispatch(session: AgentSession, settings: Settings, text: str) -> bool:
    """Run a slash command; False when `text` is not one."""
    command, _, args = text.partition(" ")
    command = command.lower()
    handlers: dict[str, Callable[[], object]] = {
        "/help": lambda: console.print(HELP),
        "/good": lambda: _good(session),
        "/bad": lambda: _bad(session, args.strip()),
        "/prefs": lambda: _prefs(session, args),
        "/reports": lambda: _reports(session),
        "/new": lambda: _new(session),
        "/history": lambda: _history(session),
        "/resume": lambda: _resume(session, args.strip()),
        "/trace": lambda: _trace(session, settings, args),
        "/stats": lambda: _stats(session, settings, args),
    }
    handler = handlers.get(command)
    if handler is None:
        if command.startswith("/") and command[1:].isalpha():
            console.print(f"[yellow]Unknown command {command}; see /help.[/yellow]")
            return True
        return False
    try:
        handler()
    except Exception as exc:
        _report_failure("command_failed", exc)
    return True


def _repl(session: AgentSession, settings: Settings) -> None:
    read = _make_reader()
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
        if _dispatch(session, settings, text):
            continue
        _run_turn(session, text, read)


# -- startup --------------------------------------------------------------------------


def _print_check(check: health.Check) -> None:
    if check.ok:
        console.print(f"[green]✓[/green] {check.message}")
    elif check.fatal:
        console.print(f"[red]✗ {check.message}[/red]")
    else:
        console.print(f"[yellow]! {check.message}[/yellow]")


def _banner(
    session: AgentSession, settings: Settings, langsmith: tracing.TracingStatus, resumed: bool
) -> None:
    user = session.user
    access = user.scope_label
    if user.categories:
        access += f" ({', '.join(user.categories)})"
    trace_line = f"on (project {settings.langsmith_project})" if langsmith.enabled else "off"
    conversation = "resumed" if resumed else "new"
    lines = [
        f"[bold]{user.name}[/bold] ({user.user_id}), role: {user.role}",
        f"Data access: {access}",
        f"Model: {settings.gemini_model}, fallback {settings.gemini_fallback_model}, "
        f"aux {settings.gemini_aux_model}",
        f"LangSmith tracing: {trace_line}",
        f"Conversation: {session.conversation_id[:8]} ({conversation})",
    ]
    console.print(Panel("\n".join(lines), title="Retail analytics agent", expand=False))
    console.print("Ask a question about sales, or type /help for commands.")


def _start_session(user: auth.UserContext, settings: Settings) -> AgentSession | None:
    runner = BigQueryRunner.from_settings(settings)
    with console.status("Checking connections..."):
        checks = [health.check_gemini(settings), health.check_bigquery(runner, settings)]
    for check in checks:
        _print_check(check)
    if any(c.fatal for c in checks):
        return None
    try:
        with console.status("Loading schema and knowledge..."):
            return create_session(user, settings, runner=runner)
    except Exception as exc:
        kind = errors.classify(exc)
        log_event("startup_failed", level="error", error_kind=kind.value, error=type(exc).__name__)
        detail = f" ({exc})" if isinstance(exc, errors.ConfigurationError) else ""
        console.print(f"[red]{errors.user_message(kind)}{detail}[/red]")
        return None


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
        session = _start_session(user, settings)
    except KeyboardInterrupt:
        return 130
    if session is None:
        return 2

    resumed: Conversation | None = None
    if args.resume:
        resumed = _switch(session, args.resume)
    elif args.continue_last:
        last = session.last_conversation()
        if last is None:
            console.print("No previous conversation; starting a new one.")
        else:
            resumed = _switch(session, last.id)
    log_event(
        "session_start",
        role=user.role,
        category_groups=list(user.category_groups),
        categories=list(user.categories),
        langsmith=tracing_status.enabled,
        resumed=resumed is not None,
    )
    _banner(session, settings, tracing_status, resumed is not None)
    if resumed is not None:
        _show_resumed(session, resumed)
    try:
        _repl(session, settings)
    finally:
        log_event("session_end")
    console.print("Bye!")
    return 0
