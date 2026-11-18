"""Command-line chat interface."""

import argparse
import sys
import uuid
from collections.abc import Callable

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.table import Table

from retail_agent import auth
from retail_agent.observability import logger
from retail_agent.observability.logger import log_event
from retail_agent.settings import get_settings

EXIT_COMMANDS = {"/exit", "/quit"}

console = Console()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="retail-agent", description="Chat with your retail data in plain language."
    )
    parser.add_argument("--user", help="demo user id from config/users.yaml (e.g. ceo, head_men)")
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


def _repl(user: auth.UserContext) -> None:
    read = _make_reader()
    console.print(
        f"Hi {user.name} ({user.role}, access: {user.scope_label}). "
        "Ask a question about sales, or type /exit to quit."
    )
    while True:
        try:
            text = read("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in EXIT_COMMANDS:
            break
        logger.new_trace_id()
        log_event("user_message", length=len(text))
        console.print("[dim]The agent is not wired up yet.[/dim]")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    logger.configure(settings.log_path, debug=args.debug)

    try:
        users = auth.load_users()
        user = auth.load_user(args.user) if args.user else _choose_user(users)
    except auth.UserConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    except (EOFError, KeyboardInterrupt):
        return 130

    logger.user_id_var.set(user.user_id)
    logger.conversation_id_var.set(uuid.uuid4().hex)
    log_event("session_start", role=user.role, departments=list(user.departments))
    try:
        _repl(user)
    finally:
        log_event("session_end")
    console.print("Bye!")
    return 0
