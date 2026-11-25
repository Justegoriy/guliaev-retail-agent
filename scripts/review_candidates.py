"""Audit the Golden Bucket: the human part of the otherwise automatic update loop.

    python scripts/review_candidates.py            # quarantined candidates
    python scripts/review_candidates.py --audit    # sampled auto-approved trios
    python scripts/review_candidates.py --list     # all candidates and their status

Candidates are approved automatically by `scripts/promote_candidates.py`; people look only
at what the pipeline was unsure about (quarantined) and at a sample of what it approved.
For a quarantined candidate: [a]pprove, [e]dit then approve, [r]eject, [s]kip or [q]uit.
On approval every SQL query must pass the SQL Guard and a BigQuery dry-run; the notes
default to the method the pipeline wrote, never to the user's answer (it holds the
author's numbers). For a sampled trio: [k]eep or [d]eprecate.
"""

import argparse
import sys

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

from retail_agent.auth import UserContext
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.errors import AgentError
from retail_agent.knowledge.golden import (
    Candidate,
    CandidateStore,
    approve_candidate,
    deprecate_trio,
    mark_audited,
    reject_candidate,
    slugify,
)
from retail_agent.knowledge.promotion import audit_queue
from retail_agent.observability import logger
from retail_agent.safety import sql_guard
from retail_agent.settings import get_settings

console = Console()
REVIEWER = UserContext("golden_review", "Golden Bucket reviewer", "ceo")  # full access, no ACL


def ask(prompt: str, default: str = "") -> str:
    shown = default if len(default) <= 60 else "Enter = keep"
    suffix = f" \\[{escape(shown)}]" if default else ""
    answer = console.input(f"{prompt}{suffix}: ").strip()
    return answer or default


def ask_sql(current: tuple[str, ...]) -> list[str]:
    console.print(
        "Enter the SQL queries; finish each query with a line containing only ';;', "
        "finish the list with an empty line. Just press Enter to keep the current SQL."
    )
    queries: list[str] = []
    lines: list[str] = []
    while True:
        line = console.input("sql> ")
        if line.strip() == ";;":
            if lines:
                queries.append("\n".join(lines))
            lines = []
        elif not line.strip() and not lines:
            break
        else:
            lines.append(line)
    if lines:
        queries.append("\n".join(lines))
    return queries or list(current)


def show(candidate: Candidate, index: int, total: int) -> None:
    console.rule(f"Candidate {index}/{total}: {candidate.id}")
    console.print(
        f"[bold]Question:[/bold] {candidate.question}\n"
        f"[dim]role: {candidate.user_role} | trace: {candidate.trace_id} | "
        f"created: {candidate.created_at}[/dim]\n"
        f"[bold]Tags:[/bold] {', '.join(candidate.tags) or '-'}"
    )
    for i, query in enumerate(candidate.sql, start=1):
        console.print(Syntax(query, "sql", word_wrap=True), f"[dim]query {i}[/dim]")
    console.print(Panel(candidate.answer_summary, title="Answer the user liked"))
    if candidate.checks:
        console.print(
            f"[bold]Pipeline:[/bold] {escape(str(candidate.checks.get('reason', '')))} "
            f"[dim](score {candidate.checks.get('score', '-')}, "
            f"leaks {candidate.checks.get('leaks') or '-'})[/dim]"
        )


def make_validator(runner: BigQueryRunner, default_limit: int):
    def validate(sql: str) -> None:
        guarded = sql_guard.check_and_rewrite(
            sql, REVIEWER, dataset=runner.dataset_id, default_limit=default_limit
        )
        runner.dry_run(guarded.sql)

    return validate


def review(store: CandidateStore, trios_path, validate) -> None:
    pending = [c for c in store.all() if c.status == "quarantined"]
    if not pending:
        console.print("No quarantined candidates.")
        return
    for i, candidate in enumerate(pending, start=1):
        show(candidate, i, len(pending))
        while True:
            action = ask(r"\[a]pprove / \[e]dit / \[r]eject / \[s]kip / \[q]uit", "s").lower()
            if action == "q":
                return
            if action == "s":
                break
            if action == "r":
                reject_candidate(candidate, store=store, note=ask("Reason (optional)"))
                console.print("[yellow]Rejected.[/yellow]")
                break
            if action in ("a", "e"):
                edits = {}
                if action == "e":
                    edits = {
                        "question": ask("Question", candidate.question),
                        "sql": ask_sql(candidate.sql),
                    }
                edits["report"] = ask(
                    "Analyst notes: method and metric definitions, no numbers",
                    candidate.method,
                )
                if not edits["report"]:
                    console.print("[red]Notes are required.[/red]")
                    continue
                question = edits.get("question", candidate.question)
                tags = ask("Tags, comma-separated", ", ".join(candidate.tags))
                trio_id = ask("Trio id", slugify(question))
                try:
                    trio = approve_candidate(
                        candidate,
                        store=store,
                        trios_path=trios_path,
                        validate_sql=validate,
                        trio_id=trio_id,
                        tags=[t.strip() for t in tags.split(",") if t.strip()],
                        **edits,
                    )
                except (AgentError, ValueError) as exc:
                    console.print(f"[red]Not approved: {exc}[/red] Edit the SQL or reject.")
                    continue
                console.print(f"[green]Approved as trio {trio.id!r}.[/green]")
                break


def audit(store: CandidateStore, trios_path) -> None:
    _, sampled = audit_queue(store, trios_path)
    pending = [t for t in sampled if not (t.source or {}).get("audited_at")]
    if not pending:
        console.print("No sampled trios waiting for the audit.")
        return
    for i, trio in enumerate(pending, start=1):
        console.rule(f"Auto-approved trio {i}/{len(pending)}: {trio.id}")
        console.print(f"[bold]Question:[/bold] {trio.question}")
        for query in trio.sql:
            console.print(Syntax(query, "sql", word_wrap=True))
        console.print(Panel(trio.report, title="Method notes"))
        action = ask(r"\[k]eep / \[d]eprecate / \[q]uit", "k").lower()
        if action == "q":
            return
        if action == "d":
            deprecate_trio(trios_path, trio.id, ask("Reason", "failed the audit"))
            console.print("[yellow]Deprecated.[/yellow]")
        else:
            mark_audited(trios_path, trio.id)
            console.print("[green]Kept.[/green]")


def list_all(store: CandidateStore) -> None:
    for c in store.all():
        detail = f" -> {c.trio_id}" if c.trio_id else ""
        console.print(f"{c.id}  {c.status:<9} {c.created_at}  {c.question}{detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list all candidates and exit")
    parser.add_argument("--audit", action="store_true", help="audit sampled auto-approved trios")
    args = parser.parse_args(argv)
    settings = get_settings()
    logger.configure(settings.log_path)
    store = CandidateStore(settings.golden_candidates_path)
    if args.list:
        list_all(store)
        return 0
    if args.audit:
        try:
            audit(store, settings.golden_trios_path)
        except (EOFError, KeyboardInterrupt):
            console.print()
        return 0
    runner = BigQueryRunner.from_settings(settings)
    try:
        review(
            store, settings.golden_trios_path, make_validator(runner, settings.sql_default_limit)
        )
    except (EOFError, KeyboardInterrupt):
        console.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
