"""Where the agent struggles, from `agent.jsonl` (system-level learning loop).

Groups the problem signals of the logged turns - failed turns, SQL errors by type, SQL
attempts exhausted, SQL Guard rejections, input-guard blocks, fallbacks, 👎 feedback - and
ranks the turns that collected the most of them, with the question (from the masked LLM
payload log) and the trace id for `/trace`. The output says what to fix next: a missing
golden trio, a glossary term, a prompt rule or a guard that is too strict.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Group
from rich.table import Table
from rich.text import Text

from retail_agent.observability.inspect import Event

WEIGHTS = {
    "turn_failed": 3,
    "sql_attempts_exhausted": 3,
    "bad_feedback": 3,
    "sql_error": 1,
    "sql_rejected": 1,
    "guard_blocked": 1,
    "fallback_used": 1,
}
LABELS = {
    "turn_failed": "turn failed",
    "sql_attempts_exhausted": "SQL attempts exhausted",
    "bad_feedback": "negative feedback",
    "sql_error": "SQL error",
    "sql_rejected": "SQL Guard rejection",
    "guard_blocked": "input guard block",
    "fallback_used": "fallback model used",
}


@dataclass
class TurnProblems:
    trace_id: str
    user_id: str | None = None
    ts: str = ""
    question: str = ""
    problems: Counter = field(default_factory=Counter)
    details: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(WEIGHTS[k] * n for k, n in self.problems.items())


@dataclass
class FailureReport:
    turns: int  # turns in the period
    problem_turns: list[TurnProblems]  # sorted, worst first
    totals: Counter  # problem kind -> occurrences
    sql_errors: Counter  # "ERROR(syntax)" / "rejected: pii_projection" -> count
    blocks: Counter  # input-guard category -> count
    turn_errors: Counter  # error kind of failed turns -> count
    repeated: list[tuple[str, int]]  # questions with problems more than once


def _question(payloads: list[Event]) -> str:
    """The user's message: the last human message of the turn's first main-model request."""
    for record in payloads:
        if record.get("purpose") != "main":
            continue
        request = record.get("request") or []
        humans = [m.get("text", "") for m in request if m.get("type") == "human"]
        if humans:
            return " ".join(str(humans[-1]).replace("\ufeff", "").split())
    return ""


def analyze(
    events: Iterable[Event],
    payloads: Iterable[Event] = (),
    *,
    since: datetime | None = None,
    user_id: str | None = None,
    max_sql_failures: int = 3,
) -> FailureReport:
    turns: dict[str, TurnProblems] = {}
    seen: set[str] = set()
    totals: Counter = Counter()
    sql_errors: Counter = Counter()
    blocks: Counter = Counter()
    turn_errors: Counter = Counter()

    def hit(e: Event, kind: str, detail: str) -> None:
        trace_id = e["trace_id"]
        turn = turns.setdefault(trace_id, TurnProblems(trace_id, e.get("user_id"), e.get("ts", "")))
        turn.problems[kind] += 1
        turn.details.append(detail)
        totals[kind] += 1

    for e in events:
        trace_id = e.get("trace_id")
        if not trace_id:
            continue
        if since is not None and e.get("ts", "") < since.isoformat():
            continue
        if user_id is not None and e.get("user_id") != user_id:
            continue
        name = e.get("event")
        if name == "turn_start":
            seen.add(trace_id)
        elif name == "turn_end" and e.get("status") in ("error", "cancelled"):
            kind = e.get("error_kind") or e.get("status")
            turn_errors[kind] += 1
            hit(e, "turn_failed", f"turn {e.get('status')}: {kind}")
        elif name == "sql_attempt_failed":
            status = str(e.get("status", "?"))
            sql_errors[status] += 1
            hit(e, "sql_error", status)
            if int(e.get("failures") or 0) >= max_sql_failures:
                hit(e, "sql_attempts_exhausted", f"{max_sql_failures} SQL attempts failed")
        elif name == "sql_rejected":
            rule = e.get("rule", "?")
            sql_errors[f"rejected: {rule}"] += 1
            hit(e, "sql_rejected", f"SQL Guard: {rule}")
        elif name == "guard_blocked":
            category = e.get("category", "?")
            blocks[category] += 1
            hit(e, "guard_blocked", f"blocked as {category}: {e.get('reason', '')}"[:120])
        elif name == "fallback_used":
            turn = turns.get(trace_id)
            if turn is None or not turn.problems["fallback_used"]:  # once per turn
                hit(e, "fallback_used", f"fallback ({e.get('reason', '?')})")
        elif name == "feedback" and e.get("rating") == "bad":
            comment = e.get("comment")
            hit(e, "bad_feedback", f"👎 {comment}" if comment else "👎")

    for trace_id in set(turns) - seen:  # e.g. start-up checks: not a user turn
        for kind, n in turns.pop(trace_id).problems.items():
            totals[kind] -= n
    totals = +totals
    by_trace: dict[str, list[Event]] = defaultdict(list)
    for record in payloads:
        if record.get("trace_id") in turns:
            by_trace[record["trace_id"]].append(record)
    for trace_id, turn in turns.items():
        turn.question = _question(by_trace.get(trace_id, []))

    ranked = sorted(turns.values(), key=lambda t: (-t.score, t.ts))
    questions = Counter(t.question.lower() for t in ranked if t.question)
    repeated = [(q, n) for q, n in questions.most_common() if n > 1]
    return FailureReport(
        turns=len(seen),
        problem_turns=ranked,
        totals=totals,
        sql_errors=sql_errors,
        blocks=blocks,
        turn_errors=turn_errors,
        repeated=repeated,
    )


def _counter_table(title: str, counter: Counter, label: str) -> Table:
    table = Table(title=title, show_lines=False)
    table.add_column(label)
    table.add_column("count", justify="right")
    for key, n in counter.most_common():
        table.add_row(Text(str(key)), str(n))
    return table


def render(report: FailureReport, top: int = 10) -> Group:
    parts: list = []
    with_problems = len(report.problem_turns)
    parts.append(Text(f"{with_problems} of {report.turns} turns had problems."))
    if not report.problem_turns:
        return Group(*parts)
    totals = Counter({LABELS[k]: n for k, n in report.totals.items()})
    parts.append(_counter_table("Problems", totals, "kind"))
    if report.sql_errors:
        parts.append(_counter_table("SQL errors by type", report.sql_errors, "type"))
    if report.turn_errors:
        parts.append(_counter_table("Failed turns by cause", report.turn_errors, "cause"))
    if report.blocks:
        parts.append(_counter_table("Input-guard blocks", report.blocks, "category"))

    worst = Table(title=f"Top {min(top, with_problems)} problem turns", show_lines=False)
    for column in ("trace", "user", "time (UTC)", "score", "question", "problems"):
        worst.add_column(column, overflow="fold")
    for t in report.problem_turns[:top]:
        worst.add_row(
            t.trace_id,
            t.user_id or "-",
            t.ts[:16].replace("T", " "),
            str(t.score),
            Text(t.question[:160] or "-"),
            Text("; ".join(dict.fromkeys(t.details))[:300]),
        )
    parts.append(worst)
    if report.repeated:
        repeated = Table(title="Questions that failed more than once", show_lines=False)
        repeated.add_column("question", overflow="fold")
        repeated.add_column("turns", justify="right")
        for q, n in report.repeated[:top]:
            repeated.add_row(Text(q[:200]), str(n))
        parts.append(repeated)
    parts.append(
        Text(
            "Next: open a turn with `python -m retail_agent.observability.inspect trace <id>`; "
            "recurring SQL errors on one topic usually call for a golden trio or a glossary term.",
            style="dim",
        )
    )
    return Group(*parts)
