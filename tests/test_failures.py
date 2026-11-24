from datetime import UTC, datetime

from rich.console import Console

from retail_agent.observability import failures


def ev(name, trace, user="ceo", ts="2026-11-21T10:00:00", **fields):
    return {"event": name, "trace_id": trace, "user_id": user, "ts": ts, **fields}


EVENTS = [
    ev("turn_start", "t1"),
    ev("sql_attempt_failed", "t1", status="ERROR(syntax)", failures=1),
    ev("sql_attempt_failed", "t1", status="ERROR(runtime)", failures=2),
    ev("sql_attempt_failed", "t1", status="ERROR(syntax)", failures=3),
    ev("fallback_used", "t1", reason="cooldown"),
    ev("fallback_used", "t1", reason="cooldown"),
    ev("turn_end", "t1", status="error", error_kind="sql_failed"),
    ev("turn_start", "t2", user="denim_manager"),
    ev("sql_rejected", "t2", user="denim_manager", rule="pii_projection"),
    ev("turn_end", "t2", user="denim_manager", status="ok"),
    ev("turn_start", "t3"),
    ev("guard_blocked", "t3", category="off_topic", reason="poem"),
    ev("turn_end", "t3", status="blocked"),
    ev("feedback", "t3", rating="bad", comment="too long"),
    ev("turn_start", "t4", ts="2026-11-01T10:00:00"),
    ev("turn_end", "t4", ts="2026-11-01T10:00:00", status="ok"),
    ev("fallback_used", "startup", reason="quota"),  # health check, not a turn
]
PAYLOADS = [
    {
        "trace_id": "t1",
        "purpose": "main",
        "request": [
            {"type": "system", "text": "rules"},
            {"type": "human", "text": "old question"},
            {"type": "ai", "text": "old answer"},
            {"type": "human", "text": "\ufeffWhy did  churn grow?"},
        ],
    },
    {"trace_id": "t1", "purpose": "guard", "request": [{"type": "human", "text": "x"}]},
]


def test_problems_are_grouped_and_turns_ranked():
    report = failures.analyze(EVENTS, PAYLOADS, max_sql_failures=3)

    assert report.turns == 4
    assert [t.trace_id for t in report.problem_turns] == ["t1", "t3", "t2"]
    worst = report.problem_turns[0]
    assert worst.question == "Why did churn grow?"
    assert worst.problems["fallback_used"] == 1  # counted once per turn
    assert worst.score == 3 + 3 + 3 + 1  # failed, exhausted, 3 SQL errors, fallback
    assert report.sql_errors == {
        "ERROR(syntax)": 2,
        "ERROR(runtime)": 1,
        "rejected: pii_projection": 1,
    }
    assert report.blocks == {"off_topic": 1}
    assert report.turn_errors == {"sql_failed": 1}
    assert report.totals["bad_feedback"] == 1
    assert report.totals["fallback_used"] == 1  # the start-up check is not counted


def test_filters_by_period_and_user():
    since = datetime(2026, 11, 10, tzinfo=UTC)
    report = failures.analyze(EVENTS, since=since, user_id="denim_manager")
    assert report.turns == 1
    assert [t.trace_id for t in report.problem_turns] == ["t2"]


def test_render_lists_the_worst_turns():
    console = Console(width=200, height=200, record=True, file=None)
    with console.capture() as capture:
        console.print(failures.render(failures.analyze(EVENTS, PAYLOADS), top=2))
    text = capture.get()
    assert "3 of 4 turns had problems." in text
    assert "SQL attempts exhausted" in text and "Why did churn grow?" in text
    assert "t2" not in text.split("Top 2 problem turns")[1]


def test_no_problems():
    report = failures.analyze([ev("turn_start", "t"), ev("turn_end", "t", status="ok")])
    console = Console(width=120)
    with console.capture() as capture:
        console.print(failures.render(report))
    assert "0 of 1 turns had problems." in capture.get()
