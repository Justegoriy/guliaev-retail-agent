import io
import json

import pytest
from rich.console import Console

from harness import make_settings
from retail_agent.observability import inspect

T1, T2, T3 = "aaaa111122223333", "aaab444455556666", "bbbb777788889999"


def ev(event, trace=T1, user="ceo", ts="2026-11-22T10:00:00.000+00:00", **fields):
    return {
        "ts": ts,
        "level": fields.pop("level", "info"),
        "event": event,
        "trace_id": trace,
        "conversation_id": "conv1",
        "user_id": user,
        **fields,
    }


EVENTS = [
    ev("turn_start", length=20),
    ev("input_guard", category="analysis", source="llm", pii_request=False, blocked=False),
    ev("golden_retrieved", method="embedding", ids=["revenue_monthly"], scores=[0.8]),
    ev(
        "prompt_built",
        persona_version="p1",
        glossary_version="g1",
        golden_ids=["revenue_monthly"],
        preferences=["format=table (explicit)"],
        chars=9000,
    ),
    ev("llm_error", level="warning", model="gemini-main", error="LlmCoolingDown", detail="x"),
    ev("fallback_used", level="warning", from_model="gemini-main", to_model="gemini-lite"),
    ev(
        "llm_call",
        ts="2026-11-22T10:00:01.500+00:00",
        model="gemini-lite",
        purpose="main",
        model_role="fallback",
        latency_ms=1400,
        input_tokens=1000,
        output_tokens=100,
        tool_calls=["run_sql"],
    ),
    ev("run_sql", purpose="revenue", tables=["order_items"], acl=False),
    ev("sql_rejected", level="warning", rule="pii", reason="email column", sql="SELECT email"),
    ev("sql_attempt_failed", level="warning", status="SQL_REJECTED", failures=1),
    ev("sql_dry_run", sql="SELECT SUM(sale_price) FROM t LIMIT 1000", bytes=5_000_000),
    ev(
        "sql_executed",
        sql="SELECT SUM(sale_price) FROM t LIMIT 1000",
        rows=1,
        bytes=5_000_000,
        duration_ms=900,
        cache_hit=False,
        source="bigquery",
    ),
    ev("tool_call", tool="run_sql", status="ok", latency_ms=950, args="{'sql': '...'}"),
    ev("llm_call", model="gemini-lite", purpose="main", input_tokens=1200, output_tokens=200),
    ev("langsmith_run", run_id="run-1", project="retail-agent"),
    ev(
        "turn_end",
        ts="2026-11-22T10:00:04.000+00:00",
        status="ok",
        latency_ms=4000,
        sql_count=1,
        sql_failures=1,
        bq_bytes=5_000_000,
    ),
    ev("feedback", rating="good"),
    # Second turn of the same user: blocked by the guard.
    ev("turn_start", trace=T2, length=30),
    ev("input_guard", trace=T2, category="injection", source="rules", blocked=True),
    ev("guard_blocked", trace=T2, category="injection", reason="rules"),
    ev("turn_end", trace=T2, status="blocked", latency_ms=1000, sql_count=0, sql_failures=0),
    # Another user's turn that failed.
    ev("turn_start", trace=T3, user="denim_manager", length=10),
    ev("run_sql", trace=T3, user="denim_manager", purpose="jeans", tables=["products"], acl=True),
    ev("confirmation_requested", trace=T3, user="denim_manager", action="delete_reports"),
    ev("confirmation_rejected", trace=T3, user="denim_manager", action="delete_reports"),
    ev("pii_masked", trace=T3, user="denim_manager", where="tool_result", count=2),
    ev(
        "turn_end",
        trace=T3,
        user="denim_manager",
        status="error",
        error_kind="bq_unavailable",
        latency_ms=500,
    ),
]


def render(renderable) -> str:
    # Both sizes: with TERM=dumb rich ignores a width given alone.
    console = Console(file=io.StringIO(), width=250, height=100)
    console.print(renderable)
    return console.file.getvalue()


def test_read_jsonl_skips_torn_lines(tmp_path):
    path = tmp_path / "agent.jsonl"
    path.write_text(json.dumps(EVENTS[0]) + "\n{not json\n\n", encoding="utf-8")
    assert list(inspect.read_jsonl(path)) == [EVENTS[0]]
    assert list(inspect.read_jsonl(tmp_path / "missing.jsonl")) == []


def test_trace_resolution_by_prefix_and_owner():
    assert inspect.resolve_trace_id(EVENTS, T1) == T1
    assert inspect.resolve_trace_id(EVENTS, "aaaa") == T1
    assert inspect.resolve_trace_id(EVENTS, "aaa") is None  # ambiguous
    assert inspect.resolve_trace_id(EVENTS, "bbbb", user_id="ceo") is None
    assert inspect.last_trace_id(EVENTS, "ceo") == T2
    assert inspect.last_trace_id(EVENTS) == T3


def test_trace_events_stop_at_turn_end():
    events = inspect.trace_events(EVENTS, T1)
    assert events[0]["event"] == "turn_start" and events[-1]["event"] == "turn_end"
    assert all(e["event"] != "feedback" for e in events)


def test_render_trace_shows_the_turn():
    events = inspect.trace_events(EVENTS, T1)
    text = render(inspect.render_trace(T1, events, langsmith_url="https://smith/run-1"))
    for expected in (
        "category=analysis",
        "revenue_monthly (0.80)",
        "persona p1",
        "format=table (explicit)",
        "gemini-lite [main/fallback] 1000→100 tokens, 1400 ms, calls: run_sql",
        "gemini-main → gemini-lite",
        "rule=pii",
        "SELECT SUM(sale_price) FROM t LIMIT 1000",
        "would scan 5.0 MB",
        "status=ok",
        "LangSmith: https://smith/run-1",
    ):
        assert expected in text, expected
    assert "1500" in text  # offset of the first llm_call in ms
    assert inspect.langsmith_run_id(events) == "run-1"


def test_render_trace_full_shows_payloads():
    payloads = [
        {
            "trace_id": T1,
            "call_id": "c1",
            "model": "gemini-lite",
            "purpose": "main",
            "latency_ms": 10,
            "request": [
                {"type": "system", "text": "You are a retail analyst."},
                {"type": "human", "text": "Revenue 2024?"},
            ],
            "response": {
                "type": "ai",
                "text": "",
                "tool_calls": [{"name": "run_sql", "args": {"sql": "SELECT 1"}}],
            },
        },
        {"trace_id": T1, "call_id": "c2", "model": "m", "request": [], "error": "Boom: 503"},
    ]
    text = render(inspect.render_trace(T1, inspect.trace_events(EVENTS, T1), payloads))
    assert "You are a retail analyst." in text
    assert 'run_sql({"sql": "SELECT 1"})' in text
    assert "[error] Boom: 503" in text


def test_trace_payloads_are_filtered_by_trace(tmp_path):
    path = tmp_path / "payloads.jsonl"
    path.write_text(
        "\n".join(json.dumps({"trace_id": t, "call_id": t[-2:]}) for t in (T1, T2, T1)),
        encoding="utf-8",
    )
    assert [p["call_id"] for p in inspect.trace_payloads(path, T1)] == ["33", "33"]


def test_percentile():
    assert inspect.percentile([], 50) is None
    assert inspect.percentile([3, 1, 2], 50) == 2
    assert inspect.percentile(list(range(1, 101)), 95) == 95


def test_compute_stats(tmp_path):
    settings = make_settings(tmp_path, llm_price_per_1m_input=1.0, llm_price_per_1m_output=10.0)
    s = inspect.compute_stats(EVENTS, settings)
    assert s.turns == 3
    assert s.statuses == {"ok": 1, "blocked": 1, "error": 1}
    assert s.success_rate == pytest.approx(2 / 3)
    assert s.error_kinds == {"bq_unavailable": 1}
    assert sorted(s.latencies_ms) == [1000, 4000]
    assert (s.llm_calls, s.input_tokens, s.output_tokens) == (2, 2200, 300)
    assert s.cost_usd == pytest.approx((2200 * 1.0 + 300 * 10.0) / 1e6)
    assert s.bq_bytes == 5_000_000
    assert (s.turns_with_sql, s.turns_with_correction, s.corrections_succeeded) == (1, 1, 1)
    assert s.sql_rejected == {"pii": 1}
    assert s.guard_blocked == {"injection": 1}
    assert s.pii_masked == {"tool_result": 2}
    assert s.acl_queries == 1
    assert (s.fallbacks, s.llm_errors) == (1, 1)
    assert s.confirmations == {"requested": 1, "rejected": 1}
    assert s.feedback == {"good": 1}

    mine = inspect.compute_stats(EVENTS, settings, user_id="ceo")
    assert mine.turns == 2 and mine.success_rate == 1.0
    text = render(inspect.render_stats(mine))
    assert "Success rate" in text and "100%" in text and "4.0 s" in text


def test_standalone_main(tmp_path, monkeypatch, capsys):
    log = tmp_path / "agent.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in EVENTS), encoding="utf-8")
    settings = make_settings(tmp_path)
    monkeypatch.setattr(inspect, "get_settings", lambda: settings)
    assert inspect.main(["stats"]) == 0
    assert "Turns" in capsys.readouterr().out
    assert inspect.main(["trace", "aaab"]) == 0
    assert "injection" in capsys.readouterr().out
    assert inspect.main(["trace", "zzz"]) == 1
