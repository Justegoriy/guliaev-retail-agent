"""Chat history is kept 90 days; reports stay until deleted; the audit log one year."""

import json
from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage

from harness import Harness
from retail_agent import retention
from retail_agent.knowledge.golden import CandidateStore

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)


def iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")


def age_conversation(h: Harness, conversation_id: str, days: float) -> None:
    h.storage._write(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (iso(days), conversation_id)
    )


def thread_exists(h: Harness, conversation_id: str) -> bool:
    config = {"configurable": {"thread_id": conversation_id}}
    return h.session.graph.checkpointer.get_tuple(config) is not None


def test_old_conversations_are_purged_with_their_checkpoints(tmp_path):
    h = Harness(tmp_path, conversation_id="old")
    h.turn("hello", AIMessage("Hi!"))
    h.session.switch_conversation("recent")
    h.turn("hello again", AIMessage("Hi again!"))
    report_id = h.storage.save_report("ceo", "old", "Q1 review", "# Q1").id
    h.storage.audit("ceo", "report_exported", [report_id])
    age_conversation(h, "old", 91)
    age_conversation(h, "recent", 89)

    result = retention.purge_expired(h.settings, h.storage, h.session.graph.checkpointer, now=NOW)
    assert result.conversations == 1
    assert h.storage.get_conversation("old") is None and not thread_exists(h, "old")
    assert h.storage.get_conversation("recent") is not None and thread_exists(h, "recent")
    # Saved reports and the audit log follow their own rules.
    assert h.storage.get_report("ceo", report_id) is not None
    assert h.storage.audit_entries("ceo")


def test_resuming_a_purged_conversation_finds_nothing(tmp_path):
    h = Harness(tmp_path, conversation_id="old")
    h.turn("hello", AIMessage("Hi!"))
    age_conversation(h, "old", 120)
    retention.purge_expired(h.settings, h.storage, h.session.graph.checkpointer, now=NOW)
    assert h.session.conversations() == []
    assert h.session.state().get("messages") in (None, [])


def write_lines(path, *ages, extra=()):
    lines = [json.dumps({"ts": iso(a), "event": f"e{a}"}) for a in ages]
    path.write_text("\n".join([*lines, *extra]) + "\n", encoding="utf-8")


def test_logs_and_payloads_keep_only_the_last_90_days(tmp_path):
    h = Harness(tmp_path)
    write_lines(h.settings.log_path, 100, 91, 89, 1, extra=["not json"])
    write_lines(h.settings.llm_payload_log_path, 200, 2)
    result = retention.purge_expired(h.settings, h.storage, h.session.graph.checkpointer, now=NOW)
    assert (result.log_lines, result.payload_lines) == (2, 1)
    kept = h.settings.log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in kept[:2]] == ["e89", "e1"]
    assert "not json" in kept  # unparseable lines are never dropped by mistake
    assert len(h.settings.llm_payload_log_path.read_text(encoding="utf-8").splitlines()) == 1


def test_old_golden_candidates_are_dropped(tmp_path):
    h = Harness(tmp_path)
    store = CandidateStore(tmp_path / "cand.jsonl")
    for trace, days in (("old", 95), ("new", 5)):
        store.add(
            question="Monthly revenue",
            sql=["SELECT 1"],
            answer_summary="Revenue: $300.",
            user_role="ceo",
            trace_id=trace,
        )
        store.replace_all(
            [c if c.trace_id != trace else c.__class__(**{**c.__dict__, "created_at": iso(days)})
             for c in store.all()]
        )  # fmt: skip
    result = retention.purge_expired(
        h.settings, h.storage, h.session.graph.checkpointer, store, now=NOW
    )
    assert result.candidates == 1
    assert [c.trace_id for c in store.all()] == ["new"]


def test_nothing_to_purge_is_quiet(tmp_path):
    h = Harness(tmp_path)
    h.turn("hello", AIMessage("Hi!"))
    result = retention.purge_expired(h.settings, h.storage, h.session.graph.checkpointer)
    assert not any(result.as_dict().values())


def test_retention_period_is_configurable(tmp_path):
    h = Harness(tmp_path, chat_retention_days=7, conversation_id="week_old")
    h.turn("hello", AIMessage("Hi!"))
    age_conversation(h, "week_old", 8)
    result = retention.purge_expired(h.settings, h.storage, h.session.graph.checkpointer, now=NOW)
    assert result.conversations == 1
