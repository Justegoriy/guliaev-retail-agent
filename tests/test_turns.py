import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from harness import REVENUE_SQL, Harness, calls, sql_call, tool_call
from retail_agent.observability import inspect, logger
from retail_agent.turns import CANCELLED_MESSAGE, Turn


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    logger.configure(harness.settings.log_path)
    logger.user_id_var.set(harness.user.user_id)
    yield harness
    logger._log_path = None


def events(h: Harness, name: str) -> list[dict]:
    return [e for e in inspect.read_jsonl(h.settings.log_path) if e["event"] == name]


def test_answered_turn_is_logged_once_with_its_metrics(h):
    h.model.script.extend([sql_call(REVENUE_SQL), AIMessage("Revenue was 300.")])
    stages: list[str] = []

    result = Turn(h.session).start("revenue in 2024", on_stage=stages.append)

    assert result.status == "ok" and result.answer == "Revenue was 300."
    assert stages  # progress was reported
    (start,) = events(h, "turn_start")
    (end,) = events(h, "turn_end")
    assert start["trace_id"] == end["trace_id"] == result.trace_id
    assert end["status"] == "ok" and end["sql_count"] == 1 and end["llm_calls"] >= 2


def test_a_confirmation_pauses_the_turn_and_resume_finishes_it(h):
    report = h.storage.save_report("ceo", "conv1", "Q1 review", "text")
    h.model.script.append(
        calls(tool_call("request_report_deletion", {"report_ids": [report.id]}, "d1"))
    )
    turn = Turn(h.session)

    paused = turn.start(f"delete report {report.id}")

    assert paused.status == "pending" and paused.outcome.pending
    assert events(h, "turn_end") == []  # the turn is not over while it waits
    h.model.script.append(AIMessage("Deleted."))
    (pending,) = paused.outcome.pending
    done = turn.resume({pending.interrupt_id: "yes"})
    assert done.status == "ok" and done.trace_id == paused.trace_id
    assert h.storage.list_reports("ceo") == []
    (end,) = events(h, "turn_end")
    assert end["trace_id"] == turn.trace_id


def test_failures_become_a_message_with_the_trace_id(h):
    result = Turn(h.session).start("revenue")  # nothing scripted: the model call fails

    assert result.status == "error" and result.outcome is None
    assert result.trace_id in result.message
    (end,) = events(h, "turn_end")
    assert end["status"] == "error" and end["error"] == "AssertionError"


def test_step_limit_has_its_own_message(h, monkeypatch):
    def too_long(*args, **kwargs):
        raise GraphRecursionError("limit")

    monkeypatch.setattr(h.session, "run_turn", too_long)
    result = Turn(h.session).start("everything about everything")

    assert result.status == "error" and result.error_kind == "step_limit"
    assert "too complex" in result.message
    assert events(h, "turn_end")[0]["error_kind"] == "step_limit"


def test_ctrl_c_cancels_and_leaves_a_usable_thread(h):
    def interrupted(messages):
        raise KeyboardInterrupt

    h.model.script.append(interrupted)
    result = Turn(h.session).start("revenue")

    assert result.status == "cancelled" and result.message == CANCELLED_MESSAGE
    assert events(h, "turn_end")[0]["status"] == "cancelled"
    assert h.turn("again", AIMessage("Hello.")).answer == "Hello."


class HostStopped(BaseException):
    """What a UI framework raises to stop a script (e.g. Streamlit's StopException)."""


def test_host_stop_cancels_the_turn_and_propagates(h):
    def stopped(messages):
        raise HostStopped

    h.model.script.append(stopped)
    with pytest.raises(HostStopped):
        Turn(h.session).start("revenue")

    assert events(h, "turn_end")[0]["status"] == "cancelled"
    assert h.turn("again", AIMessage("Hello.")).answer == "Hello."
