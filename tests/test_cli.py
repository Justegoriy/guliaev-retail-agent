from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from harness import Harness
from retail_agent import cli
from retail_agent.auth import UserContext
from retail_agent.errors import ServiceUnavailable
from retail_agent.graph.builder import PendingConfirmation, TurnOutcome
from retail_agent.observability import callbacks, inspect, logger
from retail_agent.observability.logger import log_event
from retail_agent.tools import TurnState

PENDING = PendingConfirmation(
    interrupt_id="i1",
    action="delete_reports",
    question="Delete 1 report?",
    summary="1 report will be deleted.",
    items=[{"id": "r1", "title": "Q1", "created_at": "2026-11-21T10:00:00", "snippet": "..."}],
    digest="d",
)


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.deps = SimpleNamespace(tool_ctx=SimpleNamespace(turn=TurnState()))

    def run_turn(self, text, trace_id, on_stage=None):
        self.calls.append(("turn", text))
        if text.startswith("interrupt"):
            raise KeyboardInterrupt
        if text.startswith("fail"):
            raise ServiceUnavailable("bigquery", "network down")
        callbacks._add_usage(trace_id, 100, 20)
        if text.startswith("delete"):
            return TurnOutcome("", False, trace_id, pending=(PENDING,))
        return TurnOutcome(f"answer to {text}", False, trace_id)

    def resume(self, answers, trace_id, on_stage=None):
        self.calls.append(("resume", answers))
        return TurnOutcome("resumed", False, trace_id)

    def cancel_turn(self):
        self.calls.append(("cancel",))


@pytest.fixture
def log_path(tmp_path):
    path = tmp_path / "agent.jsonl"
    logger.configure(path)
    yield path
    logger._log_path = None


def turn_ends(path):
    return [e for e in inspect.read_jsonl(path) if e["event"] == "turn_end"]


def reader(*answers):
    queue = list(answers)

    def read(prompt):
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return read


@pytest.mark.parametrize(
    ("answer", "new"),
    [
        ("y", False),
        ("no", False),
        ("yes do it", False),
        ("what about Swim?", True),
        ("show revenue by month instead", True),
    ],
)
def test_looks_like_new_message(answer, new):
    assert cli.looks_like_new_message(answer) is new


def test_answer_is_passed_to_resume():
    session = FakeSession()
    cli._run_turn(session, "delete report r1", reader("y"))
    assert session.calls == [("turn", "delete report r1"), ("resume", {"i1": "y"})]


def test_ctrl_c_means_no():
    session = FakeSession()
    cli._run_turn(session, "delete report r1", reader(KeyboardInterrupt()))
    assert session.calls[-1] == ("resume", {"i1": "no"})


def test_new_question_cancels_and_runs_as_next_turn():
    session = FakeSession()
    cli._run_turn(session, "delete report r1", reader("and what was revenue in 2024?"))
    assert session.calls == [
        ("turn", "delete report r1"),
        ("resume", {}),  # unanswered = no
        ("turn", "and what was revenue in 2024?"),
    ]


def test_turn_end_has_usage_and_trace(log_path, capsys):
    session = FakeSession()
    cli._run_turn(session, "revenue?", reader())
    [end] = turn_ends(log_path)
    assert end["status"] == "ok"
    assert (end["llm_calls"], end["input_tokens"], end["output_tokens"]) == (1, 100, 20)
    assert f"trace: {end['trace_id']}" in capsys.readouterr().out


def test_ctrl_c_during_turn_cancels_it(log_path, capsys):
    session = FakeSession()
    cli._run_turn(session, "interrupt me", reader())
    assert session.calls == [("turn", "interrupt me"), ("cancel",)]
    [end] = turn_ends(log_path)
    assert end["status"] == "cancelled"
    assert "Cancelled" in capsys.readouterr().out


def test_error_prints_user_message_not_traceback(log_path, capsys):
    session = FakeSession()
    cli._run_turn(session, "fail please", reader())
    [end] = turn_ends(log_path)
    assert end["status"] == "error" and end["error_kind"] == "bq_unavailable"
    out = capsys.readouterr().out
    assert "data warehouse is temporarily unavailable" in out
    assert "Traceback" not in out and end["trace_id"] in out


# -- commands on a real session -------------------------------------------------------


def make_harness(tmp_path, user=None):
    harness = Harness(tmp_path, **({"user": user} if user else {}))
    logger.configure(harness.settings.log_path)
    logger.user_id_var.set(harness.user.user_id)
    return harness


@pytest.fixture
def h(tmp_path):
    yield make_harness(tmp_path)
    logger._log_path = None
    logger.user_id_var.set(None)


@pytest.fixture
def run(monkeypatch):
    def run(harness, *lines):
        """Feed REPL lines; model answers come from harness.model.script."""
        monkeypatch.setattr(cli, "_make_reader", lambda: reader(*lines, EOFError()))
        cli._repl(harness.session, harness.settings)

    return run


def test_history_resume_and_new(h, run, capsys):
    h.model.script.extend([AIMessage("Revenue grew."), AIMessage("It was lower.")])
    run(h, "Monthly revenue 2024", "And 2023?", "/new", "/history")
    out = capsys.readouterr().out
    assert "Monthly revenue 2024" in out and "New conversation" in out
    first = h.session.conversations()[-1].id
    run(h, f"/resume {first[:6]}")
    out = capsys.readouterr().out
    assert "Resumed" in out and "It was lower." in out
    assert h.session.conversation_id == first


def test_resume_of_another_users_conversation_is_refused(h, run, capsys):
    h.storage.record_turn("someone-elses", "denim_manager", "secret")
    run(h, "/resume someone-elses")
    assert "belongs to another user" in capsys.readouterr().out
    assert h.session.conversation_id != "someone-elses"


def test_trace_and_stats_commands(h, run, capsys):
    h.model.script.append(AIMessage("Hello."))
    run(h, "hi", "/trace", "/trace --full", "/stats", "/stats all", "/nonsense", "/help")
    out = capsys.readouterr().out
    assert "turn_start" in out and "turn_end" in out
    assert "No model payloads" in out
    assert "Success rate" in out and "100%" in out
    assert "Unknown command /nonsense" in out
    assert "/resume <id>" in out


def test_stats_all_needs_full_access(tmp_path, run, capsys):
    manager = UserContext("denim_manager", "Bob", "manager", brands=("Diesel",))
    try:
        run(make_harness(tmp_path, manager), "/stats all")
    finally:
        logger._log_path = None
        logger.user_id_var.set(None)
    assert "Only full-access roles" in capsys.readouterr().out


def test_trace_of_another_user_is_not_found(h, run, capsys):
    logger.user_id_var.set("someone")
    other = logger.new_trace_id()
    log_event("turn_start", length=1)
    logger.user_id_var.set(h.user.user_id)
    run(h, f"/trace {other}")
    assert "No turn of yours" in capsys.readouterr().out
