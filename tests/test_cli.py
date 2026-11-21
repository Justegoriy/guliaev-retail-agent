from types import SimpleNamespace

import pytest

from retail_agent import cli
from retail_agent.graph.builder import PendingConfirmation, TurnOutcome
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

    def run_turn(self, text, trace_id):
        self.calls.append(("turn", text))
        if text.startswith("delete"):
            return TurnOutcome("", False, trace_id, pending=(PENDING,))
        return TurnOutcome(f"answer to {text}", False, trace_id)

    def resume(self, answers, trace_id):
        self.calls.append(("resume", answers))
        return TurnOutcome("resumed", False, trace_id)


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
