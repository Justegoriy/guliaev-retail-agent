"""Turn streaming (CLI stages), cancellation and conversation helpers of AgentSession."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness import Harness, sql_call
from retail_agent.graph.builder import STAGE_START, stage_after

SQL = "SELECT 1 AS month, 2 AS revenue FROM `bigquery-public-data.thelook_ecommerce.order_items`"


def test_stages_follow_the_graph(tmp_path):
    h = Harness(tmp_path)
    h.model.script.extend([sql_call(SQL), AIMessage("Revenue grew.")])
    stages: list[str] = []
    out = h.session.run_turn("Monthly revenue 2024", on_stage=stages.append)
    assert out.answer.startswith("Revenue grew.")
    assert stages[0] == STAGE_START
    assert "Querying BigQuery..." in stages
    assert stages.index("Querying BigQuery...") < stages.index("Writing answer...")


def test_stage_after_mapping():
    assert stage_after("input_guard", {}) == "Thinking..."
    assert stage_after("agent", {"messages": [sql_call(SQL)]}) == "Querying BigQuery..."
    assert stage_after("agent", {"messages": [AIMessage("done")]}) == "Checking the answer..."
    assert stage_after("tools", None) == "Writing answer..."
    assert stage_after("__interrupt__", ()) is None


def test_cancelled_turn_leaves_a_usable_thread(tmp_path):
    h = Harness(tmp_path)
    h.model.script.append(sql_call(SQL))

    def cancel(stage: str) -> None:
        if stage == "Querying BigQuery...":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        h.session.run_turn("Monthly revenue 2024", on_stage=cancel)
    h.session.cancel_turn()

    out = h.turn("Hi again", AIMessage("Hello."))
    assert out.answer.startswith("Hello.")
    messages = h.messages
    # Every tool call has its answer, otherwise the next model request is invalid.
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    for m in messages:
        if isinstance(m, AIMessage):
            assert {c["id"] for c in m.tool_calls} <= answered
    assert isinstance(messages[-2], HumanMessage)


def test_history_and_resume_helpers(tmp_path):
    h = Harness(tmp_path)
    h.turn("Monthly revenue 2024", AIMessage("Revenue grew."))
    h.turn("And in 2023?", AIMessage("It was lower."))
    [conv] = h.session.conversations()
    assert conv.title == "Monthly revenue 2024" and conv.turns == 2
    assert h.session.find_conversation(conv.id[:4]).id == conv.id
    assert h.session.find_conversation("zzz") is None
    pairs = h.session.recent_exchanges(3)
    assert pairs[-1][0] == "And in 2023?" and pairs[-1][1].startswith("It was lower.")
