from typing import TypedDict

import pytest
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from retail_agent.auth import UserContext
from retail_agent.safety import confirmation
from retail_agent.safety.confirmation import ConfirmationError, Decision
from retail_agent.tools import REGISTRY, ToolContext, ToolRegistry, ToolSpec, build_tools

CEO = UserContext("ceo", "Alice", "ceo")


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES ", "д", "да", True])
def test_yes_answers(answer):
    assert confirmation.is_yes(answer)


@pytest.mark.parametrize(
    "answer", ["", "n", "no", "yes please", "yep", "ok", "sure", "y?", "1", None, False]
)
def test_everything_else_is_no(answer):
    assert not confirmation.is_yes(answer)


# -- require() inside a graph -------------------------------------------------------------


class State(TypedDict, total=False):
    decision: str


def run_require(answer, *, shown_digest="d1", current_digest="d1"):
    """Pause in a node that asks for confirmation, then resume with `answer`."""
    audit: list[tuple[str, list[str]]] = []
    performed: list[bool] = []

    def node(state: State) -> State:
        with confirmation.scope("test_tool"):
            decision = confirmation.require(
                "delete_reports",
                question="Delete 1 report?",
                summary="1 report",
                items=[{"id": "r1", "title": "t"}],
                digest=current_digest,
                audit=lambda action, ids, details: audit.append((action, ids)),
            )
            if decision is Decision.APPROVED:
                confirmation.ensure_approved("delete_reports")
                performed.append(True)
        return {"decision": str(decision)}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({}, config)
    snapshot = graph.get_state(config)
    (pending,) = snapshot.interrupts
    assert pending.value["type"] == "confirm" and pending.value["question"] == "Delete 1 report?"
    assert audit == [("confirmation_requested", ["r1"])]
    reply = {"answer": answer, "digest": shown_digest} if shown_digest else answer
    result = graph.invoke(Command(resume=reply), config)
    return result["decision"], performed, audit


def test_yes_approves_and_allows_the_side_effect():
    decision, performed, audit = run_require("yes")
    assert decision == "approved" and performed == [True]
    assert audit[-1] == ("confirmation_approved", ["r1"])


def test_anything_else_rejects():
    decision, performed, audit = run_require("delete them all, I said")
    assert decision == "rejected" and performed == []
    assert audit[-1] == ("confirmation_rejected", ["r1"])


def test_changed_targets_make_the_answer_stale_even_if_yes():
    decision, performed, _ = run_require("yes", shown_digest="old", current_digest="new")
    assert decision == "stale" and performed == []


def test_plain_string_reply_is_accepted():
    decision, performed, _ = run_require("y", shown_digest=None)
    assert decision == "approved" and performed == [True]


# -- enforcement --------------------------------------------------------------------------------


def make_ctx() -> ToolContext:
    return ToolContext(user=CEO, conversation_id="c", runner=None, schema=None)  # type: ignore[arg-type]


def test_side_effect_without_approval_is_refused():
    with confirmation.scope("t"), pytest.raises(ConfirmationError, match="not confirmed"):
        confirmation.ensure_approved("delete_reports")
    with pytest.raises(ConfirmationError):
        confirmation.ensure_approved("delete_reports")


def test_approval_does_not_leak_out_of_its_scope():
    with confirmation.scope("t"):
        confirmation._scope.get().approved.add("export_report")
        confirmation.ensure_approved("export_report")
    with confirmation.scope("t"), pytest.raises(ConfirmationError):
        confirmation.ensure_approved("export_report")


def test_tool_declared_with_confirmation_that_skips_it_is_caught():
    """A requires_confirmation tool that performs its effect without require() fails."""

    def sloppy_delete() -> str:
        confirmation.ensure_approved("delete_reports")
        return "DELETED"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "sloppy_delete",
            lambda ctx: StructuredTool.from_function(
                sloppy_delete, name="sloppy_delete", description="test"
            ),
            side_effects=True,
            requires_confirmation=True,
        )
    )
    (tool,) = build_tools(make_ctx(), registry)
    with pytest.raises(ConfirmationError, match="not confirmed"):
        tool.invoke({})


def test_tool_not_declared_cannot_ask_for_confirmation():
    def sneaky() -> str:
        confirmation.require("delete_reports", question="?", summary="", items=[], digest="")
        return "never"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "sneaky",
            lambda ctx: StructuredTool.from_function(sneaky, name="sneaky", description="test"),
        )
    )
    (tool,) = build_tools(make_ctx(), registry)
    with pytest.raises(ConfirmationError, match="requires_confirmation=True"):
        tool.invoke({})


def test_requires_confirmation_tool_must_be_sync_structured_tool():
    async def run() -> str:
        return "x"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "async_tool",
            lambda ctx: StructuredTool.from_function(
                coroutine=run, name="async_tool", description="test"
            ),
            requires_confirmation=True,
        )
    )
    with pytest.raises(ValueError, match="sync tool"):
        build_tools(make_ctx(), registry)


def test_registered_side_effect_tools_are_declared():
    specs = {s.name: s for s in REGISTRY.specs()}
    for name in ("request_report_deletion", "export_report"):
        assert specs[name].side_effects and specs[name].requires_confirmation
    assert specs["save_report"].side_effects and not specs["save_report"].requires_confirmation
    for name in ("run_sql", "list_reports", "search_reports", "get_report"):
        assert not specs[name].side_effects
