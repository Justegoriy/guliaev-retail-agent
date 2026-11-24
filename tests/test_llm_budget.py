"""At most MAX_LLM_CALLS_PER_TURN model calls per question (input guard, agent, post-turn)."""

from langchain_core.messages import AIMessage

from harness import REVENUE_SQL, Harness, ScriptedChat, sql_call
from retail_agent.graph import nodes
from retail_agent.graph.memory import ContextExtraction


def calls_of_turn(h: Harness) -> int:
    return h.session.deps.tool_ctx.turn.llm_calls


def test_typical_question_uses_three_calls(tmp_path):
    h = Harness(tmp_path, max_agent_steps=10)
    h.turn("monthly revenue 2024", sql_call(REVENUE_SQL), AIMessage("Revenue: $300."))
    # guard + two agent calls (query, answer) + post-turn extraction
    assert calls_of_turn(h) == 4
    assert nodes.FINAL_CALL_NOTE not in h.model.system_prompt()


def test_last_allowed_agent_call_must_answer(tmp_path):
    h = Harness(tmp_path, max_agent_steps=10)
    out = h.turn(
        "why did revenue drop?",
        sql_call(REVENUE_SQL, "c1"),
        sql_call(REVENUE_SQL, "c2"),
        AIMessage("Revenue fell because of fewer orders."),
    )
    # guard (1) + agent (2, 3, 4) + extraction (5): the third agent call is the last one.
    assert len(h.model.requests) == 3
    assert nodes.FINAL_CALL_NOTE not in h.model.system_prompt(1)
    assert nodes.FINAL_CALL_NOTE in h.model.system_prompt(2)
    assert out.answer == "Revenue fell because of fewer orders."
    assert calls_of_turn(h) == 5
    assert ContextExtraction in h.aux.prompts  # the reserved post-turn call still ran


def test_tool_calls_on_the_last_call_are_dropped(tmp_path):
    h = Harness(tmp_path, max_agent_steps=10)
    out = h.turn(
        "deep dive",
        sql_call(REVENUE_SQL, "c1"),
        sql_call(REVENUE_SQL, "c2"),
        sql_call(REVENUE_SQL, "c3"),  # ignored: no budget left for another step
    )
    assert out.answer == nodes.BUDGET_ANSWER
    assert h.runner.execute.call_count == 2
    assert calls_of_turn(h) <= h.settings.max_llm_calls_per_turn


def test_rules_only_guard_leaves_one_more_agent_call(tmp_path):
    h = Harness(tmp_path, max_agent_steps=10, input_guard_llm=False)
    h.turn(
        "deep dive",
        sql_call(REVENUE_SQL, "c1"),
        sql_call(REVENUE_SQL, "c2"),
        sql_call(REVENUE_SQL, "c3"),
        AIMessage("Done."),
    )
    assert len(h.model.requests) == 4
    assert nodes.FINAL_CALL_NOTE in h.model.system_prompt(3)


def test_budget_is_per_turn(tmp_path):
    h = Harness(tmp_path, max_agent_steps=10)
    h.turn("q1", sql_call(REVENUE_SQL, "c1"), sql_call(REVENUE_SQL, "c2"), AIMessage("A1."))
    h.turn("q2", sql_call(REVENUE_SQL, "c3"), AIMessage("A2."))
    assert calls_of_turn(h) == 4
    assert nodes.FINAL_CALL_NOTE not in h.model.system_prompt()


def test_summary_waits_when_the_turn_used_its_budget(tmp_path):
    h = Harness(tmp_path, max_agent_steps=10, history_summarize_after_turns=1)
    h.turn("q1", sql_call(REVENUE_SQL, "c1"), sql_call(REVENUE_SQL, "c2"), AIMessage("A1."))
    h.turn("q2", sql_call(REVENUE_SQL, "c3"), sql_call(REVENUE_SQL, "c4"), AIMessage("A2."))
    assert calls_of_turn(h) == 5
    assert "history_summary" not in h.session.state() or not h.session.state()["history_summary"]
    h.turn("q3", AIMessage("A3."))  # a cheap turn folds the history
    assert h.session.state()["history_summary"]


def test_last_call_gets_the_history_without_tool_parts(tmp_path):
    from langchain_core.messages import HumanMessage, ToolMessage

    h = Harness(tmp_path, max_agent_steps=10)
    answer = ScriptedChat()
    answer.script.append(AIMessage("Texas spends less."))
    h.session.deps.answer_model = answer
    h.turn("why?", sql_call(REVENUE_SQL, "c1"), sql_call(REVENUE_SQL, "c2"))
    (request,) = answer.requests
    assert nodes.FINAL_CALL_NOTE in request[0].text
    assert not any(isinstance(m, ToolMessage) for m in request)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in request)
    data = [m.text for m in request if isinstance(m, HumanMessage)]
    assert any("Data from your queries (run_sql)" in t and "OK" in t for t in data)
    assert len(h.model.requests) == 2  # the tool-bound model was not used for the last step


def test_without_tool_parts_merges_neighbours():
    from langchain_core.messages import HumanMessage, ToolMessage

    from retail_agent.graph import memory

    call = {"name": "run_sql", "args": {}, "id": "c1"}
    out = memory.without_tool_parts(
        [
            HumanMessage("q"),
            AIMessage("", tool_calls=[call]),
            ToolMessage("OK: rows=1", tool_call_id="c1", name="run_sql"),
            AIMessage("Checking one more thing.", tool_calls=[{**call, "id": "c2"}]),
            ToolMessage("OK: rows=2", tool_call_id="c2", name="run_sql"),
        ]
    )
    assert [type(m).__name__ for m in out] == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert out[0].text == "q\n\nData from your queries (run_sql):\nOK: rows=1"
    assert out[1].text == "Checking one more thing."
    assert out[2].text == "Data from your queries (run_sql):\nOK: rows=2"
    assert not any("called" in m.text.lower() for m in out)  # nothing to imitate
