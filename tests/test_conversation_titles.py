"""Conversations get a short summarized title once, from the post-turn extraction call."""

import sqlite3

import pytest
from langchain_core.messages import AIMessage

from harness import REVENUE_SQL, Harness, sql_call
from retail_agent.data.storage import Storage
from retail_agent.graph import memory
from retail_agent.graph.memory import ContextExtraction
from retail_agent.safety.input_guard import GuardVerdict


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def titled(prompt: str) -> ContextExtraction:
    title = "Monthly Revenue 2024" if "no title yet" in prompt else ""
    return ContextExtraction(question="Monthly revenue in 2024", conversation_title=title)


def conversation(h: Harness):
    return h.storage.get_conversation("conv1")


def test_first_turn_gets_a_summarized_title_once(h):
    h.aux.handlers[ContextExtraction] = titled
    h.turn(
        "hey can u show me how much money we made each month in 2024",
        sql_call(REVENUE_SQL),
        AIMessage("Revenue: $300."),
    )
    conv = conversation(h)
    assert conv.title == "Monthly Revenue 2024" and conv.title_generated
    assert "no title yet" in h.aux.prompts[ContextExtraction][0]

    h.turn("and only for Jeans?", sql_call(REVENUE_SQL, "c2"), AIMessage("Jeans: $100."))
    assert conversation(h).title == "Monthly Revenue 2024"
    assert "no title yet" not in h.aux.prompts[ContextExtraction][1]
    assert conversation(h).turns == 2


def test_turn_without_sql_is_titled_without_preference_signals(h):
    h.aux.handlers[GuardVerdict] = lambda p: GuardVerdict(category="data_question", reason="")
    h.aux.handlers[ContextExtraction] = lambda p: ContextExtraction(
        conversation_title="Available Data Overview"
    )
    h.turn("What data is available to me?", AIMessage("Four tables..."))
    assert conversation(h).title == "Available Data Overview"
    (prompt,) = h.aux.prompts[ContextExtraction]
    assert memory.NO_PREFERENCES in prompt
    assert h.session.state().get("analysis_context") is None


def test_greeting_is_not_a_title_and_the_next_turn_is(h):
    h.aux.handlers[GuardVerdict] = lambda p: GuardVerdict(category="greeting", reason="")
    h.turn("hi", AIMessage("Hello! Ask me about sales."))
    assert ContextExtraction not in h.aux.prompts
    assert conversation(h).title == "hi" and not conversation(h).title_generated

    h.aux.handlers[GuardVerdict] = lambda p: GuardVerdict(category="analysis", reason="")
    h.aux.handlers[ContextExtraction] = titled
    h.turn("monthly revenue for 2024", sql_call(REVENUE_SQL), AIMessage("Revenue: $300."))
    assert conversation(h).title == "Monthly Revenue 2024"


def test_failed_call_is_retried_unusable_title_keeps_the_message(h):
    def down(prompt):
        raise TimeoutError

    h.aux.handlers[ContextExtraction] = down
    h.turn("monthly revenue for 2024", sql_call(REVENUE_SQL), AIMessage("Revenue: $300."))
    assert not conversation(h).title_generated  # the call failed: try again next turn

    h.aux.handlers[ContextExtraction] = lambda p: ContextExtraction(conversation_title='"."')
    h.turn("and for 2023?", sql_call(REVENUE_SQL, "c2"), AIMessage("Revenue: $280."))
    conv = conversation(h)
    assert conv.title == "monthly revenue for 2024" and conv.title_generated

    h.turn("and for 2022?", sql_call(REVENUE_SQL, "c3"), AIMessage("Revenue: $250."))
    assert "no title yet" not in h.aux.prompts[ContextExtraction][-1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"Q1 2025 Monthly Revenue."', "Q1 2025 Monthly Revenue"),
        ("  Texas   vs California\nSpending ", "Texas vs California Spending"),
        ("", None),
        ("..", None),
        ("Orders of john.doe@example.com", None),
        (
            "A Very Long Title That Keeps Going Well Beyond Sixty Characters In Total Length",
            "A Very Long Title That Keeps Going Well Beyond Sixty",
        ),
    ],
)
def test_clean_title(raw, expected):
    assert memory.clean_title(raw) == expected


def test_title_is_set_only_on_the_owners_conversation(tmp_path):
    storage = Storage(tmp_path / "app.db")
    storage.record_turn("c1", "alice", "first question")
    storage.set_conversation_title("c1", "bob", "Hijacked")
    assert storage.get_conversation("c1").title == "first question"
    storage.set_conversation_title("c1", "alice", None)
    conv = storage.get_conversation("c1")
    assert conv.title == "first question" and conv.title_generated


def test_existing_database_gets_the_new_column(tmp_path):
    path = tmp_path / "app.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, owner_user_id TEXT NOT NULL, "
            "title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "turns INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute("INSERT INTO conversations VALUES ('c1', 'alice', 'old', 't', 't', 3)")
    storage = Storage(path)
    conv = storage.get_conversation("c1")
    assert conv.title == "old" and conv.turns == 3 and not conv.title_generated


def test_untitled_conversations_for_retitling(tmp_path):
    storage = Storage(tmp_path / "app.db")
    storage.record_turn("c1", "alice", "first")
    storage.record_turn("c2", "alice", "second")
    storage.record_turn("c3", "bob", "third")
    storage.set_conversation_title("c2", "alice", "Titled")
    assert {c.id for c in storage.untitled_conversations()} == {"c1", "c3"}
    assert [c.id for c in storage.untitled_conversations("alice")] == ["c1"]
