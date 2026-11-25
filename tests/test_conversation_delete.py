"""A user deletes one of their conversations; reports saved from it stay."""

import pytest
from langchain_core.messages import AIMessage

from harness import Harness
from retail_agent.data.storage import ConversationAccessError


def thread_exists(h: Harness, conversation_id: str) -> bool:
    config = {"configurable": {"thread_id": conversation_id}}
    return h.session.graph.checkpointer.get_tuple(config) is not None


def test_deleting_another_conversation_keeps_the_current_one(tmp_path):
    h = Harness(tmp_path, conversation_id="old")
    h.turn("first question", AIMessage("First answer."))
    report_id = h.storage.save_report("ceo", "old", "Q1 review", "# Q1").id
    h.session.switch_conversation("current")
    h.turn("second question", AIMessage("Second answer."))

    assert h.session.delete_conversation("old") is True
    assert h.storage.get_conversation("old") is None and not thread_exists(h, "old")
    assert h.session.conversation_id == "current" and thread_exists(h, "current")
    assert [c.id for c in h.session.conversations()] == ["current"]
    assert h.storage.get_report("ceo", report_id) is not None  # reports stay
    entry = h.storage.audit_entries("ceo")[0]
    assert entry.action == "conversation_deleted" and entry.target_ids == ("old",)
    assert "first question" not in str(entry.details)  # no content in the audit


def test_deleting_the_current_conversation_starts_a_new_one(tmp_path):
    h = Harness(tmp_path, conversation_id="current")
    h.turn("question", AIMessage("Answer."))
    assert h.session.delete_conversation("current") is True
    assert h.session.conversation_id != "current"
    assert h.session.state().get("messages") in (None, [])
    h.turn("new question", AIMessage("New answer."))  # the new conversation works
    assert [c.title for c in h.session.conversations()] == ["new question"]


def test_cannot_delete_someone_elses_or_a_missing_conversation(tmp_path):
    h = Harness(tmp_path)
    h.storage.record_turn("bobs", "bob", "bob's question")
    with pytest.raises(ConversationAccessError):
        h.session.delete_conversation("bobs")
    assert h.storage.get_conversation("bobs") is not None
    assert h.session.delete_conversation("missing") is False
