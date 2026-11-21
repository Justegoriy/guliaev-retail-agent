import pytest

from retail_agent.data.storage import ConversationAccessError, Storage


@pytest.fixture
def storage(tmp_path):
    db = Storage(tmp_path / "app.db")
    yield db
    db.close()


def save(storage: Storage, owner="alice", conversation="c1", title="Q1 review", **kwargs):
    defaults = dict(content="# Summary\nRevenue grew 5% in Jeans.", entities=["Jeans"])
    return storage.save_report(owner, conversation, title, **{**defaults, **kwargs})


def test_save_and_get_round_trip(storage):
    report = save(storage, source_queries=["SELECT 1"])
    loaded = storage.get_report("alice", report.id)
    assert loaded == report
    assert loaded.entities == ("Jeans",) and loaded.source_queries == ("SELECT 1",)
    assert report.snippet() == "Summary Revenue grew 5% in Jeans."


def test_other_users_reports_are_invisible(storage):
    mine = save(storage, owner="alice")
    theirs = save(storage, owner="bob", title="Bob's Jeans report")
    assert storage.get_report("alice", theirs.id) is None
    assert storage.get_reports("alice", [mine.id, theirs.id]) == [mine]
    assert [r.id for r in storage.list_reports("alice")] == [mine.id]
    assert [r.id for r in storage.search_reports("alice", "jeans")] == [mine.id]
    assert storage.soft_delete_reports("alice", [theirs.id]) == []
    assert storage.get_report("bob", theirs.id) is not None


def test_soft_delete_hides_the_report(storage):
    report = save(storage)
    other = save(storage, title="Other")
    assert storage.soft_delete_reports("alice", [report.id, "missing"]) == [report.id]
    assert storage.get_report("alice", report.id) is None
    assert [r.id for r in storage.list_reports("alice")] == [other.id]
    assert storage.search_reports("alice", "Q1") == []
    # Deleting again is a no-op; the row is still in the table (soft delete).
    assert storage.soft_delete_reports("alice", [report.id]) == []
    rows = storage._query("SELECT deleted_at FROM reports WHERE id = ?", (report.id,))
    assert rows[0]["deleted_at"]


def test_search_matches_title_content_and_entities_case_insensitively(storage):
    by_title = save(storage, title="Levi's deep dive", content="text", entities=[])
    by_content = save(storage, title="A", content="Customer 4711 churned", entities=[])
    by_entity = save(storage, title="B", content="text", entities=["California"])
    assert [r.id for r in storage.search_reports("alice", "LEVI'S")] == [by_title.id]
    assert [r.id for r in storage.search_reports("alice", "4711")] == [by_content.id]
    assert [r.id for r in storage.search_reports("alice", "california")] == [by_entity.id]
    assert storage.search_reports("alice", "   ") == []


def test_search_treats_like_wildcards_literally(storage):
    save(storage, title="Growth 5%", content="x", entities=[])
    save(storage, title="Growth 50", content="x", entities=[])
    assert [r.title for r in storage.search_reports("alice", "5%")] == ["Growth 5%"]
    assert storage.search_reports("alice", "_") == []


def test_list_by_conversation_newest_first(storage):
    first = save(storage, conversation="c1", title="first")
    save(storage, conversation="c2", title="elsewhere")
    second = save(storage, conversation="c1", title="second")
    reports = storage.list_reports("alice", conversation_id="c1")
    assert [r.id for r in reports] == [second.id, first.id]


def test_conversation_turns_and_owner_check(storage):
    storage.record_turn("conv", "alice", "  monthly   revenue for 2024 ")
    storage.record_turn("conv", "alice", "and for Jeans?")
    conversation = storage.get_conversation("conv")
    assert conversation.title == "monthly revenue for 2024" and conversation.turns == 2
    with pytest.raises(ConversationAccessError):
        storage.record_turn("conv", "bob", "hi")
    with pytest.raises(ConversationAccessError):
        storage.check_conversation_owner("conv", "bob")
    storage.check_conversation_owner("new-conversation", "bob")
    assert [c.id for c in storage.list_conversations("alice")] == ["conv"]
    assert storage.list_conversations("bob") == []


def test_audit_log(storage):
    storage.audit("alice", "reports_deleted", ["r1", "r2"], {"soft": True}, trace_id="t1")
    storage.audit("bob", "report_exported", ["r3"])
    entries = storage.audit_entries("alice")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.action == "reports_deleted" and entry.target_ids == ("r1", "r2")
    assert entry.details == {"soft": True} and entry.trace_id == "t1"
    assert [e.action for e in storage.audit_entries()] == ["report_exported", "reports_deleted"]


def test_schema_creation_is_idempotent(tmp_path):
    first = Storage(tmp_path / "app.db")
    report = save(first)
    first.close()
    second = Storage(tmp_path / "app.db")
    assert second.get_report("alice", report.id) == report
    second.close()
