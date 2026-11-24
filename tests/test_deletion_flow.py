import pytest
from langchain_core.messages import AIMessage, ToolMessage

from harness import CEO, REVENUE_SQL, Harness, calls, tool_call
from retail_agent.auth import UserContext
from retail_agent.data.storage import ConversationAccessError
from retail_agent.tools.report_tools import report_markdown

BOB = UserContext("bob", "Bob", "ceo")
EMAIL = "john.doe@example.com"


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def save_call(title: str, call_id: str, **extra) -> dict:
    args = {
        "title": title,
        "question": f"{title} with action items",
        "content_markdown": f"# {title}\nJeans revenue grew 5%.",
        **extra,
    }
    return tool_call("save_report", args, call_id)


def delete_call(call_id: str = "d1", **args) -> AIMessage:
    return calls(tool_call("request_report_deletion", args, call_id))


def tool_results(h: Harness, name: str) -> list[str]:
    return [m.text for m in h.messages if isinstance(m, ToolMessage) and m.name == name]


def live_ids(h: Harness, user=CEO) -> set[str]:
    return {r.id for r in h.storage.list_reports(user.user_id)}


def save_two(h: Harness) -> list[str]:
    h.turn(
        "save both reports",
        calls(save_call("Q1 review", "s1"), save_call("Q2 review", "s2")),
        AIMessage("Saved two reports."),
    )
    return h.session.state()["session_report_ids"]


# -- saving ------------------------------------------------------------------------------------


def test_parallel_saves_both_land_in_session_reports(h):
    ids = save_two(h)
    assert len(ids) == 2 and set(ids) == live_ids(h)
    assert all(r.startswith("SAVED: report_id=") for r in tool_results(h, "save_report"))
    h.turn("thanks", AIMessage("You're welcome."))
    prompt = h.model.system_prompt()
    assert "# Reports saved in this conversation" in prompt
    assert f"- {ids[0]}: Q1 review" in prompt and f"- {ids[1]}: Q2 review" in prompt


def test_saved_report_is_masked_and_keeps_its_sql(h):
    h.turn(
        "monthly revenue for 2024",
        calls(tool_call("run_sql", {"sql": REVENUE_SQL, "purpose": "revenue"}, "q1")),
        AIMessage("Revenue: $300."),
    )
    h.turn(
        "save it as a report",
        calls(
            save_call(
                "Revenue 2024",
                "s1",
                question=f"Monthly revenue in 2024 and the top buyer {EMAIL}",
                content_markdown=f"Top buyer {EMAIL}",
                entities=["revenue", EMAIL],
            )
        ),
        AIMessage("Saved."),
    )
    (report,) = h.storage.list_reports("ceo")
    assert EMAIL not in report.content and "[REDACTED]" in report.content
    assert report.question == "Monthly revenue in 2024 and the top buyer [REDACTED]"
    assert report.entities == ("revenue", "[REDACTED]")
    # Built from the previous turn's analysis: its SQL is recorded as the source.
    assert report.source_queries == (REVENUE_SQL,)
    assert report.conversation_id == "conv1"


# -- deletion ------------------------------------------------------------------------------------


def test_delete_the_report_just_made_asks_about_exactly_it(h):
    first, last = save_two(h)
    out = h.turn(
        "delete the report you just made",
        lambda msgs: delete_call(report_ids=[h.session.state()["session_report_ids"][-1]]),
    )
    assert out.answer == ""
    (pending,) = out.pending
    assert pending.action == "delete_reports" and pending.question == "Delete 1 report?"
    assert [i["id"] for i in pending.items] == [last]
    assert pending.items[0]["title"] == "Q2 review"
    assert live_ids(h) == {first, last}  # nothing happens before the answer

    done = h.resume("y", AIMessage("Deleted report Q2 review."))
    assert done.answer == "Deleted report Q2 review." and not done.pending
    assert live_ids(h) == {first}
    assert tool_results(h, "request_report_deletion") == [f"DELETED: 1 reports [{last}]."]
    actions = [e.action for e in reversed(h.storage.audit_entries("ceo"))]
    assert actions == ["confirmation_requested", "confirmation_approved", "reports_deleted"]
    assert h.storage.audit_entries("ceo")[0].target_ids == (last,)


def test_delete_this_conversation_no_then_yes(h):
    ids = save_two(h)
    other = h.storage.save_report("ceo", "another-conversation", "Old report", "text")

    out = h.turn("delete all reports from this conversation", delete_call(this_conversation=True))
    (pending,) = out.pending
    assert {i["id"] for i in pending.items} == set(ids)
    assert pending.question == "Delete 2 reports?"

    h.resume("no", AIMessage("Deletion cancelled."))
    assert live_ids(h) == {*ids, other.id}
    assert tool_results(h, "request_report_deletion") == ["CANCELLED by user: nothing was deleted."]

    out = h.turn(
        "delete all reports from this conversation", delete_call("d2", this_conversation=True)
    )
    assert {i["id"] for i in out.pending[0].items} == set(ids)
    h.resume("yes", AIMessage("Deleted 2 reports."))
    assert live_ids(h) == {other.id}
    assert set(h.storage.audit_entries("ceo")[0].target_ids) == set(ids)


def test_yes_in_the_chat_is_not_a_confirmation(h):
    ids = save_two(h)
    h.turn("delete my reports", AIMessage("Are you sure you want to delete them? Say yes."))
    out = h.turn("yes", delete_call(this_conversation=True))
    assert out.pending  # the tool still asks through the interrupt
    assert live_ids(h) == set(ids)


def test_no_match_returns_without_asking(h):
    save_two(h)
    out = h.turn(
        "delete reports mentioning Swim",
        delete_call(mention="Swim"),
        AIMessage("No reports mention Swim."),
    )
    assert not out.pending and out.answer == "No reports mention Swim."
    assert tool_results(h, "request_report_deletion")[0].startswith("NO_MATCH")
    assert h.storage.audit_entries("ceo") == []


def test_no_match_is_not_followed_by_broader_criteria_in_the_same_turn(h):
    """Seen live: "Q1 2025" -> "2025" -> "report" until unrelated reports were offered."""
    ids = save_two(h)
    out = h.turn(
        "delete the report you just made",
        delete_call("d1", mention="Q3 forecast"),
        delete_call("d2", mention="review"),
        AIMessage("Nothing matched; which report do you mean?"),
    )
    assert not out.pending and live_ids(h) == set(ids)
    first, second = tool_results(h, "request_report_deletion")
    assert first.startswith("NO_MATCH: no reports")
    assert second.startswith("NO_MATCH: an earlier deletion request in this turn")
    # The next turn starts fresh: the user names the reports and is asked as usual.
    out = h.turn("delete the reviews", delete_call("d3", mention="review"))
    assert {i["id"] for i in out.pending[0].items} == set(ids)


def test_mention_finds_reports_by_entity(h):
    h.turn(
        "save",
        calls(save_call("Brand report", "s1", entities=["Levi's"]), save_call("Other", "s2")),
        AIMessage("Saved."),
    )
    out = h.turn("delete reports about Levi's", delete_call(mention="levi's"))
    assert [i["title"] for i in out.pending[0].items] == ["Brand report"]


def test_other_users_report_is_never_offered_for_deletion(h):
    bobs = h.storage.save_report("bob", "bob-conversation", "Bob's report", "Jeans")
    out = h.turn(
        f"delete report {bobs.id}",
        delete_call(report_ids=[bobs.id]),
        AIMessage("That report is not among yours."),
    )
    assert not out.pending
    (result,) = tool_results(h, "request_report_deletion")
    assert result.startswith("NO_MATCH") and bobs.id in result and "someone else's" in result
    assert h.storage.get_report("bob", bobs.id) is not None


def test_changed_set_between_preview_and_answer_cancels(h):
    ids = save_two(h)
    h.turn("delete this conversation's reports", delete_call(this_conversation=True))
    h.storage.save_report("ceo", "conv1", "Saved meanwhile", "text")  # e.g. from another window
    h.resume("yes", AIMessage("The set changed, please confirm again."))
    assert tool_results(h, "request_report_deletion")[0].startswith("CANCELLED: the set")
    assert set(ids) <= live_ids(h)
    assert h.storage.audit_entries("ceo")[0].action == "confirmation_stale"


def test_new_message_cancels_the_pending_confirmation(h):
    ids = save_two(h)
    h.turn("delete this conversation's reports", delete_call(this_conversation=True))
    out = h.turn(
        "actually, what was revenue in 2024?",
        AIMessage("Deletion cancelled."),  # the paused turn finishes first
        AIMessage("Let me check."),
    )
    assert out.answer == "Let me check." and not out.pending
    assert live_ids(h) == set(ids)
    assert tool_results(h, "request_report_deletion") == ["CANCELLED by user: nothing was deleted."]


def test_pending_confirmation_survives_restart(h):
    first, last = save_two(h)
    h.turn("delete the last report", delete_call(report_ids=[last]))
    h.session = h.new_session()
    (pending,) = h.session.pending()
    assert [i["id"] for i in pending.items] == [last]
    h.resume("y", AIMessage("Deleted."))
    assert live_ids(h) == {first}


def test_finished_parallel_calls_are_not_rerun_on_resume(h):
    _, last = save_two(h)
    h.turn(
        "revenue 2024, and delete the last report",
        calls(
            tool_call("run_sql", {"sql": REVENUE_SQL, "purpose": "revenue"}, "q1"),
            tool_call("request_report_deletion", {"report_ids": [last]}, "d1"),
        ),
    )
    assert h.runner.execute.call_count == 1
    h.resume("y", AIMessage("Done."))
    assert h.runner.execute.call_count == 1
    assert len(tool_results(h, "run_sql")) == 1
    assert last not in live_ids(h)


def test_resume_without_pending_is_an_error(h):
    h.turn("hello", AIMessage("Hi!"))
    with pytest.raises(RuntimeError, match="no pending confirmation"):
        h.session.resume({})


# -- export ---------------------------------------------------------------------------------------


def export_call(report_id: str, call_id: str = "e1") -> AIMessage:
    return calls(tool_call("export_report", {"report_id": report_id}, call_id))


def test_export_needs_confirmation(h):
    _, last = save_two(h)
    target = h.settings.exports_dir / f"{last}.md"

    out = h.turn("export this report", export_call(last))
    (pending,) = out.pending
    assert pending.action == "export_report" and last in pending.question
    assert not target.exists()

    h.resume("n", AIMessage("Not exported."))
    assert not target.exists()

    h.turn("export this report", export_call(last, "e2"))
    h.resume("yes", AIMessage("Exported."))
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "# Q2 review"
    assert "**Question:** Q2 review with action items" in lines
    entry = h.storage.audit_entries("ceo")[0]
    assert entry.action == "report_exported" and entry.target_ids == (last,)


def test_other_users_report_is_not_exported(h):
    bobs = h.storage.save_report("bob", "b", "Bob's report", "text")
    out = h.turn("export it", export_call(bobs.id), AIMessage("Not found."))
    assert not out.pending
    assert tool_results(h, "export_report")[0].startswith("NOT_FOUND")
    assert not h.settings.exports_dir.exists()


# -- library tools and conversations ----------------------------------------------------------


def test_list_search_get_only_own_reports(h):
    ids = save_two(h)
    bobs = h.storage.save_report("bob", "b", "Q3 review", "Jeans")
    h.turn(
        "list, search and open",
        calls(
            tool_call("list_reports", {}, "l1"),
            tool_call("search_reports", {"query": "review"}, "f1"),
            tool_call("get_report", {"report_id": ids[0]}, "g1"),
            tool_call("get_report", {"report_id": bobs.id}, "g2"),
        ),
        AIMessage("Here they are."),
    )
    (listed,) = tool_results(h, "list_reports")
    assert listed.startswith("OK: 2 reports") and "(this conversation)" in listed
    (found,) = tool_results(h, "search_reports")
    assert ids[0] in found and ids[1] in found and bobs.id not in found
    own, foreign = tool_results(h, "get_report")
    assert own.startswith(f"OK: report {ids[0]}") and "Jeans revenue grew 5%." in own
    assert foreign == f"NOT_FOUND: there is no report {bobs.id!r} among the user's reports."


def test_conversation_is_recorded_and_protected(h):
    h.turn("monthly revenue", AIMessage("ok"))
    h.turn("and for Jeans?", AIMessage("ok"))
    conversation = h.storage.get_conversation("conv1")
    assert conversation.owner_user_id == "ceo" and conversation.turns == 2
    assert conversation.title == "monthly revenue"
    with pytest.raises(ConversationAccessError):
        h.new_session(user=BOB)


def test_report_is_question_answer_and_sql(h):
    h.turn(
        "monthly revenue for 2024",
        calls(tool_call("run_sql", {"sql": REVENUE_SQL, "purpose": "revenue"}, "q1")),
        AIMessage("Revenue: $300."),
    )
    h.turn("save it as a report", calls(save_call("Revenue 2024", "s1")), AIMessage("Saved."))
    (report,) = h.storage.list_reports("ceo")
    h.turn(
        "show it",
        calls(tool_call("get_report", {"report_id": report.id}, "g1")),
        AIMessage("Here it is."),
    )
    (shown,) = tool_results(h, "get_report")
    assert "Question: Revenue 2024 with action items" in shown
    assert "SQL queries: 1" in shown and "Jeans revenue grew 5%." in shown

    markdown = report_markdown(report)
    assert (
        markdown.index("**Question:**") < markdown.index("Jeans revenue") < markdown.index("## SQL")
    )
    assert f"```sql\n{REVENUE_SQL}\n```" in markdown


def test_save_without_a_question_is_an_argument_error(h):
    args = {"title": "Q1", "content_markdown": "text"}
    h.turn("save it", calls(tool_call("save_report", args, "s1")), AIMessage("Retry."))
    assert tool_results(h, "save_report")[0].startswith("ERROR(invalid_args)")
    assert h.storage.list_reports("ceo") == []
