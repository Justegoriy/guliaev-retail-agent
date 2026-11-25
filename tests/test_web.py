"""The Streamlit app over a harness-backed backend (scripted model, mocked BigQuery)."""

from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from harness import CEO, Harness, calls, tool_call
from retail_agent.auth import UserContext
from retail_agent.health import Check
from retail_agent.observability import inspect, logger
from retail_agent.observability.tracing import TracingStatus
from retail_agent.web import backend as web

APP = Path(__file__).resolve().parents[1] / "src" / "retail_agent" / "web" / "app.py"
BOB = UserContext("bob", "Bob", "ceo")
TIMEOUT = 30


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    harness.conversation_id = None  # like production: every session starts a new conversation
    logger.configure(harness.settings.log_path)
    yield harness
    logger._log_path = None


def backend_for(h: Harness, ready: bool = True) -> web.Backend:
    checks = [Check("Gemini", True, "ok"), Check("BigQuery", ready, "ok", fatal=not ready)]
    return web.Backend(
        settings=h.settings,
        users={"ceo": CEO, "bob": BOB},
        checks=checks,
        langsmith=TracingStatus(enabled=False),
        session_factory=h.new_session if ready else None,
    )


class App:
    """AppTest that keeps the fake backend patched in for every run."""

    def __init__(self, backend: web.Backend) -> None:
        st.cache_resource.clear()
        self.backend = backend
        self.at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)

    def run(self, step=None) -> AppTest:
        with patch.object(web, "load_backend", return_value=self.backend):
            (step or (lambda: self.at.run()))()
        assert not self.at.exception, self.at.exception
        return self.at

    def send(self, text: str) -> AppTest:
        return self.run(lambda: self.at.chat_input[0].set_value(text).run())

    def click(self, key: str) -> AppTest:
        return self.run(lambda: self.at.button(key=key).click().run())

    def view(self, name: str) -> AppTest:
        return self.run(lambda: self.at.segmented_control(key="view").set_value(name).run())

    def texts(self) -> str:
        parts = [m.value for m in self.at.markdown] + [c.value for c in self.at.caption]
        return "\n".join(parts)


@pytest.fixture
def app(h):
    return App(backend_for(h))


def turn_ends(h: Harness) -> list[dict]:
    return [e for e in inspect.read_jsonl(h.settings.log_path) if e["event"] == "turn_end"]


def test_start_page_has_user_picker_status_and_chat(app):
    at = app.run()

    assert at.sidebar.selectbox(key="user_id").value == "ceo"
    assert "Demo sign-in: a signed token" in app.texts()
    assert len(at.chat_input) == 1
    assert at.sidebar.expander[0].label == "System status"


def test_a_question_is_answered_with_its_trace_id(h, app):
    app.run()
    h.model.script.append(AIMessage("Revenue was **300**."))

    app.send("revenue in 2024")

    texts = app.texts()
    assert "revenue in 2024" in texts and "Revenue was **300**." in texts
    (end,) = turn_ends(h)
    assert end["status"] == "ok" and f"trace `{end['trace_id']}`" in texts
    assert end["user_id"] == "ceo"


def test_thumbs_up_marks_the_answer_good(h, app):
    app.run()
    h.model.script.append(AIMessage("Hello."))
    app.send("hi")

    app.run(lambda: app.at.feedback[0].set_value(1).run())

    events = [e["event"] for e in inspect.read_jsonl(h.settings.log_path)]
    assert "feedback" in events
    assert not app.at.feedback  # one rating per answer


def test_thumbs_down_takes_an_optional_comment(h, app):
    app.run()
    h.model.script.append(AIMessage("A very long answer."))
    app.send("hi")
    app.run(lambda: app.at.feedback[0].set_value(0).run())

    app.at.text_input[0].set_value("too long")
    app.run(lambda: app.at.form_submit_button[0].click().run())

    (fb,) = [e for e in inspect.read_jsonl(h.settings.log_path) if e["event"] == "feedback"]
    assert fb["rating"] == "bad" and "too long" in fb["comment"]


def delete_request(h: Harness, app: App) -> str:
    report = h.storage.save_report("ceo", "other", "Q1 review", "text")
    app.run()
    h.model.script.append(
        calls(tool_call("request_report_deletion", {"report_ids": [report.id]}, "d1"))
    )
    app.send(f"delete report {report.id}")
    return report.id


def test_deletion_waits_for_the_delete_button(h, app):
    delete_request(h, app)
    assert "Delete 1 report?" in app.texts()
    assert len(h.storage.list_reports("ceo")) == 1

    h.model.script.append(AIMessage("Deleted."))
    app.click("confirm_yes")

    assert h.storage.list_reports("ceo") == []
    assert "Deleted." in app.texts()
    assert [e["status"] for e in turn_ends(h)] == ["ok"]


def test_cancel_button_keeps_the_report(h, app):
    delete_request(h, app)
    h.model.script.append(AIMessage("Nothing was deleted."))

    app.click("confirm_no")

    assert len(h.storage.list_reports("ceo")) == 1
    assert "Nothing was deleted." in app.texts()


def test_a_new_message_cancels_the_pending_confirmation(h, app):
    delete_request(h, app)
    h.model.script.extend([AIMessage("Not deleted."), AIMessage("Hello.")])

    app.send("hello instead")

    assert len(h.storage.list_reports("ceo")) == 1
    assert not any(b.key == "confirm_yes" for b in app.at.button)
    assert "Hello." in app.texts()
    assert len(turn_ends(h)) == 2


def test_preferences_can_be_set_forgotten_and_validated(h, app):
    app.run()
    app.view("Preferences")
    app.at.selectbox(key="pref_key").set_value("format")
    app.at.text_input(key="pref_value").set_value("table")
    app.click("pref_save")

    assert h.preferences.get_profile("ceo").get("format").value == "table"
    assert "Saved: format = table" in [s.value for s in app.at.success][0]
    assert app.at.dataframe[0].value["value"].tolist() == ["table"]

    app.at.text_input(key="pref_value").set_value("hologram")
    app.click("pref_save")
    assert app.at.warning and "Not saved" in app.at.warning[0].value

    app.click("pref_forget")
    assert h.preferences.get_profile("ceo").items == ()


def test_reports_are_listed_with_a_download(h, app):
    h.storage.save_report("ceo", "c1", "Q3 review", "# Q3\n\nJeans grew.")
    h.storage.save_report("bob", "c2", "Bob's report", "secret")
    app.run()

    at = app.view("Reports")

    labels = [e.label for e in at.expander]
    assert any("Q3 review" in label for label in labels)
    assert not any("Bob" in label for label in labels)
    assert "Jeans grew." in app.texts()
    assert len(at.get("download_button")) == 1


def test_switching_user_shows_none_of_the_previous_users_data(h, app):
    app.run()
    h.model.script.append(AIMessage("CEO answer."))
    app.send("ceo question")
    h.storage.save_report("ceo", "c1", "CEO report", "text")

    app.run(lambda: app.at.sidebar.selectbox(key="user_id").set_value("bob").run())

    texts = app.texts()
    assert "ceo question" not in texts and "CEO answer." not in texts
    assert not any(b.key and b.key.startswith("conv_") for b in app.at.sidebar.button)
    app.view("Reports")
    assert "CEO report" not in app.texts()


def test_conversations_can_be_switched_in_the_sidebar(h, app):
    app.run()
    h.model.script.append(AIMessage("First answer."))
    app.send("first question")
    app.click("new_conversation")
    assert "First answer." not in app.texts()

    (old,) = [b for b in app.at.sidebar.button if b.key and b.key.startswith("conv_")]
    app.click(old.key)

    assert "First answer." in app.texts()


def test_startup_failure_is_explained(h):
    app = App(backend_for(h, ready=False))
    at = app.run()

    assert at.error and "could not start" in at.error[0].value
    assert not at.chat_input


def test_token_in_the_url_signs_in_with_its_scope(h, app):
    from retail_agent import auth

    manager = UserContext("nike_manager", "Nora", "manager", ("Nike",))
    app.at.query_params["token"] = auth.issue_token(manager, h.settings)
    app.run()
    texts = app.texts()
    assert "Signed in with a token: **Nora** (manager)" in texts and "Data access: Nike" in texts
    assert not app.at.sidebar.selectbox  # no user picker when the frontend passes a token
    assert app.at.session_state["agent"].user == manager  # built from the verified token


def test_bad_token_in_the_url_is_refused(h, app):
    app.at.query_params["token"] = "garbage"
    app.run()
    errors = [e.value for e in app.at.error]
    assert any("Sign-in failed: invalid token" in e for e in errors)
    assert not any("could not start" in e for e in errors)
    assert not app.at.chat_input


def conversation_ids(h):
    return [c.id for c in h.storage.list_conversations("ceo")]


def test_conversation_is_deleted_after_confirmation(h, app):
    app.run()
    h.model.script.append(AIMessage("First answer."))
    app.send("first question")
    app.click("new_conversation")
    h.model.script.append(AIMessage("Second answer."))
    app.send("second question")
    current, old = conversation_ids(h)  # newest first

    app.click(f"del_{old}")
    assert any("for good" in w.value for w in app.at.sidebar.warning)
    app.click(f"del_no_{old}")  # cancel keeps it
    assert old in conversation_ids(h) and not app.at.sidebar.warning

    app.click(f"del_{old}")
    app.click(f"del_yes_{old}")
    assert conversation_ids(h) == [current]
    assert "Second answer." in app.texts()  # the open conversation is untouched
    assert "Conversation deleted." in [s.value for s in app.at.success]


def test_deleting_the_open_conversation_clears_the_chat(h, app):
    app.run()
    h.model.script.append(AIMessage("Only answer."))
    app.send("only question")
    (only,) = conversation_ids(h)
    app.click(f"del_{only}")
    app.click(f"del_yes_{only}")
    assert conversation_ids(h) == []
    assert "Only answer." not in app.texts()
