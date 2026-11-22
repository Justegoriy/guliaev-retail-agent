"""Demo web interface: the same agent core as the CLI, in a browser tab (TZ D28).

Run with `python -m retail_agent.web`. There is no authentication: the user is picked in
the sidebar, and each browser tab gets its own `AgentSession` for that user.
"""

import time
from collections.abc import Callable

import streamlit as st

from retail_agent.graph.builder import AgentSession, PendingConfirmation
from retail_agent.observability.logger import log_event
from retail_agent.personalization.profile import PreferenceRejected, display
from retail_agent.turns import Turn, TurnResult
from retail_agent.web import backend as web

CONVERSATIONS_SHOWN = 12
REPORTS_SHOWN = 50
VIEWS = {
    "Chat": ":material/chat:",
    "Preferences": ":material/tune:",
    "Reports": ":material/description:",
}

st.set_page_config(page_title="Retail analytics agent", page_icon=":material/insights:")


@st.cache_resource(show_spinner="Checking Gemini and BigQuery, loading schema...")
def get_backend() -> web.Backend:
    return web.load_backend()


state = st.session_state


# -- per-tab state --------------------------------------------------------------------------


def reset_view_state() -> None:
    """Forget everything tied to the previous user or conversation."""
    state.turn = None  # a Turn waiting for a confirmation
    state.last = None  # {"trace_id", "conversation_id", "url"} of the last answered turn
    state.notice = None  # (level, text) of a failed or cancelled turn
    state.feedback = {}  # trace_id -> message shown after рџ‘Ќ/рџ‘Ћ
    state.flash = None  # (level, text) shown once at the top of a view


def close_pending() -> None:
    """Leaving a conversation with an open confirmation cancels it (D12)."""
    turn: Turn | None = state.get("turn")
    if turn is not None and turn.session.pending():
        turn.resume({})
    state.turn = None


def ensure_session(backend: web.Backend, user_id: str) -> AgentSession:
    if state.get("session_user") != user_id or state.get("agent") is None:
        close_pending()
        state.agent = backend.session(backend.users[user_id])
        state.session_user = user_id
        reset_view_state()
    session: AgentSession = state.agent
    web.bind_context(session)
    return session


# -- sidebar --------------------------------------------------------------------------------


def sidebar(backend: web.Backend) -> AgentSession | None:
    with st.sidebar:
        st.header("Retail analytics agent")
        users = backend.users
        user_id = st.selectbox(
            "User",
            list(users),
            format_func=lambda u: f"{users[u].name} В· {users[u].role}",
            key="user_id",
        )
        user = users[user_id]
        st.caption(
            f":material/lock_open: Demo without authentication.  \n"
            f"Data access: {web.access_label(user)}"
        )
        session = ensure_session(backend, user_id) if backend.ready else None
        if session is not None:
            conversations_panel(session)
        status_panel(backend)
    return session


def conversations_panel(session: AgentSession) -> None:
    st.subheader("Conversations")
    if st.button(
        "New conversation", icon=":material/add:", key="new_conversation", width="stretch"
    ):
        close_pending()
        session.new_conversation()
        log_event("conversation_started")
        reset_view_state()
        st.rerun()
    for c in session.conversations(limit=CONVERSATIONS_SHOWN):
        current = c.id == session.conversation_id
        title = c.title if len(c.title) <= 40 else c.title[:39] + "вЂ¦"
        clicked = st.button(
            title,
            key=f"conv_{c.id}",
            type="primary" if current else "secondary",
            help=f"{c.turns} turns В· updated {web.short_date(c.updated_at)} UTC",
            width="stretch",
        )
        if clicked and not current:
            close_pending()
            session.switch_conversation(c.id)
            web.bind_context(session)
            log_event("conversation_resumed", turns=c.turns, client="web")
            reset_view_state()
            st.rerun()


def status_panel(backend: web.Backend) -> None:
    s = backend.settings
    with st.expander("System status", icon=":material/monitor_heart:"):
        for check in backend.checks:
            if check.ok:
                icon = ":green[:material/check_circle:]"
            elif check.fatal:
                icon = ":red[:material/error:]"
            else:
                icon = ":orange[:material/warning:]"
            st.markdown(f"{icon} **{check.name}**: {check.message}")
        st.markdown(f"Model: `{s.gemini_model}`, fallback `{s.gemini_fallback_model}`")
        if backend.langsmith.enabled:
            st.markdown(f"LangSmith tracing: on (project `{s.langsmith_project}`)")
        else:
            st.markdown("LangSmith tracing: off")


# -- chat -----------------------------------------------------------------------------------


def run_turn(
    session: AgentSession, turn: Turn, action: Callable[[Callable[[str], None]], TurnResult]
) -> None:
    """Run one graph call with a live stage line, remember the result, refresh the page."""
    state.notice = None
    with (
        st.chat_message("assistant"),
        st.status(":shimmer[Checking request...]", type="compact") as status,
    ):
        started = time.perf_counter()
        result = action(lambda stage: status.update(label=f":shimmer[{stage}]"))
        seconds = time.perf_counter() - started
        failed = result.status in ("error", "cancelled")
        status.update(
            label=f"{'Stopped' if failed else 'Done'} in {seconds:.1f} s",
            state="error" if failed else "complete",
        )
    state.turn = turn if result.status == "pending" else None
    if result.message:
        state.notice = ("warning" if result.status == "cancelled" else "error", result.message)
    if result.status in ("ok", "blocked"):
        backend = get_backend()
        url = (
            web.langsmith_url(backend.settings, result.trace_id)
            if backend.langsmith.enabled
            else None
        )
        state.last = {
            "trace_id": result.trace_id,
            "conversation_id": session.conversation_id,
            "url": url,
        }
    st.rerun()


def send(session: AgentSession, text: str) -> None:
    close_pending()
    with st.chat_message("user"):
        st.markdown(text)
    turn = Turn(session)
    run_turn(session, turn, lambda on_stage: turn.start(text, on_stage))


def confirmation(session: AgentSession, pending: tuple[PendingConfirmation, ...]) -> None:
    with st.chat_message("assistant"):
        for item in pending:
            st.markdown(f"**{item.question}**")
            if item.items:
                rows = [
                    {
                        "id": str(i.get("id", "")),
                        "title": str(i.get("title", "")),
                        "created (UTC)": web.short_date(str(i.get("created_at", ""))),
                        "preview": str(i.get("snippet", "")),
                    }
                    for i in item.items
                ]
                st.dataframe(rows, hide_index=True, width="stretch")
            elif item.summary:
                st.caption(item.summary)
        label = web.confirm_label([p.action for p in pending])
        with st.container(horizontal=True):
            yes = st.button(label, key="confirm_yes", type="primary", icon=":material/check:")
            no = st.button("Cancel", key="confirm_no", icon=":material/close:")
        st.caption("Sending a new message cancels this request.")
    if yes or no:
        turn: Turn = state.get("turn") or Turn(session)
        answers = {p.interrupt_id: "yes" for p in pending} if yes else {}
        run_turn(session, turn, lambda on_stage: turn.resume(answers, on_stage))


def feedback(session: AgentSession, trace_id: str) -> None:
    given = state.feedback.get(trace_id)
    if given:
        st.caption(f":material/check: {given}")
        return
    value = st.feedback("thumbs", key=f"fb_{trace_id}")
    if value is None:
        return
    if value == 1:
        result = session.mark_good()
        state.feedback[trace_id] = result.message
        st.rerun()
    with st.form(f"bad_{trace_id}", border=False):
        comment = st.text_input(
            "What was wrong? (optional)", placeholder="e.g. too long, wrong period"
        )
        if st.form_submit_button("Send feedback"):
            result = session.mark_bad(comment.strip())
            state.feedback[trace_id] = result.message
            st.rerun()


def chat_view(session: AgentSession) -> None:
    exchanges = web.transcript(session)
    if not exchanges:
        st.caption(
            "Ask about sales, products or customers, e.g. *Top 5 categories by revenue in "
            "2024* or *How many new customers did we get last month?*"
        )
    for question, answer in exchanges:
        with st.chat_message("user"):
            st.markdown(question)
        if answer:
            with st.chat_message("assistant"):
                st.markdown(answer)

    last = state.last
    pending = session.pending()
    if last and last["conversation_id"] == session.conversation_id and not pending:
        trace_line = f"trace `{last['trace_id']}`"
        if last["url"]:
            trace_line += f" В· [LangSmith]({last['url']})"
        st.caption(trace_line)
        feedback(session, last["trace_id"])
    if state.notice:
        level, text = state.notice
        (st.warning if level == "warning" else st.error)(text)
    if pending:
        confirmation(session, pending)

    text = st.chat_input("Ask a question about the data", submit_mode="disable")
    if text and text.strip():
        send(session, text.strip())


# -- preferences ----------------------------------------------------------------------------


def preferences_view(session: AgentSession) -> None:
    prefs = session.preferences
    if prefs is None:
        st.info("Preferences are not available in this session.")
        return
    profile = session.profile()
    st.subheader("Your preferences")
    if profile.items:
        rows = [
            {
                "key": p.key,
                "value": display(p.value),
                "source": p.source,
                "confidence": round(p.confidence, 2),
                "updated (UTC)": web.short_date(p.updated_at),
            }
            for p in profile.items
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption(
            "No saved preferences yet. Set one below or tell the agent, e.g. "
            "*answer briefly, as a table*."
        )

    specs = prefs.config.keys
    with st.container(border=True):
        st.markdown("**Set a preference**")
        key = st.selectbox(
            "Key",
            list(specs),
            format_func=lambda k: f"{k}: {specs[k].description}" if specs[k].description else k,
            key="pref_key",
        )
        st.caption(f"Allowed: {specs[key].allowed()}")
        value = st.text_input("Value", key="pref_value")
        if st.button("Save", icon=":material/save:", key="pref_save"):
            try:
                saved = session.set_preference(key, value.strip(), via="web")
            except PreferenceRejected as exc:
                st.warning(f"Not saved: {exc}")
            else:
                state.flash = ("success", f"Saved: {saved.key} = {display(saved.value)}")
                st.rerun()

    if profile.items:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            forget_key = st.selectbox(
                "Forget one", [p.key for p in profile.items], key="pref_forget_key"
            )
            if st.button("Forget", icon=":material/delete:", key="pref_forget"):
                session.forget_preference(forget_key)
                state.flash = ("success", f"Forgot {forget_key}.")
                st.rerun()
            with st.popover("Reset all", icon=":material/restart_alt:"):
                st.markdown("Remove all your saved preferences?")
                if st.button("Yes, reset", type="primary", key="pref_reset"):
                    removed = session.reset_preferences()
                    state.flash = ("success", f"Preferences reset ({removed} removed).")
                    st.rerun()


# -- reports --------------------------------------------------------------------------------


def reports_view(session: AgentSession) -> None:
    reports = session.reports(limit=REPORTS_SHOWN)
    st.subheader("Your reports")
    if not reports:
        st.caption("No saved reports yet. Ask for an analysis and say *save it as a report*.")
        return
    st.caption("To delete reports, ask in the chat; deletion always asks for confirmation.")
    for r in reports:
        here = " В· this conversation" if r.conversation_id == session.conversation_id else ""
        with st.expander(f"{r.title} В· {web.short_date(r.created_at)} UTC{here}"):
            st.markdown(r.content)
            st.download_button(
                "Download .md",
                r.content,
                file_name=f"report-{r.id}.md",
                mime="text/markdown",
                icon=":material/download:",
                key=f"dl_{r.id}",
                on_click="ignore",
            )


# -- page -----------------------------------------------------------------------------------


def main() -> None:
    backend = get_backend()
    session = sidebar(backend)
    if session is None:
        st.error("The agent could not start. See *System status* in the sidebar.")
        for check in backend.checks:
            if check.fatal:
                st.markdown(f"- **{check.name}**: {check.message}")
        if backend.startup_error:
            st.markdown(f"- {backend.startup_error}")
        st.stop()

    view = st.segmented_control(
        "View",
        list(VIEWS),
        format_func=lambda v: f"{VIEWS[v]} {v}",
        default="Chat",
        required=True,
        key="view",
        label_visibility="collapsed",
    )
    if state.get("flash"):
        level, text = state.flash
        (st.success if level == "success" else st.warning)(text)
        state.flash = None
    if view == "Preferences":
        preferences_view(session)
    elif view == "Reports":
        reports_view(session)
    else:
        chat_view(session)


main()
