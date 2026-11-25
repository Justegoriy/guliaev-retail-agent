"""Graph assembly, the SQLite checkpointer and the per-user agent session.

    START -> input_guard -(blocked)-> output_guard -> END
                         -(ok)-> agent -(tool calls)-> tools -> agent
                                       -(answer)-> output_guard -> update_context -> END

`thread_id = conversation_id`: history and state survive restarts in `checkpoints.db`.
The user's message is PII-masked before it enters the state, so neither checkpoints
nor traces ever contain what the user typed in as personal data.

Tools that need the user's confirmation pause the graph with `interrupt()`; the session
reports them as `TurnOutcome.pending` and continues with `resume(answers)`.
"""

import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel

from retail_agent import retention
from retail_agent.auth import UserContext
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.data.storage import Conversation, ConversationAccessError, Report, Storage
from retail_agent.graph import memory
from retail_agent.graph.nodes import (
    AgentDeps,
    AgentNodes,
    route_after_agent,
    route_after_guard,
    route_after_output,
    tool_error_message,
)
from retail_agent.graph.state import AgentState
from retail_agent.knowledge import promotion
from retail_agent.knowledge.golden import CandidateStore, GeminiEmbedder, GoldenIndex
from retail_agent.knowledge.schema import SchemaCatalog
from retail_agent.llm import build_chat_model
from retail_agent.observability import logger, tracing
from retail_agent.observability.logger import log_event
from retail_agent.personalization import profile as preference_profile
from retail_agent.personalization import signals
from retail_agent.personalization.profile import (
    Preference,
    PreferenceRejected,
    Preferences,
    UserProfile,
)
from retail_agent.prompts.context import PromptSources, ReloadingFile, persona_text
from retail_agent.safety import pii
from retail_agent.settings import Settings
from retail_agent.tools import ToolContext, build_tools

# input_guard, output_guard and update_context run once per turn besides agent/tools steps.
FIXED_STEPS = 3

StageCallback = Callable[[str], None]
STAGE_START = "Checking request..."
CANCELLED_TEXT = "(The user cancelled this request before it finished.)"
CANCELLED_TOOL_TEXT = "Cancelled by the user."
DATA_TOOLS = frozenset({"run_sql", "describe_dataset"})


def stage_after(node: str, update: Any) -> str | None:
    """What the turn is doing next, after `node` produced `update` (for the CLI status)."""
    if node == "input_guard":
        return "Thinking..."
    if node == "agent":
        messages = (update.get("messages") or []) if isinstance(update, dict) else []
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            names = {c["name"] for c in last.tool_calls}
            return "Querying BigQuery..." if names & DATA_TOOLS else "Working on it..."
        return "Checking the answer..."
    if node == "tools":
        return "Writing answer..."
    if node == "output_guard":
        return "Updating conversation context..."
    return None


def build_graph(
    deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    nodes = AgentNodes(deps)
    graph = StateGraph(AgentState)
    graph.add_node("input_guard", nodes.input_guard)
    graph.add_node("agent", nodes.agent)
    graph.add_node("tools", ToolNode(deps.tools, handle_tool_errors=tool_error_message))
    graph.add_node("output_guard", nodes.output_guard)
    graph.add_node("update_context", nodes.update_context)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", route_after_guard, ["agent", "output_guard"])
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "output_guard"])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "output_guard", route_after_output, {"end": END, "update_context": "update_context"}
    )
    graph.add_edge("update_context", END)
    return graph.compile(checkpointer=checkpointer)


def open_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The tool node runs tool calls in worker threads; the saver serialises access itself.
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


@dataclass(frozen=True)
class PendingConfirmation:
    """The graph is paused in a tool that waits for the user's yes/no."""

    interrupt_id: str
    action: str  # delete_reports | export_report | ...
    question: str  # e.g. "Delete 2 reports?"
    summary: str
    items: list[dict[str, Any]]
    digest: str


@dataclass(frozen=True)
class TurnOutcome:
    answer: str
    blocked: bool
    trace_id: str
    pending: tuple[PendingConfirmation, ...] = ()  # non-empty: answer them via `resume`


@dataclass(frozen=True)
class FeedbackResult:
    message: str  # for the user
    candidate_id: str | None = None  # set when the answer became a Golden Bucket candidate


class AgentSession:
    """One user's conversation(s) in one process; the CLI talks only to this class."""

    def __init__(
        self,
        deps: AgentDeps,
        checkpointer: BaseCheckpointSaver,
        conversation_id: str | None = None,
    ) -> None:
        self.deps = deps
        self.settings = deps.settings
        self.graph = build_graph(deps, checkpointer)
        self.conversation_id = ""
        self.switch_conversation(conversation_id or uuid.uuid4().hex)

    @property
    def user(self) -> UserContext:
        return self.deps.tool_ctx.user

    @property
    def storage(self) -> Storage | None:
        return self.deps.tool_ctx.storage

    def switch_conversation(self, conversation_id: str) -> None:
        """Raises `ConversationAccessError` for another user's conversation."""
        if self.storage is not None:
            self.storage.check_conversation_owner(conversation_id, self.user.user_id)
        self.conversation_id = conversation_id
        self.deps.tool_ctx.conversation_id = conversation_id
        logger.conversation_id_var.set(conversation_id)

    def new_conversation(self) -> str:
        self.switch_conversation(uuid.uuid4().hex)
        return self.conversation_id

    def config(self, trace_id: str) -> RunnableConfig:
        config = tracing.turn_config(
            self.settings,
            trace_id=trace_id,
            conversation_id=self.conversation_id,
            user_role=self.user.role,
        )
        config["configurable"] = {"thread_id": self.conversation_id}
        config["recursion_limit"] = self.settings.max_agent_steps * 2 + FIXED_STEPS
        return config

    def _snapshot(self):
        return self.graph.get_state({"configurable": {"thread_id": self.conversation_id}})

    def state(self) -> dict[str, Any]:
        return dict(self._snapshot().values)

    def pending(self) -> tuple[PendingConfirmation, ...]:
        """Confirmations the conversation is waiting for (also after a restart)."""
        out = []
        for item in self._snapshot().interrupts:
            value = item.value if isinstance(item.value, dict) else {}
            if value.get("type") != "confirm":
                continue
            out.append(
                PendingConfirmation(
                    interrupt_id=item.id,
                    action=str(value.get("action", "")),
                    question=str(value.get("question", "Proceed?")),
                    summary=str(value.get("summary", "")),
                    items=list(value.get("items") or []),
                    digest=str(value.get("digest", "")),
                )
            )
        return tuple(out)

    def cancel_turn(self) -> None:
        """Close a turn stopped mid-graph (Ctrl+C) so the thread stays valid.

        Tool calls without results would make the next model request invalid, so they get
        a "cancelled" result, and the turn ends with a short assistant note. A pending
        confirmation is left alone: the next message supersedes it (D12).
        """
        snapshot = self._snapshot()
        if not snapshot.next or self.pending():
            return
        messages = snapshot.values.get("messages") or []
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        closing: list[Any] = []
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, AIMessage):
                closing += [
                    ToolMessage(CANCELLED_TOOL_TEXT, tool_call_id=c["id"], name=c["name"])
                    for c in message.tool_calls
                    if c["id"] not in answered
                ]
        closing.append(AIMessage(CANCELLED_TEXT))
        self.graph.update_state(
            {"configurable": {"thread_id": self.conversation_id}},
            {"messages": closing},
            as_node="update_context",
        )
        log_event("turn_cancelled", interrupted_at=list(snapshot.next))

    def _outcome(self, values: dict[str, Any], trace_id: str) -> TurnOutcome:
        pending = self.pending()
        if pending:
            return TurnOutcome(answer="", blocked=False, trace_id=trace_id, pending=pending)
        last = values["messages"][-1]
        answer = last.text if isinstance(last, AIMessage) else ""
        return TurnOutcome(answer=answer, blocked=bool(values.get("blocked")), trace_id=trace_id)

    def _stream(self, graph_input: Any, trace_id: str, on_stage: StageCallback | None) -> dict:
        """Run the graph, reporting progress after each node; returns the final state."""
        stream = self.graph.stream(graph_input, self.config(trace_id), stream_mode="updates")
        try:
            for chunk in stream:
                for node, update in chunk.items():
                    stage = stage_after(node, update)
                    if stage and on_stage is not None:
                        on_stage(stage)
        finally:
            stream.close()
        return self.state()

    def run_turn(
        self, text: str, trace_id: str | None = None, on_stage: StageCallback | None = None
    ) -> TurnOutcome:
        """Run one turn; raises on failures the CLI turns into user messages.

        A confirmation left unanswered (D12: it lives until the next user input) is
        cancelled first, so the new message never resumes it by accident.
        """
        trace_id = trace_id or logger.new_trace_id()
        if self.pending():
            log_event("confirmation_superseded", level="warning")
            self.resume({}, trace_id)
        masked, _ = pii.mask_text(text, where="user_input")
        if self.storage is not None:
            self.storage.record_turn(self.conversation_id, self.user.user_id, masked)
        turn_input = {
            "messages": [HumanMessage(masked)],
            "trace_id": trace_id,
            "blocked": False,
            "guard": None,
        }
        if on_stage is not None:
            on_stage(STAGE_START)
        values = self._stream(turn_input, trace_id, on_stage)
        return self._outcome(values, trace_id)

    # -- conversations ----------------------------------------------------------------

    def conversations(self, limit: int = 20) -> list[Conversation]:
        if self.storage is None:
            return []
        return self.storage.list_conversations(self.user.user_id, limit=limit)

    def find_conversation(self, id_or_prefix: str) -> Conversation | None:
        """The user's conversation by full id or a unique prefix (as shown by `/history`).

        Raises `ConversationAccessError` for the full id of another user's conversation.
        """
        wanted = id_or_prefix.strip().lower()
        if self.storage is None or not wanted:
            return None
        other = self.storage.get_conversation(wanted)
        if other is not None and other.owner_user_id != self.user.user_id:
            log_event("conversation_access_denied", level="warning", target=wanted)
            raise ConversationAccessError(wanted)
        matches = [
            c
            for c in self.storage.list_conversations(self.user.user_id, limit=10_000)
            if c.id.startswith(wanted)
        ]
        exact = [c for c in matches if c.id == wanted]
        if exact:
            return exact[0]
        return matches[0] if len(matches) == 1 else None

    def last_conversation(self) -> Conversation | None:
        recent = self.conversations(limit=1)
        return recent[0] if recent else None

    def recent_exchanges(self, n: int = 3) -> list[tuple[str, str]]:
        """(user, assistant) pairs of the last `n` turns of the current conversation."""
        return memory.recent_exchanges(self.state().get("messages") or [], n)

    def reports(self, limit: int = 20) -> list[Report]:
        if self.storage is None:
            return []
        return self.storage.list_reports(self.user.user_id, limit=limit)

    @property
    def preferences(self) -> Preferences | None:
        return self.deps.tool_ctx.preferences

    def _require_preferences(self) -> Preferences:
        if self.preferences is None:
            raise PreferenceRejected("preferences are not available in this session")
        return self.preferences

    def profile(self) -> UserProfile:
        prefs = self.preferences
        return prefs.get_profile(self.user.user_id) if prefs else UserProfile()

    def set_preference(self, key: str, value: str, via: str = "cli") -> Preference:
        """`/prefs set` or the web form; raises `PreferenceRejected` with a reason for the user."""
        return self._require_preferences().set_explicit(
            self.user.user_id, key, value, via=via, conversation_id=self.conversation_id
        )

    def forget_preference(self, key: str) -> bool:
        return self._require_preferences().forget(self.user.user_id, key)

    def reset_preferences(self) -> int:
        return self._require_preferences().reset(self.user.user_id)

    def _feedback(self, rating: str, comment: str = "") -> tuple[dict[str, Any], str]:
        """Log the feedback on the last answer and turn it into preference signals."""
        state = self.state()
        trace_id = state.get("trace_id")
        if trace_id:
            logger.trace_id_var.set(trace_id)  # the feedback belongs to the answer's trace
        comment = pii.mask_text(comment, where="feedback")[0][:300]
        log_event("feedback", rating=rating, **({"comment": comment} if comment else {}))
        answer = memory.final_answer(memory.current_turn(state.get("messages") or []))
        prefs = self.preferences
        if prefs is not None and answer:
            try:
                rated = any(
                    s.kind == "feedback" and s.trace_id == trace_id
                    for s in prefs.storage.signals(self.user.user_id)
                )
                if rated:  # one answer gives at most one feedback signal set
                    return state, answer
                prefs.add_signals(
                    self.user.user_id,
                    signals.feedback_signals(rating, answer, comment),
                    conversation_id=self.conversation_id,
                    trace_id=trace_id,
                )
            except Exception as exc:
                log_event(
                    "preference_extraction_failed",
                    level="warning",
                    stage="feedback",
                    error=type(exc).__name__,
                )
        return state, answer

    def _rate_precedents(self, state: dict[str, Any], rating: str) -> None:
        """Count the rating against the Golden Bucket trios that were in the answer's prompt;
        trios with bad ratings from several users are deprecated (`knowledge.promotion`)."""
        storage, trio_ids = self.deps.tool_ctx.storage, state.get("golden_ids") or []
        trace_id = state.get("trace_id")
        if storage is None or not trio_ids or not trace_id:
            return
        try:
            deprecated = promotion.record_feedback(
                storage,
                self.deps.settings.golden_trios_path,
                trio_ids=trio_ids,
                user_id=self.user.user_id,
                rating=rating,
                trace_id=trace_id,
                index=self.deps.golden,
            )
        except Exception as exc:
            log_event("golden_feedback_failed", level="warning", error=type(exc).__name__)
            return
        log_event("golden_feedback", rating=rating, trio_ids=trio_ids, deprecated=deprecated)

    def mark_bad(self, comment: str = "") -> FeedbackResult:
        """`/bad [comment]`: logged for analysis; "too long" and the like adjust the profile."""
        state, answer = self._feedback("bad", comment)
        if answer:
            self._rate_precedents(state, "bad")
        if not answer:
            return FeedbackResult("Thanks for the feedback! There is no answer to rate yet.")
        hint = "" if comment else " Add a comment next time, e.g. /bad too long."
        return FeedbackResult(f"Thanks for the feedback, it helps to improve the answers.{hint}")

    def mark_good(self) -> FeedbackResult:
        """`/good` on the last answer: if it is backed by successful SQL, it becomes a
        Golden Bucket candidate for analysts to review (question, SQL, answer summary)."""
        state, answer = self._feedback("good")
        if answer:
            self._rate_precedents(state, "good")
        trace_id = state.get("trace_id")
        thanks = "Thanks for the feedback!"
        messages = state.get("messages") or []
        turn = memory.current_turn(messages)
        succeeded = any(
            isinstance(m, ToolMessage) and m.name == "run_sql" and m.text.startswith("OK")
            for m in turn
        )
        context = state.get("analysis_context")
        store = self.deps.candidates
        if store is None or not trace_id or not answer or self.pending():
            return FeedbackResult(thanks)
        if not succeeded or not context or not context.get("sql"):
            return FeedbackResult(
                f"{thanks} The last answer did not run a data query, so there is nothing "
                "to add to the analysts' knowledge base."
            )
        candidate = store.add(
            question=context.get("question") or memory.last_user_text(messages),
            sql=context["sql"],
            answer_summary=answer,
            user_role=self.user.role,
            trace_id=trace_id,
            tags=context.get("entities") or [],
            user_id=self.user.user_id,
            brands=self.user.brands,
        )
        if candidate is None:
            return FeedbackResult(f"{thanks} This answer is already waiting for review.")
        return FeedbackResult(
            f"{thanks} The question and its SQL were queued as a candidate example for "
            f"the analysts' knowledge base ({candidate.id}); it is checked automatically.",
            candidate_id=candidate.id,
        )

    def resume(
        self,
        answers: Mapping[str, str],
        trace_id: str | None = None,
        on_stage: StageCallback | None = None,
    ) -> TurnOutcome:
        """Answer the pending confirmations (interrupt id -> user's text) and continue.

        Unanswered ones count as "no". Each answer carries the digest of the preview the
        user saw, so the tool can tell whether its targets changed in the meantime.
        """
        trace_id = trace_id or self.state().get("trace_id") or logger.new_trace_id()
        logger.trace_id_var.set(trace_id)
        replies = {
            p.interrupt_id: {"answer": answers.get(p.interrupt_id, "no"), "digest": p.digest}
            for p in self.pending()
        }
        if not replies:
            raise RuntimeError("there is no pending confirmation to answer")
        values = self._stream(Command(resume=replies), trace_id, on_stage)
        return self._outcome(values, trace_id)


def _aux_factory(settings: Settings):
    @cache
    def factory(schema: type[BaseModel]):
        return build_chat_model("guard", settings=settings, structured_output=schema)

    return factory


def load_golden_index(settings: Settings) -> GoldenIndex:
    """Trios with cached embeddings; without an API key, keyword retrieval only."""
    embeddings = None
    if settings.google_api_key is not None:
        embeddings = GeminiEmbedder(
            settings.google_api_key.get_secret_value(),
            settings.gemini_embedding_model,
            timeout_s=settings.embedding_timeout_s,
        )
    return GoldenIndex.from_file(
        settings.golden_trios_path,
        embeddings=embeddings,
        index_path=settings.golden_index_path,
        model_name=settings.gemini_embedding_model,
        top_k=settings.golden_top_k,
        min_score=settings.golden_min_score,
        keyword_min_score=settings.golden_keyword_min_score,
    )


@dataclass(frozen=True)
class AgentResources:
    """What every user's session shares: clients, schema, knowledge, storage (thread-safe)."""

    settings: Settings
    runner: BigQueryRunner
    schema: SchemaCatalog
    storage: Storage
    preferences: Preferences
    golden: GoldenIndex
    candidates: CandidateStore
    checkpointer: BaseCheckpointSaver
    aux: Any

    def session(self, user: UserContext, conversation_id: str | None = None) -> AgentSession:
        """A session for `user`: tools bound to their access rights, own conversations."""
        tool_ctx = ToolContext(
            user=user,
            conversation_id=conversation_id or "",
            runner=self.runner,
            schema=self.schema,
            settings=self.settings,
            storage=self.storage,
            preferences=self.preferences,
        )
        tools = build_tools(tool_ctx)
        deps = AgentDeps(
            tool_ctx=tool_ctx,
            tools=tools,
            agent_model=build_chat_model("main", tools, settings=self.settings),
            answer_model=build_chat_model("main", settings=self.settings),
            settings=self.settings,
            prompt_sources=PromptSources(
                persona=ReloadingFile(self.settings.persona_path, "persona", persona_text)
            ),
            aux=self.aux,
            golden=self.golden,
            candidates=self.candidates,
        )
        log_event("agent_ready", tools=[t.name for t in tools], golden_mode=self.golden.mode)
        return AgentSession(deps, self.checkpointer, conversation_id)


def create_resources(settings: Settings, runner: BigQueryRunner | None = None) -> AgentResources:
    """Production wiring: BigQuery, schema, Gemini, golden index, SQLite storage/checkpoints."""
    runner = runner or BigQueryRunner.from_settings(settings)
    storage = Storage(settings.app_db_path)
    checkpointer = open_checkpointer(settings.checkpoint_db_path)
    candidates = CandidateStore(settings.golden_candidates_path)
    try:
        retention.purge_expired(settings, storage, checkpointer, candidates)
    except Exception as exc:  # retention must never stop the agent from starting
        log_event("retention_failed", level="error", error=type(exc).__name__, detail=str(exc))
    return AgentResources(
        settings=settings,
        runner=runner,
        schema=SchemaCatalog.load(runner),
        storage=storage,
        preferences=Preferences(
            storage, preference_profile.load_config(settings.preferences_config_path)
        ),
        golden=load_golden_index(settings),
        candidates=candidates,
        checkpointer=checkpointer,
        aux=_aux_factory(settings),
    )


def create_session(
    user: UserContext,
    settings: Settings,
    conversation_id: str | None = None,
    runner: BigQueryRunner | None = None,
) -> AgentSession:
    return create_resources(settings, runner).session(user, conversation_id)
