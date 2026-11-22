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
from collections.abc import Mapping
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

from retail_agent.auth import UserContext
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.data.storage import Storage
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
from retail_agent.knowledge.golden import CandidateStore, GeminiEmbedder, GoldenIndex
from retail_agent.knowledge.schema import SchemaCatalog
from retail_agent.llm import build_chat_model
from retail_agent.observability import logger, tracing
from retail_agent.observability.logger import log_event
from retail_agent.safety import pii
from retail_agent.settings import Settings
from retail_agent.tools import ToolContext, build_tools

# input_guard, output_guard and update_context run once per turn besides agent/tools steps.
FIXED_STEPS = 3


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

    def _outcome(self, values: dict[str, Any], trace_id: str) -> TurnOutcome:
        pending = self.pending()
        if pending:
            return TurnOutcome(answer="", blocked=False, trace_id=trace_id, pending=pending)
        last = values["messages"][-1]
        answer = last.text if isinstance(last, AIMessage) else ""
        return TurnOutcome(answer=answer, blocked=bool(values.get("blocked")), trace_id=trace_id)

    def run_turn(self, text: str, trace_id: str | None = None) -> TurnOutcome:
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
        values = self.graph.invoke(turn_input, self.config(trace_id))
        return self._outcome(values, trace_id)

    def mark_good(self) -> FeedbackResult:
        """`/good` on the last answer: if it is backed by successful SQL, it becomes a
        Golden Bucket candidate for analysts to review (question, SQL, answer summary)."""
        state = self.state()
        trace_id = state.get("trace_id")
        if trace_id:
            logger.trace_id_var.set(trace_id)  # the feedback belongs to the answer's trace
        log_event("feedback", rating="good")
        thanks = "Thanks for the feedback!"
        messages = state.get("messages") or []
        turn = memory.current_turn(messages)
        answer = memory.final_answer(turn)
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
        )
        if candidate is None:
            return FeedbackResult(f"{thanks} This answer is already waiting for review.")
        return FeedbackResult(
            f"{thanks} The question and its SQL were sent to analysts for review as a "
            f"candidate example ({candidate.id}).",
            candidate_id=candidate.id,
        )

    def resume(self, answers: Mapping[str, str], trace_id: str | None = None) -> TurnOutcome:
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
        values = self.graph.invoke(Command(resume=replies), self.config(trace_id))
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


def create_session(
    user: UserContext, settings: Settings, conversation_id: str | None = None
) -> AgentSession:
    """Production wiring: BigQuery, schema, tools, Gemini models, SQLite storage/checkpoints."""
    runner = BigQueryRunner.from_settings(settings)
    schema = SchemaCatalog.load(runner)
    tool_ctx = ToolContext(
        user=user,
        conversation_id=conversation_id or "",
        runner=runner,
        schema=schema,
        settings=settings,
        storage=Storage(settings.app_db_path),
    )
    tools = build_tools(tool_ctx)
    deps = AgentDeps(
        tool_ctx=tool_ctx,
        tools=tools,
        agent_model=build_chat_model("main", tools, settings=settings),
        settings=settings,
        aux=_aux_factory(settings),
        golden=load_golden_index(settings),
        candidates=CandidateStore(settings.golden_candidates_path),
    )
    log_event("agent_ready", tools=[t.name for t in tools], golden_mode=deps.golden.mode)
    return AgentSession(deps, open_checkpointer(settings.checkpoint_db_path), conversation_id)
