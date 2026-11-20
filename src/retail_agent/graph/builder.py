"""Graph assembly, the SQLite checkpointer and the per-user agent session.

    START -> input_guard -(blocked)-> output_guard -> END
                         -(ok)-> agent -(tool calls)-> tools -> agent
                                       -(answer)-> output_guard -> update_context -> END

`thread_id = conversation_id`: history and state survive restarts in `checkpoints.db`.
The user's message is PII-masked before it enters the state, so neither checkpoints
nor traces ever contain what the user typed in as personal data.
"""

import sqlite3
import uuid
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from retail_agent.auth import UserContext
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.graph.nodes import (
    AgentDeps,
    AgentNodes,
    route_after_agent,
    route_after_guard,
    route_after_output,
    tool_error_message,
)
from retail_agent.graph.state import AgentState
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
class TurnOutcome:
    answer: str
    blocked: bool
    trace_id: str


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

    def switch_conversation(self, conversation_id: str) -> None:
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

    def state(self) -> dict[str, Any]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": self.conversation_id}})
        return dict(snapshot.values)

    def run_turn(self, text: str, trace_id: str | None = None) -> TurnOutcome:
        """Run one turn; raises on failures the CLI turns into user messages."""
        trace_id = trace_id or logger.new_trace_id()
        masked, _ = pii.mask_text(text, where="user_input")
        turn_input = {
            "messages": [HumanMessage(masked)],
            "trace_id": trace_id,
            "blocked": False,
            "guard": None,
        }
        values = self.graph.invoke(turn_input, self.config(trace_id))
        last = values["messages"][-1]
        answer = last.text if isinstance(last, AIMessage) else ""
        return TurnOutcome(answer=answer, blocked=bool(values.get("blocked")), trace_id=trace_id)


def _aux_factory(settings: Settings):
    @cache
    def factory(schema: type[BaseModel]):
        return build_chat_model("guard", settings=settings, structured_output=schema)

    return factory


def create_session(
    user: UserContext, settings: Settings, conversation_id: str | None = None
) -> AgentSession:
    """Production wiring: BigQuery, schema, tools, Gemini models, SQLite checkpoints."""
    runner = BigQueryRunner.from_settings(settings)
    schema = SchemaCatalog.load(runner)
    tool_ctx = ToolContext(
        user=user,
        conversation_id=conversation_id or "",
        runner=runner,
        schema=schema,
        settings=settings,
    )
    tools = build_tools(tool_ctx)
    deps = AgentDeps(
        tool_ctx=tool_ctx,
        tools=tools,
        agent_model=build_chat_model("main", tools, settings=settings),
        settings=settings,
        aux=_aux_factory(settings),
    )
    log_event("agent_ready", tools=[t.name for t in tools])
    return AgentSession(deps, open_checkpointer(settings.checkpoint_db_path), conversation_id)
