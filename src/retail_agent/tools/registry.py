"""Tool registry: every capability is a `ToolSpec`, the core only asks for the user's tools.

A tool is built per session by its factory from a `ToolContext`, so the user, the
conversation and the services are captured in a closure: the LLM cannot substitute them
through tool arguments. Adding a capability (charts, e-mail, web search) means a new module
that registers its specs; the graph and the prompt do not change.
"""

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, StructuredTool

from retail_agent.auth import UserContext
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.data.storage import Storage
from retail_agent.knowledge.schema import SchemaCatalog
from retail_agent.personalization.profile import Preferences
from retail_agent.safety import confirmation
from retail_agent.settings import Settings, get_settings


@dataclass(frozen=True)
class ExecutedQuery:
    sql: str  # after the SQL Guard: qualified, ACL-filtered, limited
    purpose: str
    rows: int
    columns: tuple[str, ...]
    source: str = "run_sql"
    original_sql: str = ""  # as written by the model (passed the Guard), for follow-ups


class TurnState:
    """Per-turn counters shared by the tools; reset at the start of every turn.

    The tool node may run several tool calls of one model step in parallel threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.sql_failures = 0
        self.bq_bytes = 0
        self.executed: list[ExecutedQuery] = []
        # A deletion request in this turn matched nothing: the model must not go on to try
        # broader criteria by itself ("Q1 2025" -> "2025" -> "report").
        self.deletion_no_match = False
        self.llm_calls = 0  # model calls of this turn (agent and auxiliary), for the budget

    def reset(self) -> None:
        with self._lock:
            self.sql_failures = 0
            self.bq_bytes = 0
            self.executed = []
            self.deletion_no_match = False
            self.llm_calls = 0

    def count_llm_call(self) -> int:
        with self._lock:
            self.llm_calls += 1
            return self.llm_calls

    def add_failure(self) -> int:
        with self._lock:
            self.sql_failures += 1
            return self.sql_failures

    def reserve_bytes(self, estimate: int, budget: int) -> bool:
        """Book `estimate` bytes against the turn budget; False if they do not fit."""
        with self._lock:
            if self.bq_bytes + estimate > budget:
                return False
            self.bq_bytes += estimate
            return True

    def settle_bytes(self, reserved: int, actual: int) -> None:
        with self._lock:
            self.bq_bytes += actual - reserved

    def record(self, query: ExecutedQuery) -> None:
        with self._lock:
            self.executed.append(query)

    @property
    def executed_sql(self) -> list[str]:
        with self._lock:
            return [q.sql for q in self.executed]


@dataclass
class ToolContext:
    user: UserContext
    conversation_id: str
    runner: BigQueryRunner
    schema: SchemaCatalog
    settings: Settings = field(default_factory=get_settings)
    storage: Storage | None = None  # needed only by the report tools, checked when they run
    preferences: Preferences | None = None  # None: no personalization (preference tools say so)
    turn: TurnState = field(default_factory=TurnState)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    factory: Callable[[ToolContext], BaseTool]
    allowed_roles: frozenset[str] | None = None  # None = all roles
    side_effects: bool = False  # changes state outside the chat
    requires_confirmation: bool = False  # must ask the user through `safety.confirmation`
    capability: str = ""  # one plain-language line for the user-facing help (describe_capabilities)

    def allows(self, user: UserContext) -> bool:
        return self.allowed_roles is None or user.role in self.allowed_roles


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"Tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def specs_for(self, user: UserContext) -> list[ToolSpec]:
        return [s for s in self._specs.values() if s.allows(user)]

    def build(self, ctx: ToolContext) -> list[BaseTool]:
        return [_build(spec, ctx) for spec in self.specs_for(ctx.user)]


def _build(spec: ToolSpec, ctx: ToolContext) -> BaseTool:
    tool = spec.factory(ctx)
    if tool.name != spec.name:
        raise ValueError(f"Factory of {spec.name!r} built a tool named {tool.name!r}")
    if spec.requires_confirmation:
        if not isinstance(tool, StructuredTool) or tool.func is None:
            raise ValueError(f"{spec.name!r} requires confirmation, so it must be a sync tool")
        # Only inside this scope can the tool ask the user and perform its side effect.
        tool.func = confirmation.scoped(spec.name, tool.func)
    # Tool metadata reaches the callbacks, so `tool_call` log events carry these flags.
    tool.metadata = {
        **(tool.metadata or {}),
        "side_effects": spec.side_effects,
        "requires_confirmation": spec.requires_confirmation,
    }
    return tool


REGISTRY = ToolRegistry()


def register(specs: ToolSpec | Iterable[ToolSpec]) -> None:
    for spec in [specs] if isinstance(specs, ToolSpec) else specs:
        REGISTRY.register(spec)
