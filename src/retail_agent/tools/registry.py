"""Tool registry: every capability is a `ToolSpec`, the core only asks for the user's tools.

A tool is built per session by its factory from a `ToolContext`, so the user, the
conversation and the services are captured in a closure: the LLM cannot substitute them
through tool arguments. Adding a capability (charts, e-mail, web search) means a new module
that registers its specs; the graph and the prompt do not change.
"""

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool

from retail_agent.auth import UserContext
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.knowledge.schema import SchemaCatalog
from retail_agent.settings import Settings, get_settings


@dataclass(frozen=True)
class ExecutedQuery:
    sql: str  # after the SQL Guard: qualified, ACL-filtered, limited
    purpose: str
    rows: int
    columns: tuple[str, ...]
    source: str = "run_sql"  # or "benchmark"
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

    def reset(self) -> None:
        with self._lock:
            self.sql_failures = 0
            self.bq_bytes = 0
            self.executed = []

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
    storage: Any = None
    turn: TurnState = field(default_factory=TurnState)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    factory: Callable[[ToolContext], BaseTool]
    allowed_roles: frozenset[str] | None = None  # None = all roles
    side_effects: bool = False  # changes state outside the chat
    requires_confirmation: bool = False  # must ask the user through `safety.confirmation`

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
