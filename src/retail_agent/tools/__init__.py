"""Agent tools. Importing a tool module registers its `ToolSpec`s in `REGISTRY`."""

from langchain_core.tools import BaseTool

from retail_agent.tools import data_tools, preference_tools, report_tools  # noqa: F401
from retail_agent.tools.registry import (
    REGISTRY,
    ExecutedQuery,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    TurnState,
    register,
)

__all__ = [
    "REGISTRY",
    "ExecutedQuery",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "TurnState",
    "build_tools",
    "register",
]


def build_tools(ctx: ToolContext, registry: ToolRegistry = REGISTRY) -> list[BaseTool]:
    """Tools available to `ctx.user`, bound to this session's context."""
    return registry.build(ctx)
