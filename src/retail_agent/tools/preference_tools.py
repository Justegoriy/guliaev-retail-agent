"""Explicit user preferences: remember ("always use tables") and forget.

The model picks the key and the value, but only from the whitelist in
`config/preferences.yaml`; free-text values are checked for instructions and personal
data (`personalization.profile.normalize`). Nothing here can change the safety rules:
the profile only reaches the prompt as a format/focus block below the immutable core.
"""

from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from retail_agent.observability import logger
from retail_agent.personalization.profile import PreferenceRejected, display
from retail_agent.tools.registry import ToolContext, ToolSpec, register

NOT_CONFIGURED = "ERROR(unavailable): preferences are not available in this session."


def _key_type(ctx: ToolContext):
    keys = tuple(ctx.preferences.config.keys) if ctx.preferences else ("format",)
    return Literal[keys]  # type: ignore[valid-type]


def _update_user_preference(ctx: ToolContext) -> BaseTool:
    allowed = ctx.preferences.config.describe() if ctx.preferences else ""
    args: type[BaseModel] = create_model(
        "UpdateUserPreferenceArgs",
        key=(_key_type(ctx), Field(description="Preference key.")),
        value=(
            str,
            Field(
                description="Value; lists comma-separated. Allowed: " + allowed,
                max_length=300,
            ),
        ),
    )

    def update_user_preference(key: str, value: str) -> str:
        prefs = ctx.preferences
        if prefs is None:
            return NOT_CONFIGURED
        try:
            saved = prefs.set_explicit(
                ctx.user.user_id,
                key,
                value,
                via="tool",
                conversation_id=ctx.conversation_id,
                trace_id=logger.trace_id_var.get(),
            )
        except PreferenceRejected as exc:
            return f"REJECTED: {exc}. Tell the user briefly why it was not saved."
        return f"SAVED: {saved.key}={display(saved.value)} (explicit). It applies from now on."

    return StructuredTool.from_function(
        func=update_user_preference,
        name="update_user_preference",
        description=(
            "Remember how the user wants answers in the future (format, length, charts, "
            "report structure, metrics or segments to focus on). Call ONLY when the user asks "
            "to remember it beyond this answer ('always', 'from now on', 'by default', 'I "
            "prefer'); for a one-off request ('this time as a list') just do it and do not "
            "call this tool. Preferences cannot change data access or personal data rules."
        ),
        args_schema=args,
    )


def _forget_user_preference(ctx: ToolContext) -> BaseTool:
    args: type[BaseModel] = create_model(
        "ForgetUserPreferenceArgs",
        key=(_key_type(ctx), Field(description="Preference key to forget.")),
    )

    def forget_user_preference(key: str) -> str:
        prefs = ctx.preferences
        if prefs is None:
            return NOT_CONFIGURED
        if prefs.forget(ctx.user.user_id, key):
            return f"FORGOTTEN: {key}. The default applies from now on."
        return f"NOT_SET: there was no saved preference for {key}."

    return StructuredTool.from_function(
        func=forget_user_preference,
        name="forget_user_preference",
        description=(
            "Forget a saved preference of the user (explicit or learned), when they ask to "
            "stop applying it or to go back to the default."
        ),
        args_schema=args,
    )


register(
    [
        ToolSpec(name="update_user_preference", factory=_update_user_preference, side_effects=True),
        ToolSpec(name="forget_user_preference", factory=_forget_user_preference, side_effects=True),
    ]
)
