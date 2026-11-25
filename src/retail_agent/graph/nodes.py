"""Graph nodes: input_guard -> agent <-> tools -> output_guard -> update_context.

Nodes get the session's services from `AgentDeps` (built once per CLI session), so the
user and their access rights never pass through the model or the persisted state.
"""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolInvocationError
from langgraph.types import Send
from pydantic import BaseModel

from retail_agent.graph import memory
from retail_agent.graph.state import AgentState, AnalysisContext, GuardDecision
from retail_agent.knowledge import golden
from retail_agent.knowledge.golden import CandidateStore, GoldenIndex
from retail_agent.observability.logger import log_event
from retail_agent.personalization import signals
from retail_agent.personalization.profile import UserProfile
from retail_agent.prompts.context import (
    PromptInputs,
    PromptSources,
    SessionReport,
    build_system_prompt,
)
from retail_agent.safety import input_guard, pii
from retail_agent.settings import Settings
from retail_agent.tools.registry import ToolContext

# schema -> runnable that takes a prompt and returns an instance of the schema
AuxFactory = Callable[[type[BaseModel]], Runnable[Any, Any]]

EMPTY_ANSWER = "I could not produce an answer this time. Please rephrase the question."
# The last agent call the turn's LLM budget allows runs without tools and with this note.
FINAL_CALL_NOTE = """

# Last step of this turn
This is your last model call for this question: answer the user now from the tool results \
above, without calling any tools. If some data is still missing, say what is missing and \
suggest a narrower follow-up question."""
BUDGET_ANSWER = (
    "I reached the limit of steps for one question before I could finish the analysis. "
    'Please ask a narrower question, or say "continue" to go on from here.'
)
# Turns that are not about the data get no precedents (saves an embedding call).
NO_PRECEDENTS_CATEGORIES = frozenset(
    {"greeting", "about_assistant", "preferences", "report_management"}
)
# Turns without SQL that still get the post-turn extraction call, because the message may
# reveal a preference ("shorter, please", "as a table?"). Turns with SQL always get it.
SIGNAL_CATEGORIES = frozenset({"follow_up", "preferences"})
# Turns that say nothing about what the conversation is for: no title is made from them.
NO_TITLE_CATEGORIES = frozenset({"greeting"})


@dataclass
class AgentDeps:
    tool_ctx: ToolContext
    tools: list[BaseTool]
    agent_model: Runnable[Any, AIMessage]  # tools already bound
    settings: Settings
    # The model without tools, for the last step of a turn; None: agent_model.
    answer_model: Runnable[Any, AIMessage] | None = None
    aux: AuxFactory | None = None  # None: rules-only guard, no context extraction/summaries
    prompt_sources: PromptSources = field(default_factory=PromptSources)
    today: Callable[[], date] = date.today
    golden: GoldenIndex | None = None
    candidates: CandidateStore | None = None

    def session_reports(self, ids: Sequence[str]) -> list[SessionReport]:
        """Reports of this conversation that still exist, with their titles."""
        storage = self.tool_ctx.storage
        if storage is None:
            return [SessionReport(report_id=i, title="(untitled)") for i in ids]
        reports = storage.get_reports(self.tool_ctx.user.user_id, ids)
        return [SessionReport(report_id=r.id, title=r.title) for r in reports]


def tool_error_message(exc: Exception) -> str:
    """Tool exceptions become results the model can react to, instead of a crashed turn."""
    if isinstance(exc, ToolInvocationError):
        return f"ERROR(invalid_args): {exc.message} Fix the arguments and retry."
    log_event("tool_failed", level="error", error=type(exc).__name__, detail=str(exc)[:500])
    return "ERROR(internal): the tool failed unexpectedly; do not retry it."


def mask_ai_message(message: AIMessage, *, where: str) -> AIMessage:
    content = message.content
    if isinstance(content, str):
        masked, count = pii.mask_text(content, where=where)
        new_content: Any = masked
    else:
        count, new_content = 0, []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text, n = pii.mask_text(block["text"], where=where)
                block, count = {**block, "text": text}, count + n
            new_content.append(block)
    return message.model_copy(update={"content": new_content}) if count else message


class AgentNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self._aux_runnables: dict[type[BaseModel], Runnable[Any, Any]] = {}
        self._aux_lock = threading.Lock()
        # (trace_id, system prompt, ids of the Golden Bucket trios in it)
        self._prompt_cache: tuple[str, str, list[str]] | None = None

    # -- helpers ----------------------------------------------------------------------

    def _aux_call(self, schema: type[BaseModel], config: RunnableConfig | None):
        if self.deps.aux is None:
            return None
        with self._aux_lock:
            if schema not in self._aux_runnables:
                self._aux_runnables[schema] = self.deps.aux(schema)
            runnable = self._aux_runnables[schema]

        def call(prompt: str):
            self.deps.tool_ctx.turn.count_llm_call()
            result = runnable.invoke([HumanMessage(prompt)], config)
            if not isinstance(result, schema):
                raise TypeError(f"aux model returned {type(result).__name__}")
            return result

        return call

    def precedents(self, state: AgentState) -> list[golden.Retrieved]:
        """Golden trios for this turn, found by the message plus the topic it continues."""
        index = self.deps.golden
        category = (state.get("guard") or {}).get("category")
        if index is None or category in NO_PRECEDENTS_CATEGORIES:
            return []
        query = memory.retrieval_query(state["messages"], state.get("analysis_context"))
        try:
            return index.retrieve(query)
        except Exception as exc:
            log_event(
                "golden_degraded", level="warning", stage="retrieve", error=type(exc).__name__
            )
            return []

    def profile(self) -> UserProfile:
        """Read at the start of every turn, so a change applies to the very next answer."""
        prefs = self.deps.tool_ctx.preferences
        if prefs is None:
            return UserProfile()
        try:
            return prefs.get_profile(self.deps.tool_ctx.user.user_id)
        except Exception as exc:
            log_event("preferences_unavailable", level="warning", error=type(exc).__name__)
            return UserProfile()

    def system_prompt(self, state: AgentState) -> str:
        """Built once per turn: persona and glossary are re-read only if their files changed."""
        trace_id = state.get("trace_id") or ""
        if trace_id and self._prompt_cache and self._prompt_cache[0] == trace_id:
            return self._prompt_cache[1]
        deps = self.deps
        persona = deps.prompt_sources.persona.get()
        glossary = deps.prompt_sources.glossary.get()
        guard = state.get("guard") or {}
        report_ids = state.get("session_report_ids") or []
        precedents = self.precedents(state)
        profile = self.profile()
        prefs = deps.tool_ctx.preferences
        inputs = PromptInputs(
            user=deps.tool_ctx.user,
            today=deps.today(),
            schema=deps.tool_ctx.schema.schema_prompt(),
            persona=persona.text,
            glossary=glossary.text,
            preferences=profile.prompt_block(prefs.config) if prefs else "",
            history_summary=state.get("history_summary"),
            analysis_context=state.get("analysis_context"),
            session_reports=deps.session_reports(report_ids) if report_ids else [],
            pii_request=bool(guard.get("pii_request")),
            golden=golden.precedents_block(precedents) if precedents else "",
        )
        prompt = build_system_prompt(inputs)
        log_event(
            "prompt_built",
            persona_version=persona.version,
            glossary_version=glossary.version,
            golden_ids=[p.trio.id for p in precedents],
            preferences=profile.labels(),
            chars=len(prompt),
            has_summary=bool(inputs.history_summary),
            has_analysis_context=bool(inputs.analysis_context),
            pii_note=inputs.pii_request,
        )
        self._prompt_cache = (trace_id, prompt, [p.trio.id for p in precedents])
        return prompt

    def golden_ids(self, state: AgentState) -> list[str]:
        cache = self._prompt_cache
        if cache and cache[0] == (state.get("trace_id") or ""):
            return list(cache[2])
        return []

    # -- nodes ------------------------------------------------------------------------

    def input_guard(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        self.deps.tool_ctx.turn.reset()
        messages = state["messages"]
        text = memory.last_user_text(messages)
        context = state.get("analysis_context")
        classifier = (
            self._aux_call(input_guard.GuardVerdict, config)
            if self.deps.settings.input_guard_llm
            else None
        )
        result = input_guard.check(
            text,
            history=memory.recent_exchanges(messages[:-1]),
            analysis_question=context["question"] if context else None,
            classifier=classifier,
        )
        update: dict[str, Any] = {
            "blocked": result.blocked,
            "guard": GuardDecision(
                category=result.category,
                reason=result.reason,
                source=result.source,
                pii_request=result.pii_request,
            ),
        }
        if result.blocked:
            update["messages"] = [AIMessage(input_guard.refusal(result.category))]
        return update

    def _final_agent_call(self) -> bool:
        """Whether the next agent call is the last one the turn's LLM budget allows.

        One call stays reserved for the post-turn extraction when an aux model is set."""
        turn = self.deps.tool_ctx.turn
        reserve = 1 if self.deps.aux is not None else 0
        return turn.llm_calls + 1 + reserve >= self.deps.settings.max_llm_calls_per_turn

    def agent(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        deps = self.deps
        history = memory.prepare_history(
            state["messages"], max_turns=deps.settings.history_summarize_after_turns
        )
        final = self._final_agent_call()
        system = self.system_prompt(state) + (FINAL_CALL_NOTE if final else "")
        if final and deps.answer_model is not None:
            # No tools declared and no function calls in the history: the model can only answer.
            history = memory.without_tool_parts(history)
        request: list[BaseMessage] = [SystemMessage(system), *history]
        model = (deps.answer_model or deps.agent_model) if final else deps.agent_model
        calls = deps.tool_ctx.turn.count_llm_call()
        if final:
            log_event("llm_budget_final_call", llm_calls=calls)
        response = model.invoke(request, config)
        if final and response.tool_calls:
            log_event("llm_budget_reached", level="warning", dropped_calls=len(response.tool_calls))
            response = AIMessage(content=response.text or BUDGET_ANSWER, id=response.id)
        if not response.tool_calls:
            response = mask_ai_message(response, where="agent_answer")
        return {"messages": [response], "golden_ids": self.golden_ids(state)}

    def output_guard(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {}
        masked = mask_ai_message(last, where="output")
        if not masked.text.strip():
            log_event("empty_answer", level="warning")
            masked = masked.model_copy(update={"content": EMPTY_ANSWER})
        log_event("answer", chars=len(masked.text), blocked=bool(state.get("blocked")))
        return {"messages": [masked]} if masked is not last else {}

    def _needs_title(self, state: AgentState) -> bool:
        """The conversation is still titled by its first message and this turn says what
        it is about (a greeting does not)."""
        ctx = self.deps.tool_ctx
        if ctx.storage is None or ctx.conversation_id is None:
            return False
        if (state.get("guard") or {}).get("category") in NO_TITLE_CATEGORIES:
            return False
        try:
            conversation = ctx.storage.get_conversation(ctx.conversation_id)
        except Exception as exc:
            log_event("conversation_title_failed", level="warning", error=type(exc).__name__)
            return False
        return conversation is not None and not conversation.title_generated

    def _extract(
        self, state: AgentState, config: RunnableConfig, *, want_title: bool
    ) -> memory.ContextExtraction | None:
        """The post-turn aux call: after successful SQL (analysis context), on turns that
        may carry preference signals ("shorter, please" usually runs no query), and once
        per conversation for its title."""
        executed = self.deps.tool_ctx.turn.executed
        has_sql = any(q.source == "run_sql" for q in executed)
        prefs = self.deps.tool_ctx.preferences
        category = (state.get("guard") or {}).get("category")
        signal_turn = prefs is not None and category in SIGNAL_CATEGORIES
        extractor = self._aux_call(memory.ContextExtraction, config)
        if extractor is None or not (has_sql or signal_turn or want_title):
            return None
        rules = ""
        if prefs is not None and (has_sql or signal_turn):
            rules = signals.extraction_rules(prefs.config, self.profile().labels())
        turn = memory.current_turn(state["messages"])
        prompt = memory.extraction_prompt(
            turn, executed, state.get("analysis_context"), rules, want_title=want_title
        )
        try:
            return extractor(prompt)
        except Exception as exc:
            log_event("context_extraction_failed", level="warning", error=type(exc).__name__)
            if prefs is not None:
                log_event("preference_extraction_failed", level="warning", stage="extract")
            return None

    def _learn_preferences(
        self,
        state: AgentState,
        extraction: memory.ContextExtraction | None,
        context: AnalysisContext | None,
    ) -> None:
        prefs = self.deps.tool_ctx.preferences
        if prefs is None:
            return
        found = signals.usage_signals(prefs.config, context)
        if extraction is not None and extraction.preference_signals:
            turns = memory.split_turns(state["messages"])
            previous = memory.final_answer(turns[-2]) if len(turns) > 1 else ""
            found += signals.from_extraction(
                extraction.preference_signals,
                message=memory.last_user_text(state["messages"]),
                previous_answer_chars=len(previous) or None,
            )
        if not found:
            return
        try:
            prefs.add_signals(
                self.deps.tool_ctx.user.user_id,
                found,
                conversation_id=self.deps.tool_ctx.conversation_id,
                trace_id=state.get("trace_id"),
            )
        except Exception as exc:
            log_event(
                "preference_extraction_failed",
                level="warning",
                stage="store",
                error=type(exc).__name__,
            )

    def _set_title(self, raw: str) -> None:
        # An unusable title keeps the first message for good: asking again on every turn
        # would cost an aux call each time. A failed call (None extraction) is retried.
        title = memory.clean_title(raw)
        if title is None:
            log_event("conversation_title_skipped", chars=len(raw))
        ctx = self.deps.tool_ctx
        try:
            ctx.storage.set_conversation_title(ctx.conversation_id, ctx.user.user_id, title)
        except Exception as exc:
            log_event("conversation_title_failed", level="warning", error=type(exc).__name__)
            return
        if title is not None:
            log_event("conversation_titled", title=title)

    def update_context(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = state["messages"]
        settings = self.deps.settings
        update: dict[str, Any] = {}
        want_title = self._needs_title(state)
        extraction = self._extract(state, config, want_title=want_title)
        if want_title and extraction is not None:
            self._set_title(extraction.conversation_title)
        context = memory.build_analysis_context(
            memory.current_turn(messages), self.deps.tool_ctx.turn.executed, extraction
        )
        if context is not None:
            update["analysis_context"] = context
        self._learn_preferences(state, extraction, context)
        # The summary threshold is soft: when this turn used its whole LLM budget, the
        # summary waits for the next turn.
        within_budget = self.deps.tool_ctx.turn.llm_calls < settings.max_llm_calls_per_turn
        folded = memory.summarize(
            messages,
            state.get("history_summary"),
            keep_turns=settings.history_keep_turns,
            summarize_after=settings.history_summarize_after_turns,
            summarizer=self._aux_call(memory.HistorySummary, config) if within_budget else None,
        )
        if folded is not None:
            update["history_summary"], update["messages"] = folded
        return update


# -- routing ---------------------------------------------------------------------------


def route_after_guard(state: AgentState) -> str:
    return "output_guard" if state.get("blocked") else "agent"


def route_after_agent(state: AgentState) -> str | list[Send]:
    """One `tools` task per tool call: when a call pauses for confirmation, the calls that
    already finished in the same step keep their results and are not re-run on resume."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return [Send("tools", [call]) for call in last.tool_calls]
    return "output_guard"


def route_after_output(state: AgentState) -> str:
    return "end" if state.get("blocked") else "update_context"
