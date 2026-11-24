"""Conversation memory: what history the model sees, the analysis context, summaries.

- `prepare_history`: the turns not yet folded into the summary (at most
  `HISTORY_SUMMARIZE_AFTER_TURNS`); tool results older than the last
  `FULL_TOOL_RESULT_TURNS` turns are cut to the status line and the first rows, and
  tool calls without results (an interrupted turn) are dropped, since Gemini rejects them.
- `build_analysis_context`: after a turn with successful `run_sql`, the SQL and result
  summary are taken deterministically, the question/period/filters from one post-turn
  aux-model call (`extraction_prompt`), which also returns the user's preference signals.
- `summarize`: past the threshold the oldest turns are folded into `history_summary`,
  leaving the last `HISTORY_KEEP_TURNS` turns verbatim.
"""

from collections.abc import Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from pydantic import BaseModel, Field

from retail_agent.graph.state import AnalysisContext
from retail_agent.observability.logger import log_event
from retail_agent.personalization.signals import PreferenceSignalOut
from retail_agent.safety import pii
from retail_agent.tools.registry import ExecutedQuery

FULL_TOOL_RESULT_TURNS = 2
COMPACT_RESULT_ROWS = 5
SUMMARY_RESULT_ROWS = 3
MAX_RESULT_SUMMARY_CHARS = 1500
MAX_SUMMARY_CHARS = 3000
MAX_COMPACT_OTHER_CHARS = 600

Turn = list[BaseMessage]


# -- turns ------------------------------------------------------------------------------


def split_turns(messages: Sequence[BaseMessage]) -> list[Turn]:
    """Group messages into turns; each turn starts with a user message."""
    turns: list[Turn] = []
    for message in messages:
        if isinstance(message, HumanMessage) or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def current_turn(messages: Sequence[BaseMessage]) -> Turn:
    turns = split_turns(messages)
    return turns[-1] if turns else []


def last_user_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def final_answer(turn: Turn) -> str:
    for message in reversed(turn):
        if isinstance(message, AIMessage) and not message.tool_calls and message.text:
            return message.text
    return ""


def recent_exchanges(messages: Sequence[BaseMessage], n: int = 3) -> list[tuple[str, str]]:
    """(user, assistant answer) of the last `n` turns in `messages`."""
    return [
        (turn[0].text, final_answer(turn))
        for turn in split_turns(messages)[-n:]
        if isinstance(turn[0], HumanMessage)
    ]


def retrieval_query(messages: Sequence[BaseMessage], context: AnalysisContext | None) -> str:
    """Text for golden-example retrieval: the latest message plus the topic it continues."""
    parts = [last_user_text(messages)]
    if context and context.get("question"):
        parts.append(context["question"])
    return "\n".join(p for p in parts if p)


# -- history for the model ----------------------------------------------------------------


def compact_tool_result(text: str, rows: int = COMPACT_RESULT_ROWS) -> str:
    """Status line, CSV header and the first `rows` rows of a tool result."""
    if not text.startswith("OK"):
        if len(text) <= MAX_COMPACT_OTHER_CHARS:
            return text
        return text[:MAX_COMPACT_OTHER_CHARS] + " ... [cut]"
    blocks = text.split("\n\n")
    head = blocks[0]
    lines = head.splitlines()
    if len(blocks) > 1:
        table = blocks[1].splitlines()
        kept = table[: rows + 1]
        more = len(table) - len(kept)
        lines += kept + ([f"... [{more} more rows not shown]"] if more > 0 else [])
    return "\n".join(lines)


def _without_unanswered_calls(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    called: set[str] = set()
    out: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            ids = {c["id"] for c in message.tool_calls}
            if not ids <= answered:
                if message.text:
                    out.append(AIMessage(content=message.text, id=message.id))
                continue
            called |= ids
        elif isinstance(message, ToolMessage) and message.tool_call_id not in called:
            continue
        out.append(message)
    return out


def prepare_history(
    messages: Sequence[BaseMessage],
    *,
    max_turns: int,
    full_tool_turns: int = FULL_TOOL_RESULT_TURNS,
) -> list[BaseMessage]:
    turns = split_turns(messages)[-max_turns:]
    out: list[BaseMessage] = []
    for i, turn in enumerate(turns):
        old = i < len(turns) - full_tool_turns
        for message in turn:
            if old and isinstance(message, ToolMessage):
                message = message.model_copy(update={"content": compact_tool_result(message.text)})
            out.append(message)
    return _without_unanswered_calls(out)


TOOL_DATA_HEADER = "Data from your queries"


def without_tool_parts(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """The history as plain text, for the last agent step of a turn (run without tools).

    With function calls in the history a model may keep calling functions even when none
    are declared. Assistant messages that only called tools are dropped (any text they had
    is kept); tool results become a user-side block "Data from your queries". Nothing in
    the result looks like a tool call, so the model has no pattern to imitate. Consecutive
    messages of the same side are merged.
    """
    out: list[BaseMessage] = []

    def add(message: BaseMessage) -> None:
        if not message.text.strip():
            return
        if out and type(out[-1]) is type(message):
            merged = f"{out[-1].text}\n\n{message.text}".strip()
            out[-1] = type(message)(content=merged)
        else:
            out.append(message)

    for message in messages:
        if isinstance(message, ToolMessage):
            add(
                HumanMessage(
                    content=f"{TOOL_DATA_HEADER} ({message.name or 'tool'}):\n{message.text}"
                )
            )
        elif isinstance(message, AIMessage):
            add(AIMessage(content=message.text))
        elif isinstance(message, HumanMessage):
            add(HumanMessage(content=message.text))
    return out


# -- analysis context ---------------------------------------------------------------------


class ContextExtraction(BaseModel):
    """One post-turn aux call: the analysis context and the user's preference signals."""

    question: str = Field(
        "",
        description="The user's analytical question, rewritten to be understandable on its "
        "own (resolve 'it', 'the same', 'and for Swim?' using the previous topic).",
    )
    period: str | None = Field(None, description="Analysed period, e.g. '2024, monthly'.")
    filters: list[str] = Field(
        default_factory=list, description="Filters, e.g. 'category = Jeans'."
    )
    entities: list[str] = Field(
        default_factory=list, description="Metrics, categories, brands or segments analysed."
    )
    preference_signals: list[PreferenceSignalOut] = Field(
        default_factory=list, description="Preference signals; usually none."
    )
    conversation_title: str = Field(
        "",
        description="A short title for the conversation, only when asked for one; otherwise empty.",
    )


EXTRACTION_PROMPT = """\
Describe the turn of a conversation with a data analyst assistant that has just ended. \
Use only the information below; never include personal data.

Analysis context: describe the analysis the assistant has just done (question, period, \
filters, entities), for use as context of the next turn. If no queries were run, leave \
these fields empty.
Previous analysis topic: {previous}
User's message: {message}
Queries run (purpose): {purposes}
Beginning of the assistant's answer:
<<<
{answer}
>>>

{preferences}

{title}"""

NO_PREFERENCES = "Preference signals: always return an empty list."
NO_TITLE = "Conversation title: return an empty string."
TITLE_RULES = (
    "Conversation title: this conversation has no title yet. Write one that tells the user, "
    "in a list of their conversations, what it is about: 3 to 6 words in English, Title "
    'Case, naming the topic, metric, segment and period when known (e.g. "Q1 2025 Monthly '
    'Revenue", "Texas vs California Spending", "Churn Spike Analysis", "Available Data '
    'Overview"). No quotes, no final period, no personal data, no words like "Question '
    'about" or "User asks".'
)
MAX_TITLE_CHARS = 60
MIN_TITLE_CHARS = 3


def extraction_prompt(
    turn: Turn,
    executed: Sequence[ExecutedQuery],
    previous: AnalysisContext | None,
    preference_rules: str = "",
    *,
    want_title: bool = False,
) -> str:
    queries = [q for q in executed if q.source == "run_sql"]
    return EXTRACTION_PROMPT.format(
        previous=(previous or {}).get("question") or "none",
        message=turn[0].text if turn else "",
        purposes="; ".join(q.purpose for q in queries) or "none",
        answer=final_answer(turn)[:800],
        preferences=preference_rules or NO_PREFERENCES,
        title=TITLE_RULES if want_title else NO_TITLE,
    )


def clean_title(text: str) -> str | None:
    """A model-written conversation title made safe to show, or None if unusable."""
    title = " ".join(text.split()).strip(" \"'`*#.:;-")
    title = pii.mask_text(title)[0]
    if len(title) < MIN_TITLE_CHARS or pii.REDACTED in title:
        return None
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return title or None


def result_summary(turn: Turn) -> str:
    parts = [
        compact_tool_result(m.text, rows=SUMMARY_RESULT_ROWS)
        for m in turn
        if isinstance(m, ToolMessage) and m.name == "run_sql" and m.text.startswith("OK")
    ]
    text = "\n\n".join(parts)
    if len(text) > MAX_RESULT_SUMMARY_CHARS:
        text = text[:MAX_RESULT_SUMMARY_CHARS] + " ... [cut]"
    return text


def build_analysis_context(
    turn: Turn,
    executed: Sequence[ExecutedQuery],
    extraction: ContextExtraction | None,
) -> AnalysisContext | None:
    """New context after a turn with successful `run_sql`; None means "keep the old one".

    Without an extraction (no aux model, or it failed) the SQL and result summary are
    still kept, with the user's message as the question.
    """
    queries = [q for q in executed if q.source == "run_sql"]
    if not queries:
        return None
    message = turn[0].text if turn else ""
    extraction = extraction or ContextExtraction(question=message)

    def clean(text: str) -> str:
        return pii.mask_text(text)[0]

    context = AnalysisContext(
        question=clean(extraction.question or message),
        sql=[q.original_sql or q.sql for q in queries],
        result_summary=result_summary(turn),
        period=clean(extraction.period) if extraction.period else None,
        filters=[clean(f) for f in extraction.filters],
        entities=[clean(e) for e in extraction.entities],
    )
    log_event(
        "analysis_context_updated",
        queries=len(queries),
        period=context["period"],
        filters=context["filters"],
    )
    return context


# -- summarization ------------------------------------------------------------------------


class HistorySummary(BaseModel):
    summary: str = Field(description="Summary of the conversation so far, at most 15 lines.")


SUMMARY_PROMPT = """\
Summarize the earlier part of a conversation between an executive and a data analyst \
assistant, so the assistant can continue it. Keep: the questions asked, key numbers with \
their periods, conclusions, metric definitions used, saved report ids. Omit greetings and \
failed attempts. Never include personal data. At most 15 short lines.

Previous summary: {previous}

Conversation to add:
{turns}"""


def render_turns(turns: Sequence[Turn]) -> str:
    lines = []
    for turn in turns:
        for message in turn:
            if isinstance(message, HumanMessage):
                lines.append(f"User: {message.text}")
            elif isinstance(message, AIMessage):
                for call in message.tool_calls:
                    args = call["args"]
                    detail = args.get("sql") or ", ".join(f"{k}={v}" for k, v in args.items())
                    lines.append(f"Tool call {call['name']}: {detail}")
                if message.text:
                    lines.append(f"Assistant: {message.text}")
            elif isinstance(message, ToolMessage):
                lines.append(f"Tool result: {compact_tool_result(message.text, rows=3)}")
    return "\n".join(lines)


def summarize(
    messages: Sequence[BaseMessage],
    previous_summary: str | None,
    *,
    keep_turns: int,
    summarize_after: int,
    summarizer: Callable[[str], HistorySummary] | None,
) -> tuple[str, list[RemoveMessage]] | None:
    """Fold the oldest turns into the summary once there are more than `summarize_after`."""
    turns = split_turns(messages)
    if summarizer is None or len(turns) <= summarize_after:
        return None
    old = turns[: len(turns) - keep_turns]
    prompt = SUMMARY_PROMPT.format(previous=previous_summary or "none", turns=render_turns(old))
    try:
        summary = summarizer(prompt).summary.strip()
    except Exception as exc:
        log_event("history_summary_failed", level="warning", error=type(exc).__name__)
        return None
    if not summary:
        return None
    summary = pii.mask_text(summary[:MAX_SUMMARY_CHARS])[0]
    removals = [RemoveMessage(id=m.id) for turn in old for m in turn if m.id]
    log_event("history_summarized", turns=len(old), messages=len(removals), chars=len(summary))
    return summary, removals
