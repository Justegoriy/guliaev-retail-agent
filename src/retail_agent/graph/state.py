"""Conversation state persisted by the checkpointer between turns and restarts.

The user is not part of the state: it comes from the running session, so a resumed
conversation is always served with the current user's access rights.
"""

from typing import TypedDict

from langgraph.graph import MessagesState


class AnalysisContext(TypedDict):
    """What the conversation is about now; short follow-ups are resolved against it."""

    question: str  # the last analytical question, rephrased to stand alone
    sql: list[str]  # SQL of the last analysis, as written by the model (passed the Guard)
    result_summary: str  # columns and first rows of each result
    period: str | None
    filters: list[str]
    entities: list[str]


class GuardDecision(TypedDict):
    category: str  # analysis | follow_up | data_question | ... | off_topic | malicious
    reason: str
    source: str  # rules | llm | rules_only (guard model unavailable or disabled)
    pii_request: bool  # asks for personal data: allowed through with a prompt note


class AgentState(MessagesState):
    trace_id: str
    blocked: bool
    guard: GuardDecision | None
    analysis_context: AnalysisContext | None
    history_summary: str | None
    session_report_ids: list[str]
