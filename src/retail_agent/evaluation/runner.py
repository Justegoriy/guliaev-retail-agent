"""Run eval cases through the agent the way the CLI does (`Turn`), then score them.

Three layers: deterministic checks (`checks.py`), execution accuracy against a reference
query, and an LLM judge with a fixed rubric. Confirmations are always answered "no", so a
run never deletes or exports anything.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from retail_agent.auth import UserContext
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.evaluation.cases import Case
from retail_agent.evaluation.checks import TurnRecord, check_turn, found, numbers_in
from retail_agent.graph import memory
from retail_agent.graph.builder import AgentSession
from retail_agent.observability import inspect, logger
from retail_agent.personalization.profile import display
from retail_agent.safety import sql_guard
from retail_agent.turns import Turn

JUDGE_PASS = 3
JUDGE_ATTEMPTS = 2
JUDGE_RETRY_S = 20.0
REPORTS_LIMIT = 1000
EVIDENCE_CHARS = 6000
ANSWER_CHARS = 6000

JUDGE_SYSTEM = """You grade answers of a retail analytics assistant for company executives.
The assistant answers questions about an e-commerce dataset by running SQL in BigQuery.
The user sees only the ASSISTANT text, never the evidence: an answer that refers to data
without stating it ("here is the overview") does not fulfil the intent.
Score each criterion from 1 (bad) to 5 (excellent):
- intent: the final answer does what the user asked (see the expected behaviour); a correct
  refusal counts as fulfilling the intent when refusing is the expected behaviour.
- grounded: numbers and claims come from the query results shown as evidence; nothing is
  invented. If there is no evidence and the answer has no numbers, score 5.
- format: the layout matches the user's saved preferences (if any) and suits the content.
- action_items: concrete, data-backed recommendations; only when required, otherwise null.
- overall: your overall grade.
Be strict and short in the reasoning (at most 3 sentences)."""


class JudgeVerdict(BaseModel):
    intent: int = Field(ge=1, le=5)
    grounded: int = Field(ge=1, le=5)
    format: int = Field(ge=1, le=5)
    action_items: int | None = Field(default=None, ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    reasoning: str


Judge = Callable[[list[BaseMessage]], JudgeVerdict]


@dataclass
class EvalEnv:
    session_factory: Callable[[UserContext], AgentSession]
    runner: BigQueryRunner
    users: Mapping[str, UserContext]
    judge: Judge | None = None
    log_path: Any = None  # the run's agent log, for token usage


@dataclass
class CaseResult:
    id: str
    category: str
    user: str
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    accuracy: dict[str, Any] | None = None
    judge: dict[str, Any] | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _turn_messages(session: AgentSession) -> list[BaseMessage]:
    turns = memory.split_turns(session.state().get("messages") or [])
    return list(turns[-1]) if turns else []


def run_turn(session: AgentSession, text: str) -> TurnRecord:
    started = time.perf_counter()
    turn = Turn(session)
    result = turn.start(text)
    confirmation = False
    while result.status == "pending":
        confirmation = True
        result = turn.resume({})  # unanswered confirmations count as "no"
    messages = _turn_messages(session) if result.status != "error" else []
    executed = session.deps.tool_ctx.turn.executed
    return TurnRecord(
        text=text,
        answer=result.answer if result.outcome else (result.message or ""),
        status=result.status,
        blocked=result.status == "blocked",
        trace_id=result.trace_id,
        tools=[c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls],
        tool_results=[(m.name or "", m.text) for m in messages if isinstance(m, ToolMessage)],
        sql=[q.original_sql or q.sql for q in executed],
        confirmation=confirmation,
        error=result.error_kind,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def reference_values(case: Case, user: UserContext, runner: BigQueryRunner) -> list[float]:
    """The reference numbers, from the reference SQL run under the case user's access."""
    ref = case.reference
    assert ref is not None
    guarded = sql_guard.check_and_rewrite(ref.sql, user)
    df = runner.execute(guarded.sql).df
    return [float(v) for v in df[ref.column].head(ref.rows) if v is not None]


def score_accuracy(case: Case, answer: str, values: list[float]) -> dict[str, Any]:
    ref = case.reference
    assert ref is not None
    numbers = numbers_in(answer)
    matched = [v for v in values if found(v, numbers)]
    share = len(matched) / len(values) if values else 0.0
    return {
        "column": ref.column,
        "reference": values,
        "matched": matched,
        "share": round(share, 3),
        "passed": bool(values) and share >= ref.min_match,
    }


def judge_messages(case: Case, records: list[TurnRecord], profile: str) -> list[BaseMessage]:
    assert case.judge is not None
    conversation = "\n\n".join(
        f"USER: {r.text}\nASSISTANT: {r.answer[:ANSWER_CHARS]}" for r in records
    )
    evidence = "\n\n".join(
        f"[{name}]\n{text}" for r in records for name, text in r.tool_results if text
    )[:EVIDENCE_CHARS]
    body = (
        f"Expected behaviour: {case.judge.intent}\n"
        f"Action items required: {'yes' if case.judge.action_items else 'no'}\n"
        f"User's saved preferences: {profile or 'none'}\n\n"
        f"Conversation:\n{conversation}\n\n"
        f"Query results seen by the assistant (evidence):\n{evidence or '(none)'}"
    )
    return [SystemMessage(JUDGE_SYSTEM), HumanMessage(body)]


def _usage(log_path, trace_ids: set[str]) -> tuple[int, int, int]:
    if log_path is None:
        return 0, 0, 0
    ends = [
        e
        for e in inspect.read_jsonl(log_path)
        if e.get("event") == "turn_end" and e.get("trace_id") in trace_ids
    ]
    return (
        sum(e.get("llm_calls", 0) for e in ends),
        sum(e.get("input_tokens", 0) for e in ends),
        sum(e.get("output_tokens", 0) for e in ends),
    )


def _setup(case: Case, session: AgentSession) -> list[str]:
    for key, value in case.profile.items():
        session.set_preference(key, value, via="eval")
    ids = []
    storage = session.storage
    assert storage is not None or not case.reports
    for r in case.reports:
        owner = r.owner or case.user
        conversation = session.conversation_id if r.this_conversation else f"eval-{case.id}"
        ids.append(storage.save_report(owner, conversation, r.title, r.content).id)
    return ids


def _fill(text: str, report_ids: list[str]) -> str:
    for i, report_id in enumerate(report_ids):
        text = text.replace(f"{{r{i}}}", report_id)
    return text


def run_case(case: Case, env: EvalEnv) -> CaseResult:
    result = CaseResult(case.id, case.category, case.user)
    started = time.perf_counter()
    user = env.users[case.user]
    try:
        logger.user_id_var.set(user.user_id)
        session = env.session_factory(user)
        existing = {r.id for r in session.reports(limit=REPORTS_LIMIT)}  # from earlier cases
        report_ids = _setup(case, session)
        for i, spec in enumerate(case.turns, start=1):
            record = run_turn(session, _fill(spec.text, report_ids))
            result.turns.append(record)
            result.failures += [f"turn {i}: {f}" for f in check_turn(record, spec.expect)]
        last = result.turns[-1]
        result.failures += [f"final: {f}" for f in check_turn(last, case.expect)]
        result.failures += _check_after(case, session, existing)
        if case.reference is not None:
            values = reference_values(case, user, env.runner)
            result.accuracy = score_accuracy(case, last.answer, values)
            if not result.accuracy["passed"]:
                result.failures.append(
                    f"accuracy: {len(result.accuracy['matched'])}/{len(values)} reference "
                    f"numbers of {case.reference.column} found in the answer"
                )
        if case.judge is not None and env.judge is not None:
            profile = ", ".join(f"{p.key}={display(p.value)}" for p in session.profile().items)
            result.judge = _judge(env, case, result, profile)
    except Exception as exc:  # a broken case must not stop the run
        result.error = f"{type(exc).__name__}: {exc}"[:500]
        result.failures.append(f"error: {result.error}")
    trace_ids = {t.trace_id for t in result.turns}
    result.llm_calls, result.input_tokens, result.output_tokens = _usage(env.log_path, trace_ids)
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    result.passed = not result.failures
    return result


def _judge(env: EvalEnv, case: Case, result: CaseResult, profile: str) -> dict[str, Any]:
    """The verdict; a judge outage is recorded, not counted against the agent."""
    assert env.judge is not None and case.judge is not None
    messages = judge_messages(case, result.turns, profile)
    error: Exception | None = None
    for attempt in range(JUDGE_ATTEMPTS):
        try:
            verdict = env.judge(messages)
            break
        except Exception as exc:
            error = exc
            if attempt + 1 < JUDGE_ATTEMPTS:
                time.sleep(JUDGE_RETRY_S)
    else:
        return {"error": f"{type(error).__name__}: {error}"[:300]}
    if verdict.overall < JUDGE_PASS:
        result.failures.append(f"judge: overall {verdict.overall}/5")
    if case.judge.action_items and (verdict.action_items or 0) < JUDGE_PASS:
        result.failures.append(f"judge: action items {verdict.action_items}/5")
    return verdict.model_dump()


def _check_after(case: Case, session: AgentSession, existing: set[str]) -> list[str]:
    failures = []
    profile = session.profile()
    for key, expected in case.profile_after.items():
        item = profile.get(key)
        got = display(item.value) if item else None
        if got != expected:
            failures.append(f"profile {key}={got}, expected {expected}")
    if case.reports_after is not None:
        live = len({r.id for r in session.reports(limit=REPORTS_LIMIT)} - existing)
        if live != case.reports_after:
            failures.append(f"{live} reports left, expected {case.reports_after}")
    return failures
