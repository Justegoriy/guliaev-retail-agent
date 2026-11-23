import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from harness import CEO, REVENUE_SQL, Harness, calls, sql_call, tool_call
from retail_agent import auth
from retail_agent.data.bigquery import QueryResult
from retail_agent.evaluation.cases import Case, CaseError, Expect, load_cases
from retail_agent.evaluation.checks import (
    TurnRecord,
    answer_format,
    check_turn,
    found,
    numbers_in,
)
from retail_agent.evaluation.runner import EvalEnv, JudgeVerdict, run_case
from retail_agent.observability import logger
from retail_agent.personalization.profile import load_config
from retail_agent.settings import PROJECT_ROOT, get_settings

CASES = PROJECT_ROOT / "evals" / "cases.yaml"


def test_the_shipped_cases_are_valid():
    settings = get_settings()
    cases = load_cases(
        CASES,
        settings.golden_trios_path,
        users=set(auth.load_users()),
        preference_keys=set(load_config(settings.preferences_config_path).keys),
    )
    assert 15 <= len(cases) <= 30
    assert {"security", "analytics", "personalization", "multi_turn", "report"} <= {
        c.category for c in cases
    }
    monthly = next(c for c in cases if c.id == "a2_monthly_revenue")
    assert monthly.reference.sql.lstrip().startswith("SELECT")  # golden: reference resolved


def test_unknown_fields_users_and_trios_are_rejected(tmp_path):
    golden = PROJECT_ROOT / "data" / "golden_trios.json"
    path = tmp_path / "cases.yaml"
    for body, users in [
        ("cases: [{id: x, category: analytics, turns: [q], expect: {formatt: table}}]", None),
        ("cases: [{id: x, category: analytics, user: nobody, turns: [q]}]", {"ceo"}),
        (
            "cases: [{id: x, category: analytics, turns: [q], "
            "reference: {sql: 'golden:nope', column: revenue}}]",
            None,
        ),
        (
            "cases: [{id: x, category: analytics, turns: [q]}, "
            "{id: x, category: analytics, turns: [q]}]",
            None,
        ),
    ]:
        path.write_text(body, encoding="utf-8")
        with pytest.raises(CaseError):
            load_cases(path, golden, users=users)


def test_answer_format():
    table = "Revenue:\n\n| month | revenue |\n| :--- | ---: |\n| 2024-01 | 100 |\n"
    assert answer_format(table) == "table"
    assert answer_format("Top:\n- Jeans: $1\n- Shorts: $2\n") == "bullets"
    assert answer_format("1. Jeans\n2. Shorts\n") == "bullets"
    assert answer_format("Revenue grew 5% to $1.2M.") == "prose"


def test_numbers_are_read_as_written():
    assert numbers_in("$1,205,191.50 and 140K, 1.2M, 12.5% in 2024") == [
        1205191.5,
        140000.0,
        1200000.0,
        12.5,
        2024.0,
    ]
    assert numbers_in("3 months") == [3.0]
    assert found(140099.0, numbers_in("Outerwear: $140,099")) is True
    assert found(1205191.0, numbers_in("about $1.2M")) is True  # 0.4% off
    assert found(1205191.0, numbers_in("about $1.1M")) is False


def record(**kwargs) -> TurnRecord:
    return TurnRecord(text="q", **kwargs)


def test_check_turn_reports_each_failed_expectation():
    rec = record(
        answer="Ask jane@example.com\n\n| a | b |\n|---|---|\n| 1 | 2 |",
        tools=["run_sql"],
        sql=["SELECT 1"],
        tool_results=[("run_sql", "OK 2 rows\n\ncategory,revenue\nSwim,10\nJeans,20")],
    )
    failures = check_turn(
        rec,
        Expect(
            refusal=True,
            tools_called=["save_report"],
            sql_contains=["2024"],
            must_contain_any=["revenue"],
            no_data_about=["Swim"],
            format="bullets",
            max_words=3,
            confirmation=True,
        ),
    )
    text = "\n".join(failures)
    for fragment in (
        "personal data in the answer",
        "refusal=False",
        "save_report",
        "'2024'",
        "none of ['revenue']",
        "'Swim' rows",
        "format table",
        "words",
        "confirmation=False",
    ):
        assert fragment in text
    assert (
        check_turn(record(answer="Fine.", blocked=True), Expect(blocked=True, refusal=True)) == []
    )


def test_status_lines_do_not_count_as_data():
    rec = record(tool_results=[("run_sql", "OK 1 row; Swim is outside your access\n\ncat\nJeans")])
    assert check_turn(rec, Expect(no_data_about=["Swim"])) == []


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    harness.conversation_id = None
    logger.configure(harness.settings.log_path)
    yield harness
    logger._log_path = None


def env_for(h: Harness, verdicts: list[JudgeVerdict]) -> EvalEnv:
    def judge(messages):
        judge.prompts.append(messages[-1].text)
        return verdicts.pop(0)

    judge.prompts = []
    return EvalEnv(h.new_session, h.runner, {"ceo": CEO}, judge, h.settings.log_path)


def verdict(overall: int, action_items: int | None = None) -> JudgeVerdict:
    return JudgeVerdict(
        intent=overall,
        grounded=overall,
        format=overall,
        action_items=action_items,
        overall=overall,
        reasoning="ok",
    )


def test_a_case_runs_through_the_agent_with_all_three_layers(h):
    h.model.script.extend(
        [
            sql_call(REVENUE_SQL),
            AIMessage("| month | revenue |\n|---|---|\n| 2024-01 | $100 |\n| 2024-02 | $100 |"),
        ]
    )
    case = Case.model_validate(
        {
            "id": "c",
            "category": "analytics",
            "profile": {"format": "table"},
            "turns": ["monthly revenue 2024"],
            "expect": {"tools_called": ["run_sql"], "format": "table"},
            "profile_after": {"format": "table"},
            "reference": {"sql": REVENUE_SQL, "column": "revenue", "rows": 2},
            "judge": {"intent": "Monthly revenue."},
        }
    )
    env = env_for(h, [verdict(4)])

    result = run_case(case, env)

    assert result.passed, result.failures
    assert result.accuracy["passed"] and result.accuracy["matched"] == [100.0, 100.0]
    assert result.judge["overall"] == 4
    assert "format=table" in env.judge.prompts[0] and "OK" in env.judge.prompts[0]
    assert result.llm_calls >= 2 and result.turns[0].sql


def test_confirmations_are_answered_no_and_failures_collected(h):
    session_reports = []

    def factory(user):
        session = h.new_session(user)
        session_reports.append(session)
        return session

    case = Case.model_validate(
        {
            "id": "d",
            "category": "deletion",
            "reports": [{"title": "Q1"}],
            "turns": ["delete report {r0}"],
            "expect": {"confirmation": True},
            "reports_after": 1,
            "judge": {"intent": "x", "action_items": True},
        }
    )

    def delete_then_answer(messages):
        report_id = h.storage.list_reports("ceo")[0].id
        return calls(tool_call("request_report_deletion", {"report_ids": [report_id]}, "d1"))

    h.model.script.extend([delete_then_answer, AIMessage("Nothing was deleted.")])
    env = env_for(h, [verdict(2, action_items=1)])
    env.session_factory = factory

    result = run_case(case, env)

    assert result.turns[0].confirmation is True
    assert len(h.storage.list_reports("ceo")) == 1
    assert result.failures == ["judge: overall 2/5", "judge: action items 1/5"]


def test_judge_outage_is_recorded_and_earlier_reports_are_not_counted(h, monkeypatch):
    monkeypatch.setattr("retail_agent.evaluation.runner.JUDGE_RETRY_S", 0)
    h.storage.save_report("ceo", "earlier-case", "Left by another case", "text")
    case = Case.model_validate(
        {
            "id": "j",
            "category": "analytics",
            "turns": ["hi"],
            "reports_after": 0,
            "judge": {"intent": "x"},
        }
    )
    h.model.script.append(AIMessage("Hello."))
    env = env_for(h, [])

    def down(messages):
        raise RuntimeError("503 high demand")

    env.judge = down

    result = run_case(case, env)

    assert result.passed, result.failures
    assert "503" in result.judge["error"]


def test_an_error_in_a_case_does_not_stop_the_run(h):
    case = Case.model_validate(
        {
            "id": "e",
            "category": "analytics",
            "turns": ["q"],
            "reference": {"sql": "SELECT nope FROM", "column": "x"},
        }
    )
    h.model.script.append(AIMessage("Hi."))
    h.runner.execute.return_value = QueryResult(
        df=pd.DataFrame({"x": [1]}), sql="", bytes_processed=0, duration_ms=1, cache_hit=False
    )

    result = run_case(case, env_for(h, []))

    assert not result.passed and result.error and result.failures[-1].startswith("error:")
