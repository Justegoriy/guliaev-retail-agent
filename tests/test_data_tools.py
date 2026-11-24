import json
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from retail_agent.auth import UserContext
from retail_agent.data.bigquery import QueryResult
from retail_agent.errors import QueryTooExpensive, ServiceUnavailable, SqlInvalid
from retail_agent.knowledge.schema import SchemaCatalog
from retail_agent.observability import logger
from retail_agent.observability.callbacks import ObservabilityCallbackHandler
from retail_agent.settings import Settings
from retail_agent.tools import REGISTRY, ToolContext, ToolRegistry, ToolSpec, build_tools
from retail_agent.tools.data_tools import STOP_MESSAGE, benchmark_sql

DATASET = "bigquery-public-data.thelook_ecommerce"
CEO = UserContext("ceo", "Alice", "ceo")
JEANS = UserContext("diesel", "Carlos", "manager", ("Diesel",))
GOOD_SQL = "SELECT status, COUNT(*) AS n FROM orders GROUP BY status"


def make_settings(**kwargs) -> Settings:
    defaults = dict(
        max_sql_failures=3,
        result_rows_to_llm=5,
        sql_default_limit=1000,
        turn_max_bq_bytes=10_000,
        bq_dataset=DATASET,
    )
    return Settings(_env_file=None, **{**defaults, **kwargs})


def make_schema() -> SchemaCatalog:
    meta = [
        {
            "name": "users",
            "num_rows": 10,
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "email", "type": "STRING"},
            ],
        }
    ]
    return SchemaCatalog.from_metadata(DATASET, meta)


def result(df: pd.DataFrame, bytes_processed: int = 100) -> QueryResult:
    return QueryResult(
        df=df, sql="", bytes_processed=bytes_processed, duration_ms=1, cache_hit=False
    )


def make_ctx(user=CEO, df=None, dry_bytes=100, **settings) -> ToolContext:
    runner = MagicMock()
    runner.dataset_id = DATASET
    runner.dry_run.return_value = dry_bytes
    frame = df if df is not None else pd.DataFrame({"status": ["Complete"], "n": [5]})
    runner.execute.return_value = result(frame, bytes_processed=dry_bytes)
    return ToolContext(
        user=user,
        conversation_id="c1",
        runner=runner,
        schema=make_schema(),
        settings=make_settings(**settings),
    )


def tools_by_name(ctx: ToolContext) -> dict:
    return {t.name: t for t in build_tools(ctx)}


def run_sql(ctx: ToolContext, sql: str = GOOD_SQL) -> str:
    return tools_by_name(ctx)["run_sql"].invoke({"sql": sql, "purpose": "test"})


# -- run_sql -----------------------------------------------------------------------------


def test_success_returns_csv_and_records_guarded_sql():
    ctx = make_ctx()
    out = run_sql(ctx)
    assert out.startswith("OK: rows=1, showing first 1")
    assert "status,n\nComplete,5" in out
    executed = ctx.runner.execute.call_args.args[0]
    assert "`bigquery-public-data`.thelook_ecommerce.orders" in executed
    assert "LIMIT 1000" in executed
    assert ctx.turn.executed_sql == [executed]
    assert ctx.turn.executed[0].purpose == "test"
    assert ctx.turn.sql_failures == 0


def test_restricted_user_query_is_acl_filtered():
    ctx = make_ctx(user=JEANS)
    run_sql(ctx, "SELECT COUNT(*) AS n FROM products")
    assert "brand IN ('Diesel')" in ctx.runner.execute.call_args.args[0]


def test_rejected_query_is_not_sent_to_bigquery():
    ctx = make_ctx()
    out = run_sql(ctx, "DELETE FROM orders WHERE TRUE")
    assert out.startswith("ERROR(rejected): Only read-only")
    assert out.endswith("Attempts left: 2.")
    ctx.runner.dry_run.assert_not_called()
    assert ctx.turn.sql_failures == 1


def test_pii_projection_is_rejected():
    out = run_sql(make_ctx(), "SELECT email FROM users")
    assert out.startswith("ERROR(rejected)") and "PII" in out


def test_syntax_error_from_dry_run():
    ctx = make_ctx()
    ctx.runner.dry_run.side_effect = SqlInvalid("Unrecognized name: foo at [1:8]")
    out = run_sql(ctx)
    assert out.startswith("ERROR(syntax): Unrecognized name: foo")
    assert "Attempts left: 2." in out
    ctx.runner.execute.assert_not_called()


def test_runtime_error_from_execute_refunds_budget():
    ctx = make_ctx()
    ctx.runner.execute.side_effect = SqlInvalid("division by zero")
    out = run_sql(ctx)
    assert out.startswith("ERROR(runtime): division by zero")
    assert ctx.turn.bq_bytes == 0
    assert ctx.turn.sql_failures == 1


def test_query_over_per_query_limit_is_too_expensive():
    ctx = make_ctx()
    ctx.runner.dry_run.side_effect = QueryTooExpensive(5_000_000_000, 2_000_000_000)
    out = run_sql(ctx)
    assert out.startswith("ERROR(too_expensive): Query would scan 5.00 GB")
    assert "Add filters or aggregate" in out
    ctx.runner.execute.assert_not_called()


def test_turn_budget_exhausted():
    ctx = make_ctx(dry_bytes=6_000, turn_max_bq_bytes=10_000)
    assert run_sql(ctx).startswith("OK")
    assert ctx.turn.bq_bytes == 6_000
    out = run_sql(ctx)
    assert out.startswith("ERROR(too_expensive)")
    assert "only 0.00 GB of this turn's data budget remain" in out
    assert ctx.runner.execute.call_count == 1


def test_budget_counts_actual_bytes_not_estimate():
    ctx = make_ctx(dry_bytes=6_000)
    ctx.runner.execute.return_value = result(pd.DataFrame({"n": [1]}), bytes_processed=0)
    run_sql(ctx)
    run_sql(ctx)
    assert ctx.turn.bq_bytes == 0
    assert ctx.runner.execute.call_count == 2


def test_empty_result_counts_as_failure():
    ctx = make_ctx(df=pd.DataFrame({"n": []}))
    out = run_sql(ctx)
    assert out.startswith("EMPTY: 0 rows. Check filter values")
    assert ctx.turn.sql_failures == 1
    assert ctx.turn.executed == []


def test_failure_limit_stops_querying():
    ctx = make_ctx()
    outputs = [run_sql(ctx, "DROP TABLE orders") for _ in range(3)]
    assert "Attempts left: 1." in outputs[1]
    assert "No attempts left" in outputs[2]
    assert run_sql(ctx) == STOP_MESSAGE
    ctx.runner.dry_run.assert_not_called()


def test_new_turn_resets_failures_and_budget():
    ctx = make_ctx(dry_bytes=10_000)
    run_sql(ctx)
    for _ in range(3):
        run_sql(ctx, "DROP TABLE orders")
    ctx.turn.reset()
    assert run_sql(ctx).startswith("OK")


@pytest.mark.parametrize("failing", ["dry_run", "execute"])
def test_service_down_is_reported_without_counting_failure(failing):
    ctx = make_ctx()
    getattr(ctx.runner, failing).side_effect = ServiceUnavailable("bigquery", "timeout")
    out = run_sql(ctx)
    assert out == "ERROR(service): BigQuery unavailable, do not retry."
    assert ctx.turn.sql_failures == 0
    assert ctx.turn.bq_bytes == 0


def test_large_result_is_truncated_with_numeric_summary():
    df = pd.DataFrame({"month": [f"2024-{m:02d}" for m in range(1, 13)], "rev": range(1, 13)})
    out = run_sql(make_ctx(df=df))
    assert out.startswith("OK: rows=12, showing first 5")
    assert "2024-05" in out and "2024-06" not in out
    assert "Numeric columns over all 12 rows:\n- rev: min=1, max=12, mean=6.5, sum=78" in out
    assert "row limit" not in out


def test_result_at_row_limit_is_flagged():
    out = run_sql(make_ctx(df=pd.DataFrame({"id": range(10)}), sql_default_limit=10))
    assert "the result hit the row limit (10)" in out


def test_pii_in_result_values_is_masked():
    df = pd.DataFrame({"note": ["contact john.doe@example.com"], "n": [1]})
    out = run_sql(make_ctx(df=df), "SELECT 'x' AS note, 1 AS n")
    assert "john.doe@example.com" not in out
    assert "[REDACTED]" in out


# -- describe_dataset --------------------------------------------------------------------


def test_describe_dataset_hides_pii_and_shows_access():
    full = tools_by_name(make_ctx())["describe_dataset"].invoke({})
    assert "Personal data (email) is never shown." in full
    assert "all brands" in full
    restricted = tools_by_name(make_ctx(user=JEANS))["describe_dataset"].invoke({})
    assert "brands Diesel" in restricted
    assert "company-wide totals" in restricted


# -- get_company_benchmark ---------------------------------------------------------------


def benchmark(ctx: ToolContext, **args) -> str:
    args = {"metric": "revenue", "start_date": "2024-01-01", "end_date": "2024-12-31", **args}
    return tools_by_name(ctx)["get_company_benchmark"].invoke(args)


def test_benchmark_total_ignores_acl():
    ctx = make_ctx(user=JEANS, df=pd.DataFrame({"revenue": [1234567.891], "orders": [10]}))
    out = benchmark(ctx)
    assert out.startswith("OK: company-wide revenue (all brands, anonymized totals)")
    assert "2024-01-01..2024-12-31" in out
    assert "Definition: SUM(order_items.sale_price)" in out
    assert out.endswith("revenue=1234567.89")
    sql = ctx.runner.execute.call_args.args[0]
    assert "category" not in sql
    assert "DATE '2024-01-01' AND DATE '2024-12-31'" in sql
    assert ctx.turn.executed[0].purpose == "company benchmark: revenue"


def test_benchmark_monthly_returns_only_requested_metric():
    df = pd.DataFrame({"month": ["2024-01", "2024-02"], "revenue": [1.0, 2.0], "orders": [3, 4]})
    out = benchmark(make_ctx(df=df), metric="orders", granularity="month")
    assert out.endswith("month,orders\n2024-01,3\n2024-02,4")
    assert "revenue" not in out.split("\n\n")[-1]


def test_benchmark_without_orders():
    out = benchmark(make_ctx(df=pd.DataFrame({"revenue": [None]})))
    assert out.endswith("No orders in this period.")


def test_benchmark_service_down():
    ctx = make_ctx()
    ctx.runner.dry_run.side_effect = ServiceUnavailable("bigquery")
    assert benchmark(ctx) == "ERROR(service): BigQuery unavailable, do not retry."


@pytest.mark.parametrize(
    "args",
    [
        {"start_date": "2024-02-01", "end_date": "2024-01-01"},
        {"metric": "revenue_by_category"},
        {"start_date": "2010-01-01", "end_date": "2024-01-01"},
        {"granularity": "week"},
        {"end_date": "2024-01-01; DROP TABLE x"},
    ],
)
def test_benchmark_rejects_invalid_arguments(args):
    ctx = make_ctx()
    with pytest.raises(ValidationError):
        benchmark(ctx, **args)
    ctx.runner.execute.assert_not_called()


def test_benchmark_sql_month_grouping():
    sql = benchmark_sql(DATASET, date(2024, 1, 1), date(2024, 3, 31), "month")
    assert "FORMAT_DATE('%Y-%m', DATE(created_at)) AS month" in sql
    assert sql.endswith("GROUP BY month\nORDER BY month")


# -- registry ----------------------------------------------------------------------------


def dummy_factory(name):
    def factory(ctx):
        return StructuredTool.from_function(func=lambda: "ok", name=name, description=name)

    return factory


def test_data_tools_available_to_every_role():
    for user in (CEO, JEANS):
        names = {s.name for s in REGISTRY.specs_for(user)}
        assert {"describe_dataset", "run_sql", "get_company_benchmark"} <= names


def test_registry_filters_by_role_and_sets_flags():
    registry = ToolRegistry()
    registry.register(ToolSpec("everyone", dummy_factory("everyone")))
    registry.register(
        ToolSpec(
            "ceo_only",
            dummy_factory("ceo_only"),
            allowed_roles=frozenset({"ceo"}),
            side_effects=True,
            requires_confirmation=True,
        )
    )
    assert [t.name for t in build_tools(make_ctx(user=JEANS), registry)] == ["everyone"]
    tools = {t.name: t for t in build_tools(make_ctx(user=CEO), registry)}
    assert set(tools) == {"everyone", "ceo_only"}
    assert tools["ceo_only"].metadata == {"side_effects": True, "requires_confirmation": True}
    assert tools["everyone"].metadata["side_effects"] is False


def test_registry_rejects_duplicates_and_misnamed_tools():
    registry = ToolRegistry()
    registry.register(ToolSpec("a", dummy_factory("a")))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ToolSpec("a", dummy_factory("a")))
    registry.register(ToolSpec("b", dummy_factory("not_b")))
    with pytest.raises(ValueError, match="named 'not_b'"):
        build_tools(make_ctx(), registry)


def test_tool_call_log_carries_registry_flags(tmp_path):
    logger.configure(tmp_path / "agent.jsonl")
    try:
        tool = tools_by_name(make_ctx())["run_sql"]
        tool.invoke(
            {"sql": GOOD_SQL, "purpose": "t"},
            config={"callbacks": [ObservabilityCallbackHandler()]},
        )
        events = [
            json.loads(line)
            for line in (tmp_path / "agent.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    finally:
        logger._log_path = None
    call = next(e for e in events if e["event"] == "tool_call")
    assert call["tool"] == "run_sql"
    assert call["side_effects"] is False
    assert call["requires_confirmation"] is False


# -- BigQuery ----------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("granularity", ["total", "month"])
def test_benchmark_sql_runs_on_bigquery(granularity):
    from retail_agent.data.bigquery import BigQueryRunner
    from retail_agent.settings import get_settings

    runner = BigQueryRunner.from_settings(get_settings())
    sql = benchmark_sql(DATASET, date(2024, 1, 1), date(2024, 3, 31), granularity)
    df = runner.execute(sql).df
    assert len(df) == (3 if granularity == "month" else 1)
    row = df.iloc[0]
    assert row["revenue"] > 0 and row["orders"] > 0
    assert row["avg_order_value"] == pytest.approx(row["revenue"] / row["orders"])
    assert 0 < row["return_rate"] < 1
