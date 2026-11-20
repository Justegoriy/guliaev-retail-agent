"""Data tools: dataset description, guarded SQL and company-wide benchmarks.

Tool results are plain text addressed to the LLM and start with a status word:
`OK`, `ERROR(<kind>)`, `EMPTY` or `STOP`. Failures never raise, so a bad query or an
unavailable warehouse ends in an explanation to the user instead of a crashed turn.
"""

from datetime import date, timedelta
from typing import Literal

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, model_validator

from retail_agent.errors import QueryTooExpensive, ServiceUnavailable, SqlInvalid, SqlRejected
from retail_agent.observability.logger import log_event
from retail_agent.safety import pii, sql_guard
from retail_agent.tools.registry import ExecutedQuery, ToolContext, ToolSpec, register

STOP_MESSAGE = (
    "STOP: failure limit reached. Explain to the user what went wrong and suggest how to rephrase."
)
SERVICE_DOWN_MESSAGE = "ERROR(service): BigQuery unavailable, do not retry."
MAX_BENCHMARK_DAYS = 10 * 366


class BudgetExceeded(Exception):
    def __init__(self, estimate: int, remaining: int):
        self.estimate = estimate
        self.remaining = remaining
        super().__init__(
            f"Query would scan {_gb(estimate)}, but only {_gb(remaining)} of this turn's "
            "data budget remain"
        )


class ExecutionFailed(Exception):
    """The query passed the dry-run but BigQuery failed it at run time (e.g. division by zero)."""


def _gb(n: int) -> str:
    return f"{max(n, 0) / 1e9:.2f} GB"


def _run_within_budget(ctx: ToolContext, sql: str) -> pd.DataFrame:
    """Dry-run, book the bytes against the turn budget, execute and mask PII."""
    estimate = ctx.runner.dry_run(sql)
    budget = ctx.settings.turn_max_bq_bytes
    if not ctx.turn.reserve_bytes(estimate, budget):
        raise BudgetExceeded(estimate, budget - ctx.turn.bq_bytes)
    try:
        result = ctx.runner.execute(sql)
    except SqlInvalid as exc:
        ctx.turn.settle_bytes(estimate, 0)
        raise ExecutionFailed(str(exc)) from exc
    except Exception:
        ctx.turn.settle_bytes(estimate, 0)
        raise
    ctx.turn.settle_bytes(estimate, result.bytes_processed)
    df, _ = pii.mask_dataframe(result.df, where="tool_result")
    return df


# -- formatting --------------------------------------------------------------------------


def _fmt_number(value) -> str:
    if pd.isna(value):
        return "null"
    value = float(value)
    if value.is_integer() and abs(value) < 1e15:
        return f"{int(value)}"
    return f"{value:.2f}" if abs(value) >= 100 else f"{value:.4g}"


def to_csv(df: pd.DataFrame) -> str:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].round(4)
    return out.to_csv(index=False, lineterminator="\n").strip()


def numeric_summary(df: pd.DataFrame) -> str:
    lines = []
    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        stats = ", ".join(
            f"{name}={_fmt_number(getattr(series, name)())}"
            for name in ("min", "max", "mean", "sum")
        )
        lines.append(f"- {column}: {stats}")
    return "\n".join(lines)


# -- describe_dataset --------------------------------------------------------------------


def access_note(ctx: ToolContext) -> str:
    user = ctx.user
    if user.has_full_access:
        return "Your data access: all product categories."
    return (
        f"Your data access: {user.scope_label} (categories: {', '.join(user.categories)}). "
        "For other categories only company-wide totals are available, without any breakdown."
    )


def _describe_dataset(ctx: ToolContext) -> BaseTool:
    def describe_dataset() -> str:
        return f"{ctx.schema.describe()}\n\n{access_note(ctx)}"

    return StructuredTool.from_function(
        func=describe_dataset,
        name="describe_dataset",
        description=(
            "Describe the available data for the user: tables, fields, relationships, "
            "example questions and which product categories the user can access. "
            "Use when the user asks what data or analyses are available."
        ),
    )


# -- run_sql -----------------------------------------------------------------------------


class RunSqlArgs(BaseModel):
    sql: str = Field(description="One BigQuery Standard SQL SELECT statement.")
    purpose: str = Field(
        description="What this query answers, in a few words (used in logs and reports)."
    )


def _failure(ctx: ToolContext, status: str, message: str) -> str:
    failures = ctx.turn.add_failure()
    left = max(ctx.settings.max_sql_failures - failures, 0)
    log_event("sql_attempt_failed", level="warning", status=status, failures=failures)
    hint = (
        f"Attempts left: {left}."
        if left
        else "No attempts left: explain to the user what went wrong and suggest how to rephrase."
    )
    return f"{status}: {message} {hint}"


def format_result(df: pd.DataFrame, *, max_rows: int, row_limit: int) -> str:
    shown = min(len(df), max_rows)
    parts = [f"OK: rows={len(df)}, showing first {shown}", to_csv(df.head(shown))]
    if len(df) > shown:
        summary = numeric_summary(df)
        if summary:
            parts.append(f"Numeric columns over all {len(df)} rows:\n{summary}")
    if len(df) >= row_limit:
        parts.append(
            f"Note: the result hit the row limit ({row_limit}), so it may be incomplete. "
            "Aggregate in SQL instead of summing these rows."
        )
    return "\n\n".join(parts)


def _run_sql(ctx: ToolContext) -> BaseTool:
    settings = ctx.settings

    def run_sql(sql: str, purpose: str) -> str:
        if ctx.turn.sql_failures >= settings.max_sql_failures:
            return STOP_MESSAGE
        try:
            guarded = sql_guard.check_and_rewrite(
                sql,
                ctx.user,
                dataset=ctx.runner.dataset_id,
                default_limit=settings.sql_default_limit,
            )
        except SqlRejected as exc:
            return _failure(ctx, "ERROR(rejected)", str(exc))

        log_event("run_sql", purpose=purpose, tables=sorted(guarded.tables))
        try:
            df = _run_within_budget(ctx, guarded.sql)
        except SqlInvalid as exc:
            return _failure(ctx, "ERROR(syntax)", str(exc))
        except ExecutionFailed as exc:
            return _failure(ctx, "ERROR(runtime)", str(exc))
        except (QueryTooExpensive, BudgetExceeded) as exc:
            return _failure(ctx, "ERROR(too_expensive)", f"{exc}. Add filters or aggregate.")
        except ServiceUnavailable:
            return SERVICE_DOWN_MESSAGE

        if df.empty:
            return _failure(
                ctx,
                "EMPTY",
                "0 rows. Check filter values (case, spelling, date range). "
                "Use SELECT DISTINCT <col> to verify values.",
            )
        ctx.turn.record(
            ExecutedQuery(
                sql=guarded.sql,
                purpose=purpose,
                rows=len(df),
                columns=tuple(str(c) for c in df.columns),
                original_sql=sql.strip(),
            )
        )
        return format_result(
            df, max_rows=settings.result_rows_to_llm, row_limit=settings.sql_default_limit
        )

    scope = (
        ""
        if ctx.user.has_full_access
        else " Results are automatically restricted to the product categories the user can "
        "access; do not try to work around it."
    )
    return StructuredTool.from_function(
        func=run_sql,
        name="run_sql",
        description=(
            "Run one read-only BigQuery Standard SQL SELECT over the store tables "
            f"(`{ctx.runner.dataset_id}.<table>`) and get the result as CSV. Personal data "
            "columns cannot be selected; identify customers by users.id."
            f"{scope} Returns 'OK: ...' with rows, 'ERROR(<kind>): ...' or 'EMPTY: ...' "
            "(fix the query and retry while attempts remain) or 'STOP: ...' (stop querying)."
        ),
        args_schema=RunSqlArgs,
    )


# -- get_company_benchmark ---------------------------------------------------------------

Metric = Literal["revenue", "orders", "avg_order_value", "customers", "items_sold", "return_rate"]

METRIC_DEFINITIONS: dict[str, str] = {
    "revenue": "SUM(order_items.sale_price) of items not Cancelled or Returned, USD",
    "orders": "distinct orders with at least one item not Cancelled or Returned",
    "avg_order_value": "revenue / orders",
    "customers": "distinct customers with at least one item not Cancelled or Returned",
    "items_sold": "number of order items not Cancelled or Returned",
    "return_rate": "Returned items / items not Cancelled",
}

_SOLD = "status NOT IN ('Cancelled', 'Returned')"
_METRIC_SQL: dict[str, str] = {
    "revenue": f"SUM(IF({_SOLD}, sale_price, 0))",
    "orders": f"COUNT(DISTINCT IF({_SOLD}, order_id, NULL))",
    "avg_order_value": (
        f"SAFE_DIVIDE(SUM(IF({_SOLD}, sale_price, 0)), COUNT(DISTINCT IF({_SOLD}, order_id, NULL)))"
    ),
    "customers": f"COUNT(DISTINCT IF({_SOLD}, user_id, NULL))",
    "items_sold": f"COUNTIF({_SOLD})",
    "return_rate": "SAFE_DIVIDE(COUNTIF(status = 'Returned'), COUNTIF(status != 'Cancelled'))",
}


class BenchmarkArgs(BaseModel):
    metric: Metric = Field(description="Company-wide metric to compute.")
    start_date: date = Field(description="First day of the period, YYYY-MM-DD (inclusive).")
    end_date: date = Field(description="Last day of the period, YYYY-MM-DD (inclusive).")
    granularity: Literal["total", "month"] = Field(
        "total", description="'total' for one number, 'month' for a monthly series."
    )

    @model_validator(mode="after")
    def _check_period(self) -> "BenchmarkArgs":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        if self.end_date - self.start_date > timedelta(days=MAX_BENCHMARK_DAYS):
            raise ValueError("the period must not exceed 10 years")
        return self


def benchmark_sql(dataset: str, start: date, end: date, granularity: str) -> str:
    """One fixed query with every metric, so repeated calls hit the result cache."""
    metrics = ",\n  ".join(f"{expr} AS {name}" for name, expr in _METRIC_SQL.items())
    monthly = granularity == "month"
    period = "FORMAT_DATE('%Y-%m', DATE(created_at)) AS month,\n  " if monthly else ""
    group = "\nGROUP BY month\nORDER BY month" if monthly else ""
    return (
        f"SELECT\n  {period}{metrics}\n"
        f"FROM `{dataset}.order_items`\n"
        f"WHERE DATE(created_at) BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'"
        f"{group}"
    )


def _get_company_benchmark(ctx: ToolContext) -> BaseTool:
    def get_company_benchmark(
        metric: Metric,
        start_date: date,
        end_date: date,
        granularity: Literal["total", "month"] = "total",
    ) -> str:
        sql = benchmark_sql(ctx.runner.dataset_id, start_date, end_date, granularity)
        try:
            df = _run_within_budget(ctx, sql)
        except ServiceUnavailable:
            return SERVICE_DOWN_MESSAGE
        except (BudgetExceeded, QueryTooExpensive) as exc:
            return f"ERROR(too_expensive): {exc}. Try a shorter period."
        except (SqlInvalid, ExecutionFailed) as exc:
            log_event("benchmark_failed", level="error", metric=metric, detail=str(exc))
            return "ERROR(internal): the benchmark query failed; do not retry."

        purpose = f"company benchmark: {metric}"
        ctx.turn.record(
            ExecutedQuery(sql=sql, purpose=purpose, rows=len(df), columns=(), source="benchmark")
        )
        header = (
            f"OK: company-wide {metric} (all categories, anonymized totals), "
            f"{start_date.isoformat()}..{end_date.isoformat()}.\n"
            f"Definition: {METRIC_DEFINITIONS[metric]}."
        )
        if granularity == "month":
            if df.empty:
                return f"{header}\nNo orders in this period."
            return f"{header}\n\n{to_csv(df[['month', metric]])}"
        value = df[metric].iloc[0] if not df.empty else None
        if value is None or pd.isna(value):
            return f"{header}\nNo orders in this period."
        return f"{header}\n{metric}={_fmt_number(value)}"

    return StructuredTool.from_function(
        func=get_company_benchmark,
        name="get_company_benchmark",
        description=(
            "Company-wide anonymized totals for benchmarking; no breakdown by category, "
            "product, brand or customer. Covers all categories, including those outside "
            "the user's access. Use to compare the user's categories with the whole "
            "company, or when the user asks about data outside their access."
        ),
        args_schema=BenchmarkArgs,
    )


register(
    [
        ToolSpec(name="describe_dataset", factory=_describe_dataset),
        ToolSpec(name="run_sql", factory=_run_sql),
        ToolSpec(name="get_company_benchmark", factory=_get_company_benchmark),
    ]
)
