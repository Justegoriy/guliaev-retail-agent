"""Data tools: dataset description and guarded SQL.

Tool results are plain text addressed to the LLM and start with a status word:
`OK`, `ERROR(<kind>)`, `EMPTY` or `STOP`. Failures never raise, so a bad query or an
unavailable warehouse ends in an explanation to the user instead of a crashed turn.
"""

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from retail_agent.errors import QueryTooExpensive, ServiceUnavailable, SqlInvalid, SqlRejected
from retail_agent.observability.logger import log_event
from retail_agent.safety import pii, sql_guard
from retail_agent.tools.registry import ExecutedQuery, ToolContext, ToolSpec, register

STOP_MESSAGE = (
    "STOP: failure limit reached. Explain to the user what went wrong and suggest how to rephrase."
)
SERVICE_DOWN_MESSAGE = "ERROR(service): BigQuery unavailable, do not retry."


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
        return "Your data access: all brands."
    return (
        f"Your data access: brands {user.scope_label}. "
        "Other brands and company-wide figures are not available: every table holds only "
        "the rows of these brands."
    )


def _describe_dataset(ctx: ToolContext) -> BaseTool:
    def describe_dataset() -> str:
        return (
            f"{ctx.schema.describe()}\n\n{access_note(ctx)}\n\n"
            "(The user does not see this text: in your answer, name each table with what "
            "it holds, give a few example questions and state the user's data access.)"
        )

    return StructuredTool.from_function(
        func=describe_dataset,
        name="describe_dataset",
        description=(
            "Describe the available data for the user: tables, fields, relationships, "
            "example questions and which brands the user can access. "
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

        log_event(
            "run_sql",
            purpose=purpose,
            tables=sorted(guarded.tables),
            acl=not ctx.user.has_full_access,
        )
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
        else " Results are automatically restricted to the brands the user can "
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


register(
    [
        ToolSpec(
            name="describe_dataset",
            factory=_describe_dataset,
            capability="Describe the available data: tables, fields and the analyses they allow",
        ),
        ToolSpec(
            name="run_sql",
            factory=_run_sql,
            capability="Answer questions about sales, orders, products and customers with "
            "read-only queries, fixing a failed query by itself",
        ),
    ]
)
