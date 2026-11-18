"""BigQuery access: dry-run cost check, guarded execution, retries and a small result cache.

The runner executes SQL as given; safety rules (read-only, whitelist, PII, ACL) are
enforced earlier by `safety.sql_guard`.
"""

import contextlib
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import pandas as pd
import requests
from google.api_core import exceptions as gexc
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from retail_agent.errors import AgentError, QueryTooExpensive, ServiceUnavailable, SqlInvalid
from retail_agent.observability.logger import log_event
from retail_agent.settings import Settings

T = TypeVar("T")

QUERY_LABELS = {"app": "retail_agent"}

_TRANSIENT_TYPES = (
    gexc.ServiceUnavailable,
    gexc.InternalServerError,
    gexc.TooManyRequests,
    gexc.DeadlineExceeded,
    gexc.BadGateway,
    gexc.GatewayTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionError,
)
# BigQuery reports rate limits as 403 with these reasons.
_TRANSIENT_FORBIDDEN_REASONS = {"rateLimitExceeded", "backendError"}

_ERROR_NOISE = re.compile(r"https?://\S+|Location:\s*\S+|Job ID:\s*\S+|^\d{3}\s+(GET|POST)\b")


@dataclass(frozen=True)
class QueryResult:
    df: pd.DataFrame
    sql: str
    bytes_processed: int
    duration_ms: int
    cache_hit: bool


def _error_reason(exc: BaseException) -> str | None:
    errors = getattr(exc, "errors", None) or []
    if errors and isinstance(errors[0], dict):
        return errors[0].get("reason")
    return None


def clean_error_message(exc: BaseException) -> str:
    """Error text without URLs, job ids and HTTP prefixes - safe to show to the LLM."""
    errors = getattr(exc, "errors", None) or []
    message = None
    if errors and isinstance(errors[0], dict):
        message = errors[0].get("message")
    message = message or getattr(exc, "message", None) or str(exc)
    message = _ERROR_NOISE.sub("", message)
    return " ".join(message.split()).strip(" :")


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    return isinstance(exc, gexc.Forbidden) and _error_reason(exc) in _TRANSIENT_FORBIDDEN_REASONS


class BigQueryRunner:
    def __init__(
        self,
        project_id: str | None = None,
        dataset_id: str = "bigquery-public-data.thelook_ecommerce",
        *,
        max_bytes_billed: int = 2_000_000_000,
        timeout_s: float = 60,
        client: bigquery.Client | None = None,
        attempts: int = 3,
        retry_wait: wait_base | None = None,
        cache_ttl_s: float = 600,
        cache_size: int = 100,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.max_bytes_billed = max_bytes_billed
        self.timeout_s = timeout_s
        self._client = client
        self._attempts = attempts
        self._retry_wait = retry_wait or wait_exponential_jitter(initial=1, max=8)
        self._cache: OrderedDict[str, tuple[float, QueryResult]] = OrderedDict()
        self._cache_ttl_s = cache_ttl_s
        self._cache_size = cache_size
        self._clock = clock

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> "BigQueryRunner":
        return cls(
            project_id=settings.google_cloud_project,
            dataset_id=settings.bq_dataset,
            max_bytes_billed=settings.bq_max_bytes_billed,
            timeout_s=settings.bq_timeout_s,
            **kwargs,
        )

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            try:
                self._client = bigquery.Client(project=self.project_id)
            except DefaultCredentialsError as exc:
                raise ServiceUnavailable(
                    "bigquery",
                    "no Google credentials; run `gcloud auth application-default login`",
                ) from exc
        return self._client

    # -- public API ---------------------------------------------------------------

    def dry_run(self, sql: str) -> int:
        """Validate the query and return the number of bytes it would scan."""
        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = self._call("dry_run", lambda: self.client.query(sql, job_config=config))
        bytes_processed = int(job.total_bytes_processed or 0)
        log_event("sql_dry_run", sql=sql, bytes=bytes_processed)
        if bytes_processed > self.max_bytes_billed:
            raise QueryTooExpensive(bytes_processed, self.max_bytes_billed)
        return bytes_processed

    def execute(self, sql: str) -> QueryResult:
        key = sql.strip()
        cached = self._cache_get(key)
        if cached is not None:
            log_event("sql_executed", sql=sql, rows=len(cached.df), source="memory_cache")
            return replace(cached, df=cached.df.copy(), bytes_processed=0, cache_hit=True)

        config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.max_bytes_billed,
            use_query_cache=True,
            labels=QUERY_LABELS,
        )
        started = time.perf_counter()

        def run() -> tuple[pd.DataFrame, Any]:
            job = self.client.query(sql, job_config=config)
            try:
                rows = job.result(timeout=self.timeout_s)
            except TimeoutError:
                with contextlib.suppress(Exception):
                    job.cancel()
                raise ServiceUnavailable(
                    "bigquery", f"query timed out after {self.timeout_s:.0f}s"
                ) from None
            return rows.to_dataframe(create_bqstorage_client=False), job

        df, job = self._call("execute", run)
        result = QueryResult(
            df=df,
            sql=sql,
            bytes_processed=int(job.total_bytes_processed or 0),
            duration_ms=int((time.perf_counter() - started) * 1000),
            cache_hit=bool(job.cache_hit),
        )
        log_event(
            "sql_executed",
            sql=sql,
            rows=len(df),
            bytes=result.bytes_processed,
            duration_ms=result.duration_ms,
            cache_hit=result.cache_hit,
            source="bigquery",
        )
        self._cache_put(key, result)
        return replace(result, df=df.copy())

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Backwards-compatible helper from the original `bq_client.py`."""
        return self.execute(sql).df

    def get_table_metadata(self, table_name: str) -> dict[str, Any]:
        table_ref = f"{self.dataset_id}.{table_name}"
        table = self._call("get_table", lambda: self.client.get_table(table_ref))
        return {
            "name": table_name,
            "num_rows": int(table.num_rows or 0),
            "columns": [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or "",
                }
                for field in table.schema
            ],
        }

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        return self.get_table_metadata(table_name)["columns"]

    # -- internals ----------------------------------------------------------------

    def _call(self, op: str, fn: Callable[[], T]) -> T:
        def log_retry(state: RetryCallState) -> None:
            exc = state.outcome.exception() if state.outcome else None
            log_event(
                "bq_retry",
                level="warning",
                op=op,
                attempt=state.attempt_number,
                error=type(exc).__name__,
                detail=clean_error_message(exc) if exc else "",
            )

        retrying = Retrying(
            stop=stop_after_attempt(self._attempts),
            wait=self._retry_wait,
            retry=retry_if_exception(_is_transient),
            before_sleep=log_retry,
            reraise=True,
        )
        try:
            return retrying(fn)
        except AgentError:
            raise
        except gexc.BadRequest as exc:
            if _error_reason(exc) == "bytesBilledLimitExceeded":
                raise QueryTooExpensive(None, self.max_bytes_billed) from exc
            raise SqlInvalid(clean_error_message(exc)) from exc
        except gexc.NotFound as exc:
            raise SqlInvalid(clean_error_message(exc)) from exc
        except Exception as exc:
            detail = clean_error_message(exc)
            log_event("bq_error", level="error", op=op, error=type(exc).__name__, detail=detail)
            raise ServiceUnavailable("bigquery", detail) from exc

    def _cache_get(self, key: str) -> QueryResult | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, result = entry
        if self._clock() >= expires_at:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return result

    def _cache_put(self, key: str, result: QueryResult) -> None:
        self._cache[key] = (self._clock() + self._cache_ttl_s, result)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
