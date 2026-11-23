from unittest.mock import MagicMock

import pandas as pd
import pytest
from google.api_core import exceptions as gexc
from tenacity import wait_none

from retail_agent.data.bigquery import BigQueryRunner, clean_error_message
from retail_agent.errors import QueryTooExpensive, ServiceUnavailable, SqlInvalid


def make_job(df=None, bytes_processed=1000, cache_hit=False):
    job = MagicMock()
    job.total_bytes_processed = bytes_processed
    job.cache_hit = cache_hit
    job.result.return_value.to_dataframe.return_value = (
        df if df is not None else pd.DataFrame({"n": [1]})
    )
    return job


def make_runner(client, **kwargs):
    kwargs.setdefault("retry_wait", wait_none())
    return BigQueryRunner(project_id="p", client=client, **kwargs)


def bad_request(message, reason="invalidQuery"):
    return gexc.BadRequest(
        f"400 POST https://bigquery.googleapis.com/x: {message}\n\nLocation: US\nJob ID: abc",
        errors=[{"message": message, "reason": reason}],
    )


def test_execute_returns_result_with_job_config():
    client = MagicMock()
    client.query.return_value = make_job(bytes_processed=42, cache_hit=True)
    runner = make_runner(client, max_bytes_billed=123)

    result = runner.execute("SELECT 1")

    assert result.df["n"].tolist() == [1]
    assert result.bytes_processed == 42
    assert result.cache_hit is True
    config = client.query.call_args.kwargs["job_config"]
    assert config.maximum_bytes_billed == 123
    assert config.labels == {"app": "retail_agent"}


def test_api_calls_are_bounded_and_retried_only_by_the_runner():
    """The client's own retry would keep a network outage going for 10 minutes."""
    client = MagicMock()
    client.query.return_value = make_job()
    runner = make_runner(client, timeout_s=7)
    runner.dry_run("SELECT 1")
    runner.execute("SELECT 1")
    for call in client.query.call_args_list:
        assert call.kwargs["retry"] is None
        assert call.kwargs["job_retry"] is None
        assert call.kwargs["timeout"] == 7


def test_retries_transient_errors_then_succeeds():
    client = MagicMock()
    client.query.side_effect = [gexc.ServiceUnavailable("down"), make_job()]
    result = make_runner(client).execute("SELECT 1")
    assert client.query.call_count == 2
    assert len(result.df) == 1


def test_rate_limit_forbidden_is_retried():
    client = MagicMock()
    limited = gexc.Forbidden("slow down", errors=[{"reason": "rateLimitExceeded"}])
    client.query.side_effect = [limited, make_job()]
    make_runner(client).execute("SELECT 1")
    assert client.query.call_count == 2


def test_gives_up_after_attempts():
    client = MagicMock()
    client.query.side_effect = gexc.InternalServerError("boom")
    with pytest.raises(ServiceUnavailable):
        make_runner(client, attempts=3).execute("SELECT 1")
    assert client.query.call_count == 3


def test_bad_request_is_not_retried_and_cleaned():
    client = MagicMock()
    client.query.side_effect = bad_request("Unrecognized name: foo at [1:8]")
    with pytest.raises(SqlInvalid) as info:
        make_runner(client).dry_run("SELECT foo")
    assert client.query.call_count == 1
    assert str(info.value) == "Unrecognized name: foo at [1:8]"


def test_clean_error_message_without_structured_errors():
    exc = gexc.BadRequest("400 POST https://x.y/z: Syntax error at [1:1]\nJob ID: j1")
    assert "https" not in clean_error_message(exc)
    assert "j1" not in clean_error_message(exc)
    assert "Syntax error at [1:1]" in clean_error_message(exc)


def test_bytes_billed_limit_maps_to_too_expensive():
    client = MagicMock()
    client.query.side_effect = bad_request("exceeded limit", reason="bytesBilledLimitExceeded")
    with pytest.raises(QueryTooExpensive):
        make_runner(client).execute("SELECT 1")


def test_dry_run_returns_bytes_and_enforces_limit():
    client = MagicMock()
    client.query.return_value = make_job(bytes_processed=500)
    runner = make_runner(client, max_bytes_billed=1000)
    assert runner.dry_run("SELECT 1") == 500
    assert client.query.call_args.kwargs["job_config"].dry_run is True

    client.query.return_value = make_job(bytes_processed=5000)
    with pytest.raises(QueryTooExpensive):
        runner.dry_run("SELECT 1")


def test_timeout_cancels_job_and_is_not_retried():
    client = MagicMock()
    job = make_job()
    job.result.side_effect = TimeoutError()
    client.query.return_value = job
    with pytest.raises(ServiceUnavailable, match="timed out"):
        make_runner(client).execute("SELECT 1")
    job.cancel.assert_called_once()
    assert client.query.call_count == 1


def test_unexpected_error_maps_to_service_unavailable():
    client = MagicMock()
    client.query.side_effect = gexc.Forbidden("Access Denied", errors=[{"reason": "accessDenied"}])
    with pytest.raises(ServiceUnavailable):
        make_runner(client).execute("SELECT 1")
    assert client.query.call_count == 1


def test_memory_cache_hit_expiry_and_isolation():
    now = [0.0]
    client = MagicMock()
    client.query.side_effect = lambda *a, **k: make_job()
    runner = make_runner(client, cache_ttl_s=60, clock=lambda: now[0])

    first = runner.execute("SELECT 1")
    first.df.loc[0, "n"] = 99
    second = runner.execute(" SELECT 1 ")
    assert client.query.call_count == 1
    assert second.cache_hit is True
    assert second.bytes_processed == 0
    assert second.df["n"].tolist() == [1]

    now[0] = 61
    runner.execute("SELECT 1")
    assert client.query.call_count == 2


def test_cache_size_is_bounded():
    client = MagicMock()
    client.query.side_effect = lambda *a, **k: make_job()
    runner = make_runner(client, cache_size=2)
    for sql in ("SELECT 1", "SELECT 2", "SELECT 3"):
        runner.execute(sql)
    runner.execute("SELECT 1")
    assert client.query.call_count == 4


def test_get_table_metadata():
    field = MagicMock(field_type="STRING", mode="NULLABLE", description=None)
    field.name = "status"
    client = MagicMock()
    client.get_table.return_value = MagicMock(num_rows=10, schema=[field])
    runner = make_runner(client)
    meta = runner.get_table_metadata("orders")
    client.get_table.assert_called_once_with(
        "bigquery-public-data.thelook_ecommerce.orders", retry=None, timeout=runner.timeout_s
    )
    assert meta == {
        "name": "orders",
        "num_rows": 10,
        "columns": [{"name": "status", "type": "STRING", "mode": "NULLABLE", "description": ""}],
    }
