from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from google.api_core import exceptions as gexc
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError
from google.genai import errors as genai_errors

from harness import make_settings
from retail_agent import health
from retail_agent.errors import ServiceUnavailable, SqlInvalid


def gemini(error=None):
    def factory(api_key):
        models = MagicMock()
        if error is not None:
            models.get.side_effect = error
        return SimpleNamespace(models=models)

    return factory


def client_error(code, status):
    return genai_errors.ClientError(code, {"error": {"code": code, "status": status}})


def test_gemini_ok(tmp_path):
    check = health.check_gemini(make_settings(tmp_path, google_api_key="k"), gemini())
    assert check.ok and "Gemini key OK" in check.message


def test_gemini_missing_key_is_fatal(tmp_path):
    check = health.check_gemini(make_settings(tmp_path, google_api_key=None), gemini())
    assert check.fatal and "GOOGLE_API_KEY is not set" in check.message


@pytest.mark.parametrize(
    ("error", "fatal", "text"),
    [
        (client_error(400, "INVALID_ARGUMENT"), True, "rejected GOOGLE_API_KEY"),
        (client_error(403, "PERMISSION_DENIED"), True, "rejected GOOGLE_API_KEY"),
        (client_error(404, "NOT_FOUND"), True, "was not found"),
        (client_error(429, "RESOURCE_EXHAUSTED"), False, "429"),
        (httpx.ConnectTimeout("slow"), False, "not reachable"),
    ],
)
def test_gemini_failures(tmp_path, error, fatal, text):
    check = health.check_gemini(make_settings(tmp_path, google_api_key="k"), gemini(error))
    assert not check.ok and check.fatal is fatal and text in check.message


def runner_failing(cause):
    runner = MagicMock()
    runner.dataset_id = "ds"
    error = ServiceUnavailable("bigquery", "detail")
    error.__cause__ = cause
    runner.dry_run.side_effect = error
    return runner


def test_bigquery_ok(tmp_path):
    runner = MagicMock(dataset_id="ds")
    check = health.check_bigquery(runner, make_settings(tmp_path))
    assert check.ok
    runner.dry_run.assert_called_once_with("SELECT 1")


@pytest.mark.parametrize(
    ("cause", "fatal", "text"),
    [
        (DefaultCredentialsError("none"), True, "gcloud auth application-default login"),
        (RefreshError("expired"), True, "invalid or expired"),
        (gexc.Forbidden("BigQuery API has not been used"), True, "GOOGLE_CLOUD_PROJECT"),
        (OSError("Project was not passed and could not be determined"), True, "set GOOGLE_CLOUD"),
        (gexc.ServiceUnavailable("down"), False, "not reachable"),
        (TransportError("token endpoint unreachable"), False, "not reachable"),
    ],
)
def test_bigquery_failures(tmp_path, cause, fatal, text):
    check = health.check_bigquery(runner_failing(cause), make_settings(tmp_path))
    assert not check.ok and check.fatal is fatal and text in check.message


def test_bigquery_rejecting_select_1_means_a_bad_project(tmp_path):
    runner = MagicMock()
    runner.dry_run.side_effect = SqlInvalid("ProjectId must be non-empty")
    settings = make_settings(tmp_path, google_cloud_project="nope")
    check = health.check_bigquery(runner, settings)
    assert check.fatal and "nope" in check.message and "GOOGLE_CLOUD_PROJECT" in check.message
