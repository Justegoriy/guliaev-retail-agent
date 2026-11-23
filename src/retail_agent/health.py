"""Startup checks: is the Gemini key usable and can we query BigQuery?

Each check returns a `Check` instead of raising, so the CLI can print a clear hint
instead of a traceback. A fatal check means the agent cannot work at all (missing or
rejected credentials); a non-fatal failure is a temporary outage worth a warning.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google import genai
from google.api_core import exceptions as gexc
from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError, TransportError
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from retail_agent.data.bigquery import BigQueryRunner, clean_error_message
from retail_agent.errors import ServiceUnavailable, SqlInvalid
from retail_agent.observability.logger import log_event
from retail_agent.settings import Settings

GEMINI_CHECK_TIMEOUT_MS = 10_000
ADC_HINT = "run `gcloud auth application-default login`"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    message: str
    fatal: bool = False


def _log(check: Check) -> Check:
    log_event(
        "health_check",
        level="info" if check.ok else ("error" if check.fatal else "warning"),
        check=check.name,
        ok=check.ok,
        fatal=check.fatal,
        message=check.message,
    )
    return check


def _gemini_client(api_key: str) -> Any:
    return genai.Client(
        api_key=api_key, http_options=genai_types.HttpOptions(timeout=GEMINI_CHECK_TIMEOUT_MS)
    )


def check_gemini(
    settings: Settings, client_factory: Callable[[str], Any] = _gemini_client
) -> Check:
    """Ask for the main model's metadata: validates the key and the model name, no quota."""
    model = settings.gemini_model
    if settings.google_api_key is None:
        return _log(
            Check("gemini", False, "GOOGLE_API_KEY is not set; add it to .env.", fatal=True)
        )
    # Keep the client referenced: google-genai closes its HTTP client once it is collected.
    client = client_factory(settings.google_api_key.get_secret_value())
    try:
        client.models.get(model=model)
    except genai_errors.ClientError as exc:
        if exc.code == 404:
            message = f"Gemini model {model!r} was not found; check GEMINI_MODEL in .env."
            return _log(Check("gemini", False, message, fatal=True))
        if exc.code in (400, 401, 403):
            message = f"Gemini rejected GOOGLE_API_KEY ({exc.status or exc.code}); check the key."
            return _log(Check("gemini", False, message, fatal=True))
        message = f"Gemini answered {exc.code} ({exc.status}); requests may fail for a while."
        return _log(Check("gemini", False, message))
    except Exception as exc:
        log_event("gemini_check_failed", level="warning", error=type(exc).__name__, detail=str(exc))
        message = f"Gemini is not reachable right now ({type(exc).__name__}); answers may fail."
        return _log(Check("gemini", False, message))
    return _log(Check("gemini", True, f"Gemini key OK ({model})"))


def check_bigquery(runner: BigQueryRunner, settings: Settings) -> Check:
    """`SELECT 1` dry run: needs credentials and a project to bill, scans nothing."""
    try:
        runner.dry_run("SELECT 1")
    except ServiceUnavailable as exc:
        cause = exc.__cause__
        if isinstance(cause, DefaultCredentialsError):
            message = f"No Google Cloud credentials for BigQuery; {ADC_HINT}."
            return _log(Check("bigquery", False, message, fatal=True))
        if isinstance(cause, TransportError):
            # Could not reach Google's token endpoint: a network problem, not bad credentials.
            message = f"BigQuery is not reachable right now ({type(cause).__name__})."
            return _log(Check("bigquery", False, message))
        if isinstance(cause, GoogleAuthError | gexc.Unauthenticated):
            message = f"Google Cloud credentials are invalid or expired; {ADC_HINT}."
            return _log(Check("bigquery", False, message, fatal=True))
        if isinstance(cause, gexc.Forbidden):
            project = settings.google_cloud_project or "the default project"
            message = (
                f"BigQuery refused access for {project}: {clean_error_message(cause)}. "
                "Check GOOGLE_CLOUD_PROJECT and that the BigQuery API is enabled."
            )
            return _log(Check("bigquery", False, message, fatal=True))
        if isinstance(cause, OSError) and "project" in str(cause).lower():
            message = "No Google Cloud project to run queries in; set GOOGLE_CLOUD_PROJECT in .env."
            return _log(Check("bigquery", False, message, fatal=True))
        message = f"BigQuery is not reachable right now ({exc.detail or 'unknown error'})."
        return _log(Check("bigquery", False, message))
    except SqlInvalid as exc:
        # `SELECT 1` is valid SQL: BigQuery rejecting it means the project is wrong.
        project = settings.google_cloud_project or "(not set)"
        message = (
            f"BigQuery rejected the project {project}: {exc}. Check GOOGLE_CLOUD_PROJECT in .env."
        )
        return _log(Check("bigquery", False, message, fatal=True))
    except Exception as exc:
        message = f"BigQuery check failed ({type(exc).__name__}: {exc})."
        return _log(Check("bigquery", False, message))
    return _log(Check("bigquery", True, f"BigQuery OK ({runner.dataset_id})"))
