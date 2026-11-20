"""Domain errors and their classification into user-facing messages.

`SqlRejected` and `SqlInvalid` are returned to the LLM so it can rewrite the query;
the others end the current action with a user-friendly message.
"""

from enum import StrEnum

import httpx
import httpx2
from google.api_core import exceptions as gexc
from google.auth.exceptions import GoogleAuthError
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)


class AgentError(Exception):
    user_message = "Something went wrong while processing your request."


class SqlRejected(AgentError):
    """The query violates a safety rule (not read-only, forbidden table, PII projection...)."""

    user_message = "This query is not allowed."

    def __init__(self, message: str, rule: str = "policy"):
        self.rule = rule
        super().__init__(message)


class SqlInvalid(AgentError):
    """BigQuery rejected the query (syntax error, unknown column...)."""

    user_message = "The query could not be executed."


class QueryTooExpensive(AgentError):
    def __init__(self, bytes_needed: int | None, limit: int):
        self.bytes_needed = bytes_needed
        self.limit = limit
        needed = f"{bytes_needed / 1e9:.2f} GB" if bytes_needed is not None else "more"
        super().__init__(f"Query would scan {needed}, limit is {limit / 1e9:.2f} GB")

    user_message = (
        "This question needs too much data to answer in one go. "
        "Try narrowing the period or the set of products."
    )


class ServiceUnavailable(AgentError):
    def __init__(self, service: str, detail: str = ""):
        self.service = service
        self.detail = detail
        super().__init__(f"{service} unavailable" + (f": {detail}" if detail else ""))

    @property
    def user_message(self) -> str:
        names = {"bigquery": "The data warehouse", "llm": "The language model"}
        return (
            f"{names.get(self.service, self.service)} is temporarily unavailable. "
            "Please try again in a minute."
        )


class GuardBlocked(AgentError):
    """The request was blocked by the input guard (prompt injection, off-topic, PII request)."""

    user_message = "I can't help with that request."


class ConfigurationError(AgentError):
    """Missing or invalid configuration (API key, model name...)."""

    user_message = "The assistant is not configured correctly."


class LlmQuotaExhausted(AgentError):
    """The model's daily quota is used up; retrying before the reset is pointless."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(f"daily quota exhausted for {model}")

    user_message = "The daily limit of language model requests has been reached."


class ErrorKind(StrEnum):
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_RATE_LIMITED = "llm_rate_limited"
    BQ_UNAVAILABLE = "bq_unavailable"
    QUOTA = "quota"
    AUTH = "auth"
    INTERNAL = "internal"


# google-genai talks to the API through httpx or its fork httpx2 and lets their
# timeouts and connection errors through unwrapped.
HTTP_TRANSPORT_ERRORS = (httpx.TransportError, httpx2.TransportError)

_LLM_UNAVAILABLE_TYPES = (
    ModelAPIError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelNotFoundError,
    *HTTP_TRANSPORT_ERRORS,
)
_AUTH_TYPES = (
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    GoogleAuthError,
    gexc.Unauthenticated,
    ConfigurationError,
)
_BQ_TRANSIENT_TYPES = (
    gexc.ServiceUnavailable,
    gexc.InternalServerError,
    gexc.TooManyRequests,
    gexc.DeadlineExceeded,
)


def classify(exc: BaseException) -> ErrorKind:
    """Map any exception that ends a turn to a coarse kind for logs and user messages."""
    if isinstance(exc, LlmQuotaExhausted):
        return ErrorKind.QUOTA
    if isinstance(exc, ServiceUnavailable):
        return ErrorKind.BQ_UNAVAILABLE if exc.service == "bigquery" else ErrorKind.LLM_UNAVAILABLE
    if isinstance(exc, ModelRateLimitError):
        return ErrorKind.LLM_RATE_LIMITED
    if isinstance(exc, _AUTH_TYPES):
        return ErrorKind.AUTH
    if isinstance(exc, _LLM_UNAVAILABLE_TYPES):
        return ErrorKind.LLM_UNAVAILABLE
    if isinstance(exc, _BQ_TRANSIENT_TYPES):
        return ErrorKind.BQ_UNAVAILABLE
    return ErrorKind.INTERNAL


_USER_MESSAGES = {
    ErrorKind.LLM_UNAVAILABLE: (
        "The language model is temporarily unavailable. Please try again in a minute."
    ),
    ErrorKind.LLM_RATE_LIMITED: (
        "The language model is receiving too many requests right now. "
        "Please wait a minute and try again."
    ),
    ErrorKind.BQ_UNAVAILABLE: (
        "The data warehouse is temporarily unavailable. Please try again in a minute."
    ),
    ErrorKind.QUOTA: (
        "The daily limit of language model requests has been reached. "
        "Please try again tomorrow or ask the administrator to raise the quota."
    ),
    ErrorKind.AUTH: (
        "The assistant cannot access one of its services because of a configuration "
        "or credentials problem. Please contact support."
    ),
    ErrorKind.INTERNAL: "Something went wrong on our side. Please try again.",
}


def user_message(kind: ErrorKind, trace_id: str | None = None) -> str:
    text = _USER_MESSAGES[kind]
    return f"{text} (Reference: {trace_id})" if trace_id else text
