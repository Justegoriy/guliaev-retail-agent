import httpx
import pytest
from google.api_core import exceptions as gexc
from google.auth.exceptions import DefaultCredentialsError
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleAuthenticationError,
    GoogleModelNotFoundError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
)

from retail_agent.errors import (
    ConfigurationError,
    ErrorKind,
    LlmQuotaExhausted,
    ServiceUnavailable,
    SqlInvalid,
    classify,
    user_message,
)


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (LlmQuotaExhausted("m"), ErrorKind.QUOTA),
        (GoogleRateLimitError("429"), ErrorKind.LLM_RATE_LIMITED),
        (GoogleAPIError(code=503, response_json={"error": {}}), ErrorKind.LLM_UNAVAILABLE),
        (GoogleModelNotFoundError("404"), ErrorKind.LLM_UNAVAILABLE),
        (httpx.ReadTimeout("slow"), ErrorKind.LLM_UNAVAILABLE),
        (httpx.ConnectError("offline"), ErrorKind.LLM_UNAVAILABLE),
        (ServiceUnavailable("llm"), ErrorKind.LLM_UNAVAILABLE),
        (ServiceUnavailable("bigquery", "timeout"), ErrorKind.BQ_UNAVAILABLE),
        (gexc.ServiceUnavailable("down"), ErrorKind.BQ_UNAVAILABLE),
        (GoogleAuthenticationError("401"), ErrorKind.AUTH),
        (GooglePermissionDeniedError("403"), ErrorKind.AUTH),
        (DefaultCredentialsError("no adc"), ErrorKind.AUTH),
        (ConfigurationError("no key"), ErrorKind.AUTH),
        (SqlInvalid("bad"), ErrorKind.INTERNAL),
        (ValueError("bug"), ErrorKind.INTERNAL),
    ],
)
def test_classify(exc, kind):
    assert classify(exc) is kind


@pytest.mark.parametrize("kind", list(ErrorKind))
def test_user_message_is_friendly_and_has_reference(kind):
    text = user_message(kind, "abc123")
    assert text.endswith("(Reference: abc123)")
    for jargon in ("Exception", "429", "503", "traceback"):
        assert jargon not in text
    assert user_message(kind) == text.removesuffix(" (Reference: abc123)")
