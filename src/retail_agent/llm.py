"""Gemini chat models with rate limiting, retries, daily-quota handling and fallback.

    build_chat_model("main")  = retry(Gemini main).with_fallbacks([retry(Gemini fallback)])
    build_chat_model("guard") = retry(Gemini aux)             # no fallback: guard degrades to rules

Retries cover per-minute 429 (waiting for the server's `retryDelay`), 5xx and network
timeouts, within `LLM_RETRY_BUDGET_S` per model: an overloaded model can hold each request
for almost the whole timeout, and the fallback is the better bet by then.
A 429 for a per-day quota is not retried: the model is marked exhausted for the
rest of the process and later calls fail before sending a request, so the fallback takes
over at once. Quotas are per model, so each model name has its own client-side limiter.
"""

import asyncio
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from google.genai import errors as genai_errors
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import Runnable
from langchain_core.runnables.retry import RunnableRetry
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field
from tenacity import (
    RetryCallState,
    stop_after_attempt,
    stop_before_delay,
    wait_exponential_jitter,
)

from retail_agent.errors import HTTP_TRANSPORT_ERRORS, ConfigurationError, LlmQuotaExhausted
from retail_agent.observability.logger import log_event
from retail_agent.settings import Settings, get_settings

Purpose = Literal["main", "guard"]

RETRY_ATTEMPTS = 3
# Per-minute limits come with retryDelay <= 60s; anything longer is not worth waiting for.
MAX_HONORED_RETRY_DELAY_S = 60.0
# Without a per-day quotaId, a delay this long still means "come back much later".
DAILY_RETRY_DELAY_S = 600.0
_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_backoff = wait_exponential_jitter(initial=1, max=20)


# -- rate limiting ------------------------------------------------------------------


class SlidingWindowRateLimiter(BaseRateLimiter):
    """At most `rpm` requests in any 60-second window, like Gemini's per-minute quotas.

    `InMemoryRateLimiter` is a token bucket that starts empty and spaces every request
    by 60/rpm seconds (12 s at 5 RPM); here a burst of up to `rpm` calls goes out at once.
    """

    def __init__(
        self,
        rpm: float,
        *,
        name: str = "",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_requests = max(1, int(rpm))
        self.window_s = 60.0 * self.max_requests / rpm
        self.name = name
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[float] = deque()
        self._lock = threading.Lock()

    def _reserve(self) -> float:
        """Take a slot and return 0, or return how long to wait for the next one."""
        with self._lock:
            now = self._clock()
            while self._sent and now - self._sent[0] >= self.window_s:
                self._sent.popleft()
            if len(self._sent) < self.max_requests:
                self._sent.append(now)
                return 0.0
            return self._sent[0] + self.window_s - now

    def _log_wait(self, wait: float) -> None:
        if wait >= 1:
            log_event("llm_rate_limit_wait", model=self.name, wait_s=round(wait, 1))

    def acquire(self, *, blocking: bool = True) -> bool:
        while (wait := self._reserve()) > 0:
            if not blocking:
                return False
            self._log_wait(wait)
            self._sleep(wait)
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        while (wait := self._reserve()) > 0:
            if not blocking:
                return False
            self._log_wait(wait)
            await asyncio.sleep(wait)
        return True


_limiters: dict[str, SlidingWindowRateLimiter] = {}
_limiters_lock = threading.Lock()


def _limiter_for(model: str, rpm: float) -> SlidingWindowRateLimiter:
    with _limiters_lock:
        if model not in _limiters:
            _limiters[model] = SlidingWindowRateLimiter(rpm, name=model)
        return _limiters[model]


# -- daily quota ----------------------------------------------------------------------

_exhausted: set[str] = set()
_exhausted_lock = threading.Lock()


def is_exhausted(model: str) -> bool:
    with _exhausted_lock:
        return model in _exhausted


def mark_exhausted(model: str, quota_id: str | None = None) -> None:
    with _exhausted_lock:
        if model in _exhausted:
            return
        _exhausted.add(model)
    log_event("quota_exhausted", level="warning", model=model, quota_id=quota_id)


def reset_llm_state() -> None:
    """Forget exhausted models and rate-limit windows (new process semantics, for tests)."""
    with _exhausted_lock:
        _exhausted.clear()
    with _limiters_lock:
        _limiters.clear()


# -- error inspection -----------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitInfo:
    daily: bool
    retry_delay_s: float | None
    quota_id: str | None


_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")


def _parse_duration(value: Any) -> float | None:
    match = _DURATION.match(value) if isinstance(value, str) else None
    return float(match.group(1)) if match else None


def _api_error(exc: BaseException | None) -> genai_errors.APIError | None:
    """The original google-genai error; LangChain re-raises it as the `__cause__`."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, genai_errors.APIError):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def rate_limit_info(exc: BaseException) -> RateLimitInfo:
    """Read `QuotaFailure` and `RetryInfo` from a 429 response."""
    api = _api_error(exc)
    details = api.details if api is not None else None
    error = details.get("error", details) if isinstance(details, dict) else {}
    quota_ids: list[str] = []
    retry_delay = None
    for item in error.get("details") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("@type", ""))
        if kind.endswith("QuotaFailure"):
            quota_ids += [
                str(v.get("quotaId", ""))
                for v in item.get("violations") or []
                if isinstance(v, dict)
            ]
        elif kind.endswith("RetryInfo"):
            retry_delay = _parse_duration(item.get("retryDelay"))
    daily_ids = [q for q in quota_ids if "perday" in q.lower()]
    daily = bool(daily_ids) or (retry_delay is not None and retry_delay >= DAILY_RETRY_DELAY_S)
    quota_id = (daily_ids or quota_ids or [None])[0]
    return RateLimitInfo(daily=daily, retry_delay_s=retry_delay, quota_id=quota_id)


def is_transient(exc: BaseException) -> bool:
    """Worth retrying the same model: per-minute 429, 5xx, timeouts, network errors."""
    if isinstance(exc, LlmQuotaExhausted):
        return False
    if isinstance(exc, ModelRateLimitError):
        info = rate_limit_info(exc)
        return not info.daily and (info.retry_delay_s or 0) <= MAX_HONORED_RETRY_DELAY_S
    if isinstance(exc, ModelAPIError | ModelConnectionError | ModelTimeoutError):
        return True
    if isinstance(exc, HTTP_TRANSPORT_ERRORS):
        return True
    api = _api_error(exc)
    return api is not None and api.code in _TRANSIENT_STATUS_CODES


def _retry_wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, ModelRateLimitError):
        delay = rate_limit_info(exc).retry_delay_s
        if delay is not None:
            return min(delay, MAX_HONORED_RETRY_DELAY_S) + random.uniform(0, 1)
    return _backoff(state)


# -- model and wrappers ---------------------------------------------------------------


class GeminiChat(ChatGoogleGenerativeAI):
    """Gemini model that skips exhausted quotas and rate-limits only real requests.

    The limiter is applied here rather than through `rate_limiter`, which LangChain
    acquires before `_generate`: calls that fail fast on an exhausted quota must not
    use up rate-limit slots.
    """

    request_limiter: BaseRateLimiter | None = Field(default=None, exclude=True)

    @property
    def model_name(self) -> str:
        return self.model.rsplit("/", 1)[-1]

    def _raise_if_daily(self, exc: ModelRateLimitError) -> None:
        info = rate_limit_info(exc)
        if info.daily:
            mark_exhausted(self.model_name, info.quota_id)
            raise LlmQuotaExhausted(self.model_name) from exc

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if is_exhausted(self.model_name):
            raise LlmQuotaExhausted(self.model_name)
        if self.request_limiter is not None:
            self.request_limiter.acquire(blocking=True)
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except ModelRateLimitError as exc:
            self._raise_if_daily(exc)
            raise

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if is_exhausted(self.model_name):
            raise LlmQuotaExhausted(self.model_name)
        if self.request_limiter is not None:
            await self.request_limiter.aacquire(blocking=True)
        try:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except ModelRateLimitError as exc:
            self._raise_if_daily(exc)
            raise


class LoggedRetry(RunnableRetry):
    """`with_retry` that honors `retryDelay`, logs retries and marks fallback use."""

    model_name: str
    fallback_from: str | None = None
    budget_s: float = 30.0

    @property
    def _kwargs_retrying(self) -> dict[str, Any]:
        kwargs = super()._kwargs_retrying
        kwargs["stop"] = stop_after_attempt(self.max_attempt_number) | stop_before_delay(
            self.budget_s
        )
        kwargs["wait"] = _retry_wait
        kwargs["before_sleep"] = self._log_retry
        return kwargs

    def _log_retry(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log_event(
            "llm_retry",
            level="warning",
            model=self.model_name,
            attempt=state.attempt_number,
            error=type(exc).__name__,
            wait_s=round(state.upcoming_sleep, 1),
        )

    def _log_fallback(self) -> None:
        if self.fallback_from is not None:
            log_event(
                "fallback_used",
                level="warning",
                from_model=self.fallback_from,
                to_model=self.model_name,
            )

    def _invoke(self, input_, run_manager, config, **kwargs):  # type: ignore[override]
        self._log_fallback()
        return super()._invoke(input_, run_manager, config, **kwargs)

    async def _ainvoke(self, input_, run_manager, config, **kwargs):  # type: ignore[override]
        self._log_fallback()
        return await super()._ainvoke(input_, run_manager, config, **kwargs)


def _chat_model(settings: Settings, model: str, purpose: Purpose, role: str) -> GeminiChat:
    if settings.google_api_key is None:
        raise ConfigurationError("GOOGLE_API_KEY is not set; add it to .env")
    rpm = settings.llm_rpm if model == settings.gemini_model else settings.llm_aux_rpm
    return GeminiChat(
        model=model,
        google_api_key=settings.google_api_key,
        timeout=settings.llm_timeout_s,
        max_retries=0,
        disable_streaming=True,
        request_limiter=_limiter_for(model, rpm),
        metadata={"purpose": purpose, "model_role": role},
        tags=[f"llm:{purpose}"],
    )


def _resilient(
    settings: Settings,
    model: GeminiChat,
    tools: Sequence[Any] | None,
    fallback_from: str | None = None,
) -> LoggedRetry:
    bound = model.bind_tools(tools) if tools else model
    return LoggedRetry(
        bound=bound,
        kwargs={},
        config={},
        retry_exception_types=is_transient,
        max_attempt_number=RETRY_ATTEMPTS,
        model_name=model.model_name,
        fallback_from=fallback_from,
        budget_s=settings.llm_retry_budget_s,
    )


def build_chat_model(
    purpose: Purpose = "main",
    tools: Sequence[Any] | None = None,
    *,
    settings: Settings | None = None,
) -> Runnable[LanguageModelInput, AIMessage]:
    """Chat model for the agent ("main") or auxiliary calls ("guard").

    Tools are bound to each model before the retry/fallback wrappers, which do not
    support `bind_tools` themselves.
    """
    settings = settings or get_settings()
    if purpose == "main":
        primary_name, fallback_name = settings.gemini_model, settings.gemini_fallback_model
    else:
        primary_name, fallback_name = settings.gemini_aux_model, None

    primary = _resilient(settings, _chat_model(settings, primary_name, purpose, "primary"), tools)
    if not fallback_name or fallback_name == primary_name:
        return primary
    fallback = _resilient(
        settings,
        _chat_model(settings, fallback_name, purpose, "fallback"),
        tools,
        fallback_from=primary_name,
    )
    return primary.with_fallbacks([fallback])
