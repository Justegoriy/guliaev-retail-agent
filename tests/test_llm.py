import json
import time

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import models as genai_models
from google.genai import types
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from retail_agent import llm
from retail_agent.errors import ConfigurationError, ErrorKind, LlmQuotaExhausted, classify
from retail_agent.observability import logger
from retail_agent.settings import Settings

QUESTION = [HumanMessage("hi")]
DAILY = {
    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
    "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
}
PER_MINUTE = {
    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
    "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}],
}


def retry_info(delay: str) -> dict:
    return {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}


def client_error(code: int, *details: dict) -> genai_errors.ClientError:
    body = {"error": {"code": code, "message": "x", "status": "S", "details": list(details)}}
    return genai_errors.ClientError(code, body)


def server_error(code: int = 503) -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"error": {"code": code, "message": "busy"}})


def ok(text: str = "OK") -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=3, candidates_token_count=1, total_token_count=4
        ),
    )


class FakeGemini:
    """Scripted `Models.generate_content`: per model, a queue of responses or errors."""

    def __init__(self, **script: list) -> None:
        self.script = {model.replace("_", "-"): list(items) for model, items in script.items()}
        self.calls: list[str] = []
        self.configs: list = []

    def __call__(self, *, model: str, contents, config=None):
        name = model.rsplit("/", 1)[-1]
        self.calls.append(name)
        self.configs.append(config)
        outcome = self.script[name].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        google_api_key="test-key",
        gemini_model="main-m",
        gemini_fallback_model="fb-m",
        gemini_aux_model="aux-m",
        llm_rpm=1000,
        llm_aux_rpm=1000,
        llm_retry_budget_s=300,
    )


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    llm.reset_llm_state()
    log_path = tmp_path / "agent.jsonl"
    logger.configure(log_path)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    yield {"log": log_path, "sleeps": sleeps}
    llm.reset_llm_state()


@pytest.fixture
def fake(monkeypatch):
    def install(**script):
        fake = FakeGemini(**script)
        monkeypatch.setattr(
            genai_models.Models, "generate_content", lambda _self, **kwargs: fake(**kwargs)
        )
        return fake

    return install


def events(log_path, name: str) -> list[dict]:
    if not log_path.exists():
        return []
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return [r for r in records if r["event"] == name]


def test_simple_call(settings, fake):
    gemini = fake(main_m=[ok("hello")])
    reply = llm.build_chat_model("main", settings=settings).invoke(QUESTION)
    assert reply.text == "hello"
    assert gemini.calls == ["main-m"]


def test_per_minute_rate_limit_waits_for_retry_delay(settings, fake, isolated):
    gemini = fake(main_m=[client_error(429, PER_MINUTE, retry_info("7s")), ok()])
    llm.build_chat_model("main", settings=settings).invoke(QUESTION)
    assert gemini.calls == ["main-m", "main-m"]
    assert 7 <= isolated["sleeps"][0] <= 8
    assert events(isolated["log"], "llm_retry")[0]["model"] == "main-m"


def test_server_errors_and_timeouts_are_retried(settings, fake):
    gemini = fake(main_m=[server_error(503), httpx.ReadTimeout("slow"), ok()])
    llm.build_chat_model("main", settings=settings).invoke(QUESTION)
    assert gemini.calls == ["main-m"] * 3


def test_daily_quota_switches_to_fallback_for_the_rest_of_the_session(settings, fake, isolated):
    gemini = fake(main_m=[client_error(429, DAILY, retry_info("3600s"))], fb_m=[ok("a"), ok("b")])
    model = llm.build_chat_model("main", settings=settings)

    assert model.invoke(QUESTION).text == "a"
    assert model.invoke(QUESTION).text == "b"

    assert gemini.calls == ["main-m", "fb-m", "fb-m"]
    assert llm.is_exhausted("main-m")
    quota = events(isolated["log"], "quota_exhausted")
    assert len(quota) == 1
    assert quota[0]["quota_id"] == "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    assert isolated["sleeps"] == []


def test_unknown_model_falls_back_without_retries(settings, fake, isolated):
    gemini = fake(main_m=[client_error(404)], fb_m=[ok("from fallback")])
    reply = llm.build_chat_model("main", settings=settings).invoke(QUESTION)
    assert reply.text == "from fallback"
    assert gemini.calls == ["main-m", "fb-m"]
    used = events(isolated["log"], "fallback_used")
    assert used == [used[0]] and used[0]["from_model"] == "main-m"
    assert used[0]["to_model"] == "fb-m"


def test_retry_budget_hands_over_to_fallback(settings, fake):
    settings = settings.model_copy(update={"llm_retry_budget_s": 0})
    gemini = fake(main_m=[server_error()], fb_m=[ok()])
    llm.build_chat_model("main", settings=settings).invoke(QUESTION)
    assert gemini.calls == ["main-m", "fb-m"]


def test_all_attempts_fail(settings, fake):
    gemini = fake(main_m=[server_error()] * 3, fb_m=[server_error()] * 3)
    with pytest.raises(Exception) as info:
        llm.build_chat_model("main", settings=settings).invoke(QUESTION)
    assert gemini.calls == ["main-m"] * 3 + ["fb-m"] * 3
    assert classify(info.value) is ErrorKind.LLM_UNAVAILABLE


def test_overloaded_main_model_is_skipped_during_cooldown(settings, fake, isolated, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(llm, "_clock", lambda: now[0])
    settings = settings.model_copy(update={"llm_primary_cooldown_s": 120})
    gemini = fake(main_m=[server_error(503)] * 3 + [ok("main again")], fb_m=[ok("a"), ok("b")])
    model = llm.build_chat_model("main", settings=settings)

    assert model.invoke(QUESTION).text == "a"
    assert model.invoke(QUESTION).text == "b"
    assert gemini.calls == ["main-m"] * 3 + ["fb-m", "fb-m"]
    [cooldown] = events(isolated["log"], "llm_cooldown")
    assert cooldown["model"] == "main-m" and cooldown["seconds"] == 120
    reasons = [e["reason"] for e in events(isolated["log"], "fallback_used")]
    assert reasons == ["cooldown", "cooldown"]

    now[0] += 121
    assert model.invoke(QUESTION).text == "main again"


def test_non_transient_error_does_not_start_cooldown(settings, fake, isolated):
    fake(main_m=[client_error(404)], fb_m=[ok()])
    llm.build_chat_model("main", settings=settings).invoke(QUESTION)
    assert llm.cooldown_left("main-m") == 0
    assert events(isolated["log"], "llm_cooldown") == []


def test_cooldown_disabled_and_not_used_without_fallback(settings, fake, isolated):
    off = settings.model_copy(update={"llm_primary_cooldown_s": 0})
    fake(main_m=[server_error()] * 3, fb_m=[ok()])
    llm.build_chat_model("main", settings=off).invoke(QUESTION)
    assert llm.cooldown_left("main-m") == 0

    fake(aux_m=[server_error()] * 3)
    with pytest.raises(Exception) as info:
        llm.build_chat_model("guard", settings=settings).invoke(QUESTION)
    assert classify(info.value) is ErrorKind.LLM_UNAVAILABLE
    assert llm.cooldown_left("aux-m") == 0


def test_cooling_down_error_is_not_retried_and_means_unavailable():
    exc = llm.LlmCoolingDown("main-m", 30)
    assert not llm.is_transient(exc)
    assert classify(exc) is ErrorKind.LLM_UNAVAILABLE


def test_both_models_out_of_daily_quota(settings, fake):
    fake(main_m=[client_error(429, DAILY)], fb_m=[client_error(429, DAILY)])
    with pytest.raises(LlmQuotaExhausted) as info:
        llm.build_chat_model("main", settings=settings).invoke(QUESTION)
    assert classify(info.value) is ErrorKind.QUOTA


def test_guard_uses_aux_model_without_fallback(settings, fake):
    gemini = fake(aux_m=[client_error(404)])
    with pytest.raises(Exception) as info:
        llm.build_chat_model("guard", settings=settings).invoke(QUESTION)
    assert gemini.calls == ["aux-m"]
    assert classify(info.value) is ErrorKind.LLM_UNAVAILABLE


def test_tools_are_bound_to_primary_and_fallback(settings, fake):
    @tool
    def run_sql(sql: str) -> str:
        """Run a query."""
        return sql

    gemini = fake(main_m=[client_error(404)], fb_m=[ok()])
    llm.build_chat_model("main", [run_sql], settings=settings).invoke(QUESTION)
    for config in gemini.configs:
        names = [f.name for t in config.tools for f in t.function_declarations]
        assert names == ["run_sql"]


def test_missing_api_key(settings):
    settings = settings.model_copy(update={"google_api_key": None})
    with pytest.raises(ConfigurationError):
        llm.build_chat_model("main", settings=settings)


def test_rate_limit_info_parsing():
    unwrapped = genai_errors.ClientError(429, {"details": [PER_MINUTE, retry_info("12.5s")]})
    info = llm.rate_limit_info(unwrapped)
    assert info.daily is False
    assert info.retry_delay_s == 12.5

    assert llm.rate_limit_info(client_error(429, retry_info("7200s"))).daily is True
    assert llm.rate_limit_info(client_error(429)).retry_delay_s is None


def test_sliding_window_limiter():
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = llm.SlidingWindowRateLimiter(2, clock=lambda: now[0], sleep=sleep)
    assert limiter.acquire() and limiter.acquire()
    assert sleeps == []
    assert limiter.acquire(blocking=False) is False

    now[0] = 10
    limiter.acquire()
    assert sleeps == [50]
