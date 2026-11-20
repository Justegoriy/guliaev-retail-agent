import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from retail_agent.observability import logger
from retail_agent.observability.callbacks import MAX_FIELD_CHARS, ObservabilityCallbackHandler

EMAIL = "jane.doe@example.com"


class FailingChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise TimeoutError("model did not answer")


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "agent.jsonl"
    logger.configure(path)
    logger.new_trace_id()

    def read(file=path) -> list[dict]:
        return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]

    return read


def test_llm_call_logs_tokens_and_payload(tmp_path, log):
    payloads = tmp_path / "llm_payloads.jsonl"
    handler = ObservabilityCallbackHandler(payloads)
    reply = AIMessage(
        "Write to " + EMAIL,
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    )
    model = GenericFakeChatModel(messages=iter([reply]), metadata={"purpose": "main"})

    model.invoke([HumanMessage(f"who is {EMAIL}?")], config={"callbacks": [handler]})

    [event] = [r for r in log() if r["event"] == "llm_call"]
    assert event["purpose"] == "main"
    assert (event["input_tokens"], event["output_tokens"]) == (10, 4)
    assert event["latency_ms"] >= 0

    [payload] = log(payloads)
    assert payload["call_id"] == event["call_id"]
    assert payload["trace_id"] == event["trace_id"]
    assert payload["request"][0]["type"] == "human"
    assert EMAIL not in json.dumps(payload)


def test_payloads_are_optional(tmp_path, log):
    model = GenericFakeChatModel(messages=iter([AIMessage("ok")]))
    model.invoke("hi", config={"callbacks": [ObservabilityCallbackHandler(None)]})
    assert [r["event"] for r in log()] == ["llm_call"]
    assert not (tmp_path / "llm_payloads.jsonl").exists()


def test_llm_error_is_classified(log):
    with pytest.raises(TimeoutError):
        FailingChat().invoke("hi", config={"callbacks": [ObservabilityCallbackHandler()]})
    [event] = log()
    assert event["event"] == "llm_error"
    assert event["error"] == "TimeoutError"
    assert event["kind"] == "internal"


def test_tool_call_is_masked_and_clipped(log):
    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        return f"contact {EMAIL} " + "x" * 5000

    lookup.invoke({"query": EMAIL}, config={"callbacks": [ObservabilityCallbackHandler()]})

    [event] = log()
    assert event["event"] == "tool_call"
    assert event["tool"] == "lookup"
    assert event["status"] == "ok"
    assert EMAIL not in event["args"] and EMAIL not in event["result"]
    assert len(event["result"]) < MAX_FIELD_CHARS + 50


def test_tool_error(log):
    @tool
    def broken(query: str) -> str:
        """Always fails."""
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        broken.invoke({"query": "q"}, config={"callbacks": [ObservabilityCallbackHandler()]})
    [event] = log()
    assert (event["event"], event["status"], event["error"]) == (
        "tool_call",
        "error",
        "RuntimeError",
    )
