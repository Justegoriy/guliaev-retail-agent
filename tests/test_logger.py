import json

from retail_agent.observability import logger


def test_log_event_writes_json_with_context(tmp_path):
    path = tmp_path / "logs" / "agent.jsonl"
    logger.configure(path)
    logger.user_id_var.set("ceo")
    logger.conversation_id_var.set("conv1")
    trace_id = logger.new_trace_id()

    logger.log_event("session_start", role="ceo", categories=[])

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "session_start"
    assert record["user_id"] == "ceo"
    assert record["conversation_id"] == "conv1"
    assert record["trace_id"] == trace_id
    assert record["role"] == "ceo"


def test_strings_are_masked(tmp_path, monkeypatch):
    monkeypatch.setattr(logger.pii, "mask_text", lambda s: (s.replace("secret", "***"), 1))
    path = tmp_path / "agent.jsonl"
    logger.configure(path)

    logger.log_event("x", text="a secret", nested={"items": ["secret"]})

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["text"] == "a ***"
    assert record["nested"] == {"items": ["***"]}
