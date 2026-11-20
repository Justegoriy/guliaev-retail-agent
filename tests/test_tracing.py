import os
from unittest.mock import MagicMock

import pytest
from langchain_core.tracers.langchain import LangChainTracer

from retail_agent.observability import logger, tracing
from retail_agent.observability.callbacks import ObservabilityCallbackHandler
from retail_agent.settings import Settings


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    logger.configure(tmp_path / "agent.jsonl")
    for name in tracing._ENV_TRACING_FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    yield
    tracing.setup_langsmith(Settings(_env_file=None))


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="key", **kwargs)


def config_for(settings: Settings):
    return tracing.turn_config(settings, trace_id="t1", conversation_id="c1", user_role="ceo")


def test_disabled_by_default_and_env_tracing_switched_off():
    settings = Settings(_env_file=None)
    status = tracing.setup_langsmith(settings)
    assert status == tracing.TracingStatus(enabled=False)
    assert all(os.environ[name] == "false" for name in tracing._ENV_TRACING_FLAGS)
    [handler] = config_for(settings)["callbacks"]
    assert isinstance(handler, ObservabilityCallbackHandler)


def test_missing_key_warns():
    status = tracing.setup_langsmith(Settings(_env_file=None, langsmith_tracing=True))
    assert not status.enabled
    assert "LANGSMITH_API_KEY" in status.warning


def test_unreachable_langsmith_disables_tracing():
    client = MagicMock()
    client.list_projects.side_effect = RuntimeError("401 invalid key")
    settings = make_settings()

    status = tracing.setup_langsmith(settings, client_factory=lambda **kw: client)

    assert not status.enabled
    assert "tracing is off" in status.warning
    assert len(config_for(settings)["callbacks"]) == 1


def test_enabled_tracing_adds_tracer_with_anonymizer():
    client = MagicMock()
    client.list_projects.return_value = iter([])
    factory = MagicMock(return_value=client)
    settings = make_settings(langsmith_project="proj")

    status = tracing.setup_langsmith(settings, client_factory=factory)

    assert status.enabled
    assert factory.call_args.kwargs["anonymizer"] is tracing.anonymize
    config = config_for(settings)
    assert config["run_name"] == "turn"
    assert config["metadata"] == {"trace_id": "t1", "conversation_id": "c1", "user_role": "ceo"}
    assert config["tags"] == ["ceo"]
    tracer = config["callbacks"][1]
    assert isinstance(tracer, LangChainTracer)
    assert tracer.project_name == "proj"


def test_anonymizer_masks_nested_strings():
    masked = tracing.anonymize({"messages": [{"content": "mail jane.doe@example.com"}], "n": 1})
    assert "jane.doe@example.com" not in str(masked)
    assert masked["n"] == 1
