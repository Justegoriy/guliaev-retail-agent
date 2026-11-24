from retail_agent.settings import PROJECT_ROOT, Settings


def test_defaults_and_path_resolution(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    s = Settings(_env_file=None)
    assert s.google_api_key is None
    assert s.bq_dataset == "bigquery-public-data.thelook_ecommerce"
    assert s.log_path == PROJECT_ROOT / "logs" / "agent.jsonl"
    assert s.llm_payload_log_path.name == "llm_payloads.jsonl"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MAX_SQL_FAILURES", "5")
    monkeypatch.setenv("INPUT_GUARD_LLM", "false")
    monkeypatch.setenv("GOOGLE_API_KEY", "key-value-42")
    s = Settings(_env_file=None)
    assert s.max_sql_failures == 5
    assert s.input_guard_llm is False
    assert "key-value-42" not in repr(s)
    assert s.google_api_key.get_secret_value() == "key-value-42"
