"""Application settings, read from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini
    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    gemini_aux_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"
    llm_rpm: float = Field(5, gt=0)
    llm_timeout_s: float = Field(60, gt=0)
    llm_price_per_1m_input: float = 0.75
    llm_price_per_1m_output: float = 3.75

    # BigQuery
    google_cloud_project: str | None = None
    bq_dataset: str = "bigquery-public-data.thelook_ecommerce"
    bq_max_bytes_billed: int = Field(2_000_000_000, gt=0)
    turn_max_bq_bytes: int = Field(10_000_000_000, gt=0)
    bq_timeout_s: float = Field(60, gt=0)

    # Agent behaviour
    max_sql_failures: int = Field(3, ge=1)
    max_agent_steps: int = Field(15, ge=1)
    result_rows_to_llm: int = Field(50, ge=1)
    sql_default_limit: int = Field(1000, ge=1)
    input_guard_llm: bool = True
    history_keep_turns: int = Field(6, ge=1)
    history_summarize_after_turns: int = Field(10, ge=1)

    # Storage & logs (relative paths are resolved against the project root)
    app_db_path: Path = Path("data/app.db")
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    log_path: Path = Path("logs/agent.jsonl")
    log_llm_payloads: bool = True

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "retail-agent"

    @field_validator("google_api_key", "langsmith_api_key", "google_cloud_project", mode="before")
    @classmethod
    def _empty_to_none(cls, value):
        return value or None

    @field_validator("app_db_path", "checkpoint_db_path", "log_path")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def llm_payload_log_path(self) -> Path:
        return self.log_path.with_name("llm_payloads.jsonl")


@lru_cache
def get_settings() -> Settings:
    return Settings()
