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
    llm_aux_rpm: float = Field(10, gt=0)
    llm_timeout_s: float = Field(60, gt=0)
    llm_retry_budget_s: float = Field(30, ge=0)
    llm_primary_cooldown_s: float = Field(300, ge=0)
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
    max_agent_steps: int = Field(5, ge=1)  # graph recursion guard; the budget is below
    max_llm_calls_per_turn: int = Field(5, ge=2)  # guard + agent + post-turn calls
    result_rows_to_llm: int = Field(50, ge=1)
    sql_default_limit: int = Field(1000, ge=1)
    input_guard_llm: bool = True
    history_keep_turns: int = Field(6, ge=1)
    history_summarize_after_turns: int = Field(10, ge=1)

    # Golden Bucket
    golden_top_k: int = Field(3, ge=0)
    golden_min_score: float = Field(0.65, ge=0, le=1)  # cosine similarity of embeddings
    golden_keyword_min_score: float = Field(0.3, ge=0, le=1)  # share of query keywords
    embedding_timeout_s: float = Field(10, gt=0)
    golden_trios_path: Path = Path("data/golden_trios.json")
    golden_index_path: Path = Path("data/golden_index.npz")
    golden_candidates_path: Path = Path("data/golden_candidates.jsonl")

    # User preferences (keys, values, thresholds)
    preferences_config_path: Path = Path("config/preferences.yaml")

    # Storage & logs (relative paths are resolved against the project root)
    app_db_path: Path = Path("data/app.db")
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    exports_dir: Path = Path("exports")
    log_path: Path = Path("logs/agent.jsonl")
    log_llm_payloads: bool = True

    # Authentication (JWT with the user's scope, D4)
    auth_jwt_secret: SecretStr | None = None  # unset: a local secret in dev_jwt_secret_path
    auth_jwt_issuer: str = "retail-agent-dev"
    auth_jwt_audience: str = "retail-agent"
    auth_token_ttl_s: int = Field(8 * 3600, gt=0)
    dev_jwt_secret_path: Path = Path("data/dev_jwt_secret")

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "retail-agent"

    @field_validator(
        "google_api_key",
        "langsmith_api_key",
        "google_cloud_project",
        "auth_jwt_secret",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, value):
        return value or None

    @field_validator(
        "app_db_path",
        "checkpoint_db_path",
        "exports_dir",
        "log_path",
        "golden_trios_path",
        "golden_index_path",
        "golden_candidates_path",
        "preferences_config_path",
        "dev_jwt_secret_path",
    )
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def llm_payload_log_path(self) -> Path:
        return self.log_path.with_name("llm_payloads.jsonl")


@lru_cache
def get_settings() -> Settings:
    return Settings()
