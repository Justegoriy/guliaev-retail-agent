"""PII policy and masking. Text masking is a placeholder until step 3."""

from functools import lru_cache
from pathlib import Path

import yaml

from retail_agent.settings import CONFIG_DIR

PII_POLICY_FILE = CONFIG_DIR / "pii_policy.yaml"


@lru_cache
def pii_columns(path: Path = PII_POLICY_FILE) -> dict[str, frozenset[str]]:
    """Table name -> PII column names."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        table: frozenset(c.lower() for c in columns)
        for table, columns in (data.get("pii_columns") or {}).items()
    }


def mask_text(text: str) -> str:
    return text
