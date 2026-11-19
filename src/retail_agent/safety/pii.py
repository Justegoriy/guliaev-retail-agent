"""PII policy and masking of free text and query results (D5, D6).

The SQL Guard keeps PII columns out of query output; this module is the second
(results before the LLM) and third (final answer, logs, reports) line of defence.
Masking functions never log the masked values, only where and how many.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from retail_agent.settings import CONFIG_DIR

PII_POLICY_FILE = CONFIG_DIR / "pii_policy.yaml"
REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class PiiPolicy:
    columns: Mapping[str, frozenset[str]]
    patterns: Mapping[str, re.Pattern[str]]

    @property
    def all_columns(self) -> frozenset[str]:
        return frozenset().union(*self.columns.values())


@lru_cache
def load_policy(path: Path = PII_POLICY_FILE) -> PiiPolicy:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    columns = {
        table: frozenset(c.lower() for c in cols)
        for table, cols in (data.get("pii_columns") or {}).items()
    }
    patterns = {name: re.compile(rx) for name, rx in (data.get("patterns") or {}).items()}
    return PiiPolicy(columns=columns, patterns=patterns)


def pii_columns(path: Path = PII_POLICY_FILE) -> Mapping[str, frozenset[str]]:
    """Table name -> PII column names."""
    return load_policy(path).columns


def _report(where: str | None, count: int, **fields: Any) -> None:
    if where is None or count == 0:
        return
    # Imported lazily: the logger itself masks through this module.
    from retail_agent.observability.logger import log_event

    log_event("pii_masked", where=where, count=count, **fields)


def _mask_str(text: str, policy: PiiPolicy) -> tuple[str, int]:
    total = 0
    for pattern in policy.patterns.values():
        text, n = pattern.subn(REDACTED, text)
        total += n
    return text, total


def mask_text(text: str, *, where: str | None = None) -> tuple[str, int]:
    """Replace emails, phone numbers and street addresses in `text`."""
    if not text:
        return text, 0
    masked, count = _mask_str(text, load_policy())
    _report(where, count)
    return masked, count


def _mask_value(value: Any, policy: PiiPolicy) -> tuple[Any, int]:
    if isinstance(value, str):
        return _mask_str(value, policy)
    if isinstance(value, dict):
        total = 0
        out = {}
        for key, item in value.items():
            pii_key = isinstance(key, str) and key.lower() in policy.all_columns
            if pii_key and item is not None:
                out[key], n = REDACTED, 1
            else:
                out[key], n = _mask_value(item, policy)
            total += n
        return out, total
    if isinstance(value, list | tuple):
        masked = [_mask_value(item, policy) for item in value]
        return [m for m, _ in masked], sum(n for _, n in masked)
    return value, 0


def mask_dataframe(df: pd.DataFrame, *, where: str | None = None) -> tuple[pd.DataFrame, int]:
    """Redact PII-named columns entirely and regex-mask the remaining text values."""
    policy = load_policy()
    out = df.copy()
    total = 0
    redacted_columns = []
    for column in out.columns:
        series = out[column]
        if str(column).lower() in policy.all_columns:
            notna = series.notna()
            total += int(notna.sum())
            out[column] = series.astype(object).where(~notna, REDACTED)
            redacted_columns.append(str(column))
        elif pd.api.types.is_string_dtype(series) or series.dtype == object:
            masked = series.map(lambda v: _mask_value(v, policy))
            counts = masked.map(lambda pair: pair[1])
            if counts.any():
                out[column] = masked.map(lambda pair: pair[0])
                total += int(counts.sum())
    _report(where, total, columns=redacted_columns)
    return out, total
