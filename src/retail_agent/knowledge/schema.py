"""Schema of the four whitelisted tables, enriched with business notes and PII flags.

Loaded once per process: from `data/schema_cache.json` when it is fresh, otherwise from
BigQuery (and the cache file is rewritten). A stale cache is still used if BigQuery is down.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.errors import ServiceUnavailable
from retail_agent.observability.logger import log_event
from retail_agent.safety.pii import pii_columns
from retail_agent.settings import PROJECT_ROOT

TABLES = ("orders", "order_items", "products", "users")
SCHEMA_CACHE_FILE = PROJECT_ROOT / "data" / "schema_cache.json"
CACHE_MAX_AGE_S = 7 * 24 * 3600

ORDER_STATUSES = "Processing, Shipped, Complete, Returned, Cancelled"

TABLE_PURPOSE = {
    "orders": "One row per customer order.",
    "order_items": "One row per item sold; the source of revenue.",
    "products": "Product catalogue; `department` (Men/Women) defines data access.",
    "users": "Customers. Contains PII.",
}

COLUMN_NOTES: dict[str, dict[str, str]] = {
    "orders": {
        "order_id": "primary key",
        "user_id": "-> users.id",
        "status": f"order status: {ORDER_STATUSES}",
        "gender": "customer gender, M/F",
        "created_at": "when the order was placed (UTC)",
        "returned_at": "NULL unless returned",
        "shipped_at": "NULL until shipped",
        "delivered_at": "NULL until delivered",
        "num_of_item": "number of items in the order",
    },
    "order_items": {
        "id": "primary key",
        "order_id": "-> orders.order_id",
        "user_id": "-> users.id",
        "product_id": "-> products.id",
        "inventory_item_id": "internal inventory id (inventory table not available)",
        "status": f"item status: {ORDER_STATUSES}",
        "created_at": "when the item was ordered (UTC)",
        "sale_price": "price paid for the item, USD; revenue = SUM(sale_price)",
    },
    "products": {
        "id": "primary key",
        "cost": "unit cost to the company, USD; margin = sale_price - cost",
        "category": "product category (26 values, e.g. Jeans, Swim, Accessories)",
        "name": "product name",
        "brand": "brand name",
        "retail_price": "list price, USD",
        "department": "Men or Women",
        "sku": "stock keeping unit",
        "distribution_center_id": "warehouse id (distribution_centers table not available)",
    },
    "users": {
        "id": "primary key, safe to show as a customer identifier",
        "age": "age in years",
        "gender": "M/F",
        "state": "state / region",
        "city": "city",
        "country": "country",
        "traffic_source": "acquisition channel: Search, Organic, Facebook, Email, Display",
        "created_at": "sign-up time (UTC)",
    },
}

RELATIONSHIPS = (
    "order_items.order_id = orders.order_id",
    "order_items.product_id = products.id",
    "order_items.user_id = users.id",
    "orders.user_id = users.id",
)

EXAMPLE_ANALYSES = (
    "Revenue, orders and average order value by period, category, brand or department",
    "Top products and brands, margins (sale price vs cost)",
    "Return and cancellation rates",
    "Customer segments by age, gender, country and acquisition channel",
    "Repeat customers, cohorts by sign-up month, delivery times",
)


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str
    pii: bool


@dataclass(frozen=True)
class Table:
    name: str
    num_rows: int
    columns: tuple[Column, ...]

    @property
    def purpose(self) -> str:
        return TABLE_PURPOSE.get(self.name, "")


class SchemaCatalog:
    def __init__(self, dataset: str, tables: dict[str, Table]):
        self.dataset = dataset
        self.tables = tables

    @classmethod
    def from_metadata(cls, dataset: str, metadata: list[dict[str, Any]]) -> "SchemaCatalog":
        pii = pii_columns()
        tables = {}
        for meta in metadata:
            name = meta["name"]
            notes = COLUMN_NOTES.get(name, {})
            columns = tuple(
                Column(
                    name=c["name"],
                    type=c["type"],
                    description=c.get("description") or notes.get(c["name"], ""),
                    pii=c["name"].lower() in pii.get(name, frozenset()),
                )
                for c in meta["columns"]
            )
            tables[name] = Table(name=name, num_rows=meta.get("num_rows", 0), columns=columns)
        return cls(dataset, tables)

    @classmethod
    def load(
        cls,
        runner: BigQueryRunner,
        cache_path: Path = SCHEMA_CACHE_FILE,
        max_age_s: float = CACHE_MAX_AGE_S,
        refresh: bool = False,
    ) -> "SchemaCatalog":
        cached = _read_cache(cache_path, runner.dataset_id)
        if cached and not refresh and cached["age_s"] < max_age_s:
            log_event("schema_loaded", source="cache")
            return cls.from_metadata(runner.dataset_id, cached["tables"])
        try:
            metadata = [runner.get_table_metadata(t) for t in TABLES]
        except ServiceUnavailable:
            if cached:
                log_event("schema_loaded", level="warning", source="stale_cache")
                return cls.from_metadata(runner.dataset_id, cached["tables"])
            raise
        _write_cache(cache_path, runner.dataset_id, metadata)
        log_event("schema_loaded", source="bigquery")
        return cls.from_metadata(runner.dataset_id, metadata)

    def column_names(self, table: str) -> set[str]:
        return {c.name for c in self.tables[table].columns}

    def schema_prompt(self) -> str:
        """Compact schema for the system prompt."""
        lines = [f"Dataset `{self.dataset}`. Always use fully qualified table names."]
        for table in self.tables.values():
            lines.append(f"\n{table.name} (~{table.num_rows:,} rows): {table.purpose}")
            for c in table.columns:
                flag = " [PII: never select]" if c.pii else ""
                note = f" - {c.description}" if c.description else ""
                lines.append(f"  {c.name} {c.type}{flag}{note}")
        lines.append("\nJoins: " + "; ".join(RELATIONSHIPS))
        return "\n".join(lines)

    def describe(self) -> str:
        """User-facing description of the available data (markdown)."""
        parts = ["I can analyse four tables of the online store:\n"]
        for table in self.tables.values():
            visible = [c.name for c in table.columns if not c.pii]
            parts.append(f"- **{table.name}** (~{table.num_rows:,} rows) - {table.purpose}")
            parts.append(f"  Fields: {', '.join(visible)}")
            hidden = [c.name for c in table.columns if c.pii]
            if hidden:
                parts.append(f"  Personal data ({', '.join(hidden)}) is never shown.")
        parts.append("\nExamples of what you can ask about:")
        parts.extend(f"- {a}" for a in EXAMPLE_ANALYSES)
        return "\n".join(parts)


def _read_cache(path: Path, dataset: str) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        tables = data["tables"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if data.get("dataset") != dataset or {t.get("name") for t in tables} != set(TABLES):
        return None
    age_s = (datetime.now(UTC) - fetched_at).total_seconds()
    return {"tables": tables, "age_s": age_s}


def _write_cache(path: Path, dataset: str, metadata: list[dict[str, Any]]) -> None:
    payload = {
        "dataset": dataset,
        "fetched_at": datetime.now(UTC).isoformat(),
        "tables": metadata,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        log_event("schema_cache_write_failed", level="warning", error=str(exc))
