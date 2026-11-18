import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from retail_agent.errors import ServiceUnavailable
from retail_agent.knowledge.schema import TABLES, SchemaCatalog

DATASET = "bigquery-public-data.thelook_ecommerce"

COLUMNS = {
    "orders": ["order_id", "user_id", "status"],
    "order_items": ["id", "order_id", "product_id", "sale_price"],
    "products": ["id", "department", "cost"],
    "users": ["id", "first_name", "email", "age"],
}


def metadata(table):
    return {
        "name": table,
        "num_rows": 100,
        "columns": [
            {"name": c, "type": "STRING", "mode": "NULLABLE", "description": ""}
            for c in COLUMNS[table]
        ],
    }


def make_runner():
    runner = MagicMock()
    runner.dataset_id = DATASET
    runner.get_table_metadata.side_effect = metadata
    return runner


def write_cache(path, age=timedelta(0), dataset=DATASET):
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "fetched_at": (datetime.now(UTC) - age).isoformat(),
                "tables": [metadata(t) for t in TABLES],
            }
        ),
        encoding="utf-8",
    )


def test_load_from_bigquery_writes_cache(tmp_path):
    cache = tmp_path / "schema_cache.json"
    catalog = SchemaCatalog.load(make_runner(), cache_path=cache)
    assert set(catalog.tables) == set(TABLES)
    assert json.loads(cache.read_text(encoding="utf-8"))["dataset"] == DATASET


def test_fresh_cache_skips_bigquery(tmp_path):
    cache = tmp_path / "schema_cache.json"
    write_cache(cache)
    runner = make_runner()
    SchemaCatalog.load(runner, cache_path=cache)
    runner.get_table_metadata.assert_not_called()


def test_stale_cache_is_refreshed(tmp_path):
    cache = tmp_path / "schema_cache.json"
    write_cache(cache, age=timedelta(days=30))
    runner = make_runner()
    SchemaCatalog.load(runner, cache_path=cache)
    assert runner.get_table_metadata.call_count == len(TABLES)


def test_stale_cache_used_when_bigquery_down(tmp_path):
    cache = tmp_path / "schema_cache.json"
    write_cache(cache, age=timedelta(days=30))
    runner = make_runner()
    runner.get_table_metadata.side_effect = ServiceUnavailable("bigquery")
    assert set(SchemaCatalog.load(runner, cache_path=cache).tables) == set(TABLES)


def test_no_cache_and_bigquery_down_raises(tmp_path):
    runner = make_runner()
    runner.get_table_metadata.side_effect = ServiceUnavailable("bigquery")
    with pytest.raises(ServiceUnavailable):
        SchemaCatalog.load(runner, cache_path=tmp_path / "missing.json")


def test_cache_for_other_dataset_is_ignored(tmp_path):
    cache = tmp_path / "schema_cache.json"
    write_cache(cache, dataset="other.dataset")
    runner = make_runner()
    SchemaCatalog.load(runner, cache_path=cache)
    assert runner.get_table_metadata.called


def test_pii_flags_notes_and_rendering(tmp_path):
    catalog = SchemaCatalog.load(make_runner(), cache_path=tmp_path / "c.json")
    users = {c.name: c for c in catalog.tables["users"].columns}
    assert users["email"].pii and users["first_name"].pii
    assert not users["id"].pii and not users["age"].pii

    sale_price = next(c for c in catalog.tables["order_items"].columns if c.name == "sale_price")
    assert "revenue" in sale_price.description

    prompt = catalog.schema_prompt()
    assert "email STRING [PII: never select]" in prompt
    assert "order_items.product_id = products.id" in prompt

    described = catalog.describe()
    assert "Fields: id, age" in described
    assert "Personal data (first_name, email) is never shown." in described
