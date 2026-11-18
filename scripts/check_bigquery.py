"""Smoke check of BigQuery access: a real query, a dry-run on broken SQL, and schema loading.

Usage: python scripts/check_bigquery.py
"""

from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.errors import AgentError, SqlInvalid
from retail_agent.knowledge.schema import SchemaCatalog
from retail_agent.observability import logger
from retail_agent.settings import get_settings


def main() -> int:
    settings = get_settings()
    logger.configure(settings.log_path)
    runner = BigQueryRunner.from_settings(settings)
    orders = f"`{settings.bq_dataset}.orders`"
    try:
        result = runner.execute(f"SELECT COUNT(*) AS orders FROM {orders}")
        print(f"orders: {int(result.df['orders'][0]):,} ({result.bytes_processed:,} bytes)")

        try:
            runner.dry_run(f"SELECT no_such_column FROM {orders}")
            print("dry-run on broken SQL: unexpectedly passed")
            return 1
        except SqlInvalid as exc:
            print(f"dry-run on broken SQL -> SqlInvalid: {exc}")

        catalog = SchemaCatalog.load(runner, refresh=True)
        for table in catalog.tables.values():
            print(f"{table.name}: {len(table.columns)} columns, ~{table.num_rows:,} rows")
    except AgentError as exc:
        print(f"FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
