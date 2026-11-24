import pytest
import sqlglot
from sqlglot import exp

from retail_agent.auth import UserContext
from retail_agent.errors import SqlRejected
from retail_agent.safety.sql_guard import check_and_rewrite

DATASET = "bigquery-public-data.thelook_ecommerce"
CEO = UserContext("ceo", "CEO", "ceo")


def guard(sql, user=CEO, limit=1000):
    return check_and_rewrite(sql, user, dataset=DATASET, default_limit=limit)


def rejected(sql, rule=None):
    with pytest.raises(SqlRejected) as info:
        guard(sql)
    if rule:
        assert info.value.rule == rule, str(info.value)
    return info.value


# -- read-only ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders WHERE TRUE",
        "DROP TABLE orders",
        "INSERT INTO orders SELECT * FROM orders",
        "UPDATE orders SET status = 'x' WHERE TRUE",
        "MERGE orders t USING orders s ON t.order_id = s.order_id WHEN MATCHED THEN DELETE",
        "CREATE TABLE x AS SELECT 1",
        "TRUNCATE TABLE orders",
        "EXPORT DATA OPTIONS(uri='gs://b/*.csv', format='CSV') AS SELECT * FROM orders",
    ],
)
def test_non_select_statements_are_rejected(sql):
    rejected(sql, "read_only")


def test_multiple_statements_are_rejected():
    rejected("SELECT 1; DROP TABLE x", "single_statement")


def test_unparseable_sql_is_rejected():
    rejected("SELECT FROM WHERE (", "parse")


def test_empty_sql_is_rejected():
    rejected("  ;  ", "parse")


# -- table whitelist ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM thelook_ecommerce.INFORMATION_SCHEMA.TABLES",
        "SELECT * FROM `region-us`.INFORMATION_SCHEMA.JOBS",
        "SELECT * FROM other_dataset.orders",
        "SELECT * FROM `other-project.thelook_ecommerce.orders`",
        "SELECT * FROM distribution_centers",
        "SELECT * FROM `bigquery-public-data.thelook_ecommerce.events`",
    ],
)
def test_tables_outside_whitelist_are_rejected(sql):
    rejected(sql, "table_whitelist")


def test_table_functions_are_rejected():
    rejected("SELECT * FROM ML.PREDICT(MODEL m, TABLE orders)", "table_function")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT status FROM orders",
        "SELECT status FROM thelook_ecommerce.orders",
        "SELECT status FROM `bigquery-public-data.thelook_ecommerce.orders`",
        "SELECT status FROM `bigquery-public-data`.thelook_ecommerce.orders",
    ],
)
def test_whitelisted_tables_with_any_valid_prefix(sql):
    assert guard(sql).tables == {"orders"}


def test_cte_names_are_not_tables():
    result = guard(
        "WITH orders_cte AS (SELECT order_id, status FROM orders) "
        "SELECT status, COUNT(*) AS n FROM orders_cte GROUP BY status"
    )
    assert result.tables == {"orders"}
    assert "FROM orders_cte" in result.sql


def test_cte_shadowing_a_table_is_rejected():
    rejected("WITH users AS (SELECT 'a@b.c' AS email) SELECT email FROM users", "cte_name")


# -- PII projections ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT email FROM users",
        "SELECT u.first_name AS n FROM users u",
        "SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM users",
        "SELECT x FROM (SELECT email AS x FROM users)",
        "WITH c AS (SELECT id, email FROM users) SELECT id FROM c",
        "SELECT id, (SELECT email FROM users LIMIT 1) AS e FROM orders",
        "SELECT ARRAY_AGG(email) AS emails FROM users",
        "SELECT ST_ASTEXT(user_geom) AS g FROM users",
        "SELECT FIRST_VALUE(email) OVER (ORDER BY id) AS e FROM users",
        "SELECT CASE WHEN age > 30 THEN email END AS e FROM users",
        "SELECT o.order_id, u.street_address FROM orders o JOIN users u ON o.user_id = u.id",
        "SELECT latitude, longitude FROM users",
        "SELECT id FROM orders UNION ALL SELECT postal_code FROM users",
        "SELECT `EMAIL` FROM users",
    ],
)
def test_pii_in_output_is_rejected(sql):
    rejected(sql, "pii_projection")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM users",
        "SELECT u.* FROM users u",
        "SELECT o.*, u.* FROM orders o JOIN users u ON o.user_id = u.id",
        "SELECT * FROM orders o JOIN users u ON o.user_id = u.id",
        "SELECT * EXCEPT (email) FROM users",
        "WITH c AS (SELECT * FROM users) SELECT id FROM c",
    ],
)
def test_star_over_users_is_rejected(sql):
    error = rejected(sql, "pii_star")
    assert "explicitly" in str(error)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT u FROM users u",
        "SELECT TO_JSON_STRING(u) AS j FROM users AS u",
        "SELECT users FROM users",
    ],
)
def test_whole_users_row_is_rejected(sql):
    rejected(sql, "pii_row")


def test_unnest_of_pii_is_rejected():
    rejected("SELECT x FROM users, UNNEST([email]) AS x", "pii_projection")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(DISTINCT email) AS customers FROM users",
        "SELECT COUNT(email) AS n, APPROX_COUNT_DISTINCT(email) AS m FROM users",
        "SELECT COUNTIF(email LIKE '%@gmail.com') AS gmail FROM users",
        "SELECT id, created_at FROM users WHERE email LIKE '%@gmail.com'",
        "SELECT o.order_id FROM orders o JOIN users u ON o.user_id = u.id AND u.email IS NOT NULL",
        "SELECT state, COUNT(*) AS n FROM users GROUP BY state, postal_code",
        "SELECT SUM(IF(REGEXP_CONTAINS(email, r'@example'), 1, 0)) AS n FROM users",
        "SELECT id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY created_at) AS rn FROM users",
        # Stars over orders leave out `gender`: it is a customer attribute (test_exposure).
        "SELECT * EXCEPT (gender) FROM orders "
        "WHERE user_id IN (SELECT id FROM users WHERE email LIKE '%.com')",
        "SELECT * EXCEPT (gender) FROM orders o "
        "WHERE EXISTS (SELECT * FROM users u WHERE u.id = o.user_id)",
        "SELECT * FROM (SELECT id, created_at FROM users)",
        "SELECT c.* FROM (SELECT id, created_at FROM users) c",
        "SELECT o.* EXCEPT (gender) FROM orders o JOIN users u ON o.user_id = u.id",
        "SELECT COUNT(*) AS n FROM users",
    ],
)
def test_allowed_pii_usage(sql):
    guard(sql)


# -- rewriting ---------------------------------------------------------------------------


def test_tables_are_fully_qualified_and_keep_alias():
    result = guard("SELECT orders.status, o2.order_id FROM orders JOIN orders o2 USING (order_id)")
    tables = list(sqlglot.parse_one(result.sql, read="bigquery").find_all(exp.Table))
    assert {(t.catalog, t.db, t.name) for t in tables} == {
        ("bigquery-public-data", "thelook_ecommerce", "orders")
    }
    assert {t.alias for t in tables} == {"orders", "o2"}
    assert "`bigquery-public-data`.thelook_ecommerce.orders AS orders" in result.sql


def test_ceo_query_has_no_acl_filter():
    result = guard("SELECT category FROM products")
    assert "WHERE" not in result.sql


def test_limit_is_added_to_outer_query_only():
    result = guard("SELECT status FROM (SELECT status FROM orders) t")
    tree = sqlglot.parse_one(result.sql, read="bigquery")
    assert tree.args["limit"].expression.this == "1000"
    assert result.sql.count("LIMIT") == 1


def test_limit_is_added_to_union():
    result = guard("SELECT status FROM orders UNION ALL SELECT status FROM orders")
    assert result.sql.endswith("LIMIT 1000")
    assert result.sql.count("LIMIT") == 1


def test_existing_small_limit_is_kept():
    result = guard("SELECT status FROM orders LIMIT 10")
    assert result.sql.endswith("LIMIT 10")
    assert result.sql.count("LIMIT") == 1


def test_large_limit_is_capped():
    assert guard("SELECT status FROM orders LIMIT 100000").sql.endswith("LIMIT 1000")


def test_inner_limits_are_untouched():
    result = guard("SELECT * FROM (SELECT status FROM orders LIMIT 5000) t", limit=100)
    assert "LIMIT 5000" in result.sql
    assert result.sql.endswith("LIMIT 100")


def test_rejection_is_logged_without_values(tmp_path):
    from retail_agent.observability import logger

    path = tmp_path / "agent.jsonl"
    logger.configure(path)
    rejected("SELECT email FROM users WHERE email = 'john.doe@example.com'")
    text = path.read_text(encoding="utf-8")
    assert '"event": "sql_rejected"' in text
    assert '"rule": "pii_projection"' in text
    assert "john.doe@example.com" not in text
