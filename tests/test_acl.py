import pytest
import sqlglot
from sqlglot import exp

from retail_agent.auth import UserContext
from retail_agent.errors import SqlRejected
from retail_agent.safety import acl
from retail_agent.safety.sql_guard import check_and_rewrite

DATASET = "bigquery-public-data.thelook_ecommerce"
CEO = UserContext("ceo", "CEO", "ceo")
JEANS = UserContext("diesel_manager", "Diesel manager", "manager", ("Diesel",))
JEANS_FILTER = "brand IN ('Diesel')"

QUERIES = [
    "SELECT brand, COUNT(*) AS n FROM products GROUP BY brand",
    "SELECT p.brand, SUM(oi.sale_price) AS revenue FROM order_items oi "
    "JOIN products p ON oi.product_id = p.id GROUP BY 1",
    "SELECT orders.status, COUNT(*) AS n FROM orders GROUP BY orders.status",
    "SELECT u.state, COUNT(DISTINCT o.order_id) AS orders FROM orders AS o "
    "JOIN users AS u ON o.user_id = u.id GROUP BY u.state",
    "WITH rev AS (SELECT product_id, SUM(sale_price) AS r FROM order_items GROUP BY 1) "
    "SELECT p.brand, SUM(rev.r) AS r FROM rev JOIN products p ON p.id = rev.product_id GROUP BY 1",
    "SELECT category, COUNT(*) AS n FROM products WHERE brand IN ('Diesel', 'Nike') GROUP BY 1",
    "SELECT o.order_id, o.status FROM orders o "
    "WHERE EXISTS (SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)",
    "SELECT id FROM users UNION ALL SELECT user_id FROM orders",
    "SELECT t.n FROM (SELECT COUNT(*) AS n FROM thelook_ecommerce.order_items) t",
    "SELECT status FROM `bigquery-public-data.thelook_ecommerce.orders`",
    "SELECT AVG(num_of_item) AS avg_items FROM orders",
]


def guard(sql, user):
    return check_and_rewrite(sql, user, dataset=DATASET, default_limit=1000)


def acl_selects(brands):
    """SQL of every SELECT in the ACL templates that reads a base table directly."""
    selects = set()
    for template in acl._subqueries(DATASET, brands).values():
        for select in sqlglot.parse_one(template, read="bigquery").find_all(exp.Select):
            if isinstance(select.args["from_"].this, exp.Table):
                selects.add(select.sql(dialect="bigquery"))
    return selects


@pytest.mark.parametrize("sql", QUERIES)
def test_every_table_is_read_only_through_acl_filter(sql):
    result = guard(sql, JEANS)
    allowed = acl_selects(("Diesel",))
    tree = sqlglot.parse_one(result.sql, read="bigquery")
    tables = [t for t in tree.find_all(exp.Table) if t.db]  # skip CTE references
    assert tables
    for table in tables:
        enclosing = table.find_ancestor(exp.Select)
        assert enclosing.sql(dialect="bigquery") in allowed, table.sql()
    if "Nike" not in sql:
        assert "Nike" not in result.sql


@pytest.mark.parametrize("sql", QUERIES)
def test_ceo_is_not_filtered(sql):
    result = guard(sql, CEO)
    assert JEANS_FILTER not in result.sql
    assert "REPLACE" not in result.sql


def test_alias_is_preserved():
    result = guard("SELECT o.status FROM orders AS o", JEANS)
    subquery = sqlglot.parse_one(result.sql, read="bigquery").args["from_"].this
    assert isinstance(subquery, exp.Subquery)
    assert subquery.alias == "o"


def test_table_name_becomes_alias_when_missing():
    result = guard("SELECT orders.status FROM orders", JEANS)
    subquery = sqlglot.parse_one(result.sql, read="bigquery").args["from_"].this
    assert subquery.alias == "orders"


def test_join_of_several_tables_wraps_each():
    result = guard(
        "SELECT p.brand, COUNT(DISTINCT o.order_id) AS n FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.order_id "
        "JOIN products p ON p.id = oi.product_id GROUP BY 1",
        JEANS,
    )
    tree = sqlglot.parse_one(result.sql, read="bigquery")
    aliases = {s.alias for s in tree.find_all(exp.Subquery) if s.alias}
    assert {"o", "oi", "p"} <= aliases
    assert result.tables == {"orders", "order_items", "products"}


def test_cte_body_is_filtered():
    result = guard(
        "WITH c AS (SELECT product_id FROM order_items) SELECT COUNT(*) AS n FROM c", JEANS
    )
    cte = sqlglot.parse_one(result.sql, read="bigquery").find(exp.CTE)
    assert JEANS_FILTER in cte.sql(dialect="bigquery")


def test_orders_num_of_item_counts_only_allowed_items():
    result = guard("SELECT SUM(num_of_item) AS items FROM orders", JEANS)
    orders = sqlglot.parse_one(result.sql, read="bigquery").args["from_"].this.this
    assert "REPLACE (a.num_of_item AS num_of_item)" in orders.sql(dialect="bigquery")
    assert "COUNT(*) AS num_of_item" in orders.sql(dialect="bigquery")


def test_multiple_brands_with_special_characters():
    user = UserContext("x", "X", "manager", ("Levi's", "7 For All Mankind", "Ben & Jerry"))
    tree = sqlglot.parse_one(guard("SELECT id FROM products", user).sql, read="bigquery")
    where = tree.args["from_"].this.this.args["where"].this
    assert [e.this for e in where.expressions] == ["7 For All Mankind", "Ben & Jerry", "Levi's"]


def test_brand_literal_cannot_inject_sql():
    rogue = UserContext("x", "X", "manager", ("Diesel') OR TRUE --",))
    sql = guard("SELECT id FROM products", rogue).sql
    tree = sqlglot.parse_one(sql, read="bigquery")
    where = tree.args["from_"].this.this.args["where"].this
    assert isinstance(where, exp.In)
    assert [e.this for e in where.expressions] == ["Diesel') OR TRUE --"]


def test_empty_scope_is_rejected():
    with pytest.raises(SqlRejected):
        guard("SELECT id FROM products", UserContext("x", "X", "manager", ()))


def bq_runner():
    from retail_agent.data.bigquery import BigQueryRunner
    from retail_agent.settings import get_settings

    return BigQueryRunner.from_settings(get_settings())


@pytest.mark.network
@pytest.mark.parametrize("user_id", ["denim_manager", "outdoor_manager"])
def test_manager_sees_exactly_their_brands_in_bigquery(user_id):
    from retail_agent.auth import load_user

    user = load_user(user_id)
    runner = bq_runner()
    queries = [
        "SELECT DISTINCT brand FROM products",
        "SELECT DISTINCT p.brand FROM order_items oi "
        "JOIN `bigquery-public-data.thelook_ecommerce.products` p ON p.id = oi.product_id",
    ]
    for sql in queries:
        # Equality also catches a misspelt brand in users.yaml (it would match nothing).
        assert set(runner.execute(guard(sql, user).sql).df["brand"]) == set(user.brands)
    rewritten = guard("SELECT COUNT(*) AS n FROM products WHERE brand = 'Nike'", user)
    assert int(runner.execute(rewritten.sql).df["n"][0]) == 0


@pytest.mark.network
def test_items_revenue_and_orders_are_consistent_in_bigquery():
    runner = bq_runner()
    items = runner.execute(
        guard(
            "SELECT COUNT(*) AS items, COUNT(DISTINCT order_id) AS orders FROM order_items", JEANS
        ).sql
    ).df
    orders = runner.execute(
        guard("SELECT COUNT(*) AS orders, SUM(num_of_item) AS items FROM orders", JEANS).sql
    ).df
    assert int(orders["orders"][0]) == int(items["orders"][0])
    assert int(orders["items"][0]) == int(items["items"][0])


@pytest.mark.network
@pytest.mark.parametrize("sql", QUERIES)
def test_rewritten_queries_pass_bigquery_dry_run(sql):
    assert bq_runner().dry_run(guard(sql, JEANS).sql) > 0
