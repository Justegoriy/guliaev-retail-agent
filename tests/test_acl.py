import pytest
import sqlglot
from sqlglot import exp

from retail_agent.auth import UserContext
from retail_agent.errors import SqlRejected
from retail_agent.safety import acl
from retail_agent.safety.sql_guard import check_and_rewrite

DATASET = "bigquery-public-data.thelook_ecommerce"
CEO = UserContext("ceo", "CEO", "ceo")
CFO = UserContext("cfo", "CFO", "c_level")
HEAD_MEN = UserContext("head_men", "Head of Men", "executive", ("Men",))

QUERIES = [
    "SELECT category, COUNT(*) AS n FROM products GROUP BY category",
    "SELECT p.category, SUM(oi.sale_price) AS revenue FROM order_items oi "
    "JOIN products p ON oi.product_id = p.id GROUP BY 1",
    "SELECT orders.status, COUNT(*) AS n FROM orders GROUP BY orders.status",
    "SELECT u.state, COUNT(DISTINCT o.order_id) AS orders FROM orders AS o "
    "JOIN users AS u ON o.user_id = u.id GROUP BY u.state",
    "WITH rev AS (SELECT product_id, SUM(sale_price) AS r FROM order_items GROUP BY 1) "
    "SELECT p.brand, SUM(rev.r) AS r FROM rev JOIN products p ON p.id = rev.product_id GROUP BY 1",
    "SELECT department, COUNT(*) AS n FROM products "
    "WHERE department IN ('Men', 'Women') GROUP BY 1",
    "SELECT * FROM orders o "
    "WHERE EXISTS (SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)",
    "SELECT id FROM users UNION ALL SELECT user_id FROM orders",
    "SELECT t.n FROM (SELECT COUNT(*) AS n FROM thelook_ecommerce.order_items) t",
    "SELECT status FROM `bigquery-public-data.thelook_ecommerce.orders`",
]


def guard(sql, user):
    return check_and_rewrite(sql, user, dataset=DATASET, default_limit=1000)


def acl_selects(departments):
    """SQL of every SELECT in the ACL templates that reads a base table directly."""
    selects = set()
    for template in acl._subqueries(DATASET, departments).values():
        for select in sqlglot.parse_one(template, read="bigquery").find_all(exp.Select):
            if isinstance(select.args["from_"].this, exp.Table):
                selects.add(select.sql(dialect="bigquery"))
    return selects


@pytest.mark.parametrize("sql", QUERIES)
def test_every_table_is_read_only_through_acl_filter(sql):
    result = guard(sql, HEAD_MEN)
    allowed = acl_selects(("Men",))
    tree = sqlglot.parse_one(result.sql, read="bigquery")
    tables = list(tree.find_all(exp.Table))
    assert tables
    for table in tables:
        if not table.db:  # CTE reference
            continue
        enclosing = table.find_ancestor(exp.Select)
        assert enclosing.sql(dialect="bigquery") in allowed, table.sql()
    if "Women" not in sql:
        assert "Women" not in result.sql


@pytest.mark.parametrize("user", [CEO, CFO])
@pytest.mark.parametrize("sql", QUERIES)
def test_full_access_roles_are_not_filtered(sql, user):
    result = guard(sql, user)
    tree = sqlglot.parse_one(result.sql, read="bigquery")
    assert not any(
        isinstance(s.args.get("from_") and s.args["from_"].this, exp.Subquery)
        and "department IN" in s.sql()
        for s in tree.find_all(exp.Select)
    )
    assert "department IN ('Men')" not in result.sql


def test_alias_is_preserved():
    result = guard("SELECT o.status FROM orders AS o", HEAD_MEN)
    tree = sqlglot.parse_one(result.sql, read="bigquery")
    subquery = tree.args["from_"].this
    assert isinstance(subquery, exp.Subquery)
    assert subquery.alias == "o"


def test_table_name_becomes_alias_when_missing():
    result = guard("SELECT orders.status FROM orders", HEAD_MEN)
    subquery = sqlglot.parse_one(result.sql, read="bigquery").args["from_"].this
    assert subquery.alias == "orders"


def test_join_of_several_tables_wraps_each():
    result = guard(
        "SELECT p.category, COUNT(DISTINCT o.order_id) AS n FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.order_id "
        "JOIN products p ON p.id = oi.product_id GROUP BY 1",
        HEAD_MEN,
    )
    tree = sqlglot.parse_one(result.sql, read="bigquery")
    aliases = {s.alias for s in tree.find_all(exp.Subquery) if s.alias}
    assert {"o", "oi", "p"} <= aliases
    assert result.tables == {"orders", "order_items", "products"}


def test_cte_body_is_filtered():
    result = guard(
        "WITH c AS (SELECT product_id FROM order_items) SELECT COUNT(*) AS n FROM c", HEAD_MEN
    )
    cte = sqlglot.parse_one(result.sql, read="bigquery").find(exp.CTE)
    assert "department IN ('Men')" in cte.sql(dialect="bigquery")


def test_multiple_departments():
    both = UserContext("x", "X", "executive", ("Women", "Men"))
    assert "department IN ('Men', 'Women')" in guard("SELECT id FROM products", both).sql


def test_unknown_department_is_rejected():
    rogue = UserContext("x", "X", "executive", ("Men') OR TRUE --",))
    with pytest.raises(SqlRejected):
        guard("SELECT id FROM products", rogue)


@pytest.mark.network
def test_head_men_sees_only_men_in_bigquery():
    from retail_agent.data.bigquery import BigQueryRunner
    from retail_agent.settings import get_settings

    runner = BigQueryRunner.from_settings(get_settings())
    queries = {
        "products": "SELECT DISTINCT department FROM products",
        "order_items": "SELECT DISTINCT p.department FROM order_items oi "
        "JOIN `bigquery-public-data.thelook_ecommerce.products` p ON p.id = oi.product_id",
    }
    for sql in queries.values():
        rewritten = guard(sql, HEAD_MEN).sql
        assert set(runner.execute(rewritten).df["department"]) == {"Men"}
    # Asking for Women explicitly still returns nothing.
    rewritten = guard("SELECT COUNT(*) AS n FROM products WHERE department = 'Women'", HEAD_MEN)
    assert int(runner.execute(rewritten.sql).df["n"][0]) == 0


@pytest.mark.network
@pytest.mark.parametrize("sql", QUERIES)
def test_rewritten_queries_pass_bigquery_dry_run(sql):
    from retail_agent.data.bigquery import BigQueryRunner
    from retail_agent.settings import get_settings

    runner = BigQueryRunner.from_settings(get_settings())
    assert runner.dry_run(guard(sql, HEAD_MEN).sql) > 0
