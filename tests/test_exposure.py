"""Customer attributes reach the output only aggregated over groups of customers (D7)."""

import pytest

from retail_agent.auth import UserContext
from retail_agent.errors import SqlRejected
from retail_agent.safety.sql_guard import check_and_rewrite

DATASET = "bigquery-public-data.thelook_ecommerce"
CEO = UserContext("ceo", "CEO", "ceo")
DIESEL = UserContext("diesel", "Diesel manager", "manager", ("Diesel",))
SPEND = (
    "SELECT u.id, u.state, SUM(oi.sale_price) AS spend "
    "FROM users u JOIN order_items oi ON oi.user_id = u.id GROUP BY 1, 2"
)

ALLOWED = [
    "SELECT state, COUNT(*) AS n FROM users GROUP BY state",
    "SELECT gender, COUNT(*) AS n FROM orders GROUP BY gender",
    "SELECT traffic_source, COUNT(DISTINCT u.id) AS n FROM users u GROUP BY 1",
    "SELECT DISTINCT state FROM users",
    "SELECT state, ROUND(AVG(age), 1) AS a FROM users GROUP BY state "
    "UNION ALL SELECT 'all', AVG(age) FROM users",
    # Per-customer attributes inside, aggregated away outside.
    f"WITH c AS ({SPEND}) SELECT state, AVG(spend) AS avg_spend FROM c GROUP BY state",
    "WITH s AS (SELECT state, COUNT(*) AS n FROM users GROUP BY state) "
    "SELECT state, n FROM s ORDER BY n DESC",
    "SELECT x.s, COUNT(*) AS n FROM (SELECT state AS s FROM users) x GROUP BY 1",
    # Filtering by an attribute is allowed; customers are shown by id only.
    "SELECT id FROM users WHERE state = 'Texas'",
    "SELECT u.id, SUM(oi.sale_price) AS s FROM users u JOIN order_items oi "
    "ON oi.user_id = u.id WHERE u.state = 'Texas' GROUP BY u.id ORDER BY s DESC LIMIT 10",
    # Product ids are not customer identifiers.
    "SELECT p.id, u.gender, COUNT(*) AS n FROM order_items oi "
    "JOIN products p ON p.id = oi.product_id JOIN users u ON u.id = oi.user_id "
    "GROUP BY p.id, u.gender",
    "SELECT p.category, u.gender, SUM(oi.sale_price) AS revenue FROM order_items oi "
    "JOIN products p ON p.id = oi.product_id JOIN users u ON u.id = oi.user_id GROUP BY 1, 2",
]

REJECTED = [
    "SELECT id, age FROM users",
    "SELECT city FROM users",  # one row per customer, even without an id
    "SELECT order_id, gender FROM orders",
    "SELECT DISTINCT id, state FROM users",
    SPEND,
    "SELECT u.id, ANY_VALUE(u.city) AS c FROM users u GROUP BY u.id",
    "SELECT u.id, AVG(u.age) AS a FROM users u GROUP BY u.id",  # an average of one person
    f"WITH c AS ({SPEND}) SELECT * FROM c",
    "SELECT x.s FROM (SELECT state AS s FROM users) x",
    "SELECT u.state AS s, u.id AS who FROM users u",
    "SELECT * EXCEPT (first_name, last_name, email, street_address, postal_code, latitude, "
    "longitude, user_geom) FROM users",
    "SELECT o.order_id, (SELECT u.city FROM users u WHERE u.id = o.user_id) AS c FROM orders o",
    "SELECT state, ARRAY_AGG(STRUCT(id, city)) AS a FROM users GROUP BY state",
    "SELECT id, state FROM users UNION ALL SELECT 1, 'x'",
    "SELECT u.id, CASE WHEN u.age > 40 THEN 'old' ELSE 'young' END AS seg FROM users u",
    "SELECT u.id, ROW_NUMBER() OVER (PARTITION BY u.state ORDER BY u.created_at) AS rn "
    "FROM users u",
]


def guard(sql, user=CEO):
    return check_and_rewrite(sql, user, dataset=DATASET, default_limit=1000)


@pytest.mark.parametrize("user", [CEO, DIESEL])
@pytest.mark.parametrize("sql", ALLOWED)
def test_aggregated_attributes_are_allowed(sql, user):
    guard(sql, user)


@pytest.mark.parametrize("user", [CEO, DIESEL])
@pytest.mark.parametrize("sql", REJECTED)
def test_attribute_tied_to_an_individual_is_rejected(sql, user):
    with pytest.raises(SqlRejected) as err:
        guard(sql, user)
    assert err.value.rule in {"pii_attribute", "pii_star"}


def test_rejection_names_the_attributes_and_the_fix():
    with pytest.raises(SqlRejected) as err:
        guard("SELECT id, age, state FROM users")
    message = str(err.value)
    assert err.value.rule == "pii_attribute"
    assert "(age, state)" in message and "GROUP BY state" in message
    assert "filtering by it in WHERE is allowed" in message
