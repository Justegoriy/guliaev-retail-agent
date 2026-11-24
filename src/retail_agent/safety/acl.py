"""Row-level access control by brand (D1-D3, T4).

For brand managers every base-table reference is replaced by a filtered subquery
that keeps the original alias, so any SQL the LLM writes only sees rows related to
the user's brands:

- products    -> products of allowed brands
- order_items -> items of allowed products
- orders      -> orders containing at least one allowed item; `num_of_item` is
                 recomputed to count only the allowed items of the order
- users       -> users who bought at least one allowed item

So "items sold", "revenue" and "orders" are consistent: a mixed order counts as one
order of the user's brands with only their items and revenue.
"""

from sqlglot import exp, parse_one

from retail_agent.auth import UserContext
from retail_agent.errors import SqlRejected


def _subqueries(dataset: str, brands: tuple[str, ...]) -> dict[str, str]:
    if not brands:
        # Brands come from validated users.yaml; this is a defensive check only.
        raise SqlRejected("empty brand scope", rule="acl")
    # Literals are escaped by sqlglot, so a brand name like "Levi's" cannot break the string.
    allowed = ", ".join(exp.Literal.string(b).sql("bigquery") for b in sorted(brands))
    products = f"`{dataset}.products`"
    allowed_items = (
        f"SELECT * FROM `{dataset}.order_items` WHERE product_id IN "
        f"(SELECT id FROM {products} WHERE brand IN ({allowed}))"
    )
    return {
        "products": f"SELECT * FROM {products} WHERE brand IN ({allowed})",
        "order_items": allowed_items,
        "orders": (
            "SELECT o.* REPLACE (a.num_of_item AS num_of_item) "
            f"FROM `{dataset}.orders` AS o JOIN "
            f"(SELECT order_id, COUNT(*) AS num_of_item FROM ({allowed_items}) GROUP BY order_id)"
            " AS a ON o.order_id = a.order_id"
        ),
        "users": (
            f"SELECT * FROM `{dataset}.users` WHERE id IN (SELECT user_id FROM ({allowed_items}))"
        ),
    }


def _is_dataset_table(table: exp.Table, dataset: str) -> bool:
    return f"{table.catalog}.{table.db}".lower() == dataset.lower()


def apply(tree: exp.Expression, user: UserContext, dataset: str) -> exp.Expression:
    """Wrap every fully-qualified `dataset` table in `tree` into its ACL subquery (in place)."""
    if user.has_full_access:
        return tree
    templates = _subqueries(dataset, user.brands)
    tables = [t for t in tree.find_all(exp.Table) if _is_dataset_table(t, dataset)]
    for table in tables:
        template = templates.get(table.name.lower())
        if template is None:
            raise SqlRejected(f"table {table.name} is not available", rule="acl")
        table.replace(
            exp.Subquery(
                this=parse_one(template, read="bigquery"),
                alias=exp.TableAlias(this=exp.to_identifier(table.alias or table.name)),
            )
        )
    return tree
