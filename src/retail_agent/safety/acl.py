"""Row-level access control by product department (D1-D3, T4).

For users without full access every base-table reference is replaced by a filtered
subquery that keeps the original alias, so any SQL the LLM writes only sees rows
related to the user's departments:

- products    -> products of allowed departments
- order_items -> items of allowed products
- orders      -> orders containing at least one allowed item
- users       -> users who bought at least one allowed item

Note: `orders.num_of_item` and similar order-level fields still describe the whole order.
"""

from sqlglot import exp, parse_one

from retail_agent.auth import DEPARTMENTS, UserContext
from retail_agent.errors import SqlRejected


def _quote(fq_table: str) -> str:
    return f"`{fq_table}`"


def _subqueries(dataset: str, departments: tuple[str, ...]) -> dict[str, str]:
    unknown = set(departments) - DEPARTMENTS
    if unknown or not departments:
        # Departments come from validated users.yaml; this is a defensive check only.
        raise SqlRejected(f"invalid department scope: {sorted(unknown) or 'empty'}")
    depts = ", ".join(f"'{d}'" for d in sorted(departments))
    products = _quote(f"{dataset}.products")
    order_items = _quote(f"{dataset}.order_items")
    allowed_products = f"SELECT id FROM {products} WHERE department IN ({depts})"
    allowed_items = f"SELECT * FROM {order_items} WHERE product_id IN ({allowed_products})"
    return {
        "products": f"SELECT * FROM {products} WHERE department IN ({depts})",
        "order_items": allowed_items,
        "orders": (
            f"SELECT * FROM {_quote(f'{dataset}.orders')} "
            f"WHERE order_id IN (SELECT order_id FROM ({allowed_items}))"
        ),
        "users": (
            f"SELECT * FROM {_quote(f'{dataset}.users')} "
            f"WHERE id IN (SELECT user_id FROM ({allowed_items}))"
        ),
    }


def _is_dataset_table(table: exp.Table, dataset: str) -> bool:
    return f"{table.catalog}.{table.db}".lower() == dataset.lower()


def apply(tree: exp.Expression, user: UserContext, dataset: str) -> exp.Expression:
    """Wrap every fully-qualified `dataset` table in `tree` into its ACL subquery (in place)."""
    if user.has_full_access:
        return tree
    templates = _subqueries(dataset, user.departments)
    tables = [t for t in tree.find_all(exp.Table) if _is_dataset_table(t, dataset)]
    for table in tables:
        template = templates.get(table.name.lower())
        if template is None:
            raise SqlRejected(f"table {table.name} is not available")
        table.replace(
            exp.Subquery(
                this=parse_one(template, read="bigquery"),
                alias=exp.TableAlias(this=exp.to_identifier(table.alias or table.name)),
            )
        )
    return tree
