"""SQL Guard: every query the LLM writes passes through `check_and_rewrite` before BigQuery.

Checks: single read-only statement, whitelisted tables only, no PII in query output (D6).
Rewrites: fully-qualified table names, department ACL (T4), mandatory outer LIMIT.
Rejections raise `SqlRejected` with a message addressed to the LLM so it can fix the query.
"""

from typing import NamedTuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from retail_agent.auth import UserContext
from retail_agent.errors import SqlRejected
from retail_agent.observability.logger import log_event
from retail_agent.safety import acl, pii
from retail_agent.settings import get_settings

ALLOWED_TABLES = ("orders", "order_items", "products", "users")
PII_TABLE = "users"

_FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "TruncateTable",
        "Command", "Export", "LoadData", "Copy", "Grant", "Revoke", "Declare", "Set", "Use",
        "Transaction", "Commit", "Rollback",
    )
    if hasattr(exp, name)
)  # fmt: skip
_COUNT_FUNCS = (exp.Count, exp.ApproxDistinct, exp.CountIf)
# PII inside a boolean test yields only true/false, same as using it in WHERE.
_BOOLEAN_FUNCS = (exp.Predicate, exp.RegexpLike, exp.StartsWith, exp.EndsWith)


class GuardResult(NamedTuple):
    sql: str
    tables: set[str]


def check_and_rewrite(
    sql: str,
    user: UserContext,
    *,
    dataset: str | None = None,
    default_limit: int | None = None,
) -> GuardResult:
    settings = get_settings()
    dataset = dataset or settings.bq_dataset
    default_limit = default_limit or settings.sql_default_limit
    try:
        tree = _parse(sql)
        _check_read_only(tree)
        base_tables = _check_tables(tree, dataset)
        _check_pii_projections(tree, base_tables)
        _qualify(base_tables, dataset)
        tree = acl.apply(tree, user, dataset)
        _ensure_limit(tree, default_limit)
    except SqlRejected as exc:
        log_event("sql_rejected", level="warning", rule=exc.rule, reason=str(exc), sql=sql)
        raise
    return GuardResult(
        sql=tree.sql(dialect="bigquery"),
        tables={t.name.lower() for t in base_tables},
    )


# -- 1-2. parsing and read-only ----------------------------------------------------------


def _parse(sql: str) -> exp.Expression:
    try:
        statements = [s for s in sqlglot.parse(sql, read="bigquery") if s is not None]
    except (ParseError, TokenError) as exc:
        errors = getattr(exc, "errors", None) or []
        detail = errors[0].get("description") if errors else str(exc).splitlines()[0]
        raise SqlRejected(
            f"Could not parse the query as BigQuery SQL: {detail}", rule="parse"
        ) from None
    if not statements:
        raise SqlRejected("The query is empty.", rule="parse")
    if len(statements) > 1:
        raise SqlRejected(
            "Only one statement per call is allowed; send a single SELECT.",
            rule="single_statement",
        )
    tree = statements[0]
    while isinstance(tree, exp.Subquery) and not tree.alias and not tree.args.get("limit"):
        tree = tree.this
    return tree


def _check_read_only(tree: exp.Expression) -> None:
    if not isinstance(tree, exp.Query) or any(
        isinstance(node, _FORBIDDEN_NODES) for node in tree.walk()
    ):
        raise SqlRejected(
            "Only read-only SELECT queries are allowed (no DML, DDL or scripting).",
            rule="read_only",
        )


# -- 3. table whitelist ------------------------------------------------------------------


def _cte_names(tree: exp.Expression) -> set[str]:
    names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    shadowing = names & set(ALLOWED_TABLES)
    if shadowing:
        raise SqlRejected(
            f"CTE name {sorted(shadowing)[0]!r} shadows a dataset table; rename the CTE.",
            rule="cte_name",
        )
    return names


def _check_tables(tree: exp.Expression, dataset: str) -> list[exp.Table]:
    """Validate every table reference; return the nodes that point at dataset tables."""
    project, _, dataset_id = dataset.rpartition(".")
    ctes = _cte_names(tree)
    allowed = ", ".join(ALLOWED_TABLES)
    base: list[exp.Table] = []
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SqlRejected(
                f"Table functions are not allowed. Query only the tables: {allowed}.",
                rule="table_function",
            )
        name = table.name.lower()
        if not table.db and not table.catalog and name in ctes:
            continue
        db_ok = not table.db or table.db.lower() == dataset_id.lower()
        catalog_ok = not table.catalog or table.catalog.lower() == project.lower()
        if name not in ALLOWED_TABLES or not db_ok or not catalog_ok:
            full = ".".join(p for p in (table.catalog, table.db, table.name) if p)
            raise SqlRejected(
                f"Table {full!r} is not available. Only these tables of {dataset} "
                f"can be queried: {allowed}.",
                rule="table_whitelist",
            )
        base.append(table)
    return base


# -- 4. PII in query output --------------------------------------------------------------


def _in_allowed_context(node: exp.Expression, stop: exp.Expression, *, boolean: bool) -> bool:
    """True if `node` only feeds a count, a boolean test or a window partition/order."""
    child, parent = node, node.parent
    while child is not stop and parent is not None:
        if isinstance(parent, _COUNT_FUNCS):
            return True
        if boolean and isinstance(parent, _BOOLEAN_FUNCS):
            return True
        if boolean and isinstance(parent, exp.Window) and child.arg_key != "this":
            return True
        child, parent = parent, parent.parent
    return False


def _output_is_boolean(select: exp.Select) -> bool:
    """Subqueries of EXISTS / IN (...) only produce a true/false test for the outer query."""
    node = select
    while isinstance(node.parent, exp.Subquery | exp.Paren):
        node = node.parent
    parent = node.parent
    return isinstance(parent, exp.Exists) or (
        isinstance(parent, exp.In) and node.arg_key == "query"
    )


def _users_aliases(select: exp.Select, base_ids: set[int]) -> set[str]:
    """Names under which the base `users` table is visible in this SELECT's FROM/JOIN."""
    sources = []
    from_ = select.args.get("from_") or select.args.get("from")
    if from_ is not None:
        sources.append(from_.this)
    sources.extend(join.this for join in select.args.get("joins") or [])
    return {
        (s.alias or s.name).lower()
        for s in sources
        if isinstance(s, exp.Table) and id(s) in base_ids and s.name.lower() == PII_TABLE
    }


def _star_hides_pii(star: exp.Star, pii_cols: frozenset[str]) -> bool:
    excluded = {c.name.lower() for c in star.args.get("except_") or star.args.get("except") or []}
    return pii_cols <= excluded


def _pii_error(column: str) -> SqlRejected:
    return SqlRejected(
        f"Column {column!r} is personal data (PII) and must not appear in the query output. "
        "You may use it in WHERE / JOIN / GROUP BY or inside COUNT(DISTINCT ...); "
        "identify customers by users.id instead.",
        rule="pii_projection",
    )


def _star_error() -> SqlRejected:
    return SqlRejected(
        "SELECT * over the users table would return personal data (PII). "
        "List the needed non-PII columns explicitly.",
        rule="pii_star",
    )


def _check_pii_projections(tree: exp.Expression, base_tables: list[exp.Table]) -> None:
    pii_cols = pii.load_policy().columns.get(PII_TABLE, frozenset())
    all_pii = pii.load_policy().all_columns
    base_ids = {id(t) for t in base_tables}

    for select in tree.find_all(exp.Select):
        if _output_is_boolean(select):
            continue
        users_aliases = _users_aliases(select, base_ids)
        for projection in select.expressions:
            for node in projection.walk():
                if node.find_ancestor(exp.Select) is not select:
                    continue  # nested SELECTs are checked on their own
                if isinstance(node, exp.Star):
                    owner = node.parent if isinstance(node.parent, exp.Column) else None
                    qualifier = owner.table.lower() if owner is not None else None
                    exposes_users = qualifier in users_aliases if qualifier else bool(users_aliases)
                    if (
                        exposes_users
                        and not _star_hides_pii(node, pii_cols)
                        and not _in_allowed_context(node, projection, boolean=False)
                    ):
                        raise _star_error()
                elif (
                    isinstance(node, exp.Column)
                    and not isinstance(node.this, exp.Star)
                    and not isinstance(node.parent, exp.Star)  # * EXCEPT (...) list
                ):
                    name = node.name.lower()
                    if name in all_pii and not _in_allowed_context(node, projection, boolean=True):
                        raise _pii_error(node.name)
                    if not node.table and name in users_aliases:
                        raise SqlRejected(
                            f"{node.name!r} refers to the whole users row, which contains "
                            "personal data (PII). Select individual non-PII columns.",
                            rule="pii_row",
                        )

    # Arrays unnested in FROM produce rows just like a projection does.
    for unnest in tree.find_all(exp.Unnest):
        if not isinstance(unnest.parent, exp.From | exp.Join):
            continue
        for column in unnest.find_all(exp.Column):
            if column.find_ancestor(exp.Select, exp.Unnest) is not unnest:
                continue
            if column.name.lower() in all_pii and not _in_allowed_context(
                column, unnest, boolean=True
            ):
                raise _pii_error(column.name)


# -- 5. qualification, 7. LIMIT ----------------------------------------------------------


def _qualify(base_tables: list[exp.Table], dataset: str) -> None:
    project, _, dataset_id = dataset.rpartition(".")
    for table in base_tables:
        if not table.alias:
            table.set("alias", exp.TableAlias(this=exp.to_identifier(table.name)))
        table.set("this", exp.to_identifier(table.name.lower()))
        table.set("db", exp.to_identifier(dataset_id))
        table.set("catalog", exp.to_identifier(project, quoted=True))


def _ensure_limit(tree: exp.Expression, default_limit: int) -> None:
    limit = tree.args.get("limit")
    if limit is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(default_limit)))
        return
    value = limit.args.get("expression")
    if isinstance(value, exp.Literal) and value.is_int and int(value.this) > default_limit:
        limit.set("expression", exp.Literal.number(default_limit))
