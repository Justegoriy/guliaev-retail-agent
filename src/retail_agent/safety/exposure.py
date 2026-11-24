"""Customer attributes only in aggregates (D7): the attribute-exposure rule of the SQL Guard.

Customer attributes (`age`, `gender`, `city`, `state`, `country`, `traffic_source`) may
reach the query output only aggregated over groups of customers. Anything that ties an
attribute to one individual is rejected: an attribute next to a customer or order id,
one output row per customer or order, `ANY_VALUE(state)` grouped by customer, and so on.
Filtering by an attribute (`WHERE state = 'Texas'`) stays allowed: the attribute itself
is not in the output, and customers are referred to by their id.

The check follows the query's scopes bottom-up (CTEs, derived tables, set operations,
scalar subqueries) and tracks, for every output column, whether it carries an attribute
or an identifier, and, for every scope, whether its rows are individual (one row per
customer, order or order item, or finer). Only the final output is judged: inner scopes
may hold per-customer attributes as long as an outer scope aggregates them away.
"""

from dataclasses import dataclass, field

from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from retail_agent.errors import SqlRejected

ATTRIBUTES: dict[str, frozenset[str]] = {
    "users": frozenset({"age", "gender", "city", "state", "country", "traffic_source"}),
    "orders": frozenset({"gender"}),
}
# Columns whose value points at one customer: a customer id, or an order (item) id,
# since every order belongs to exactly one customer.
IDENTIFIERS: dict[str, frozenset[str]] = {
    "users": frozenset({"id"}),
    "orders": frozenset({"order_id", "user_id"}),
    "order_items": frozenset({"id", "order_id", "user_id"}),
}
# Tables whose rows each belong to one customer.
INDIVIDUAL_TABLES = frozenset({"users", "orders", "order_items"})
# Aggregates that return one of the input values: they do not hide whose value it is.
_PASS_THROUGH_AGGS = tuple(
    getattr(exp, name)
    for name in ("AnyValue", "ArrayAgg", "GroupConcat", "Max", "Min", "First", "Last")
    if hasattr(exp, name)
)

ATTRIBUTE_LIST = "age, gender, city, state, country, traffic_source"
REJECTION = (
    "Customer attributes ({attrs}) may be shown only aggregated over groups of customers, "
    "never next to a customer or order id and never one row per customer or order. "
    "Group by the attribute (e.g. GROUP BY state with COUNT/SUM/AVG), or drop it from the "
    "output; filtering by it in WHERE is allowed."
)


@dataclass
class Taint:
    attribute: set[str] = field(default_factory=set)  # attribute names carried
    identifier: bool = False

    def add(self, other: "Taint") -> None:
        self.attribute |= other.attribute
        self.identifier = self.identifier or other.identifier

    @property
    def clean(self) -> bool:
        return not self.attribute and not self.identifier


@dataclass
class ScopeInfo:
    outputs: dict[str, Taint]  # output column name (lower) -> taint
    individual: bool  # rows are per customer / order / order item, or finer


def _table_outputs(name: str) -> dict[str, Taint]:
    outputs: dict[str, Taint] = {}
    for column in ATTRIBUTES.get(name, ()):
        outputs.setdefault(column, Taint()).attribute.add(column)
    for column in IDENTIFIERS.get(name, ()):
        outputs.setdefault(column, Taint()).identifier = True
    return outputs


def _table_info(table: exp.Table) -> ScopeInfo:
    name = table.name.lower()
    return ScopeInfo(outputs=_table_outputs(name), individual=name in INDIVIDUAL_TABLES)


class _Analyzer:
    def __init__(self) -> None:
        self.infos: dict[int, ScopeInfo] = {}  # id(scope.expression) -> info

    def source_info(self, source: object) -> ScopeInfo:
        if isinstance(source, Scope):
            return self.infos[id(source.expression)]
        if isinstance(source, exp.Table):
            return _table_info(source)
        return ScopeInfo(outputs={}, individual=False)

    # -- columns and expressions ------------------------------------------------------------

    def column_taint(self, column: exp.Column, scope: Scope) -> Taint:
        name = column.name.lower()
        sources = scope.selected_sources
        if column.table:
            entry = sources.get(column.table)
            candidates = [entry[1]] if entry else []
        else:
            candidates = [source for _, source in sources.values()]
        taint = Taint()
        for source in candidates:
            found = self.source_info(source).outputs.get(name)
            if found is not None:
                taint.add(found)
        return taint

    def star_outputs(self, star: exp.Expression, scope: Scope) -> dict[str, Taint]:
        qualifier = star.table if isinstance(star, exp.Column) else ""
        sources = scope.selected_sources
        if qualifier:
            entry = sources.get(qualifier)
            chosen = [entry[1]] if entry else []
        else:
            chosen = [source for _, source in sources.values()]
        node = star.this if isinstance(star, exp.Column) else star
        except_list = node.args.get("except_") or node.args.get("except") or []
        excluded = {e.name.lower() for e in except_list}
        outputs: dict[str, Taint] = {}
        for source in chosen:
            for name, taint in self.source_info(source).outputs.items():
                if name not in excluded:
                    outputs.setdefault(name, Taint()).add(taint)
        return outputs

    def expression_taint(
        self, expression: exp.Expression, scope: Scope, *, launders: bool
    ) -> Taint:
        """Taint of a projection; `launders`: aggregates over groups hide individual values."""
        taint = Taint()
        for node in expression.walk():
            if isinstance(node, exp.Subquery | exp.Select) and node is not expression:
                inner = self.infos.get(id(node.this if isinstance(node, exp.Subquery) else node))
                if inner is not None:
                    for found in inner.outputs.values():
                        taint.add(found)
                continue
            if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star):
                continue
            if node.find_ancestor(exp.Select) is not scope.expression:
                continue  # belongs to a nested scope, handled above
            if launders and self._laundered(node, expression):
                continue
            taint.add(self.column_taint(node, scope))
        return taint

    @staticmethod
    def _laundered(column: exp.Column, stop: exp.Expression) -> bool:
        node = column.parent
        while node is not None and node is not stop.parent:
            if isinstance(node, exp.Window):
                return False  # row-level value
            if isinstance(node, exp.AggFunc):
                return not isinstance(node, _PASS_THROUGH_AGGS)
            node = node.parent
        return False

    # -- scopes -----------------------------------------------------------------------------

    def analyze(self, scope: Scope) -> ScopeInfo:
        expression = scope.expression
        if isinstance(expression, exp.SetOperation):
            left, right = (self._branch(expression.args[side]) for side in ("this", "expression"))
            outputs = {name: Taint() for name in left.outputs}
            for branch in (left, right):
                for i, taint in enumerate(branch.outputs.values()):
                    if i < len(outputs):
                        list(outputs.values())[i].add(taint)
            info = ScopeInfo(outputs=outputs, individual=left.individual or right.individual)
        elif isinstance(expression, exp.Select):
            info = self._select(scope, expression)
        else:
            info = ScopeInfo(outputs={}, individual=False)
        self.infos[id(expression)] = info
        return info

    def _branch(self, node: exp.Expression) -> ScopeInfo:
        while isinstance(node, exp.Subquery):
            node = node.this
        # A branch that was not analysed (unexpected shape) is treated as the worst case.
        return self.infos.get(id(node), ScopeInfo(outputs={}, individual=True))

    def _select(self, scope: Scope, select: exp.Select) -> ScopeInfo:
        group = select.args.get("group")
        has_agg = any(
            agg.find_ancestor(exp.Window) is None and agg.find_ancestor(exp.Select) is select
            for projection in select.selects
            for agg in projection.find_all(exp.AggFunc)
        )
        aggregated = bool(group) or has_agg
        distinct = bool(select.args.get("distinct"))
        sources_individual = any(
            self.source_info(source).individual for _, source in scope.selected_sources.values()
        )

        # First pass without laundering: which outputs identify an individual.
        raw: dict[str, Taint] = {}
        for projection in select.selects:
            if isinstance(projection, exp.Star) or (
                isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
            ):
                for name, taint in self.star_outputs(projection, scope).items():
                    raw.setdefault(name, Taint()).add(taint)
                continue
            raw.setdefault(projection.alias_or_name.lower(), Taint()).add(
                self.expression_taint(projection, scope, launders=False)
            )

        if group:
            keys = self._group_keys(select, group)
            individual = any(
                self.expression_taint(key, scope, launders=False).identifier for key in keys
            )
        elif aggregated:
            individual = False  # one row
        elif distinct:
            individual = any(t.identifier for t in raw.values())
        else:
            individual = sources_individual

        if aggregated and not individual:
            outputs: dict[str, Taint] = {}
            for projection in select.selects:
                outputs.setdefault(projection.alias_or_name.lower(), Taint()).add(
                    self.expression_taint(projection, scope, launders=True)
                )
        else:
            outputs = raw
        return ScopeInfo(outputs=outputs, individual=individual)

    @staticmethod
    def _group_keys(select: exp.Select, group: exp.Group) -> list[exp.Expression]:
        by_alias = {p.alias_or_name.lower(): p for p in select.selects}
        keys: list[exp.Expression] = []
        for key in group.expressions:
            if isinstance(key, exp.Literal) and key.is_int:
                position = int(key.this) - 1
                if 0 <= position < len(select.selects):
                    keys.append(select.selects[position])
                continue
            if isinstance(key, exp.Column) and not key.table:
                alias = by_alias.get(key.name.lower())
                if alias is not None and isinstance(alias, exp.Alias):
                    keys.append(alias)
                    continue
            keys.append(key)
        return keys


def check(tree: exp.Expression) -> None:
    """Raise `SqlRejected` if the query output ties a customer attribute to an individual."""
    analyzer = _Analyzer()
    root: ScopeInfo | None = None
    for scope in traverse_scope(tree):
        info = analyzer.analyze(scope)
        if scope.is_root:
            root = info
    if root is None:
        return
    exposed: set[str] = set()
    for taint in root.outputs.values():
        if taint.attribute and (root.individual or taint.identifier):
            exposed |= taint.attribute
    if exposed:
        raise SqlRejected(REJECTION.format(attrs=", ".join(sorted(exposed))), rule="pii_attribute")
