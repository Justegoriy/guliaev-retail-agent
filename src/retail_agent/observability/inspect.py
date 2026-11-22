"""Read the event log back: one turn's timeline (`/trace`) and aggregate metrics (`/stats`).

Used by the CLI commands and standalone:

    python -m retail_agent.observability.inspect trace <id-or-prefix> [--full]
    python -m retail_agent.observability.inspect stats [--user ceo]
"""

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from retail_agent.settings import Settings, get_settings

Event = dict[str, Any]

# Fields every record has; the timeline shows only the event-specific rest.
COMMON_FIELDS = frozenset({"ts", "level", "event", "trace_id", "conversation_id", "user_id"})
SQL_EVENTS = frozenset({"sql_dry_run", "sql_executed", "sql_rejected"})
MAX_DETAIL_CHARS = 300
LEVEL_STYLES = {"warning": "yellow", "error": "red"}


def read_jsonl(path: Path) -> Iterator[Event]:
    """Records of a JSONL file; a missing file or a torn last line is not an error."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _turn_starts(events: Iterable[Event], user_id: str | None) -> list[Event]:
    return [
        e
        for e in events
        if e.get("event") == "turn_start"
        and e.get("trace_id")
        and (user_id is None or e.get("user_id") == user_id)
    ]


def resolve_trace_id(events: Iterable[Event], id_or_prefix: str, user_id: str | None = None):
    """Full trace id for an id or a unique prefix among the user's turns, else None."""
    wanted = id_or_prefix.strip().lower()
    if not wanted:
        return None
    ids = list(dict.fromkeys(e["trace_id"] for e in _turn_starts(events, user_id)))
    if wanted in ids:
        return wanted
    matches = [i for i in ids if i.startswith(wanted)]
    return matches[0] if len(matches) == 1 else None


def last_trace_id(
    events: Iterable[Event], user_id: str | None = None, conversation_id: str | None = None
) -> str | None:
    starts = [
        e
        for e in _turn_starts(events, user_id)
        if conversation_id is None or e.get("conversation_id") == conversation_id
    ]
    return starts[-1]["trace_id"] if starts else None


def trace_events(events: Iterable[Event], trace_id: str) -> list[Event]:
    """The turn's events, from `turn_start` to `turn_end` (later records reuse the id)."""
    out: list[Event] = []
    for e in events:
        if e.get("trace_id") != trace_id:
            continue
        out.append(e)
        if e.get("event") == "turn_end":
            break
    return out


def trace_payloads(path: Path, trace_id: str) -> list[Event]:
    return [r for r in read_jsonl(path) if r.get("trace_id") == trace_id]


# -- /trace ---------------------------------------------------------------------------


def _clip(text: Any, limit: int = MAX_DETAIL_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _bytes(value: Any) -> str:
    if not isinstance(value, int | float):
        return "-"
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= size:
            return f"{value / size:.1f} {unit}"
    return f"{int(value)} B"


def _ts(event: Event) -> datetime | None:
    try:
        return datetime.fromisoformat(str(event.get("ts")))
    except ValueError:
        return None


def describe(event: Event) -> str:
    """One-line summary of an event for the timeline."""
    e = event
    name = e.get("event")
    if name == "turn_start":
        return f"message of {e.get('length')} chars"
    if name == "input_guard":
        flags = " BLOCKED" if e.get("blocked") else ""
        pii = " pii_request" if e.get("pii_request") else ""
        return f"category={e.get('category')} source={e.get('source')}{pii}{flags}"
    if name == "golden_retrieved":
        pairs = zip(e.get("ids") or [], e.get("scores") or [], strict=False)
        found = ", ".join(f"{i} ({s:.2f})" for i, s in pairs) or "none above threshold"
        return f"{e.get('method')}: {found}"
    if name == "prompt_built":
        prefs = "; ".join(e.get("preferences") or []) or "none"
        golden = ", ".join(e.get("golden_ids") or []) or "none"
        return (
            f"persona {e.get('persona_version')}, glossary {e.get('glossary_version')}, "
            f"golden: {golden}, preferences: {prefs}, {e.get('chars')} chars"
        )
    if name == "llm_call":
        tools = ", ".join(e.get("tool_calls") or [])
        return (
            f"{e.get('model')} [{e.get('purpose')}/{e.get('model_role')}] "
            f"{e.get('input_tokens')}→{e.get('output_tokens')} tokens, {e.get('latency_ms')} ms"
            + (f", calls: {tools}" if tools else "")
        )
    if name == "llm_error":
        return f"{e.get('model')} {e.get('error')}: {_clip(e.get('detail'), 160)}"
    if name == "llm_retry":
        return (
            f"{e.get('model')} attempt {e.get('attempt')} {e.get('error')}, wait {e.get('wait_s')}s"
        )
    if name == "fallback_used":
        return f"{e.get('from_model')} → {e.get('to_model')} ({e.get('reason')})"
    if name == "tool_call":
        detail = e.get("args") if e.get("status") == "ok" else e.get("detail")
        return f"{e.get('tool')} {e.get('status')} {e.get('latency_ms')} ms: {_clip(detail)}"
    if name == "run_sql":
        acl = " (category ACL)" if e.get("acl") else ""
        return f"{e.get('purpose')} | tables: {', '.join(e.get('tables') or [])}{acl}"
    if name == "sql_dry_run":
        return f"would scan {_bytes(e.get('bytes'))}"
    if name == "sql_executed":
        if e.get("source") == "memory_cache":
            return f"{e.get('rows')} rows from memory cache"
        cache = ", BigQuery cache" if e.get("cache_hit") else ""
        return f"{e.get('rows')} rows, {_bytes(e.get('bytes'))}, {e.get('duration_ms')} ms{cache}"
    if name == "sql_rejected":
        return f"rule={e.get('rule')}: {_clip(e.get('reason'), 200)}"
    if name == "turn_end":
        parts = [f"status={e.get('status')}"]
        if e.get("error_kind"):
            parts.append(f"error={e.get('error_kind')}")
        for key, label in (
            ("latency_ms", "ms"),
            ("llm_calls", "LLM calls"),
            ("input_tokens", "in tokens"),
            ("output_tokens", "out tokens"),
            ("sql_count", "SQL"),
            ("sql_failures", "SQL failures"),
        ):
            if e.get(key) is not None:
                parts.append(f"{e[key]} {label}")
        if e.get("bq_bytes"):
            parts.append(_bytes(e["bq_bytes"]))
        return ", ".join(parts)
    rest = {k: v for k, v in e.items() if k not in COMMON_FIELDS}
    return _clip(" ".join(f"{k}={v}" for k, v in rest.items()))


def _sql_panels(events: list[Event]) -> list[RenderableType]:
    """The SQL as the guard rewrote it (dry run text), with its cost and outcome."""
    panels: list[RenderableType] = []
    for e in events:
        if e.get("event") not in SQL_EVENTS:
            continue
        if e["event"] == "sql_executed" and e.get("source") == "memory_cache":
            continue
        if e["event"] == "sql_dry_run":
            title, style = f"SQL dry run: {_bytes(e.get('bytes'))}", "cyan"
        elif e["event"] == "sql_executed":
            title = (
                f"SQL executed: {e.get('rows')} rows, {_bytes(e.get('bytes'))}, "
                f"{e.get('duration_ms')} ms"
            )
            style = "green"
        else:
            title, style = f"SQL rejected ({e.get('rule')}): {_clip(e.get('reason'), 120)}", "red"
        sql = str(e.get("sql") or "")
        if e["event"] == "sql_executed" and any(
            p.get("event") == "sql_dry_run" and p.get("sql") == sql for p in events
        ):
            panels.append(Text(f"  {title}", style=style))
            continue
        panels.append(
            Panel(Syntax(sql, "sql", word_wrap=True), title=Text(title), border_style=style)
        )
    return panels


def _payload_panels(payloads: list[Event]) -> list[RenderableType]:
    panels: list[RenderableType] = []
    for p in payloads:
        lines: list[str] = []
        for m in p.get("request") or []:
            head = f"[{m.get('type')}{' ' + m['name'] if m.get('name') else ''}]"
            lines.append(f"{head} {m.get('text', '')}")
            for call in m.get("tool_calls") or []:
                lines.append(
                    f"  → {call.get('name')}({json.dumps(call.get('args'), ensure_ascii=False)})"
                )
        lines.append("")
        if "response" in p:
            response = p["response"] or {}
            lines.append(f"[response] {response.get('text', '')}")
            for call in response.get("tool_calls") or []:
                lines.append(
                    f"  → {call.get('name')}({json.dumps(call.get('args'), ensure_ascii=False)})"
                )
        else:
            lines.append(f"[error] {p.get('error')}")
        title = (
            f"LLM call {p.get('call_id')}: {p.get('model')} [{p.get('purpose')}], "
            f"{p.get('latency_ms')} ms"
        )
        panels.append(Panel(Text("\n".join(lines)), title=Text(title), border_style="magenta"))
    return panels


def render_trace(
    trace_id: str,
    events: list[Event],
    payloads: list[Event] | None = None,
    langsmith_url: str | None = None,
) -> RenderableType:
    if not events:
        return Text(f"No events for trace {trace_id}.", style="yellow")
    first = events[0]
    started = _ts(first)
    header = Text.assemble(
        ("trace ", "bold"),
        (trace_id, "bold cyan"),
        f"  user={first.get('user_id')}  conversation={str(first.get('conversation_id'))[:12]}",
        f"  {str(first.get('ts'))[:19].replace('T', ' ')} UTC",
    )
    table = Table(show_lines=False, expand=True)
    table.add_column("+ms", justify="right", no_wrap=True)
    table.add_column("event", no_wrap=True)
    table.add_column("details", overflow="fold")
    for e in events:
        ts = _ts(e)
        offset = int((ts - started).total_seconds() * 1000) if ts and started else 0
        table.add_row(
            str(offset),
            Text(str(e.get("event")), style=LEVEL_STYLES.get(str(e.get("level")), "")),
            Text(describe(e)),
        )
    parts: list[RenderableType] = [header, table, *_sql_panels(events)]
    if langsmith_url:
        parts.append(Text(f"LangSmith: {langsmith_url}", style="blue"))
    if payloads is not None:
        if payloads:
            parts.append(Text("Model conversation (llm_payloads.jsonl):", style="bold"))
            parts.extend(_payload_panels(payloads))
        else:
            parts.append(
                Text("No model payloads for this trace (LOG_LLM_PAYLOADS is off?).", style="yellow")
            )
    return Group(*parts)


def langsmith_run_id(events: list[Event]) -> str | None:
    for e in events:
        if e.get("event") == "langsmith_run" and e.get("run_id"):
            return str(e["run_id"])
    return None


# -- /stats ---------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass
class Stats:
    turns: int = 0
    statuses: Counter = field(default_factory=Counter)
    error_kinds: Counter = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)
    llm_calls: int = 0
    llm_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    bq_bytes: int = 0
    sql_executed: int = 0
    sql_failed_attempts: int = 0
    sql_rejected: Counter = field(default_factory=Counter)
    turns_with_sql: int = 0
    turns_with_correction: int = 0
    corrections_succeeded: int = 0
    guard_blocked: Counter = field(default_factory=Counter)
    guard_degraded: int = 0
    pii_requests: int = 0
    pii_masked: Counter = field(default_factory=Counter)
    acl_queries: int = 0
    fallbacks: int = 0
    llm_retries: int = 0
    bq_retries: int = 0
    confirmations: Counter = field(default_factory=Counter)
    feedback: Counter = field(default_factory=Counter)

    @property
    def success_rate(self) -> float | None:
        done = self.statuses["ok"] + self.statuses["blocked"]
        return done / self.turns if self.turns else None

    @property
    def correction_rate(self) -> float | None:
        return self.turns_with_correction / self.turns_with_sql if self.turns_with_sql else None

    @property
    def correction_success_rate(self) -> float | None:
        if not self.turns_with_correction:
            return None
        return self.corrections_succeeded / self.turns_with_correction


def compute_stats(events: Iterable[Event], settings: Settings, user_id: str | None = None) -> Stats:
    s = Stats()
    for e in events:
        if user_id is not None and e.get("user_id") != user_id:
            continue
        name = e.get("event")
        if name == "turn_end":
            s.turns += 1
            status = str(e.get("status"))
            s.statuses[status] += 1
            if status == "error":
                s.error_kinds[str(e.get("error_kind"))] += 1
            if status in ("ok", "blocked") and isinstance(e.get("latency_ms"), int | float):
                s.latencies_ms.append(e["latency_ms"])
            s.bq_bytes += int(e.get("bq_bytes") or 0)
            failures = int(e.get("sql_failures") or 0)
            executed = int(e.get("sql_count") or 0)
            if failures or executed:
                s.turns_with_sql += 1
            if failures:
                s.turns_with_correction += 1
                if executed and status == "ok":
                    s.corrections_succeeded += 1
        elif name == "llm_call":
            s.llm_calls += 1
            s.input_tokens += int(e.get("input_tokens") or 0)
            s.output_tokens += int(e.get("output_tokens") or 0)
        elif name == "llm_error":
            s.llm_errors += 1
        elif name == "sql_executed":
            s.sql_executed += 1
        elif name == "sql_attempt_failed":
            s.sql_failed_attempts += 1
        elif name == "sql_rejected":
            s.sql_rejected[str(e.get("rule"))] += 1
        elif name == "input_guard" and e.get("pii_request"):
            s.pii_requests += 1
        elif name == "guard_blocked":
            s.guard_blocked[str(e.get("category"))] += 1
        elif name == "guard_degraded":
            s.guard_degraded += 1
        elif name == "pii_masked":
            s.pii_masked[str(e.get("where"))] += int(e.get("count") or 1)
        elif name == "run_sql" and e.get("acl"):
            s.acl_queries += 1
        elif name == "fallback_used":
            s.fallbacks += 1
        elif name == "llm_retry":
            s.llm_retries += 1
        elif name == "bq_retry":
            s.bq_retries += 1
        elif isinstance(name, str) and name.startswith("confirmation_"):
            s.confirmations[name.removeprefix("confirmation_")] += 1
        elif name == "feedback":
            s.feedback[str(e.get("rating"))] += 1
    s.cost_usd = (
        s.input_tokens * settings.llm_price_per_1m_input
        + s.output_tokens * settings.llm_price_per_1m_output
    ) / 1e6
    return s


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _counts(counter: Counter) -> str:
    return ", ".join(f"{k}: {v}" for k, v in counter.most_common()) or "-"


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value / 1000:.1f} s"


def render_stats(stats: Stats, title: str = "Agent statistics") -> RenderableType:
    s = stats
    table = Table(title=title, show_header=False, show_lines=False)
    table.add_column("metric", style="bold")
    table.add_column("value", overflow="fold")
    rows = [
        ("Turns", f"{s.turns} ({_counts(s.statuses)})"),
        ("Success rate", _pct(s.success_rate)),
        ("Errors by kind", _counts(s.error_kinds)),
        (
            "Latency p50 / p95",
            f"{_ms(percentile(s.latencies_ms, 50))} / {_ms(percentile(s.latencies_ms, 95))}",
        ),
        ("LLM calls / errors", f"{s.llm_calls} / {s.llm_errors}"),
        ("Tokens in / out", f"{s.input_tokens:,} / {s.output_tokens:,}"),
        ("Cost estimate", f"${s.cost_usd:.4f}"),
        ("BigQuery bytes", _bytes(s.bq_bytes)),
        ("SQL executed / failed attempts", f"{s.sql_executed} / {s.sql_failed_attempts}"),
        (
            "SQL self-correction",
            f"{_pct(s.correction_rate)} of turns with SQL needed a fix; "
            f"{_pct(s.correction_success_rate)} of them succeeded",
        ),
        ("SQL rejected by rule", _counts(s.sql_rejected)),
        ("Guard blocked", _counts(s.guard_blocked)),
        ("Guard degraded (rules only)", str(s.guard_degraded)),
        ("PII requests / masked", f"{s.pii_requests} / {_counts(s.pii_masked)}"),
        ("Category-ACL queries", str(s.acl_queries)),
        (
            "Fallback / LLM retries / BQ retries",
            f"{s.fallbacks} / {s.llm_retries} / {s.bq_retries}",
        ),
        ("Confirmations", _counts(s.confirmations)),
        ("Feedback", _counts(s.feedback)),
    ]
    for name, value in rows:
        table.add_row(name, value)
    return table


# -- standalone -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m retail_agent.observability.inspect",
        description="Inspect the agent's event log.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    trace = sub.add_parser("trace", help="timeline of one turn")
    trace.add_argument("trace_id", nargs="?", help="trace id or prefix (default: the last turn)")
    trace.add_argument("--full", action="store_true", help="also show the model conversation")
    trace.add_argument("--user", help="only this user's turns")
    stats = sub.add_parser("stats", help="aggregate metrics")
    stats.add_argument("--user", help="only this user's events")
    args = parser.parse_args(argv)

    settings = get_settings()
    console = Console()
    events = list(read_jsonl(settings.log_path))
    if args.command == "stats":
        title = f"Agent statistics ({args.user})" if args.user else "Agent statistics"
        console.print(render_stats(compute_stats(events, settings, args.user), title))
        return 0

    if args.trace_id:
        trace_id = resolve_trace_id(events, args.trace_id, args.user)
    else:
        trace_id = last_trace_id(events, args.user)
    if trace_id is None:
        console.print(f"[yellow]No unique trace matches {args.trace_id or '(last)'}.[/yellow]")
        return 1
    turn = trace_events(events, trace_id)
    payloads = trace_payloads(settings.llm_payload_log_path, trace_id) if args.full else None
    url = None
    run_id = langsmith_run_id(turn)
    if run_id is not None:
        from retail_agent.observability import tracing

        if tracing.setup_langsmith(settings).enabled:
            url = tracing.run_url(run_id)
    console.print(render_trace(trace_id, turn, payloads, url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
