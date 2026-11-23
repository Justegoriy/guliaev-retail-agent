"""Run the eval cases against the real agent (Gemini + BigQuery) and save the results.

    python evals/run_eval.py                      # all cases
    python evals/run_eval.py --only security      # one or more categories
    python evals/run_eval.py --case a2_top_categories --no-judge

Each run uses its own SQLite databases and log under `evals/results/runs/<timestamp>/`, so it
never touches real users' reports, preferences or conversations. The summary is written to
`evals/results/<timestamp>.json` for comparing versions.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from retail_agent import auth, health
from retail_agent.console import tolerate_unencodable_output
from retail_agent.evaluation.cases import CaseError, load_cases
from retail_agent.evaluation.runner import CaseResult, EvalEnv, JudgeVerdict, run_case
from retail_agent.graph.builder import create_resources
from retail_agent.llm import build_chat_model
from retail_agent.observability import inspect, logger, tracing
from retail_agent.personalization.profile import load_config
from retail_agent.settings import PROJECT_ROOT, get_settings

EVALS = PROJECT_ROOT / "evals"
console = Console()


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cases", type=Path, default=EVALS / "cases.yaml")
    p.add_argument("--only", action="append", default=[], help="category (repeatable, or a,b)")
    p.add_argument("--case", action="append", default=[], help="case id (repeatable)")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    p.add_argument("--pause", type=float, default=2.0, help="seconds between cases")
    p.add_argument("--out", type=Path, default=EVALS / "results")
    return p.parse_args(argv)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _models_used(payload_log: Path) -> dict[str, dict[str, int]]:
    """purpose -> model -> answered calls: shows how much went to the fallback model."""
    used: dict[str, Counter] = {}
    for record in inspect.read_jsonl(payload_log):
        if record.get("response") is not None:
            used.setdefault(record.get("purpose", "?"), Counter())[record.get("model", "?")] += 1
    return {purpose: dict(models) for purpose, models in used.items()}


def _summary(results: list[CaseResult], payload_log: Path) -> dict:
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        c = by_category.setdefault(r.category, {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += r.passed
    judged = [r.judge["overall"] for r in results if r.judge and "overall" in r.judge]
    with_reference = [r for r in results if r.accuracy is not None]
    return {
        "total": len(results),
        "passed": sum(r.passed for r in results),
        "by_category": by_category,
        "judge_mean": round(sum(judged) / len(judged), 2) if judged else None,
        "judge_unavailable": sum(1 for r in results if r.judge and "error" in r.judge),
        "models_used": _models_used(payload_log),
        "accuracy_passed": sum(r.accuracy["passed"] for r in with_reference),
        "accuracy_total": len(with_reference),
        "llm_calls": sum(r.llm_calls for r in results),
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
    }


def _render(results: list[CaseResult], summary: dict) -> None:
    table = Table(title="Eval results", show_lines=False)
    for column in ("case", "category", "pass", "accuracy", "judge", "s", "failures"):
        table.add_column(column, overflow="fold")
    for r in results:
        accuracy = "-" if r.accuracy is None else f"{r.accuracy['share']:.0%}"
        if r.judge is None:
            judge = "-"
        else:
            judge = f"{r.judge['overall']}/5" if "overall" in r.judge else "n/a"
        table.add_row(
            r.id,
            r.category,
            Text("PASS", style="green") if r.passed else Text("FAIL", style="red"),
            accuracy,
            judge,
            f"{r.latency_ms / 1000:.0f}",
            Text("; ".join(r.failures)[:300]),
        )
    console.print(table)
    parts = [f"{cat} {c['passed']}/{c['total']}" for cat, c in summary["by_category"].items()]
    console.print(f"Passed {summary['passed']}/{summary['total']}: " + ", ".join(parts))
    if summary["judge_mean"] is not None:
        console.print(f"Judge mean: {summary['judge_mean']}/5")
    if summary["judge_unavailable"]:
        console.print(
            f"[yellow]Judge unavailable for {summary['judge_unavailable']} case(s)[/yellow]"
        )
    for purpose, models in summary["models_used"].items():
        console.print(
            f"Model calls ({purpose}): " + ", ".join(f"{m} {n}" for m, n in models.items())
        )
    console.print(
        f"Execution accuracy: {summary['accuracy_passed']}/{summary['accuracy_total']}; "
        f"LLM calls {summary['llm_calls']}, tokens in {summary['input_tokens']:,} / "
        f"out {summary['output_tokens']:,}"
    )


def main(argv: list[str] | None = None) -> int:
    tolerate_unencodable_output()
    args = _args(argv)
    base = get_settings()
    users = auth.load_users()
    try:
        cases = load_cases(
            args.cases,
            base.golden_trios_path,
            users=set(users),
            preference_keys=set(load_config(base.preferences_config_path).keys),
        )
    except CaseError as exc:
        console.print(f"[red]Invalid cases: {exc}[/red]")
        return 2
    only = {c for item in args.only for c in item.split(",") if c}
    cases = [
        c
        for c in cases
        if (not only or c.category in only) and (not args.case or c.id in args.case)
    ]
    if not cases:
        console.print("[yellow]No cases selected.[/yellow]")
        return 2

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = args.out / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = base.model_copy(
        update={
            "app_db_path": run_dir / "app.db",
            "checkpoint_db_path": run_dir / "checkpoints.db",
            "log_path": run_dir / "agent.jsonl",
            "golden_candidates_path": run_dir / "golden_candidates.jsonl",
            "exports_dir": run_dir / "exports",
        }
    )
    logger.configure(settings.log_path)
    tracing.setup_langsmith(settings)
    resources = create_resources(settings)
    checks = [health.check_gemini(settings), health.check_bigquery(resources.runner, settings)]
    for check in checks:
        if not check.ok:
            console.print(f"[{'red' if check.fatal else 'yellow'}]{check.name}: {check.message}")
    if any(c.fatal for c in checks):
        return 2

    judge = None
    if not args.no_judge:
        model = build_chat_model("main", settings=settings, structured_output=JudgeVerdict)
        judge = model.invoke
    env = EvalEnv(resources.session, resources.runner, users, judge, settings.log_path)

    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        console.print(f"[{i}/{len(cases)}] {case.id} ({case.category}, {case.user})...", end=" ")
        result = run_case(case, env)
        results.append(result)
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(f"{status} {result.latency_ms / 1000:.0f} s")
        if i < len(cases):
            time.sleep(args.pause)

    summary = _summary(results, settings.llm_payload_log_path)
    _render(results, summary)
    out = args.out / f"{stamp}.json"
    payload = {
        "started": stamp,
        "commit": _git_commit(),
        "models": {
            "main": settings.gemini_model,
            "fallback": settings.gemini_fallback_model,
            "aux": settings.gemini_aux_model,
            "judge": None if args.no_judge else settings.gemini_model,
        },
        "filters": {"only": sorted(only), "case": args.case},
        "summary": summary,
        "cases": [r.to_dict() for r in results],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    console.print(
        f"Saved {out.relative_to(PROJECT_ROOT)} (run data: {run_dir.relative_to(PROJECT_ROOT)})"
    )
    security = summary["by_category"].get("security")
    return 1 if security and security["passed"] < security["total"] else 0


if __name__ == "__main__":
    sys.exit(main())
