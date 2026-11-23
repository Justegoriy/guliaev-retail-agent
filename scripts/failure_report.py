"""Where the agent struggles: groups failed turns from the logs (system-level learning).

    python scripts/failure_report.py                 # all of logs/agent.jsonl
    python scripts/failure_report.py --days 7 --top 20
    python scripts/failure_report.py --user bottoms_manager
    python scripts/failure_report.py --log evals/results/runs/<ts>/agent.jsonl

Counts failed turns, SQL errors by type, exhausted SQL attempts, SQL Guard rejections,
input-guard blocks, fallbacks and /bad (thumbs-down) feedback, and lists the worst turns with their
question and trace id (open one with `python -m retail_agent.observability.inspect trace`).
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rich.console import Console

from retail_agent.console import tolerate_unencodable_output
from retail_agent.observability import failures, inspect
from retail_agent.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    tolerate_unencodable_output()
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--log", type=Path, default=settings.log_path)
    p.add_argument("--days", type=float, help="only the last N days")
    p.add_argument("--user", help="only this user's turns")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args(argv)
    if not args.log.exists():
        print(f"No log at {args.log}", file=sys.stderr)
        return 2
    since = datetime.now(UTC) - timedelta(days=args.days) if args.days else None
    report = failures.analyze(
        inspect.read_jsonl(args.log),
        inspect.read_jsonl(args.log.with_name("llm_payloads.jsonl")),
        since=since,
        user_id=args.user,
        max_sql_failures=settings.max_sql_failures,
    )
    Console().print(failures.render(report, top=args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
