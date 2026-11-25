"""Check Golden Bucket candidates automatically and approve, quarantine or reject them.

    python scripts/promote_candidates.py           # process the queue
    python scripts/promote_candidates.py --dry-run # show the queue, change nothing

In production this is a worker triggered by new candidates (Pub/Sub); here it is run on
demand. See `retail_agent.knowledge.promotion` for the checks. New trios are used from the
agent's next start; quarantined candidates and a sample of approved trios go to
`scripts/review_candidates.py` for the audit.
"""

import argparse
import sys

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.table import Table

from retail_agent.auth import UserContext
from retail_agent.console import tolerate_unencodable_output
from retail_agent.data.bigquery import BigQueryRunner
from retail_agent.graph.builder import _aux_factory, load_golden_index
from retail_agent.knowledge.golden import CandidateStore
from retail_agent.knowledge.promotion import CandidateReview, Promoter
from retail_agent.observability import logger
from retail_agent.safety import sql_guard
from retail_agent.settings import get_settings

console = Console()


def main(argv: list[str] | None = None) -> int:
    tolerate_unencodable_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="list the queue and exit")
    args = parser.parse_args(argv)
    settings = get_settings()
    logger.configure(settings.log_path)
    store = CandidateStore(settings.golden_candidates_path)
    pending = store.pending()
    if args.dry_run or not pending:
        console.print(f"{len(pending)} candidate(s) in the queue.")
        for c in pending:
            console.print(f"  {c.id}  {c.user_role:<8} {c.question}")
        return 0

    runner = BigQueryRunner.from_settings(settings)

    def check_sql(sql: str, author: UserContext) -> str:
        guarded = sql_guard.check_and_rewrite(
            sql, author, dataset=runner.dataset_id, default_limit=settings.sql_default_limit
        )
        runner.dry_run(guarded.sql)
        return guarded.sql

    judge = _aux_factory(settings)(CandidateReview) if settings.google_api_key else None
    promoter = Promoter(
        store=store,
        trios_path=settings.golden_trios_path,
        check_sql=check_sql,
        run_sql=lambda sql: runner.execute(sql).df,
        judge=(lambda prompt: judge.invoke([HumanMessage(prompt)])) if judge else None,
        index=load_golden_index(settings),
    )
    outcomes = promoter.run()
    table = Table(title="Golden Bucket promotion")
    for column in ("candidate", "decision", "reason", "trio"):
        table.add_column(column)
    for o in outcomes:
        table.add_row(o.candidate_id, o.decision, o.reason, o.trio_id or "")
    console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
