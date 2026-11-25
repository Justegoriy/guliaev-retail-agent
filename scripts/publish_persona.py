"""Publish the persona (the analyst's tone) through the automated quality gate.

    python scripts/publish_persona.py check            # static checks of the draft only
    python scripts/publish_persona.py publish          # checks + eval gate, then publish
    python scripts/publish_persona.py versions         # published versions
    python scripts/publish_persona.py rollback [ID]    # restore an earlier version

Edit `config/persona_draft.md`, then publish. Nothing is approved by hand: the draft goes
live only if it passes the checks and the `persona` eval suite (`evals/gate.yaml`). The
agent re-reads the active persona on the next turn, so no restart or redeploy is needed.
"""

import argparse
import subprocess
import sys

from rich.console import Console

from retail_agent.console import tolerate_unencodable_output
from retail_agent.observability import logger
from retail_agent.prompts.persona_gate import PersonaPublisher, static_problems
from retail_agent.settings import CONFIG_DIR, PROJECT_ROOT, get_settings

console = Console()
DRAFT = CONFIG_DIR / "persona_draft.md"
VERSIONS = CONFIG_DIR / "persona_versions"


def run_gate(draft) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evals" / "run_eval.py"),
        "--gate",
        "--suite",
        "persona",
        "--persona",
        str(draft),
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    tolerate_unencodable_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["check", "publish", "versions", "rollback"])
    parser.add_argument("version", nargs="?", help="version id for rollback")
    args = parser.parse_args(argv)
    settings = get_settings()
    logger.configure(settings.log_path)
    publisher = PersonaPublisher(settings.persona_path, DRAFT, VERSIONS, run_gate)

    if args.command == "check":
        problems = static_problems(DRAFT.read_text(encoding="utf-8"))
        for problem in problems:
            console.print(f"[red]- {problem}[/red]")
        console.print("[green]Static checks passed.[/green]" if not problems else "")
        return 1 if problems else 0
    if args.command == "versions":
        for path in publisher.versions():
            console.print(path.stem)
        return 0
    result = publisher.rollback(args.version) if args.command == "rollback" else publisher.publish()
    colour = {"published": "green", "unchanged": "cyan", "inconclusive": "yellow"}.get(
        result.status, "red"
    )
    console.print(f"[bold {colour}]{result.status.upper()}[/bold {colour}] {result.version or ''}")
    for reason in result.reasons:
        console.print(f"  - {reason}")
    return {"published": 0, "unchanged": 0, "inconclusive": 3}.get(result.status, 1)


if __name__ == "__main__":
    sys.exit(main())
