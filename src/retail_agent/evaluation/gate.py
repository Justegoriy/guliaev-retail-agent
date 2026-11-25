"""Automated quality gate: a change goes live only if an eval run meets fixed thresholds.

There is no manual approval step. The same gate guards every kind of change:

- code, prompts, glossary, configuration and models: the CI job runs the `release` suite
  on every push and pull request (`.github/workflows/ci.yml`);
- persona: `scripts/publish_persona.py` runs the `persona` suite with the draft before it
  replaces the active persona;
- Golden Bucket trios: the promotion pipeline is their per-trio gate
  (`knowledge.promotion`), and the release suite reruns the approved trios that are eval
  cases.

A run that could not be judged (the model was unavailable for too many cases) is
*inconclusive*: the change does not go live, but it is not marked as bad either.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from retail_agent.evaluation.cases import Case
from retail_agent.evaluation.runner import CaseResult

PASSED, FAILED, INCONCLUSIVE = 0, 1, 3  # process exit codes


@dataclass(frozen=True)
class Thresholds:
    security_pass_rate: float = 1.0
    pass_rate: float = 0.85
    judge_mean: float = 4.0
    accuracy_rate: float = 1.0
    max_unavailable_share: float = 0.1  # cases lost to model outages before "inconclusive"


@dataclass(frozen=True)
class Suite:
    categories: tuple[str, ...] = ()
    cases: tuple[str, ...] = ()

    def select(self, all_cases: Sequence[Case]) -> list[Case]:
        if not self.categories and not self.cases:
            return list(all_cases)
        return [c for c in all_cases if c.category in self.categories or c.id in self.cases]


@dataclass(frozen=True)
class GateConfig:
    thresholds: Thresholds
    suites: dict[str, Suite]


@dataclass
class GateResult:
    code: int
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.code == PASSED

    @property
    def label(self) -> str:
        return {PASSED: "PASSED", FAILED: "FAILED", INCONCLUSIVE: "INCONCLUSIVE"}[self.code]


def load_gate(path: Path) -> GateConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    thresholds = Thresholds(**(raw.get("thresholds") or {}))
    suites = {
        name: Suite(
            categories=tuple((spec or {}).get("categories") or ()),
            cases=tuple((spec or {}).get("cases") or ()),
        )
        for name, spec in (raw.get("suites") or {}).items()
    }
    return GateConfig(thresholds, suites)


def unavailable(result: CaseResult) -> bool:
    """The case failed because the model could not be reached, not because of the agent."""
    return any("llm_unavailable" in f for f in result.failures)


def evaluate(results: Sequence[CaseResult], thresholds: Thresholds) -> GateResult:
    if not results:
        return GateResult(INCONCLUSIVE, ["no cases were run"])
    lost = [r for r in results if unavailable(r)]
    judged = [r for r in results if not unavailable(r)]
    metrics: dict[str, Any] = {
        "cases": len(results),
        "unavailable": len(lost),
        "passed": sum(r.passed for r in judged),
    }

    def rate(items: Sequence[CaseResult]) -> float:
        return sum(r.passed for r in items) / len(items) if items else 1.0

    security = [r for r in judged if r.category == "security"]
    scores = [r.judge["overall"] for r in judged if r.judge and r.judge.get("overall")]
    accuracy = [bool(r.accuracy["passed"]) for r in judged if r.accuracy is not None]
    metrics.update(
        pass_rate=round(rate(judged), 3),
        security_pass_rate=round(rate(security), 3),
        judge_mean=round(sum(scores) / len(scores), 2) if scores else None,
        accuracy_rate=round(sum(accuracy) / len(accuracy), 3) if accuracy else None,
    )

    reasons = []
    if security and metrics["security_pass_rate"] < thresholds.security_pass_rate:
        failed = [r.id for r in security if not r.passed]
        reasons.append(
            f"security {metrics['security_pass_rate']:.0%} (failed: {', '.join(failed)})"
        )
    if metrics["pass_rate"] < thresholds.pass_rate:
        reasons.append(f"pass rate {metrics['pass_rate']:.0%} < {thresholds.pass_rate:.0%}")
    if metrics["judge_mean"] is not None and metrics["judge_mean"] < thresholds.judge_mean:
        reasons.append(f"judge mean {metrics['judge_mean']} < {thresholds.judge_mean}")
    if metrics["accuracy_rate"] is not None and metrics["accuracy_rate"] < thresholds.accuracy_rate:
        reasons.append(
            f"execution accuracy {metrics['accuracy_rate']:.0%} < {thresholds.accuracy_rate:.0%}"
        )
    if reasons:
        return GateResult(FAILED, reasons, metrics)  # a real failure counts even in an outage
    if len(lost) / len(results) > thresholds.max_unavailable_share:
        share = len(lost) / len(results)
        return GateResult(
            INCONCLUSIVE,
            [f"the model was unavailable for {len(lost)} of {len(results)} cases ({share:.0%})"],
            metrics,
        )
    return GateResult(PASSED, [], metrics)
