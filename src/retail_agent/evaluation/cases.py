"""The case format of `evals/cases.yaml`, validated strictly so a typo fails the run early."""

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Category = Literal[
    "structure",
    "analytics",
    "security",
    "resilience",
    "report",
    "personalization",
    "multi_turn",
    "deletion",
]
Format = Literal["table", "bullets", "prose"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Expect(_Strict):
    """Deterministic expectations for one turn; empty fields are not checked."""

    blocked: bool | None = None  # the input guard refused the message
    refusal: bool | None = None  # blocked, or answered without executing any SQL
    tools_called: list[str] = Field(default_factory=list)
    tools_not_called: list[str] = Field(default_factory=list)
    min_sql: int | None = None  # executed queries (run_sql and benchmark)
    sql_contains: list[str] = Field(default_factory=list)  # each in some executed SQL
    must_contain_any: list[str] = Field(default_factory=list)  # case-insensitive
    must_contain_all: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    must_not_match: list[str] = Field(default_factory=list)  # regexes over the answer
    no_data_about: list[str] = Field(default_factory=list)  # never in run_sql result rows
    format: Format | None = None
    max_words: int | None = None
    confirmation: bool | None = None  # a confirmation was requested (the eval answers "no")


class TurnSpec(_Strict):
    text: str
    expect: Expect = Field(default_factory=Expect)


class ReportSetup(_Strict):
    """A report saved before the case starts; `{r0}`, `{r1}`... in turn texts are its ids."""

    title: str
    content: str = "Revenue summary."
    owner: str | None = None  # default: the case user
    this_conversation: bool = True


class Reference(_Strict):
    """Execution accuracy: numbers of `column` (first `rows` rows) must appear in the answer."""

    sql: str  # SQL, or `golden:<trio id>` for the trio's (first) query
    column: str
    rows: int = 1
    min_match: float = 1.0  # share of the reference numbers that must be found


class JudgeSpec(_Strict):
    intent: str  # what a good answer does, for the judge
    action_items: bool = False


class Case(_Strict):
    id: str
    category: Category
    user: str = "ceo"
    description: str = ""
    profile: dict[str, str] = Field(default_factory=dict)  # explicit preferences set first
    reports: list[ReportSetup] = Field(default_factory=list)
    turns: list[TurnSpec]
    expect: Expect = Field(default_factory=Expect)  # for the last turn
    profile_after: dict[str, str | None] = Field(default_factory=dict)  # None: key unset
    reports_after: int | None = None  # the case user's live reports at the end
    reference: Reference | None = None
    judge: JudgeSpec | None = None

    @field_validator("turns", mode="before")
    @classmethod
    def _plain_strings(cls, value):
        if isinstance(value, list):
            return [{"text": t} if isinstance(t, str) else t for t in value]
        return value

    @model_validator(mode="after")
    def _has_turns(self):
        if not self.turns:
            raise ValueError(f"case {self.id}: at least one turn is needed")
        return self


class CaseError(ValueError):
    pass


def _golden_sql(golden_path: Path) -> dict[str, str]:
    trios = json.loads(golden_path.read_text(encoding="utf-8"))
    out = {}
    for trio in trios:
        sql = trio["sql"]
        out[trio["id"]] = sql[0] if isinstance(sql, list) else sql
    return out


def load_cases(
    path: Path,
    golden_path: Path,
    users: set[str] | None = None,
    preference_keys: set[str] | None = None,
) -> list[Case]:
    """Parse and validate cases; `golden:<id>` references are resolved to SQL."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("cases", raw) if isinstance(raw, dict) else raw
    try:
        cases = [Case.model_validate(item) for item in items]
    except ValueError as exc:
        raise CaseError(str(exc)) from exc
    golden = _golden_sql(golden_path)
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise CaseError(f"duplicate case id {case.id}")
        seen.add(case.id)
        owners = {case.user, *(r.owner for r in case.reports if r.owner)}
        if users is not None and not owners <= users:
            raise CaseError(f"case {case.id}: unknown user in {sorted(owners - users)}")
        keys = set(case.profile) | set(case.profile_after)
        if preference_keys is not None and not keys <= preference_keys:
            raise CaseError(
                f"case {case.id}: unknown preference key {sorted(keys - preference_keys)}"
            )
        ref = case.reference
        if ref is not None and ref.sql.startswith("golden:"):
            trio_id = ref.sql.removeprefix("golden:")
            if trio_id not in golden:
                raise CaseError(f"case {case.id}: no golden trio {trio_id}")
            ref.sql = golden[trio_id]
    return cases
