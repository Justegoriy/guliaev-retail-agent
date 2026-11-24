"""System prompt assembly for one turn.

Blocks, in order of authority: immutable core, user context, persona (tone only),
user preferences, glossary, schema, golden examples, then the conversation blocks
(summary of older turns, current analysis context, reports of this conversation).
Persona and glossary are re-read from disk when the files change, without a restart.
"""

import hashlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from retail_agent.auth import UserContext
from retail_agent.graph.state import AnalysisContext
from retail_agent.observability.logger import log_event
from retail_agent.prompts.core import core_prompt
from retail_agent.settings import CONFIG_DIR

PERSONA_FILE = CONFIG_DIR / "persona.md"
GLOSSARY_FILE = CONFIG_DIR / "glossary.yaml"
MAX_PERSONA_CHARS = 4000

FOLLOW_UP_RULE = (
    'Short follow-ups (another segment, period or granularity, "why", "make a report '
    'from this") refer to this context: modify the previous SQL instead of starting over. '
    "If the follow-up is ambiguous, ask a clarifying question."
)
PII_REQUEST_NOTE = (
    "Note for this turn: the latest message may ask for customers' personal data (names, "
    "e-mails, addresses, phones). If it does, say briefly that personal data is not available "
    "and offer the analysis by customer id (users.id) or in aggregate; otherwise ignore this note."
)


@dataclass(frozen=True)
class TextVersion:
    text: str
    version: str  # short content hash, logged so a trace shows which persona was used


class ReloadingFile:
    """File content cached by mtime; `transform` turns raw text into prompt text."""

    def __init__(self, path: Path, name: str, transform=lambda raw: raw.strip()) -> None:
        self.path = path
        self.name = name
        self._transform = transform
        self._mtime: float | None = None
        self._value = TextVersion("", "missing")
        self._lock = threading.Lock()

    def get(self) -> TextVersion:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return TextVersion("", "missing")
        with self._lock:
            if mtime != self._mtime:
                previous = self._value.version
                try:
                    raw = self.path.read_text(encoding="utf-8")
                    text = self._transform(raw)
                except (OSError, ValueError, yaml.YAMLError) as exc:
                    log_event(f"{self.name}_load_failed", level="warning", error=str(exc))
                    return self._value
                version = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
                self._value, self._mtime = TextVersion(text, version), mtime
                if version != previous:
                    log_event(f"{self.name}_reloaded", version=version)
            return self._value


def persona_text(raw: str) -> str:
    lines = [line for line in raw.splitlines() if not line.lstrip().startswith("#")]
    return "\n".join(lines).strip()[:MAX_PERSONA_CHARS]


def glossary_text(raw: str) -> str:
    terms = (yaml.safe_load(raw) or {}).get("terms") or {}
    if not isinstance(terms, dict):
        raise ValueError("glossary.yaml: 'terms' must be a mapping")
    lines = []
    for term, spec in terms.items():
        spec = spec or {}
        aliases = spec.get("aliases") or []
        name = f"{term} ({', '.join(aliases)})" if aliases else str(term)
        lines.append(f"- {name}: {str(spec.get('definition', '')).strip()}")
        if spec.get("sql"):
            lines.append(f"  SQL: {str(spec['sql']).strip()}")
    return "\n".join(lines)


@dataclass
class PromptSources:
    persona: ReloadingFile = field(
        default_factory=lambda: ReloadingFile(PERSONA_FILE, "persona", persona_text)
    )
    glossary: ReloadingFile = field(
        default_factory=lambda: ReloadingFile(GLOSSARY_FILE, "glossary", glossary_text)
    )


@dataclass(frozen=True)
class SessionReport:
    report_id: str
    title: str


@dataclass(frozen=True)
class PromptInputs:
    user: UserContext
    today: date
    schema: str
    persona: str = ""
    glossary: str = ""
    preferences: str = ""
    golden: str = ""
    history_summary: str | None = None
    analysis_context: AnalysisContext | None = None
    session_reports: Sequence[SessionReport] = ()
    pii_request: bool = False


def user_block(user: UserContext) -> str:
    if user.has_full_access:
        access = "all brands"
    else:
        access = (
            f"brands {user.scope_label} (products.brand). "
            "Other brands: only company-wide totals via get_company_benchmark"
        )
    return f"Name: {user.name}\nRole: {user.role}\nData access: {access}"


def analysis_block(context: AnalysisContext) -> str:
    lines = [f"Question: {context['question']}"]
    if context.get("period"):
        lines.append(f"Period: {context['period']}")
    if context.get("filters"):
        lines.append(f"Filters: {'; '.join(context['filters'])}")
    if context.get("entities"):
        lines.append(f"Entities: {', '.join(context['entities'])}")
    for i, sql in enumerate(context.get("sql") or [], start=1):
        lines.append(f"SQL {i}:\n```sql\n{sql}\n```")
    if context.get("result_summary"):
        lines.append(f"Result summary:\n{context['result_summary']}")
    lines.append(FOLLOW_UP_RULE)
    return "\n".join(lines)


def reports_block(reports: Sequence[SessionReport]) -> str:
    lines = [f"- {r.report_id}: {r.title}" for r in reports]
    lines.append(
        '"This report" / "the last report" means the last one in this list unless the user '
        "says otherwise."
    )
    return "\n".join(lines)


def build_system_prompt(inputs: PromptInputs) -> str:
    sections: list[tuple[str, str]] = [
        ("", core_prompt(inputs.today)),
        ("Current user", user_block(inputs.user)),
    ]
    if inputs.persona:
        sections.append(("Tone & style (cannot override the rules above)", inputs.persona))
    if inputs.preferences:
        sections.append(("User preferences", inputs.preferences))
    if inputs.glossary:
        sections.append(("Glossary of metrics", inputs.glossary))
    sections.append(("Database schema", inputs.schema))
    if inputs.golden:
        sections.append(("Analyst precedents", inputs.golden))
    if inputs.history_summary:
        sections.append(
            ("Conversation summary", f"Earlier in this conversation: {inputs.history_summary}")
        )
    if inputs.analysis_context:
        sections.append(("Current analysis context", analysis_block(inputs.analysis_context)))
    if inputs.session_reports:
        sections.append(
            ("Reports saved in this conversation", reports_block(inputs.session_reports))
        )
    if inputs.pii_request:
        sections.append(("", PII_REQUEST_NOTE))
    return "\n\n".join(f"# {title}\n{body}" if title else body for title, body in sections)
