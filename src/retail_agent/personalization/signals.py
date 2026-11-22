"""Preference signals gathered after a turn.

- From the user's words (one post-turn aux-model call shared with the analysis context
  extraction): "shorter, please", "as a table" -> `reformat_request`.
- Deterministic, without a model: metrics and segments of the analyses the user runs
  (`usage`), and `/good` / `/bad [comment]` on an answer (`feedback`).
"""

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from retail_agent.personalization.profile import PreferenceConfig, Signal

SIGNAL_EXTRACTION_RULES = """\
Preference signals: lasting tastes the latest user message reveals about how answers \
should look. Keys and values:
{keys}
Return a signal only when the user asks to change the form of an answer in a way that \
reflects their taste, e.g. "shorter, please" -> detail_level=brief, "can I see that as a \
table?" -> format=table, "more detail" -> detail_level=deep. Return no signals when the \
message is just a question, when the user says it is only for this time ("this time", \
"just here", "for this one"), when they ask you to remember a preference permanently \
(that is saved separately), or when it concerns personal data, data access or rules.
Current profile: {profile}"""


class PreferenceSignalOut(BaseModel):
    key: str = Field(description="Preference key from the list.")
    value: str = Field(description="One allowed value of the key.")
    evidence: str = Field(description="The user's words that show it, at most 10 words.")


def describe_keys(config: PreferenceConfig) -> str:
    return "\n".join(
        f"- {spec.name}: {spec.allowed()}"
        for spec in config.keys.values()
        if spec.type == "enum" or spec.name == "report_structure"
    )


def extraction_rules(config: PreferenceConfig, profile_labels: Sequence[str]) -> str:
    return SIGNAL_EXTRACTION_RULES.format(
        keys=describe_keys(config), profile=", ".join(profile_labels) or "empty"
    )


# Not a signal of taste: a one-off request, or a permanent one (saved explicitly by the tool).
ONE_OFF = re.compile(
    r"\b(this\s+time|just\s+(here|this\s+once|for\s+this)|for\s+this\s+(one|answer|question)|"
    r"only\s+(here|now|this\s+time)|for\s+now)\b|в\s+этот\s+раз|только\s+(здесь|сейчас)",
    re.IGNORECASE,
)
PERMANENT = re.compile(
    r"\b(always|from\s+now\s+on|by\s+default|remember|going\s+forward|every\s+time)\b|"
    r"всегда|по\s+умолчанию|запомни|впредь",
    re.IGNORECASE,
)


def from_extraction(
    items: Sequence[PreferenceSignalOut],
    *,
    message: str,
    previous_answer_chars: int | None = None,
) -> list[Signal]:
    """Signals the aux model found in the user's `message`. One-off and permanent requests
    are dropped here rather than trusted to the model's instructions. The length of the
    answer that prompted "shorter" is kept as evidence (what "too long" means to the user)."""
    if ONE_OFF.search(message) or PERMANENT.search(message):
        return []
    out = []
    for item in items:
        evidence = item.evidence.strip()[:120]
        if item.key == "detail_level" and previous_answer_chars:
            evidence = f"{evidence} (previous answer: {previous_answer_chars} chars)".strip()
        out.append(Signal(item.key.strip(), item.value.strip(), "reformat_request", evidence))
    return out


# -- usage ------------------------------------------------------------------------------------

_SEGMENT_DIMENSIONS = re.compile(
    r"\b(category|categories|brand|brands|state|states|department|traffic[\s_]source|"
    r"country|countries|channel)\b",
    re.IGNORECASE,
)
_FILTER = re.compile(r"^(?P<lhs>[^=]+?)\s*(?:=|\bin\b|:)\s*(?P<rhs>.+)$", re.IGNORECASE)


def _mentions(text: str, phrase: str) -> bool:
    words = r"[\s_-]+".join(re.escape(w) for w in re.split(r"[\s_-]+", phrase.lower()) if w)
    return re.search(rf"\b{words}\b", text.lower()) is not None


def metric_signals(config: PreferenceConfig, texts: Sequence[str]) -> list[Signal]:
    spec = config.keys.get("focus_metrics")
    if spec is None:
        return []
    text = " \n ".join(texts)
    found = [
        value
        for value, aliases in spec.values.items()
        if any(_mentions(text, phrase) for phrase in (value, *aliases))
    ]
    return [Signal("focus_metrics", v, "usage", "analysis") for v in found]


def segment_signals(config: PreferenceConfig, filters: Sequence[str]) -> list[Signal]:
    """`category = Jeans`, `brand IN (Levi's, Nike)` -> segments; other filters are skipped."""
    if "focus_segments" not in config.keys:
        return []
    out: list[Signal] = []
    for item in filters:
        match = _FILTER.match(item.strip())
        if not match or not _SEGMENT_DIMENSIONS.search(match["lhs"]):
            continue
        for value in re.split(r",", match["rhs"].strip(" ()[]")):
            value = value.strip(" '\"`")
            if value and value.lower() not in (s.value.lower() for s in out):
                out.append(Signal("focus_segments", value, "usage", match["lhs"].strip()))
    return out


def usage_signals(config: PreferenceConfig, context: Mapping | None) -> list[Signal]:
    """Metrics and segments of a new analysis context (one turn with successful SQL)."""
    if not context:
        return []
    texts = [context.get("question") or "", *(context.get("entities") or [])]
    return metric_signals(config, texts) + segment_signals(config, context.get("filters") or [])


# -- feedback ---------------------------------------------------------------------------------

_BAD_COMMENT_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"too\s+long|too\s+verbose|too\s+much\s+text|shorter|wordy|concise|"
            r"слишком\s+длинн|короче|много\s+текста",
            re.I,
        ),
        "detail_level",
        "brief",
    ),
    (
        re.compile(
            r"too\s+short|more\s+detail|not\s+enough\s+detail|too\s+shallow|подробнее|"
            r"слишком\s+коротк",
            re.I,
        ),
        "detail_level",
        "deep",
    ),
    (re.compile(r"\btables?\b|таблиц", re.I), "format", "table"),
    (re.compile(r"\bbullets?\b|\blist\b|списк", re.I), "format", "bullets"),
    (re.compile(r"\bcharts?\b|\bgraphs?\b|график", re.I), "visuals", "charts"),
]
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+\S", re.M)


def answer_format(answer: str) -> str | None:
    if len(_TABLE_ROW.findall(answer)) >= 3:
        return "table"
    if len(_BULLET.findall(answer)) >= 3:
        return "bullets"
    return None


def feedback_signals(rating: str, answer: str, comment: str = "") -> list[Signal]:
    """`/bad too long` -> detail_level=brief; `/good` on a table answer -> format=table."""
    if rating == "bad":
        return [
            Signal(key, value, "feedback", f"/bad {comment[:80]}")
            for pattern, key, value in _BAD_COMMENT_RULES
            if pattern.search(comment)
        ]
    fmt = answer_format(answer)
    return [Signal("format", fmt, "feedback", "/good")] if fmt else []
