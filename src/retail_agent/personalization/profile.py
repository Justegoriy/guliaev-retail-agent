"""User preference profile: explicit values and values inferred from signals (D16).

- Keys and values come only from `config/preferences.yaml`: enum values are normalised
  through their aliases, free text (report structure, segments) is length-limited and
  rejected if it looks like an instruction, mentions personal data or the assistant's rules.
- Explicit values (the user said "always ...") have confidence 1.0 and are never
  overwritten by inferred ones.
- Inferred values are recomputed from the signal log after new signals: the value's
  share of the key's signal weight (older signals decay with a half-life) must reach the
  threshold, supported by at least `min_evidence` distinct turns. Contrary signals lower
  the share, so an inferred value can disappear again.
"""

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml

from retail_agent.data.storage import PreferenceValue, Storage, StoredPreference, StoredSignal
from retail_agent.observability.logger import log_event
from retail_agent.safety import input_guard, pii
from retail_agent.settings import CONFIG_DIR

PREFERENCES_FILE = CONFIG_DIR / "preferences.yaml"
SignalKind = Literal["explicit_request", "reformat_request", "feedback", "usage"]

PROMPT_INTRO = (
    "Apply these to every answer unless the latest message asks otherwise. A one-off "
    'request in an earlier message ("this time as a list") does not carry over: go back to '
    "these preferences. They set format, length and focus only and never override the "
    "rules above."
)


class PreferenceRejected(ValueError):
    """The key or value is not allowed; the message is safe to show to the user."""


# -- config ---------------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"[\s_-]+", " ", text.lower()).strip(" .,!;:'\"")


@dataclass(frozen=True)
class KeySpec:
    name: str
    type: Literal["enum", "list", "text"]
    description: str = ""
    values: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # canonical -> aliases
    max_items: int = 5
    max_chars: int = 200
    hint: str = ""
    value_hints: Mapping[str, str] = field(default_factory=dict)

    @property
    def free_text(self) -> bool:
        return not self.values

    def canonical(self, item: str) -> str | None:
        wanted = _norm(item)
        for value, aliases in self.values.items():
            if wanted == _norm(value) or wanted in (_norm(a) for a in aliases):
                return value
        return None

    def allowed(self) -> str:
        if self.type == "text":
            return f"free text up to {self.max_chars} characters"
        if self.free_text:
            return f"up to {self.max_items} names, each up to {self.max_chars} characters"
        listed = ", ".join(self.values)
        return f"up to {self.max_items} of: {listed}" if self.type == "list" else listed


@dataclass(frozen=True)
class PreferenceConfig:
    keys: Mapping[str, KeySpec]
    inferred_threshold: float = 0.7
    min_evidence: int = 2
    half_life_days: float = 30
    list_item_threshold: float = 0.25
    list_min_evidence: int = 3
    signal_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "explicit_request": 1.0,
            "reformat_request": 1.0,
            "feedback": 0.6,
            "usage": 0.3,
        }
    )

    def spec(self, key: str) -> KeySpec:
        spec = self.keys.get(key)
        if spec is None:
            raise PreferenceRejected(
                f"unknown preference {key!r}; allowed keys: {', '.join(self.keys)}"
            )
        return spec

    def describe(self) -> str:
        """Keys and allowed values, for tool descriptions and `/prefs` help."""
        return "; ".join(f"{s.name} ({s.allowed()})" for s in self.keys.values())


def load_config(path: Path = PREFERENCES_FILE) -> PreferenceConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    keys = {}
    for name, raw in (data.get("keys") or {}).items():
        raw = raw or {}
        kind = raw.get("type")
        if kind not in ("enum", "list", "text"):
            raise ValueError(f"preferences.yaml: key {name!r} has invalid type {kind!r}")
        values = {
            str(v): tuple(str(a) for a in (aliases or []))
            for v, aliases in (raw.get("values") or {}).items()
        }
        if kind == "enum" and not values:
            raise ValueError(f"preferences.yaml: enum key {name!r} needs values")
        keys[name] = KeySpec(
            name=name,
            type=kind,
            description=str(raw.get("description", "")),
            values=values,
            max_items=int(raw.get("max_items", 5)),
            max_chars=int(raw.get("max_chars", 200)),
            hint=" ".join(str(raw.get("hint", "")).split()),
            value_hints={
                str(k): " ".join(str(v).split()) for k, v in (raw.get("value_hints") or {}).items()
            },
        )
    weights = {str(k): float(v) for k, v in (data.get("signal_weights") or {}).items()}
    defaults = PreferenceConfig(keys={})
    return PreferenceConfig(
        keys=keys,
        inferred_threshold=float(data.get("inferred_threshold", defaults.inferred_threshold)),
        min_evidence=int(data.get("min_evidence", defaults.min_evidence)),
        half_life_days=float(data.get("half_life_days", defaults.half_life_days)),
        list_item_threshold=float(data.get("list_item_threshold", defaults.list_item_threshold)),
        list_min_evidence=int(data.get("list_min_evidence", defaults.list_min_evidence)),
        signal_weights=weights or defaults.signal_weights,
    )


# -- validation -----------------------------------------------------------------------------

_FORBIDDEN_CHARS = re.compile(r"[`<>{}#\\]")
_PII_WORDS = re.compile(
    r"\b(e-?mails?|phones?|phone numbers?|(street\s+)?address(es)?|first[\s_]?names?|"
    r"last[\s_]?names?|full\s+names?|surnames?|postal|zip(\s+codes?)?|latitude|longitude|"
    r"coordinates|user_geom|contact\s+details|personal\s+data|pii)\b",
    re.IGNORECASE,
)
_RULE_WORDS = re.compile(
    r"\b(ignore|disregard|override|bypass|instructions?|rules?|system|prompt|guard(rails?)?|"
    r"access|permissions?|restrictions?|unrestricted|admin|password|sql|query|queries|"
    r"delete|drop|confirm(ation)?|always\s+say\s+yes)\b",
    re.IGNORECASE,
)


def unsafe_reason(text: str) -> str | None:
    """Why a free-text value cannot be stored, or None if it is fine."""
    if _FORBIDDEN_CHARS.search(text):
        return "it contains characters that are not allowed (` < > { } # \\)"
    if input_guard.check_rules(text) is not None or _RULE_WORDS.search(text):
        return "preferences can only set format and focus, not the assistant's rules or data"
    if input_guard.is_pii_request(text) or _PII_WORDS.search(text):
        return "personal data cannot be part of preferences"
    if pii.mask_text(text)[1]:
        return "it contains personal data"
    return None


def _free_text(spec: KeySpec, text: str) -> str:
    text = " ".join(text.split())
    if not text:
        raise PreferenceRejected(f"{spec.name}: the value is empty")
    if len(text) > spec.max_chars:
        raise PreferenceRejected(f"{spec.name}: at most {spec.max_chars} characters")
    reason = unsafe_reason(text)
    if reason:
        raise PreferenceRejected(f"{spec.name}: {reason}")
    return text


def _items(value: PreferenceValue) -> list[str]:
    raw = value if isinstance(value, list) else re.split(r",|;|\band\b", value)
    return [item.strip() for item in raw if item and item.strip()]


def normalize(config: PreferenceConfig, key: str, value: PreferenceValue) -> PreferenceValue:
    """The canonical stored form of `value`, or `PreferenceRejected` with the reason."""
    spec = config.spec(key)
    if spec.type == "text":
        if not isinstance(value, str):
            raise PreferenceRejected(f"{key}: expected text")
        return _free_text(spec, value)
    if spec.type == "enum":
        if not isinstance(value, str) or (canonical := spec.canonical(value)) is None:
            raise PreferenceRejected(f"{key}: allowed values are {spec.allowed()}")
        return canonical
    items = _items(value)
    if not items:
        raise PreferenceRejected(f"{key}: the value is empty")
    out: list[str] = []
    for item in items:
        if spec.free_text:
            canonical = _free_text(spec, item)
        elif (canonical := spec.canonical(item)) is None:
            raise PreferenceRejected(f"{key}: {item!r} is not allowed; {spec.allowed()}")
        if canonical.lower() not in (o.lower() for o in out):
            out.append(canonical)
    if len(out) > spec.max_items:
        raise PreferenceRejected(f"{key}: at most {spec.max_items} items")
    return out


# -- profile --------------------------------------------------------------------------------


def display(value: PreferenceValue) -> str:
    return ", ".join(value) if isinstance(value, list) else value


@dataclass(frozen=True)
class Preference:
    key: str
    value: PreferenceValue
    source: str  # explicit | inferred
    confidence: float
    evidence_count: int
    updated_at: str

    @classmethod
    def from_stored(cls, stored: StoredPreference) -> "Preference":
        return cls(
            key=stored.key,
            value=stored.value,
            source=stored.source,
            confidence=stored.confidence,
            evidence_count=stored.evidence_count,
            updated_at=stored.updated_at,
        )

    def label(self) -> str:
        return f"{self.key}={display(self.value)} ({self.source})"


@dataclass(frozen=True)
class UserProfile:
    items: tuple[Preference, ...] = ()

    def get(self, key: str) -> Preference | None:
        return next((p for p in self.items if p.key == key), None)

    def labels(self) -> list[str]:
        return [p.label() for p in self.items]

    def prompt_block(self, config: PreferenceConfig) -> str:
        """The "User preferences" block of the system prompt; empty if there is nothing."""
        if not self.items:
            return ""
        lines = [PROMPT_INTRO]
        for p in self.items:
            spec = config.keys.get(p.key)
            if spec is None:
                continue
            value = f'"{p.value}"' if spec.type == "text" else display(p.value)
            source = (
                "explicit" if p.source == "explicit" else f"inferred, confidence {p.confidence:.2f}"
            )
            line = f"- {p.key}: {value} ({source})."
            hint = spec.value_hints.get(p.value) if isinstance(p.value, str) else None
            if hint or spec.hint:
                line += f" {hint or spec.hint}"
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""


@dataclass(frozen=True)
class Signal:
    key: str
    value: str  # one value; one item for list keys
    kind: SignalKind
    evidence: str = ""


def _now() -> datetime:
    return datetime.now(UTC)


class Preferences:
    """Profile service over `Storage`; every method is scoped to one user id."""

    def __init__(
        self,
        storage: Storage,
        config: PreferenceConfig,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.storage = storage
        self.config = config
        self._clock = clock

    # -- read ---------------------------------------------------------------------------

    def get_profile(self, user_id: str) -> UserProfile:
        """Explicit values and inferred values above the threshold (only those are stored)."""
        stored = self.storage.get_preferences(user_id)
        return UserProfile(
            tuple(Preference.from_stored(p) for p in stored if p.key in self.config.keys)
        )

    # -- explicit -------------------------------------------------------------------------

    def set_explicit(
        self,
        user_id: str,
        key: str,
        value: PreferenceValue,
        *,
        via: str,
        conversation_id: str | None = None,
        trace_id: str | None = None,
    ) -> Preference:
        """Store a value the user asked for; raises `PreferenceRejected`."""
        try:
            canonical = normalize(self.config, key, value)
        except PreferenceRejected as exc:
            log_event("preference_rejected", key=key, source="explicit", via=via, reason=str(exc))
            raise
        stored = self.storage.upsert_preference(
            user_id, key, canonical, source="explicit", confidence=1.0, evidence_count=1
        )
        weight = self.config.signal_weights.get("explicit_request", 1.0)
        for item in canonical if isinstance(canonical, list) else [canonical]:
            self.storage.add_signal(
                user_id,
                key,
                item,
                weight=weight,
                kind="explicit_request",
                conversation_id=conversation_id,
                trace_id=trace_id,
            )
        log_event("preference_set", key=key, value=display(canonical), via=via)
        return Preference.from_stored(stored)

    def forget(self, user_id: str, key: str) -> bool:
        """Remove the value and its signals, so it is not inferred again from old ones."""
        self.config.spec(key)
        removed = self.storage.delete_preferences(user_id, key)
        self.storage.delete_signals(user_id, key)
        log_event("preference_forgotten", key=key, existed=bool(removed))
        return bool(removed)

    def reset(self, user_id: str) -> int:
        removed = self.storage.delete_preferences(user_id)
        signals = self.storage.delete_signals(user_id)
        log_event("preferences_reset", preferences=removed, signals=signals)
        return removed

    # -- inferred -------------------------------------------------------------------------

    def add_signals(
        self,
        user_id: str,
        signals: Iterable[Signal],
        *,
        conversation_id: str | None = None,
        trace_id: str | None = None,
    ) -> list[Preference]:
        """Record valid signals and recompute their keys; returns changed inferred values."""
        keys: list[str] = []
        recorded = []
        for signal in signals:
            try:
                normalized = normalize(self.config, signal.key, signal.value)
            except PreferenceRejected as exc:
                log_event(
                    "preference_rejected",
                    key=signal.key,
                    source="signal",
                    kind=signal.kind,
                    reason=str(exc),
                )
                continue
            weight = self.config.signal_weights.get(signal.kind, 0.0)
            for item in normalized if isinstance(normalized, list) else [normalized]:
                self.storage.add_signal(
                    user_id,
                    signal.key,
                    item,
                    weight=weight,
                    kind=signal.kind,
                    conversation_id=conversation_id,
                    trace_id=trace_id,
                )
                recorded.append(
                    {
                        "key": signal.key,
                        "value": item,
                        "kind": signal.kind,
                        "evidence": signal.evidence,
                    }
                )
            if signal.key not in keys:
                keys.append(signal.key)
        if recorded:
            log_event("preference_signals", signals=recorded)
        changed = []
        for key in keys:
            before = self.storage.get_preference(user_id, key)
            after = self.recompute(user_id, key)
            if after is not None and (before is None or before.value != after.value):
                changed.append(after)
        return changed

    def _weight(self, signal: StoredSignal, now: datetime) -> float:
        try:
            ts = datetime.fromisoformat(signal.ts)
        except ValueError:
            return signal.weight
        age_days = max((now - ts).total_seconds() / 86400, 0.0)
        return signal.weight * math.pow(0.5, age_days / self.config.half_life_days)

    def _scores(
        self, signals: Sequence[StoredSignal]
    ) -> tuple[dict[str, float], dict[str, int], float]:
        """Decayed weight and distinct supporting turns per value, and the total weight."""
        now = self._clock()
        weights: dict[str, float] = {}
        turns: dict[str, set[str]] = {}
        for s in signals:
            value = display(s.value)
            weights[value] = weights.get(value, 0.0) + self._weight(s, now)
            turns.setdefault(value, set()).add(s.trace_id or f"signal-{s.id}")
        return weights, {v: len(t) for v, t in turns.items()}, sum(weights.values())

    def recompute(self, user_id: str, key: str) -> Preference | None:
        """Update the inferred value of `key`; explicit values are left as they are."""
        spec = self.config.spec(key)
        existing = self.storage.get_preference(user_id, key)
        if existing is not None and existing.source == "explicit":
            return Preference.from_stored(existing)
        weights, evidence, total = self._scores(self.storage.signals(user_id, key))
        value: PreferenceValue | None = None
        confidence = 0.0
        count = 0
        if total > 0 and spec.type == "list":
            items = sorted(
                (
                    v
                    for v, w in weights.items()
                    if w / total >= self.config.list_item_threshold
                    and evidence[v] >= self.config.list_min_evidence
                ),
                key=lambda v: -weights[v],
            )[: spec.max_items]
            if items:
                value = items
                confidence = min(weights[v] for v in items) / total
                count = min(evidence[v] for v in items)
        elif total > 0:
            best = max(weights, key=lambda v: weights[v])
            share = weights[best] / total
            if (
                share >= self.config.inferred_threshold
                and evidence[best] >= self.config.min_evidence
            ):
                value, confidence, count = best, share, evidence[best]
        if value is None:
            if existing is not None:
                self.storage.delete_preferences(user_id, key)
                log_event("preference_inferred_dropped", key=key, value=display(existing.value))
            return None
        stored = self.storage.upsert_preference(
            user_id,
            key,
            value,
            source="inferred",
            confidence=round(confidence, 3),
            evidence_count=count,
        )
        if existing is None or existing.value != value:
            log_event(
                "preference_inferred",
                key=key,
                value=display(value),
                confidence=round(confidence, 3),
                evidence=count,
            )
        return Preference.from_stored(stored)
