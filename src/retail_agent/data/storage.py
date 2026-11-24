"""Application database (SQLite): saved reports, conversations, audit log, user preferences.

Every report query is scoped to its owner and skips soft-deleted rows: the repository
has no method that returns another user's report, so ownership cannot be forgotten in
a tool. The schema is created with `CREATE TABLE IF NOT EXISTS` on open.
"""

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    entities TEXT NOT NULL DEFAULT '[]',
    source_queries TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS reports_owner ON reports (owner_user_id, deleted_at);
CREATE INDEX IF NOT EXISTS reports_conversation ON reports (conversation_id);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turns INTEGER NOT NULL DEFAULT 0,
    title_generated INTEGER NOT NULL DEFAULT 0  -- 1: title settled (summarized or kept)
);
CREATE INDEX IF NOT EXISTS conversations_owner ON conversations (owner_user_id, updated_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target_ids TEXT NOT NULL DEFAULT '[]',
    details TEXT NOT NULL DEFAULT '{}',
    trace_id TEXT
);

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('explicit', 'inferred')),
    confidence REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS preference_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    weight REAL NOT NULL,
    kind TEXT NOT NULL,
    conversation_id TEXT,
    trace_id TEXT,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS preference_signals_user ON preference_signals (user_id, key);
"""

MAX_TITLE_CHARS = 200
MAX_CONVERSATION_TITLE_CHARS = 80


class ConversationAccessError(Exception):
    """The conversation belongs to another user."""


@dataclass(frozen=True)
class Report:
    id: str
    owner_user_id: str
    conversation_id: str
    title: str
    content: str
    entities: tuple[str, ...]
    source_queries: tuple[str, ...]
    created_at: str
    updated_at: str

    def snippet(self, chars: int = 120) -> str:
        lines = [line.strip("#*- ").strip() for line in self.content.splitlines()]
        text = " ".join(line for line in lines if line)
        return text if len(text) <= chars else text[: chars - 3].rstrip() + "..."


@dataclass(frozen=True)
class Conversation:
    id: str
    owner_user_id: str
    title: str
    created_at: str
    updated_at: str
    turns: int
    title_generated: bool = False  # False: titled by its first message, not settled yet


def _conversation(row: sqlite3.Row) -> Conversation:
    data = dict(row)
    data["title_generated"] = bool(data.get("title_generated"))
    return Conversation(**data)


@dataclass(frozen=True)
class AuditEntry:
    id: int
    ts: str
    user_id: str
    action: str
    target_ids: tuple[str, ...]
    details: dict[str, Any]
    trace_id: str | None


PreferenceValue = str | list[str]


@dataclass(frozen=True)
class StoredPreference:
    user_id: str
    key: str
    value: PreferenceValue
    source: str  # explicit | inferred
    confidence: float
    evidence_count: int
    updated_at: str


@dataclass(frozen=True)
class StoredSignal:
    id: int
    user_id: str
    key: str
    value: PreferenceValue
    weight: float
    kind: str  # explicit_request | reformat_request | feedback | usage
    conversation_id: str | None
    trace_id: str | None
    ts: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _like_pattern(text: str) -> str:
    escaped = text.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


_REPORT_COLUMNS = (
    "id, owner_user_id, conversation_id, title, content, entities, source_queries, "
    "created_at, updated_at"
)


def _report(row: sqlite3.Row) -> Report:
    return Report(
        id=row["id"],
        owner_user_id=row["owner_user_id"],
        conversation_id=row["conversation_id"],
        title=row["title"],
        content=row["content"],
        entities=tuple(json.loads(row["entities"])),
        source_queries=tuple(json.loads(row["source_queries"])),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class Storage:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Tool calls run in worker threads; one connection guarded by a lock is enough here.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Columns added after a database was created (`CREATE TABLE IF NOT EXISTS` skips them)."""
        columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(conversations)")}
        if "title_generated" not in columns:
            self._conn.execute(
                "ALTER TABLE conversations ADD COLUMN title_generated INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _write(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._lock, self._conn:
            return self._conn.execute(sql, params).rowcount

    # -- reports ----------------------------------------------------------------------

    def save_report(
        self,
        owner_user_id: str,
        conversation_id: str,
        title: str,
        content: str,
        entities: Iterable[str] = (),
        source_queries: Iterable[str] = (),
    ) -> Report:
        now = _now()
        while True:
            report_id = uuid.uuid4().hex[:8]
            try:
                self._write(
                    f"INSERT INTO reports ({_REPORT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        report_id,
                        owner_user_id,
                        conversation_id,
                        title.strip()[:MAX_TITLE_CHARS],
                        content,
                        json.dumps(list(entities), ensure_ascii=False),
                        json.dumps(list(source_queries), ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                continue  # id collision of the short random id
            report = self.get_report(owner_user_id, report_id)
            assert report is not None
            return report

    def get_report(self, owner_user_id: str, report_id: str) -> Report | None:
        rows = self._query(
            f"SELECT {_REPORT_COLUMNS} FROM reports "
            "WHERE id = ? AND owner_user_id = ? AND deleted_at IS NULL",
            (report_id, owner_user_id),
        )
        return _report(rows[0]) if rows else None

    def get_reports(self, owner_user_id: str, report_ids: Iterable[str]) -> list[Report]:
        """The caller's live reports among `report_ids`, oldest first; others are skipped."""
        ids = list(dict.fromkeys(report_ids))
        if not ids:
            return []
        marks = ", ".join("?" * len(ids))
        rows = self._query(
            f"SELECT {_REPORT_COLUMNS} FROM reports WHERE id IN ({marks}) "
            "AND owner_user_id = ? AND deleted_at IS NULL ORDER BY created_at, rowid",
            (*ids, owner_user_id),
        )
        return [_report(r) for r in rows]

    def list_reports(
        self, owner_user_id: str, *, conversation_id: str | None = None, limit: int = 20
    ) -> list[Report]:
        """Newest first."""
        sql = (
            f"SELECT {_REPORT_COLUMNS} FROM reports WHERE owner_user_id = ? AND deleted_at IS NULL"
        )
        params: list[Any] = [owner_user_id]
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        return [_report(r) for r in self._query(sql, params)]

    def search_reports(self, owner_user_id: str, query: str, *, limit: int = 50) -> list[Report]:
        """Case-insensitive substring match in the title, the content or the entities."""
        query = query.strip()
        if not query:
            return []
        pattern = _like_pattern(query)
        rows = self._query(
            f"SELECT {_REPORT_COLUMNS} FROM reports "
            "WHERE owner_user_id = ? AND deleted_at IS NULL AND ("
            "lower(title) LIKE ? ESCAPE '\\' OR lower(content) LIKE ? ESCAPE '\\' "
            "OR lower(entities) LIKE ? ESCAPE '\\') "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (owner_user_id, pattern, pattern, pattern, limit),
        )
        return [_report(r) for r in rows]

    def soft_delete_reports(self, owner_user_id: str, report_ids: Iterable[str]) -> list[str]:
        """Mark the caller's live reports deleted; returns the ids actually deleted."""
        ids = [r.id for r in self.get_reports(owner_user_id, report_ids)]
        if not ids:
            return []
        marks = ", ".join("?" * len(ids))
        now = _now()
        self._write(
            f"UPDATE reports SET deleted_at = ?, updated_at = ? WHERE id IN ({marks}) "
            "AND owner_user_id = ? AND deleted_at IS NULL",
            (now, now, *ids, owner_user_id),
        )
        return ids

    # -- conversations ------------------------------------------------------------------

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        rows = self._query("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        return _conversation(rows[0]) if rows else None

    def check_conversation_owner(self, conversation_id: str, owner_user_id: str) -> None:
        conversation = self.get_conversation(conversation_id)
        if conversation is not None and conversation.owner_user_id != owner_user_id:
            raise ConversationAccessError(conversation_id)

    def record_turn(self, conversation_id: str, owner_user_id: str, first_message: str) -> None:
        """Create the conversation on its first turn (titled by the message), count turns."""
        self.check_conversation_owner(conversation_id, owner_user_id)
        now = _now()
        title = " ".join(first_message.split())[:MAX_CONVERSATION_TITLE_CHARS] or "(untitled)"
        self._write(
            "INSERT INTO conversations (id, owner_user_id, title, created_at, updated_at, turns) "
            "VALUES (?, ?, ?, ?, ?, 1) "
            "ON CONFLICT (id) DO UPDATE SET turns = turns + 1, updated_at = excluded.updated_at",
            (conversation_id, owner_user_id, title, now, now),
        )

    def set_conversation_title(
        self, conversation_id: str, owner_user_id: str, title: str | None
    ) -> None:
        """Settle the title (only the owner's conversation): replace the first-message
        title with a summarized one, or keep it when no usable summary was produced."""
        self._write(
            "UPDATE conversations SET title = COALESCE(?, title), title_generated = 1 "
            "WHERE id = ? AND owner_user_id = ?",
            (title, conversation_id, owner_user_id),
        )

    def untitled_conversations(self, owner_user_id: str | None = None) -> list[Conversation]:
        """Conversations still titled by their first message (for retitling old ones)."""
        sql = "SELECT * FROM conversations WHERE title_generated = 0"
        params: tuple[str, ...] = ()
        if owner_user_id is not None:
            sql += " AND owner_user_id = ?"
            params = (owner_user_id,)
        return [_conversation(r) for r in self._query(sql + " ORDER BY updated_at DESC", params)]

    def list_conversations(self, owner_user_id: str, limit: int = 20) -> list[Conversation]:
        rows = self._query(
            "SELECT * FROM conversations WHERE owner_user_id = ? "
            "ORDER BY updated_at DESC, rowid DESC LIMIT ?",
            (owner_user_id, limit),
        )
        return [_conversation(r) for r in rows]

    # -- audit log --------------------------------------------------------------------------

    def audit(
        self,
        user_id: str,
        action: str,
        target_ids: Iterable[str] = (),
        details: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._write(
            "INSERT INTO audit_log (ts, user_id, action, target_ids, details, trace_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                _now(),
                user_id,
                action,
                json.dumps(list(target_ids)),
                json.dumps(details or {}, ensure_ascii=False, default=str),
                trace_id,
            ),
        )

    # -- user preferences --------------------------------------------------------------------

    def get_preferences(self, user_id: str) -> list[StoredPreference]:
        rows = self._query(
            "SELECT * FROM user_preferences WHERE user_id = ? ORDER BY key", (user_id,)
        )
        return [StoredPreference(**{**dict(r), "value": json.loads(r["value"])}) for r in rows]

    def get_preference(self, user_id: str, key: str) -> StoredPreference | None:
        return next((p for p in self.get_preferences(user_id) if p.key == key), None)

    def upsert_preference(
        self,
        user_id: str,
        key: str,
        value: PreferenceValue,
        *,
        source: str,
        confidence: float,
        evidence_count: int,
    ) -> StoredPreference:
        self._write(
            "INSERT INTO user_preferences "
            "(user_id, key, value, source, confidence, evidence_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (user_id, key) DO UPDATE SET value = excluded.value, "
            "source = excluded.source, confidence = excluded.confidence, "
            "evidence_count = excluded.evidence_count, updated_at = excluded.updated_at",
            (
                user_id,
                key,
                json.dumps(value, ensure_ascii=False),
                source,
                confidence,
                evidence_count,
                _now(),
            ),
        )
        stored = self.get_preference(user_id, key)
        assert stored is not None
        return stored

    def delete_preferences(self, user_id: str, key: str | None = None) -> int:
        """Delete one key (or all keys) of the user; returns the number of rows deleted."""
        if key is None:
            return self._write("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
        return self._write(
            "DELETE FROM user_preferences WHERE user_id = ? AND key = ?", (user_id, key)
        )

    def add_signal(
        self,
        user_id: str,
        key: str,
        value: PreferenceValue,
        *,
        weight: float,
        kind: str,
        conversation_id: str | None = None,
        trace_id: str | None = None,
        ts: str | None = None,
    ) -> None:
        self._write(
            "INSERT INTO preference_signals "
            "(user_id, key, value, weight, kind, conversation_id, trace_id, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                key,
                json.dumps(value, ensure_ascii=False),
                weight,
                kind,
                conversation_id,
                trace_id,
                ts or _now(),
            ),
        )

    def signals(self, user_id: str, key: str | None = None) -> list[StoredSignal]:
        """Oldest first."""
        sql = "SELECT * FROM preference_signals WHERE user_id = ?"
        params: list[Any] = [user_id]
        if key is not None:
            sql += " AND key = ?"
            params.append(key)
        rows = self._query(sql + " ORDER BY id", params)
        return [StoredSignal(**{**dict(r), "value": json.loads(r["value"])}) for r in rows]

    def delete_signals(self, user_id: str, key: str | None = None) -> int:
        if key is None:
            return self._write("DELETE FROM preference_signals WHERE user_id = ?", (user_id,))
        return self._write(
            "DELETE FROM preference_signals WHERE user_id = ? AND key = ?", (user_id, key)
        )

    def audit_entries(self, user_id: str | None = None, limit: int = 100) -> list[AuditEntry]:
        """Newest first."""
        sql = "SELECT * FROM audit_log"
        params: list[Any] = []
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [
            AuditEntry(
                id=r["id"],
                ts=r["ts"],
                user_id=r["user_id"],
                action=r["action"],
                target_ids=tuple(json.loads(r["target_ids"])),
                details=json.loads(r["details"]),
                trace_id=r["trace_id"],
            )
            for r in self._query(sql, params)
        ]
