"""Demo identity: users and their data access scope, loaded from `config/users.yaml`."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import yaml

from retail_agent.settings import CONFIG_DIR

Role = Literal["ceo", "c_level", "executive"]
FULL_ACCESS_ROLES: frozenset[str] = frozenset({"ceo", "c_level"})
USERS_FILE = CONFIG_DIR / "users.yaml"


class UserConfigError(ValueError):
    """users.yaml is malformed or the requested user does not exist."""


@dataclass(frozen=True)
class UserContext:
    user_id: str
    name: str
    role: Role
    categories: tuple[str, ...] = ()

    @property
    def has_full_access(self) -> bool:
        return self.role in FULL_ACCESS_ROLES

    def can_access(self, category: str) -> bool:
        return self.has_full_access or category in self.categories

    @property
    def scope_label(self) -> str:
        return "all categories" if self.has_full_access else ", ".join(self.categories)


def _parse_user(user_id: str, raw: dict, known_categories: frozenset[str]) -> UserContext:
    if not isinstance(raw, dict):
        raise UserConfigError(f"User '{user_id}': expected a mapping, got {type(raw).__name__}")
    role = raw.get("role")
    if role not in get_args(Role):
        raise UserConfigError(f"User '{user_id}': unknown role {role!r}")
    categories = tuple(str(c) for c in raw.get("categories") or ())
    unknown = set(categories) - known_categories
    if unknown:
        raise UserConfigError(f"User '{user_id}': unknown categories {sorted(unknown)}")
    if role == "executive" and not categories:
        raise UserConfigError(f"User '{user_id}': executive must have at least one category")
    if role in FULL_ACCESS_ROLES:
        categories = ()
    return UserContext(
        user_id=user_id,
        name=str(raw.get("name") or user_id),
        role=role,
        categories=categories,
    )


def load_users(path: Path = USERS_FILE) -> dict[str, UserContext]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise UserConfigError(f"Users file not found: {path}") from exc
    raw_users = data.get("users")
    if not isinstance(raw_users, dict) or not raw_users:
        raise UserConfigError(f"{path}: 'users' must be a non-empty mapping")
    known_categories = frozenset(str(c) for c in data.get("categories") or ())
    return {
        str(uid): _parse_user(str(uid), raw, known_categories) for uid, raw in raw_users.items()
    }


def load_user(user_id: str, path: Path = USERS_FILE) -> UserContext:
    users = load_users(path)
    try:
        return users[user_id]
    except KeyError:
        known = ", ".join(sorted(users))
        raise UserConfigError(f"Unknown user '{user_id}'. Available: {known}") from None
