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
    categories: tuple[str, ...] = ()  # resolved products.category values the ACL filters on
    category_groups: tuple[str, ...] = ()  # groups granted (for display)
    extra_categories: tuple[str, ...] = ()  # categories granted individually (for display)

    @property
    def has_full_access(self) -> bool:
        return self.role in FULL_ACCESS_ROLES

    def can_access(self, category: str) -> bool:
        return self.has_full_access or category in self.categories

    @property
    def scope_label(self) -> str:
        if self.has_full_access:
            return "all categories"
        granted = self.category_groups + self.extra_categories
        return ", ".join(granted or self.categories)


def _parse_category_groups(data: dict, path: Path) -> dict[str, tuple[str, ...]]:
    raw = data.get("category_groups")
    if not isinstance(raw, dict) or not raw:
        raise UserConfigError(f"{path}: 'category_groups' must be a non-empty mapping")
    groups: dict[str, tuple[str, ...]] = {}
    owner: dict[str, str] = {}
    for group, categories in raw.items():
        groups[str(group)] = tuple(str(c) for c in categories or ())
        for category in groups[str(group)]:
            if category in owner:
                raise UserConfigError(
                    f"{path}: category {category!r} is in both {owner[category]!r} and {group!r}"
                )
            owner[category] = str(group)
    return groups


def _parse_user(user_id: str, raw: dict, groups: dict[str, tuple[str, ...]]) -> UserContext:
    if not isinstance(raw, dict):
        raise UserConfigError(f"User '{user_id}': expected a mapping, got {type(raw).__name__}")
    role = raw.get("role")
    if role not in get_args(Role):
        raise UserConfigError(f"User '{user_id}': unknown role {role!r}")

    user_groups = tuple(str(g) for g in raw.get("category_groups") or ())
    unknown_groups = set(user_groups) - set(groups)
    if unknown_groups:
        raise UserConfigError(f"User '{user_id}': unknown category groups {sorted(unknown_groups)}")
    extra = tuple(str(c) for c in raw.get("categories") or ())
    known_categories = {c for cats in groups.values() for c in cats}
    unknown = set(extra) - known_categories
    if unknown:
        raise UserConfigError(f"User '{user_id}': unknown categories {sorted(unknown)}")

    from_groups = [c for g in user_groups for c in groups[g]]
    categories = tuple(dict.fromkeys(from_groups + list(extra)))
    extra = tuple(c for c in extra if c not in from_groups)
    if role == "executive" and not categories:
        raise UserConfigError(
            f"User '{user_id}': executive must have at least one category group or category"
        )
    if role in FULL_ACCESS_ROLES:
        categories, user_groups, extra = (), (), ()
    return UserContext(
        user_id=user_id,
        name=str(raw.get("name") or user_id),
        role=role,
        categories=categories,
        category_groups=user_groups,
        extra_categories=extra,
    )


def _read(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise UserConfigError(f"Users file not found: {path}") from exc


def load_category_groups(path: Path = USERS_FILE) -> dict[str, tuple[str, ...]]:
    """Product category group -> products.category values."""
    return _parse_category_groups(_read(path), path)


def load_users(path: Path = USERS_FILE) -> dict[str, UserContext]:
    data = _read(path)
    raw_users = data.get("users")
    if not isinstance(raw_users, dict) or not raw_users:
        raise UserConfigError(f"{path}: 'users' must be a non-empty mapping")
    groups = _parse_category_groups(data, path)
    return {str(uid): _parse_user(str(uid), raw, groups) for uid, raw in raw_users.items()}


def load_user(user_id: str, path: Path = USERS_FILE) -> UserContext:
    users = load_users(path)
    try:
        return users[user_id]
    except KeyError:
        known = ", ".join(sorted(users))
        raise UserConfigError(f"Unknown user '{user_id}'. Available: {known}") from None
