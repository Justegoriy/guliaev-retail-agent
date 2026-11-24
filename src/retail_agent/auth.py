"""Demo identity: users and their data access scope, loaded from `config/users.yaml`.

Brand is the only access dimension (D1): a `manager` analyses only the brands listed for
them, the `ceo` sees everything.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import yaml

from retail_agent.settings import CONFIG_DIR

Role = Literal["ceo", "manager"]
FULL_ACCESS_ROLES: frozenset[str] = frozenset({"ceo"})
USERS_FILE = CONFIG_DIR / "users.yaml"


class UserConfigError(ValueError):
    """users.yaml is malformed or the requested user does not exist."""


@dataclass(frozen=True)
class UserContext:
    user_id: str
    name: str
    role: Role
    brands: tuple[str, ...] = ()  # products.brand values the ACL filters on (exact match)

    @property
    def has_full_access(self) -> bool:
        return self.role in FULL_ACCESS_ROLES

    def can_access(self, brand: str) -> bool:
        return self.has_full_access or brand in self.brands

    @property
    def scope_label(self) -> str:
        return "all brands" if self.has_full_access else ", ".join(self.brands)


def _parse_user(user_id: str, raw: dict) -> UserContext:
    if not isinstance(raw, dict):
        raise UserConfigError(f"User '{user_id}': expected a mapping, got {type(raw).__name__}")
    role = raw.get("role")
    if role not in get_args(Role):
        raise UserConfigError(f"User '{user_id}': unknown role {role!r}")
    brands_raw = raw.get("brands") or ()
    if isinstance(brands_raw, str) or not isinstance(brands_raw, list | tuple):
        raise UserConfigError(f"User '{user_id}': 'brands' must be a list")
    brands = tuple(dict.fromkeys(str(b).strip() for b in brands_raw))
    if any(not b for b in brands):
        raise UserConfigError(f"User '{user_id}': empty brand name")
    if role in FULL_ACCESS_ROLES:
        brands = ()
    elif not brands:
        raise UserConfigError(f"User '{user_id}': a manager must have at least one brand")
    return UserContext(
        user_id=user_id, name=str(raw.get("name") or user_id), role=role, brands=brands
    )


def _read(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise UserConfigError(f"Users file not found: {path}") from exc


def load_users(path: Path = USERS_FILE) -> dict[str, UserContext]:
    raw_users = _read(path).get("users")
    if not isinstance(raw_users, dict) or not raw_users:
        raise UserConfigError(f"{path}: 'users' must be a non-empty mapping")
    return {str(uid): _parse_user(str(uid), raw) for uid, raw in raw_users.items()}


def load_user(user_id: str, path: Path = USERS_FILE) -> UserContext:
    users = load_users(path)
    try:
        return users[user_id]
    except KeyError:
        known = ", ".join(sorted(users))
        raise UserConfigError(f"Unknown user '{user_id}'. Available: {known}") from None
