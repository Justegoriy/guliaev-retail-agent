"""Identity and data access scope: JWTs, and the demo access table `config/users.yaml`.

Brand is the only access dimension (D1): a `manager` analyses only the brands listed for
them, the `ceo` sees everything.

The agent trusts only a verified token (D4): the frontend sends a JWT whose claims carry the
user's role and brands, `verify_token` checks signature, issuer, audience and expiry, and the
resulting `UserContext` is what the SQL ACL filters by. In the prototype the CLI and the web
app play the frontend: `demo_login` issues a token from the access table and verifies it like
any other; `--token` accepts a token issued elsewhere. Tokens are HS256 with a local secret
(`AUTH_JWT_SECRET`, or a generated `data/dev_jwt_secret`); production verifies the identity
provider's RS256 tokens against its JWKS instead.
"""

import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import jwt
import yaml

from retail_agent.settings import CONFIG_DIR, Settings

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


# -- tokens ---------------------------------------------------------------------------------

JWT_ALGORITHM = "HS256"
MIN_SECRET_BYTES = 32


class AuthError(Exception):
    """The token is missing, invalid, expired or carries an unusable scope."""


def jwt_secret(settings: Settings) -> str:
    """The signing secret: `AUTH_JWT_SECRET`, or a local one generated on first use."""
    if settings.auth_jwt_secret is not None:
        secret = settings.auth_jwt_secret.get_secret_value()
        if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
            raise AuthError(f"AUTH_JWT_SECRET must be at least {MIN_SECRET_BYTES} bytes long")
        return secret
    path = settings.dev_jwt_secret_path
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        secret = secrets.token_urlsafe(48)
        path.write_text(secret, encoding="utf-8")
        return secret


def issue_token(user: UserContext, settings: Settings, *, ttl_s: int | None = None) -> str:
    """Sign a token with the user's role and brands (what the frontend would send)."""
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "sub": user.user_id,
        "name": user.name,
        "role": user.role,
        "brands": list(user.brands),
        "iat": now,
        "exp": now + (ttl_s if ttl_s is not None else settings.auth_token_ttl_s),
    }
    return jwt.encode(claims, jwt_secret(settings), algorithm=JWT_ALGORITHM)


def verify_token(token: str, settings: Settings) -> UserContext:
    """The user and scope from a verified token; raises `AuthError` otherwise."""
    try:
        claims = jwt.decode(
            token.strip(),
            jwt_secret(settings),
            algorithms=[JWT_ALGORITHM],  # never "none" or an algorithm the token picks
            audience=settings.auth_jwt_audience,
            issuer=settings.auth_jwt_issuer,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("the token has expired; sign in again") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"invalid token ({type(exc).__name__})") from exc
    try:
        return _parse_user(str(claims["sub"]), claims)
    except UserConfigError as exc:
        raise AuthError(f"the token's access scope is invalid: {exc}") from exc


def sign_in_as(user: UserContext, settings: Settings) -> UserContext:
    """Prototype sign-in: issue a token for an access-table entry and accept it like any
    token, so the session's scope always comes from a verified token."""
    return verify_token(issue_token(user, settings), settings)


def demo_login(user_id: str, settings: Settings, path: Path = USERS_FILE) -> UserContext:
    return sign_in_as(load_user(user_id, path), settings)
