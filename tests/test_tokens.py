"""The session's access scope comes only from a verified JWT (D4)."""

import time

import jwt
import pytest

from harness import make_settings
from retail_agent import auth, cli
from retail_agent.auth import AuthError, UserContext
from retail_agent.safety.sql_guard import check_and_rewrite

DENIM = UserContext("denim_manager", "Carlos Ruiz", "manager", ("Levi's", "Diesel"))
CEO = UserContext("ceo", "Alice Morgan", "ceo")


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


def claims(settings, **overrides):
    now = int(time.time())
    base = {
        "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience,
        "sub": "x",
        "name": "X",
        "role": "manager",
        "brands": ["Diesel"],
        "iat": now,
        "exp": now + 60,
    }
    return {k: v for k, v in {**base, **overrides}.items() if v is not None}


def sign(settings, key=None, algorithm="HS256", **overrides):
    return jwt.encode(
        claims(settings, **overrides), key or auth.jwt_secret(settings), algorithm=algorithm
    )


def test_round_trip_keeps_role_and_brands(settings):
    token = auth.issue_token(DENIM, settings)
    assert auth.verify_token(token, settings) == DENIM
    assert auth.verify_token(auth.issue_token(CEO, settings), settings) == CEO


def test_scope_comes_from_the_token_not_the_access_table(settings):
    # A brand that no demo user has: the token alone decides what the ACL filters by.
    user = auth.verify_token(sign(settings, sub="nike_manager", brands=["Nike"]), settings)
    assert user.brands == ("Nike",) and not user.has_full_access
    sql = check_and_rewrite("SELECT id FROM products", user, dataset="d.s").sql
    assert "brand IN ('Nike')" in sql


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda s: sign(s, key="another-secret-of-at-least-32-bytes!!"), "InvalidSignatureError"),
        (lambda s: sign(s, exp=int(time.time()) - 5), "expired"),
        (lambda s: sign(s, aud="other-app"), "InvalidAudienceError"),
        (lambda s: sign(s, iss="someone-else"), "InvalidIssuerError"),
        (lambda s: sign(s, exp=None), "MissingRequiredClaimError"),
        (lambda s: jwt.encode(claims(s), None, algorithm="none"), "InvalidAlgorithmError"),
        (lambda s: sign(s, brands=[]), "at least one brand"),
        (lambda s: sign(s, role="admin"), "unknown role"),
        (lambda s: sign(s, brands="Diesel"), "'brands' must be a list"),
        (lambda s: "not-a-token", "DecodeError"),
    ],
)
def test_invalid_tokens_are_rejected(settings, make, message):
    with pytest.raises(AuthError, match=message):
        auth.verify_token(make(settings), settings)


def test_tampered_payload_is_rejected(settings):
    header, _, signature = auth.issue_token(DENIM, settings).split(".")
    forged = jwt.utils.base64url_encode(
        jwt.api_jws.json.dumps(claims(settings, role="ceo")).encode()
    ).decode()
    with pytest.raises(AuthError, match="InvalidSignatureError"):
        auth.verify_token(f"{header}.{forged}.{signature}", settings)


def test_ceo_token_ignores_brands(settings):
    user = auth.verify_token(sign(settings, role="ceo", brands=["Diesel"]), settings)
    assert user.has_full_access and user.brands == ()


def test_dev_secret_is_generated_once(settings):
    assert not settings.dev_jwt_secret_path.exists()
    first = auth.jwt_secret(settings)
    assert len(first) >= auth.MIN_SECRET_BYTES
    assert auth.jwt_secret(settings) == first
    assert settings.dev_jwt_secret_path.read_text(encoding="utf-8") == first


def test_configured_secret_is_used_and_must_be_long(tmp_path):
    good = make_settings(tmp_path, auth_jwt_secret="x" * 40)
    assert auth.jwt_secret(good) == "x" * 40
    assert not good.dev_jwt_secret_path.exists()
    with pytest.raises(AuthError, match="at least 32 bytes"):
        auth.jwt_secret(make_settings(tmp_path, auth_jwt_secret="short"))


def test_demo_login_goes_through_a_token(settings):
    user = auth.demo_login("denim_manager", settings)
    assert user.brands == ("Levi's", "Diesel", "7 For All Mankind", "True Religion")
    assert settings.dev_jwt_secret_path.exists()  # a token was signed and verified


# -- CLI sign-in --------------------------------------------------------------------------------


def sign_in(settings, *argv, env=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.delenv(cli.TOKEN_ENV, raising=False)
        if env:
            monkeypatch.setenv(cli.TOKEN_ENV, env)
    return cli._sign_in(cli._parse_args(list(argv)), settings)


def test_cli_user_signs_in_with_a_demo_token(settings, monkeypatch):
    user = sign_in(settings, "--user", "outdoor_manager", monkeypatch=monkeypatch)
    assert user.brands == ("The North Face", "Columbia", "Carhartt")


def test_cli_token_argument_and_environment(settings, monkeypatch):
    token = auth.issue_token(DENIM, settings)
    assert sign_in(settings, "--token", token, monkeypatch=monkeypatch) == DENIM
    assert sign_in(settings, env=token, monkeypatch=monkeypatch) == DENIM
    # An explicit --user wins over a token left in the environment.
    assert sign_in(settings, "--user", "ceo", env=token, monkeypatch=monkeypatch).role == "ceo"


def test_cli_rejects_a_bad_token(settings, monkeypatch):
    with pytest.raises(AuthError):
        sign_in(settings, "--token", "garbage", monkeypatch=monkeypatch)


def test_cli_print_token_prints_a_verifiable_token(settings, monkeypatch, capsys):
    assert (
        sign_in(settings, "--user", "denim_manager", "--print-token", monkeypatch=monkeypatch)
        is None
    )
    token = capsys.readouterr().out.strip()
    assert auth.verify_token(token, settings).user_id == "denim_manager"


def test_user_and_token_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli._parse_args(["--user", "ceo", "--token", "t"])
