import pytest

from retail_agent import auth


def test_demo_users_load():
    users = auth.load_users()
    assert set(users) == {"ceo", "cfo", "head_men", "head_women"}
    assert users["ceo"].has_full_access
    assert users["cfo"].has_full_access
    assert users["head_men"].departments == ("Men",)
    assert not users["head_women"].has_full_access


def test_access_scope():
    head_men = auth.load_user("head_men")
    assert head_men.can_access("Men")
    assert not head_men.can_access("Women")
    assert auth.load_user("ceo").can_access("Women")


def test_unknown_user_lists_available():
    with pytest.raises(auth.UserConfigError, match="Available: ceo, cfo"):
        auth.load_user("nobody")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("users:\n  x: {role: admin}\n", "unknown role"),
        ("users:\n  x: {role: executive}\n", "at least one department"),
        ("users:\n  x: {role: executive, departments: [Kids]}\n", "unknown departments"),
        ("users: {}\n", "non-empty mapping"),
    ],
)
def test_invalid_config(tmp_path, body, message):
    path = tmp_path / "users.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(auth.UserConfigError, match=message):
        auth.load_users(path)


def test_full_access_roles_ignore_departments(tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text("users:\n  x: {role: ceo, departments: [Men]}\n", encoding="utf-8")
    assert auth.load_users(path)["x"].departments == ()
