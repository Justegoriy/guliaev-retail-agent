import pytest

from retail_agent import auth


def test_demo_users_load():
    users = auth.load_users()
    assert set(users) == {"ceo", "cfo", "jeans_manager", "outerwear_manager"}
    assert users["ceo"].has_full_access
    assert users["cfo"].has_full_access
    assert users["jeans_manager"].categories == ("Jeans",)
    assert users["outerwear_manager"].categories == ("Outerwear & Coats", "Blazers & Jackets")
    assert not users["outerwear_manager"].has_full_access


def test_access_scope():
    jeans = auth.load_user("jeans_manager")
    assert jeans.can_access("Jeans")
    assert not jeans.can_access("Swim")
    assert jeans.scope_label == "Jeans"
    assert auth.load_user("ceo").can_access("Swim")
    assert auth.load_user("ceo").scope_label == "all categories"


def test_unknown_user_lists_available():
    with pytest.raises(auth.UserConfigError, match="Available: ceo, cfo"):
        auth.load_user("nobody")


CATEGORIES = "categories: [Jeans, Swim]\n"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("users:\n  x: {role: admin}\n", "unknown role"),
        ("users:\n  x: {role: executive}\n" + CATEGORIES, "at least one category"),
        ("users:\n  x: {role: executive, categories: [Kids]}\n" + CATEGORIES, "unknown categories"),
        (
            "users:\n  x: {role: executive, categories: [jeans]}\n" + CATEGORIES,
            "unknown categories",
        ),
        ("users:\n  x: {role: executive, categories: [Jeans]}\n", "unknown categories"),
        ("users: {}\n", "non-empty mapping"),
    ],
)
def test_invalid_config(tmp_path, body, message):
    path = tmp_path / "users.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(auth.UserConfigError, match=message):
        auth.load_users(path)


def test_full_access_roles_ignore_categories(tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text("users:\n  x: {role: ceo, categories: [Jeans]}\n" + CATEGORIES, "utf-8")
    assert auth.load_users(path)["x"].categories == ()
