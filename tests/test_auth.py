import pytest

from retail_agent import auth


def test_demo_users_load():
    users = auth.load_users()
    assert set(users) == {"ceo", "denim_manager", "outdoor_manager"}
    assert users["ceo"].has_full_access and users["ceo"].brands == ()
    denim = users["denim_manager"]
    assert denim.role == "manager" and not denim.has_full_access
    assert denim.brands == ("Levi's", "Diesel", "7 For All Mankind", "True Religion")
    assert users["outdoor_manager"].brands == ("The North Face", "Columbia", "Carhartt")


def test_access_scope():
    denim = auth.load_user("denim_manager")
    assert denim.can_access("Levi's")
    assert not denim.can_access("levi's")  # exact products.brand values
    assert not denim.can_access("Columbia")
    assert denim.scope_label == "Levi's, Diesel, 7 For All Mankind, True Religion"
    ceo = auth.load_user("ceo")
    assert ceo.can_access("Columbia") and ceo.scope_label == "all brands"


def test_unknown_user_lists_available():
    with pytest.raises(auth.UserConfigError, match="Available: ceo, denim_manager"):
        auth.load_user("nobody")


def write(tmp_path, users):
    path = tmp_path / "users.yaml"
    path.write_text(f"users:\n{users}\n", encoding="utf-8")
    return path


def test_brands_are_stripped_and_deduplicated(tmp_path):
    path = write(tmp_path, "  x: {role: manager, brands: [' Diesel ', Diesel, Levi's]}")
    assert auth.load_users(path)["x"].brands == ("Diesel", "Levi's")


@pytest.mark.parametrize(
    ("users", "message"),
    [
        ("  x: {role: admin}", "unknown role"),
        ("  x: {role: c_level}", "unknown role"),
        ("  x: {role: manager}", "at least one brand"),
        ("  x: {role: manager, brands: []}", "at least one brand"),
        ("  x: {role: manager, brands: Diesel}", "'brands' must be a list"),
        ("  x: {role: manager, brands: ['  ']}", "empty brand name"),
        ("  x: plain", "expected a mapping"),
    ],
)
def test_invalid_user(tmp_path, users, message):
    with pytest.raises(auth.UserConfigError, match=message):
        auth.load_users(write(tmp_path, users))


def test_invalid_config(tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text("users: {}\n", encoding="utf-8")
    with pytest.raises(auth.UserConfigError, match="'users' must be a non-empty mapping"):
        auth.load_users(path)


def test_ceo_ignores_brand_scope(tmp_path):
    user = auth.load_users(write(tmp_path, "  x: {role: ceo, brands: [Diesel]}"))["x"]
    assert user.brands == () and user.has_full_access
