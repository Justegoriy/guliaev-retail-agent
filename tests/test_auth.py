import pytest

from retail_agent import auth


def test_demo_users_load():
    users = auth.load_users()
    assert set(users) == {"ceo", "cfo", "bottoms_manager", "outerwear_manager"}
    assert users["ceo"].has_full_access
    assert users["cfo"].has_full_access
    bottoms = users["bottoms_manager"]
    assert bottoms.category_groups == ("Bottoms",)
    assert bottoms.categories == (
        "Jeans",
        "Pants",
        "Pants & Capris",
        "Shorts",
        "Skirts",
        "Leggings",
    )
    assert not users["outerwear_manager"].has_full_access


def test_category_groups_cover_all_26_categories_once():
    groups = auth.load_category_groups()
    categories = [c for cats in groups.values() for c in cats]
    assert len(categories) == len(set(categories)) == 26


def test_access_scope():
    bottoms = auth.load_user("bottoms_manager")
    assert bottoms.can_access("Jeans")
    assert not bottoms.can_access("Swim")
    assert bottoms.scope_label == "Bottoms"
    assert auth.load_user("ceo").can_access("Swim")
    assert auth.load_user("ceo").scope_label == "all categories"


def test_unknown_user_lists_available():
    with pytest.raises(auth.UserConfigError, match="Available: bottoms_manager, ceo"):
        auth.load_user("nobody")


GROUPS = "category_groups:\n  Bottoms: [Jeans, Shorts]\n  Active & Swim: [Active, Swim]\n"


def write(tmp_path, users):
    path = tmp_path / "users.yaml"
    path.write_text(f"users:\n{users}\n{GROUPS}", encoding="utf-8")
    return path


def test_groups_and_individual_categories_combine(tmp_path):
    path = write(
        tmp_path, "  x: {role: executive, category_groups: [Bottoms], categories: [Swim, Jeans]}"
    )
    user = auth.load_users(path)["x"]
    assert user.categories == ("Jeans", "Shorts", "Swim")
    assert user.extra_categories == ("Swim",)
    assert user.scope_label == "Bottoms, Swim"


def test_individual_categories_only(tmp_path):
    user = auth.load_users(write(tmp_path, "  x: {role: executive, categories: [Swim]}"))["x"]
    assert user.categories == ("Swim",)
    assert user.scope_label == "Swim"


@pytest.mark.parametrize(
    ("users", "message"),
    [
        ("  x: {role: admin}", "unknown role"),
        ("  x: {role: executive}", "at least one category group or category"),
        ("  x: {role: executive, category_groups: [Kids]}", "unknown category groups"),
        ("  x: {role: executive, categories: [jeans]}", "unknown categories"),
    ],
)
def test_invalid_user(tmp_path, users, message):
    with pytest.raises(auth.UserConfigError, match=message):
        auth.load_users(write(tmp_path, users))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("users: {}\n" + GROUPS, "'users' must be a non-empty mapping"),
        ("users:\n  x: {role: ceo}\n", "'category_groups' must be a non-empty mapping"),
        (
            "users:\n  x: {role: ceo}\ncategory_groups:\n  A: [Jeans]\n  B: [Jeans]\n",
            "'Jeans' is in both 'A' and 'B'",
        ),
    ],
)
def test_invalid_config(tmp_path, body, message):
    path = tmp_path / "users.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(auth.UserConfigError, match=message):
        auth.load_users(path)


def test_full_access_roles_ignore_scope(tmp_path):
    user = auth.load_users(write(tmp_path, "  x: {role: ceo, category_groups: [Bottoms]}"))["x"]
    assert user.categories == () and user.category_groups == ()
