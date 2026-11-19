import json

import pandas as pd
import pytest

from retail_agent.observability import logger
from retail_agent.safety.pii import REDACTED, load_policy, mask_dataframe, mask_text


def test_policy_loads_columns_and_patterns():
    policy = load_policy()
    assert {"email", "first_name", "user_geom"} <= policy.columns["users"]
    assert "age" not in policy.all_columns and "id" not in policy.all_columns
    assert {"email", "phone", "street_address"} <= set(policy.patterns)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Contact john.doe+x@mail.example.com now", f"Contact {REDACTED} now"),
        ("call (555) 123-4567 or +1 555 123 4567", f"call {REDACTED} or {REDACTED}"),
        ("tel 555-123-4567.", f"tel {REDACTED}."),
        ("+7 999 123-45-67", REDACTED),
        ("lives at 8163 Johnson Ferry Road, Atlanta", f"lives at {REDACTED}, Atlanta"),
        ("ship to 393 Barbara Mews Suite 254", f"ship to {REDACTED}"),
        ("2177 Thomas Stravenue Apt. 637", REDACTED),
    ],
)
def test_mask_text_replaces_pii(text, expected):
    masked, count = mask_text(text)
    assert masked == expected
    assert count == expected.count(REDACTED)


@pytest.mark.parametrize(
    "text",
    [
        "Revenue grew 12.5% to $1,234,567.89 in 2024-03",
        "Order 1234567890 and user 98765",
        "Speedo Boys 8-20 Youth Circle Sound Splice Xtra Life Lycra Jammer",
        "Hanes Men's 3 Pack Crew Neck T-Shirt",
        "Men's Striped Fall / Winter Flannel Sleepwear 2-Piece Set",
        "WHERE email LIKE '%@gmail.com'",
        "Top states: California 10 123, Texas 7 456",
        "",
    ],
)
def test_mask_text_keeps_regular_text(text):
    assert mask_text(text) == (text, 0)


def test_mask_dataframe_redacts_pii_columns_and_values():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "Email": ["a@b.com", None, "c@d.org"],
            "note": ["write to x@y.com", "ok", None],
            "revenue": [10.5, 20.0, 30.0],
            "address": [{"street_address": "1 Main Street", "city": "Austin"}, None, None],
        }
    )
    masked, count = mask_dataframe(df)
    assert masked["Email"][0] == masked["Email"][2] == REDACTED
    assert pd.isna(masked["Email"][1])
    assert masked["note"][:2].tolist() == [f"write to {REDACTED}", "ok"]
    assert pd.isna(masked["note"][2])
    assert masked["address"][0] == {"street_address": REDACTED, "city": "Austin"}
    assert masked["user_id"].tolist() == [1, 2, 3]
    assert masked["revenue"].tolist() == [10.5, 20.0, 30.0]
    assert count == 4
    assert df["Email"][0] == "a@b.com"  # input is not modified


def test_mask_dataframe_without_pii_is_unchanged():
    df = pd.DataFrame({"category": ["Jeans", "Tops"], "n": [1, 2]})
    masked, count = mask_dataframe(df)
    assert count == 0
    pd.testing.assert_frame_equal(masked, df)


def test_masking_events_are_logged_without_values(tmp_path):
    path = tmp_path / "agent.jsonl"
    logger.configure(path)
    mask_text("mail me: jane@example.com", where="answer")
    mask_dataframe(pd.DataFrame({"email": ["jane@example.com"]}), where="run_sql")
    mask_text("nothing here", where="answer")

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(r["event"], r["where"], r["count"]) for r in records] == [
        ("pii_masked", "answer", 1),
        ("pii_masked", "run_sql", 1),
    ]
    assert records[1]["columns"] == ["email"]
    assert "jane@example.com" not in path.read_text(encoding="utf-8")


def test_logger_masks_pii_in_events(tmp_path):
    path = tmp_path / "agent.jsonl"
    logger.configure(path)
    logger.log_event("sql_executed", sql="SELECT 1 FROM users WHERE email = 'bob@corp.com'")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert "bob@corp.com" not in record["sql"]
    assert REDACTED in record["sql"]
