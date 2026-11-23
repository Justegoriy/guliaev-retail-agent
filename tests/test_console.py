import io
import sys

from retail_agent.console import tolerate_unencodable_output


def test_unencodable_characters_are_replaced_not_raised(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1251", newline="\n")
    monkeypatch.setattr(sys, "stdout", stream)
    tolerate_unencodable_output()
    print("Выручка 👎 ok")
    stream.flush()
    assert raw.getvalue().decode("cp1251") == "Выручка ? ok\n"
