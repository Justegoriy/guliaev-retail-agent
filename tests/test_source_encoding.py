"""Source and config files are UTF-8 without mojibake.

`web/app.py` once had "·", "…" and emoji saved as UTF-8 bytes read in cp1251 ("В·", "вЂ¦"),
which the web app then showed to users.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# UTF-8 lead bytes 0xC2, 0xE2, 0xF0 followed by a continuation byte, decoded as cp1251.
MOJIBAKE = re.compile(r"В[ -¿]|вЂ|в„|рџ|Р[ЂЃ‚ѓ„…†‡€‰Љ‹ЊЌЋЏ]")
FILES = [
    *ROOT.glob("src/**/*.py"),
    *ROOT.glob("scripts/*.py"),
    *ROOT.glob("evals/*.py"),
    *ROOT.glob("evals/*.yaml"),
    *ROOT.glob("config/*"),
    *ROOT.glob("docs/*.md"),
    ROOT / "README.md",
]


def test_no_mojibake_in_sources():
    bad = []
    for path in FILES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if MOJIBAKE.search(line):
                bad.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:80]}")
    assert not bad, "\n".join(bad)
