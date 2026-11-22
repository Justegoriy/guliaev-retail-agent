"""`python -m retail_agent.web [--server.port N]`: the demo web interface on localhost only."""

import sys
from pathlib import Path


def main() -> int:
    from streamlit.web import cli as stcli

    app = Path(__file__).with_name("app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app),
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
        *sys.argv[1:],
    ]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
