"""Console output that never crashes on characters the terminal's encoding lacks."""

import sys


def tolerate_unencodable_output() -> None:
    """Print `?` instead of raising on characters stdout/stderr cannot encode.

    A Windows console or pipe often uses a legacy code page (cp1251, cp437): an emoji or a
    typographic symbol from a log, a question or a model answer must not kill the command.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
