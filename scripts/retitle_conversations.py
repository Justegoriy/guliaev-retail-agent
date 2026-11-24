"""Give conversations created before summarized titles a short title.

    python scripts/retitle_conversations.py             # all users
    python scripts/retitle_conversations.py --user ceo --dry-run

New conversations get their title from the post-turn extraction call; this script does the
same once for old ones, from their first exchange (one Flash-Lite call per conversation).
"""

import argparse
import sys

from langchain_core.messages import HumanMessage
from rich.console import Console

from retail_agent.console import tolerate_unencodable_output
from retail_agent.data.storage import Storage
from retail_agent.graph import memory
from retail_agent.graph.builder import open_checkpointer
from retail_agent.llm import build_chat_model
from retail_agent.observability import logger
from retail_agent.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    tolerate_unencodable_output()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--user", help="only this user's conversations")
    p.add_argument("--dry-run", action="store_true", help="show titles without saving them")
    args = p.parse_args(argv)

    settings = get_settings()
    logger.configure(settings.log_path)
    console = Console()
    storage = Storage(settings.app_db_path)
    checkpointer = open_checkpointer(settings.checkpoint_db_path)
    extractor = build_chat_model(
        "guard", settings=settings, structured_output=memory.ContextExtraction
    )

    conversations = storage.untitled_conversations(args.user)
    if not conversations:
        console.print("All conversations already have titles.")
        return 0
    for conv in conversations:
        saved = checkpointer.get_tuple({"configurable": {"thread_id": conv.id}})
        messages = saved.checkpoint["channel_values"].get("messages", []) if saved else []
        turns = memory.split_turns(messages)
        if not turns:
            console.print(f"{conv.id[:8]}  no history, kept: {conv.title}")
            continue
        prompt = memory.extraction_prompt(turns[0], [], None, want_title=True)
        try:
            raw = extractor.invoke([HumanMessage(prompt)]).conversation_title
        except Exception as exc:
            console.print(f"{conv.id[:8]}  failed ({type(exc).__name__}), kept: {conv.title}")
            continue
        title = memory.clean_title(raw)
        console.print(f"{conv.id[:8]}  {conv.title[:50]!r} -> {title or '(kept)'}")
        if not args.dry_run:
            storage.set_conversation_title(conv.id, conv.owner_user_id, title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
