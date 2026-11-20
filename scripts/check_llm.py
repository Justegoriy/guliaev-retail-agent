"""Smoke check of the LLM layer: one call to the main model and one to the guard model.

Usage: python scripts/check_llm.py [--main-model NAME]

Pass a non-existent `--main-model` to see the fallback model answer instead
(`llm_error` + `fallback_used` events in logs/agent.jsonl).
"""

import argparse
import uuid

from langchain_core.messages import HumanMessage

from retail_agent import errors
from retail_agent.llm import build_chat_model
from retail_agent.observability import logger, tracing
from retail_agent.settings import get_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-model", help="override GEMINI_MODEL for this check")
    args = parser.parse_args()

    settings = get_settings()
    if args.main_model:
        settings = settings.model_copy(update={"gemini_model": args.main_model})
    logger.configure(settings.log_path)
    status = tracing.setup_langsmith(settings)
    print(
        f"LangSmith: {'on' if status.enabled else 'off'}"
        + (f" ({status.warning})" if status.warning else "")
    )
    trace_id = logger.new_trace_id()
    config = tracing.turn_config(
        settings, trace_id=trace_id, conversation_id=uuid.uuid4().hex, user_role="check"
    )

    question = [HumanMessage("Reply with exactly one word: OK")]
    for purpose in ("main", "guard"):
        try:
            reply = build_chat_model(purpose, settings=settings).invoke(question, config=config)
        except Exception as exc:
            kind = errors.classify(exc)
            print(f"{purpose}: FAILED [{kind}] {type(exc).__name__}: {exc}")
            print(errors.user_message(kind, trace_id))
            return 1
        model = reply.response_metadata.get("model_name", "?")
        usage = reply.usage_metadata or {}
        print(
            f"{purpose}: {reply.text.strip()!r} from {model} "
            f"(in={usage.get('input_tokens')}, out={usage.get('output_tokens')})"
        )
    print(f"trace_id: {trace_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
