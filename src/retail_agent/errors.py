"""Domain errors.

`SqlRejected` and `SqlInvalid` are returned to the LLM so it can rewrite the query;
the others end the current action with a user-friendly message.
"""


class AgentError(Exception):
    user_message = "Something went wrong while processing your request."


class SqlRejected(AgentError):
    """The query violates a safety rule (not read-only, forbidden table, PII projection...)."""

    user_message = "This query is not allowed."

    def __init__(self, message: str, rule: str = "policy"):
        self.rule = rule
        super().__init__(message)


class SqlInvalid(AgentError):
    """BigQuery rejected the query (syntax error, unknown column...)."""

    user_message = "The query could not be executed."


class QueryTooExpensive(AgentError):
    def __init__(self, bytes_needed: int | None, limit: int):
        self.bytes_needed = bytes_needed
        self.limit = limit
        needed = f"{bytes_needed / 1e9:.2f} GB" if bytes_needed is not None else "more"
        super().__init__(f"Query would scan {needed}, limit is {limit / 1e9:.2f} GB")

    user_message = (
        "This question needs too much data to answer in one go. "
        "Try narrowing the period or the set of products."
    )


class ServiceUnavailable(AgentError):
    def __init__(self, service: str, detail: str = ""):
        self.service = service
        self.detail = detail
        super().__init__(f"{service} unavailable" + (f": {detail}" if detail else ""))

    @property
    def user_message(self) -> str:
        names = {"bigquery": "The data warehouse", "llm": "The language model"}
        return (
            f"{names.get(self.service, self.service)} is temporarily unavailable. "
            "Please try again in a minute."
        )


class GuardBlocked(AgentError):
    """The request was blocked by the input guard (prompt injection, off-topic, PII request)."""

    user_message = "I can't help with that request."
