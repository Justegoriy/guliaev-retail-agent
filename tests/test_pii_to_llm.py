"""PII never reaches a model (D6, confirmed by OpsFleet): every model input is checked.

The user writes personal data, a query result contains it, the conversation is long
enough to be summarized; then every request to the agent model, every auxiliary prompt
(input guard, context extraction, title, summary) and every embedding query is scanned.
"""

import pandas as pd
from langchain_core.messages import AIMessage

from harness import REVENUE_SQL, Harness, sql_call
from retail_agent.data.bigquery import QueryResult
from retail_agent.knowledge.golden import Trio
from test_golden import BagOfWords, make_index

EMAIL = "jane.roe@example.com"
PHONE = "+1 415 555 0134"
ADDRESS = "742 Evergreen Terrace"
RAW = (EMAIL, PHONE, ADDRESS, "Jane", "Roe")


class RecordingEmbeddings(BagOfWords):
    def __init__(self):
        super().__init__()
        self.queries: list[str] = []

    def embed_query(self, text):
        self.queries.append(text)
        return super().embed_query(text)


def leaked(text: str) -> list[str]:
    return [value for value in RAW[:3] if value in text]


def test_no_model_input_contains_personal_data(tmp_path):
    embeddings = RecordingEmbeddings()
    trios = [Trio("revenue", "Monthly revenue", (REVENUE_SQL,), "Sum sale_price.", ("revenue",))]
    index = make_index(tmp_path, embeddings=embeddings, trios=trios, min_score=0.1)
    h = Harness(tmp_path, golden=index, history_summarize_after_turns=2, history_keep_turns=1)
    h.runner.execute.return_value = QueryResult(
        df=pd.DataFrame(
            {
                "user_id": [7],
                "email": [EMAIL],
                "first_name": ["Jane"],
                "note": [f"call {PHONE} or visit {ADDRESS}"],
                "revenue": [120.0],
            }
        ),
        sql="",
        bytes_processed=1,
        duration_ms=1,
        cache_hit=False,
    )
    message = f"monthly revenue for 2024 of {EMAIL}, phone {PHONE}, {ADDRESS}"
    for i in range(4):
        h.turn(message, sql_call(REVENUE_SQL, f"c{i}"), AIMessage(f"Revenue of {EMAIL}: $120."))

    inputs = [m.text for request in h.model.requests for m in request]
    inputs += [p for prompts in h.aux.prompts.values() for p in prompts]
    inputs += embeddings.queries
    assert h.model.requests and embeddings.queries
    assert {"GuardVerdict", "ContextExtraction", "HistorySummary"} <= {
        schema.__name__ for schema in h.aux.prompts
    }
    found = {value for text in inputs for value in leaked(text)}
    assert found == set(), f"personal data reached a model: {found}"
    # The redacted placeholders show the data was there and was masked, not just absent.
    assert any("[REDACTED]" in text for text in inputs)
