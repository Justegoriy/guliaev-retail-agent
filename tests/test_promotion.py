"""Golden Bucket candidates are promoted automatically; bad trios are deprecated."""

import random
from datetime import UTC, datetime

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from harness import REVENUE_SQL, Harness, sql_call
from retail_agent.auth import UserContext
from retail_agent.data.storage import Storage
from retail_agent.errors import SqlRejected
from retail_agent.knowledge import golden
from retail_agent.knowledge.golden import CandidateStore, Retrieved, Trio, load_trios
from retail_agent.knowledge.promotion import (
    CandidateReview,
    Promoter,
    PromotionPolicy,
    audit_queue,
    leaked_numbers,
    record_feedback,
)
from test_golden import TRIOS, BagOfWords, make_index

METHOD = (
    "Revenue is SUM(sale_price) of order items not Cancelled or Returned, grouped by "
    "FORMAT_DATE('%Y-%m', created_at) over the requested year."
)
GOOD = CandidateReview(answers_question=5, reusable=True, method=METHOD, reason="correct")
DF = pd.DataFrame({"month": ["2024-01", "2024-02"], "revenue": [24974.12, 21409.5]})


def add(store, trace_id="t1", user_id="denim_manager", **kwargs):
    defaults = dict(
        question="Monthly revenue in 2024",
        sql=[REVENUE_SQL],
        answer_summary="Revenue was $24,974.12 in January and 21,409.5 in February.",
        user_role="manager",
        trace_id=trace_id,
        tags=["revenue", "diesel"],
        user_id=user_id,
        brands=["Diesel"],
    )
    return store.add(**{**defaults, **kwargs})


class World:
    def __init__(self, tmp_path, review=GOOD, df=DF, index=None, policy=None):
        self.store = CandidateStore(tmp_path / "cand.jsonl")
        self.trios_path = tmp_path / "trios.json"
        golden.save_trios(self.trios_path, TRIOS[:1])
        self.checked: list[tuple[str, UserContext]] = []
        self.prompts: list[str] = []
        self.review, self.df = review, df

        def check_sql(sql, author):
            self.checked.append((sql, author))
            if "email" in sql:
                raise SqlRejected("personal data")
            return f"guarded:{sql}"

        def judge(prompt):
            self.prompts.append(prompt)
            review = self.review(prompt) if callable(self.review) else self.review
            if isinstance(review, Exception):
                raise review
            return review

        self.promoter = Promoter(
            store=self.store,
            trios_path=self.trios_path,
            check_sql=check_sql,
            run_sql=lambda sql: self.df,
            judge=judge,
            index=index,
            policy=policy or PromotionPolicy(),
            rng=random.Random(0),
        )

    def run(self):
        return self.promoter.run()

    def status(self, candidate_id):
        return next(c for c in self.store.all() if c.id == candidate_id)


def test_good_candidate_is_approved_with_method_notes_only(tmp_path):
    w = World(tmp_path)
    candidate = add(w.store)
    (outcome,) = w.run()
    assert outcome.decision == "approved"
    trio = load_trios(w.trios_path)[-1]
    assert trio.id == outcome.trio_id and trio.report == METHOD
    assert "24,974" not in trio.report and "21409" not in trio.report
    assert trio.source["approved_by"] == "auto" and trio.source["checks"]["score"] == 5
    assert trio.tags == ("revenue",)  # the author's brand is not a tag
    stored = w.status(candidate.id)
    assert stored.status == "approved" and stored.method == METHOD
    # The SQL is checked under the author's own scope, not with full access.
    ((_, author),) = w.checked
    assert author.role == "manager" and author.brands == ("Diesel",)


@pytest.mark.parametrize(
    "method",
    [
        "Revenue was 24974.12 in January, so sum sale_price by month.",  # number from the rows
        "Sum sale_price by month; January had $24,974 for this brand.",  # amount
        "Sum sale_price by month; it grew by 16% month over month.",  # percentage
    ],
)
def test_method_that_repeats_results_is_quarantined(tmp_path, method):
    w = World(tmp_path, review=GOOD.model_copy(update={"method": method}))
    candidate = add(w.store)
    (outcome,) = w.run()
    assert outcome.decision == "quarantined" and "leak" in outcome.reason
    assert len(load_trios(w.trios_path)) == 1
    assert w.status(candidate.id).status == "quarantined"


def test_years_and_numbers_from_the_question_are_not_leaks():
    rows = "month,revenue\n2024-01,10"
    found = leaked_numbers(
        "Take the top 10 months of 2024 by SUM(sale_price).",
        answer="Top 10 months of 2024: ...",
        rows=rows,
        question="Top 10 months of 2024 by revenue",
    )
    assert found == []


@pytest.mark.parametrize(
    ("update", "decision"),
    [
        ({"answers_question": 3}, "quarantined"),
        ({"answers_question": 2}, "rejected"),
        ({"reusable": False}, "rejected"),
    ],
)
def test_judge_thresholds(tmp_path, update, decision):
    w = World(tmp_path, review=GOOD.model_copy(update=update))
    add(w.store)
    (outcome,) = w.run()
    assert outcome.decision == decision
    assert len(load_trios(w.trios_path)) == 1


def test_bad_or_empty_sql_is_rejected_before_the_judge(tmp_path):
    w = World(tmp_path)
    add(w.store, sql=["SELECT email FROM users"])
    w.df = DF
    (outcome,) = w.run()
    assert outcome.decision == "rejected" and "SQL check failed" in outcome.reason

    w = World(tmp_path / "empty", df=pd.DataFrame({"revenue": []}))
    add(w.store)
    (outcome,) = w.run()
    assert outcome.decision == "rejected" and "no rows" in outcome.reason
    assert w.prompts == []


def test_author_without_a_usable_scope_is_rejected(tmp_path):
    w = World(tmp_path)
    add(w.store, brands=[])  # a manager without brands
    (outcome,) = w.run()
    assert outcome.decision == "rejected" and "author scope" in outcome.reason
    assert w.checked == []


def test_duplicate_of_an_approved_trio_is_rejected(tmp_path):
    index = make_index(tmp_path, embeddings=BagOfWords())
    w = World(tmp_path, index=index, policy=PromotionPolicy(duplicate_score=0.5))
    add(w.store, question=TRIOS[0].question)
    (outcome,) = w.run()
    assert outcome.decision == "rejected" and f"duplicate of trio {TRIOS[0].id}" in outcome.reason
    assert w.prompts == []


def test_keyword_matches_are_not_treated_as_duplicates(tmp_path):
    class KeywordIndex:
        def retrieve(self, text, k=None):
            return [Retrieved(TRIOS[0], 1.0, "keywords")]

    w = World(tmp_path, index=KeywordIndex())
    add(w.store)
    assert w.run()[0].decision == "approved"


def test_unavailable_judge_defers_the_candidate(tmp_path):
    w = World(tmp_path, review=TimeoutError())
    candidate = add(w.store)
    (outcome,) = w.run()
    assert outcome.decision == "deferred"
    assert w.store.pending() == [candidate]  # tried again on the next run


def test_personal_data_never_reaches_the_judge(tmp_path):
    df = pd.DataFrame({"user_id": [1], "email": ["john.doe@example.com"], "revenue": [5.0]})
    w = World(tmp_path, df=df)
    add(w.store, question="Top customer of john.doe@example.com")
    w.run()
    (prompt,) = w.prompts
    assert "john.doe@example.com" not in prompt and "[REDACTED]" in prompt


def test_authors_are_rate_limited_per_day(tmp_path):
    w = World(tmp_path, policy=PromotionPolicy(max_per_author_per_day=2))
    for i in range(3):
        add(w.store, trace_id=f"t{i}", question=f"Monthly revenue in 202{i}")
    add(w.store, trace_id="other", user_id="outdoor_manager", question="Revenue by brand")
    outcomes = w.run()
    assert len(outcomes) == 3
    (waiting,) = w.store.pending()
    assert waiting.user_id == "denim_manager"
    assert w.run() == []  # the allowance for today is used up


def test_sampled_trios_and_quarantine_form_the_audit_queue(tmp_path):
    unsure = GOOD.model_copy(update={"answers_question": 3})
    w = World(tmp_path, policy=PromotionPolicy(audit_share=1.0))
    w.review = lambda prompt: unsure if "by brand" in prompt else GOOD
    add(w.store, trace_id="a")
    add(w.store, trace_id="b", question="Revenue by brand in 2024")
    w.run()
    quarantined, sampled = audit_queue(w.store, w.trios_path)
    assert [c.trace_id for c in quarantined] == ["b"]
    assert len(sampled) == 1 and sampled[0].source["approved_by"] == "auto"


# -- deprecation ---------------------------------------------------------------------------------


def test_trio_is_deprecated_after_bad_ratings_from_two_users(tmp_path):
    storage = Storage(tmp_path / "app.db")
    trios_path = tmp_path / "trios.json"
    golden.save_trios(trios_path, TRIOS)
    index = make_index(tmp_path, embeddings=BagOfWords())
    trio_id = TRIOS[0].id

    def rate(user, rating, trace):
        return record_feedback(
            storage, trios_path, trio_ids=[trio_id], user_id=user, rating=rating,
            trace_id=trace, index=index,
        )  # fmt: skip

    assert rate("alice", "bad", "t1") == []
    assert rate("alice", "bad", "t2") == []  # the same user again is not enough
    assert rate("bob", "good", "t3") == []
    assert rate("bob", "bad", "t3") == [trio_id]  # re-rating an answer replaces the rating
    deprecated = next(t for t in load_trios(trios_path) if t.id == trio_id)
    assert deprecated.status == "deprecated" and "2 users" in deprecated.source["deprecated_why"]
    assert trio_id not in [t.id for t in index.trios]


def test_good_ratings_outweigh_bad_ones(tmp_path):
    storage = Storage(tmp_path / "app.db")
    trios_path = tmp_path / "trios.json"
    golden.save_trios(trios_path, TRIOS)
    trio_id = TRIOS[0].id
    for i, (user, rating) in enumerate(
        [("a", "good"), ("b", "good"), ("c", "good"), ("d", "bad"), ("e", "bad")]
    ):
        record_feedback(
            storage, trios_path, trio_ids=[trio_id], user_id=user, rating=rating, trace_id=str(i)
        )
    assert next(t for t in load_trios(trios_path) if t.id == trio_id).status == "approved"


def test_bad_rating_in_the_chat_counts_against_the_trios_in_the_prompt(tmp_path):
    trios = [
        Trio("revenue", "Monthly revenue", (REVENUE_SQL,), "Sum sale_price.", ("revenue",)),
        Trio("returns", "Return rate by category", ("SELECT 1",), "Returned / not cancelled."),
    ]
    golden.save_trios(tmp_path / "trios.json", trios)
    index = make_index(tmp_path, embeddings=BagOfWords(), trios=trios, min_score=0.3)
    h = Harness(tmp_path, golden=index, golden_trios_path=tmp_path / "trios.json")
    h.turn("monthly revenue for 2024", sql_call(REVENUE_SQL), AIMessage("Revenue: $300."))
    assert h.session.state()["golden_ids"] == ["revenue"]
    h.session.mark_bad("wrong numbers")
    assert h.storage.trio_feedback_counts("revenue") == (0, 1, 1)
    assert h.storage.trio_feedback_counts("returns") == (0, 0, 0)


def test_candidate_records_the_authors_scope(tmp_path):
    diesel = UserContext("diesel_manager", "Dee", "manager", ("Diesel",))
    h = Harness(tmp_path, user=diesel)
    h.turn("monthly revenue for 2024", sql_call(REVENUE_SQL), AIMessage("Revenue: $300."))
    result = h.session.mark_good()
    assert "checked automatically" in result.message
    (candidate,) = h.candidates.pending()
    assert candidate.user_id == "diesel_manager" and candidate.brands == ("Diesel",)


def test_now_is_injectable_for_the_daily_allowance(tmp_path):
    w = World(tmp_path)
    w.promoter.now = lambda: datetime(2030, 1, 1, tzinfo=UTC)
    add(w.store)
    assert w.run()[0].decision == "approved"
