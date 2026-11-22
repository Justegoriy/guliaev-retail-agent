import json
import zlib

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from harness import REVENUE_SQL, Harness, sql_call
from retail_agent import auth
from retail_agent.errors import SqlRejected
from retail_agent.graph.memory import ContextExtraction
from retail_agent.knowledge import golden
from retail_agent.knowledge.golden import (
    CandidateStore,
    GoldenIndex,
    Trio,
    approve_candidate,
    load_trios,
    reject_candidate,
    slugify,
)
from retail_agent.observability import logger
from retail_agent.safety import sql_guard
from retail_agent.safety.input_guard import GuardVerdict
from retail_agent.settings import PROJECT_ROOT

TRIOS_FILE = PROJECT_ROOT / "data" / "golden_trios.json"
DATASET = "bigquery-public-data.thelook_ecommerce"
EMAIL = "john.doe@example.com"


class BagOfWords(Embeddings):
    """Deterministic stand-in for Gemini embeddings: hashed keyword counts."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.document_calls = 0
        self.query_calls = 0
        self.fail_queries = False
        self.fail_documents = False

    def _vector(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for word in golden.keywords(text):
            v[zlib.crc32(word.encode()) % self.dim] += 1
        return v.tolist()

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.fail_documents:
            raise TimeoutError("embeddings down")
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        self.query_calls += 1
        if self.fail_queries:
            raise TimeoutError("embeddings down")
        return self._vector(text)


def trio(id_: str, question: str, tags=(), status="approved") -> Trio:
    return Trio(
        id=id_, question=question, sql=("SELECT 1",), report="notes", tags=tags, status=status
    )


TRIOS = [
    trio("churn", "What is our monthly customer churn rate?", ("churn", "retention")),
    trio("returns", "Which categories have the highest return rate?", ("returns", "category")),
    trio("aov", "What is the average order value by traffic source?", ("aov", "marketing")),
    trio("old", "Old churn definition", ("churn",), status="deprecated"),
]


def make_index(tmp_path, embeddings=None, trios=TRIOS, **kwargs) -> GoldenIndex:
    return GoldenIndex(
        trios,
        embeddings=embeddings,
        index_path=tmp_path / "golden_index.npz",
        model_name="fake",
        min_score=kwargs.pop("min_score", 0.3),
        **kwargs,
    )


@pytest.fixture
def events(tmp_path):
    logger.configure(tmp_path / "agent.jsonl")
    yield lambda: [
        json.loads(line)
        for line in (tmp_path / "agent.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    logger._log_path = None


# -- seed trios ------------------------------------------------------------------------------


def test_seed_trios_are_valid():
    trios = load_trios(TRIOS_FILE)
    assert 10 <= len(trios) <= 15
    assert all(t.status == "approved" and t.tags and t.report for t in trios)
    ids = {t.id for t in trios}
    for topic in (
        "revenue_monthly",
        "top_customers",
        "states_underspending",
        "churn_monthly",
        "brand_comparison",
        "return_rate_category",
        "aov_traffic_source",
        "quarterly_report",
        "new_customer_cohorts",
        "margin_by_category",
    ):
        assert topic in ids


@pytest.mark.parametrize("user_id", ["ceo", "bottoms_manager"])
def test_seed_sql_passes_the_guard(user_id):
    user = auth.load_user(user_id)
    for t in load_trios(TRIOS_FILE):
        for sql in t.sql:
            sql_guard.check_and_rewrite(sql, user, dataset=DATASET, default_limit=1000)


@pytest.mark.network
def test_seed_sql_dry_runs_on_bigquery():
    from retail_agent.data.bigquery import BigQueryRunner
    from retail_agent.settings import get_settings

    runner = BigQueryRunner.from_settings(get_settings())
    ceo = auth.load_user("ceo")
    for t in load_trios(TRIOS_FILE):
        for sql in t.sql:
            guarded = sql_guard.check_and_rewrite(sql, ceo, dataset=DATASET, default_limit=1000)
            assert runner.dry_run(guarded.sql) > 0, t.id


def test_load_trios_rejects_bad_files(tmp_path):
    path = tmp_path / "trios.json"
    path.write_text(json.dumps([t.to_json() for t in (TRIOS[0], TRIOS[0])]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate trio id"):
        load_trios(path)
    path.write_text(json.dumps([{**TRIOS[0].to_json(), "sql": []}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_trios(path)


# -- retrieval -------------------------------------------------------------------------------------


def test_embedding_retrieval_ranks_and_skips_deprecated(tmp_path, events):
    index = make_index(tmp_path, BagOfWords())
    assert index.mode == "embedding" and "old" not in [t.id for t in index.trios]
    hits = index.retrieve("Why did customer churn grow last month?")
    assert hits[0].trio.id == "churn" and hits[0].method == "embedding"
    assert index.retrieve("What's the weather in Paris tomorrow?") == []
    logged = [e for e in events() if e["event"] == "golden_retrieved"]
    assert logged[0]["ids"][0] == "churn" and logged[0]["method"] == "embedding"
    assert logged[1]["ids"] == [] and logged[1]["best_score"] < logged[1]["threshold"]


def test_index_is_cached_until_trios_change(tmp_path, events):
    first = BagOfWords()
    make_index(tmp_path, first)
    assert first.document_calls == 1

    again = BagOfWords()
    index = make_index(tmp_path, again)
    assert again.document_calls == 0 and index.mode == "embedding"
    assert index.retrieve("churn rate")[0].trio.id == "churn"

    changed = BagOfWords()
    make_index(tmp_path, changed, trios=[*TRIOS, trio("margin", "Gross margin by category")])
    assert changed.document_calls == 1
    names = [e["event"] for e in events()]
    assert names.count("golden_index_built") == 2 and names.count("golden_index_loaded") == 1


def test_query_failure_falls_back_to_keywords_and_cools_down(tmp_path, events):
    now = [0.0]
    embeddings = BagOfWords()
    index = make_index(tmp_path, embeddings, clock=lambda: now[0])
    embeddings.fail_queries = True
    hits = index.retrieve("monthly churn rate")
    assert hits[0].trio.id == "churn" and hits[0].method == "keywords"
    index.retrieve("return rate by category")
    assert embeddings.query_calls == 1  # skipped during the cooldown
    now[0] += golden.EMBEDDING_COOLDOWN_S + 1
    embeddings.fail_queries = False
    assert index.retrieve("monthly churn rate")[0].method == "embedding"
    degraded = [e for e in events() if e["event"] == "golden_degraded"]
    assert len(degraded) == 1 and degraded[0]["stage"] == "query"


def test_index_build_failure_means_keyword_mode(tmp_path, events):
    embeddings = BagOfWords()
    embeddings.fail_documents = True
    index = make_index(tmp_path, embeddings)
    assert index.mode == "keywords"
    assert index.retrieve("which categories are returned most?")[0].trio.id == "returns"
    assert embeddings.query_calls == 0
    assert any(e["event"] == "golden_degraded" and e["stage"] == "index" for e in events())


def test_keyword_retrieval_without_embeddings(tmp_path):
    index = make_index(tmp_path, None)
    assert [h.trio.id for h in index.retrieve("average order value per traffic source")] == ["aov"]
    assert index.retrieve("hello there") == []
    assert index.retrieve("   ") == []


def test_missing_trios_file_gives_an_empty_index(tmp_path, events):
    index = GoldenIndex.from_file(
        tmp_path / "missing.json", embeddings=BagOfWords(), index_path=None
    )
    assert index.trios == [] and index.retrieve("churn") == []
    assert any(e["event"] == "golden_load_failed" for e in events())


def test_precedents_block():
    two_queries = Trio("drivers", "Why did churn grow?", ("SELECT 1", "SELECT 2"), "Decompose.")
    text = golden.precedents_block([golden.Retrieved(two_queries, 0.81, "embedding")])
    assert text.startswith(golden.PRECEDENTS_INTRO)
    assert "## drivers (similarity 0.81)" in text
    assert "SQL (2 queries, one per step):" in text and "```sql\nSELECT 2\n```" in text
    assert "Analyst notes: Decompose." in text


# -- candidates ----------------------------------------------------------------------------


def add(store: CandidateStore, trace_id="t1", **kwargs):
    defaults = dict(
        question="Monthly revenue in 2024",
        sql=[REVENUE_SQL],
        answer_summary=f"Revenue grew; top buyer {EMAIL}",
        user_role="ceo",
        trace_id=trace_id,
        tags=["revenue", "monthly"],
    )
    return store.add(**{**defaults, **kwargs})


def test_candidate_is_masked_and_deduplicated(tmp_path):
    store = CandidateStore(tmp_path / "cand.jsonl")
    candidate = add(store)
    assert candidate.status == "candidate" and candidate.id.startswith("cand_")
    assert EMAIL not in candidate.answer_summary and "[REDACTED]" in candidate.answer_summary
    assert add(store) is None
    assert EMAIL not in (tmp_path / "cand.jsonl").read_text(encoding="utf-8")
    assert store.pending() == [candidate]


def test_approved_candidate_becomes_a_retrievable_trio(tmp_path):
    trios_path = tmp_path / "trios.json"
    golden.save_trios(trios_path, TRIOS[:2])
    store = CandidateStore(tmp_path / "cand.jsonl")
    candidate = add(store)
    validated = []

    trio_ = approve_candidate(
        candidate, store=store, trios_path=trios_path, validate_sql=validated.append
    )
    assert validated == [REVENUE_SQL]
    assert trio_.id == "monthly_revenue_in_2024" and trio_.status == "approved"
    assert trio_.source["candidate_id"] == candidate.id and trio_.source["trace_id"] == "t1"
    assert [t.id for t in load_trios(trios_path)][-1] == trio_.id
    (stored,) = store.all()
    assert stored.status == "approved" and stored.trio_id == trio_.id
    assert store.pending() == []

    # The next start re-embeds the changed file and finds the new trio.
    index = GoldenIndex.from_file(
        trios_path, embeddings=BagOfWords(), index_path=None, min_score=0.3
    )
    assert index.retrieve("revenue by month for 2024")[0].trio.id == trio_.id


def test_approval_with_edits_and_unique_id(tmp_path):
    trios_path = tmp_path / "trios.json"
    golden.save_trios(trios_path, [trio("monthly_revenue", "x")])
    store = CandidateStore(tmp_path / "cand.jsonl")
    new = approve_candidate(
        add(store),
        store=store,
        trios_path=trios_path,
        validate_sql=lambda sql: None,
        trio_id="monthly revenue",
        report="Revenue = sold items.",
        sql=["SELECT 1", "SELECT 2"],
        tags=["revenue"],
    )
    assert new.id == "monthly_revenue_2" and new.sql == ("SELECT 1", "SELECT 2")
    assert new.report == "Revenue = sold items." and new.tags == ("revenue",)


def test_invalid_sql_is_not_approved(tmp_path):
    trios_path = tmp_path / "trios.json"
    golden.save_trios(trios_path, TRIOS[:1])
    store = CandidateStore(tmp_path / "cand.jsonl")
    candidate = add(store, sql=["SELECT email FROM users"])

    def validate(sql):
        raise SqlRejected("personal data column email in the output")

    with pytest.raises(SqlRejected):
        approve_candidate(candidate, store=store, trios_path=trios_path, validate_sql=validate)
    assert len(load_trios(trios_path)) == 1
    assert store.pending() == [candidate]


def test_reject_candidate(tmp_path):
    store = CandidateStore(tmp_path / "cand.jsonl")
    candidate = add(store)
    reject_candidate(candidate, store=store, note="too specific")
    (stored,) = store.all()
    assert stored.status == "rejected" and stored.note == "too specific" and stored.reviewed_at


def test_slugify_cuts_on_word_boundary():
    assert slugify("What is the gender split of our buyers in 2025 by revenue?") == (
        "what_is_the_gender_split_of_our_buyers_in_2025"
    )
    assert slugify("Churn, by month!") == "churn_by_month"
    assert slugify("x" * 60) == "x" * 48
    assert slugify("???") == "trio"


# -- in the graph --------------------------------------------------------------------------


def golden_harness(tmp_path) -> Harness:
    index = make_index(tmp_path, BagOfWords())
    return Harness(tmp_path, golden=index)


def test_precedents_reach_the_prompt(tmp_path, events):
    h = golden_harness(tmp_path)
    h.turn("Why did customer churn increase last month?", AIMessage("Let me check."))
    prompt = h.model.system_prompt()
    assert "# Analyst precedents" in prompt and "## churn (similarity" in prompt
    assert prompt.index("# Database schema") < prompt.index("# Analyst precedents")
    built = next(e for e in events() if e["event"] == "prompt_built")
    assert built["golden_ids"][0] == "churn"


def test_follow_up_retrieval_uses_the_analysis_topic(tmp_path):
    h = golden_harness(tmp_path)
    h.aux.handlers[ContextExtraction] = lambda p: ContextExtraction(
        question="Monthly customer churn rate in 2025"
    )
    h.turn("churn by month in 2025", sql_call(REVENUE_SQL), AIMessage("Churn: ..."))
    h.turn("and for Swim?", AIMessage("ok"))
    assert "## churn (similarity" in h.model.system_prompt()


def test_no_precedents_for_greetings(tmp_path):
    h = golden_harness(tmp_path)
    h.aux.handlers[GuardVerdict] = lambda p: GuardVerdict(category="greeting", reason="hi")
    h.turn("hi, what is churn?", AIMessage("Hello!"))
    assert "# Analyst precedents" not in h.model.system_prompt()
    assert h.golden._embeddings.query_calls == 0


def test_good_turns_a_successful_answer_into_a_candidate(tmp_path, events):
    h = Harness(tmp_path)
    out = h.turn("monthly revenue for 2024", sql_call(REVENUE_SQL), AIMessage("Revenue: $300."))
    result = h.session.mark_good()
    assert result.candidate_id
    (candidate,) = h.candidates.pending()
    assert candidate.question == "Monthly revenue in 2024"  # standalone, from the context
    assert candidate.sql == (REVENUE_SQL,) and candidate.answer_summary == "Revenue: $300."
    assert candidate.trace_id == out.trace_id and candidate.user_role == "ceo"
    assert candidate.tags == ("revenue",)
    assert "already waiting" in h.session.mark_good().message
    feedback = [e for e in events() if e["event"] == "feedback"]
    assert feedback and feedback[0]["trace_id"] == out.trace_id


def test_good_without_sql_records_nothing(tmp_path):
    h = Harness(tmp_path)
    h.turn("monthly revenue for 2024", sql_call(REVENUE_SQL), AIMessage("Revenue: $300."))
    h.turn("thanks!", AIMessage("You're welcome."))
    result = h.session.mark_good()
    assert result.candidate_id is None and "did not run a data query" in result.message
    assert h.candidates.all() == []


def test_good_after_failed_sql_records_nothing(tmp_path):
    h = Harness(tmp_path)
    h.turn("emails please", sql_call("SELECT email FROM users"), AIMessage("Not available."))
    assert h.session.mark_good().candidate_id is None
