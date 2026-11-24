"""Graph test harness: scripted agent model, fake aux model, mocked BigQuery, real SQLite."""

from collections.abc import Callable
from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from retail_agent.auth import UserContext
from retail_agent.data.bigquery import QueryResult
from retail_agent.data.storage import Storage
from retail_agent.graph.builder import AgentSession, open_checkpointer
from retail_agent.graph.memory import ContextExtraction, HistorySummary
from retail_agent.graph.nodes import AgentDeps
from retail_agent.knowledge.golden import CandidateStore
from retail_agent.knowledge.schema import SchemaCatalog
from retail_agent.personalization.profile import Preferences, load_config
from retail_agent.prompts.context import PromptSources, ReloadingFile, glossary_text, persona_text
from retail_agent.safety.input_guard import GuardVerdict
from retail_agent.settings import Settings
from retail_agent.tools import ToolContext, build_tools

DATASET = "bigquery-public-data.thelook_ecommerce"
CEO = UserContext("ceo", "Alice", "ceo")
REVENUE_SQL = (
    "SELECT FORMAT_DATE('%Y-%m', DATE(created_at)) AS month, SUM(sale_price) AS revenue "
    f"FROM `{DATASET}.order_items` WHERE EXTRACT(YEAR FROM created_at) = 2024 GROUP BY month"
)

Reply = AIMessage | Callable[[list[BaseMessage]], AIMessage]


class ScriptedChat(BaseChatModel):
    """Answers with the scripted replies in order and records every request."""

    script: list[Any] = Field(default_factory=list)
    requests: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        if not self.script:
            raise AssertionError("the agent model was called more times than scripted")
        item = self.script.pop(0)
        reply = item(messages) if callable(item) else item
        return ChatResult(generations=[ChatGeneration(message=reply.model_copy())])

    def system_prompt(self, i: int = -1) -> str:
        first = self.requests[i][0]
        assert isinstance(first, SystemMessage)
        return first.text


def tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


def calls(*items: dict[str, Any]) -> AIMessage:
    """An AI message with the given tool calls (made with `tool_call`)."""
    return AIMessage(content="", tool_calls=list(items))


def sql_call(sql: str, call_id: str = "c1") -> AIMessage:
    return calls(tool_call("run_sql", {"sql": sql, "purpose": "revenue"}, call_id))


class FakeAux:
    """Aux model stand-in: schema -> handler(prompt) returning an instance of the schema."""

    def __init__(self) -> None:
        self.prompts: dict[type, list[str]] = {}
        self.handlers: dict[type, Callable[[str], Any]] = {
            GuardVerdict: lambda p: GuardVerdict(category="analysis", reason="ok"),
            ContextExtraction: lambda p: ContextExtraction(
                question="Monthly revenue in 2024", period="2024, monthly", entities=["revenue"]
            ),
            HistorySummary: lambda p: HistorySummary(summary="User asked about revenue."),
        }

    def __call__(self, schema):
        def run(messages):
            prompt = messages[0].text
            self.prompts.setdefault(schema, []).append(prompt)
            return self.handlers[schema](prompt)

        return RunnableLambda(run)


def make_settings(tmp_path, **kwargs) -> Settings:
    defaults = dict(
        bq_dataset=DATASET,
        log_path=tmp_path / "agent.jsonl",
        log_llm_payloads=False,
        checkpoint_db_path=tmp_path / "checkpoints.db",
        app_db_path=tmp_path / "app.db",
        exports_dir=tmp_path / "exports",
        max_agent_steps=5,
        history_keep_turns=2,
        history_summarize_after_turns=3,
        result_rows_to_llm=50,
        dev_jwt_secret_path=tmp_path / "dev_jwt_secret",
    )
    return Settings(_env_file=None, **{**defaults, **kwargs})


def make_schema() -> SchemaCatalog:
    meta = [
        {
            "name": "order_items",
            "num_rows": 10,
            "columns": [
                {"name": "sale_price", "type": "FLOAT"},
                {"name": "created_at", "type": "TIMESTAMP"},
            ],
        }
    ]
    return SchemaCatalog.from_metadata(DATASET, meta)


def revenue_df(months: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {"month": [f"2024-{m:02d}" for m in range(1, months + 1)], "revenue": [100.0] * months}
    )


class Harness:
    def __init__(
        self, tmp_path, user=CEO, aux=True, conversation_id="conv1", golden=None, **settings
    ):
        self.tmp_path = tmp_path
        self.golden = golden
        self.candidates = CandidateStore(tmp_path / "golden_candidates.jsonl")
        self.settings = make_settings(tmp_path, **settings)
        self.runner = MagicMock()
        self.runner.dataset_id = DATASET
        self.runner.dry_run.return_value = 100
        self.runner.execute.return_value = QueryResult(
            df=revenue_df(), sql="", bytes_processed=100, duration_ms=1, cache_hit=False
        )
        self.storage = Storage(self.settings.app_db_path)
        self.preferences = Preferences(self.storage, load_config())
        self.model = ScriptedChat()
        self.aux = FakeAux() if aux else None
        self.persona = tmp_path / "persona.md"
        self.persona.write_text("Be brief.", encoding="utf-8")
        glossary = tmp_path / "glossary.yaml"
        glossary.write_text(
            "terms:\n  churn:\n    definition: no orders this month\n", encoding="utf-8"
        )
        self.sources = PromptSources(
            persona=ReloadingFile(self.persona, "persona", persona_text),
            glossary=ReloadingFile(glossary, "glossary", glossary_text),
        )
        self.user = user
        self.conversation_id = conversation_id
        self.session = self.new_session()

    def new_session(self, user: UserContext | None = None) -> AgentSession:
        ctx = ToolContext(
            user=user or self.user,
            conversation_id="",
            runner=self.runner,
            schema=make_schema(),
            settings=self.settings,
            storage=self.storage,
            preferences=self.preferences,
        )
        deps = AgentDeps(
            tool_ctx=ctx,
            tools=build_tools(ctx),
            agent_model=self.model,
            settings=self.settings,
            aux=self.aux,
            prompt_sources=self.sources,
            today=lambda: date(2026, 11, 20),
            golden=self.golden,
            candidates=self.candidates,
        )
        checkpointer = open_checkpointer(self.settings.checkpoint_db_path)
        return AgentSession(deps, checkpointer, self.conversation_id)

    def turn(self, text: str, *replies: Reply):
        self.model.script.extend(replies)
        return self.session.run_turn(text)

    def resume(self, answer: str, *replies: Reply):
        """Answer every pending confirmation with `answer`."""
        self.model.script.extend(replies)
        pending = self.session.pending()
        return self.session.resume({p.interrupt_id: answer for p in pending})

    @property
    def messages(self) -> list[BaseMessage]:
        return self.session.state()["messages"]
