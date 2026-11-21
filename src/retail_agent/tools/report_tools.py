"""Report library tools: save, list, search, read, delete (with confirmation), export.

Ownership is never an argument: the user comes from the session's `ToolContext` and every
storage call is scoped to them. Another user's report and a missing one get the same
"not found", so the tools do not reveal which reports exist.

Deletion and export pause the graph for the user's confirmation (`safety.confirmation`);
the tool is re-run on resume, so it recomputes its targets and checks they are still
exactly the ones shown in the preview.
"""

import hashlib
from collections.abc import Iterable, Sequence
from typing import Annotated, Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from retail_agent.data.storage import Report, Storage
from retail_agent.observability import logger
from retail_agent.observability.logger import log_event
from retail_agent.safety import confirmation, pii
from retail_agent.safety.confirmation import Decision
from retail_agent.tools.registry import ToolContext, ToolSpec, register

MAX_LISTED = 50
MAX_DELETION_CANDIDATES = 100
EXPORTS_HEADER = "<!-- exported by retail-agent; report {id}, created {created_at} -->\n"


def _storage(ctx: ToolContext) -> Storage:
    if ctx.storage is None:
        raise RuntimeError("report storage is not configured")
    return ctx.storage


def _audit(ctx: ToolContext) -> confirmation.AuditSink:
    def audit(action: str, target_ids: list[str], details: dict[str, Any]) -> None:
        _storage(ctx).audit(
            ctx.user.user_id, action, target_ids, details, trace_id=logger.trace_id_var.get()
        )

    return audit


def _clean(text: str) -> str:
    return pii.mask_text(text, where="report")[0]


def _line(report: Report, conversation_id: str | None = None) -> str:
    here = " (this conversation)" if report.conversation_id == conversation_id else ""
    return f"- {report.id} | {report.created_at[:16].replace('T', ' ')} | {report.title}{here}"


def _preview(report: Report) -> dict[str, Any]:
    return {
        "id": report.id,
        "title": report.title,
        "created_at": report.created_at,
        "snippet": report.snippet(),
    }


def digest(reports: Iterable[Report]) -> str:
    """Identity of the exact set (and versions) of reports the user is asked about."""
    key = "\n".join(sorted(f"{r.id}@{r.updated_at}" for r in reports))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


# -- save_report --------------------------------------------------------------------------


class SaveReportArgs(BaseModel):
    title: str = Field(
        min_length=1, max_length=200, description="Short title, e.g. 'Q1 2024 sales review'."
    )
    content_markdown: str = Field(min_length=1, description="The full report in Markdown.")
    entities: list[str] = Field(
        default_factory=list,
        description="Brands, products, categories, states, metrics, segments and customer "
        "ids (users.id) the report mentions; used to find the report later. No personal data.",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]
    analysis_context: Annotated[dict | None, InjectedState("analysis_context")] = None


def source_queries(ctx: ToolContext, analysis_context: dict | None) -> list[str]:
    """SQL behind the report: this turn's queries, then those of the analysis it continues."""
    this_turn = [q.original_sql or q.sql for q in ctx.turn.executed]
    earlier = (analysis_context or {}).get("sql") or []
    return _dedupe([*this_turn, *earlier])


def _save_report(ctx: ToolContext) -> BaseTool:
    def save_report(
        title: str,
        content_markdown: str,
        tool_call_id: str,
        entities: list[str] | None = None,
        analysis_context: dict | None = None,
    ) -> Command:
        report = _storage(ctx).save_report(
            owner_user_id=ctx.user.user_id,
            conversation_id=ctx.conversation_id,
            title=_clean(title),
            content=_clean(content_markdown),
            entities=_dedupe(_clean(e.strip()) for e in entities or []),
            source_queries=source_queries(ctx, analysis_context),
        )
        log_event(
            "report_saved",
            report_id=report.id,
            chars=len(report.content),
            entities=len(report.entities),
            source_queries=len(report.source_queries),
        )
        text = f"SAVED: report_id={report.id}, title={report.title!r}."
        return Command(
            update={
                "session_report_ids": [report.id],
                "messages": [ToolMessage(text, tool_call_id=tool_call_id, name="save_report")],
            }
        )

    return StructuredTool.from_function(
        func=save_report,
        name="save_report",
        description=(
            "Save a report to the user's report library. Call only when the user asked to "
            "save it or agreed to your offer. Returns the report id."
        ),
        args_schema=SaveReportArgs,
    )


# -- list / search / get --------------------------------------------------------------------


class ListReportsArgs(BaseModel):
    scope: Literal["all", "this_conversation"] = Field(
        "all", description="'this_conversation' for reports saved in this conversation only."
    )
    limit: int = Field(20, ge=1, le=MAX_LISTED, description="Maximum number of reports.")


def _list_reports(ctx: ToolContext) -> BaseTool:
    def list_reports(scope: str = "all", limit: int = 20) -> str:
        conversation = ctx.conversation_id if scope == "this_conversation" else None
        reports = _storage(ctx).list_reports(
            ctx.user.user_id, conversation_id=conversation, limit=limit
        )
        if not reports:
            return "EMPTY: no saved reports" + (" in this conversation." if conversation else ".")
        lines = [_line(r, ctx.conversation_id) for r in reports]
        return f"OK: {len(reports)} reports, newest first (id | created UTC | title)\n" + "\n".join(
            lines
        )

    return StructuredTool.from_function(
        func=list_reports,
        name="list_reports",
        description="List the user's saved reports (id, creation time, title), newest first.",
        args_schema=ListReportsArgs,
    )


class SearchReportsArgs(BaseModel):
    query: str = Field(
        min_length=1,
        description="Word or phrase to find in report titles, texts and entities, "
        "e.g. a brand, category, state or customer id.",
    )


def _search_reports(ctx: ToolContext) -> BaseTool:
    def search_reports(query: str) -> str:
        reports = _storage(ctx).search_reports(ctx.user.user_id, query, limit=MAX_LISTED)
        if not reports:
            return f"EMPTY: no saved reports mention {query!r}."
        lines = [_line(r, ctx.conversation_id) for r in reports]
        return f"OK: {len(reports)} reports mention {query!r} (id | created UTC | title)\n" + (
            "\n".join(lines)
        )

    return StructuredTool.from_function(
        func=search_reports,
        name="search_reports",
        description=(
            "Find the user's saved reports that mention a word or phrase in the title, "
            "the text or the entities (case-insensitive)."
        ),
        args_schema=SearchReportsArgs,
    )


class GetReportArgs(BaseModel):
    report_id: str = Field(description="Report id from save_report, list_reports or search.")


def _not_found(report_id: str) -> str:
    return f"NOT_FOUND: there is no report {report_id!r} among the user's reports."


def _get_report(ctx: ToolContext) -> BaseTool:
    def get_report(report_id: str) -> str:
        report = _storage(ctx).get_report(ctx.user.user_id, report_id.strip())
        if report is None:
            return _not_found(report_id)
        return (
            f"OK: report {report.id}, {report.title!r}, created {report.created_at}\n"
            f"Entities: {', '.join(report.entities) or 'none'}\n\n{report.content}"
        )

    return StructuredTool.from_function(
        func=get_report,
        name="get_report",
        description="Read one of the user's saved reports in full.",
        args_schema=GetReportArgs,
    )


# -- request_report_deletion ------------------------------------------------------------------


class DeletionArgs(BaseModel):
    report_ids: list[str] | None = Field(
        None, description="Ids of reports to delete, e.g. the report saved a moment ago."
    )
    mention: str | None = Field(
        None,
        description="Delete reports that mention this word or phrase (title, text or "
        "entities), e.g. 'client 123' or 'Levi's'.",
    )
    this_conversation: bool = Field(
        False, description="True to delete all reports saved in this conversation."
    )


def deletion_candidates(
    ctx: ToolContext,
    report_ids: Sequence[str] | None,
    mention: str | None,
    this_conversation: bool,
) -> tuple[list[Report], list[str]]:
    """The user's own reports matching any criterion, and requested ids that are not theirs."""
    storage, owner = _storage(ctx), ctx.user.user_id
    requested = _dedupe(i.strip() for i in report_ids or [])
    found: dict[str, Report] = {r.id: r for r in storage.get_reports(owner, requested)}
    not_found = [i for i in requested if i not in found]
    if mention and mention.strip():
        for report in storage.search_reports(owner, mention, limit=MAX_DELETION_CANDIDATES):
            found.setdefault(report.id, report)
    if this_conversation:
        conversation = storage.list_reports(
            owner, conversation_id=ctx.conversation_id, limit=MAX_DELETION_CANDIDATES
        )
        for report in conversation:
            found.setdefault(report.id, report)
    candidates = sorted(found.values(), key=lambda r: (r.created_at, r.id))
    return candidates, not_found


def _not_found_note(not_found: Sequence[str]) -> str:
    if not not_found:
        return ""
    return (
        f" Not among the user's reports (missing or someone else's, never deletable): "
        f"{', '.join(not_found)}."
    )


def _request_report_deletion(ctx: ToolContext) -> BaseTool:
    def request_report_deletion(
        report_ids: list[str] | None = None,
        mention: str | None = None,
        this_conversation: bool = False,
    ) -> str:
        if not report_ids and not (mention and mention.strip()) and not this_conversation:
            return (
                "ERROR(invalid_args): say which reports to delete: report_ids, mention "
                "or this_conversation=true."
            )
        candidates, not_found = deletion_candidates(ctx, report_ids, mention, this_conversation)
        note = _not_found_note(not_found)
        if not candidates:
            log_event("deletion_no_match", not_found=len(not_found))
            return f"NO_MATCH: no reports of the user match, nothing was deleted.{note}"

        ids = [r.id for r in candidates]
        set_digest = digest(candidates)
        log_event("deletion_requested", report_ids=ids, digest=set_digest)
        count = len(candidates)
        summary = f"{count} report{'s' if count != 1 else ''} will be deleted." + (
            f" Not found among your reports: {', '.join(not_found)}." if not_found else ""
        )
        decision = confirmation.require(
            "delete_reports",
            question=f"Delete {count} report{'s' if count != 1 else ''}?",
            summary=summary,
            items=[_preview(r) for r in candidates],
            digest=set_digest,
            audit=_audit(ctx),
        )
        if decision is Decision.STALE:
            log_event("deletion_cancelled", reason="stale", report_ids=ids)
            return (
                "CANCELLED: the set of matching reports changed before the user answered; "
                "nothing was deleted. Ask the user again."
            )
        if decision is Decision.REJECTED:
            log_event("deletion_cancelled", reason="user", report_ids=ids)
            return "CANCELLED by user: nothing was deleted."

        confirmation.ensure_approved("delete_reports")
        deleted = _storage(ctx).soft_delete_reports(ctx.user.user_id, ids)
        _audit(ctx)("reports_deleted", deleted, {"digest": set_digest, "soft": True})
        log_event("deletion_confirmed", report_ids=deleted)
        return f"DELETED: {len(deleted)} reports [{', '.join(deleted)}].{note}"

    return StructuredTool.from_function(
        func=request_report_deletion,
        name="request_report_deletion",
        description=(
            "Delete the user's own saved reports. Pass report_ids, a mention to search for, "
            "and/or this_conversation=true. The system shows the matching reports to the "
            "user and asks for confirmation itself; you cannot confirm. Results: "
            "DELETED, CANCELLED, NO_MATCH (nothing matched, nothing asked). Report the "
            "result as is."
        ),
        args_schema=DeletionArgs,
    )


# -- export_report ------------------------------------------------------------------------------


class ExportArgs(BaseModel):
    report_id: str = Field(description="Id of the user's report to export.")


def _export_report(ctx: ToolContext) -> BaseTool:
    def export_report(report_id: str) -> str:
        report = _storage(ctx).get_report(ctx.user.user_id, report_id.strip())
        if report is None:
            return _not_found(report_id)
        exports_dir = ctx.settings.exports_dir
        target = exports_dir / f"{report.id}.md"
        decision = confirmation.require(
            "export_report",
            question=f"Export report {report.id} to {target.name} in {exports_dir.name}/?",
            summary=f"The report will be written to {target}.",
            items=[_preview(report)],
            digest=digest([report]),
            audit=_audit(ctx),
        )
        if decision is not Decision.APPROVED:
            log_event("export_cancelled", report_id=report.id, reason=str(decision))
            return "CANCELLED: the report was not exported."

        confirmation.ensure_approved("export_report")
        exports_dir.mkdir(parents=True, exist_ok=True)
        header = EXPORTS_HEADER.format(id=report.id, created_at=report.created_at)
        target.write_text(f"{header}# {report.title}\n\n{report.content}\n", encoding="utf-8")
        _audit(ctx)("report_exported", [report.id], {"path": str(target)})
        log_event("report_exported", report_id=report.id, path=str(target))
        return f"EXPORTED: report {report.id} written to {target}."

    return StructuredTool.from_function(
        func=export_report,
        name="export_report",
        description=(
            "Export one of the user's saved reports to a Markdown file. The system asks "
            "the user for confirmation itself. Results: EXPORTED, CANCELLED, NOT_FOUND."
        ),
        args_schema=ExportArgs,
    )


register(
    [
        ToolSpec(name="save_report", factory=_save_report, side_effects=True),
        ToolSpec(name="list_reports", factory=_list_reports),
        ToolSpec(name="search_reports", factory=_search_reports),
        ToolSpec(name="get_report", factory=_get_report),
        ToolSpec(
            name="request_report_deletion",
            factory=_request_report_deletion,
            side_effects=True,
            requires_confirmation=True,
        ),
        ToolSpec(
            name="export_report",
            factory=_export_report,
            side_effects=True,
            requires_confirmation=True,
        ),
    ]
)
