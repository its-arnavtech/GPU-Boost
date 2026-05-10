"""Human-readable report building for GPUBoost agent runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gpuboost.schemas.agent import (
    AgentAction,
    AgentEvent,
    AgentRunResult,
)


@dataclass(slots=True)
class AgentReportSection:
    """One stable section in an agent report."""

    title: str
    items: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentReport:
    """Structured human-readable summary of an agent run."""

    title: str
    status: str
    summary: str
    sections: list[AgentReportSection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the report as JSON-serializable data."""

        return asdict(self)


def build_agent_report(result: AgentRunResult) -> AgentReport:
    """Build a concise structured report from an agent run result."""

    sections = [
        _build_goal_section(result),
        _build_plan_section(result),
    ]

    results_section = _build_results_section(result)
    if results_section is not None:
        sections.append(results_section)

    if result.warnings:
        sections.append(
            AgentReportSection(
                title="Warnings",
                items=list(result.warnings),
            ),
        )

    errors = failed_action_messages(result)
    if result.error:
        errors.insert(0, result.error)
    if errors:
        sections.append(
            AgentReportSection(
                title="Errors",
                items=errors,
            ),
        )

    if result.events:
        sections.append(
            AgentReportSection(
                title="Events",
                items=[format_event_line(event) for event in result.events[-5:]],
            ),
        )

    return AgentReport(
        title="GPUBoost Agent Report",
        status=result.status,
        summary=_summary_for_status(result.status),
        sections=sections,
        warnings=list(result.warnings),
        error=result.error,
    )


def count_actions_by_status(result: AgentRunResult) -> dict[str, int]:
    """Return action counts grouped by action status."""

    counts: dict[str, int] = {}
    for action in result.plan.actions:
        counts[action.status] = counts.get(action.status, 0) + 1
    return counts


def failed_action_messages(result: AgentRunResult) -> list[str]:
    """Return stable failure messages for failed actions."""

    messages = []
    for action in result.plan.actions:
        if action.status != "failed":
            continue

        if action.error:
            messages.append(f"{action.id}: {action.error}")
        else:
            messages.append(f"{action.id}: failed")
    return messages


def format_action_line(action: AgentAction) -> str:
    """Format one action for report display."""

    return f"{action.id}: {action.name} [{action.status}]"


def format_event_line(event: AgentEvent) -> str:
    """Format one event for report display."""

    return f"{event.level}: {event.message}"


def _build_goal_section(result: AgentRunResult) -> AgentReportSection:
    items = [
        f"Kind: {result.goal.kind}",
        f"Description: {result.goal.description}",
    ]
    if result.goal.script_path:
        items.append(f"Script path: {result.goal.script_path}")

    return AgentReportSection(title="Goal", items=items)


def _build_plan_section(result: AgentRunResult) -> AgentReportSection:
    items = [f"Action count: {len(result.plan.actions)}"]
    items.extend(format_action_line(action) for action in result.plan.actions)
    return AgentReportSection(title="Plan", items=items)


def _build_results_section(
    result: AgentRunResult,
) -> AgentReportSection | None:
    counts = count_actions_by_status(result)
    if not counts:
        return None

    items = [
        f"{status}: {count}"
        for status, count in sorted(counts.items())
    ]
    return AgentReportSection(title="Results", items=items)


def _summary_for_status(status: str) -> str:
    if status == "ok":
        return "The agent workflow completed successfully."
    if status == "partial":
        return "The agent workflow completed with some non-fatal failures."
    if status == "error":
        return "The agent workflow failed."
    return f"The agent workflow finished with status: {status}."
