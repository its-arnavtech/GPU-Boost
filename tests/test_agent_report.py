"""Tests for Phase 5.7 agent report builder."""

from __future__ import annotations

from gpuboost.agent.report import (
    AgentReport,
    AgentReportSection,
    build_agent_report,
    count_actions_by_status,
    failed_action_messages,
)
from gpuboost.schemas.agent import (
    AgentAction,
    AgentEvent,
    AgentGoal,
    AgentPlan,
    AgentRunResult,
)


def test_agent_report_section_creation() -> None:
    section = AgentReportSection(
        title="Goal",
        items=["Kind: optimize_script"],
    )

    assert section.title == "Goal"
    assert section.items == ["Kind: optimize_script"]


def test_agent_report_to_dict() -> None:
    report = AgentReport(
        title="GPUBoost Agent Report",
        status="ok",
        summary="The agent workflow completed successfully.",
        sections=[AgentReportSection(title="Goal", items=["Kind: optimize_script"])],
        warnings=["Review generated diff."],
        error=None,
    )

    data = report.to_dict()

    assert data["title"] == "GPUBoost Agent Report"
    assert data["status"] == "ok"
    assert data["sections"][0]["title"] == "Goal"
    assert data["warnings"] == ["Review generated diff."]
    assert data["error"] is None


def test_build_report_for_ok_result() -> None:
    report = build_agent_report(_make_result(status="ok"))

    assert report.status == "ok"
    assert report.summary == "The agent workflow completed successfully."
    assert _section_titles(report)[:2] == ["Goal", "Plan"]


def test_build_report_for_partial_result() -> None:
    report = build_agent_report(
        _make_result(
            status="partial",
            actions=[
                _make_action("optional_action", status="failed", required=False),
                _make_action("summarize_results", status="completed"),
            ],
        ),
    )

    assert report.status == "partial"
    assert report.summary == (
        "The agent workflow completed with some non-fatal failures."
    )


def test_build_report_for_error_result() -> None:
    report = build_agent_report(
        _make_result(
            status="error",
            actions=[_make_action("inspect_system", status="failed")],
            error="No handler registered for action: inspect_system",
        ),
    )

    assert report.status == "error"
    assert report.summary == "The agent workflow failed."
    assert report.error == "No handler registered for action: inspect_system"


def test_goal_section_includes_script_path() -> None:
    report = build_agent_report(_make_result())
    goal_section = _section(report, "Goal")

    assert "Script path: train.py" in goal_section.items


def test_plan_section_includes_action_statuses() -> None:
    report = build_agent_report(
        _make_result(
            actions=[
                _make_action("inspect_system", status="completed"),
                _make_action("analyze_code", status="skipped"),
            ],
        ),
    )
    plan_section = _section(report, "Plan")

    assert "Action count: 2" in plan_section.items
    assert "inspect_system: inspect_system [completed]" in plan_section.items
    assert "analyze_code: analyze_code [skipped]" in plan_section.items


def test_warning_section_appears_when_warnings_exist() -> None:
    report = build_agent_report(
        _make_result(warnings=["Patch plan contains no edits."]),
    )

    warning_section = _section(report, "Warnings")

    assert warning_section.items == ["Patch plan contains no edits."]
    assert report.warnings == ["Patch plan contains no edits."]


def test_error_section_appears_when_failed_action_exists() -> None:
    report = build_agent_report(
        _make_result(
            status="partial",
            actions=[
                _make_action(
                    "analyze_code",
                    status="failed",
                    error="Code analysis failed.",
                    required=False,
                ),
            ],
        ),
    )

    error_section = _section(report, "Errors")

    assert error_section.items == ["analyze_code: Code analysis failed."]


def test_events_section_includes_last_five_events_only() -> None:
    events = [
        _make_event(index)
        for index in range(7)
    ]
    report = build_agent_report(_make_result(events=events))
    events_section = _section(report, "Events")

    assert events_section.items == [
        "info: Event 2",
        "info: Event 3",
        "info: Event 4",
        "info: Event 5",
        "info: Event 6",
    ]


def test_count_actions_by_status_works() -> None:
    result = _make_result(
        actions=[
            _make_action("inspect_system", status="completed"),
            _make_action("analyze_code", status="failed", required=False),
            _make_action("generate_diff", status="skipped", required=False),
            _make_action("summarize_results", status="completed"),
        ],
    )

    assert count_actions_by_status(result) == {
        "completed": 2,
        "failed": 1,
        "skipped": 1,
    }


def test_failed_action_messages_works() -> None:
    result = _make_result(
        actions=[
            _make_action(
                "analyze_code",
                status="failed",
                error="Code analysis failed.",
                required=False,
            ),
            _make_action("generate_diff", status="failed", required=False),
        ],
    )

    assert failed_action_messages(result) == [
        "analyze_code: Code analysis failed.",
        "generate_diff: failed",
    ]


def test_report_handles_empty_action_list() -> None:
    report = build_agent_report(_make_result(actions=[]))
    plan_section = _section(report, "Plan")

    assert plan_section.items == ["Action count: 0"]
    assert "Results" not in _section_titles(report)


def test_report_handles_no_events() -> None:
    report = build_agent_report(_make_result(events=[]))

    assert "Events" not in _section_titles(report)


def _make_goal() -> AgentGoal:
    return AgentGoal(
        id="goal_001",
        kind="optimize_script",
        description="Optimize train.py for NVIDIA GPU performance",
        script_path="train.py",
    )


def _make_action(
    action_id: str,
    *,
    status: str = "completed",
    error: str | None = None,
    required: bool = True,
) -> AgentAction:
    return AgentAction(
        id=action_id,
        name=action_id,
        description=f"Synthetic action: {action_id}",
        required=required,
        status=status,
        error=error,
    )


def _make_event(index: int = 0) -> AgentEvent:
    return AgentEvent(
        timestamp="2026-01-01T00:00:00+00:00",
        action_id=None,
        level="info",
        message=f"Event {index}",
    )


def _make_result(
    *,
    status: str = "ok",
    actions: list[AgentAction] | None = None,
    events: list[AgentEvent] | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> AgentRunResult:
    goal = _make_goal()
    return AgentRunResult(
        generated_at="2026-01-01T00:00:00+00:00",
        goal=goal,
        plan=AgentPlan(
            id="plan_goal_001",
            goal=goal,
            actions=actions
            if actions is not None
            else [_make_action("inspect_system")],
        ),
        status=status,
        events=events if events is not None else [_make_event()],
        warnings=warnings or [],
        error=error,
    )


def _section(report: AgentReport, title: str) -> AgentReportSection:
    for section in report.sections:
        if section.title == title:
            return section
    raise AssertionError(f"Missing report section: {title}")


def _section_titles(report: AgentReport) -> list[str]:
    return [section.title for section in report.sections]
