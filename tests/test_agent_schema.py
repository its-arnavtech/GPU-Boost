"""Tests for Phase 5.1 agent schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from gpuboost.schemas.agent import (
    AgentAction,
    AgentEvent,
    AgentGoal,
    AgentPlan,
    AgentRunResult,
    create_timestamp,
)


def test_agent_goal_creation() -> None:
    goal = _make_goal()

    assert goal.id == "goal_001"
    assert goal.kind == "optimize_script"
    assert goal.description == "Optimize train.py for NVIDIA GPU performance"
    assert goal.script_path == "train.py"
    assert goal.options == {"quick": True}
    assert goal.constraints == ["do_not_modify_original_file"]


def test_agent_action_creation_with_defaults() -> None:
    action = AgentAction(
        id="action_001",
        name="inspect_system",
        description="Inspect the available GPU and software environment.",
        required=True,
    )

    assert action.id == "action_001"
    assert action.name == "inspect_system"
    assert action.required is True
    assert action.depends_on == []
    assert action.inputs == {}
    assert action.status == "pending"
    assert action.error is None


def test_agent_plan_creation() -> None:
    goal = _make_goal()
    action = _make_action()
    plan = AgentPlan(
        id="plan_001",
        goal=goal,
        actions=[action],
        warnings=["Review generated diffs before applying."],
    )

    assert plan.id == "plan_001"
    assert plan.goal == goal
    assert plan.actions == [action]
    assert plan.warnings == ["Review generated diffs before applying."]


def test_agent_event_creation() -> None:
    event = AgentEvent(
        timestamp="2026-01-01T00:00:00+00:00",
        action_id="action_001",
        level="info",
        message="System inspection completed.",
        data={"cuda_available": True},
    )

    assert event.timestamp == "2026-01-01T00:00:00+00:00"
    assert event.action_id == "action_001"
    assert event.level == "info"
    assert event.message == "System inspection completed."
    assert event.data == {"cuda_available": True}


def test_agent_run_result_creation() -> None:
    goal = _make_goal()
    plan = _make_plan(goal=goal)
    event = _make_event()
    result = AgentRunResult(
        generated_at="2026-01-01T00:00:01+00:00",
        goal=goal,
        plan=plan,
        status="ok",
        events=[event],
        warnings=["Patch output is review-only."],
        error=None,
    )

    assert result.generated_at == "2026-01-01T00:00:01+00:00"
    assert result.goal == goal
    assert result.plan == plan
    assert result.status == "ok"
    assert result.events == [event]
    assert result.warnings == ["Patch output is review-only."]
    assert result.error is None
    assert result.artifacts == {}


def test_to_dict_nesting_works() -> None:
    goal = _make_goal()
    result = AgentRunResult(
        generated_at="2026-01-01T00:00:01+00:00",
        goal=goal,
        plan=_make_plan(goal=goal),
        status="ok",
        events=[_make_event()],
    )

    data = result.to_dict()

    assert data["goal"]["id"] == "goal_001"
    assert data["plan"]["goal"]["kind"] == "optimize_script"
    assert data["plan"]["actions"][0]["status"] == "pending"
    assert data["events"][0]["data"]["recommendation_count"] == 3
    assert data["warnings"] == []
    assert data["error"] is None
    assert data["artifacts"] == {}


def test_json_serialization_works() -> None:
    goal = _make_goal()
    result = AgentRunResult(
        generated_at="2026-01-01T00:00:01+00:00",
        goal=goal,
        plan=_make_plan(goal=goal),
        status="ok",
        events=[_make_event()],
    )

    serialized = json.dumps(result.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["goal"]["script_path"] == "train.py"
    assert deserialized["plan"]["actions"][0]["name"] == "generate_recommendations"
    assert deserialized["events"][0]["message"] == "Recommendations generated."
    assert deserialized["artifacts"] == {}


def test_default_list_and_dict_fields_are_isolated_between_instances() -> None:
    first_goal = AgentGoal(
        id="first_goal",
        kind="optimize_script",
        description="First goal",
    )
    second_goal = AgentGoal(
        id="second_goal",
        kind="explain_run",
        description="Second goal",
    )
    first_action = AgentAction(
        id="first_action",
        name="analyze_code",
        description="Analyze code.",
        required=True,
    )
    second_action = AgentAction(
        id="second_action",
        name="summarize_results",
        description="Summarize results.",
        required=False,
    )
    first_plan = AgentPlan(id="first_plan", goal=first_goal)
    second_plan = AgentPlan(id="second_plan", goal=second_goal)
    first_event = AgentEvent(
        timestamp="2026-01-01T00:00:00+00:00",
        action_id=None,
        level="info",
        message="First event.",
    )
    second_event = AgentEvent(
        timestamp="2026-01-01T00:00:01+00:00",
        action_id=None,
        level="info",
        message="Second event.",
    )
    first_result = AgentRunResult(
        generated_at="2026-01-01T00:00:02+00:00",
        goal=first_goal,
        plan=first_plan,
        status="partial",
    )
    second_result = AgentRunResult(
        generated_at="2026-01-01T00:00:03+00:00",
        goal=second_goal,
        plan=second_plan,
        status="ok",
    )

    first_goal.options["quick"] = True
    first_goal.constraints.append("do_not_modify_original_file")
    first_action.depends_on.append("inspect_system")
    first_action.inputs["script_path"] = "train.py"
    first_plan.actions.append(first_action)
    first_plan.warnings.append("First warning.")
    first_event.data["phase"] = "5.1"
    first_result.events.append(first_event)
    first_result.warnings.append("Result warning.")

    assert first_goal.options == {"quick": True}
    assert first_goal.constraints == ["do_not_modify_original_file"]
    assert second_goal.options == {}
    assert second_goal.constraints == []
    assert first_action.depends_on == ["inspect_system"]
    assert first_action.inputs == {"script_path": "train.py"}
    assert second_action.depends_on == []
    assert second_action.inputs == {}
    assert first_plan.actions == [first_action]
    assert first_plan.warnings == ["First warning."]
    assert second_plan.actions == []
    assert second_plan.warnings == []
    assert first_event.data == {"phase": "5.1"}
    assert second_event.data == {}
    assert first_result.events == [first_event]
    assert first_result.warnings == ["Result warning."]
    assert second_result.events == []
    assert second_result.warnings == []


def test_timestamp_helper_returns_non_empty_utc_iso_string() -> None:
    timestamp = create_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert timestamp
    assert parsed.tzinfo == timezone.utc


def test_agent_action_status_defaults_to_pending() -> None:
    action = _make_action()

    assert action.status == "pending"


def test_agent_action_error_defaults_to_none() -> None:
    action = _make_action()

    assert action.error is None


def _make_goal() -> AgentGoal:
    return AgentGoal(
        id="goal_001",
        kind="optimize_script",
        description="Optimize train.py for NVIDIA GPU performance",
        script_path="train.py",
        options={"quick": True},
        constraints=["do_not_modify_original_file"],
    )


def _make_action() -> AgentAction:
    return AgentAction(
        id="action_001",
        name="generate_recommendations",
        description="Generate deterministic recommendations.",
        required=True,
        depends_on=["run_quick_benchmark"],
        inputs={"quick": True},
    )


def _make_plan(*, goal: AgentGoal | None = None) -> AgentPlan:
    return AgentPlan(
        id="plan_001",
        goal=goal or _make_goal(),
        actions=[_make_action()],
    )


def _make_event() -> AgentEvent:
    return AgentEvent(
        timestamp="2026-01-01T00:00:00+00:00",
        action_id="action_001",
        level="info",
        message="Recommendations generated.",
        data={"recommendation_count": 3},
    )
