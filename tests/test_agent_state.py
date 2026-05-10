"""Tests for Phase 5.2 agent state."""

from __future__ import annotations

from gpuboost.agent import AgentState
from gpuboost.schemas.agent import AgentAction, AgentGoal, AgentPlan


def test_agent_state_creation() -> None:
    goal = _make_goal()
    plan = _make_plan(goal)
    state = AgentState(
        goal=goal,
        plan=plan,
        gpu_profile={"gpu_name": "NVIDIA Test GPU"},
        benchmark_result={"status": "ok"},
        advisor_result={"recommendations": 2},
        code_analysis={"findings": 1},
        patch_plan={"suggestions": 1},
        diff="--- train.py\n+++ train.py\n",
        metadata={"phase": "5.2"},
    )

    assert state.goal == goal
    assert state.plan == plan
    assert state.gpu_profile == {"gpu_name": "NVIDIA Test GPU"}
    assert state.benchmark_result == {"status": "ok"}
    assert state.advisor_result == {"recommendations": 2}
    assert state.code_analysis == {"findings": 1}
    assert state.patch_plan == {"suggestions": 1}
    assert state.diff == "--- train.py\n+++ train.py\n"
    assert state.metadata == {"phase": "5.2"}


def test_default_collections_are_isolated_between_instances() -> None:
    first = AgentState(goal=_make_goal("first"))
    second = AgentState(goal=_make_goal("second"))

    first.warnings.append("first warning")
    first.completed_actions.append("action_001")
    first.failed_actions.append("action_002")
    first.events.append(
        first.add_event(
            level="info",
            message="First state event.",
        ),
    )
    first.metadata["owner"] = "first"

    assert first.warnings == ["first warning"]
    assert first.completed_actions == ["action_001"]
    assert first.failed_actions == ["action_002"]
    assert len(first.events) == 2
    assert first.metadata == {"owner": "first"}
    assert second.warnings == []
    assert second.completed_actions == []
    assert second.failed_actions == []
    assert second.events == []
    assert second.metadata == {}


def test_add_event_appends_and_returns_event() -> None:
    state = AgentState(goal=_make_goal())

    event = state.add_event(
        level="info",
        message="System inspected.",
        action_id="inspect_system",
        data={"cuda_available": True},
    )

    assert event in state.events
    assert event.level == "info"
    assert event.message == "System inspected."
    assert event.action_id == "inspect_system"
    assert event.data == {"cuda_available": True}
    assert event.timestamp


def test_add_warning_appends_warning_and_event() -> None:
    state = AgentState(goal=_make_goal())

    state.add_warning(
        message="Patch output is review-only.",
        action_id="generate_diff",
    )

    assert state.warnings == ["Patch output is review-only."]
    assert len(state.events) == 1
    assert state.events[0].level == "warning"
    assert state.events[0].message == "Patch output is review-only."
    assert state.events[0].action_id == "generate_diff"


def test_mark_completed_appends_action_once_and_adds_event() -> None:
    state = AgentState(goal=_make_goal())

    state.mark_completed("inspect_system")
    state.mark_completed("inspect_system")

    assert state.completed_actions == ["inspect_system"]
    assert len(state.events) == 2
    assert state.events[0].level == "info"
    assert state.events[0].action_id == "inspect_system"
    assert state.events[0].message == "Action completed: inspect_system"


def test_mark_failed_appends_action_once_and_adds_event() -> None:
    state = AgentState(goal=_make_goal())

    state.mark_failed("run_quick_benchmark", "Benchmark timed out.")
    state.mark_failed("run_quick_benchmark", "Benchmark timed out.")

    assert state.failed_actions == ["run_quick_benchmark"]
    assert len(state.events) == 2
    assert state.events[0].level == "error"
    assert state.events[0].action_id == "run_quick_benchmark"
    assert state.events[0].message == "Action failed: run_quick_benchmark"
    assert state.events[0].data == {"error": "Benchmark timed out."}


def test_has_failures_true_false() -> None:
    state = AgentState(goal=_make_goal())

    assert state.has_failures() is False

    state.mark_failed("analyze_code", "Unable to parse source.")

    assert state.has_failures() is True


def test_summary_counts_returns_expected_values() -> None:
    state = AgentState(goal=_make_goal())

    state.add_warning("First warning.")
    state.mark_completed("inspect_system")
    state.mark_failed("analyze_code", "Unable to parse source.")

    assert state.summary_counts() == {
        "warnings": 1,
        "completed_actions": 1,
        "failed_actions": 1,
        "events": 3,
    }


def test_to_dict_serializes_nested_goal_and_events() -> None:
    goal = _make_goal()
    state = AgentState(goal=goal, plan=_make_plan(goal))
    state.add_event(
        level="info",
        message="Recommendations generated.",
        action_id="generate_recommendations",
        data={"recommendation_count": 3},
    )

    data = state.to_dict()

    assert data["goal"]["id"] == "goal_001"
    assert data["goal"]["options"] == {"quick": True}
    assert data["plan"]["actions"][0]["name"] == "generate_recommendations"
    assert data["events"][0]["message"] == "Recommendations generated."
    assert data["events"][0]["data"] == {"recommendation_count": 3}


def test_metadata_can_store_arbitrary_values() -> None:
    state = AgentState(goal=_make_goal())

    state.metadata["attempt"] = 1
    state.metadata["dry_run"] = True
    state.metadata["labels"] = ["phase_5", "state"]
    state.metadata["artifact"] = {"kind": "patch_plan", "count": 1}

    assert state.metadata == {
        "attempt": 1,
        "dry_run": True,
        "labels": ["phase_5", "state"],
        "artifact": {"kind": "patch_plan", "count": 1},
    }


def _make_goal(goal_id: str = "goal_001") -> AgentGoal:
    return AgentGoal(
        id=goal_id,
        kind="optimize_script",
        description="Optimize train.py for NVIDIA GPU performance",
        script_path="train.py",
        options={"quick": True},
        constraints=["do_not_modify_original_file"],
    )


def _make_plan(goal: AgentGoal) -> AgentPlan:
    return AgentPlan(
        id="plan_001",
        goal=goal,
        actions=[
            AgentAction(
                id="action_001",
                name="generate_recommendations",
                description="Generate deterministic recommendations.",
                required=True,
            ),
        ],
    )
