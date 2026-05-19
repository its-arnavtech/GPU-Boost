"""Integration tests for Phase 5 agent core components."""

from __future__ import annotations

from pathlib import Path

from gpuboost.agent.actions import (
    ANALYZE_CODE,
    CREATE_PATCH_PLAN,
    GENERATE_DIFF,
    GENERATE_RECOMMENDATIONS,
    INSPECT_SYSTEM,
    RUN_MODEL_INFERENCE,
    RUN_TRIAL_WORKSPACE,
    RUN_QUICK_BENCHMARK,
    SUMMARIZE_RESULTS,
)
from gpuboost.agent.executor import AgentExecutor, default_handlers
from gpuboost.agent.planner import plan_for_goal
from gpuboost.agent.report import build_agent_report
from gpuboost.agent.state import AgentState
from gpuboost.schemas.agent import AgentAction, AgentGoal, AgentPlan


ALL_ACTIONS = {
    INSPECT_SYSTEM,
    RUN_QUICK_BENCHMARK,
    GENERATE_RECOMMENDATIONS,
    ANALYZE_CODE,
    CREATE_PATCH_PLAN,
    GENERATE_DIFF,
    RUN_TRIAL_WORKSPACE,
    RUN_MODEL_INFERENCE,
    SUMMARIZE_RESULTS,
}


def test_optimize_script_with_script_path_runs_fake_core_workflow() -> None:
    goal = _make_goal(script_path="train.py")
    plan = plan_for_goal(goal)
    result = AgentExecutor(handlers=_fake_handlers()).execute_plan(plan)
    report = build_agent_report(result)

    assert result.status == "ok"
    assert all(action.status == "completed" for action in result.plan.actions)
    assert not any(action.status == "failed" for action in result.plan.actions)
    assert report.status == "ok"
    assert {"Goal", "Plan"}.issubset(_section_titles(report))
    assert result.events


def test_optimize_script_without_script_path_skips_code_actions_and_completes() -> None:
    goal = _make_goal(script_path=None)
    plan = plan_for_goal(goal)

    action_names = [action.name for action in plan.actions]
    assert ANALYZE_CODE not in action_names
    assert CREATE_PATCH_PLAN not in action_names
    assert GENERATE_DIFF not in action_names
    assert plan.warnings == [
        "No script_path provided; code analysis and patch planning actions were skipped."
    ]

    result = AgentExecutor(handlers=_fake_handlers()).execute_plan(plan)

    assert result.status == "ok"
    assert all(action.status == "completed" for action in result.plan.actions)


def test_optional_action_failure_returns_partial_and_report_error_section() -> None:
    goal = _make_goal(script_path="train.py")
    plan = plan_for_goal(goal)
    handlers = _fake_handlers()
    handlers[ANALYZE_CODE] = _raise("analysis failed")

    result = AgentExecutor(handlers=handlers).execute_plan(plan)
    report = build_agent_report(result)
    statuses = _statuses_by_id(result.plan)

    assert result.status == "partial"
    assert statuses[ANALYZE_CODE] == "failed"
    assert statuses[GENERATE_DIFF] == "completed"
    assert statuses[SUMMARIZE_RESULTS] == "skipped"
    assert "Errors" in _section_titles(report)


def test_required_action_failure_returns_error_and_stops_execution() -> None:
    goal = _make_goal(script_path="train.py")
    plan = plan_for_goal(goal)
    handlers = _fake_handlers()
    handlers[RUN_QUICK_BENCHMARK] = _raise("benchmark failed")

    result = AgentExecutor(handlers=handlers).execute_plan(plan)
    report = build_agent_report(result)
    statuses = _statuses_by_id(result.plan)

    assert result.status == "error"
    assert result.error == "benchmark failed"
    assert statuses[RUN_QUICK_BENCHMARK] == "failed"
    assert statuses[GENERATE_RECOMMENDATIONS] == "pending"
    assert statuses[ANALYZE_CODE] == "pending"
    assert report.status == "error"


def test_dependency_skip_current_executor_behavior_is_explicit() -> None:
    goal = _make_goal(script_path="train.py")
    plan = plan_for_goal(goal)
    handlers = _fake_handlers()
    handlers[ANALYZE_CODE] = _raise("analysis failed")

    result = AgentExecutor(handlers=handlers).execute_plan(plan)
    statuses = _statuses_by_id(result.plan)

    assert statuses[ANALYZE_CODE] == "failed"
    assert statuses[CREATE_PATCH_PLAN] == "skipped"
    assert statuses[GENERATE_DIFF] == "completed"
    assert statuses[SUMMARIZE_RESULTS] == "skipped"
    assert result.status == "partial"


def test_default_real_handler_map_keys() -> None:
    assert set(default_handlers()) == ALL_ACTIONS


def test_fake_workflow_does_not_modify_original_file(tmp_path) -> None:
    source_path = tmp_path / "train.py"
    original_source = "value = 1\n"
    source_path.write_text(original_source, encoding="utf-8")

    goal = _make_goal(script_path=str(source_path))
    plan = plan_for_goal(goal)
    result = AgentExecutor(
        handlers=_fake_handlers(source_path=source_path),
    ).execute_plan(plan)

    assert result.status == "ok"
    assert source_path.read_text(encoding="utf-8") == original_source


def _fake_handlers(
    *,
    source_path: Path | None = None,
) -> dict[str, object]:
    def inspect_system(state: AgentState, action: AgentAction) -> None:
        state.gpu_profile = {"gpus": [{"name": "NVIDIA Test GPU"}]}

    def run_quick_benchmark(state: AgentState, action: AgentAction) -> None:
        suite = {"results": [{"name": "Quick Benchmark"}]}
        state.benchmark_result = suite
        state.metadata["_benchmark_suite"] = suite

    def generate_recommendations(state: AgentState, action: AgentAction) -> None:
        state.advisor_result = {"recommendations": [{"id": "rec_001"}]}

    def analyze_code(state: AgentState, action: AgentAction) -> None:
        analysis = {"findings": [{"id": "finding_001"}]}
        state.code_analysis = analysis
        state.metadata["_code_analysis"] = analysis

    def create_patch_plan(state: AgentState, action: AgentAction) -> None:
        patch_plan = {"suggestions": [{"id": "patch_001"}]}
        state.patch_plan = patch_plan
        state.metadata["_patch_plan"] = patch_plan
        if source_path is None:
            state.metadata["_source_text"] = "value = 1\n"
        else:
            state.metadata["_source_text"] = source_path.read_text(encoding="utf-8")

    def generate_diff(state: AgentState, action: AgentAction) -> None:
        state.diff = "--- train.py\n+++ train.py\n-value = 1\n+value = 2"

    def summarize_results(state: AgentState, action: AgentAction) -> None:
        state.metadata["summary"] = {
            "has_gpu_profile": state.gpu_profile is not None,
            "has_benchmark_result": state.benchmark_result is not None,
            "recommendation_count": 1,
            "code_finding_count": 1 if state.code_analysis else 0,
            "patch_suggestion_count": 1 if state.patch_plan else 0,
            "has_diff": bool(state.diff),
            "has_trial_result": "trial_result" in state.metadata,
            "warning_count": len(state.warnings),
            "failed_action_count": len(state.failed_actions),
        }

    return {
        INSPECT_SYSTEM: inspect_system,
        RUN_QUICK_BENCHMARK: run_quick_benchmark,
        GENERATE_RECOMMENDATIONS: generate_recommendations,
        ANALYZE_CODE: analyze_code,
        CREATE_PATCH_PLAN: create_patch_plan,
        GENERATE_DIFF: generate_diff,
        SUMMARIZE_RESULTS: summarize_results,
    }


def _make_goal(*, script_path: str | None) -> AgentGoal:
    return AgentGoal(
        id="goal_001",
        kind="optimize_script",
        description="Optimize train.py for NVIDIA GPU performance",
        script_path=script_path,
    )


def _raise(message: str):
    def handler(state: AgentState, action: AgentAction) -> None:
        raise RuntimeError(message)

    return handler


def _statuses_by_id(plan: AgentPlan) -> dict[str, str]:
    return {action.id: action.status for action in plan.actions}


def _section_titles(report) -> set[str]:
    return {section.title for section in report.sections}
