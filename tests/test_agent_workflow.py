"""Tests for Phase 5.10 internal optimize_script workflow helpers."""

from __future__ import annotations

from pathlib import Path

from gpuboost.agent.actions import (
    ANALYZE_CODE,
    CREATE_PATCH_PLAN,
    GENERATE_DIFF,
    GENERATE_RECOMMENDATIONS,
    INSPECT_SYSTEM,
    RUN_TRIAL_WORKSPACE,
    RUN_QUICK_BENCHMARK,
    SUMMARIZE_RESULTS,
)
from gpuboost.agent.state import AgentState
from gpuboost.agent.workflow import (
    create_optimize_script_goal,
    run_optimize_script_workflow,
)
from gpuboost.schemas.agent import AgentAction, AgentRunResult


def test_create_optimize_script_goal_with_script_path() -> None:
    goal = create_optimize_script_goal(
        script_path="train.py",
        quick=False,
        goal_id="goal_123",
    )

    assert goal.id == "goal_123"
    assert goal.kind == "optimize_script"
    assert goal.description == "Optimize train.py for NVIDIA GPU performance"
    assert goal.script_path == "train.py"
    assert goal.options == {
        "quick": False,
        "trial": False,
        "test_command": None,
    }


def test_create_optimize_script_goal_without_script_path() -> None:
    goal = create_optimize_script_goal()

    assert goal.id == "optimize_script"
    assert goal.description == (
        "Analyze this system for NVIDIA GPU optimization opportunities"
    )
    assert goal.script_path is None
    assert goal.options == {
        "quick": True,
        "trial": False,
        "test_command": None,
    }


def test_goal_includes_safety_constraints() -> None:
    goal = create_optimize_script_goal(script_path="train.py")

    assert goal.constraints == [
        "do_not_modify_original_file",
        "review_patches_only",
    ]


def test_run_optimize_script_workflow_with_fake_handlers_returns_result_and_report(
) -> None:
    result, report = run_optimize_script_workflow(
        script_path="train.py",
        handlers=_fake_handlers(),
    )

    assert isinstance(result, AgentRunResult)
    assert result.status == "ok"
    assert report.status == "ok"
    assert result.events


def test_run_optimize_script_workflow_defaults_quick_true() -> None:
    result, _report = run_optimize_script_workflow(
        script_path="train.py",
        handlers=_fake_handlers(),
    )

    assert result.goal.options == {
        "quick": True,
        "trial": False,
        "test_command": None,
    }


def test_workflow_passes_trial_and_test_command_options() -> None:
    result, _report = run_optimize_script_workflow(
        script_path="train.py",
        handlers=_fake_handlers(),
        trial=True,
        test_command="python -c pass",
    )

    assert result.goal.options["trial"] is True
    assert result.goal.options["test_command"] == "python -c pass"
    assert RUN_TRIAL_WORKSPACE in [action.name for action in result.plan.actions]


def test_workflow_with_script_path_includes_code_patch_diff_actions() -> None:
    result, _report = run_optimize_script_workflow(
        script_path="train.py",
        handlers=_fake_handlers(),
    )
    action_names = [action.name for action in result.plan.actions]

    assert ANALYZE_CODE in action_names
    assert CREATE_PATCH_PLAN in action_names
    assert GENERATE_DIFF in action_names


def test_workflow_without_script_path_skips_code_patch_diff_actions() -> None:
    result, _report = run_optimize_script_workflow(
        script_path=None,
        handlers=_fake_handlers(),
    )
    action_names = [action.name for action in result.plan.actions]

    assert ANALYZE_CODE not in action_names
    assert CREATE_PATCH_PLAN not in action_names
    assert GENERATE_DIFF not in action_names
    assert result.status == "ok"


def test_workflow_preserves_no_source_edit_safety(tmp_path) -> None:
    source_path = tmp_path / "train.py"
    original_source = "value = 1\n"
    source_path.write_text(original_source, encoding="utf-8")

    result, _report = run_optimize_script_workflow(
        script_path=str(source_path),
        handlers=_fake_handlers(source_path=source_path),
    )

    assert result.status == "ok"
    assert source_path.read_text(encoding="utf-8") == original_source


def test_workflow_returns_error_report_when_required_fake_handler_fails() -> None:
    handlers = _fake_handlers()
    handlers[RUN_QUICK_BENCHMARK] = _raise("benchmark failed")

    result, report = run_optimize_script_workflow(
        script_path="train.py",
        handlers=handlers,
    )

    assert result.status == "error"
    assert result.error == "benchmark failed"
    assert report.status == "error"
    assert report.error == "benchmark failed"


def test_workflow_can_use_injected_handlers_instead_of_default_handlers() -> None:
    marker: dict[str, bool] = {}
    handlers = _fake_handlers(marker=marker)

    result, report = run_optimize_script_workflow(
        script_path="train.py",
        handlers=handlers,
    )

    assert result.status == "ok"
    assert report.status == "ok"
    assert marker == {
        INSPECT_SYSTEM: True,
        RUN_QUICK_BENCHMARK: True,
        GENERATE_RECOMMENDATIONS: True,
        ANALYZE_CODE: True,
        CREATE_PATCH_PLAN: True,
        GENERATE_DIFF: True,
        SUMMARIZE_RESULTS: True,
    }


def _fake_handlers(
    *,
    source_path: Path | None = None,
    marker: dict[str, bool] | None = None,
) -> dict[str, object]:
    def mark(name: str) -> None:
        if marker is not None:
            marker[name] = True

    def inspect_system(state: AgentState, action: AgentAction) -> None:
        mark(INSPECT_SYSTEM)
        state.gpu_profile = {"gpus": [{"name": "NVIDIA Test GPU"}]}

    def run_quick_benchmark(state: AgentState, action: AgentAction) -> None:
        mark(RUN_QUICK_BENCHMARK)
        suite = {"results": [{"name": "Quick Benchmark"}]}
        state.benchmark_result = suite
        state.metadata["_benchmark_suite"] = suite

    def generate_recommendations(state: AgentState, action: AgentAction) -> None:
        mark(GENERATE_RECOMMENDATIONS)
        state.advisor_result = {"recommendations": [{"id": "rec_001"}]}

    def analyze_code(state: AgentState, action: AgentAction) -> None:
        mark(ANALYZE_CODE)
        analysis = {"findings": [{"id": "finding_001"}]}
        state.code_analysis = analysis
        state.metadata["_code_analysis"] = analysis

    def create_patch_plan(state: AgentState, action: AgentAction) -> None:
        mark(CREATE_PATCH_PLAN)
        patch_plan = {"suggestions": [{"id": "patch_001"}]}
        state.patch_plan = patch_plan
        state.metadata["_patch_plan"] = patch_plan
        if source_path is None:
            state.metadata["_source_text"] = "value = 1\n"
        else:
            state.metadata["_source_text"] = source_path.read_text(encoding="utf-8")

    def generate_diff(state: AgentState, action: AgentAction) -> None:
        mark(GENERATE_DIFF)
        state.diff = "--- train.py\n+++ train.py\n-value = 1\n+value = 2"

    def summarize_results(state: AgentState, action: AgentAction) -> None:
        mark(SUMMARIZE_RESULTS)
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


def _raise(message: str):
    def handler(state: AgentState, action: AgentAction) -> None:
        raise RuntimeError(message)

    return handler
