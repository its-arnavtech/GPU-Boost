"""Deterministic action handlers for GPUBoost agent execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gpuboost.advisor import engine as advisor_engine
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
from gpuboost.agent.state import AgentState
from gpuboost.benchmarks import runner as benchmark_runner
from gpuboost.code_analysis import runner as code_analysis_runner
from gpuboost.inspector import profile as profile_module
from gpuboost.model import inference as model_inference
from gpuboost.patching import diff as patch_diff
from gpuboost.patching import planner as patch_planner
from gpuboost.schemas.agent import AgentAction
from gpuboost.trial import engine as trial_engine

if TYPE_CHECKING:
    from gpuboost.agent.executor import ActionHandler


def handle_inspect_system(state: AgentState, action: AgentAction) -> None:
    """Collect and store the current GPUBoost system profile."""

    profile = profile_module.collect_profile()
    state.gpu_profile = profile.to_dict()
    _add_warnings(state, getattr(profile, "warnings", []), action.id)
    state.add_event(
        level="info",
        message="System inspection completed.",
        action_id=action.id,
    )


def handle_run_quick_benchmark(state: AgentState, action: AgentAction) -> None:
    """Run and store the quick benchmark suite result."""

    device_index = int(action.inputs.get("device_index", 0))
    suite = benchmark_runner.run_quick_benchmark(device_index=device_index)
    state.benchmark_result = suite.to_dict()
    state.metadata["_benchmark_suite"] = suite
    _add_warnings(state, getattr(suite, "warnings", []), action.id)
    state.add_event(
        level="info",
        message="Quick benchmark completed.",
        action_id=action.id,
    )


def handle_generate_recommendations(
    state: AgentState,
    action: AgentAction,
) -> None:
    """Generate and store advisor recommendations from benchmark results."""

    suite = state.metadata.get("_benchmark_suite")
    if state.benchmark_result is None or suite is None:
        raise ValueError(
            "Benchmark result is required before generating recommendations."
        )

    advisor = advisor_engine.generate_advisor_result(suite)
    state.advisor_result = advisor.to_dict()
    _add_warnings(state, getattr(advisor, "warnings", []), action.id)
    state.add_event(
        level="info",
        message="Recommendations generated.",
        action_id=action.id,
    )


def handle_analyze_code(state: AgentState, action: AgentAction) -> None:
    """Analyze a Python script and store code analysis artifacts."""

    script_path = _get_script_path(state, action)
    if not script_path:
        raise ValueError("script_path is required for code analysis.")

    analysis = code_analysis_runner.analyze_python_file(script_path)
    state.code_analysis = analysis.to_dict()
    state.metadata["_code_analysis"] = analysis
    _add_warnings(state, getattr(analysis, "warnings", []), action.id)
    state.add_event(
        level="info",
        message="Code analysis completed.",
        action_id=action.id,
    )

    if analysis.status == "error":
        raise ValueError(analysis.error or "Code analysis failed.")


def handle_create_patch_plan(state: AgentState, action: AgentAction) -> None:
    """Create a safe patch plan from stored code analysis."""

    script_path = _get_script_path(state, action)
    if not script_path:
        raise ValueError("script_path is required for patch planning.")

    analysis = state.metadata.get("_code_analysis")
    if analysis is None:
        raise ValueError("Code analysis is required before creating a patch plan.")

    source_text = Path(script_path).read_text(encoding="utf-8")
    patch_plan = patch_planner.create_patch_plan_from_analysis(
        source_text,
        analysis,
    )
    state.patch_plan = patch_plan.to_dict()
    state.metadata["_patch_plan"] = patch_plan
    state.metadata["_source_text"] = source_text
    _add_warnings(state, getattr(patch_plan, "warnings", []), action.id)
    state.add_event(
        level="info",
        message="Patch plan created.",
        action_id=action.id,
    )


def handle_generate_diff(state: AgentState, action: AgentAction) -> None:
    """Generate and store a reviewable diff without writing files."""

    patch_plan = state.metadata.get("_patch_plan")
    if patch_plan is None:
        raise ValueError("Patch plan is required before generating a diff.")

    source_text = state.metadata.get("_source_text")
    if source_text is None:
        raise ValueError("Source text is required before generating a diff.")

    diff, warnings = patch_diff.generate_patch_plan_diff(source_text, patch_plan)
    state.diff = diff
    _add_warnings(state, warnings, action.id)
    state.add_event(
        level="info",
        message="Diff generated.",
        action_id=action.id,
    )


def handle_run_trial_workspace(state: AgentState, action: AgentAction) -> None:
    """Run a safe patch trial against a copied workspace file."""

    script_path = _get_script_path(state, action)
    if not script_path:
        raise ValueError("script_path is required for trial workspace validation.")

    patch_plan = state.metadata.get("_patch_plan")
    if patch_plan is None:
        raise ValueError("Patch plan is required before running a trial workspace.")

    test_command = action.inputs.get("test_command")
    if test_command is None:
        test_command = state.goal.options.get("test_command")
    trial_result = trial_engine.run_patch_trial(
        original_file=script_path,
        patch_plan=patch_plan,
        test_command=str(test_command) if test_command is not None else None,
    )
    state.metadata["_trial_result"] = trial_result
    state.metadata["trial_result"] = trial_result.to_dict()
    _add_warnings(state, trial_result.warnings, action.id)
    state.add_event(
        level="info",
        message=f"Trial workspace completed with status: {trial_result.status}.",
        action_id=action.id,
        data={"status": trial_result.status},
    )

    if trial_result.status in {"failed", "error"}:
        raise ValueError(trial_result.error or "Trial workspace validation failed.")


def handle_run_model_inference(state: AgentState, action: AgentAction) -> None:
    """Run optional local model inference over safe agent artifacts."""

    provider = state.metadata.get("_model_provider")
    result = model_inference.run_model_inference(state, provider=provider)
    state.metadata["_model_result"] = result
    state.metadata["model_result"] = result.to_dict()
    _add_warnings(state, result.warnings, action.id)
    state.add_event(
        level="info",
        message=f"Model inference completed with status: {result.status}.",
        action_id=action.id,
        data={
            "fallback_used": result.fallback_used,
            "status": result.status,
        },
    )

    if result.status == "error" and not result.fallback_used:
        raise ValueError(result.error or "Model inference failed.")


def handle_summarize_results(state: AgentState, action: AgentAction) -> None:
    """Store a lightweight summary of collected agent artifacts."""

    state.metadata["summary"] = {
        "has_gpu_profile": state.gpu_profile is not None,
        "has_benchmark_result": state.benchmark_result is not None,
        "recommendation_count": _count_items(
            state.advisor_result,
            "recommendations",
        ),
        "code_finding_count": _count_items(state.code_analysis, "findings"),
        "patch_suggestion_count": _count_items(
            state.patch_plan,
            "suggestions",
        ),
        "has_diff": bool(state.diff),
        "has_trial_result": "trial_result" in state.metadata,
        "has_model_result": "model_result" in state.metadata,
        "warning_count": len(state.warnings),
        "failed_action_count": len(state.failed_actions),
    }
    state.add_event(
        level="info",
        message="Results summarized.",
        action_id=action.id,
    )


def default_handlers() -> dict[str, "ActionHandler"]:
    """Return the default deterministic agent action handlers."""

    return {
        INSPECT_SYSTEM: handle_inspect_system,
        RUN_QUICK_BENCHMARK: handle_run_quick_benchmark,
        GENERATE_RECOMMENDATIONS: handle_generate_recommendations,
        ANALYZE_CODE: handle_analyze_code,
        CREATE_PATCH_PLAN: handle_create_patch_plan,
        GENERATE_DIFF: handle_generate_diff,
        RUN_TRIAL_WORKSPACE: handle_run_trial_workspace,
        RUN_MODEL_INFERENCE: handle_run_model_inference,
        SUMMARIZE_RESULTS: handle_summarize_results,
    }


def _get_script_path(state: AgentState, action: AgentAction) -> str | None:
    script_path = action.inputs.get("script_path") or state.goal.script_path
    if script_path is None:
        return None
    return str(script_path)


def _add_warnings(
    state: AgentState,
    warnings: list[str],
    action_id: str,
) -> None:
    for warning in warnings:
        state.add_warning(warning, action_id=action_id)


def _count_items(data: dict[str, object] | None, key: str) -> int:
    if data is None:
        return 0

    items = data.get(key)
    if isinstance(items, list):
        return len(items)
    return 0
