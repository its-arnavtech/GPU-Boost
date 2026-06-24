"""Agent working-memory helpers for GPUBoost."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ACTION_REGISTRY": ("gpuboost.agent.actions", "ACTION_REGISTRY"),
    "ANALYZE_CODE": ("gpuboost.agent.actions", "ANALYZE_CODE"),
    "CREATE_PATCH_PLAN": ("gpuboost.agent.actions", "CREATE_PATCH_PLAN"),
    "GENERATE_DIFF": ("gpuboost.agent.actions", "GENERATE_DIFF"),
    "GENERATE_RECOMMENDATIONS": (
        "gpuboost.agent.actions",
        "GENERATE_RECOMMENDATIONS",
    ),
    "INSPECT_SYSTEM": ("gpuboost.agent.actions", "INSPECT_SYSTEM"),
    "RUN_MODEL_INFERENCE": ("gpuboost.agent.actions", "RUN_MODEL_INFERENCE"),
    "RUN_QUICK_BENCHMARK": ("gpuboost.agent.actions", "RUN_QUICK_BENCHMARK"),
    "SUMMARIZE_RESULTS": ("gpuboost.agent.actions", "SUMMARIZE_RESULTS"),
    "ActionDefinition": ("gpuboost.agent.actions", "ActionDefinition"),
    "ActionHandler": ("gpuboost.agent.executor", "ActionHandler"),
    "AgentExecutor": ("gpuboost.agent.executor", "AgentExecutor"),
    "AgentReport": ("gpuboost.agent.report", "AgentReport"),
    "AgentReportSection": ("gpuboost.agent.report", "AgentReportSection"),
    "AgentState": ("gpuboost.agent.state", "AgentState"),
    "build_agent_report": ("gpuboost.agent.report", "build_agent_report"),
    "count_actions_by_status": (
        "gpuboost.agent.report",
        "count_actions_by_status",
    ),
    "create_agent_action": ("gpuboost.agent.actions", "create_agent_action"),
    "create_optimize_script_goal": (
        "gpuboost.agent.workflow",
        "create_optimize_script_goal",
    ),
    "create_optimize_script_plan": (
        "gpuboost.agent.planner",
        "create_optimize_script_plan",
    ),
    "default_handlers": ("gpuboost.agent.executor", "default_handlers"),
    "failed_action_messages": (
        "gpuboost.agent.report",
        "failed_action_messages",
    ),
    "format_action_line": ("gpuboost.agent.report", "format_action_line"),
    "format_event_line": ("gpuboost.agent.report", "format_event_line"),
    "get_action_definition": ("gpuboost.agent.actions", "get_action_definition"),
    "handle_analyze_code": ("gpuboost.agent.handlers", "handle_analyze_code"),
    "handle_create_patch_plan": (
        "gpuboost.agent.handlers",
        "handle_create_patch_plan",
    ),
    "handle_generate_diff": ("gpuboost.agent.handlers", "handle_generate_diff"),
    "handle_generate_recommendations": (
        "gpuboost.agent.handlers",
        "handle_generate_recommendations",
    ),
    "handle_inspect_system": ("gpuboost.agent.handlers", "handle_inspect_system"),
    "handle_run_model_inference": (
        "gpuboost.agent.handlers",
        "handle_run_model_inference",
    ),
    "handle_run_quick_benchmark": (
        "gpuboost.agent.handlers",
        "handle_run_quick_benchmark",
    ),
    "handle_summarize_results": (
        "gpuboost.agent.handlers",
        "handle_summarize_results",
    ),
    "is_known_action": ("gpuboost.agent.actions", "is_known_action"),
    "list_action_definitions": ("gpuboost.agent.actions", "list_action_definitions"),
    "plan_for_goal": ("gpuboost.agent.planner", "plan_for_goal"),
    "run_optimize_script_workflow": (
        "gpuboost.agent.workflow",
        "run_optimize_script_workflow",
    ),
    "handlers": ("gpuboost.agent.handlers", None),
    "workflow": ("gpuboost.agent.workflow", None),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
