"""Agent working-memory helpers for GPUBoost."""

from gpuboost.agent.actions import (
    ACTION_REGISTRY,
    ANALYZE_CODE,
    CREATE_PATCH_PLAN,
    GENERATE_DIFF,
    GENERATE_RECOMMENDATIONS,
    INSPECT_SYSTEM,
    RUN_QUICK_BENCHMARK,
    SUMMARIZE_RESULTS,
    ActionDefinition,
    create_agent_action,
    get_action_definition,
    is_known_action,
    list_action_definitions,
)
from gpuboost.agent.executor import (
    ActionHandler,
    AgentExecutor,
    default_handlers,
)
from gpuboost.agent.handlers import (
    handle_analyze_code,
    handle_create_patch_plan,
    handle_generate_diff,
    handle_generate_recommendations,
    handle_inspect_system,
    handle_run_quick_benchmark,
    handle_summarize_results,
)
from gpuboost.agent.planner import (
    create_optimize_script_plan,
    plan_for_goal,
)
from gpuboost.agent.report import (
    AgentReport,
    AgentReportSection,
    build_agent_report,
    count_actions_by_status,
    failed_action_messages,
    format_action_line,
    format_event_line,
)
from gpuboost.agent.state import AgentState
from gpuboost.agent.workflow import (
    create_optimize_script_goal,
    run_optimize_script_workflow,
)

__all__ = [
    "ACTION_REGISTRY",
    "ANALYZE_CODE",
    "CREATE_PATCH_PLAN",
    "GENERATE_DIFF",
    "GENERATE_RECOMMENDATIONS",
    "INSPECT_SYSTEM",
    "RUN_QUICK_BENCHMARK",
    "SUMMARIZE_RESULTS",
    "ActionDefinition",
    "ActionHandler",
    "AgentExecutor",
    "AgentReport",
    "AgentReportSection",
    "AgentState",
    "build_agent_report",
    "count_actions_by_status",
    "create_agent_action",
    "create_optimize_script_goal",
    "create_optimize_script_plan",
    "default_handlers",
    "failed_action_messages",
    "format_action_line",
    "format_event_line",
    "get_action_definition",
    "handle_analyze_code",
    "handle_create_patch_plan",
    "handle_generate_diff",
    "handle_generate_recommendations",
    "handle_inspect_system",
    "handle_run_quick_benchmark",
    "handle_summarize_results",
    "is_known_action",
    "list_action_definitions",
    "plan_for_goal",
    "run_optimize_script_workflow",
]
