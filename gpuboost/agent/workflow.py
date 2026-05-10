"""Internal Phase 5 workflow helpers for GPUBoost agent runs."""

from __future__ import annotations

from gpuboost.agent.executor import ActionHandler, AgentExecutor, default_handlers
from gpuboost.agent.planner import plan_for_goal
from gpuboost.agent.report import AgentReport, build_agent_report
from gpuboost.schemas.agent import AgentGoal, AgentRunResult


def create_optimize_script_goal(
    script_path: str | None = None,
    quick: bool = True,
    goal_id: str = "optimize_script",
) -> AgentGoal:
    """Create a deterministic optimize_script goal."""

    if script_path:
        description = f"Optimize {script_path} for NVIDIA GPU performance"
    else:
        description = "Analyze this system for NVIDIA GPU optimization opportunities"

    return AgentGoal(
        id=goal_id,
        kind="optimize_script",
        description=description,
        script_path=script_path,
        options={"quick": quick},
        constraints=[
            "do_not_modify_original_file",
            "review_patches_only",
        ],
    )


def run_optimize_script_workflow(
    script_path: str | None = None,
    handlers: dict[str, ActionHandler] | None = None,
    quick: bool = True,
) -> tuple[AgentRunResult, AgentReport]:
    """Run the deterministic optimize_script workflow without CLI integration."""

    goal = create_optimize_script_goal(
        script_path=script_path,
        quick=quick,
    )
    plan = plan_for_goal(goal)
    executor = AgentExecutor(
        handlers=handlers if handlers is not None else default_handlers(),
    )
    result = executor.execute_plan(plan)
    report = build_agent_report(result)
    return result, report
