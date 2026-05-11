"""Internal Phase 5 workflow helpers for GPUBoost agent runs."""

from __future__ import annotations

from gpuboost.agent.executor import ActionHandler, AgentExecutor, default_handlers
from gpuboost.agent.planner import plan_for_goal
from gpuboost.agent.report import AgentReport, build_agent_report
from gpuboost.history.builder import build_history_run_record
from gpuboost.history.store import insert_history_run
from gpuboost.schemas.agent import AgentGoal, AgentRunResult


def create_optimize_script_goal(
    script_path: str | None = None,
    quick: bool = True,
    goal_id: str = "optimize_script",
    trial: bool = False,
    test_command: str | None = None,
) -> AgentGoal:
    """Create a deterministic optimize_script goal."""

    if script_path:
        description = f"Optimize {script_path} for NVIDIA GPU performance"
    else:
        description = "Analyze this system for NVIDIA GPU optimization opportunities"

    options = {
        "quick": quick,
        "trial": trial,
        "test_command": test_command,
    }

    return AgentGoal(
        id=goal_id,
        kind="optimize_script",
        description=description,
        script_path=script_path,
        options=options,
        constraints=[
            "do_not_modify_original_file",
            "review_patches_only",
        ],
    )


def run_optimize_script_workflow(
    script_path: str | None = None,
    handlers: dict[str, ActionHandler] | None = None,
    quick: bool = True,
    trial: bool = False,
    test_command: str | None = None,
    save_history: bool = False,
    history_db_path: str | None = None,
) -> tuple[AgentRunResult, AgentReport]:
    """Run the deterministic optimize_script workflow without CLI integration."""

    goal = create_optimize_script_goal(
        script_path=script_path,
        quick=quick,
        trial=trial,
        test_command=test_command,
    )
    plan = plan_for_goal(goal)
    executor = AgentExecutor(
        handlers=handlers if handlers is not None else default_handlers(),
    )
    result = executor.execute_plan(plan)
    report = build_agent_report(result)
    if save_history:
        try:
            record = build_history_run_record(result)
            insert_history_run(record, db_path=history_db_path)
            result.artifacts["history_run_id"] = record.run_id
        except Exception as error:  # noqa: BLE001 - history is best effort
            result.warnings.append(f"Failed to save history: {error}")
    return result, report
