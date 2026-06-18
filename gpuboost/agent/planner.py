"""Deterministic planning helpers for GPUBoost agent goals."""

from __future__ import annotations

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
    create_agent_action,
)
from gpuboost.schemas.agent import AgentGoal, AgentPlan


NO_SCRIPT_PATH_WARNING = (
    "No script_path provided; code analysis and patch planning actions were skipped."
)
TRIAL_REQUIRES_SCRIPT_PATH_WARNING = (
    "Trial mode requires script_path; trial action was skipped."
)


def create_optimize_script_plan(goal: AgentGoal) -> AgentPlan:
    """Create a deterministic optimize_script plan without executing actions."""

    actions = [
        create_agent_action(INSPECT_SYSTEM),
        create_agent_action(
            RUN_QUICK_BENCHMARK,
            depends_on=[INSPECT_SYSTEM],
        ),
        create_agent_action(
            GENERATE_RECOMMENDATIONS,
            depends_on=[RUN_QUICK_BENCHMARK],
        ),
    ]
    warnings: list[str] = []

    trial_requested = bool(goal.options.get("trial"))
    model_requested = bool(goal.options.get("model") or goal.options.get("model_artifact_path"))

    if goal.script_path:
        script_inputs = {"script_path": goal.script_path}
        actions.extend(
            [
                create_agent_action(
                    ANALYZE_CODE,
                    inputs=script_inputs,
                    depends_on=[INSPECT_SYSTEM],
                ),
                create_agent_action(
                    CREATE_PATCH_PLAN,
                    inputs=script_inputs,
                    depends_on=[ANALYZE_CODE],
                ),
                create_agent_action(
                    GENERATE_DIFF,
                    inputs=script_inputs,
                    depends_on=[CREATE_PATCH_PLAN],
                ),
            ],
        )
        if trial_requested:
            trial_inputs = dict(script_inputs)
            test_command = goal.options.get("test_command")
            if test_command is not None:
                trial_inputs["test_command"] = test_command
            actions.append(
                create_agent_action(
                    RUN_TRIAL_WORKSPACE,
                    inputs=trial_inputs,
                    depends_on=[GENERATE_DIFF],
                )
            )
    else:
        warnings.append(NO_SCRIPT_PATH_WARNING)
        if trial_requested:
            warnings.append(TRIAL_REQUIRES_SCRIPT_PATH_WARNING)

    if model_requested:
        actions.append(
            create_agent_action(
                RUN_MODEL_INFERENCE,
                depends_on=[action.id for action in actions],
            ),
        )

    prior_action_ids = [action.id for action in actions]
    actions.append(
        create_agent_action(
            SUMMARIZE_RESULTS,
            depends_on=prior_action_ids,
        ),
    )

    return AgentPlan(
        id=f"plan_{goal.id}",
        goal=goal,
        actions=actions,
        warnings=warnings,
    )


def plan_for_goal(goal: AgentGoal) -> AgentPlan:
    """Create a deterministic plan for a supported goal."""

    if goal.kind == "optimize_script":
        return create_optimize_script_plan(goal)

    return AgentPlan(
        id=f"plan_{goal.id}",
        goal=goal,
        actions=[],
        warnings=[f"Unsupported agent goal kind: {goal.kind}"],
    )
