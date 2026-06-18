"""Tests for Phase 5.4 deterministic planner."""

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
)
from gpuboost.agent.planner import (
    NO_SCRIPT_PATH_WARNING,
    TRIAL_REQUIRES_SCRIPT_PATH_WARNING,
    create_optimize_script_plan,
    plan_for_goal,
)
from gpuboost.schemas.agent import AgentAction, AgentGoal


ALL_OPTIMIZE_ACTIONS = [
    INSPECT_SYSTEM,
    RUN_QUICK_BENCHMARK,
    GENERATE_RECOMMENDATIONS,
    ANALYZE_CODE,
    CREATE_PATCH_PLAN,
    GENERATE_DIFF,
    SUMMARIZE_RESULTS,
]

NO_SCRIPT_ACTIONS = [
    INSPECT_SYSTEM,
    RUN_QUICK_BENCHMARK,
    GENERATE_RECOMMENDATIONS,
    SUMMARIZE_RESULTS,
]


def test_optimize_script_with_script_path_includes_all_seven_actions() -> None:
    plan = create_optimize_script_plan(_make_goal(script_path="train.py"))

    assert [action.name for action in plan.actions] == ALL_OPTIMIZE_ACTIONS
    assert len(plan.actions) == 7


def test_optimize_script_without_script_path_skips_code_patch_diff_actions() -> None:
    plan = create_optimize_script_plan(_make_goal(script_path=None))

    assert [action.name for action in plan.actions] == NO_SCRIPT_ACTIONS
    assert ANALYZE_CODE not in [action.name for action in plan.actions]
    assert CREATE_PATCH_PLAN not in [action.name for action in plan.actions]
    assert GENERATE_DIFF not in [action.name for action in plan.actions]


def test_missing_script_path_adds_warning() -> None:
    plan = create_optimize_script_plan(_make_goal(script_path=None))

    assert plan.warnings == [NO_SCRIPT_PATH_WARNING]


def test_action_ordering_is_deterministic() -> None:
    first_plan = create_optimize_script_plan(_make_goal(script_path="train.py"))
    second_plan = create_optimize_script_plan(_make_goal(script_path="train.py"))

    assert [action.id for action in first_plan.actions] == ALL_OPTIMIZE_ACTIONS
    assert [action.id for action in second_plan.actions] == ALL_OPTIMIZE_ACTIONS


def test_dependencies_are_correct() -> None:
    plan = create_optimize_script_plan(_make_goal(script_path="train.py"))
    dependencies = {action.id: action.depends_on for action in plan.actions}

    assert dependencies[INSPECT_SYSTEM] == []
    assert dependencies[RUN_QUICK_BENCHMARK] == [INSPECT_SYSTEM]
    assert dependencies[GENERATE_RECOMMENDATIONS] == [RUN_QUICK_BENCHMARK]
    assert dependencies[ANALYZE_CODE] == [INSPECT_SYSTEM]
    assert dependencies[CREATE_PATCH_PLAN] == [ANALYZE_CODE]
    assert dependencies[GENERATE_DIFF] == [CREATE_PATCH_PLAN]


def test_action_inputs_include_script_path_where_expected() -> None:
    plan = create_optimize_script_plan(_make_goal(script_path="train.py"))
    inputs = {action.id: action.inputs for action in plan.actions}

    assert inputs[INSPECT_SYSTEM] == {}
    assert inputs[RUN_QUICK_BENCHMARK] == {}
    assert inputs[GENERATE_RECOMMENDATIONS] == {}
    assert inputs[ANALYZE_CODE] == {"script_path": "train.py"}
    assert inputs[CREATE_PATCH_PLAN] == {"script_path": "train.py"}
    assert inputs[GENERATE_DIFF] == {"script_path": "train.py"}
    assert inputs[SUMMARIZE_RESULTS] == {}


def test_summarize_results_depends_on_all_included_prior_actions() -> None:
    plan_with_script = create_optimize_script_plan(_make_goal(script_path="train.py"))
    plan_without_script = create_optimize_script_plan(_make_goal(script_path=None))

    assert plan_with_script.actions[-1].id == SUMMARIZE_RESULTS
    assert plan_with_script.actions[-1].depends_on == ALL_OPTIMIZE_ACTIONS[:-1]
    assert plan_without_script.actions[-1].id == SUMMARIZE_RESULTS
    assert plan_without_script.actions[-1].depends_on == NO_SCRIPT_ACTIONS[:-1]


def test_planner_includes_trial_action_when_requested_with_script_path() -> None:
    plan = create_optimize_script_plan(
        _make_goal(script_path="train.py", trial=True),
    )

    assert RUN_TRIAL_WORKSPACE in [action.name for action in plan.actions]
    assert plan.actions[-2].name == RUN_TRIAL_WORKSPACE


def test_planner_skips_trial_without_script_path_and_adds_warning() -> None:
    plan = create_optimize_script_plan(_make_goal(script_path=None, trial=True))

    assert RUN_TRIAL_WORKSPACE not in [action.name for action in plan.actions]
    assert TRIAL_REQUIRES_SCRIPT_PATH_WARNING in plan.warnings


def test_summarize_results_depends_on_trial_action_when_included() -> None:
    plan = create_optimize_script_plan(
        _make_goal(script_path="train.py", trial=True),
    )
    dependencies = {action.id: action.depends_on for action in plan.actions}

    assert dependencies[RUN_TRIAL_WORKSPACE] == [GENERATE_DIFF]
    assert dependencies[SUMMARIZE_RESULTS][-1] == RUN_TRIAL_WORKSPACE


def test_planner_includes_model_action_only_when_requested() -> None:
    default_plan = create_optimize_script_plan(_make_goal(script_path="train.py"))
    model_plan = create_optimize_script_plan(
        _make_goal(script_path="train.py", model=True),
    )

    assert [action.name for action in default_plan.actions] == ALL_OPTIMIZE_ACTIONS
    assert RUN_MODEL_INFERENCE not in [action.name for action in default_plan.actions]
    assert [action.name for action in model_plan.actions] == [
        *ALL_OPTIMIZE_ACTIONS[:-1],
        RUN_MODEL_INFERENCE,
        SUMMARIZE_RESULTS,
    ]


def test_planner_includes_model_action_when_artifact_path_is_present() -> None:
    plan = create_optimize_script_plan(
        _make_goal(script_path="train.py", model_artifact_path="artifact/manifest.json"),
    )

    assert RUN_MODEL_INFERENCE in [action.name for action in plan.actions]


def test_model_action_depends_on_prior_useful_actions_and_summary_depends_on_model(
) -> None:
    plan = create_optimize_script_plan(_make_goal(script_path="train.py", model=True))
    dependencies = {action.id: action.depends_on for action in plan.actions}

    assert dependencies[RUN_MODEL_INFERENCE] == ALL_OPTIMIZE_ACTIONS[:-1]
    assert dependencies[SUMMARIZE_RESULTS][-1] == RUN_MODEL_INFERENCE


def test_model_action_can_coexist_with_trial_action() -> None:
    plan = create_optimize_script_plan(
        _make_goal(script_path="train.py", trial=True, model=True),
    )
    action_names = [action.name for action in plan.actions]
    dependencies = {action.id: action.depends_on for action in plan.actions}

    assert RUN_TRIAL_WORKSPACE in action_names
    assert action_names[-3:] == [
        RUN_TRIAL_WORKSPACE,
        RUN_MODEL_INFERENCE,
        SUMMARIZE_RESULTS,
    ]
    assert dependencies[RUN_MODEL_INFERENCE][-1] == RUN_TRIAL_WORKSPACE


def test_unsupported_goal_kind_returns_empty_plan_with_warning() -> None:
    goal = _make_goal(kind="compare_runs", script_path=None)
    plan = plan_for_goal(goal)

    assert plan.id == "plan_goal_001"
    assert plan.goal == goal
    assert plan.actions == []
    assert plan.warnings == ["Unsupported agent goal kind: compare_runs"]


def test_plan_id_is_deterministic() -> None:
    goal = _make_goal(goal_id="goal_abc", script_path="train.py")
    plan = plan_for_goal(goal)

    assert plan.id == "plan_goal_abc"


def test_generated_actions_are_agent_actions_with_pending_status() -> None:
    plan = plan_for_goal(_make_goal(script_path="train.py"))

    assert all(isinstance(action, AgentAction) for action in plan.actions)
    assert all(action.status == "pending" for action in plan.actions)


def _make_goal(
    *,
    goal_id: str = "goal_001",
    kind: str = "optimize_script",
    script_path: str | None = "train.py",
    trial: bool = False,
    model: bool = False,
    model_artifact_path: str | None = None,
) -> AgentGoal:
    return AgentGoal(
        id=goal_id,
        kind=kind,
        description="Optimize train.py for NVIDIA GPU performance",
        script_path=script_path,
        options={
            "quick": True,
            "trial": trial,
            "model": model,
            "model_artifact_path": model_artifact_path,
        },
        constraints=["do_not_modify_original_file"],
    )
