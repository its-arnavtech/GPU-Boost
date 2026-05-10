"""Tests for Phase 5.5 agent executor skeleton."""

from __future__ import annotations

from gpuboost.agent.executor import AgentExecutor, default_handlers
from gpuboost.agent.state import AgentState
from gpuboost.schemas.agent import AgentAction, AgentGoal, AgentPlan, AgentRunResult


def test_executor_creates_agent_run_result() -> None:
    plan = _make_plan([_make_action("action_001", required=False)])
    executor = AgentExecutor(
        handlers={"action_001": _successful_handler},
    )

    result = executor.execute_plan(plan)

    assert isinstance(result, AgentRunResult)
    assert result.goal == plan.goal
    assert result.plan == plan
    assert result.generated_at


def test_successful_fake_handlers_mark_actions_completed() -> None:
    plan = _make_plan(
        [
            _make_action("action_001"),
            _make_action("action_002", depends_on=["action_001"]),
        ],
    )
    executor = AgentExecutor(
        handlers={
            "action_001": _successful_handler,
            "action_002": _successful_handler,
        },
    )

    result = executor.execute_plan(plan)

    assert result.status == "ok"
    assert [action.status for action in plan.actions] == ["completed", "completed"]
    assert [action.error for action in plan.actions] == [None, None]


def test_events_are_recorded() -> None:
    plan = _make_plan([_make_action("action_001")])
    executor = AgentExecutor(handlers={"action_001": _successful_handler})

    result = executor.execute_plan(plan)
    messages = [event.message for event in result.events]

    assert messages[0] == "Agent execution started."
    assert "Starting action: action_001" in messages
    assert "Action completed: action_001" in messages
    assert messages[-1] == "Agent execution finished with status: ok"


def test_missing_required_handler_fails_and_stops() -> None:
    plan = _make_plan(
        [
            _make_action("required_action", required=True),
            _make_action("later_action", required=True),
        ],
    )
    executor = AgentExecutor(handlers={"later_action": _successful_handler})

    result = executor.execute_plan(plan)

    assert result.status == "error"
    assert result.error == "No handler registered for action: required_action"
    assert plan.actions[0].status == "failed"
    assert plan.actions[0].error == "No handler registered for action: required_action"
    assert plan.actions[1].status == "pending"


def test_missing_optional_handler_skips_and_continues() -> None:
    plan = _make_plan(
        [
            _make_action("optional_action", required=False),
            _make_action("required_action", required=True),
        ],
    )
    executor = AgentExecutor(handlers={"required_action": _successful_handler})

    result = executor.execute_plan(plan)

    assert result.status == "ok"
    assert plan.actions[0].status == "skipped"
    assert plan.actions[0].error == (
        "No handler registered for optional action: optional_action"
    )
    assert plan.actions[1].status == "completed"
    assert result.warnings == [
        "No handler registered for optional action: optional_action",
    ]


def test_required_handler_exception_produces_status_error() -> None:
    plan = _make_plan(
        [
            _make_action("required_action", required=True),
            _make_action("later_action", required=True),
        ],
    )
    executor = AgentExecutor(
        handlers={
            "required_action": _failing_handler,
            "later_action": _successful_handler,
        },
    )

    result = executor.execute_plan(plan)

    assert result.status == "error"
    assert result.error == "handler exploded"
    assert plan.actions[0].status == "failed"
    assert plan.actions[0].error == "handler exploded"
    assert plan.actions[1].status == "pending"


def test_optional_handler_exception_produces_status_partial_and_continues() -> None:
    plan = _make_plan(
        [
            _make_action("optional_action", required=False),
            _make_action("required_action", required=True),
        ],
    )
    executor = AgentExecutor(
        handlers={
            "optional_action": _failing_handler,
            "required_action": _successful_handler,
        },
    )

    result = executor.execute_plan(plan)

    assert result.status == "partial"
    assert result.error is None
    assert plan.actions[0].status == "failed"
    assert plan.actions[1].status == "completed"


def test_dependency_failed_causes_dependent_action_to_skip() -> None:
    plan = _make_plan(
        [
            _make_action("optional_action", required=False),
            _make_action(
                "dependent_action",
                required=False,
                depends_on=["optional_action"],
            ),
        ],
    )
    executor = AgentExecutor(
        handlers={
            "optional_action": _failing_handler,
            "dependent_action": _successful_handler,
        },
    )

    result = executor.execute_plan(plan)

    assert result.status == "partial"
    assert plan.actions[0].status == "failed"
    assert plan.actions[1].status == "skipped"
    assert plan.actions[1].error == (
        "Skipped because dependency failed: optional_action"
    )
    assert result.warnings == [
        "Skipped because dependency failed: optional_action",
    ]


def test_final_status_ok_partial_error_behavior() -> None:
    ok_plan = _make_plan([_make_action("ok_action")])
    partial_plan = _make_plan(
        [
            _make_action("optional_action", required=False),
            _make_action("ok_action"),
        ],
    )
    error_plan = _make_plan([_make_action("required_action", required=True)])

    ok_result = AgentExecutor(
        handlers={"ok_action": _successful_handler},
    ).execute_plan(ok_plan)
    partial_result = AgentExecutor(
        handlers={
            "optional_action": _failing_handler,
            "ok_action": _successful_handler,
        },
    ).execute_plan(partial_plan)
    error_result = AgentExecutor().execute_plan(error_plan)

    assert ok_result.status == "ok"
    assert partial_result.status == "partial"
    assert error_result.status == "error"


def test_default_handlers_returns_dict() -> None:
    assert isinstance(default_handlers(), dict)


def test_handlers_can_mutate_agent_state_metadata() -> None:
    plan = _make_plan([_make_action("metadata_action")])
    observed_metadata: dict[str, str] = {}

    def metadata_handler(state: AgentState, action: AgentAction) -> None:
        state.metadata["handled_action"] = action.id
        observed_metadata["handled_action"] = state.metadata["handled_action"]

    executor = AgentExecutor(handlers={"metadata_action": metadata_handler})

    result = executor.execute_plan(plan)

    assert result.status == "ok"
    assert observed_metadata == {"handled_action": "metadata_action"}
    assert any(
        event.message == "Action completed: metadata_action"
        for event in result.events
    )
    assert plan.actions[0].status == "completed"


def _make_goal() -> AgentGoal:
    return AgentGoal(
        id="goal_001",
        kind="optimize_script",
        description="Synthetic executor test goal.",
    )


def _make_action(
    action_id: str,
    *,
    required: bool = True,
    depends_on: list[str] | None = None,
) -> AgentAction:
    return AgentAction(
        id=action_id,
        name=action_id,
        description=f"Synthetic action: {action_id}",
        required=required,
        depends_on=depends_on or [],
    )


def _make_plan(actions: list[AgentAction]) -> AgentPlan:
    return AgentPlan(
        id="plan_goal_001",
        goal=_make_goal(),
        actions=actions,
    )


def _successful_handler(state: AgentState, action: AgentAction) -> None:
    state.metadata[action.id] = "handled"


def _failing_handler(state: AgentState, action: AgentAction) -> None:
    raise RuntimeError("handler exploded")
