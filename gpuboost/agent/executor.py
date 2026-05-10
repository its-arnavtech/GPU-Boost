"""Generic injected-handler executor for GPUBoost agent plans."""

from __future__ import annotations

from typing import Callable

from gpuboost.agent.state import AgentState
from gpuboost.schemas.agent import (
    AgentAction,
    AgentPlan,
    AgentRunResult,
    create_timestamp,
)


ActionHandler = Callable[[AgentState, AgentAction], None]


class AgentExecutor:
    """Execute an AgentPlan with injected deterministic action handlers."""

    def __init__(self, handlers: dict[str, ActionHandler] | None = None) -> None:
        self.handlers = handlers or {}

    def execute_plan(self, plan: AgentPlan) -> AgentRunResult:
        """Execute plan actions in order and return a run result."""

        state = AgentState(goal=plan.goal, plan=plan)
        state.add_event(level="info", message="Agent execution started.")

        required_failure_error: str | None = None

        for action in plan.actions:
            failed_dependency = _first_failed_dependency(action, state)
            if failed_dependency is not None:
                action.status = "skipped"
                action.error = f"Skipped because dependency failed: {failed_dependency}"
                state.add_warning(action.error, action_id=action.id)
                continue

            handler = self.handlers.get(action.name)
            if handler is None:
                if action.required:
                    action.status = "failed"
                    action.error = f"No handler registered for action: {action.name}"
                    state.mark_failed(action.id, action.error)
                    required_failure_error = action.error
                    break

                action.status = "skipped"
                action.error = (
                    f"No handler registered for optional action: {action.name}"
                )
                state.add_warning(action.error, action_id=action.id)
                continue

            action.status = "running"
            state.add_event(
                level="info",
                message=f"Starting action: {action.name}",
                action_id=action.id,
            )

            try:
                handler(state, action)
            except Exception as error:  # noqa: BLE001 - handlers are user-injected
                action.status = "failed"
                action.error = str(error)
                state.mark_failed(action.id, action.error)
                if action.required:
                    required_failure_error = action.error
                    break
                continue

            action.status = "completed"
            action.error = None
            state.mark_completed(action.id)

        status = _compute_status(plan)
        error = required_failure_error if status == "error" else None
        state.add_event(
            level="info",
            message=f"Agent execution finished with status: {status}",
        )

        return AgentRunResult(
            generated_at=create_timestamp(),
            goal=plan.goal,
            plan=plan,
            status=status,
            events=state.events,
            warnings=state.warnings,
            error=error,
            artifacts={"diff": state.diff},
        )


def default_handlers() -> dict[str, ActionHandler]:
    """Return default deterministic action handlers."""

    from gpuboost.agent.handlers import default_handlers as _default_handlers

    return _default_handlers()


def _first_failed_dependency(
    action: AgentAction,
    state: AgentState,
) -> str | None:
    for dependency_id in action.depends_on:
        if dependency_id in state.failed_actions:
            return dependency_id
    return None


def _compute_status(plan: AgentPlan) -> str:
    if any(action.required and action.status == "failed" for action in plan.actions):
        return "error"
    if any(action.status == "failed" for action in plan.actions):
        return "partial"
    return "ok"
