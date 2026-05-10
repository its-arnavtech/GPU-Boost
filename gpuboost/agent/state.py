"""Working-memory state for a single GPUBoost agent run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gpuboost.schemas.agent import AgentEvent, AgentGoal, AgentPlan, create_timestamp


@dataclass(slots=True)
class AgentState:
    """Intermediate artifacts and events for one agent workflow."""

    goal: AgentGoal
    plan: AgentPlan | None = None
    gpu_profile: dict[str, Any] | None = None
    benchmark_result: dict[str, Any] | None = None
    advisor_result: dict[str, Any] | None = None
    code_analysis: dict[str, Any] | None = None
    patch_plan: dict[str, Any] | None = None
    diff: str | None = None
    warnings: list[str] = field(default_factory=list)
    completed_actions: list[str] = field(default_factory=list)
    failed_actions: list[str] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(
        self,
        level: str,
        message: str,
        action_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> AgentEvent:
        """Append and return a timestamped event."""

        event = AgentEvent(
            timestamp=create_timestamp(),
            action_id=action_id,
            level=level,
            message=message,
            data=data or {},
        )
        self.events.append(event)
        return event

    def add_warning(self, message: str, action_id: str | None = None) -> None:
        """Record a warning and mirror it as a warning event."""

        self.warnings.append(message)
        self.add_event(
            level="warning",
            message=message,
            action_id=action_id,
        )

    def mark_completed(self, action_id: str) -> None:
        """Mark an action completed and record an info event."""

        if action_id not in self.completed_actions:
            self.completed_actions.append(action_id)
        self.add_event(
            level="info",
            message=f"Action completed: {action_id}",
            action_id=action_id,
        )

    def mark_failed(self, action_id: str, error: str) -> None:
        """Mark an action failed and record an error event."""

        if action_id not in self.failed_actions:
            self.failed_actions.append(action_id)
        self.add_event(
            level="error",
            message=f"Action failed: {action_id}",
            action_id=action_id,
            data={"error": error},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the state as JSON-serializable data."""

        return asdict(self)

    def has_failures(self) -> bool:
        """Return whether any actions have failed."""

        return bool(self.failed_actions)

    def summary_counts(self) -> dict[str, int]:
        """Return counts of key state collections."""

        return {
            "warnings": len(self.warnings),
            "completed_actions": len(self.completed_actions),
            "failed_actions": len(self.failed_actions),
            "events": len(self.events),
        }
