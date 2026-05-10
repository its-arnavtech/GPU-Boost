"""Dataclass schemas for GPUBoost Phase 5 agent output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


AgentValue = str | int | float | bool | None


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AgentGoal:
    """A deterministic agent goal request."""

    id: str
    kind: str
    description: str
    script_path: str | None = None
    options: dict[str, AgentValue] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the goal as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class AgentAction:
    """One planned agent action."""

    id: str
    name: str
    description: str
    required: bool
    depends_on: list[str] = field(default_factory=list)
    inputs: dict[str, AgentValue] = field(default_factory=dict)
    status: str = "pending"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the action as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class AgentPlan:
    """A deterministic plan for an agent goal."""

    id: str
    goal: AgentGoal
    actions: list[AgentAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class AgentEvent:
    """A structured event emitted during an agent run."""

    timestamp: str
    action_id: str | None
    level: str
    message: str
    data: dict[str, AgentValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the event as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class AgentRunResult:
    """Result envelope for a deterministic agent run."""

    generated_at: str
    goal: AgentGoal
    plan: AgentPlan
    status: str
    events: list[AgentEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the run result as JSON-serializable data."""

        return asdict(self)
