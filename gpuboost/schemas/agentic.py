"""Schemas for human-approved agentic optimization runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


AGENTIC_OPTIMIZATION_SCHEMA_VERSION = "agentic.optimization.v1"


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


class OptimizationLifecycleStatus(str, Enum):
    """Lifecycle states for approved optimization runs."""

    ANALYZED = "ANALYZED"
    PLANNED = "PLANNED"
    TRIALED = "TRIALED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    APPLYING = "APPLYING"
    VALIDATING = "VALIDATING"
    BENCHMARKING = "BENCHMARKING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


class ApprovalState(str, Enum):
    """Approval states for an optimization plan."""

    NOT_REQUESTED = "not_requested"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class RiskLevel(str, Enum):
    """Conservative risk labels for proposed edits."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AcceptancePolicy(str, Enum):
    """Post-application acceptance policy."""

    VALIDATION_ONLY = "validation-only"
    NO_REGRESSION = "no-regression"
    MINIMUM_SPEEDUP = "minimum-speedup"
    MANUAL_REVIEW = "manual-review"


@dataclass(frozen=True, slots=True)
class ProposedEdit:
    """One exact-source edit that can be applied after approval."""

    edit_id: str
    action_id: str
    path: str
    start_line: int | None
    end_line: int | None
    expected_before: str
    replacement: str
    rationale: str
    risk: RiskLevel

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe data."""

        data = asdict(self)
        data["risk"] = self.risk.value
        return data


@dataclass(frozen=True, slots=True)
class OptimizationApproval:
    """Explicit approval tied to one immutable plan digest."""

    run_id: str
    plan_id: str
    plan_digest: str
    approved_action_ids: tuple[str, ...]
    approved_by: str
    approved_at: str
    target_file_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe data."""

        data = asdict(self)
        data["approved_action_ids"] = list(self.approved_action_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationApproval":
        """Create an approval object from persisted JSON data."""

        return cls(
            run_id=str(data["run_id"]),
            plan_id=str(data["plan_id"]),
            plan_digest=str(data["plan_digest"]),
            approved_action_ids=tuple(str(item) for item in data["approved_action_ids"]),
            approved_by=str(data["approved_by"]),
            approved_at=str(data["approved_at"]),
            target_file_hash=str(data["target_file_hash"]),
        )


@dataclass(slots=True)
class AgenticOptimizationRun:
    """Persisted human-approved optimization run record."""

    run_id: str
    target_repository_root: str
    target_file: str
    original_file_hash: str
    plan_id: str
    plan_digest: str
    created_at: str
    tool_version: str
    proposed_actions: list[dict[str, Any]]
    proposed_edits: list[dict[str, Any]]
    generated_diff: str
    patch_plan: dict[str, Any]
    lifecycle_status: OptimizationLifecycleStatus
    approval_state: ApprovalState = ApprovalState.AWAITING_APPROVAL
    approved_action_ids: list[str] = field(default_factory=list)
    approver: str | None = None
    approver_confirmation_timestamp: str | None = None
    trial_result: dict[str, Any] | None = None
    pre_application_backup_path: str | None = None
    application_result: dict[str, Any] | None = None
    validation_result: dict[str, Any] | None = None
    benchmark_result: dict[str, Any] | None = None
    rollback_result: dict[str, Any] | None = None
    final_status: str | None = None
    approval: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe data."""

        data = asdict(self)
        data["schema_version"] = AGENTIC_OPTIMIZATION_SCHEMA_VERSION
        data["lifecycle_status"] = self.lifecycle_status.value
        data["approval_state"] = self.approval_state.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgenticOptimizationRun":
        """Create a run from persisted JSON data."""

        return cls(
            run_id=str(data["run_id"]),
            target_repository_root=str(data["target_repository_root"]),
            target_file=str(data["target_file"]),
            original_file_hash=str(data["original_file_hash"]),
            plan_id=str(data["plan_id"]),
            plan_digest=str(data["plan_digest"]),
            created_at=str(data["created_at"]),
            tool_version=str(data["tool_version"]),
            proposed_actions=list(data.get("proposed_actions", [])),
            proposed_edits=list(data.get("proposed_edits", [])),
            generated_diff=str(data.get("generated_diff", "")),
            patch_plan=dict(data.get("patch_plan", {})),
            lifecycle_status=OptimizationLifecycleStatus(
                data.get("lifecycle_status", OptimizationLifecycleStatus.FAILED.value)
            ),
            approval_state=ApprovalState(
                data.get("approval_state", ApprovalState.NOT_REQUESTED.value)
            ),
            approved_action_ids=[
                str(item) for item in data.get("approved_action_ids", [])
            ],
            approver=data.get("approver"),
            approver_confirmation_timestamp=data.get(
                "approver_confirmation_timestamp"
            ),
            trial_result=data.get("trial_result"),
            pre_application_backup_path=data.get("pre_application_backup_path"),
            application_result=data.get("application_result"),
            validation_result=data.get("validation_result"),
            benchmark_result=data.get("benchmark_result"),
            rollback_result=data.get("rollback_result"),
            final_status=data.get("final_status"),
            approval=data.get("approval"),
            warnings=[str(item) for item in data.get("warnings", [])],
            error=data.get("error"),
        )
