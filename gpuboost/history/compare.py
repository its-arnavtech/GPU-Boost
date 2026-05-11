"""Safe comparison helpers for local GPUBoost history records."""

from __future__ import annotations

from gpuboost.schemas.history import (
    HistoryCompareResult,
    HistoryRunRecord,
    HistoryValue,
    create_timestamp,
)


def compare_history_runs(
    left: HistoryRunRecord,
    right: HistoryRunRecord,
) -> HistoryCompareResult:
    """Compare selected safe fields from two history records."""

    tracked_fields = {
        "status": (left.status, right.status),
        "command": (left.command, right.command),
        "goal_kind": (left.goal_kind, right.goal_kind),
        "script_sha256": (left.script_sha256, right.script_sha256),
        "gpu_name": (left.gpu_name, right.gpu_name),
        "cuda_available": (left.cuda_available, right.cuda_available),
        "trial_status": (
            left.trial_summary.get("status"),
            right.trial_summary.get("status"),
        ),
        "comparison_overall_verdict": (
            left.comparison_summary.get("overall_verdict"),
            right.comparison_summary.get("overall_verdict"),
        ),
        "has_diff": (left.metadata.get("has_diff"), right.metadata.get("has_diff")),
        "has_trial": (left.metadata.get("has_trial"), right.metadata.get("has_trial")),
        "has_comparison": (
            left.metadata.get("has_comparison"),
            right.metadata.get("has_comparison"),
        ),
    }
    changed_fields: dict[str, HistoryValue] = {}

    for field_name, (left_value, right_value) in tracked_fields.items():
        if left_value != right_value:
            changed_fields[field_name] = _format_change(left_value, right_value)

    if changed_fields:
        summary = f"Changed fields: {', '.join(changed_fields)}"
    else:
        summary = "No tracked fields changed."

    return HistoryCompareResult(
        generated_at=create_timestamp(),
        status="ok",
        left_run_id=left.run_id,
        right_run_id=right.run_id,
        summary=summary,
        changed_fields=changed_fields,
        error=None,
    )


def _format_change(left_value: object, right_value: object) -> str:
    return f"{left_value} -> {right_value}"
