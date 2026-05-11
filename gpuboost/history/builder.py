"""Build safe local history records from GPUBoost agent runs."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from gpuboost.schemas.agent import AgentRunResult
from gpuboost.schemas.history import HistoryRunRecord, HistoryValue, create_timestamp


def hash_text(text: str) -> str:
    """Return a SHA256 hex digest for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file_if_exists(filepath: str | None) -> str | None:
    """Return a SHA256 hex digest for a file, or None when unavailable."""

    if filepath is None:
        return None

    path = Path(filepath)
    if not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_history_run_record(
    result: AgentRunResult,
    command: str = "agent optimize",
    run_id: str | None = None,
) -> HistoryRunRecord:
    """Build a safe history record from an agent run result."""

    created_at = create_timestamp()
    resolved_run_id = run_id or _generate_run_id(result, created_at)
    gpu_name, cuda_available = extract_gpu_summary(result)
    action_statuses = {action.id: action.status for action in result.plan.actions}
    trial_summary = extract_trial_summary(result)
    comparison_summary = extract_comparison_summary(result)
    patch_summary = extract_patch_summary(result)

    metadata: dict[str, HistoryValue] = {
        "event_count": len(result.events),
        "action_count": len(result.plan.actions),
        "completed_action_count": sum(
            1 for action in result.plan.actions if action.status == "completed"
        ),
        "failed_action_count": sum(
            1 for action in result.plan.actions if action.status == "failed"
        ),
        "has_diff": bool(_artifact(result, "diff")),
        "has_trial": bool(trial_summary),
        "has_comparison": bool(comparison_summary),
    }

    return HistoryRunRecord(
        run_id=resolved_run_id,
        created_at=created_at,
        status=result.status,
        command=command,
        schema_version="history.run.v1",
        goal_kind=result.goal.kind,
        goal_description=result.goal.description,
        script_path=result.goal.script_path,
        script_sha256=hash_file_if_exists(result.goal.script_path),
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        benchmark_summary=extract_benchmark_summary(result),
        advisor_summary=extract_advisor_summary(result),
        code_summary=extract_code_summary(result),
        patch_summary=patch_summary,
        trial_summary=trial_summary,
        comparison_summary=comparison_summary,
        action_statuses=action_statuses,
        warnings=list(result.warnings),
        error=result.error,
        metadata=metadata,
    )


def extract_gpu_summary(result: AgentRunResult) -> tuple[str | None, bool | None]:
    """Extract safe GPU identity and CUDA availability from artifacts."""

    gpu = _as_mapping(_artifact(result, "gpu") or _artifact(result, "gpu_profile"))
    if not gpu:
        return None, None

    gpu_name = _scalar_or_none(gpu.get("gpu_name") or gpu.get("name"))
    cuda_available = _bool_or_none(gpu.get("cuda_available"))

    gpus = gpu.get("gpus")
    if gpu_name is None and isinstance(gpus, list) and gpus:
        first_gpu = _as_mapping(gpus[0])
        gpu_name = _scalar_or_none(first_gpu.get("name"))

    return str(gpu_name) if gpu_name is not None else None, cuda_available


def extract_benchmark_summary(result: AgentRunResult) -> dict[str, HistoryValue]:
    """Extract safe benchmark counts and status from artifacts."""

    return _count_summary(
        _artifact(result, "benchmark") or _artifact(result, "benchmark_result"),
        count_keys=("results", "metrics"),
        count_name="metric_count",
    )


def extract_advisor_summary(result: AgentRunResult) -> dict[str, HistoryValue]:
    """Extract safe advisor counts and status from artifacts."""

    return _count_summary(
        _artifact(result, "advisor") or _artifact(result, "advisor_result"),
        count_keys=("recommendations",),
        count_name="recommendation_count",
    )


def extract_code_summary(result: AgentRunResult) -> dict[str, HistoryValue]:
    """Extract safe code-analysis counts and status from artifacts."""

    return _count_summary(
        _artifact(result, "code") or _artifact(result, "code_analysis"),
        count_keys=("findings",),
        count_name="finding_count",
    )


def extract_patch_summary(result: AgentRunResult) -> dict[str, HistoryValue]:
    """Extract safe patch-plan counts and diff presence from artifacts."""

    summary = _count_summary(
        _artifact(result, "patch") or _artifact(result, "patch_plan"),
        count_keys=("suggestions", "edits"),
        count_name="suggestion_count",
    )
    if _artifact(result, "diff"):
        summary["has_diff"] = True
    return summary


def extract_trial_summary(result: AgentRunResult) -> dict[str, HistoryValue]:
    """Extract safe trial status fields without stdout or stderr."""

    trial = _as_mapping(_artifact(result, "trial"))
    if not trial:
        return {}

    return _pick_scalars(
        trial,
        (
            "status",
            "patch_applied",
            "syntax_check_status",
            "test_status",
            "original_file_unchanged",
        ),
    )


def extract_comparison_summary(result: AgentRunResult) -> dict[str, HistoryValue]:
    """Extract safe comparison status fields."""

    comparison = _as_mapping(_artifact(result, "comparison"))
    if not comparison:
        return {}

    return _pick_scalars(comparison, ("status", "overall_verdict"))


def _generate_run_id(result: AgentRunResult, created_at: str) -> str:
    compact_timestamp = (
        created_at.replace("-", "")
        .replace(":", "")
        .replace("+", "")
        .replace(".", "")
    )
    digest = hash_text(f"{created_at}:{result.goal.kind}:{result.goal.description}")[:8]
    return f"run_{compact_timestamp}_{digest}"


def _artifact(result: AgentRunResult, key: str) -> Any:
    return result.artifacts.get(key)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return converted
    return {}


def _count_summary(
    value: Any,
    count_keys: tuple[str, ...],
    count_name: str,
) -> dict[str, HistoryValue]:
    data = _as_mapping(value)
    if not data:
        return {}

    summary = _pick_scalars(data, ("status", "error"))
    for key in count_keys:
        items = data.get(key)
        if isinstance(items, list):
            summary[count_name] = len(items)
            break
    return summary


def _pick_scalars(
    data: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, HistoryValue]:
    summary: dict[str, HistoryValue] = {}
    for key in keys:
        value = _scalar_or_none(data.get(key))
        if value is not None:
            summary[key] = value
    return summary


def _scalar_or_none(value: Any) -> HistoryValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
