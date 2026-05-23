"""Convert local GPUBoost history records into safe dataset rows."""

from __future__ import annotations

from gpuboost.schemas.dataset import (
    DatasetLabel,
    DatasetPrivacyFlags,
    DatasetRow,
    DatasetValue,
    create_timestamp,
)
from gpuboost.schemas.history import HistoryRunRecord, HistoryValue


_COMPLETED_STATUSES = {"ok", "completed", "success", "done"}
_FAILED_STATUSES = {"failed", "error"}
_UNSAFE_KEY_PARTS = (
    "raw",
    "source_code",
    "diff",
    "stdout",
    "stderr",
    "path",
)


def history_record_to_dataset_row(
    record: HistoryRunRecord,
    row_id: str | None = None,
    split: str | None = None,
) -> DatasetRow:
    """Convert a local history record into a privacy-safe dataset row."""

    converter_warnings: list[str] = []
    action_statuses = list(record.action_statuses.values())

    features: dict[str, DatasetValue] = {
        "action_count": len(record.action_statuses),
        "completed_action_count": sum(
            1 for status in action_statuses if status in _COMPLETED_STATUSES
        ),
        "failed_action_count": sum(
            1 for status in action_statuses if status in _FAILED_STATUSES
        ),
        "has_diff": bool(record.patch_summary),
        "has_trial": record.has_trial(),
        "has_comparison": record.has_comparison(),
    }
    _add_safe_values(features, "code", record.code_summary, converter_warnings)
    _add_safe_values(features, "patch", record.patch_summary, converter_warnings)
    _add_safe_values(features, "trial", record.trial_summary, converter_warnings)

    metrics: dict[str, DatasetValue] = {}
    _add_safe_values(metrics, "benchmark", record.benchmark_summary, converter_warnings)
    _add_safe_values(
        metrics,
        "comparison",
        record.comparison_summary,
        converter_warnings,
    )

    comparison_verdict = _safe_get(record.comparison_summary, "overall_verdict")
    trial_status = _safe_get(record.trial_summary, "status")

    return DatasetRow(
        row_id=row_id or f"row_{record.run_id}",
        created_at=create_timestamp(),
        source="gpuboost_history",
        row_type="optimization_outcome",
        hardware={
            "gpu_name": record.gpu_name,
            "cuda_available": record.cuda_available,
        },
        workload={
            "command": record.command,
            "goal_kind": record.goal_kind,
            "has_script_path": record.script_path is not None,
            "script_sha256": record.script_sha256,
        },
        features=features,
        metrics=metrics,
        label=derive_label_from_history_record(record),
        privacy=DatasetPrivacyFlags(),
        split=split,
        quality_score=estimate_history_row_quality(record),
        warnings=[*record.warnings, *converter_warnings],
        metadata={
            "run_id": record.run_id,
            "status": record.status,
            "schema_version": record.schema_version,
            "comparison_verdict": comparison_verdict,
            "trial_status": trial_status,
            "error_present": record.error is not None,
        },
    )


def derive_label_from_history_record(record: HistoryRunRecord) -> DatasetLabel:
    """Derive a dataset label from comparison or trial outcome summaries."""

    verdict = record.comparison_summary.get("overall_verdict")
    if isinstance(verdict, str):
        normalized_verdict = verdict.strip().lower()
        if normalized_verdict == "improved":
            return DatasetLabel(
                value="improved",
                source="comparison",
                confidence=0.9,
            )
        if normalized_verdict == "regressed":
            return DatasetLabel(
                value="regressed",
                source="comparison",
                confidence=0.9,
            )
        if normalized_verdict == "unchanged":
            return DatasetLabel(
                value="neutral",
                source="comparison",
                confidence=0.8,
            )
        if normalized_verdict == "mixed":
            return DatasetLabel(
                value="neutral",
                source="comparison",
                confidence=0.5,
            )

    if record.status == "error":
        source = "trial" if record.trial_summary else "unknown"
        return DatasetLabel(value="failed", source=source, confidence=0.7)

    trial_status = record.trial_summary.get("status")
    if isinstance(trial_status, str) and trial_status.strip().lower() in {
        "failed",
        "error",
    }:
        return DatasetLabel(value="failed", source="trial", confidence=0.8)

    return DatasetLabel(value="unknown", source="unknown", confidence=None)


def estimate_history_row_quality(record: HistoryRunRecord) -> float:
    """Estimate deterministic row quality from available safe history signals."""

    score = 0.5
    if record.script_sha256:
        score += 0.2
    if record.benchmark_summary:
        score += 0.1
    if record.trial_summary:
        score += 0.1
    if record.comparison_summary:
        score += 0.1
    if record.status == "error":
        score -= 0.2
    return round(max(0.0, min(1.0, score)), 10)


def history_records_to_dataset_rows(
    records: list[HistoryRunRecord],
    split: str | None = None,
) -> list[DatasetRow]:
    """Convert history records to dataset rows while preserving order."""

    return [
        history_record_to_dataset_row(record, split=split)
        for record in records
    ]


def _add_safe_values(
    target: dict[str, DatasetValue],
    prefix: str,
    values: dict[str, HistoryValue],
    warnings: list[str],
) -> None:
    for key, value in values.items():
        if _is_unsafe_key(key):
            warnings.append(f"Skipped unsafe summary field: {prefix}.{key}")
            continue
        target[f"{prefix}_{key}"] = value


def _is_unsafe_key(key: str) -> bool:
    normalized = key.strip().lower()
    return any(part in normalized for part in _UNSAFE_KEY_PARTS)


def _safe_get(values: dict[str, HistoryValue], key: str) -> DatasetValue:
    if _is_unsafe_key(key):
        return None
    return values.get(key)
