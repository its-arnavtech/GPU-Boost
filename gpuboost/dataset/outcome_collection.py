"""Controlled benchmark outcome collection for GPUBoost Phase 11.8.

This module currently compares existing local benchmark JSON files. It does not
execute arbitrary benchmark commands, run user code, scrape, download, or call
external APIs.

Future support may accept explicit user-provided benchmark commands, but that
requires safety controls around allowlists, working directories, timeouts,
resource limits, and stdout/stderr handling before command execution is added.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from gpuboost.comparison.engine import compare_benchmarks
from gpuboost.schemas.comparison import ComparisonMetricValue, ComparisonResult
from gpuboost.schemas.dataset import (
    DatasetLabel,
    DatasetPrivacyFlags,
    DatasetRow,
    DatasetValue,
    create_timestamp,
)


_UNSAFE_KEY_PARTS = (
    "raw",
    "source_code",
    "diff",
    "stdout",
    "stderr",
    "path",
)


def comparison_result_to_label(comparison: ComparisonResult) -> DatasetLabel:
    """Convert a benchmark comparison verdict into a dataset label."""

    if comparison.status.strip().lower() == "error":
        return DatasetLabel(value="failed", source="comparison", confidence=0.7)

    verdict = comparison.overall_verdict.strip().lower()
    if verdict == "improved":
        return DatasetLabel(value="improved", source="comparison", confidence=0.9)
    if verdict == "regressed":
        return DatasetLabel(value="regressed", source="comparison", confidence=0.9)
    if verdict == "unchanged":
        return DatasetLabel(value="neutral", source="comparison", confidence=0.8)
    if verdict == "mixed":
        return DatasetLabel(value="neutral", source="comparison", confidence=0.5)
    return DatasetLabel(value="unknown", source="unknown", confidence=None)


def comparison_result_to_dataset_row(
    comparison: ComparisonResult,
    row_id: str,
    source: str = "controlled_experiment",
    workload_name: str | None = None,
    hardware: dict | None = None,
    features: dict | None = None,
    metadata: dict | None = None,
) -> DatasetRow:
    """Create a privacy-safe dataset row from a benchmark comparison result."""

    label = comparison_result_to_label(comparison)
    row_features = _safe_scalar_mapping(features)
    row_features.update(_comparison_features(comparison))

    row_metadata = _safe_scalar_mapping(metadata)
    row_metadata.update(
        {
            "comparison_status": comparison.status,
            "comparison_warning_count": len(comparison.warnings),
            "comparison_error_present": comparison.error is not None,
        }
    )

    return DatasetRow(
        row_id=row_id,
        created_at=create_timestamp(),
        source=source,
        row_type="controlled_experiment",
        hardware=_safe_scalar_mapping(hardware),
        workload=_workload_mapping(workload_name),
        features=row_features,
        metrics=_flatten_metric_deltas(comparison),
        label=label,
        privacy=DatasetPrivacyFlags(),
        quality_score=_quality_score_from_label(label),
        warnings=_safe_warning_list(comparison.warnings),
        metadata=row_metadata,
    )


def collect_outcome_from_benchmark_json(
    baseline_json_path: str,
    optimized_json_path: str,
    row_id: str | None = None,
    workload_name: str | None = None,
    hardware: dict | None = None,
    output_dir: str | None = None,
) -> tuple[DatasetRow, ComparisonResult]:
    """Compare two local benchmark JSON files and return a labeled dataset row."""

    baseline_path = Path(baseline_json_path)
    optimized_path = Path(optimized_json_path)
    baseline = _load_benchmark_json(baseline_path)
    optimized = _load_benchmark_json(optimized_path)
    resolved_row_id = row_id or _derive_row_id(
        baseline_path=baseline_path,
        optimized_path=optimized_path,
        baseline=baseline,
        optimized=optimized,
    )

    comparison = compare_benchmarks(baseline, optimized)
    row = comparison_result_to_dataset_row(
        comparison=comparison,
        row_id=resolved_row_id,
        workload_name=workload_name,
        hardware=hardware,
    )

    if output_dir is not None:
        _write_outcome_artifacts(output_dir, row, comparison)

    return row, comparison


def collect_outcomes_from_pairs(
    pairs: list[dict],
    output_dir: str | None = None,
) -> list[DatasetRow]:
    """Collect outcome rows from a list of local baseline/optimized JSON pairs."""

    rows = []
    for pair in pairs:
        row, _comparison = collect_outcome_from_benchmark_json(
            baseline_json_path=str(pair.get("baseline_json_path") or ""),
            optimized_json_path=str(pair.get("optimized_json_path") or ""),
            row_id=_optional_string(pair.get("row_id")),
            workload_name=_optional_string(pair.get("workload_name")),
            hardware=pair.get("hardware") if isinstance(pair.get("hardware"), dict) else None,
            output_dir=output_dir,
        )
        rows.append(row)
    return rows


def _comparison_features(comparison: ComparisonResult) -> dict[str, DatasetValue]:
    metric_directions = [
        metric.direction
        for section in comparison.sections
        for metric in section.metrics
    ]
    return {
        "overall_verdict": comparison.overall_verdict,
        "section_count": len(comparison.sections),
        "improved_metric_count": metric_directions.count("improved"),
        "regressed_metric_count": metric_directions.count("regressed"),
        "unchanged_metric_count": metric_directions.count("unchanged"),
    }


def _flatten_metric_deltas(comparison: ComparisonResult) -> dict[str, DatasetValue]:
    metrics: dict[str, DatasetValue] = {}
    for section in comparison.sections:
        for metric in section.metrics:
            metric_name = _safe_key(metric.name)
            if metric_name is None:
                continue
            _set_unique_metric(metrics, f"before_{metric_name}", metric.before)
            _set_unique_metric(metrics, f"after_{metric_name}", metric.after)
            _set_unique_metric(metrics, f"delta_{metric_name}", metric.absolute_delta)
            _set_unique_metric(
                metrics,
                f"percent_delta_{metric_name}",
                metric.percent_delta,
            )
    return metrics


def _set_unique_metric(
    metrics: dict[str, DatasetValue],
    key: str,
    value: ComparisonMetricValue,
) -> None:
    if not _is_scalar_value_safe(value):
        return

    unique_key = key
    suffix = 2
    while unique_key in metrics:
        unique_key = f"{key}_{suffix}"
        suffix += 1
    metrics[unique_key] = value


def _load_benchmark_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark JSON file not found: {path}")

    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark JSON in {path}: {exc.msg}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"Benchmark JSON must contain an object: {path}")
    return loaded


def _derive_row_id(
    baseline_path: Path,
    optimized_path: Path,
    baseline: dict,
    optimized: dict,
) -> str:
    digest_source = json.dumps(
        {
            "baseline_name": baseline_path.name,
            "optimized_name": optimized_path.name,
            "baseline": baseline,
            "optimized": optimized,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
    baseline_stem = _safe_file_stem(baseline_path.stem) or "baseline"
    optimized_stem = _safe_file_stem(optimized_path.stem) or "optimized"
    return f"controlled_{baseline_stem}_vs_{optimized_stem}_{digest}"


def _write_outcome_artifacts(
    output_dir: str,
    row: DatasetRow,
    comparison: ComparisonResult,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = _safe_file_stem(row.row_id) or "outcome"
    _write_json(output_path / f"{stem}.comparison.json", comparison.to_dict())
    _write_json(output_path / f"{stem}.dataset_row.json", row.to_dict())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _safe_scalar_mapping(values: dict | None) -> dict[str, DatasetValue]:
    if not isinstance(values, dict):
        return {}

    safe_values: dict[str, DatasetValue] = {}
    for key, value in values.items():
        key_text = str(key)
        if _is_unsafe_key(key_text) or not _is_scalar_value_safe(value):
            continue
        safe_values[key_text] = value
    return safe_values


def _workload_mapping(workload_name: str | None) -> dict[str, DatasetValue]:
    if workload_name is None:
        return {}
    if not _is_scalar_value_safe(workload_name):
        return {}
    return {"workload_name": workload_name}


def _quality_score_from_label(label: DatasetLabel) -> float:
    if label.confidence is None:
        return 0.5
    return round(max(0.0, min(1.0, label.confidence)), 10)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _safe_key(value: str) -> str | None:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_")
    if not normalized or _is_unsafe_key(normalized):
        return None
    return normalized


def _safe_file_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")


def _is_unsafe_key(key: str) -> bool:
    normalized = key.strip().lower()
    return any(part in normalized for part in _UNSAFE_KEY_PARTS)


def _safe_warning_list(warnings: list[str]) -> list[str]:
    return [warning for warning in warnings if _is_scalar_value_safe(warning)]


def _is_scalar_value_safe(value: object) -> bool:
    if not isinstance(value, str | int | float | bool) and value is not None:
        return False
    if isinstance(value, str):
        return not (
            _looks_like_unified_diff(value)
            or _looks_like_python_source(value)
            or _contains_unsafe_marker(value)
        )
    return True


def _looks_like_unified_diff(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith("--- ") and "\n+++ " in stripped


def _looks_like_python_source(value: str) -> bool:
    if len(value) < 120 or "\n" not in value:
        return False
    source_markers = ("def ", "class ", "import ", "from ")
    marker_count = sum(1 for marker in source_markers if marker in value)
    return marker_count >= 2 and ("    " in value or "\n\t" in value)


def _contains_unsafe_marker(value: str) -> bool:
    normalized = value.strip().lower()
    return any(part in normalized for part in _UNSAFE_KEY_PARTS)
