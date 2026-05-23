"""Controlled benchmark outcome collection for GPUBoost Phase 11.8/11.9.

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
from collections import Counter
from pathlib import Path
from typing import Any

from gpuboost.comparison.engine import compare_benchmarks
from gpuboost.dataset.export import (
    build_dataset_manifest,
    export_dataset_jsonl,
    export_manifest,
    export_validation_report,
)
from gpuboost.dataset.readiness import (
    analyze_training_readiness,
    write_training_readiness_reports,
)
from gpuboost.dataset.validation import validate_dataset_rows
from gpuboost.schemas.comparison import ComparisonMetricValue, ComparisonResult
from gpuboost.schemas.dataset import (
    DatasetLabel,
    DatasetPrivacyFlags,
    DatasetRow,
    DatasetValue,
    create_timestamp,
)


OUTCOME_COLLECTION_SCHEMA_VERSION = "dataset.outcome_collection.v1"

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
    features: dict | None = None,
    metadata: dict | None = None,
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
        features=features,
        metadata=metadata,
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
            features=pair.get("features") if isinstance(pair.get("features"), dict) else None,
            metadata=pair.get("metadata") if isinstance(pair.get("metadata"), dict) else None,
            output_dir=output_dir,
        )
        rows.append(row)
    return rows


def load_outcome_pairs_file(filepath: str) -> list[dict]:
    """Load and validate a local outcome pair JSON file."""

    path = Path(filepath)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid outcome pairs JSON in {path}: {exc.msg}") from exc

    if isinstance(loaded, list):
        pairs = loaded
    elif isinstance(loaded, dict) and isinstance(loaded.get("pairs"), list):
        pairs = loaded["pairs"]
    else:
        raise ValueError("Outcome pairs JSON must be a list or an object with a pairs list.")

    normalized_pairs = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"Outcome pair at index {index} must be an object.")
        normalized_pairs.append(_normalize_pair(pair, index))
    return normalized_pairs


def collect_outcomes_from_pairs_file(
    pairs_file: str,
    output_dir: str = "data/gpuboost/generated/outcomes",
    dataset_name: str = "gpuboost_controlled_outcomes",
    dataset_version: str = "0.1.0",
    validate: bool = True,
) -> dict:
    """Collect labeled outcome rows from a local pair file and export artifacts."""

    generated_at = create_timestamp()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    errors: list[dict[str, DatasetValue]] = []
    pairs = load_outcome_pairs_file(pairs_file)
    pairs_base_dir = Path(pairs_file).parent
    rows: list[DatasetRow] = []

    for index, pair in enumerate(pairs):
        resolved_pair = _resolve_pair_paths(pair, pairs_base_dir)
        try:
            row, _comparison = collect_outcome_from_benchmark_json(
                baseline_json_path=str(resolved_pair["baseline_json_path"]),
                optimized_json_path=str(resolved_pair["optimized_json_path"]),
                row_id=_optional_string(resolved_pair.get("row_id")),
                workload_name=_optional_string(resolved_pair.get("workload_name")),
                hardware=(
                    resolved_pair.get("hardware")
                    if isinstance(resolved_pair.get("hardware"), dict)
                    else None
                ),
                features=(
                    resolved_pair.get("features")
                    if isinstance(resolved_pair.get("features"), dict)
                    else None
                ),
                metadata=(
                    resolved_pair.get("metadata")
                    if isinstance(resolved_pair.get("metadata"), dict)
                    else None
                ),
                output_dir=None,
            )
        except (OSError, ValueError) as error:
            errors.append(
                {
                    "pair_index": index,
                    "row_id": _optional_string(pair.get("row_id")),
                    "error": _format_error(error),
                }
            )
            continue
        rows.append(row)

    validation_report = validate_dataset_rows(rows)
    manifest = build_dataset_manifest(
        rows=rows,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        warnings=warnings,
    )
    readiness = analyze_training_readiness(rows)

    dataset_path = output_path / "outcome_dataset.jsonl"
    manifest_path = output_path / "outcome_manifest.json"
    validation_path = output_path / "outcome_validation_report.json"
    report_json_path = output_path / "outcome_collection_report.json"
    report_md_path = output_path / "outcome_collection_report.md"

    dataset_report = export_dataset_jsonl(rows, str(dataset_path), validate=validate)
    export_manifest(manifest, str(manifest_path))
    export_validation_report(validation_report, str(validation_path))
    readiness_json_path, readiness_md_path = write_training_readiness_reports(
        readiness,
        manifest_dir=str(output_path),
    )

    label_counts = dict(Counter(row.label.value for row in rows))
    output_files = {
        "outcome_dataset_jsonl": str(dataset_path) if dataset_path.exists() else None,
        "outcome_manifest_json": str(manifest_path),
        "outcome_validation_report_json": str(validation_path),
        "outcome_collection_report_json": str(report_json_path),
        "outcome_collection_report_md": str(report_md_path),
        "training_readiness_report_json": readiness_json_path,
        "training_readiness_report_md": readiness_md_path,
    }
    if dataset_report.status == "failed" and not dataset_path.exists():
        warnings.append("Outcome dataset JSONL was not written because validation failed.")

    summary: dict[str, Any] = {
        "schema_version": OUTCOME_COLLECTION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "pairs_file": str(pairs_file),
        "output_dir": str(output_path),
        "pair_count": len(pairs),
        "collected_row_count": len(rows),
        "label_counts": label_counts,
        "validation_status": validation_report.status,
        "readiness_status": readiness["status"],
        "output_files": output_files,
        "warnings": warnings,
        "errors": errors,
    }

    _write_json(report_json_path, summary)
    report_md_path.write_text(_build_collection_markdown(summary), encoding="utf-8")
    return summary


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


def _normalize_pair(pair: dict, index: int) -> dict:
    baseline_path = pair.get("baseline_json_path")
    optimized_path = pair.get("optimized_json_path")
    if not isinstance(baseline_path, str) or not baseline_path.strip():
        raise ValueError(
            f"Outcome pair at index {index} requires baseline_json_path."
        )
    if not isinstance(optimized_path, str) or not optimized_path.strip():
        raise ValueError(
            f"Outcome pair at index {index} requires optimized_json_path."
        )

    normalized = {
        "baseline_json_path": baseline_path,
        "optimized_json_path": optimized_path,
    }
    for key in ("row_id", "workload_name"):
        value = pair.get(key)
        if value is not None:
            normalized[key] = str(value)
    for key in ("hardware", "features", "metadata"):
        value = pair.get(key)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(
                    f"Outcome pair at index {index} field {key} must be an object."
                )
            normalized[key] = value
    return normalized


def _resolve_pair_paths(pair: dict, base_dir: Path) -> dict:
    resolved = dict(pair)
    for key in ("baseline_json_path", "optimized_json_path"):
        path = Path(str(pair[key]))
        if not path.is_absolute():
            path = base_dir / path
        resolved[key] = str(path)
    return resolved


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_collection_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# GPUBoost Outcome Collection Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Pairs file: {summary['pairs_file']}",
        f"Output directory: {summary['output_dir']}",
        "",
        "## Summary",
        f"- Pairs: {summary['pair_count']}",
        f"- Collected rows: {summary['collected_row_count']}",
        f"- Validation: {summary['validation_status']}",
        f"- Readiness: {summary['readiness_status']}",
        "",
        "## Labels",
    ]
    label_counts = summary["label_counts"]
    if label_counts:
        for label, count in sorted(label_counts.items()):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Errors"])
    if summary["errors"]:
        for error in summary["errors"]:
            lines.append(
                "- "
                f"pair_index={error.get('pair_index')} "
                f"row_id={error.get('row_id') or 'none'} "
                f"error={error.get('error')}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings"])
    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("- none")

    lines.append("")
    return "\n".join(lines)


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


def _format_error(error: Exception) -> str:
    message = str(error)
    if message:
        return message
    return error.__class__.__name__


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
