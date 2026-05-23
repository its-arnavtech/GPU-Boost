"""Training-usefulness and readiness analysis for Phase 11 datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from gpuboost.dataset.validation import is_hardware_specs_context, validate_no_raw_sensitive_fields
from gpuboost.schemas.dataset import BenchmarkContextRow, DatasetRow


def analyze_training_readiness(
    rows: list[DatasetRow],
    context_rows: list[BenchmarkContextRow] | None = None,
    min_total_rows: int = 100,
    min_labeled_rows: int = 50,
    min_known_label_classes: int = 2,
) -> dict[str, Any]:
    """Analyze whether assembled rows are ready for Phase 12 training."""

    context_rows = list(context_rows or [])
    label_counts = Counter(row.label.value for row in rows)
    known_label_counts = Counter(
        row.label.value for row in rows if row.label.is_known()
    )
    source_counts = Counter(row.source for row in rows)
    split_counts = Counter(row.split or "unassigned" for row in rows)
    row_type_counts = Counter(row.row_type for row in rows)
    feature_presence = _presence_counts(row.features for row in rows)
    metric_presence = _presence_counts(row.metrics for row in rows)
    hardware_presence = _presence_counts(row.hardware for row in rows)

    labeled_rows = sum(1 for row in rows if row.is_labeled())
    unlabeled_rows = len(rows) - labeled_rows
    comparison_labels = sum(1 for row in rows if row.label.source == "comparison")
    trial_feature_rows = sum(
        1
        for row in rows
        if any(key == "has_trial" or key.startswith("trial_") for key in row.features)
    )
    hardware_feature_rows = sum(1 for row in rows if any(value is not None for value in row.hardware.values()))

    context_source_counts = Counter(row.source for row in context_rows)
    hardware_specs_count = sum(1 for row in context_rows if is_hardware_specs_context(row))
    benchmark_result_context_count = len(context_rows) - hardware_specs_count
    mlperf_row_count = context_source_counts.get("mlperf", 0)

    unsafe_row_count = sum(1 for row in rows if _row_has_unsafe_data(row))
    unsafe_context_count = sum(1 for row in context_rows if _context_row_has_unsafe_data(row))

    duplicate_row_id_count = _duplicate_value_count(row.row_id for row in rows if row.row_id)
    duplicate_script_sha256_count = _duplicate_value_count(
        row.workload.get("script_sha256")
        for row in rows
        if isinstance(row.workload.get("script_sha256"), str)
    )

    has_enough_rows = len(rows) >= min_total_rows
    has_enough_labeled_rows = labeled_rows >= min_labeled_rows
    has_enough_label_classes = len(known_label_counts) >= min_known_label_classes
    has_context_data = len(context_rows) > 0
    has_trial_features = trial_feature_rows > 0
    has_comparison_labels = comparison_labels > 0
    has_hardware_features = hardware_feature_rows > 0

    blockers: list[str] = []
    recommendations: list[str] = []

    if not has_enough_rows:
        blockers.append("Not enough total rows for training.")
        recommendations.append("Run GPUBoost with --save-history on real scripts.")
    if labeled_rows == 0:
        blockers.append("No labeled optimization outcome rows found.")
        recommendations.append("Run before/after comparisons to create improved/regressed/neutral labels.")
    elif not has_enough_labeled_rows:
        blockers.append("Not enough labeled rows for training.")
        recommendations.append("Run before/after comparisons to create improved/regressed/neutral labels.")
    if 0 < len(known_label_counts) < min_known_label_classes:
        blockers.append("Only one known label class is present.")
        recommendations.append("Collect more examples for regressed and neutral outcomes.")
    if not has_comparison_labels:
        blockers.append("No comparison-derived labels found.")
        recommendations.append("Run before/after comparisons to create improved/regressed/neutral labels.")
    if not has_context_data:
        recommendations.append("Import more MLPerf/OpenBenchmarking context rows.")
    if hardware_specs_count > 0:
        recommendations.append("Keep TechPowerUp rows as hardware context, not labels.")
    if not has_trial_features:
        recommendations.append("Capture more trial summary fields in GPUBoost history.")
    if not has_hardware_features:
        recommendations.append("Capture GPU/hardware details consistently in history rows.")

    severe_imbalance = _has_severe_label_imbalance(known_label_counts)
    if severe_imbalance:
        recommendations.append("Collect more examples for underrepresented label classes.")
    if duplicate_row_id_count:
        recommendations.append("Deduplicate repeated dataset rows before Phase 12 training.")
    if duplicate_script_sha256_count:
        recommendations.append("Diversify scripts or deduplicate repeated script_sha256 entries.")
    if unsafe_row_count or unsafe_context_count:
        recommendations.append("Remove unsafe rows before Phase 12 training.")

    recommendations = _unique_preserving_order(recommendations)

    if blockers:
        status = "not_ready"
    elif severe_imbalance or not has_trial_features or not has_hardware_features or unlabeled_rows > labeled_rows or unsafe_row_count or unsafe_context_count:
        status = "warning"
    else:
        status = "ready"

    return {
        "status": status,
        "total_rows": len(rows),
        "labeled_rows": labeled_rows,
        "unlabeled_rows": unlabeled_rows,
        "label_counts": dict(label_counts),
        "known_label_counts": dict(known_label_counts),
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "row_type_counts": dict(row_type_counts),
        "feature_presence": dict(feature_presence),
        "metric_presence": dict(metric_presence),
        "hardware_presence": dict(hardware_presence),
        "context": {
            "context_row_count": len(context_rows),
            "context_source_counts": dict(context_source_counts),
            "hardware_specs_count": hardware_specs_count,
            "benchmark_result_context_count": benchmark_result_context_count,
            "mlperf_row_count": mlperf_row_count,
        },
        "privacy": {
            "unsafe_row_count": unsafe_row_count,
            "unsafe_context_count": unsafe_context_count,
        },
        "duplicates": {
            "duplicate_row_id_count": duplicate_row_id_count,
            "duplicate_script_sha256_count": duplicate_script_sha256_count,
        },
        "usefulness": {
            "has_enough_rows": has_enough_rows,
            "has_enough_labeled_rows": has_enough_labeled_rows,
            "has_enough_label_classes": has_enough_label_classes,
            "has_context_data": has_context_data,
            "has_trial_features": has_trial_features,
            "has_comparison_labels": has_comparison_labels,
            "has_hardware_features": has_hardware_features,
        },
        "blockers": blockers,
        "recommendations": recommendations,
    }


def write_training_readiness_reports(
    report: dict[str, Any],
    manifest_dir: str = "data/gpuboost/manifests",
) -> tuple[str, str]:
    """Write machine-readable and Markdown readiness reports."""

    manifest_path = Path(manifest_dir)
    manifest_path.mkdir(parents=True, exist_ok=True)
    json_path = manifest_path / "training_readiness_report.json"
    md_path = manifest_path / "training_readiness_report.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    return str(json_path), str(md_path)


def _presence_counts(mappings: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for mapping in mappings:
        for key, value in mapping.items():
            if value is not None:
                counts[str(key)] += 1
    return counts


def _row_has_unsafe_data(row: DatasetRow) -> bool:
    if not row.is_safe_for_export():
        return True
    return bool(validate_no_raw_sensitive_fields(row.to_dict(), row.row_id))


def _context_row_has_unsafe_data(row: BenchmarkContextRow) -> bool:
    return bool(validate_no_raw_sensitive_fields(row.to_dict(), row.row_id))


def _duplicate_value_count(values: Any) -> int:
    counts = Counter(value for value in values if value)
    return sum(count - 1 for count in counts.values() if count > 1)


def _has_severe_label_imbalance(known_label_counts: Counter[str]) -> bool:
    if len(known_label_counts) < 2:
        return False
    total = sum(known_label_counts.values())
    if total == 0:
        return False
    largest = max(known_label_counts.values())
    smallest = min(known_label_counts.values())
    return (largest / total) > 0.8 or (smallest / largest) < 0.2


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Training Readiness Report",
        "",
        f"Status: {report['status']}",
        "",
        "## Row Counts",
        f"- Total rows: {report['total_rows']}",
        f"- Labeled rows: {report['labeled_rows']}",
        f"- Unlabeled rows: {report['unlabeled_rows']}",
        "",
        "## Labels",
    ]

    if report["label_counts"]:
        for key, value in sorted(report["label_counts"].items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Feature Coverage",
        ]
    )
    if report["feature_presence"]:
        for key, value in sorted(report["feature_presence"].items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Context",
            f"- Context rows: {report['context']['context_row_count']}",
            f"- Hardware specs rows: {report['context']['hardware_specs_count']}",
            f"- Benchmark result context rows: {report['context']['benchmark_result_context_count']}",
            f"- MLPerf context rows: {report['context'].get('mlperf_row_count', 0)}",
        ]
    )

    external_intake = report.get("external_intake")
    if isinstance(external_intake, dict) and external_intake:
        lines.append("")
        lines.append("## External Intake")
        for key, value in sorted(external_intake.items()):
            lines.append(f"- {key}: {json.dumps(value, sort_keys=True)}")

    lines.extend(["", "## Blockers"])
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")

    lines.extend(["", "## Recommendations"])
    if report["recommendations"]:
        for recommendation in report["recommendations"]:
            lines.append(f"- {recommendation}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "Phase 12 training should not begin until blockers are resolved.",
            "",
        ]
    )
    return "\n".join(lines)
