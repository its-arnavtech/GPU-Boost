"""JSONL and metadata export helpers for GPUBoost datasets."""

from __future__ import annotations

import json
from pathlib import Path

from gpuboost.dataset.validation import (
    validate_benchmark_context_rows,
    validate_dataset_rows,
)
from gpuboost.schemas.dataset import (
    BenchmarkContextRow,
    DatasetManifest,
    DatasetRow,
    DatasetValidationReport,
    create_timestamp,
)


def build_dataset_manifest(
    rows: list[DatasetRow],
    dataset_name: str,
    dataset_version: str,
    warnings: list[str] | None = None,
) -> DatasetManifest:
    """Build a manifest summarizing dataset rows."""

    sources: dict[str, int] = {}
    splits: dict[str, int] = {}
    labeled_count = 0

    for row in rows:
        sources[row.source] = sources.get(row.source, 0) + 1
        if row.split is not None:
            splits[row.split] = splits.get(row.split, 0) + 1
        if row.is_labeled():
            labeled_count += 1

    return DatasetManifest(
        generated_at=create_timestamp(),
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        row_count=len(rows),
        labeled_count=labeled_count,
        unlabeled_count=len(rows) - labeled_count,
        sources=sources,
        splits=splits,
        privacy_safe=all(row.is_safe_for_export() for row in rows),
        warnings=list(warnings or []),
    )


def export_dataset_jsonl(
    rows: list[DatasetRow],
    output_path: str,
    validate: bool = True,
) -> DatasetValidationReport:
    """Export dataset rows as UTF-8 JSONL when validation permits it."""

    report = validate_dataset_rows(rows)
    if validate and report.status == "failed":
        return report
    if report.status == "failed":
        return report

    _write_jsonl([row.to_dict() for row in rows], output_path)
    return report


def export_benchmark_context_jsonl(
    rows: list[BenchmarkContextRow],
    output_path: str,
    validate: bool = True,
) -> DatasetValidationReport:
    """Export benchmark context rows as UTF-8 JSONL when validation permits it."""

    report = validate_benchmark_context_rows(rows)
    if validate and report.status == "failed":
        return report
    if report.status == "failed":
        return report

    _write_jsonl([row.to_dict() for row in rows], output_path)
    return report


def export_manifest(
    manifest: DatasetManifest,
    output_path: str,
) -> None:
    """Write a dataset manifest as UTF-8 JSON."""

    _write_json(manifest.to_dict(), output_path)


def export_validation_report(
    report: DatasetValidationReport,
    output_path: str,
) -> None:
    """Write a validation report as UTF-8 JSON."""

    _write_json(report.to_dict(), output_path)


def export_dataset_bundle(
    rows: list[DatasetRow],
    output_dir: str,
    dataset_name: str,
    dataset_version: str,
) -> tuple[DatasetManifest, DatasetValidationReport]:
    """Export dataset JSONL, manifest, and validation report into a directory."""

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    report = validate_dataset_rows(rows)
    manifest = build_dataset_manifest(
        rows=rows,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

    if report.status != "failed":
        _write_jsonl(
            [row.to_dict() for row in rows],
            str(output_directory / "dataset.jsonl"),
        )
    export_manifest(manifest, str(output_directory / "manifest.json"))
    export_validation_report(report, str(output_directory / "validation_report.json"))
    return manifest, report


def _write_jsonl(records: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")


def _write_json(data: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
