"""Tests for Phase 11.5 dataset export helpers."""

from __future__ import annotations

import json

from gpuboost.dataset.export import (
    build_dataset_manifest,
    export_benchmark_context_jsonl,
    export_dataset_bundle,
    export_dataset_jsonl,
    export_manifest,
    export_validation_report,
)
from gpuboost.dataset.validation import validate_dataset_rows
from gpuboost.schemas.dataset import (
    BenchmarkContextRow,
    DatasetLabel,
    DatasetPrivacyFlags,
    DatasetRow,
)


def test_build_manifest_counts_rows_labeled_unlabeled_sources_splits() -> None:
    rows = [
        _make_row(row_id="row-001", split="train"),
        _make_row(
            row_id="row-002",
            source="manual",
            split="validation",
            label=DatasetLabel(value="unknown", source="unknown"),
        ),
    ]

    manifest = build_dataset_manifest(
        rows,
        dataset_name="local",
        dataset_version="0.1.0",
        warnings=["small dataset"],
    )

    assert manifest.row_count == 2
    assert manifest.labeled_count == 1
    assert manifest.unlabeled_count == 1
    assert manifest.sources == {"gpuboost_history": 1, "manual": 1}
    assert manifest.splits == {"train": 1, "validation": 1}
    assert manifest.privacy_safe is True
    assert manifest.warnings == ["small dataset"]


def test_export_dataset_jsonl_writes_valid_jsonl(tmp_path) -> None:
    path = tmp_path / "nested" / "dataset.jsonl"
    report = export_dataset_jsonl([_make_row()], str(path))

    lines = path.read_text(encoding="utf-8").splitlines()

    assert report.status == "passed"
    assert len(lines) == 1
    assert json.loads(lines[0])["row_id"] == "row-001"


def test_export_dataset_jsonl_refuses_failed_validation(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    report = export_dataset_jsonl([_make_row(row_id="")], str(path))

    assert report.status == "failed"
    assert not path.exists()


def test_export_benchmark_context_jsonl_writes_valid_jsonl(tmp_path) -> None:
    path = tmp_path / "contexts.jsonl"
    report = export_benchmark_context_jsonl([_make_context_row()], str(path))

    lines = path.read_text(encoding="utf-8").splitlines()

    assert report.status == "passed"
    assert json.loads(lines[0])["benchmark_name"] == "MLPerf Training"


def test_export_manifest_writes_json(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = build_dataset_manifest([_make_row()], "local", "0.1.0")

    export_manifest(manifest, str(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["dataset_name"] == "local"
    assert data["row_count"] == 1


def test_export_validation_report_writes_json(tmp_path) -> None:
    path = tmp_path / "validation_report.json"
    report = validate_dataset_rows([_make_row()])

    export_validation_report(report, str(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert data["row_count"] == 1


def test_export_dataset_bundle_writes_all_files_on_valid_rows(tmp_path) -> None:
    manifest, report = export_dataset_bundle(
        [_make_row()],
        str(tmp_path),
        dataset_name="local",
        dataset_version="0.1.0",
    )

    assert manifest.row_count == 1
    assert report.status == "passed"
    assert (tmp_path / "dataset.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "validation_report.json").exists()


def test_export_dataset_bundle_skips_dataset_jsonl_on_failed_validation(tmp_path) -> None:
    manifest, report = export_dataset_bundle(
        [_make_row(privacy=DatasetPrivacyFlags(contains_raw_diff=True))],
        str(tmp_path),
        dataset_name="local",
        dataset_version="0.1.0",
    )

    assert manifest.privacy_safe is False
    assert report.status == "failed"
    assert not (tmp_path / "dataset.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "validation_report.json").exists()


def test_parent_dirs_created(tmp_path) -> None:
    path = tmp_path / "a" / "b" / "dataset.jsonl"

    export_dataset_jsonl([_make_row()], str(path))

    assert path.exists()


def test_raw_sensitive_rows_are_not_exported(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    row = _make_row(features={"raw_source": "def train(): pass"})

    report = export_dataset_jsonl([row], str(path))

    assert report.status == "failed"
    assert not path.exists()


def test_exported_jsonl_can_be_read_back(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"

    export_dataset_jsonl([_make_row(row_id="row-001"), _make_row(row_id="row-002")], str(path))
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["row_id"] for record in records] == ["row-001", "row-002"]


def _make_row(
    row_id: str = "row-001",
    source: str = "gpuboost_history",
    split: str | None = "train",
    label: DatasetLabel | None = None,
    privacy: DatasetPrivacyFlags | None = None,
    features: dict | None = None,
) -> DatasetRow:
    return DatasetRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source=source,
        row_type="optimization_outcome",
        hardware={"gpu_name": "NVIDIA A100"},
        workload={"command": "agent optimize"},
        features=features if features is not None else {"action_count": 1},
        metrics={"benchmark_median_ms": 9.5},
        label=label or DatasetLabel(value="improved", source="comparison"),
        privacy=privacy or DatasetPrivacyFlags(),
        split=split,
        quality_score=0.9,
    )


def _make_context_row() -> BenchmarkContextRow:
    return BenchmarkContextRow(
        row_id="context-001",
        created_at="2026-01-01T00:00:00+00:00",
        source="mlperf",
        benchmark_name="MLPerf Training",
        workload_name="resnet",
        hardware_name="NVIDIA A100",
        software_stack={"cuda": "12.4"},
        metrics={"samples_per_sec": 123.4},
        units={"samples_per_sec": "samples/sec"},
        url="https://example.invalid/benchmark",
    )
