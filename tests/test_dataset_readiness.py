"""Tests for Phase 11.7 training readiness analysis."""

from __future__ import annotations

import json

from gpuboost.dataset.readiness import (
    analyze_training_readiness,
    write_training_readiness_reports,
)
from gpuboost.schemas.dataset import (
    BenchmarkContextRow,
    DatasetLabel,
    DatasetPrivacyFlags,
    DatasetRow,
)


def test_not_ready_with_zero_rows() -> None:
    report = analyze_training_readiness([])

    assert report["status"] == "not_ready"
    assert "Not enough total rows for training." in report["blockers"]


def test_not_ready_with_unlabeled_rows_only() -> None:
    rows = [_make_row(label=DatasetLabel(value="unknown", source="unknown")) for _ in range(3)]

    report = analyze_training_readiness(rows, min_total_rows=1, min_labeled_rows=1, min_known_label_classes=2)

    assert report["status"] == "not_ready"
    assert report["labeled_rows"] == 0
    assert "No labeled optimization outcome rows found." in report["blockers"]


def test_not_ready_with_one_label_class_only() -> None:
    rows = [_make_row(row_id=f"row-{index}", label=DatasetLabel(value="improved", source="comparison")) for index in range(5)]

    report = analyze_training_readiness(rows, min_total_rows=1, min_labeled_rows=1, min_known_label_classes=2)

    assert report["status"] == "not_ready"
    assert report["known_label_counts"] == {"improved": 5}
    assert "Only one known label class is present." in report["blockers"]


def test_warning_when_thresholds_met_but_imbalanced() -> None:
    rows = [
        *[_make_row(row_id=f"improved-{index}", label=DatasetLabel(value="improved", source="comparison")) for index in range(9)],
        _make_row(row_id="neutral-1", label=DatasetLabel(value="neutral", source="comparison")),
    ]

    report = analyze_training_readiness(rows, min_total_rows=5, min_labeled_rows=5, min_known_label_classes=2)

    assert report["status"] == "warning"
    assert report["known_label_counts"]["improved"] == 9
    assert report["known_label_counts"]["neutral"] == 1


def test_label_and_presence_counts_are_correct() -> None:
    rows = [
        _make_row(row_id="row-1", features={"action_count": 1, "trial_duration_seconds": 3.0}, metrics={"benchmark_median_ms": 9.5}, hardware={"gpu_name": "RTX 4090"}),
        _make_row(row_id="row-2", label=DatasetLabel(value="neutral", source="comparison"), features={"action_count": 1}, metrics={"comparison_delta_pct": 5.0}, hardware={"gpu_name": "RTX 4080", "cuda_available": True}),
    ]

    report = analyze_training_readiness(rows, min_total_rows=1, min_labeled_rows=1, min_known_label_classes=2)

    assert report["label_counts"]["improved"] == 1
    assert report["label_counts"]["neutral"] == 1
    assert report["feature_presence"]["action_count"] == 2
    assert report["metric_presence"]["benchmark_median_ms"] == 1
    assert report["hardware_presence"]["gpu_name"] == 2


def test_context_counts_detect_hardware_specs_rows() -> None:
    report = analyze_training_readiness(
        [_make_row()],
        context_rows=[
            _make_context_row(),
            _make_benchmark_result_context_row(),
        ],
        min_total_rows=1,
        min_labeled_rows=1,
        min_known_label_classes=1,
    )

    assert report["context"]["context_row_count"] == 2
    assert report["context"]["hardware_specs_count"] == 1
    assert report["context"]["benchmark_result_context_count"] == 1
    assert report["context"]["mlperf_row_count"] == 1
    assert report["context"]["context_source_counts"]["mlperf"] == 1


def test_duplicate_detection_works() -> None:
    rows = [
        _make_row(row_id="dup", script_sha256="same"),
        _make_row(row_id="dup", script_sha256="same", label=DatasetLabel(value="neutral", source="comparison")),
    ]

    report = analyze_training_readiness(rows, min_total_rows=1, min_labeled_rows=1, min_known_label_classes=2)

    assert report["duplicates"]["duplicate_row_id_count"] == 1
    assert report["duplicates"]["duplicate_script_sha256_count"] == 1


def test_unsafe_row_count_detected() -> None:
    rows = [
        _make_row(),
        _make_row(
            row_id="unsafe",
            privacy=DatasetPrivacyFlags(contains_raw_source=True),
        ),
    ]

    report = analyze_training_readiness(rows, min_total_rows=1, min_labeled_rows=1, min_known_label_classes=1)

    assert report["privacy"]["unsafe_row_count"] == 1


def test_recommendations_include_comparison_guidance_when_labels_missing() -> None:
    rows = [_make_row(label=DatasetLabel(value="failed", source="trial"))]

    report = analyze_training_readiness(rows, min_total_rows=1, min_labeled_rows=1, min_known_label_classes=1)

    assert any("before/after comparisons" in item for item in report["recommendations"])


def test_readiness_reports_are_written(tmp_path) -> None:
    report = analyze_training_readiness([], min_total_rows=1, min_labeled_rows=1, min_known_label_classes=1)

    json_path, md_path = write_training_readiness_reports(report, manifest_dir=str(tmp_path))

    json_payload = json.loads((tmp_path / "training_readiness_report.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "training_readiness_report.md").read_text(encoding="utf-8")

    assert json_path.endswith("training_readiness_report.json")
    assert md_path.endswith("training_readiness_report.md")
    assert json_payload["status"] == report["status"]
    assert "Phase 12 training should not begin until blockers are resolved." in markdown


def test_readiness_markdown_warning_message(tmp_path) -> None:
    rows = [
        *[
            _make_row(
                row_id=f"improved-{index}",
                label=DatasetLabel(value="improved", source="comparison"),
            )
            for index in range(9)
        ],
        _make_row(
            row_id="neutral-1",
            label=DatasetLabel(value="neutral", source="comparison"),
        ),
    ]
    report = analyze_training_readiness(
        rows,
        min_total_rows=5,
        min_labeled_rows=5,
        min_known_label_classes=2,
    )

    _json_path, md_path = write_training_readiness_reports(
        report,
        manifest_dir=str(tmp_path),
    )
    markdown = (tmp_path / "training_readiness_report.md").read_text(encoding="utf-8")

    assert report["status"] == "warning"
    assert not report["blockers"]
    assert md_path.endswith("training_readiness_report.md")
    assert (
        "Phase 12 training may proceed cautiously, but warnings should be reviewed first."
        in markdown
    )


def test_readiness_markdown_ready_message(tmp_path) -> None:
    rows = [
        _make_row(
            row_id="improved-1",
            label=DatasetLabel(value="improved", source="comparison"),
        ),
        _make_row(
            row_id="neutral-1",
            label=DatasetLabel(value="neutral", source="comparison"),
        ),
    ]
    report = analyze_training_readiness(
        rows,
        min_total_rows=2,
        min_labeled_rows=2,
        min_known_label_classes=2,
    )

    write_training_readiness_reports(report, manifest_dir=str(tmp_path))
    markdown = (tmp_path / "training_readiness_report.md").read_text(encoding="utf-8")

    assert report["status"] == "ready"
    assert not report["blockers"]
    assert "Phase 12 training may begin because no hard blockers remain." in markdown
    assert (
        "Phase 12 training should not begin until blockers are resolved."
        not in markdown
    )


def _make_row(
    row_id: str = "row-001",
    label: DatasetLabel | None = None,
    features: dict | None = None,
    metrics: dict | None = None,
    hardware: dict | None = None,
    privacy: DatasetPrivacyFlags | None = None,
    script_sha256: str = "sha-001",
) -> DatasetRow:
    return DatasetRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source="gpuboost_history",
        row_type="optimization_outcome",
        hardware=hardware if hardware is not None else {"gpu_name": "NVIDIA RTX 4090"},
        workload={"command": "agent optimize", "script_sha256": script_sha256},
        features=features if features is not None else {"action_count": 1, "has_trial": True},
        metrics=metrics if metrics is not None else {"benchmark_median_ms": 9.5},
        label=label or DatasetLabel(value="improved", source="comparison"),
        privacy=privacy or DatasetPrivacyFlags(),
        split="train",
        quality_score=0.9,
    )


def _make_context_row() -> BenchmarkContextRow:
    return BenchmarkContextRow(
        row_id="ctx-hw",
        created_at="2026-01-01T00:00:00+00:00",
        source="techpowerup",
        benchmark_name="TechPowerUp GPU Database",
        workload_name=None,
        hardware_name="NVIDIA RTX 4090",
        software_stack={},
        metrics={},
        units={},
        url="https://example.invalid/hw",
        metadata={
            "context_type": "hardware_specs",
            "source_kind": "gpu_specs",
            "gpu_name": "NVIDIA RTX 4090",
        },
    )


def _make_benchmark_result_context_row() -> BenchmarkContextRow:
    return BenchmarkContextRow(
        row_id="ctx-bench",
        created_at="2026-01-01T00:00:00+00:00",
        source="mlperf",
        benchmark_name="MLPerf Training",
        workload_name="resnet",
        hardware_name="NVIDIA H100",
        software_stack={"cuda": "12.4"},
        metrics={"samples_per_sec": 1000.0},
        units={"samples_per_sec": "samples/sec"},
        url="https://example.invalid/mlperf",
    )
