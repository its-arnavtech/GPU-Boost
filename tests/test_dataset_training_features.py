"""Tests for safe Phase 12 dataset training feature extraction."""

from __future__ import annotations

from gpuboost.dataset.training_features import (
    audit_training_feature_leakage,
    build_training_matrix,
    extract_training_features_from_row,
    extract_training_label_from_row,
    is_safe_training_feature_name,
    is_target_derived_feature_name,
)
from gpuboost.schemas.dataset import DatasetLabel, DatasetPrivacyFlags, DatasetRow


def test_target_derived_names_are_detected() -> None:
    for name in (
        "overall_verdict",
        "features.overall_verdict",
        "improved_metric_count",
        "before_best_images_per_sec",
        "after_best_images_per_sec",
        "delta_best_images_per_sec",
        "percent_delta_best_images_per_sec",
        "comparison_status",
        "target",
        "label",
    ):
        assert is_target_derived_feature_name(name)
        assert not is_safe_training_feature_name(name)


def test_unsafe_raw_identifier_and_secret_fields_are_excluded() -> None:
    row = _make_row(
        features={
            "raw_source": "secret source",
            "source_code": "print('secret')",
            "raw_diff": "--- a\n+++ b",
            "unified_diff": "--- a\n+++ b",
            "stdout": "noisy",
            "stderr": "noisy",
            "file_contents": "contents",
            "api_key": "key",
            "token": "token",
            "credential": "credential",
            "password": "password",
            "secret": "secret",
            "safe_count": 3,
        },
        workload={
            "script_path": "train.py",
            "absolute_path": "C:/workspace/train.py",
            "local_path": "data/gpuboost/raw/file.json",
        },
    )

    features = extract_training_features_from_row(row)

    assert features == {"features.safe_count": 3}


def test_label_row_id_split_created_at_and_privacy_are_excluded() -> None:
    row = _make_row(split="test")

    features = extract_training_features_from_row(row)

    assert "row_id" not in features
    assert "created_at" not in features
    assert "split" not in features
    assert "label" not in features
    assert "privacy" not in features


def test_safe_hardware_workload_features_metrics_and_metadata_are_kept() -> None:
    row = _make_row(
        hardware={"gpu_name": "NVIDIA Test GPU", "cuda_available": True},
        workload={"workload_name": "amp_controlled", "batch_size": 32},
        features={"workload_family": "amp", "controlled_grid": True},
        metrics={"fp32_samples_per_sec": 100.0},
        metadata={"source": "controlled", "phase": "11.10"},
    )

    features = extract_training_features_from_row(row)

    assert features == {
        "features.controlled_grid": True,
        "features.workload_family": "amp",
        "hardware.cuda_available": True,
        "hardware.gpu_name": "NVIDIA Test GPU",
        "metadata.phase": "11.10",
        "metadata.source": "controlled",
        "metrics.fp32_samples_per_sec": 100.0,
        "workload.batch_size": 32,
        "workload.workload_name": "amp_controlled",
    }


def test_unknown_labels_are_skipped() -> None:
    row = _make_row(label=DatasetLabel(value="unknown", source="unknown"))

    assert extract_training_label_from_row(row) is None
    features, labels, row_ids = build_training_matrix([row])
    assert features == []
    assert labels == []
    assert row_ids == []


def test_training_matrix_aligns_features_labels_and_row_ids() -> None:
    rows = [
        _make_row("row-1", features={"workload_family": "amp"}),
        _make_row(
            "row-2",
            label=DatasetLabel(value="regressed", source="comparison"),
            features={"workload_family": "batch"},
        ),
        _make_row("row-3", label=DatasetLabel(value="unknown", source="unknown")),
    ]

    features, labels, row_ids = build_training_matrix(rows)

    assert features == [
        {"features.workload_family": "amp"},
        {"features.workload_family": "batch"},
    ]
    assert labels == ["improved", "regressed"]
    assert row_ids == ["row-1", "row-2"]


def test_leakage_audit_passes_after_safe_extraction_and_reports_exclusions() -> None:
    row = _make_row(
        features={
            "workload_family": "amp",
            "overall_verdict": "improved",
            "improved_metric_count": 1,
        },
        metrics={
            "before_best_images_per_sec": 10.0,
            "after_best_images_per_sec": 12.0,
            "fp32_samples_per_sec": 100.0,
        },
        metadata={"comparison_verdict": "improved", "source": "controlled"},
    )

    audit = audit_training_feature_leakage([row])
    features = extract_training_features_from_row(row)

    assert audit["status"] == "passed"
    assert audit["leaked_feature_count"] == 0
    assert "features.overall_verdict" in audit["excluded_fields"]["row-1"]
    assert "features.improved_metric_count" in audit["excluded_fields"]["row-1"]
    assert "metrics.before_best_images_per_sec" in audit["excluded_fields"]["row-1"]
    assert "metrics.after_best_images_per_sec" in audit["excluded_fields"]["row-1"]
    assert "metadata.comparison_verdict" in audit["excluded_fields"]["row-1"]
    assert features == {
        "features.workload_family": "amp",
        "metadata.source": "controlled",
        "metrics.fp32_samples_per_sec": 100.0,
    }


def _make_row(
    row_id: str = "row-1",
    *,
    label: DatasetLabel | None = None,
    split: str | None = None,
    hardware: dict | None = None,
    workload: dict | None = None,
    features: dict | None = None,
    metrics: dict | None = None,
    metadata: dict | None = None,
) -> DatasetRow:
    return DatasetRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source="controlled_experiment",
        row_type="controlled_experiment",
        hardware=hardware or {},
        workload=workload or {},
        features=features or {},
        metrics=metrics or {},
        label=label or DatasetLabel(value="improved", source="comparison"),
        privacy=DatasetPrivacyFlags(),
        split=split,
        quality_score=0.9,
        metadata=metadata or {},
    )
