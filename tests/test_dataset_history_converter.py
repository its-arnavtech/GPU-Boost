"""Tests for Phase 11.2 history-to-dataset conversion."""

from __future__ import annotations

from gpuboost.dataset.history_converter import (
    derive_label_from_history_record,
    estimate_history_row_quality,
    history_record_to_dataset_row,
    history_records_to_dataset_rows,
)
from gpuboost.schemas.dataset import DatasetRow
from gpuboost.schemas.history import HistoryRunRecord


def test_converts_basic_record_to_dataset_row() -> None:
    row = history_record_to_dataset_row(_make_record())

    assert isinstance(row, DatasetRow)
    assert row.created_at
    assert row.label.value == "improved"
    assert row.quality_score == 1.0


def test_row_source_and_type_are_correct() -> None:
    row = history_record_to_dataset_row(_make_record())

    assert row.source == "gpuboost_history"
    assert row.row_type == "optimization_outcome"


def test_row_id_derived_from_run_id() -> None:
    row = history_record_to_dataset_row(_make_record(run_id="run-abc"))

    assert row.row_id == "row_run-abc"


def test_provided_row_id_is_used() -> None:
    row = history_record_to_dataset_row(_make_record(), row_id="custom-row")

    assert row.row_id == "custom-row"


def test_hardware_fields_extracted() -> None:
    row = history_record_to_dataset_row(
        _make_record(gpu_name="NVIDIA RTX 4090", cuda_available=True)
    )

    assert row.hardware["gpu_name"] == "NVIDIA RTX 4090"
    assert row.hardware["cuda_available"] is True


def test_workload_does_not_include_raw_script_path() -> None:
    row = history_record_to_dataset_row(_make_record(script_path="C:/secret/train.py"))

    assert row.workload["has_script_path"] is True
    assert "script_path" not in row.workload


def test_script_sha256_included() -> None:
    row = history_record_to_dataset_row(_make_record(script_sha256="abc123"))

    assert row.workload["script_sha256"] == "abc123"


def test_features_include_action_counts_and_summary_presence() -> None:
    row = history_record_to_dataset_row(
        _make_record(
            action_statuses={
                "benchmark": "ok",
                "patch": "completed",
                "trial": "error",
            },
            patch_summary={"edit_count": 2},
            trial_summary={"status": "ok", "duration_seconds": 3.5},
            comparison_summary={"overall_verdict": "improved"},
            code_summary={"finding_count": 4},
        )
    )

    assert row.features["action_count"] == 3
    assert row.features["completed_action_count"] == 2
    assert row.features["failed_action_count"] == 1
    assert row.features["has_diff"] is True
    assert row.features["has_trial"] is True
    assert row.features["has_comparison"] is True
    assert row.features["code_finding_count"] == 4
    assert row.features["patch_edit_count"] == 2
    assert row.features["trial_duration_seconds"] == 3.5


def test_label_improved_from_comparison() -> None:
    label = derive_label_from_history_record(
        _make_record(comparison_summary={"overall_verdict": "improved"})
    )

    assert label.value == "improved"
    assert label.source == "comparison"
    assert label.confidence == 0.9


def test_label_regressed_from_comparison() -> None:
    label = derive_label_from_history_record(
        _make_record(comparison_summary={"overall_verdict": "regressed"})
    )

    assert label.value == "regressed"
    assert label.source == "comparison"
    assert label.confidence == 0.9


def test_label_neutral_from_unchanged_and_mixed_comparison() -> None:
    unchanged = derive_label_from_history_record(
        _make_record(comparison_summary={"overall_verdict": "unchanged"})
    )
    mixed = derive_label_from_history_record(
        _make_record(comparison_summary={"overall_verdict": "mixed"})
    )

    assert unchanged.value == "neutral"
    assert unchanged.confidence == 0.8
    assert mixed.value == "neutral"
    assert mixed.confidence == 0.5


def test_label_failed_from_trial_failure() -> None:
    label = derive_label_from_history_record(
        _make_record(comparison_summary={}, trial_summary={"status": "failed"})
    )

    assert label.value == "failed"
    assert label.source == "trial"
    assert label.confidence == 0.8


def test_label_unknown_when_no_outcome() -> None:
    label = derive_label_from_history_record(
        _make_record(comparison_summary={}, trial_summary={})
    )

    assert label.value == "unknown"
    assert label.source == "unknown"
    assert label.confidence is None


def test_quality_score_increases_with_useful_data() -> None:
    minimal = _make_record(
        script_sha256=None,
        benchmark_summary={},
        trial_summary={},
        comparison_summary={},
    )
    rich = _make_record()

    assert estimate_history_row_quality(minimal) == 0.5
    assert estimate_history_row_quality(rich) > estimate_history_row_quality(minimal)


def test_quality_score_decreases_on_error() -> None:
    ok_record = _make_record(status="ok", comparison_summary={})
    error_record = _make_record(status="error", comparison_summary={})

    assert estimate_history_row_quality(error_record) == (
        estimate_history_row_quality(ok_record) - 0.2
    )


def test_records_to_dataset_rows_preserves_order() -> None:
    rows = history_records_to_dataset_rows(
        [_make_record(run_id="first"), _make_record(run_id="second")],
        split="train",
    )

    assert [row.row_id for row in rows] == ["row_first", "row_second"]
    assert [row.split for row in rows] == ["train", "train"]


def test_privacy_flags_are_safe() -> None:
    row = history_record_to_dataset_row(_make_record())

    assert row.privacy.contains_raw_source is False
    assert row.privacy.contains_raw_diff is False
    assert row.privacy.contains_stdout is False
    assert row.privacy.contains_stderr is False
    assert row.privacy.is_safe_for_export() is True


def test_raw_payload_fields_are_not_present_in_dataset_payload_sections() -> None:
    row = history_record_to_dataset_row(
        _make_record(
            code_summary={"raw_source": "secret", "finding_count": 1},
            patch_summary={"raw_diff": "--- secret", "edit_count": 2},
            benchmark_summary={"stdout": "noisy", "median_ms": 1.2},
            comparison_summary={"stderr": "noisy", "overall_verdict": "improved"},
        )
    )

    for section in (row.hardware, row.workload, row.features, row.metrics, row.metadata):
        assert "raw_source" not in section
        assert "raw_diff" not in section
        assert "stdout" not in section
        assert "stderr" not in section

    assert row.features["code_finding_count"] == 1
    assert row.features["patch_edit_count"] == 2
    assert row.metrics["benchmark_median_ms"] == 1.2


def _make_record(
    run_id: str = "run-001",
    status: str = "ok",
    script_path: str | None = "train.py",
    script_sha256: str | None = "abc123",
    gpu_name: str | None = "NVIDIA A100",
    cuda_available: bool | None = True,
    benchmark_summary: dict[str, str | int | float | bool | None] | None = None,
    code_summary: dict[str, str | int | float | bool | None] | None = None,
    patch_summary: dict[str, str | int | float | bool | None] | None = None,
    trial_summary: dict[str, str | int | float | bool | None] | None = None,
    comparison_summary: dict[str, str | int | float | bool | None] | None = None,
    action_statuses: dict[str, str] | None = None,
) -> HistoryRunRecord:
    return HistoryRunRecord(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        status=status,
        command="agent optimize",
        schema_version="history.run.v1",
        goal_kind="optimize",
        goal_description="Improve training throughput.",
        script_path=script_path,
        script_sha256=script_sha256,
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        benchmark_summary=benchmark_summary
        if benchmark_summary is not None
        else {"median_ms": 9.5},
        code_summary=code_summary if code_summary is not None else {"findings": 1},
        patch_summary=patch_summary if patch_summary is not None else {"edits": 1},
        trial_summary=trial_summary if trial_summary is not None else {"status": "ok"},
        comparison_summary=comparison_summary
        if comparison_summary is not None
        else {"overall_verdict": "improved"},
        action_statuses=action_statuses
        if action_statuses is not None
        else {"benchmark": "ok", "trial": "ok"},
        warnings=["Synthetic record."],
        error="failed" if status == "error" else None,
        metadata={"phase": 11},
    )
