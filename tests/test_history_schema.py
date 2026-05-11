"""Tests for Phase 9.1 history schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from gpuboost.schemas.history import (
    HistoryCompareResult,
    HistoryRunRecord,
    HistorySummary,
    create_timestamp,
)


def test_history_run_record_creation() -> None:
    record = _make_record()

    assert record.run_id == "run-001"
    assert record.created_at == "2026-01-01T00:00:00+00:00"
    assert record.status == "ok"
    assert record.command == "agent optimize"
    assert record.schema_version == "history.run.v1"
    assert record.goal_kind == "optimize"
    assert record.goal_description == "Improve training throughput."
    assert record.script_path == "train.py"
    assert record.script_sha256 == "abc123"
    assert record.gpu_name == "NVIDIA RTX"
    assert record.cuda_available is True


def test_history_summary_creation() -> None:
    record = _make_record()
    summary = HistorySummary(
        generated_at="2026-01-01T00:00:01+00:00",
        total_runs=1,
        runs=[record],
        warnings=["Synthetic history."],
    )

    assert summary.generated_at == "2026-01-01T00:00:01+00:00"
    assert summary.total_runs == 1
    assert summary.runs == [record]
    assert summary.warnings == ["Synthetic history."]


def test_history_compare_result_creation() -> None:
    result = HistoryCompareResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="ok",
        left_run_id="run-001",
        right_run_id="run-002",
        summary="Throughput changed.",
        changed_fields={"tokens_per_second": 12.5},
        warnings=["Comparison is schema-only."],
        error=None,
    )

    assert result.generated_at == "2026-01-01T00:00:02+00:00"
    assert result.status == "ok"
    assert result.left_run_id == "run-001"
    assert result.right_run_id == "run-002"
    assert result.summary == "Throughput changed."
    assert result.changed_fields == {"tokens_per_second": 12.5}
    assert result.warnings == ["Comparison is schema-only."]
    assert result.error is None


def test_to_dict_nesting_works() -> None:
    summary = HistorySummary(
        generated_at="2026-01-01T00:00:01+00:00",
        total_runs=1,
        runs=[_make_record()],
    )

    data = summary.to_dict()

    assert data["runs"][0]["run_id"] == "run-001"
    assert data["runs"][0]["benchmark_summary"]["median_ms"] == 9.5
    assert data["runs"][0]["action_statuses"]["benchmark"] == "ok"
    assert data["runs"][0]["warnings"] == []


def test_json_serialization_works() -> None:
    result = HistoryCompareResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="ok",
        left_run_id="run-001",
        right_run_id="run-002",
        summary="No change.",
        changed_fields={"status_changed": False},
    )

    serialized = json.dumps(result.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["left_run_id"] == "run-001"
    assert deserialized["changed_fields"]["status_changed"] is False
    assert deserialized["warnings"] == []
    assert deserialized["error"] is None


def test_default_dict_and_list_fields_are_isolated_between_instances() -> None:
    first_record = _make_minimal_record("run-001")
    second_record = _make_minimal_record("run-002")
    first_summary = HistorySummary(
        generated_at="2026-01-01T00:00:01+00:00",
        total_runs=0,
    )
    second_summary = HistorySummary(
        generated_at="2026-01-01T00:00:02+00:00",
        total_runs=0,
    )
    first_compare = HistoryCompareResult(
        generated_at="2026-01-01T00:00:03+00:00",
        status="ok",
        left_run_id="run-001",
        right_run_id="run-002",
        summary="Compared.",
    )
    second_compare = HistoryCompareResult(
        generated_at="2026-01-01T00:00:04+00:00",
        status="ok",
        left_run_id="run-003",
        right_run_id="run-004",
        summary="Compared.",
    )

    first_record.benchmark_summary["median_ms"] = 9.5
    first_record.warnings.append("First warning.")
    first_summary.runs.append(first_record)
    first_summary.warnings.append("Summary warning.")
    first_compare.changed_fields["status"] = "ok"
    first_compare.warnings.append("Compare warning.")

    assert second_record.benchmark_summary == {}
    assert second_record.warnings == []
    assert second_summary.runs == []
    assert second_summary.warnings == []
    assert second_compare.changed_fields == {}
    assert second_compare.warnings == []


def test_create_timestamp_returns_non_empty_utc_iso_string() -> None:
    timestamp = create_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert timestamp
    assert parsed.tzinfo == timezone.utc


def test_has_error_returns_true_for_error_status() -> None:
    record = _make_minimal_record("run-001", status="error")

    assert record.has_error() is True


def test_has_error_returns_true_for_error_message() -> None:
    record = _make_minimal_record("run-001", error="Benchmark failed.")

    assert record.has_error() is True


def test_has_error_returns_false_without_error() -> None:
    record = _make_minimal_record("run-001")

    assert record.has_error() is False


def test_has_trial_returns_true_when_trial_summary_present() -> None:
    record = _make_minimal_record("run-001")
    record.trial_summary["status"] = "ok"

    assert record.has_trial() is True


def test_has_trial_returns_false_when_trial_summary_empty() -> None:
    record = _make_minimal_record("run-001")

    assert record.has_trial() is False


def test_has_comparison_returns_true_when_comparison_summary_present() -> None:
    record = _make_minimal_record("run-001")
    record.comparison_summary["overall_verdict"] = "improved"

    assert record.has_comparison() is True


def test_has_comparison_returns_false_when_comparison_summary_empty() -> None:
    record = _make_minimal_record("run-001")

    assert record.has_comparison() is False


def test_latest_returns_first_run() -> None:
    first = _make_minimal_record("run-newest")
    second = _make_minimal_record("run-older")
    summary = HistorySummary(
        generated_at="2026-01-01T00:00:01+00:00",
        total_runs=2,
        runs=[first, second],
    )

    assert summary.latest() == first


def test_latest_returns_none_when_empty() -> None:
    summary = HistorySummary(
        generated_at="2026-01-01T00:00:01+00:00",
        total_runs=0,
    )

    assert summary.latest() is None


def test_record_can_represent_script_path_none() -> None:
    record = _make_minimal_record("run-001")

    assert record.script_path is None


def test_record_stores_script_hash_without_raw_source() -> None:
    record = _make_minimal_record("run-001")
    record.script_sha256 = "a" * 64
    data = record.to_dict()

    assert data["script_sha256"] == "a" * 64
    assert "source_code" not in data
    assert "raw_source" not in data
    assert "code" not in data


def test_error_record_can_carry_error_message() -> None:
    record = _make_minimal_record(
        "run-001",
        status="error",
        error="Optimization failed.",
    )

    assert record.status == "error"
    assert record.error == "Optimization failed."
    assert record.to_dict()["error"] == "Optimization failed."


def _make_record() -> HistoryRunRecord:
    return HistoryRunRecord(
        run_id="run-001",
        created_at="2026-01-01T00:00:00+00:00",
        status="ok",
        command="agent optimize",
        schema_version="history.run.v1",
        goal_kind="optimize",
        goal_description="Improve training throughput.",
        script_path="train.py",
        script_sha256="abc123",
        gpu_name="NVIDIA RTX",
        cuda_available=True,
        benchmark_summary={"median_ms": 9.5, "ok": True},
        advisor_summary={"recommendations": 2},
        code_summary={"findings": 1},
        patch_summary={"edits": 1},
        trial_summary={"status": "ok"},
        comparison_summary={"overall_verdict": "improved"},
        action_statuses={"benchmark": "ok", "advisor": "ok"},
        metadata={"phase": 9},
    )


def _make_minimal_record(
    run_id: str,
    status: str = "ok",
    error: str | None = None,
) -> HistoryRunRecord:
    return HistoryRunRecord(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        status=status,
        command="agent optimize",
        schema_version="history.run.v1",
        goal_kind="optimize",
        goal_description="Improve training throughput.",
        error=error,
    )
