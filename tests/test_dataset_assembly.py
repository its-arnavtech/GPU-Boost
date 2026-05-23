"""Tests for Phase 11.7 unified dataset assembly."""

from __future__ import annotations

import json
import os

from gpuboost.dataset.assembly import (
    assemble_training_dataset,
    load_benchmark_context_rows_jsonl,
    load_dataset_rows_jsonl,
)
from gpuboost.history.store import insert_history_run
from gpuboost.schemas.dataset import BenchmarkContextRow
from gpuboost.schemas.history import HistoryRunRecord


def test_assembles_rows_from_temp_history_db(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record(), db_path=db_path)

    summary = assemble_training_dataset(
        history_db_path=str(db_path),
        external_context_paths=[],
        output_dir=str(tmp_path / "generated"),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["dataset_row_count"] == 1
    assert summary["labeled_count"] == 1
    assert summary["label_counts"]["improved"] == 1


def test_handles_missing_history_db_with_warning(tmp_path) -> None:
    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[],
        output_dir=str(tmp_path / "generated"),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["dataset_row_count"] == 0
    assert any("History DB not found" in warning for warning in summary["warnings"])


def test_loads_external_benchmark_context_jsonl(tmp_path) -> None:
    context_path = tmp_path / "context.jsonl"
    row = _make_context_row()
    context_path.write_text(json.dumps(row.to_dict()) + "\n", encoding="utf-8")

    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[str(context_path)],
        output_dir=str(tmp_path / "generated"),
        manifest_dir=str(tmp_path / "manifests"),
    )

    loaded_rows = load_benchmark_context_rows_jsonl(
        str(tmp_path / "generated" / "benchmark_context.jsonl")
    )

    assert summary["benchmark_context_row_count"] == 1
    assert loaded_rows[0].row_id == row.row_id


def test_default_assembly_prefers_consolidated_benchmark_context(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    consolidated_rows = [_make_context_row(), _make_context_row(row_id="ctx-2", hardware_name="RTX 4080")]
    source_specific_row = _make_context_row(row_id="ctx-3", hardware_name="RTX 4070")

    (generated_dir / "benchmark_context.jsonl").write_text(
        "\n".join(json.dumps(row.to_dict()) for row in consolidated_rows) + "\n",
        encoding="utf-8",
    )
    (generated_dir / "techpowerup_gpu_specs.jsonl").write_text(
        json.dumps(source_specific_row.to_dict()) + "\n",
        encoding="utf-8",
    )
    os.utime(generated_dir / "techpowerup_gpu_specs.jsonl", (1, 1))

    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[],
        output_dir=str(generated_dir),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["benchmark_context_row_count"] == 2
    assert summary["warnings"] == [f"History DB not found: {tmp_path / 'missing.db'}"]


def test_default_assembly_merges_newer_source_specific_context_with_consolidated(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    consolidated_row = _make_context_row()
    newer_mlperf_row = _make_mlperf_context_row()

    (generated_dir / "benchmark_context.jsonl").write_text(
        json.dumps(consolidated_row.to_dict()) + "\n",
        encoding="utf-8",
    )
    (generated_dir / "mlcommons_inference_context.jsonl").write_text(
        json.dumps(newer_mlperf_row.to_dict()) + "\n",
        encoding="utf-8",
    )
    os.utime(generated_dir / "benchmark_context.jsonl", (1, 1))
    os.utime(generated_dir / "mlcommons_inference_context.jsonl", (2, 2))

    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[],
        output_dir=str(generated_dir),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["benchmark_context_row_count"] == 2


def test_fallback_loads_techpowerup_when_consolidated_absent(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    row = _make_context_row()
    (generated_dir / "techpowerup_gpu_specs.jsonl").write_text(
        json.dumps(row.to_dict()) + "\n",
        encoding="utf-8",
    )

    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[],
        output_dir=str(generated_dir),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["benchmark_context_row_count"] == 1


def test_fallback_loads_mlcommons_context_when_consolidated_absent(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    row = _make_mlperf_context_row()
    (generated_dir / "mlcommons_inference_context.jsonl").write_text(
        json.dumps(row.to_dict()) + "\n",
        encoding="utf-8",
    )

    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[],
        output_dir=str(generated_dir),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["benchmark_context_row_count"] == 1


def test_fallback_loads_techpowerup_and_mlcommons_when_consolidated_absent(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    techpowerup_row = _make_context_row()
    mlperf_row = _make_mlperf_context_row()
    (generated_dir / "techpowerup_gpu_specs.jsonl").write_text(
        json.dumps(techpowerup_row.to_dict()) + "\n",
        encoding="utf-8",
    )
    (generated_dir / "mlcommons_inference_context.jsonl").write_text(
        json.dumps(mlperf_row.to_dict()) + "\n",
        encoding="utf-8",
    )

    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[],
        output_dir=str(generated_dir),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["benchmark_context_row_count"] == 2


def test_duplicate_context_rows_are_deduped_with_summary_warning(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    row = _make_context_row()
    duplicate_path = tmp_path / "duplicate_context.jsonl"
    (generated_dir / "benchmark_context.jsonl").write_text(
        json.dumps(row.to_dict()) + "\n",
        encoding="utf-8",
    )
    duplicate_path.write_text(
        "\n".join([json.dumps(row.to_dict()), json.dumps(_make_context_row(row_id="ctx-2", hardware_name="RTX 4080").to_dict())]) + "\n",
        encoding="utf-8",
    )

    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[str(duplicate_path)],
        output_dir=str(generated_dir),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["benchmark_context_row_count"] == 2
    duplicate_warnings = [warning for warning in summary["warnings"] if "duplicate benchmark context rows" in warning]
    assert duplicate_warnings == [
        f"Skipped 1 duplicate benchmark context rows from {duplicate_path}."
    ]


def test_writes_training_dataset_and_reports(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record(), db_path=db_path)

    assemble_training_dataset(
        history_db_path=str(db_path),
        external_context_paths=[],
        output_dir=str(tmp_path / "generated"),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert (tmp_path / "generated" / "training_dataset.jsonl").exists()
    assert (tmp_path / "generated" / "training_dataset_manifest.json").exists()
    assert (tmp_path / "generated" / "training_dataset_validation_report.json").exists()
    assert (tmp_path / "manifests" / "training_readiness_report.json").exists()
    assert (tmp_path / "manifests" / "training_readiness_report.md").exists()


def test_does_not_invent_labels_or_convert_hardware_specs_to_training_labels(tmp_path) -> None:
    context_path = tmp_path / "context.jsonl"
    context_path.write_text(json.dumps(_make_context_row().to_dict()) + "\n", encoding="utf-8")

    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[str(context_path)],
        output_dir=str(tmp_path / "generated"),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["dataset_row_count"] == 0
    assert summary["benchmark_context_row_count"] == 1
    assert summary["label_counts"] == {}


def test_handles_zero_rows_gracefully(tmp_path) -> None:
    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[],
        output_dir=str(tmp_path / "generated"),
        manifest_dir=str(tmp_path / "manifests"),
    )

    assert summary["dataset_row_count"] == 0
    assert summary["readiness_status"] == "not_ready"
    assert (tmp_path / "generated" / "training_dataset.jsonl").exists()


def test_no_raw_source_diff_stdout_stderr_in_exported_rows(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(
        _make_record(
            benchmark_summary={"stdout": "loud", "median_ms": 12.0},
            code_summary={"raw_source": "secret", "finding_count": 1},
            patch_summary={"raw_diff": "--- a", "edit_count": 2},
            comparison_summary={"stderr": "oops", "overall_verdict": "improved"},
        ),
        db_path=db_path,
    )

    assemble_training_dataset(
        history_db_path=str(db_path),
        external_context_paths=[],
        output_dir=str(tmp_path / "generated"),
        manifest_dir=str(tmp_path / "manifests"),
    )

    rows = load_dataset_rows_jsonl(str(tmp_path / "generated" / "training_dataset.jsonl"))
    row = rows[0]

    for section in (
        row.hardware,
        row.workload,
        row.features,
        row.metrics,
        row.metadata,
    ):
        assert "raw_source" not in section
        assert "raw_diff" not in section
        assert "stdout" not in section
        assert "stderr" not in section


def test_split_assignment_happens_when_requested(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record(run_id="run-001"), db_path=db_path)
    insert_history_run(_make_record(run_id="run-002"), db_path=db_path)

    assemble_training_dataset(
        history_db_path=str(db_path),
        external_context_paths=[],
        output_dir=str(tmp_path / "generated"),
        manifest_dir=str(tmp_path / "manifests"),
        assign_splits=True,
        seed=7,
    )

    rows = load_dataset_rows_jsonl(str(tmp_path / "generated" / "training_dataset.jsonl"))
    assert all(row.split in {"train", "validation", "test"} for row in rows)


def _make_record(
    run_id: str = "run-001",
    benchmark_summary: dict[str, str | int | float | bool | None] | None = None,
    code_summary: dict[str, str | int | float | bool | None] | None = None,
    patch_summary: dict[str, str | int | float | bool | None] | None = None,
    comparison_summary: dict[str, str | int | float | bool | None] | None = None,
) -> HistoryRunRecord:
    return HistoryRunRecord(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        status="ok",
        command="agent optimize",
        schema_version="history.run.v1",
        goal_kind="optimize",
        goal_description="Improve throughput.",
        script_path="train.py",
        script_sha256=f"sha-{run_id}",
        gpu_name="NVIDIA RTX 4090",
        cuda_available=True,
        benchmark_summary=benchmark_summary if benchmark_summary is not None else {"median_ms": 9.5},
        advisor_summary={},
        code_summary=code_summary if code_summary is not None else {"finding_count": 1},
        patch_summary=patch_summary if patch_summary is not None else {"edit_count": 1},
        trial_summary={"status": "ok", "duration_seconds": 3.2},
        comparison_summary=comparison_summary if comparison_summary is not None else {"overall_verdict": "improved"},
        action_statuses={"benchmark": "ok", "trial": "ok"},
        warnings=[],
        metadata={"phase": 11},
    )


def _make_context_row(
    row_id: str = "techpowerup_rtx_4090",
    hardware_name: str = "NVIDIA GeForce RTX 4090",
) -> BenchmarkContextRow:
    return BenchmarkContextRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source="techpowerup",
        benchmark_name="TechPowerUp GPU Database",
        workload_name=None,
        hardware_name=hardware_name,
        software_stack={},
        metrics={},
        units={},
        url="https://example.invalid/rtx4090",
        metadata={
            "context_type": "hardware_specs",
            "source_kind": "gpu_specs",
            "gpu_name": hardware_name,
            "memory_size_mb": 24576,
        },
    )


def _make_mlperf_context_row(row_id: str = "mlperf_h100_resnet") -> BenchmarkContextRow:
    return BenchmarkContextRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source="mlperf",
        benchmark_name="MLPerf Inference",
        workload_name="resnet50",
        hardware_name="NVIDIA H100",
        software_stack={"cuda": "12.4"},
        metrics={"qps": 12345.6},
        units={"qps": "samples/s"},
        url=None,
        metadata={
            "context_type": "benchmark_result",
            "vendor": "NVIDIA",
        },
    )
