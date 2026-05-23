"""Tests for the MLCommons local intake helpers."""

from __future__ import annotations

import json

from gpuboost.dataset.assembly import assemble_training_dataset
from gpuboost.dataset.mlcommons_importer import (
    extract_mlcommons_context_rows,
    inspect_mlcommons_source,
    run_mlcommons_intake,
)
from gpuboost.dataset.readiness import analyze_training_readiness


def test_inspect_mlcommons_source_detects_requested_vendor_folders(tmp_path) -> None:
    root = tmp_path / "mlcommons"
    _write_system_json(root, "NVIDIA", "sys-a")
    _write_system_json(root, "AMD", "sys-b")

    summary = inspect_mlcommons_source(str(root))

    assert summary["vendors"]["NVIDIA"]["exists"] is True
    assert summary["vendors"]["AMD"]["exists"] is True
    assert summary["vendors"]["Google"]["exists"] is False
    assert summary["vendors"]["CoreWeave"]["exists"] is False


def test_inspect_reports_missing_folder_warning(tmp_path) -> None:
    summary = inspect_mlcommons_source(str(tmp_path / "missing"))

    assert any("Missing vendor folder" in warning for warning in summary["vendors"]["NVIDIA"]["warnings"])


def test_extractor_parses_system_desc_json_into_context_row(tmp_path) -> None:
    root = tmp_path / "mlcommons"
    _write_system_json(root, "NVIDIA", "sys-a")

    rows, warnings = extract_mlcommons_context_rows(str(root))

    assert warnings == [
        f"Missing vendor folder: {root / 'closed' / 'AMD'}",
        f"Missing vendor folder: {root / 'closed' / 'Google'}",
        f"Missing vendor folder: {root / 'closed' / 'CoreWeave'}",
    ]
    system_rows = [row for row in rows if row.metadata.get("source_kind") == "system_description"]
    assert len(system_rows) == 1
    assert system_rows[0].source == "mlperf"
    assert system_rows[0].hardware_name == "NVIDIA H100 SXM"
    assert system_rows[0].metadata["context_type"] == "hardware_specs"


def test_extractor_parses_result_json_with_qps_and_latency(tmp_path) -> None:
    root = tmp_path / "mlcommons"
    _write_system_json(root, "NVIDIA", "sys-a")
    result_path = root / "closed" / "NVIDIA" / "results" / "sys-a" / "resnet50" / "Offline" / "summary.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "SystemName": "sys-a",
                "Model": "resnet50",
                "Scenario": "Offline",
                "Result": 12345.6,
                "Latency": 7.8,
                "stdout": "do not keep",
            }
        ),
        encoding="utf-8",
    )

    rows, _ = extract_mlcommons_context_rows(str(root))

    result_rows = [row for row in rows if row.metadata.get("context_type") == "benchmark_result"]
    assert len(result_rows) == 1
    assert result_rows[0].metrics["qps"] == 12345.6
    assert result_rows[0].metrics["latency_ms"] == 7.8
    assert "stdout" not in json.dumps(result_rows[0].to_dict())


def test_extractor_parses_csv_result(tmp_path) -> None:
    root = tmp_path / "mlcommons"
    _write_system_json(root, "NVIDIA", "sys-a")
    csv_path = root / "closed" / "NVIDIA" / "results" / "out.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "SystemName,Model,Scenario,Result,Units,Accuracy\n"
        "sys-a,bert-99,Server,4321.0,samples/s,99.1\n",
        encoding="utf-8",
    )

    rows, _ = extract_mlcommons_context_rows(str(root))

    result_rows = [row for row in rows if row.metadata.get("context_type") == "benchmark_result"]
    assert len(result_rows) == 1
    assert result_rows[0].workload_name == "bert-99"
    assert result_rows[0].metrics["qps"] == 4321.0
    assert result_rows[0].metrics["accuracy"] == 99.1


def test_extractor_skips_invalid_json_safely(tmp_path) -> None:
    root = tmp_path / "mlcommons"
    invalid_path = root / "closed" / "NVIDIA" / "results" / "sys-a" / "resnet50" / "Offline" / "summary.json"
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_text("{broken", encoding="utf-8")

    rows, warnings = extract_mlcommons_context_rows(str(root))

    assert rows == []
    assert any("Skipped invalid MLCommons JSON files" in warning for warning in warnings)


def test_extractor_excludes_raw_source_stdout_stderr_and_diff_fields(tmp_path) -> None:
    root = tmp_path / "mlcommons"
    _write_system_json(root, "NVIDIA", "sys-a")
    csv_path = root / "closed" / "NVIDIA" / "results" / "out.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "SystemName,Model,Scenario,Result,source_code,stdout,stderr,raw_diff\n"
        "sys-a,resnet50,Offline,1000.0,def train(): pass,hello,oops,--- a\n",
        encoding="utf-8",
    )

    rows, _ = extract_mlcommons_context_rows(str(root))

    payload = json.dumps([row.to_dict() for row in rows])
    assert "source_code" not in payload
    assert "stdout" not in payload
    assert "stderr" not in payload
    assert "raw_diff" not in payload


def test_run_mlcommons_intake_writes_jsonl_and_reports(tmp_path) -> None:
    root = tmp_path / "mlcommons"
    output_dir = tmp_path / "generated"
    manifest_dir = tmp_path / "manifests"
    _write_system_json(root, "NVIDIA", "sys-a")
    csv_path = root / "closed" / "NVIDIA" / "results" / "out.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "SystemName,Model,Scenario,Result,Units\n"
        "sys-a,resnet50,Offline,1000.0,samples/s\n",
        encoding="utf-8",
    )

    report = run_mlcommons_intake(
        mlcommons_root=str(root),
        output_dir=str(output_dir),
        manifest_dir=str(manifest_dir),
    )

    assert report["row_count"] == 2
    assert (output_dir / "mlcommons_inference_context.jsonl").exists()
    assert (output_dir / "mlcommons_inference_validation_report.json").exists()
    assert (manifest_dir / "mlcommons_inference_intake_report.json").exists()
    assert (manifest_dir / "mlcommons_inference_intake_report.md").exists()


def test_zero_usable_rows_still_writes_report_and_does_not_fail(tmp_path) -> None:
    root = tmp_path / "mlcommons"
    output_dir = tmp_path / "generated"
    manifest_dir = tmp_path / "manifests"
    skipped_path = root / "closed" / "NVIDIA" / "results" / "sys-a" / "resnet50" / "Offline" / "mlperf_log_accuracy.json"
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.write_text(json.dumps({"tokens": [1, 2, 3]}), encoding="utf-8")

    report = run_mlcommons_intake(
        mlcommons_root=str(root),
        output_dir=str(output_dir),
        manifest_dir=str(manifest_dir),
    )

    assert report["row_count"] == 0
    assert (output_dir / "mlcommons_inference_validation_report.json").exists()
    assert (manifest_dir / "mlcommons_inference_intake_report.json").exists()
    assert report["validation_status"] in {"passed", "warning"}


def test_assembly_fallback_includes_mlcommons_context_when_consolidated_absent(tmp_path) -> None:
    generated_dir = tmp_path / "generated"
    manifests_dir = tmp_path / "manifests"
    generated_dir.mkdir()
    row = {
        "row_id": "mlperf_ctx_1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "source": "mlperf",
        "benchmark_name": "MLPerf Inference",
        "workload_name": "resnet50",
        "hardware_name": "NVIDIA H100",
        "software_stack": {"cuda": "12.4"},
        "metrics": {"qps": 1000.0},
        "units": {"qps": "samples/s"},
        "url": None,
        "metadata": {"context_type": "benchmark_result", "vendor": "NVIDIA"},
    }
    (generated_dir / "mlcommons_inference_context.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    summary = assemble_training_dataset(
        history_db_path=str(tmp_path / "missing.db"),
        external_context_paths=[],
        output_dir=str(generated_dir),
        manifest_dir=str(manifests_dir),
    )

    assert summary["benchmark_context_row_count"] == 1


def test_readiness_context_counts_mlperf_rows_as_benchmark_result_context() -> None:
    report = analyze_training_readiness(
        rows=[],
        context_rows=[
            _context_row(
                row_id="mlperf_ctx_1",
                source="mlperf",
                benchmark_name="MLPerf Inference",
                workload_name="resnet50",
                hardware_name="NVIDIA H100",
                metrics={"qps": 1000.0},
                metadata={"context_type": "benchmark_result", "vendor": "NVIDIA"},
            )
        ],
        min_total_rows=1,
        min_labeled_rows=1,
        min_known_label_classes=1,
    )

    assert report["context"]["benchmark_result_context_count"] == 1
    assert report["context"]["mlperf_row_count"] == 1


def _write_system_json(root, vendor: str, system_name: str) -> None:
    path = root / "closed" / vendor / "systems" / f"{system_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "system_name": system_name,
                "accelerator_model_name": "NVIDIA H100 SXM",
                "system_type": "datacenter",
                "framework": "TensorRT 10.1, CUDA 12.4",
            }
        ),
        encoding="utf-8",
    )


def _context_row(**kwargs):
    from gpuboost.schemas.dataset import BenchmarkContextRow

    defaults = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "software_stack": {},
        "units": {},
        "url": None,
        "notes": None,
    }
    defaults.update(kwargs)
    return BenchmarkContextRow(**defaults)
