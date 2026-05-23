"""Tests for Phase 11.3 local benchmark context importers."""

from __future__ import annotations

import json

import pytest

from gpuboost.dataset.benchmark_importers import (
    import_benchmark_context_csv,
    import_benchmark_context_json,
    normalize_source_name,
    parse_scalar,
)


def test_json_list_import(tmp_path) -> None:
    path = tmp_path / "benchmarks.json"
    path.write_text(
        json.dumps(
            [
                {
                    "benchmark_name": "MLPerf Training",
                    "workload_name": "resnet",
                    "hardware_name": "NVIDIA A100",
                    "metrics": {"samples_per_sec": 1234.5},
                    "units": {"samples_per_sec": "samples/sec"},
                    "software_stack": {"cuda": "12.4"},
                    "url": "https://example.invalid/mlperf",
                    "notes": "Local fixture.",
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = import_benchmark_context_json(str(path), source="MLPerf")

    assert len(rows) == 1
    assert rows[0].row_id == "mlperf_0"
    assert rows[0].source == "mlperf"
    assert rows[0].benchmark_name == "MLPerf Training"
    assert rows[0].metrics["samples_per_sec"] == 1234.5
    assert rows[0].units["samples_per_sec"] == "samples/sec"
    assert rows[0].software_stack["cuda"] == "12.4"


def test_json_object_with_rows_import(tmp_path) -> None:
    path = tmp_path / "benchmarks.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "benchmark_name": "OpenBenchmarking",
                        "metrics": {"score": 42},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = import_benchmark_context_json(str(path), source="OpenBenchmarking")

    assert len(rows) == 1
    assert rows[0].row_id == "openbenchmarking_0"
    assert rows[0].benchmark_name == "OpenBenchmarking"
    assert rows[0].workload_name is None


def test_json_invalid_shape_raises(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"bad": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="rows"):
        import_benchmark_context_json(str(path), source="manual")


def test_csv_import_creates_benchmark_context_row(tmp_path) -> None:
    path = tmp_path / "benchmarks.csv"
    path.write_text(
        "\n".join(
            [
                "benchmark_name,workload_name,hardware_name,metric_name,"
                "metric_value,metric_unit,cuda,framework,url,notes",
                "Rodinia,bfs,NVIDIA A100,throughput,1234.5,items/sec,"
                "12.4,CUDA,https://example.invalid/rodinia,Local fixture",
            ]
        ),
        encoding="utf-8",
    )

    rows = import_benchmark_context_csv(str(path), source="Rodinia")

    assert len(rows) == 1
    row = rows[0]
    assert row.row_id == "rodinia_0"
    assert row.benchmark_name == "Rodinia"
    assert row.workload_name == "bfs"
    assert row.hardware_name == "NVIDIA A100"
    assert row.metrics == {"throughput": 1234.5}
    assert row.units == {"throughput": "items/sec"}
    assert row.software_stack == {"cuda": 12.4, "framework": "CUDA"}


def test_csv_missing_required_columns_raises(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(
        "\n".join(["benchmark_name,metric_name", "SPEC ACCEL,score"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metric_value"):
        import_benchmark_context_csv(str(path), source="SPEC ACCEL")


def test_parse_scalar_int_float_bool_none_and_string() -> None:
    assert parse_scalar("42") == 42
    assert parse_scalar("42.5") == 42.5
    assert parse_scalar("true") is True
    assert parse_scalar("FALSE") is False
    assert parse_scalar("") is None
    assert parse_scalar(" value ") == "value"


def test_normalize_source_name_examples() -> None:
    assert normalize_source_name("MLPerf") == "mlperf"
    assert normalize_source_name("OpenBenchmarking") == "openbenchmarking"
    assert normalize_source_name("SPEC ACCEL") == "spec_accel"
    assert normalize_source_name("phoronix-suite") == "phoronix_suite"


def test_optional_fields_handled(tmp_path) -> None:
    json_path = tmp_path / "minimal.json"
    json_path.write_text(
        json.dumps([{"benchmark_name": "Minimal"}]),
        encoding="utf-8",
    )
    csv_path = tmp_path / "minimal.csv"
    csv_path.write_text(
        "\n".join(["benchmark_name,metric_name,metric_value", "Minimal,score,7"]),
        encoding="utf-8",
    )

    json_row = import_benchmark_context_json(str(json_path), source="manual")[0]
    csv_row = import_benchmark_context_csv(str(csv_path), source="manual")[0]

    assert json_row.workload_name is None
    assert json_row.hardware_name is None
    assert json_row.metrics == {}
    assert csv_row.workload_name is None
    assert csv_row.hardware_name is None
    assert csv_row.units == {}
    assert csv_row.software_stack == {}


def test_row_id_includes_normalized_source(tmp_path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(
        json.dumps([{"benchmark_name": "SPEC ACCEL"}, {"benchmark_name": "SPEC ACCEL"}]),
        encoding="utf-8",
    )

    rows = import_benchmark_context_json(str(path), source="SPEC ACCEL")

    assert [row.row_id for row in rows] == ["spec_accel_0", "spec_accel_1"]


def test_importers_do_not_call_network(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"benchmark_name": "Manual"}]), encoding="utf-8")

    def fail_network(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Network should not be called.")

    monkeypatch.setattr("socket.create_connection", fail_network)

    rows = import_benchmark_context_json(str(path), source="manual")

    assert len(rows) == 1


def test_imported_rows_json_serialize(tmp_path) -> None:
    path = tmp_path / "benchmarks.csv"
    path.write_text(
        "\n".join(["benchmark_name,metric_name,metric_value", "Manual,score,10"]),
        encoding="utf-8",
    )

    rows = import_benchmark_context_csv(str(path), source="manual")
    serialized = json.dumps([row.to_dict() for row in rows])
    deserialized = json.loads(serialized)

    assert deserialized[0]["benchmark_name"] == "Manual"
    assert deserialized[0]["metrics"]["score"] == 10


def test_supported_source_names(tmp_path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"benchmark_name": "Fixture"}]), encoding="utf-8")

    sources = [
        "mlperf",
        "openbenchmarking",
        "phoronix",
        "rodinia",
        "spec_accel",
        "manual",
    ]
    imported_sources = [
        import_benchmark_context_json(str(path), source=source)[0].source
        for source in sources
    ]

    assert imported_sources == sources
