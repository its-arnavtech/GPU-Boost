"""Tests for Phase 11.8 controlled outcome collection."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from gpuboost.dataset.outcome_collection import (
    collect_outcome_from_benchmark_json,
    collect_outcomes_from_pairs,
    collect_outcomes_from_pairs_file,
    comparison_result_to_dataset_row,
    comparison_result_to_label,
    load_outcome_pairs_file,
)
from gpuboost.dataset.validation import validate_dataset_rows
from gpuboost.schemas.comparison import (
    BenchmarkMetricDelta,
    ComparisonResult,
    ComparisonSection,
)


def test_comparison_result_to_label_improved() -> None:
    label = comparison_result_to_label(_comparison("improved"))

    assert label.value == "improved"
    assert label.source == "comparison"
    assert label.confidence == 0.9


def test_comparison_result_to_label_regressed() -> None:
    label = comparison_result_to_label(_comparison("regressed"))

    assert label.value == "regressed"
    assert label.source == "comparison"
    assert label.confidence == 0.9


def test_comparison_result_to_label_unchanged() -> None:
    label = comparison_result_to_label(_comparison("unchanged"))

    assert label.value == "neutral"
    assert label.source == "comparison"
    assert label.confidence == 0.8


def test_comparison_result_to_label_mixed() -> None:
    label = comparison_result_to_label(_comparison("mixed"))

    assert label.value == "neutral"
    assert label.source == "comparison"
    assert label.confidence == 0.5


def test_comparison_result_to_label_error() -> None:
    label = comparison_result_to_label(
        _comparison("unknown", status="error", error="No comparable metrics.")
    )

    assert label.value == "failed"
    assert label.source == "comparison"
    assert label.confidence == 0.7


def test_comparison_result_to_label_unknown() -> None:
    label = comparison_result_to_label(_comparison("unknown"))

    assert label.value == "unknown"
    assert label.source == "unknown"
    assert label.confidence is None


def test_comparison_result_to_dataset_row_creates_labeled_row() -> None:
    row = comparison_result_to_dataset_row(
        _comparison("improved"),
        row_id="row-001",
        workload_name="matrix_multiply",
        hardware={"gpu_name": "NVIDIA A100"},
        features={"experiment_index": 1},
        metadata={"operator": "local"},
    )

    assert row.row_id == "row-001"
    assert row.source == "controlled_experiment"
    assert row.row_type == "controlled_experiment"
    assert row.workload == {"workload_name": "matrix_multiply"}
    assert row.hardware == {"gpu_name": "NVIDIA A100"}
    assert row.label.value == "improved"
    assert row.features["experiment_index"] == 1
    assert row.features["overall_verdict"] == "improved"
    assert row.metadata["operator"] == "local"
    assert validate_dataset_rows([row]).status == "passed"


def test_metric_deltas_flatten_into_metrics() -> None:
    row = comparison_result_to_dataset_row(_comparison("improved"), row_id="row-001")

    assert row.metrics["before_best_fp32_tflops"] == 10.0
    assert row.metrics["after_best_fp32_tflops"] == 12.0
    assert row.metrics["delta_best_fp32_tflops"] == 2.0
    assert row.metrics["percent_delta_best_fp32_tflops"] == 20.0


def test_comparison_features_count_metric_directions() -> None:
    row = comparison_result_to_dataset_row(_mixed_comparison(), row_id="row-001")

    assert row.features["overall_verdict"] == "mixed"
    assert row.features["section_count"] == 1
    assert row.features["improved_metric_count"] == 1
    assert row.features["regressed_metric_count"] == 1
    assert row.features["unchanged_metric_count"] == 1


def test_privacy_is_safe() -> None:
    row = comparison_result_to_dataset_row(_comparison("improved"), row_id="row-001")

    assert row.privacy.contains_raw_source is False
    assert row.privacy.contains_raw_diff is False
    assert row.privacy.contains_stdout is False
    assert row.privacy.contains_stderr is False
    assert row.is_safe_for_export() is True


def test_no_raw_source_diff_stdout_stderr_in_row_dict() -> None:
    row = comparison_result_to_dataset_row(
        _comparison("improved"),
        row_id="row-001",
        hardware={"raw_source": "def train(): pass", "gpu_name": "NVIDIA A100"},
        features={"raw_diff": "--- a\n+++ b", "safe_feature": 1},
        metadata={"stdout": "loud", "stderr": "noisy", "safe_meta": True},
    )
    data = row.to_dict()

    assert _contains_key(data, "raw_source") is False
    assert _contains_key(data, "raw_diff") is False
    assert _contains_key(data, "stdout") is False
    assert _contains_key(data, "stderr") is False
    assert row.hardware == {"gpu_name": "NVIDIA A100"}
    assert row.features["safe_feature"] == 1
    assert row.metadata["safe_meta"] is True


def test_collect_outcome_from_benchmark_json_loads_local_files(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    optimized_path = tmp_path / "optimized.json"
    _write_json(baseline_path, _benchmark(10.0))
    _write_json(optimized_path, _benchmark(12.0))

    row, comparison = collect_outcome_from_benchmark_json(
        str(baseline_path),
        str(optimized_path),
        row_id="controlled-001",
        workload_name="tiny_fixture",
        hardware={"gpu_name": "fixture gpu"},
    )

    assert row.row_id == "controlled-001"
    assert row.label.value == "improved"
    assert row.workload == {"workload_name": "tiny_fixture"}
    assert row.hardware == {"gpu_name": "fixture gpu"}
    assert comparison.overall_verdict == "improved"


def test_collect_outcome_from_benchmark_json_writes_outputs(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    optimized_path = tmp_path / "optimized.json"
    output_dir = tmp_path / "out"
    _write_json(baseline_path, _benchmark(10.0))
    _write_json(optimized_path, _benchmark(12.0))

    row, _comparison = collect_outcome_from_benchmark_json(
        str(baseline_path),
        str(optimized_path),
        row_id="controlled-001",
        output_dir=str(output_dir),
    )

    comparison_path = output_dir / "controlled-001.comparison.json"
    row_path = output_dir / "controlled-001.dataset_row.json"

    assert comparison_path.exists()
    assert row_path.exists()
    assert json.loads(row_path.read_text(encoding="utf-8"))["row_id"] == row.row_id
    assert json.loads(comparison_path.read_text(encoding="utf-8"))[
        "overall_verdict"
    ] == "improved"


def test_collect_outcomes_from_pairs_processes_multiple_pairs(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    optimized_path = tmp_path / "optimized.json"
    regressed_path = tmp_path / "regressed.json"
    _write_json(baseline_path, _benchmark(10.0))
    _write_json(optimized_path, _benchmark(12.0))
    _write_json(regressed_path, _benchmark(8.0))

    rows = collect_outcomes_from_pairs(
        [
            {
                "baseline_json_path": str(baseline_path),
                "optimized_json_path": str(optimized_path),
                "row_id": "row-improved",
            },
            {
                "baseline_json_path": str(baseline_path),
                "optimized_json_path": str(regressed_path),
                "row_id": "row-regressed",
            },
        ]
    )

    assert [row.row_id for row in rows] == ["row-improved", "row-regressed"]
    assert [row.label.value for row in rows] == ["improved", "regressed"]


def test_missing_json_file_raises_clean_error(tmp_path) -> None:
    optimized_path = tmp_path / "optimized.json"
    _write_json(optimized_path, _benchmark(12.0))

    with pytest.raises(FileNotFoundError, match="Benchmark JSON file not found"):
        collect_outcome_from_benchmark_json(
            str(tmp_path / "missing.json"),
            str(optimized_path),
        )


def test_invalid_json_raises_clean_error(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    optimized_path = tmp_path / "optimized.json"
    baseline_path.write_text("{not json", encoding="utf-8")
    _write_json(optimized_path, _benchmark(12.0))

    with pytest.raises(ValueError, match="Invalid benchmark JSON"):
        collect_outcome_from_benchmark_json(str(baseline_path), str(optimized_path))


def test_no_command_execution_occurs(tmp_path, monkeypatch) -> None:
    baseline_path = tmp_path / "baseline.json"
    optimized_path = tmp_path / "optimized.json"
    _write_json(baseline_path, _benchmark(10.0))
    _write_json(optimized_path, _benchmark(12.0))

    def fail_execution(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Outcome collection must not execute commands.")

    monkeypatch.setattr(os, "system", fail_execution)
    monkeypatch.setattr(subprocess, "run", fail_execution)
    monkeypatch.setattr(subprocess, "Popen", fail_execution)

    row, comparison = collect_outcome_from_benchmark_json(
        str(baseline_path),
        str(optimized_path),
    )

    assert row.label.value == "improved"
    assert comparison.overall_verdict == "improved"


def test_load_outcome_pairs_file_supports_list_shape(tmp_path) -> None:
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps(
            [
                {
                    "row_id": "dataloader_001",
                    "workload_name": "dataloader_bottleneck",
                    "baseline_json_path": "baseline.json",
                    "optimized_json_path": "optimized.json",
                    "hardware": {"gpu_name": "NVIDIA Test GPU"},
                    "features": {"batch_size": 32},
                    "metadata": {"experiment": "local"},
                }
            ]
        ),
        encoding="utf-8",
    )

    pairs = load_outcome_pairs_file(str(pairs_path))

    assert pairs == [
        {
            "row_id": "dataloader_001",
            "workload_name": "dataloader_bottleneck",
            "baseline_json_path": "baseline.json",
            "optimized_json_path": "optimized.json",
            "hardware": {"gpu_name": "NVIDIA Test GPU"},
            "features": {"batch_size": 32},
            "metadata": {"experiment": "local"},
        }
    ]


def test_load_outcome_pairs_file_supports_object_with_pairs_shape(tmp_path) -> None:
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "baseline_json_path": "baseline.json",
                        "optimized_json_path": "optimized.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    pairs = load_outcome_pairs_file(str(pairs_path))

    assert pairs == [
        {
            "baseline_json_path": "baseline.json",
            "optimized_json_path": "optimized.json",
        }
    ]


def test_load_outcome_pairs_file_missing_required_path_fields_fail(tmp_path) -> None:
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps([{"baseline_json_path": "baseline.json"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="optimized_json_path"):
        load_outcome_pairs_file(str(pairs_path))


def test_load_outcome_pairs_file_invalid_json_shape_fails(tmp_path) -> None:
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(json.dumps({"bad": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="list or an object with a pairs list"):
        load_outcome_pairs_file(str(pairs_path))


def test_collect_outcomes_from_pairs_file_collects_improved_row(tmp_path) -> None:
    pairs_path = _write_pairs_fixture(tmp_path, optimized_value=12.0)

    summary = collect_outcomes_from_pairs_file(
        str(pairs_path),
        output_dir=str(tmp_path / "out"),
    )

    assert summary["pair_count"] == 1
    assert summary["collected_row_count"] == 1
    assert summary["label_counts"] == {"improved": 1}
    assert summary["validation_status"] == "passed"
    assert summary["errors"] == []


def test_collect_outcomes_from_pairs_file_continues_after_one_bad_pair(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    optimized_path = tmp_path / "optimized.json"
    pairs_path = tmp_path / "pairs.json"
    _write_json(baseline_path, _benchmark(10.0))
    _write_json(optimized_path, _benchmark(12.0))
    pairs_path.write_text(
        json.dumps(
            [
                {
                    "row_id": "good-row",
                    "baseline_json_path": "baseline.json",
                    "optimized_json_path": "optimized.json",
                },
                {
                    "row_id": "bad-row",
                    "baseline_json_path": "missing.json",
                    "optimized_json_path": "optimized.json",
                },
            ]
        ),
        encoding="utf-8",
    )

    summary = collect_outcomes_from_pairs_file(
        str(pairs_path),
        output_dir=str(tmp_path / "out"),
    )

    assert summary["pair_count"] == 2
    assert summary["collected_row_count"] == 1
    assert summary["label_counts"] == {"improved": 1}
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["row_id"] == "bad-row"


def test_collect_outcomes_from_pairs_file_writes_output_files(tmp_path) -> None:
    pairs_path = _write_pairs_fixture(tmp_path, optimized_value=12.0)
    output_dir = tmp_path / "out"

    collect_outcomes_from_pairs_file(str(pairs_path), output_dir=str(output_dir))

    assert (output_dir / "outcome_dataset.jsonl").exists()
    assert (output_dir / "outcome_manifest.json").exists()
    assert (output_dir / "outcome_validation_report.json").exists()
    assert (output_dir / "outcome_collection_report.json").exists()
    assert (output_dir / "outcome_collection_report.md").exists()
    assert (output_dir / "training_readiness_report.json").exists()
    assert (output_dir / "training_readiness_report.md").exists()


def test_collect_outcomes_from_pairs_file_validation_report_is_written(tmp_path) -> None:
    pairs_path = _write_pairs_fixture(tmp_path, optimized_value=12.0)
    output_dir = tmp_path / "out"

    collect_outcomes_from_pairs_file(str(pairs_path), output_dir=str(output_dir))

    validation = json.loads(
        (output_dir / "outcome_validation_report.json").read_text(encoding="utf-8")
    )
    assert validation["status"] == "passed"
    assert validation["row_count"] == 1


def test_collect_outcomes_from_pairs_file_counts_labels(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    improved_path = tmp_path / "improved.json"
    regressed_path = tmp_path / "regressed.json"
    pairs_path = tmp_path / "pairs.json"
    _write_json(baseline_path, _benchmark(10.0))
    _write_json(improved_path, _benchmark(12.0))
    _write_json(regressed_path, _benchmark(8.0))
    pairs_path.write_text(
        json.dumps(
            [
                {
                    "baseline_json_path": "baseline.json",
                    "optimized_json_path": "improved.json",
                },
                {
                    "baseline_json_path": "baseline.json",
                    "optimized_json_path": "regressed.json",
                },
            ]
        ),
        encoding="utf-8",
    )

    summary = collect_outcomes_from_pairs_file(
        str(pairs_path),
        output_dir=str(tmp_path / "out"),
    )

    assert summary["label_counts"] == {"improved": 1, "regressed": 1}


def test_collect_outcomes_from_pairs_file_stores_no_raw_benchmark_json_or_outputs(
    tmp_path,
) -> None:
    pairs_path = _write_pairs_fixture(tmp_path, optimized_value=12.0)
    output_dir = tmp_path / "out"

    collect_outcomes_from_pairs_file(str(pairs_path), output_dir=str(output_dir))

    row_jsonl = (output_dir / "outcome_dataset.jsonl").read_text(encoding="utf-8")
    row = json.loads(row_jsonl)
    forbidden = {
        "source_code",
        "\"results\"",
        "\"metrics\": [",
    }

    assert all(token not in row_jsonl for token in forbidden)
    assert _contains_key(row, "raw_source") is False
    assert _contains_key(row, "raw_diff") is False
    assert _contains_key(row, "stdout") is False
    assert _contains_key(row, "stderr") is False


def _comparison(
    verdict: str,
    status: str = "ok",
    error: str | None = None,
) -> ComparisonResult:
    sections = []
    if verdict != "unknown" or status != "error":
        sections = [
            ComparisonSection(
                title="Matrix Multiply",
                metrics=[
                    BenchmarkMetricDelta(
                        name="best_fp32_tflops",
                        unit="TFLOPS",
                        before=10.0,
                        after=12.0,
                        absolute_delta=2.0,
                        percent_delta=20.0,
                        direction="improved",
                        higher_is_better=True,
                        summary="Improved.",
                    )
                ],
                verdict=verdict,
            )
        ]
    return ComparisonResult(
        generated_at="2026-01-01T00:00:00+00:00",
        status=status,
        baseline_label="baseline",
        optimized_label="optimized",
        sections=sections,
        overall_verdict=verdict,
        warnings=[],
        error=error,
    )


def _mixed_comparison() -> ComparisonResult:
    return ComparisonResult(
        generated_at="2026-01-01T00:00:00+00:00",
        status="ok",
        baseline_label="baseline",
        optimized_label="optimized",
        sections=[
            ComparisonSection(
                title="Mixed",
                metrics=[
                    _delta("throughput", "improved"),
                    _delta("latency", "regressed"),
                    _delta("batch_size", "unchanged"),
                ],
                verdict="mixed",
            )
        ],
        overall_verdict="mixed",
    )


def _delta(name: str, direction: str) -> BenchmarkMetricDelta:
    return BenchmarkMetricDelta(
        name=name,
        unit=None,
        before=1.0,
        after=2.0,
        absolute_delta=1.0,
        percent_delta=100.0,
        direction=direction,
        higher_is_better=True,
        summary=direction,
    )


def _benchmark(value: float) -> dict:
    return {
        "results": [
            {
                "name": "Matrix Multiply",
                "metrics": [
                    {
                        "name": "best_fp32_tflops",
                        "value": value,
                        "unit": "TFLOPS",
                    }
                ],
            }
        ]
    }


def _write_json(path, data: dict) -> None:  # noqa: ANN001
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_pairs_fixture(tmp_path, optimized_value: float):  # noqa: ANN001, ANN201
    baseline_path = tmp_path / "baseline.json"
    optimized_path = tmp_path / "optimized.json"
    pairs_path = tmp_path / "pairs.json"
    _write_json(baseline_path, _benchmark(10.0))
    _write_json(optimized_path, _benchmark(optimized_value))
    pairs_path.write_text(
        json.dumps(
            [
                {
                    "row_id": "controlled-001",
                    "workload_name": "tiny_fixture",
                    "baseline_json_path": "baseline.json",
                    "optimized_json_path": "optimized.json",
                    "hardware": {"gpu_name": "fixture gpu"},
                }
            ]
        ),
        encoding="utf-8",
    )
    return pairs_path


def _contains_key(value: object, target_key: str) -> bool:
    if isinstance(value, dict):
        return any(
            key == target_key or _contains_key(item, target_key)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(item, target_key) for item in value)
    return False


def test_cli_collect_outcomes_exits_nonzero_when_all_pairs_fail(tmp_path) -> None:
    from gpuboost.cli import main as cli_main

    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps(
            [
                {
                    "baseline_json_path": "does_not_exist_a.json",
                    "optimized_json_path": "does_not_exist_b.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli_main.main(
        [
            "dataset",
            "collect-outcomes",
            str(pairs_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1


def test_path_from_text_preserves_unc_on_windows_and_normalizes_on_posix() -> None:
    import os
    from pathlib import Path

    from gpuboost.dataset.outcome_collection import _path_from_text

    # Forward-slash paths are always preserved.
    assert _path_from_text("dir/sub/file.json") == Path("dir/sub/file.json")

    if os.name == "nt":
        # On Windows, backslashes (incl. UNC) must NOT be rewritten to "/".
        assert _path_from_text(r"\\server\share\f.json") == Path(
            r"\\server\share\f.json"
        )
    else:
        # On POSIX, Windows-style backslash paths are normalized so they resolve.
        assert _path_from_text("dir\\sub\\file.json") == Path("dir/sub/file.json")
