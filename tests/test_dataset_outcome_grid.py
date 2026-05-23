"""Tests for Phase 11.10 controlled outcome grid generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from gpuboost.dataset.outcome_grid import (
    build_controlled_outcome_grid,
    write_grid_pairs_file,
    write_grid_runner_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_DIR = ROOT / "examples" / "outcome_collection" / "workloads"
GRID_RUNNER_PATH = ROOT / "scripts" / "run_outcome_experiment_grid.ps1"
RAW_FIELD_NAMES = {
    "raw_source",
    "source_code",
    "raw_diff",
    "unified_diff",
    "stdout",
    "stderr",
}


def test_build_controlled_outcome_grid_is_deterministic(tmp_path) -> None:
    first = build_controlled_outcome_grid(output_root=str(tmp_path / "grid"))
    second = build_controlled_outcome_grid(output_root=str(tmp_path / "grid"))

    assert first == second
    assert len(first) >= 120
    assert first[0]["row_id"] == "controlled_grid_dataloader_001"
    assert any(pair["row_id"] == "controlled_grid_amp_001" for pair in first)
    assert any(pair["row_id"] == "controlled_grid_batch_001" for pair in first)
    assert any(pair["row_id"] == "controlled_grid_neutral_001" for pair in first)


def test_build_controlled_outcome_grid_has_at_least_30_pairs_per_family(
    tmp_path,
) -> None:
    pairs = build_controlled_outcome_grid(output_root=str(tmp_path / "grid"))
    family_counts = Counter(pair["metadata"]["workload_family"] for pair in pairs)

    assert family_counts["dataloader"] >= 30
    assert family_counts["amp"] >= 30
    assert family_counts["batch"] >= 30
    assert family_counts["neutral_control"] >= 30
    assert any(pair["row_id"] == "controlled_grid_dataloader_030" for pair in pairs)
    assert any(pair["row_id"] == "controlled_grid_amp_030" for pair in pairs)
    assert any(pair["row_id"] == "controlled_grid_batch_030" for pair in pairs)
    assert any(pair["row_id"] == "controlled_grid_neutral_030" for pair in pairs)


def test_build_controlled_outcome_grid_interleaves_families_deterministically(
    tmp_path,
) -> None:
    pairs = build_controlled_outcome_grid(
        output_root=str(tmp_path / "grid"),
        max_pairs=8,
    )

    assert [pair["row_id"] for pair in pairs] == [
        "controlled_grid_dataloader_001",
        "controlled_grid_amp_001",
        "controlled_grid_batch_001",
        "controlled_grid_neutral_001",
        "controlled_grid_dataloader_002",
        "controlled_grid_amp_002",
        "controlled_grid_batch_002",
        "controlled_grid_neutral_002",
    ]


def test_build_controlled_outcome_grid_max_pairs_truncates(tmp_path) -> None:
    pairs = build_controlled_outcome_grid(
        output_root=str(tmp_path / "grid"),
        max_pairs=2,
    )

    assert [pair["row_id"] for pair in pairs] == [
        "controlled_grid_dataloader_001",
        "controlled_grid_amp_001",
    ]


def test_build_controlled_outcome_grid_row_ids_and_paths_are_safe(tmp_path) -> None:
    pairs = build_controlled_outcome_grid(output_root=str(tmp_path / "grid"))
    row_ids = [pair["row_id"] for pair in pairs]

    assert len(row_ids) == len(set(row_ids))
    for pair in pairs:
        assert pair["baseline_json_path"] != pair["optimized_json_path"]
        assert pair["baseline_json_path"].endswith("baseline.json")
        assert pair["optimized_json_path"].endswith("optimized.json")
        assert pair["workload_name"]
        assert pair["hardware"]["gpu_name"] == "auto"
        assert pair["features"]["controlled_grid"] is True
        assert pair["metadata"]["grid_schema_version"] == "dataset.outcome_grid.v1"


def test_build_controlled_outcome_grid_includes_neutral_family(tmp_path) -> None:
    pairs = build_controlled_outcome_grid(output_root=str(tmp_path / "grid"))
    neutral_pairs = [
        pair
        for pair in pairs
        if pair["metadata"]["workload_family"] == "neutral_control"
    ]

    assert neutral_pairs
    assert neutral_pairs[0]["row_id"] == "controlled_grid_neutral_001"
    assert neutral_pairs[0]["baseline_script"].endswith("neutral_baseline.py")
    assert neutral_pairs[0]["optimized_script"].endswith("neutral_optimized.py")


def test_grid_builder_does_not_execute_commands_or_write_outputs(tmp_path) -> None:
    pairs = build_controlled_outcome_grid(output_root=str(tmp_path / "grid"))

    for pair in pairs:
        assert not Path(pair["baseline_json_path"]).exists()
        assert not Path(pair["optimized_json_path"]).exists()


def test_write_grid_pairs_file_is_collect_outcomes_compatible(tmp_path) -> None:
    pairs = build_controlled_outcome_grid(
        output_root=str(tmp_path / "experiments" / "grid"),
        max_pairs=1,
    )
    output_path = tmp_path / "experiments" / "grid_pairs.json"

    written = write_grid_pairs_file(pairs, output_path=str(output_path))
    payload = json.loads(Path(written).read_text(encoding="utf-8"))

    assert len(payload) == 1
    assert set(payload[0]) == {
        "row_id",
        "workload_name",
        "baseline_json_path",
        "optimized_json_path",
        "hardware",
        "features",
        "metadata",
    }
    assert payload[0]["baseline_json_path"] == (
        "grid/controlled_grid_dataloader_001/baseline.json"
    )
    assert payload[0]["optimized_json_path"] == (
        "grid/controlled_grid_dataloader_001/optimized.json"
    )


def test_write_grid_runner_manifest_includes_scripts_and_args(tmp_path) -> None:
    pairs = build_controlled_outcome_grid(
        output_root=str(tmp_path / "experiments" / "grid"),
        max_pairs=1,
    )
    output_path = tmp_path / "experiments" / "grid_runner_manifest.json"

    written = write_grid_runner_manifest(pairs, output_path=str(output_path))
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    pair = payload["pairs"][0]

    assert payload["schema_version"] == "dataset.outcome_grid.v1"
    assert payload["pair_count"] == 1
    assert pair["baseline_script"].endswith("dataloader_baseline.py")
    assert pair["optimized_script"].endswith("dataloader_optimized.py")
    assert "--batch-size" in pair["baseline_args"]
    assert "--batch-size" in pair["optimized_args"]


def test_write_grid_runner_manifest_includes_neutral_family(tmp_path) -> None:
    pairs = build_controlled_outcome_grid(output_root=str(tmp_path / "grid"))
    output_path = tmp_path / "grid_runner_manifest.json"

    written = write_grid_runner_manifest(pairs, output_path=str(output_path))
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    neutral_pairs = [
        pair
        for pair in payload["pairs"]
        if pair["metadata"]["workload_family"] == "neutral_control"
    ]

    assert neutral_pairs
    assert neutral_pairs[0]["baseline_script"].endswith("neutral_baseline.py")
    assert neutral_pairs[0]["optimized_script"].endswith("neutral_optimized.py")


def test_workload_scripts_accept_smoke_args_and_emit_valid_json() -> None:
    dataloader = _run_workload(
        "dataloader_baseline.py",
        "--workload-id",
        "test_dataloader",
        "--batch-size",
        "4",
        "--feature-size",
        "32",
        "--num-batches",
        "2",
        "--warmup",
        "1",
        "--num-workers",
        "0",
        "--pin-memory",
        "false",
        "--device",
        "cpu",
    )
    amp = _run_workload(
        "amp_optimized.py",
        "--workload-id",
        "test_amp",
        "--batch-size",
        "8",
        "--feature-size",
        "64",
        "--hidden-size",
        "64",
        "--num-batches",
        "2",
        "--warmup",
        "1",
        "--amp",
        "false",
        "--device",
        "cpu",
    )

    assert dataloader["metadata"]["workload_id"] == "test_dataloader"
    assert dataloader["metadata"]["requested_device"] == "cpu"
    assert amp["metadata"]["workload_id"] == "test_amp"
    assert amp["metadata"]["amp_requested"] is False
    for payload in (dataloader, amp):
        assert payload["results"][0]["status"] == "ok"
        metric_names = {metric["name"] for metric in payload["results"][0]["metrics"]}
        assert "samples_per_sec" in metric_names
        assert "fp32_samples_per_sec" in metric_names
        assert "median_step_ms" in metric_names


def test_neutral_workload_scripts_exist_and_emit_smoke_json() -> None:
    assert (WORKLOAD_DIR / "neutral_baseline.py").exists()
    assert (WORKLOAD_DIR / "neutral_optimized.py").exists()

    payload = _run_workload(
        "neutral_baseline.py",
        "--workload-id",
        "test_neutral",
        "--batch-size",
        "4",
        "--feature-size",
        "32",
        "--hidden-size",
        "32",
        "--num-batches",
        "2",
        "--warmup",
        "1",
        "--device",
        "cpu",
    )

    assert payload["metadata"]["workload"] == "neutral_control"
    assert payload["metadata"]["workload_id"] == "test_neutral"
    metric_names = {metric["name"] for metric in payload["results"][0]["metrics"]}
    assert "best_batch_size" in metric_names
    assert "max_successful_batch_size" in metric_names
    assert "median_step_ms" in metric_names


def test_powershell_grid_runner_file_exists() -> None:
    assert GRID_RUNNER_PATH.exists()


def test_grid_pairs_file_contains_no_raw_output_fields(tmp_path) -> None:
    pairs = build_controlled_outcome_grid(
        output_root=str(tmp_path / "experiments" / "grid"),
        max_pairs=3,
    )
    output_path = tmp_path / "experiments" / "grid_pairs.json"
    write_grid_pairs_file(pairs, output_path=str(output_path))

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert not _contains_raw_field(payload)


def _run_workload(script_name: str, *args: str) -> dict:
    env = dict(os.environ)
    env["GPUBOOST_OUTCOME_SMOKE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(WORKLOAD_DIR / script_name), *args],
        check=True,
        capture_output=True,
        encoding="utf-8",
        env=env,
    )
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def _contains_raw_field(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in RAW_FIELD_NAMES:
                return True
            if _contains_raw_field(item):
                return True
    elif isinstance(value, list):
        return any(_contains_raw_field(item) for item in value)
    return False
