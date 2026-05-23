"""Tests for controlled outcome workload experiment scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_DIR = ROOT / "examples" / "outcome_collection" / "workloads"
RUNNER_PATH = ROOT / "scripts" / "run_outcome_experiments.ps1"
PAIRS_PATH = ROOT / "data" / "gpuboost" / "experiments" / "pairs.json"

WORKLOAD_SCRIPTS = [
    "dataloader_baseline.py",
    "dataloader_optimized.py",
    "amp_baseline.py",
    "amp_optimized.py",
    "batch_small_baseline.py",
    "batch_small_optimized.py",
]


def test_workload_script_files_exist() -> None:
    for script_name in WORKLOAD_SCRIPTS:
        assert (WORKLOAD_DIR / script_name).exists()


def test_runner_script_exists() -> None:
    assert RUNNER_PATH.exists()


def test_pairs_json_has_expected_structure() -> None:
    pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))

    assert [pair["row_id"] for pair in pairs] == [
        "controlled_dataloader_001",
        "controlled_amp_001",
        "controlled_batch_001",
    ]
    for pair in pairs:
        assert pair["baseline_json_path"].endswith("baseline.json")
        assert pair["optimized_json_path"].endswith("optimized.json")
        assert pair["hardware"] == {"gpu_name": "auto"}


def test_workloads_emit_valid_smoke_json() -> None:
    for script_name in WORKLOAD_SCRIPTS:
        payload = _run_workload_smoke(script_name)

        assert isinstance(payload["results"], list)
        assert payload["results"]
        result = payload["results"][0]
        assert result["status"] == "ok"
        assert isinstance(result["metrics"], list)
        assert result["metrics"]
        assert payload["metadata"]["cuda_available"] in {True, False}

        metric_names = {metric["name"] for metric in result["metrics"]}
        assert "samples_per_sec" in metric_names
        assert "median_step_ms" in metric_names
        assert "fp32_samples_per_sec" in metric_names

        for metric in result["metrics"]:
            assert isinstance(metric["value"], int | float)


def test_smoke_workloads_write_json_to_stdout_only() -> None:
    payload = _run_workload_smoke("batch_small_baseline.py", return_process=True)

    assert payload.stderr == ""
    assert json.loads(payload.stdout)["metadata"]["workload"] == "batch_size"


def _run_workload_smoke(script_name: str, return_process: bool = False):
    env = dict(os.environ)
    env["GPUBOOST_OUTCOME_SMOKE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(WORKLOAD_DIR / script_name)],
        check=True,
        capture_output=True,
        encoding="utf-8",
        env=env,
    )
    if return_process:
        return completed
    return json.loads(completed.stdout)
