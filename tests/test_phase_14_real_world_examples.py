"""Tests for Phase 14 real-world PyTorch example workloads."""

from __future__ import annotations

import importlib.util
import io
import json
import runpy
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "real_world"
EXAMPLE_SCRIPTS = [
    "pytorch_cnn_baseline.py",
    "pytorch_cnn_optimized.py",
    "transformer_toy_baseline.py",
    "transformer_toy_optimized.py",
    "dataloader_training_baseline.py",
    "dataloader_training_optimized.py",
]


def test_phase_14_real_world_files_exist() -> None:
    assert (EXAMPLE_DIR / "README.md").exists()
    for script_name in EXAMPLE_SCRIPTS:
        assert (EXAMPLE_DIR / script_name).exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is not installed in this environment.",
)
def test_phase_14_real_world_scripts_emit_benchmark_json() -> None:
    for script_name in EXAMPLE_SCRIPTS:
        stdout, stderr = _run_example_script(script_name)

        assert "Traceback" not in stderr
        assert stdout.strip()
        payload = json.loads(stdout)
        _assert_benchmark_payload(payload, script_name)


def _assert_benchmark_payload(payload: dict, script_name: str) -> None:
    assert sorted(payload) == ["metadata", "results"]
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) == 1

    result = payload["results"][0]
    assert result["status"] == "ok"
    assert isinstance(result["name"], str)
    assert isinstance(result["metrics"], list)
    assert result["metrics"]

    metric_names = {metric["name"] for metric in result["metrics"]}
    assert {"samples_per_sec", "median_step_ms"}.issubset(metric_names)
    for metric in result["metrics"]:
        assert isinstance(metric["name"], str)
        assert isinstance(metric["value"], int | float)
        assert metric["value"] >= 0
        assert metric["unit"] in {"samples/sec", "ms"}

    metadata = payload["metadata"]
    assert metadata["example"] == "real_world"
    assert metadata["cuda_available"] in {True, False}
    assert metadata["variant"] in {"baseline", "optimized"}
    assert metadata["quick"] is True
    if "cnn" in script_name:
        assert metadata["workload_family"] == "cnn_image_classification"
    elif "transformer" in script_name:
        assert metadata["workload_family"] == "toy_transformer_text_classification"
    else:
        assert metadata["workload_family"] == "dataloader_training"


def _run_example_script(script_name: str) -> tuple[str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    script_path = EXAMPLE_DIR / script_name
    original_argv = sys.argv[:]
    original_path = sys.path[:]
    try:
        sys.argv = [str(script_path), "--quick", "--benchmark-json"]
        sys.path.insert(0, str(EXAMPLE_DIR))
        with redirect_stdout(stdout), redirect_stderr(stderr):
            runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path = original_path
    return stdout.getvalue(), stderr.getvalue()
