"""End-to-end CLI smoke tests for Phase 13A."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpuboost.model.neural import torch_available


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_OUTPUT_MARKERS = (
    "Traceback",
    "state_dict",
    "model weights",
    "model_weights",
)
FORBIDDEN_RAW_KEYS = {
    "raw_source",
    "raw_diff",
    "source_code",
    "model_weights",
    "state_dict",
}


def run_cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    pythonpath = str(PROJECT_ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    return subprocess.run(
        [sys.executable, "-m", "gpuboost", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def assert_json_output(
    completed: subprocess.CompletedProcess[str],
    *,
    expected_codes: set[int] | None = None,
) -> dict[str, Any]:
    expected_codes = expected_codes or {0}
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode in expected_codes, combined_output
    for marker in FORBIDDEN_OUTPUT_MARKERS:
        assert marker not in combined_output
    assert completed.stdout.strip(), combined_output
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(completed.stdout) from exc
    assert_no_raw_or_weight_payloads(payload)
    return payload


def make_tiny_training_dataset(path: Path) -> Path:
    rows = [
        _training_row("row-1", "train", "improved", 0.0),
        _training_row("row-2", "train", "improved", 0.2),
        _training_row("row-3", "train", "regressed", 9.0),
        _training_row("row-4", "train", "regressed", 9.2),
        _training_row("row-5", "validation", "improved", 0.1),
        _training_row("row-6", "validation", "regressed", 9.1),
        _training_row("row-7", "test", "improved", 0.05),
        _training_row("row-8", "test", "regressed", 9.05),
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_model_safety_check_and_baseline_cli_smoke(tmp_path: Path) -> None:
    safety = assert_json_output(
        run_cli("model", "safety-check", "--json"),
    )
    assert safety["schema_version"] == "training.model_workflow_safety.v1"
    assert safety["command"] == "model safety-check"
    assert safety["result"]["status"] in {"ok", "warning"}
    assert safety["result"]["generated_dir_ignored"] is True

    dataset = make_tiny_training_dataset(tmp_path / "training_dataset.jsonl")
    output_dir = tmp_path / "baseline_reports"
    baselines = assert_json_output(
        run_cli(
            "model",
            "evaluate-baselines",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--json",
        ),
    )

    assert baselines["schema_version"] == "training.baseline_comparison.v1"
    assert baselines["command"] == "model evaluate-baselines"
    assert baselines["result"]["status"] == "ok"
    assert (output_dir / "baseline_comparison_report.json").exists()
    assert (output_dir / "baseline_comparison_report.md").exists()
    assert not _model_artifact_files(output_dir)


def test_train_neural_cli_smoke_uses_tiny_dataset(tmp_path: Path) -> None:
    dataset = make_tiny_training_dataset(tmp_path / "training_dataset.jsonl")
    output_dir = tmp_path / "neural_reports"
    completed = run_cli(
        "model",
        "train-neural",
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--max-epochs",
        "2",
        "--max-candidates",
        "1",
        "--json",
    )
    payload = assert_json_output(completed, expected_codes={0, 1})

    assert payload["schema_version"] == "training.neural_search_result.v1"
    assert payload["command"] == "model train-neural"
    if completed.returncode == 1:
        assert payload["result"]["status"] == "error"
        assert "Traceback" not in completed.stderr
        return

    result = payload["result"]
    assert result["status"] == "ok"
    assert result["metadata"]["candidate_count"] == 1
    assert result["patch_application_allowed"] is False
    assert result["output_files"]["json_report"].endswith(
        "neural_training_report.json"
    )
    assert (output_dir / "neural_training_report.json").exists()
    assert (output_dir / "neural_training_report.md").exists()
    assert not _model_artifact_files(output_dir)


def test_artifact_lifecycle_and_agent_model_cli_smoke(tmp_path: Path) -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable; artifact lifecycle requires it.")

    dataset = make_tiny_training_dataset(tmp_path / "training_dataset.jsonl")
    output_dir = tmp_path / "neural_reports"
    artifact_dir = tmp_path / "artifacts"
    training = assert_json_output(
        run_cli(
            "model",
            "train-neural",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--artifact-name",
            "phase13-smoke",
            "--max-epochs",
            "2",
            "--max-candidates",
            "1",
            "--save-artifact",
            "--json",
        ),
    )
    result = training["result"]
    manifest = artifact_dir / "phase13-smoke" / "manifest.json"

    assert result["status"] == "ok"
    assert result["patch_application_allowed"] is False
    assert result["artifact_validation_status"] == "ok"
    assert result["artifact_manifest_path"].endswith("phase13-smoke/manifest.json")
    assert manifest.exists()
    assert all(path.is_relative_to(tmp_path) for path in _artifact_files(artifact_dir))

    validation = assert_json_output(
        run_cli("model", "validate-artifact", str(manifest), "--json"),
    )
    assert validation["schema_version"] == "training.model_artifact_validation.v1"
    assert validation["result"]["status"] == "ok"

    prediction = assert_json_output(
        run_cli(
            "model",
            "predict-artifact",
            str(manifest),
            "--features-json",
            '{"features.safe_signal": 1.0}',
            "--json",
        ),
    )
    assert prediction["schema_version"] == "training.model_artifact_prediction.v1"
    assert prediction["result"]["status"] == "ok"
    assert prediction["result"]["metadata"]["patch_application_allowed"] is False
    assert prediction["result"]["predictions"][0]["metadata"][
        "patch_application_allowed"
    ] is False

    agent = assert_json_output(
        run_cli(
            "agent",
            "optimize",
            "examples/bad_train_sample.txt",
            "--model-artifact",
            str(manifest),
            "--json",
        ),
    )
    assert_agent_output_is_advisory(agent, model_expected=True)


def test_agent_optimize_cli_smoke_without_model_artifact() -> None:
    agent = assert_json_output(
        run_cli("agent", "optimize", "examples/bad_train_sample.txt", "--json"),
        expected_codes={0, 1},
    )
    assert_agent_output_is_advisory(agent, model_expected=False)


def assert_agent_output_is_advisory(
    payload: dict[str, Any],
    *,
    model_expected: bool,
) -> None:
    assert payload["schema_version"] == "agent.optimize.v1"
    assert payload["command"] == "agent optimize"
    assert payload["artifacts"]["raw_artifacts_included"] is False
    assert payload["artifacts"]["diff"] is None
    assert payload["result"]["artifacts"]["diff"] is None
    assert payload["result"]["artifacts"]["raw_artifacts_included"] is False
    assert payload["result"]["goal"]["options"]["model"] is model_expected

    model = payload["artifacts"]["model"]
    if model_expected:
        assert model is not None
        assert model["patch_application_allowed"] is False
        assert model["provider"] == "trained_local_model"
    else:
        assert model is None


def assert_no_raw_or_weight_payloads(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in FORBIDDEN_RAW_KEYS
            if key in {"stdout", "stderr", "diff"}:
                assert item in (None, "")
            assert_no_raw_or_weight_payloads(item)
        return
    if isinstance(value, list):
        for item in value:
            assert_no_raw_or_weight_payloads(item)


def _training_row(row_id: str, split: str, label: str, signal: float) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "schema_version": "dataset.row.v1",
        "source": "phase_13_cli_smoke",
        "row_type": "optimization_outcome",
        "hardware": {"gpu_name": "NVIDIA Test GPU"},
        "workload": {"batch_size": 32},
        "features": {
            "safe_signal": signal,
            "workload_family": "phase_13_smoke",
        },
        "metrics": {"fp32_samples_per_sec": 100.0 + signal},
        "label": {"value": label, "source": "comparison"},
        "privacy": {
            "contains_raw_source": False,
            "contains_raw_diff": False,
            "contains_stdout": False,
            "contains_stderr": False,
            "contains_sensitive_path": False,
            "notes": [],
        },
        "split": split,
        "quality_score": 1.0,
        "warnings": [],
        "metadata": {},
    }


def _artifact_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [item for item in path.rglob("*") if item.is_file()]


def _model_artifact_files(path: Path) -> list[Path]:
    artifact_suffixes = {".pt", ".pth", ".pkl", ".safetensors"}
    return [
        item
        for item in _artifact_files(path)
        if item.suffix.lower() in artifact_suffixes or item.name == "manifest.json"
    ]
