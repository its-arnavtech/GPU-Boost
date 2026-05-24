"""Phase 13C security, artifact, and data leak audit tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpuboost.agent.report import AgentReport, AgentReportSection
from gpuboost.cli import main as cli_main
from gpuboost.schemas.agent import AgentAction, AgentGoal, AgentPlan, AgentRunResult
from gpuboost.schemas.model import ModelInferenceResult, ModelPrediction
from gpuboost.security.audit import find_json_leaks


MODEL_ARTIFACT_PATTERNS = ("*.pt", "*.pth", "*.onnx", "*.safetensors", "*.pkl", "*.joblib")
LOCAL_DB_PATTERNS = ("*.db", "*.sqlite", "*.sqlite3")
PRIVATE_PATH_MARKERS = ("C:\\Users", "/Users/", "/home/", "C:\\GPU-Boost")
PRIVATE_KEY_MARKER = "-----BEGIN PRIVATE KEY-----"


def test_phase_13_git_tracking_excludes_raw_generated_model_and_db_artifacts() -> None:
    _skip_if_not_git_repo()

    assert _git_ls_files("data/gpuboost/generated") == []
    assert _git_ls_files("data/gpuboost/raw") == []
    assert _git_ls_files(*MODEL_ARTIFACT_PATTERNS) == []
    assert _git_ls_files(*LOCAL_DB_PATTERNS) == []


def test_phase_13_gitignore_covers_sensitive_data_and_artifact_classes() -> None:
    gitignore = _gitignore_lines()

    for pattern in (
        "data/gpuboost/raw/",
        "data/gpuboost/generated/",
        *MODEL_ARTIFACT_PATTERNS,
        *LOCAL_DB_PATTERNS,
        "__pycache__/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".cache/",
        ".env",
        ".env.*",
        "secrets/",
        "credentials/",
        "*.pem",
        "*.key",
        "*.token",
        "*.secret",
    ):
        assert pattern in gitignore


def test_phase_13_model_list_and_show_json_do_not_leak_artifact_payloads(
    tmp_path,
    capsys,
) -> None:
    manifest_path = _write_model_artifact_fixture(tmp_path / "fixture")

    list_exit = cli_main.main(
        ["model", "list-artifacts", "--artifacts-dir", str(tmp_path), "--json"]
    )
    list_output = capsys.readouterr().out
    list_payload = json.loads(list_output)

    show_exit = cli_main.main(["model", "show-artifact", str(manifest_path), "--json"])
    show_output = capsys.readouterr().out
    show_payload = json.loads(show_output)

    assert list_exit == 0
    assert show_exit == 0
    _assert_no_json_leaks(list_payload, list_output)
    _assert_no_json_leaks(show_payload, show_output)
    for output in (list_output, show_output):
        assert "model.pt" not in output
        assert "serialized raw model weights" not in output
        assert PRIVATE_KEY_MARKER not in output
        assert str(tmp_path) not in output


def test_phase_13_predict_json_does_not_echo_sensitive_input_features(
    monkeypatch,
    capsys,
) -> None:
    class FakeProvider:
        def __init__(self, manifest_path: str, device: str = "cpu") -> None:
            self.manifest_path = manifest_path
            self.device = device

        def predict(self, model_input) -> ModelInferenceResult:
            assert model_input.context["features"]["api_token"] == "tok-secret"
            return ModelInferenceResult(
                model_available=True,
                model_name="mlp_classifier",
                model_version="training.model_artifact.v1",
                status="ok",
                predictions=[
                    ModelPrediction(
                        id="trained_local_prediction",
                        target="optimization_outcome",
                        label="improved",
                        score=0.91,
                        confidence=0.91,
                        rationale="Advisory local prediction.",
                        metadata={
                            "provider": "trained_local_model",
                            "patch_application_allowed": False,
                        },
                    )
                ],
                metadata={
                    "provider": "trained_local_model",
                    "patch_application_allowed": False,
                },
            )

    monkeypatch.setattr(cli_main, "TrainedLocalModelProvider", FakeProvider)
    features_json = json.dumps(
        {
            "safe_signal": 1.0,
            "api_token": "tok-secret",
            "raw_source": "secret source",
            "private_key": PRIVATE_KEY_MARKER,
        }
    )

    exit_code = cli_main.main(
        [
            "model",
            "predict-artifact",
            "local/manifest.json",
            "--features-json",
            features_json,
            "--json",
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    _assert_no_json_leaks(payload, output)
    assert "tok-secret" not in output
    assert "secret source" not in output
    assert PRIVATE_KEY_MARKER not in output


def test_phase_13_agent_json_redacts_raw_diff_streams_and_model_metadata(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow_with_sensitive_artifacts,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--trial", "--json"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    _assert_no_json_leaks(payload, output)
    for forbidden in (
        "raw diff sentinel",
        "secret stdout sentinel",
        "secret stderr sentinel",
        "raw source sentinel",
        "api-token-sentinel",
        PRIVATE_KEY_MARKER,
    ):
        assert forbidden not in output
    assert payload["artifacts"]["diff"] is None
    assert payload["artifacts"]["diff_redacted"] is True
    assert payload["artifacts"]["trial"]["steps"][0]["stdout"] is None
    assert payload["artifacts"]["trial"]["steps"][0]["stderr"] is None
    assert payload["artifacts"]["model"]["patch_application_allowed"] is False


def test_phase_13_tracked_docs_and_manifests_do_not_contain_private_paths() -> None:
    _skip_if_not_git_repo()

    leaked_locations: list[str] = []
    for path_text in _git_ls_files("README.md", "docs", "data/gpuboost/manifests"):
        path = Path(path_text)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in PRIVATE_PATH_MARKERS:
            if marker in text:
                leaked_locations.append(f"{path_text}: {marker}")

    assert leaked_locations == []


def test_phase_13_model_safety_check_reports_guardrails(capsys) -> None:
    exit_code = cli_main.main(["model", "safety-check", "--json"])
    output = capsys.readouterr().out
    payload = json.loads(output)
    result = payload["result"]

    assert exit_code == 0
    assert result["status"] in {"ok", "warning"}
    assert result["patch_application_allowed"] is False
    assert result["generated_dir_ignored"] is True
    assert result["raw_data_ignored"] is True
    assert result["artifact_extensions_ignored"] is True
    assert result["local_db_artifacts_ignored"] is True
    assert result["cache_dirs_ignored"] is True
    assert result["env_secret_patterns_ignored"] is True


def _assert_no_json_leaks(payload: Any, output: str) -> None:
    leaks = find_json_leaks(payload)
    assert leaks == []
    assert "state_dict" not in output
    assert "raw model weights" not in output.lower()


def _skip_if_not_git_repo() -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        pytest.skip("git repository is not available")


def _git_ls_files(*pathspecs: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", *pathspecs],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _gitignore_lines() -> set[str]:
    lines = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def _write_model_artifact_fixture(artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.pt").write_text(
        f"serialized raw model weights {PRIVATE_KEY_MARKER}",
        encoding="utf-8",
    )
    (artifact_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "feature_names": ["features.safe_signal"],
                "categorical_features": [],
                "numeric_features": ["features.safe_signal"],
                "boolean_features": [],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "label_mapping.json").write_text(
        json.dumps({"improved": 0, "regressed": 1}),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text(
        json.dumps({"raw_source": "raw source sentinel"}),
        encoding="utf-8",
    )
    (artifact_dir / "evaluation_report.json").write_text(
        json.dumps({"stdout": "secret stdout sentinel"}),
        encoding="utf-8",
    )
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "training.model_artifact.v1",
                "artifact_type": "mlp_classifier",
                "created_at": "2026-01-01T00:00:00+00:00",
                "model_name": "mlp_classifier",
                "model_format": "torch_state_dict",
                "model_file": "model.pt",
                "feature_spec_file": "feature_spec.json",
                "label_mapping_file": "label_mapping.json",
                "training_config_file": "training_config.json",
                "evaluation_report_file": "evaluation_report.json",
                "input_size": 1,
                "output_size": 2,
                "labels": ["improved", "regressed"],
                "feature_names": ["features.safe_signal"],
                "validation_macro_f1": 0.8,
                "test_macro_f1": 0.76,
                "baseline_macro_f1": 0.7,
                "beats_baseline": True,
                "target_macro_f1": 0.85,
                "target_met": False,
                "warnings": [],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _fake_run_optimize_script_workflow_with_sensitive_artifacts(
    **kwargs,
) -> tuple[AgentRunResult, AgentReport]:
    goal = AgentGoal(
        id="optimize_script",
        kind="optimize_script",
        description="Optimize training script.",
        script_path=kwargs.get("script_path"),
        options={"quick": kwargs.get("quick", True)},
        constraints=["do_not_modify_original_file"],
    )
    plan = AgentPlan(
        id="plan_optimize_script",
        goal=goal,
        actions=[
            AgentAction(
                id="inspect_system",
                name="inspect_system",
                description="Collect system information.",
                required=True,
                status="completed",
            )
        ],
    )
    result = AgentRunResult(
        generated_at="2026-01-01T00:00:00+00:00",
        goal=goal,
        plan=plan,
        status="ok",
        artifacts={
            "diff": "--- a/train.py\n+++ b/train.py\n@@ raw diff sentinel",
            "trial": {
                "status": "passed",
                "patch_applied": True,
                "steps": [
                    {
                        "name": "run_test_command",
                        "status": "passed",
                        "stdout": "secret stdout sentinel",
                        "stderr": "secret stderr sentinel",
                    }
                ],
            },
            "model": {
                "model_available": True,
                "fallback_used": False,
                "status": "ok",
                "predictions": [
                    {
                        "label": "improved",
                        "confidence": 0.91,
                        "metadata": {"probabilities": {"improved": 0.91}},
                    }
                ],
                "metadata": {
                    "provider": "trained_local_model",
                    "api_token": "api-token-sentinel",
                    "private_key": PRIVATE_KEY_MARKER,
                    "raw_source": "raw source sentinel",
                    "patch_application_allowed": False,
                },
            },
        },
    )
    report = AgentReport(
        title="GPUBoost Agent Report",
        status="ok",
        summary="Synthetic safe report.",
        sections=[AgentReportSection(title="Results", items=["completed: 1"])],
    )
    return result, report
