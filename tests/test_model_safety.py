"""Tests for Phase 12 model workflow safety checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

from gpuboost.model.safety import verify_model_workflow_safety


def test_verify_model_workflow_safety_returns_ok_or_warning() -> None:
    result = verify_model_workflow_safety()

    assert result["schema_version"] == "training.model_workflow_safety.v1"
    assert result["status"] in {"ok", "warning"}
    assert result["generated_dir_ignored"] is True
    assert result["artifact_extensions_ignored"] is True
    assert result["raw_data_ignored"] is True
    assert result["provider_patch_application_allowed_false"] is True


def test_no_generated_artifacts_or_model_weights_are_tracked() -> None:
    generated = subprocess.run(
        ["git", "ls-files", "data/gpuboost/generated"],
        check=True,
        capture_output=True,
        text=True,
    )
    weights = subprocess.run(
        [
            "git",
            "ls-files",
            "*.pt",
            "*.pth",
            "*.onnx",
            "*.safetensors",
            "*.pkl",
            "*.joblib",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert generated.stdout.strip() == ""
    assert weights.stdout.strip() == ""


def test_phase_12_release_readiness_doc_exists() -> None:
    text = Path("docs/phase-12-release-readiness.md").read_text(encoding="utf-8")

    assert "Phase 12 Release Readiness" in text
    assert "model predictions are advisory only" in text.lower()
    assert "generated artifacts are not tracked" in text.lower() or (
        "generated artifacts are local files and are not committed" in text.lower()
    )
