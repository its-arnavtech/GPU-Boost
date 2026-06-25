"""Post-release audit guardrails for repository hygiene and README claims."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


README = Path("README.md")
AUDIT_DOC = Path("docs/post-release-audit.md")
GITIGNORE = Path(".gitignore")


def test_gitignore_covers_post_release_sensitive_and_generated_paths() -> None:
    lines = _gitignore_lines()

    for pattern in (
        ".env",
        ".env.*",
        "!.env.example",
        "data/gpuboost/raw/",
        "data/gpuboost/generated/",
        "run-output/",
        "artifacts/",
        "*.sqlite3",
        "*.safetensors",
        ".pypirc",
        ".pyright/",
        "pkg-remediation-venv/",
    ):
        assert pattern in lines


def test_gitignore_keeps_public_templates_and_fixtures_trackable() -> None:
    _skip_if_git_unavailable()

    assert _check_ignore(".env") == 0
    assert _check_ignore(".env.local") == 0
    assert _check_ignore("run-output/demo.json") == 0
    assert _check_ignore("artifacts/model.safetensors") == 0
    assert _check_ignore(".env.example") == 1
    assert _check_ignore("examples/outcome_collection/pairs.example.json") == 1
    assert _check_ignore("data/gpuboost/manifests/training_readiness_report.json") == 1


def test_readme_documents_current_release_and_install_commands() -> None:
    text = README.read_text(encoding="utf-8")

    assert "GPUBoost 0.1.2" in text
    assert "pip install gpuboost" in text
    assert 'pip install "gpuboost[all]"' in text
    assert 'pip install "gpuboost[benchmark]"' in text
    assert 'pip install "gpuboost[model]"' in text
    assert "https://pypi.org/project/gpuboost/" in text
    assert "https://github.com/its-arnavtech/GPU-Boost" in text


def test_readme_preserves_safety_model_claims() -> None:
    text = README.read_text(encoding="utf-8").lower()

    for phrase in (
        "no automatic patch application",
        "model output is advisory",
        "deterministic checks remain authoritative",
        "patch_application_allowed=false",
        "no guaranteed speedups",
        "gpu/cuda is optional",
    ):
        assert phrase in text


def test_readme_links_current_post_release_audit_evidence() -> None:
    text = README.read_text(encoding="utf-8")

    assert "docs/post-release-audit.md" in text
    assert "Validated on one local machine" in text
    assert "1044 passed" in text
    assert AUDIT_DOC.exists()


def _gitignore_lines() -> set[str]:
    return {
        line.strip()
        for line in GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _skip_if_git_unavailable() -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("git repository is not available")


def _check_ignore(path: str) -> int:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        capture_output=True,
        text=True,
    )
    return result.returncode
