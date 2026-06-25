"""Release checkpoint checks for Phase 15C."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib

import gpuboost
from gpuboost.cli import main as cli_main


def test_cli_version_outputs_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["--version"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 0
    assert captured.out.strip() == f"gpuboost {gpuboost.__version__}"
    assert captured.err == ""


def test_version_metadata_uses_package_version_source() -> None:
    init_text = Path("gpuboost/__init__.py").read_text(encoding="utf-8")
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    version_match = re.search(r'__version__ = "([^"]+)"', init_text)

    assert version_match is not None
    assert version_match.group(1) == "0.2.0"
    assert gpuboost.__version__ == version_match.group(1)
    assert 'dynamic = ["version"]' in pyproject_text
    assert 'path = "gpuboost/__init__.py"' in pyproject_text
    assert 'version = "0.2.0"' not in pyproject_text


def test_release_notes_exist_and_cover_phase_15c_topics() -> None:
    notes_path = Path("docs/release-notes.md")
    text = notes_path.read_text(encoding="utf-8")
    lowered = text.lower()

    assert notes_path.exists()
    for phrase in [
        "0.2.0",
        "human-approved agentic optimization",
        "immutable plan digest",
        "original source hash",
        "partial approvals",
        "repository root",
        "cross-repository",
        "path traversal",
        "backup",
        "benchmark acceptance",
        "automatic rollback",
        "explicit rollback",
        "model-originated patching remains forbidden",
        "safety policy",
        "0.1.2 packaging fix",
        "numpy",
        "lightweight base install",
        "kept pytorch optional",
        "phase 11 dataset readiness",
        "phase 12 local model workflow",
        "phase 13 testing and release hardening",
        "phase 14 real-world validation demos",
        "phase 15 final polish",
        "advisory-only",
        "does not apply patches without approval",
        "generated artifacts are ignored",
        "no generated artifacts included",
        "python -m gpuboost --version",
        "python -m ruff check .",
        "python -m pytest",
    ]:
        assert phrase in lowered


def test_package_metadata_keeps_lightweight_base_and_optional_ml_extras() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    optional = project["optional-dependencies"]

    assert project["name"] == "gpuboost"
    assert project["requires-python"] == ">=3.9"
    assert project["license"]["text"] == "MIT"
    assert project["urls"]["Repository"] == "https://github.com/its-arnavtech/GPU-Boost"
    assert project["urls"]["Issues"] == "https://github.com/its-arnavtech/GPU-Boost/issues"
    assert set(project["dependencies"]) == {"psutil", "nvidia-ml-py", "rich"}
    assert set(optional["benchmark"]) == {"torch", "numpy"}
    assert set(optional["model"]) == {"torch", "numpy"}
    assert set(optional["all"]) == {"torch", "numpy"}


def test_readme_links_to_release_notes() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "docs/release-notes.md" in text
    assert "python -m gpuboost --version" in text
    assert "GPUBoost 0.2.0" in text
    assert "1060 passed, 1 skipped" in text
