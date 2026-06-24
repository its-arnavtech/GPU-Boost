"""Release checkpoint checks for Phase 15C."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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
    assert version_match.group(1) == "0.1.1"
    assert gpuboost.__version__ == version_match.group(1)
    assert 'dynamic = ["version"]' in pyproject_text
    assert 'path = "gpuboost/__init__.py"' in pyproject_text
    assert 'version = "0.1.1"' not in pyproject_text


def test_release_notes_exist_and_cover_phase_15c_topics() -> None:
    notes_path = Path("docs/release-notes.md")
    text = notes_path.read_text(encoding="utf-8")
    lowered = text.lower()

    assert notes_path.exists()
    for phrase in [
        "phase 11 dataset readiness",
        "phase 12 local model workflow",
        "phase 13 testing and release hardening",
        "phase 14 real-world validation demos",
        "phase 15 final polish",
        "advisory-only",
        "no automatic patching",
        "generated artifacts are ignored",
        "no generated artifacts included",
        "python -m gpuboost --version",
        "python -m ruff check .",
        "python -m pytest",
    ]:
        assert phrase in lowered


def test_readme_links_to_release_notes() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "docs/release-notes.md" in text
    assert "python -m gpuboost --version" in text
