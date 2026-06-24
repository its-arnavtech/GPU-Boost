"""Phase 15A setup/install/dev-environment polish checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gpuboost.cli import main as cli_main


def test_setup_doc_exists_and_covers_new_user_workflow() -> None:
    text = Path("docs/setup.md").read_text(encoding="utf-8")
    lowered = text.lower()

    for phrase in [
        "prerequisites",
        "windows powershell",
        "python -m pip install -e \".[dev]\"",
        "python -m gpuboost doctor",
        "python -m ruff check .",
        "python -m pytest",
        "cuda is not required",
        "data/gpuboost/generated/",
        "do not commit",
    ]:
        assert phrase in lowered


def test_readme_links_setup_doc_and_lists_setup_validation_commands() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    for phrase in [
        "docs/setup.md",
        "python -m gpuboost doctor",
        "python -m ruff check .",
        "python -m pytest",
    ]:
        assert phrase in text


def test_pyproject_metadata_is_reasonable_for_dev_install() -> None:
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    project = pyproject["project"]
    assert project["name"] == "gpuboost"
    assert "version" in project or "version" in project.get("dynamic", [])
    if "version" in project.get("dynamic", []):
        assert pyproject["tool"]["hatch"]["version"]["path"] == "gpuboost/__init__.py"
    assert project["requires-python"] == ">=3.9"
    assert "torch" not in project["dependencies"]
    assert "psutil" in project["dependencies"]
    assert "rich" in project["dependencies"]
    assert "benchmark" in project["optional-dependencies"]
    assert "model" in project["optional-dependencies"]
    assert "all" in project["optional-dependencies"]
    assert "torch" in project["optional-dependencies"]["benchmark"]
    assert "torch" in project["optional-dependencies"]["model"]
    assert "torch" in project["optional-dependencies"]["all"]
    assert "dev" in project["optional-dependencies"]
    assert "pytest" in project["optional-dependencies"]["dev"]
    assert "ruff" in project["optional-dependencies"]["dev"]
    assert pyproject["project"]["scripts"]["gpuboost"] == "gpuboost.cli.main:main"
    assert pyproject["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert pyproject["tool"]["ruff"]["line-length"] == 88


def test_doctor_cli_json_is_lightweight_and_cuda_optional(capsys) -> None:
    exit_code = cli_main.main(["doctor", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["schema_version"] == "setup.doctor.v1"
    assert payload["status"] in {"ok", "warning"}
    assert payload["cuda_required"] is False
    assert any(
        check["name"] == "python_version" and check["status"] == "passed"
        for check in payload["checks"]
    )
    assert any(check["name"] == "optional:torch" for check in payload["checks"])
    assert any(
        check["name"] == "gitignore_generated_artifacts"
        and check["status"] == "passed"
        for check in payload["checks"]
    )
    assert captured.err == ""


def test_no_generated_data_or_model_artifacts_are_tracked() -> None:
    generated = _git_ls_files("data/gpuboost/generated")
    artifact_files = _git_ls_files(
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.safetensors",
        "*.pkl",
        "*.joblib",
    )

    assert generated == []
    assert artifact_files == []


def _git_ls_files(*patterns: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", *patterns],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_gitignore_check_flags_commented_pattern(monkeypatch) -> None:
    from gpuboost.cli import main as cli_main

    lines = [
        "data/gpuboost/generated/",
        "data/gpuboost/raw/",
        "# *.pt",  # commented out -> must be treated as missing
        "*.pth",
        "*.safetensors",
        "*.onnx",
        "*.pkl",
        "*.joblib",
        "*.db",
        "*.sqlite",
        "*.sqlite3",
    ]
    monkeypatch.setattr(
        cli_main, "_read_optional_text", lambda path: "\n".join(lines) + "\n"
    )

    result = cli_main._check_gitignore_safety()

    assert result["status"] == "failed"
    assert "*.pt" in result["message"]


def test_gitignore_check_passes_when_all_patterns_active(monkeypatch) -> None:
    from gpuboost.cli import main as cli_main

    lines = [
        "data/gpuboost/generated/",
        "data/gpuboost/raw/",
        "*.pt",
        "*.pth",
        "*.safetensors",
        "*.onnx",
        "*.pkl",
        "*.joblib",
        "*.db",
        "*.sqlite",
        "*.sqlite3",
    ]
    monkeypatch.setattr(
        cli_main, "_read_optional_text", lambda path: "\n".join(lines) + "\n"
    )

    assert cli_main._check_gitignore_safety()["status"] == "passed"
