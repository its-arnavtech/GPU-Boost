"""Installed-package-style CLI checks for lightweight commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from gpuboost import __version__


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(
    *args: str,
    cwd: Path,
    pythonpath_prefix: list[Path] | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    pythonpath_entries = [str(path) for path in (pythonpath_prefix or [])]
    pythonpath_entries.append(str(PROJECT_ROOT))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-m", "gpuboost", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_torch_blocker(tmp_path: Path) -> Path:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            [
                "import builtins",
                "import os",
                "",
                "_real_import = builtins.__import__",
                "_log_path = os.environ.get('GPUBOOST_TORCH_IMPORT_LOG')",
                "",
                "def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):",
                "    if name == 'torch' or name.startswith('torch.'):",
                "        if _log_path:",
                "            with open(_log_path, 'a', encoding='utf-8') as handle:",
                "                handle.write(name + '\\n')",
                "        raise RuntimeError('torch import blocked by test')",
                "    return _real_import(name, globals, locals, fromlist, level)",
                "",
                "builtins.__import__ = _guarded_import",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path


def _doctor_check(payload: dict[str, object], name: str) -> dict[str, object]:
    checks = payload.get("checks")
    assert isinstance(checks, list)
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return check
    raise AssertionError(f"Missing doctor check: {name}")


@pytest.mark.parametrize(
    ("args", "expected_output"),
    [
        (["--help"], "usage: gpuboost"),
        (["--version"], f"gpuboost {__version__}"),
        (["compare", "--help"], "usage: gpuboost compare"),
        (["agent", "--help"], "Run GPUBoost agent workflows."),
        (["demo", "--help"], "usage: gpuboost demo"),
    ],
)
def test_lightweight_cli_commands_do_not_import_torch(
    tmp_path: Path,
    args: list[str],
    expected_output: str,
) -> None:
    log_path = tmp_path / "torch-imports.log"
    blocker_dir = _write_torch_blocker(tmp_path)

    completed = _run_cli(
        *args,
        cwd=tmp_path,
        pythonpath_prefix=[blocker_dir],
        extra_env={"GPUBOOST_TORCH_IMPORT_LOG": str(log_path)},
    )

    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, combined_output
    assert expected_output in completed.stdout
    assert "Failed to initialize NumPy" not in combined_output
    assert "torch import blocked by test" not in combined_output
    assert not log_path.exists() or log_path.read_text(encoding="utf-8") == ""


def test_doctor_json_from_unrelated_directory_skips_repo_policy_checks(
    tmp_path: Path,
) -> None:
    completed = _run_cli("doctor", "--json", cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    gitignore_check = _doctor_check(payload, "gitignore_generated_artifacts")

    assert payload["schema_version"] == "setup.doctor.v1"
    assert payload["status"] in {"ok", "warning"}
    assert gitignore_check["status"] == "skipped"
    assert gitignore_check["applicable"] is False


def test_doctor_json_explicit_repo_root_passes_repo_policy_checks(
    tmp_path: Path,
) -> None:
    completed = _run_cli(
        "doctor",
        "--json",
        "--repo-root",
        str(PROJECT_ROOT),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    gitignore_check = _doctor_check(payload, "gitignore_generated_artifacts")

    assert payload["schema_version"] == "setup.doctor.v1"
    assert gitignore_check["status"] == "passed"
    assert gitignore_check["repo_root"] == str(PROJECT_ROOT)


def test_doctor_json_invalid_repo_root_returns_non_blocking_warning(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-repo"
    completed = _run_cli(
        "doctor",
        "--json",
        "--repo-root",
        str(missing_root),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    gitignore_check = _doctor_check(payload, "gitignore_generated_artifacts")

    assert payload["schema_version"] == "setup.doctor.v1"
    assert payload["status"] == "warning"
    assert gitignore_check["status"] == "warning"
    assert "does not exist" in gitignore_check["message"]


def test_model_safety_check_from_unrelated_directory_marks_repo_checks_skipped(
    tmp_path: Path,
) -> None:
    completed = _run_cli("model", "safety-check", "--json", cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    result = payload["result"]

    assert payload["schema_version"] == "training.model_workflow_safety.v1"
    assert payload["command"] == "model safety-check"
    assert result["status"] == "warning"
    assert result["patch_application_allowed"] is False
    assert result["provider_patch_application_allowed_false"] is True
    assert result["no_default_artifact_path_required"] is True
    assert result["generated_dir_ignored"] is None
    assert "generated_dir_ignored" in result["skipped_checks"]


def test_model_safety_check_explicit_repo_root_runs_repository_checks(
    tmp_path: Path,
) -> None:
    completed = _run_cli(
        "model",
        "safety-check",
        "--json",
        "--repo-root",
        str(PROJECT_ROOT),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    result = payload["result"]

    assert result["status"] in {"ok", "warning"}
    assert result["generated_dir_ignored"] is True
    assert result["artifact_extensions_ignored"] is True
    assert result["raw_data_ignored"] is True


def test_model_safety_check_invalid_repo_root_is_non_blocking_warning(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-repo"
    completed = _run_cli(
        "model",
        "safety-check",
        "--json",
        "--repo-root",
        str(missing_root),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    result = payload["result"]

    assert result["status"] == "warning"
    assert result["generated_dir_ignored"] is None
    assert "does not exist" in "\n".join(result["warnings"])
