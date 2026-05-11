"""Tests for Phase 7.2 temporary trial workspace helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gpuboost.schemas.trial import TrialWorkspace
from gpuboost.trial.workspace import (
    cleanup_trial_workspace,
    create_trial_workspace,
    hash_file,
    verify_original_unchanged,
    workspace_is_separate_from_original,
)


def test_create_trial_workspace_copies_file(tmp_path: Path) -> None:
    original = tmp_path / "train.py"
    original.write_text("print('hello')\n", encoding="utf-8")
    before_hash = hash_file(str(original))

    workspace, steps = create_trial_workspace(str(original), base_dir=str(tmp_path))

    assert Path(workspace.workspace_path).exists()
    assert Path(workspace.trial_file).exists()
    assert Path(workspace.trial_file).read_text(encoding="utf-8") == (
        original.read_text(encoding="utf-8")
    )
    assert verify_original_unchanged(str(original), before_hash) is True
    assert [step.name for step in steps] == ["create_workspace", "copy_source"]
    assert [step.status for step in steps] == ["passed", "passed"]

    cleanup_trial_workspace(workspace)


def test_workspace_paths_are_absolute_and_separate(tmp_path: Path) -> None:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")

    workspace, _steps = create_trial_workspace(str(original), base_dir=str(tmp_path))

    assert Path(workspace.original_file).is_absolute()
    assert Path(workspace.workspace_path).is_absolute()
    assert Path(workspace.trial_file).is_absolute()
    assert workspace_is_separate_from_original(workspace) is True

    cleanup_trial_workspace(workspace)


def test_cleanup_removes_workspace_when_enabled(tmp_path: Path) -> None:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")
    workspace, _steps = create_trial_workspace(
        str(original),
        cleanup_enabled=True,
        base_dir=str(tmp_path),
    )
    workspace_path = Path(workspace.workspace_path)

    step = cleanup_trial_workspace(workspace)

    assert step.name == "cleanup_workspace"
    assert step.status == "passed"
    assert not workspace_path.exists()


def test_cleanup_skips_when_disabled(tmp_path: Path) -> None:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")
    workspace, _steps = create_trial_workspace(
        str(original),
        cleanup_enabled=False,
        base_dir=str(tmp_path),
    )

    step = cleanup_trial_workspace(workspace)

    assert step.name == "cleanup_workspace"
    assert step.status == "skipped"
    assert Path(workspace.workspace_path).exists()

    shutil_step = TrialWorkspace(
        original_file=workspace.original_file,
        workspace_path=workspace.workspace_path,
        trial_file=workspace.trial_file,
        cleanup_enabled=True,
        created_at=workspace.created_at,
    )
    cleanup_trial_workspace(shutil_step)


def test_cleanup_never_deletes_original_file(tmp_path: Path) -> None:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")
    workspace, _steps = create_trial_workspace(str(original), base_dir=str(tmp_path))

    cleanup_trial_workspace(workspace)

    assert original.exists()
    assert original.read_text(encoding="utf-8") == "x = 1\n"


def test_missing_original_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"

    with pytest.raises(FileNotFoundError):
        create_trial_workspace(str(missing), base_dir=str(tmp_path))


def test_directory_original_path_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        create_trial_workspace(str(tmp_path), base_dir=str(tmp_path))


def test_hash_file_returns_stable_hash(tmp_path: Path) -> None:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")
    expected_hash = hashlib.sha256(original.read_bytes()).hexdigest()

    assert hash_file(str(original)) == expected_hash
    assert hash_file(str(original)) == expected_hash


def test_verify_original_unchanged_true_and_false(tmp_path: Path) -> None:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")
    before_hash = hash_file(str(original))

    assert verify_original_unchanged(str(original), before_hash) is True

    original.write_text("x = 2\n", encoding="utf-8")

    assert verify_original_unchanged(str(original), before_hash) is False


def test_create_trial_workspace_supports_base_dir(tmp_path: Path) -> None:
    base_dir = tmp_path / "trial-root"
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")

    workspace, _steps = create_trial_workspace(str(original), base_dir=str(base_dir))

    assert base_dir.exists()
    assert Path(workspace.workspace_path).parent == base_dir.resolve()

    cleanup_trial_workspace(workspace)


def test_trial_step_list_includes_create_workspace_and_copy_source(
    tmp_path: Path,
) -> None:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")

    workspace, steps = create_trial_workspace(str(original), base_dir=str(tmp_path))

    assert [step.name for step in steps] == ["create_workspace", "copy_source"]
    assert all(step.status == "passed" for step in steps)
    assert all(step.started_at for step in steps)
    assert all(step.ended_at for step in steps)

    cleanup_trial_workspace(workspace)


def test_cleanup_safety_check_refuses_original_parent_path(tmp_path: Path) -> None:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")
    workspace = TrialWorkspace(
        original_file=str(original.resolve()),
        workspace_path=str(tmp_path.resolve()),
        trial_file=str((tmp_path / "trial.py").resolve()),
        cleanup_enabled=True,
        created_at="2026-01-01T00:00:00+00:00",
    )

    step = cleanup_trial_workspace(workspace)

    assert step.name == "cleanup_workspace"
    assert step.status == "failed"
    assert original.exists()
    assert tmp_path.exists()
