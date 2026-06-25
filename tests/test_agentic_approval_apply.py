"""Tests for human-approved agentic source application."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gpuboost.agent.approved_apply import (
    apply_approved_optimization_run,
    approve_optimization_run,
    prepare_optimization_run,
    rollback_optimization_run,
    short_plan_id,
)
from gpuboost.cli import main as cli_main
from gpuboost.schemas.agentic import ApprovalState, OptimizationLifecycleStatus


SAMPLE_SOURCE = """import torch
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, num_workers=0)
for inputs, targets in loader:
    inputs = inputs.to("cuda")
    targets = targets.to("cuda")
    outputs = model(inputs)
"""


def test_prepare_persists_plan_without_mutating_source(tmp_path: Path) -> None:
    script = _write_training_script(tmp_path)

    run = prepare_optimization_run("train.py", repo_root=str(tmp_path))

    assert script.read_text(encoding="utf-8") == SAMPLE_SOURCE
    assert run.lifecycle_status == OptimizationLifecycleStatus.AWAITING_APPROVAL
    assert run.approval_state == ApprovalState.AWAITING_APPROVAL
    assert run.plan_digest
    assert run.original_file_hash
    assert "pin_memory=True" in run.generated_diff
    assert (tmp_path / ".gpuboost" / "runs" / f"{run.run_id}.json").is_file()


def test_apply_requires_approval_and_then_creates_backup(tmp_path: Path) -> None:
    script = _write_training_script(tmp_path)
    run = prepare_optimization_run("train.py", repo_root=str(tmp_path))

    with pytest.raises(ValueError, match="persisted approval"):
        apply_approved_optimization_run(run.run_id, repo_root=str(tmp_path))
    with pytest.raises(ValueError, match="Confirmation phrase"):
        approve_optimization_run(
            run.run_id,
            approved_by="tester",
            confirmation_phrase="APPLY wrong",
            repo_root=str(tmp_path),
        )

    approved = approve_optimization_run(
        run.run_id,
        approved_by="tester",
        confirmation_phrase=_confirmation_phrase(run),
        repo_root=str(tmp_path),
    )
    assert script.read_text(encoding="utf-8") == SAMPLE_SOURCE

    applied = apply_approved_optimization_run(approved.run_id, repo_root=str(tmp_path))

    source = script.read_text(encoding="utf-8")
    assert applied.lifecycle_status == OptimizationLifecycleStatus.COMPLETED
    assert applied.final_status == "completed"
    assert "pin_memory=True" in source
    assert "torch.backends.cudnn.benchmark = True" in source
    assert applied.pre_application_backup_path is not None
    assert Path(applied.pre_application_backup_path).is_file()


def test_partial_approval_applies_only_selected_action(tmp_path: Path) -> None:
    script = _write_training_script(tmp_path)
    run = prepare_optimization_run(
        "train.py",
        repo_root=str(tmp_path),
        action_ids=["patch_cudnn_benchmark_missing"],
    )
    approved = approve_optimization_run(
        run.run_id,
        approved_by="tester",
        confirmation_phrase=_confirmation_phrase(run),
        repo_root=str(tmp_path),
    )

    applied = apply_approved_optimization_run(approved.run_id, repo_root=str(tmp_path))

    source = script.read_text(encoding="utf-8")
    assert applied.final_status == "completed"
    assert "torch.backends.cudnn.benchmark = True" in source
    assert "pin_memory=True" not in source
    assert "num_workers=0" in source


def test_source_change_invalidates_approval(tmp_path: Path) -> None:
    script = _write_training_script(tmp_path)
    run = prepare_optimization_run("train.py", repo_root=str(tmp_path))
    script.write_text(SAMPLE_SOURCE + "\n# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Target file changed after planning"):
        approve_optimization_run(
            run.run_id,
            approved_by="tester",
            confirmation_phrase=_confirmation_phrase(run),
            repo_root=str(tmp_path),
        )


def test_plan_digest_tampering_invalidates_approval(tmp_path: Path) -> None:
    _write_training_script(tmp_path)
    run = prepare_optimization_run("train.py", repo_root=str(tmp_path))
    path = tmp_path / ".gpuboost" / "runs" / f"{run.run_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["plan_digest"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Plan digest mismatch"):
        approve_optimization_run(
            run.run_id,
            approved_by="tester",
            confirmation_phrase=_confirmation_phrase(run),
            repo_root=str(tmp_path),
        )


def test_widened_action_ids_after_approval_are_rejected(tmp_path: Path) -> None:
    script = _write_training_script(tmp_path)
    run = prepare_optimization_run("train.py", repo_root=str(tmp_path))
    approve_optimization_run(
        run.run_id,
        approved_by="tester",
        confirmation_phrase=_confirmation_phrase(run),
        approved_action_ids=["patch_cudnn_benchmark_missing"],
        repo_root=str(tmp_path),
    )
    # Widen the mutable run field beyond the immutable approval record.
    path = tmp_path / ".gpuboost" / "runs" / f"{run.run_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["approved_action_ids"] = [
        action["action_id"] for action in data["proposed_actions"]
    ]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match the approval record"):
        apply_approved_optimization_run(run.run_id, repo_root=str(tmp_path))

    assert script.read_text(encoding="utf-8") == SAMPLE_SOURCE


def test_validation_failure_rolls_back_source(tmp_path: Path) -> None:
    script = _write_training_script(tmp_path)
    run = prepare_optimization_run("train.py", repo_root=str(tmp_path))
    approved = approve_optimization_run(
        run.run_id,
        approved_by="tester",
        confirmation_phrase=_confirmation_phrase(run),
        repo_root=str(tmp_path),
    )

    applied = apply_approved_optimization_run(
        approved.run_id,
        repo_root=str(tmp_path),
        validation_command=f'"{sys.executable}" -c "import sys; sys.exit(1)"',
    )

    assert applied.lifecycle_status == OptimizationLifecycleStatus.ROLLED_BACK
    assert applied.final_status == "rolled_back"
    assert applied.rollback_result is not None
    assert applied.rollback_result["status"] == "passed"
    assert script.read_text(encoding="utf-8") == SAMPLE_SOURCE


def test_minimum_speedup_policy_accepts_json_benchmark_metrics(
    tmp_path: Path,
) -> None:
    script = _write_training_script(tmp_path)
    benchmark = tmp_path / "bench.py"
    benchmark.write_text(
        'import json\nprint(json.dumps({"speedup_percent": 3.5}))\n',
        encoding="utf-8",
    )
    benchmark_command = f'"{sys.executable}" bench.py'
    run = prepare_optimization_run(
        "train.py",
        repo_root=str(tmp_path),
        acceptance_policy="minimum-speedup",
        min_speedup_percent=2.0,
        benchmark_command=benchmark_command,
    )
    approved = approve_optimization_run(
        run.run_id,
        approved_by="tester",
        confirmation_phrase=_confirmation_phrase(run),
        repo_root=str(tmp_path),
    )

    applied = apply_approved_optimization_run(
        approved.run_id,
        repo_root=str(tmp_path),
        benchmark_command=benchmark_command,
        acceptance_policy="minimum-speedup",
        min_speedup_percent=2.0,
    )

    assert applied.final_status == "completed"
    assert applied.benchmark_result is not None
    assert applied.benchmark_result["status"] == "passed"
    assert "pin_memory=True" in script.read_text(encoding="utf-8")


def test_rollback_command_restores_applied_source(tmp_path: Path) -> None:
    script = _write_training_script(tmp_path)
    run = prepare_optimization_run("train.py", repo_root=str(tmp_path))
    approved = approve_optimization_run(
        run.run_id,
        approved_by="tester",
        confirmation_phrase=_confirmation_phrase(run),
        repo_root=str(tmp_path),
    )
    applied = apply_approved_optimization_run(approved.run_id, repo_root=str(tmp_path))

    rolled_back = rollback_optimization_run(applied.run_id, repo_root=str(tmp_path))

    assert rolled_back.lifecycle_status == OptimizationLifecycleStatus.ROLLED_BACK
    assert script.read_text(encoding="utf-8") == SAMPLE_SOURCE


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text(SAMPLE_SOURCE, encoding="utf-8")

    with pytest.raises(ValueError, match="escapes repository root"):
        prepare_optimization_run(str(outside), repo_root=str(repo))


def test_relative_parent_traversal_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "outside.py").write_text(SAMPLE_SOURCE, encoding="utf-8")

    with pytest.raises(ValueError, match="escapes repository root"):
        prepare_optimization_run("../outside.py", repo_root=str(repo))


def test_backup_dir_escape_is_rejected(tmp_path: Path) -> None:
    script = _write_training_script(tmp_path)
    run = prepare_optimization_run("train.py", repo_root=str(tmp_path))
    approved = approve_optimization_run(
        run.run_id,
        approved_by="tester",
        confirmation_phrase=_confirmation_phrase(run),
        repo_root=str(tmp_path),
    )

    with pytest.raises(ValueError, match="escapes repository root"):
        apply_approved_optimization_run(
            approved.run_id,
            repo_root=str(tmp_path),
            backup_dir="../escape-backups",
        )

    assert script.read_text(encoding="utf-8") == SAMPLE_SOURCE


def test_run_metadata_for_other_repository_is_rejected(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "train.py").write_text(SAMPLE_SOURCE, encoding="utf-8")
    (repo_b / "train.py").write_text(SAMPLE_SOURCE, encoding="utf-8")

    run = prepare_optimization_run("train.py", repo_root=str(repo_a))
    # Place repo_a's run record under repo_b without changing its stored root.
    source_record = (repo_a / ".gpuboost" / "runs" / f"{run.run_id}.json").read_text(
        encoding="utf-8"
    )
    destination = repo_b / ".gpuboost" / "runs"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{run.run_id}.json").write_text(source_record, encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the active repository"):
        approve_optimization_run(
            run.run_id,
            approved_by="tester",
            confirmation_phrase=_confirmation_phrase(run),
            repo_root=str(repo_b),
        )
    assert (repo_a / "train.py").read_text(encoding="utf-8") == SAMPLE_SOURCE


def test_approved_run_metadata_for_other_repository_cannot_apply(
    tmp_path: Path,
) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "train.py").write_text(SAMPLE_SOURCE, encoding="utf-8")
    (repo_b / "train.py").write_text(SAMPLE_SOURCE, encoding="utf-8")

    run = prepare_optimization_run("train.py", repo_root=str(repo_a))
    approved = approve_optimization_run(
        run.run_id,
        approved_by="tester",
        confirmation_phrase=_confirmation_phrase(run),
        repo_root=str(repo_a),
    )
    source_record = (
        repo_a / ".gpuboost" / "runs" / f"{approved.run_id}.json"
    ).read_text(encoding="utf-8")
    destination = repo_b / ".gpuboost" / "runs"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{approved.run_id}.json").write_text(
        source_record,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the active repository"):
        apply_approved_optimization_run(approved.run_id, repo_root=str(repo_b))

    assert (repo_b / "train.py").read_text(encoding="utf-8") == SAMPLE_SOURCE
    assert not (repo_b / ".gpuboost" / "backups").exists()


def test_symlink_target_escape_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text(SAMPLE_SOURCE, encoding="utf-8")
    link = repo / "linked.py"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as error:  # pragma: no cover - env dependent
        pytest.skip(f"symlink creation not permitted on this platform: {error}")

    with pytest.raises(ValueError, match="repository root"):
        prepare_optimization_run("linked.py", repo_root=str(repo))


def test_cli_prepare_approve_apply_json_flow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _write_training_script(tmp_path)

    prepare_exit = cli_main.main(
        [
            "agent",
            "optimize",
            "train.py",
            "--prepare",
            "--repo-root",
            str(tmp_path),
            "--json",
        ]
    )
    prepare_out = capsys.readouterr()
    payload = json.loads(prepare_out.out)
    run_id = payload["run"]["run_id"]
    phrase = payload["confirmation_phrase"]

    approve_exit = cli_main.main(
        [
            "agent",
            "approve",
            run_id,
            "--repo-root",
            str(tmp_path),
            "--confirm",
            phrase,
            "--json",
        ]
    )
    approve_out = capsys.readouterr()

    apply_exit = cli_main.main(
        [
            "agent",
            "apply",
            run_id,
            "--repo-root",
            str(tmp_path),
            "--json",
        ]
    )
    apply_out = capsys.readouterr()
    apply_payload = json.loads(apply_out.out)

    assert prepare_exit == 0
    assert approve_exit == 0
    assert apply_exit == 0
    assert json.loads(approve_out.out)["run"]["approval_state"] == "approved"
    assert apply_payload["run"]["final_status"] == "completed"
    assert "pin_memory=True" in script.read_text(encoding="utf-8")


def _write_training_script(root: Path) -> Path:
    script = root / "train.py"
    script.write_text(SAMPLE_SOURCE, encoding="utf-8")
    return script


def _confirmation_phrase(run: Any) -> str:
    return f"APPLY {short_plan_id(run.plan_id)}"
