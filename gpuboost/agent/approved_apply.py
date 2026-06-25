"""Human-approved source application for deterministic GPUBoost patch plans."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import py_compile
import shlex
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpuboost import __version__
from gpuboost.code_analysis.runner import analyze_python_file
from gpuboost.patching.diff import (
    apply_patch_edits_to_text,
    generate_patch_plan_diff,
)
from gpuboost.patching.planner import create_patch_plan_from_analysis
from gpuboost.schemas.agentic import (
    AGENTIC_OPTIMIZATION_SCHEMA_VERSION,
    AcceptancePolicy,
    AgenticOptimizationRun,
    ApprovalState,
    OptimizationApproval,
    OptimizationLifecycleStatus,
    ProposedEdit,
    RiskLevel,
    create_timestamp,
)
from gpuboost.schemas.patch_plan import PatchEdit, PatchPlan, PatchSuggestion
from gpuboost.trial.engine import run_patch_trial


DEFAULT_RUNS_DIR = ".gpuboost/runs"
DEFAULT_BACKUP_DIR = ".gpuboost/backups"
CONFIRMATION_PREFIX = "APPLY"


@dataclass(frozen=True, slots=True)
class ProjectTarget:
    """Resolved repository root and target path."""

    repo_root: Path
    target_path: Path
    relative_target: str


def prepare_optimization_run(
    script_path: str,
    *,
    repo_root: str | None = None,
    trial: bool = False,
    action_ids: list[str] | None = None,
    exclude_action_ids: list[str] | None = None,
    acceptance_policy: str = AcceptancePolicy.VALIDATION_ONLY.value,
    min_speedup_percent: float = 0.0,
    max_regression_percent: float = 0.0,
    benchmark_command: str | None = None,
    runs_dir: str | None = None,
) -> AgenticOptimizationRun:
    """Analyze, plan, diff, optionally trial, and persist an approval-gated run."""

    target = resolve_project_target(script_path, repo_root=repo_root)
    source_text = target.target_path.read_text(encoding="utf-8")
    original_hash = sha256_file(target.target_path)
    analysis = analyze_python_file(str(target.target_path))
    if analysis.status != "ok":
        raise ValueError(analysis.error or "Code analysis failed.")

    patch_plan = create_patch_plan_from_analysis(source_text, analysis)
    selected_plan = select_patch_plan_actions(
        patch_plan,
        include_action_ids=action_ids,
        exclude_action_ids=exclude_action_ids,
    )
    diff, diff_warnings = generate_patch_plan_diff(source_text, selected_plan)
    if selected_plan.status != "ok":
        raise ValueError(selected_plan.error or "Patch plan failed.")
    if not selected_plan.suggestions:
        raise ValueError("No deterministic patch suggestions were generated.")
    if not diff:
        raise ValueError("Patch plan did not produce a source diff.")

    run_id = _new_run_id()
    plan_id = _new_plan_id()
    policy_settings = {
        "acceptance_policy": acceptance_policy,
        "min_speedup_percent": min_speedup_percent,
        "max_regression_percent": max_regression_percent,
        "benchmark_command": benchmark_command,
    }
    plan_digest = compute_plan_digest(
        target_path=target.relative_target,
        original_hash=original_hash,
        patch_plan=selected_plan,
        diff=diff,
        policy_settings=policy_settings,
    )
    proposed_actions = _build_proposed_actions(selected_plan)
    proposed_edits = _build_proposed_edits(selected_plan)

    trial_result: dict[str, Any] | None = None
    lifecycle = OptimizationLifecycleStatus.AWAITING_APPROVAL
    if trial:
        trial_run = run_patch_trial(
            original_file=str(target.target_path),
            patch_plan=selected_plan,
            force_python_syntax=True,
        )
        trial_result = trial_run.to_dict()
        lifecycle = (
            OptimizationLifecycleStatus.TRIALED
            if trial_run.status == "passed"
            else OptimizationLifecycleStatus.FAILED
        )
        if trial_run.status in {"failed", "error"}:
            lifecycle = OptimizationLifecycleStatus.FAILED

    run = AgenticOptimizationRun(
        run_id=run_id,
        target_repository_root=str(target.repo_root),
        target_file=target.relative_target,
        original_file_hash=original_hash,
        plan_id=plan_id,
        plan_digest=plan_digest,
        created_at=create_timestamp(),
        tool_version=__version__,
        proposed_actions=proposed_actions,
        proposed_edits=proposed_edits,
        generated_diff=diff,
        patch_plan=selected_plan.to_dict(),
        lifecycle_status=lifecycle,
        approval_state=ApprovalState.AWAITING_APPROVAL,
        trial_result=trial_result,
        benchmark_result={
            "status": "not_run",
            "policy": policy_settings,
            "message": "Benchmark execution is optional and was not requested.",
        },
        warnings=list(diff_warnings),
    )
    if lifecycle == OptimizationLifecycleStatus.FAILED:
        run.approval_state = ApprovalState.NOT_REQUESTED
        run.final_status = "failed"
        run.error = "Trial validation failed."
    save_run(run, runs_dir=runs_dir)
    return run


def approve_optimization_run(
    run_id: str,
    *,
    approved_by: str,
    confirmation_phrase: str,
    approved_action_ids: list[str] | None = None,
    repo_root: str | None = None,
    runs_dir: str | None = None,
) -> AgenticOptimizationRun:
    """Create a persisted approval object for a prepared run."""

    run = load_run(run_id, repo_root=repo_root, runs_dir=runs_dir)
    if run.lifecycle_status in {
        OptimizationLifecycleStatus.FAILED,
        OptimizationLifecycleStatus.REJECTED,
        OptimizationLifecycleStatus.ROLLED_BACK,
    }:
        raise ValueError(f"Run {run_id} is not approvable: {run.lifecycle_status.value}")
    expected_phrase = f"{CONFIRMATION_PREFIX} {short_plan_id(run.plan_id)}"
    if confirmation_phrase.strip() != expected_phrase:
        raise ValueError(f"Confirmation phrase must be exactly: {expected_phrase}")
    validate_current_source_matches_run(run)
    validate_plan_digest(run)

    available_action_ids = [action["action_id"] for action in run.proposed_actions]
    selected = approved_action_ids or available_action_ids
    invalid = sorted(set(selected) - set(available_action_ids))
    if invalid:
        raise ValueError(f"Unknown action IDs: {', '.join(invalid)}")
    if not selected:
        raise ValueError("At least one action must be approved.")

    approval = OptimizationApproval(
        run_id=run.run_id,
        plan_id=run.plan_id,
        plan_digest=run.plan_digest,
        approved_action_ids=tuple(selected),
        approved_by=approved_by,
        approved_at=create_timestamp(),
        target_file_hash=run.original_file_hash,
    )
    run.approval = approval.to_dict()
    run.approval_state = ApprovalState.APPROVED
    run.approved_action_ids = list(selected)
    run.approver = approved_by
    run.approver_confirmation_timestamp = approval.approved_at
    run.lifecycle_status = OptimizationLifecycleStatus.APPROVED
    run.final_status = "approved"
    save_run(run, runs_dir=runs_dir)
    return run


def reject_optimization_run(
    run_id: str,
    *,
    repo_root: str | None = None,
    runs_dir: str | None = None,
) -> AgenticOptimizationRun:
    """Reject a prepared run and prevent later application."""

    run = load_run(run_id, repo_root=repo_root, runs_dir=runs_dir)
    run.approval_state = ApprovalState.REJECTED
    run.lifecycle_status = OptimizationLifecycleStatus.REJECTED
    run.final_status = "rejected"
    save_run(run, runs_dir=runs_dir)
    return run


def apply_approved_optimization_run(
    run_id: str,
    *,
    repo_root: str | None = None,
    runs_dir: str | None = None,
    backup_dir: str | None = None,
    dry_run: bool = False,
    validation_command: str | None = None,
    test_command: str | None = None,
    benchmark_command: str | None = None,
    validation_timeout_sec: int = 300,
    acceptance_policy: str | None = None,
    min_speedup_percent: float | None = None,
    max_regression_percent: float | None = None,
) -> AgenticOptimizationRun:
    """Apply an approved deterministic plan to the real source file."""

    run = load_run(run_id, repo_root=repo_root, runs_dir=runs_dir)
    approval = _require_valid_approval(run)
    validate_current_source_matches_run(run)
    validate_plan_digest(run)
    policy_settings = _approved_policy_settings(
        run,
        acceptance_policy=acceptance_policy,
        min_speedup_percent=min_speedup_percent,
        max_regression_percent=max_regression_percent,
        benchmark_command=benchmark_command,
    )

    target = _target_from_run(run)
    patch_plan = _patch_plan_from_dict(run.patch_plan)
    approved_edits = _collect_approved_edits(
        patch_plan, list(approval.approved_action_ids)
    )
    source_text = target.target_path.read_text(encoding="utf-8")
    preflight_warnings = _preflight_edits(source_text, approved_edits)
    if preflight_warnings:
        run.lifecycle_status = OptimizationLifecycleStatus.FAILED
        run.final_status = "failed"
        run.error = "; ".join(preflight_warnings)
        run.application_result = {
            "status": "failed",
            "dry_run": dry_run,
            "warnings": preflight_warnings,
        }
        save_run(run, runs_dir=runs_dir)
        return run

    modified_text, apply_warnings = apply_patch_edits_to_text(source_text, approved_edits)
    if apply_warnings or modified_text == source_text:
        run.lifecycle_status = OptimizationLifecycleStatus.FAILED
        run.final_status = "failed"
        run.error = "; ".join(apply_warnings) or "Approved edits made no changes."
        run.application_result = {
            "status": "failed",
            "dry_run": dry_run,
            "warnings": apply_warnings,
        }
        save_run(run, runs_dir=runs_dir)
        return run

    validation_before_write = validate_source_text(
        modified_text,
        target.target_path,
        run_commands=[],
        timeout_sec=validation_timeout_sec,
    )
    if validation_before_write["status"] != "passed":
        run.lifecycle_status = OptimizationLifecycleStatus.FAILED
        run.final_status = "failed"
        run.error = "Temporary content validation failed before source replacement."
        run.validation_result = validation_before_write
        save_run(run, runs_dir=runs_dir)
        return run

    run.lifecycle_status = OptimizationLifecycleStatus.APPLYING
    if dry_run:
        run.application_result = {
            "status": "dry_run",
            "modified_hash": sha256_text(modified_text),
            "message": "Dry run completed; source file was not modified.",
        }
        run.lifecycle_status = OptimizationLifecycleStatus.APPROVED
        run.final_status = "dry_run"
        save_run(run, runs_dir=runs_dir)
        return run

    backup_path = create_backup(run, backup_dir=backup_dir)
    run.pre_application_backup_path = str(backup_path)
    temp_path = target.target_path.with_name(
        f".{target.target_path.name}.gpuboost-{run.run_id}.tmp"
    )
    modified_hash = sha256_text(modified_text)

    try:
        _write_text_safely(temp_path, modified_text)
        os.replace(temp_path, target.target_path)
        run.application_result = {
            "status": "applied",
            "backup_path": str(backup_path),
            "modified_hash": modified_hash,
            "approved_action_ids": list(approval.approved_action_ids),
        }
        run.lifecycle_status = OptimizationLifecycleStatus.VALIDATING
        validation = run_post_application_validation(
            target.target_path,
            validation_command=validation_command,
            test_command=test_command,
            timeout_sec=validation_timeout_sec,
        )
        run.validation_result = validation
        benchmark = run_benchmark_command(
            policy_settings.get("benchmark_command"),
            cwd=target.repo_root,
            timeout_sec=validation_timeout_sec,
        )
        policy = AcceptancePolicy(policy_settings["acceptance_policy"])
        run.benchmark_result = _evaluate_acceptance_policy(
            policy,
            benchmark_result=benchmark,
            min_speedup_percent=float(policy_settings.get("min_speedup_percent", 0.0)),
            max_regression_percent=float(
                policy_settings.get("max_regression_percent", 0.0)
            ),
        )
        if validation["status"] != "passed" or _policy_failed(run.benchmark_result):
            rollback = restore_backup(
                target.target_path,
                backup_path,
                expected_backup_hash=run.original_file_hash,
            )
            run.rollback_result = rollback
            if rollback["status"] == "passed":
                run.lifecycle_status = OptimizationLifecycleStatus.ROLLED_BACK
                run.final_status = "rolled_back"
                run.error = (
                    "Validation or acceptance policy failed; source was rolled back."
                )
            else:
                run.lifecycle_status = OptimizationLifecycleStatus.FAILED
                run.final_status = "failed"
                run.error = (
                    "Validation or acceptance policy failed, and rollback failed: "
                    f"{rollback.get('error', 'unknown error')}"
                )
        else:
            run.lifecycle_status = OptimizationLifecycleStatus.COMPLETED
            run.final_status = "completed"
    except Exception as error:  # noqa: BLE001 - source safety boundary
        rollback = restore_backup(
            target.target_path,
            backup_path,
            expected_backup_hash=run.original_file_hash,
        )
        run.rollback_result = rollback
        if rollback["status"] == "passed":
            run.lifecycle_status = OptimizationLifecycleStatus.ROLLED_BACK
            run.final_status = "rolled_back"
            run.error = f"Application failed and rollback was attempted: {error}"
        else:
            run.lifecycle_status = OptimizationLifecycleStatus.FAILED
            run.final_status = "failed"
            run.error = (
                f"Application failed, and rollback failed: "
                f"{rollback.get('error', error)}"
            )
    finally:
        if temp_path.exists():
            temp_path.unlink()

    save_run(run, runs_dir=runs_dir)
    return run


def rollback_optimization_run(
    run_id: str,
    *,
    repo_root: str | None = None,
    runs_dir: str | None = None,
    force: bool = False,
) -> AgenticOptimizationRun:
    """Restore the pre-application backup for a run."""

    run = load_run(run_id, repo_root=repo_root, runs_dir=runs_dir)
    if not run.pre_application_backup_path:
        raise ValueError(f"Run {run_id} has no application backup.")
    target = _target_from_run(run)
    if not force and run.application_result:
        expected_hash = run.application_result.get("modified_hash")
        if expected_hash and sha256_file(target.target_path) != expected_hash:
            raise ValueError(
                "Target file changed after application; pass --force to rollback."
            )
    rollback = restore_backup(
        target.target_path,
        Path(run.pre_application_backup_path),
        expected_backup_hash=run.original_file_hash,
    )
    run.rollback_result = rollback
    if rollback["status"] == "passed":
        run.lifecycle_status = OptimizationLifecycleStatus.ROLLED_BACK
        run.final_status = "rolled_back"
    else:
        run.lifecycle_status = OptimizationLifecycleStatus.FAILED
        run.final_status = "failed"
        run.error = f"Rollback failed: {rollback.get('error', 'unknown error')}"
    save_run(run, runs_dir=runs_dir)
    return run


def resolve_project_target(
    script_path: str,
    *,
    repo_root: str | None = None,
) -> ProjectTarget:
    """Resolve and validate repository root and target path."""

    root = Path(repo_root).expanduser() if repo_root else Path.cwd()
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Repository root is not a directory: {root}")
    if root.is_symlink():
        raise ValueError(f"Repository root must not be a symlink: {root}")

    target = Path(script_path).expanduser()
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    _ensure_inside_root(target, root)
    if target.is_symlink():
        raise ValueError(f"Target file must not be a symlink: {target}")
    if not target.exists() or not target.is_file():
        raise ValueError(f"Target file is not a regular file: {target}")
    relative = target.relative_to(root).as_posix()
    return ProjectTarget(repo_root=root, target_path=target, relative_target=relative)


def load_run(
    run_id: str,
    *,
    repo_root: str | None = None,
    runs_dir: str | None = None,
) -> AgenticOptimizationRun:
    """Load a persisted run by ID, constrained to the active repository root."""

    expected_root = _resolve_root(repo_root)
    path = _run_path(run_id, repo_root=repo_root, runs_dir=runs_dir)
    if not path.exists():
        raise ValueError(f"Optimization run not found: {run_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != AGENTIC_OPTIMIZATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported optimization run schema for {run_id}.")
    run = AgenticOptimizationRun.from_dict(data)
    stored_root = Path(run.target_repository_root).resolve()
    if stored_root != expected_root:
        raise ValueError(
            f"Run {run_id} references repository {stored_root}, which does not "
            f"match the active repository {expected_root}. Re-prepare the plan "
            "in this repository."
        )
    return run


def save_run(run: AgenticOptimizationRun, *, runs_dir: str | None = None) -> Path:
    """Persist a run under the target repository's ignored `.gpuboost` tree."""

    root = Path(run.target_repository_root).resolve()
    base = (root / (runs_dir or DEFAULT_RUNS_DIR)).resolve()
    _ensure_inside_root(base, root)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{run.run_id}.json"
    path.write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def short_plan_id(plan_id: str) -> str:
    """Return the short plan ID used in confirmation phrases."""

    return plan_id.split("-", 1)[-1][:8]


def sha256_file(path: Path) -> str:
    """Return SHA-256 for a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Return SHA-256 for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_plan_digest(
    *,
    target_path: str,
    original_hash: str,
    patch_plan: PatchPlan,
    diff: str,
    policy_settings: dict[str, Any],
) -> str:
    """Return a canonical digest for an immutable source plan."""

    payload = {
        "target_path": target_path,
        "original_hash": original_hash,
        "actions": [
            {
                "id": suggestion.id,
                "title": suggestion.title,
                "category": suggestion.category,
                "confidence": suggestion.confidence,
                "severity": suggestion.severity,
                "edits": [edit.to_dict() for edit in suggestion.edits],
            }
            for suggestion in patch_plan.suggestions
        ],
        "diff": diff,
        "policy_settings": policy_settings,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_plan_digest(run: AgenticOptimizationRun) -> None:
    """Verify the persisted digest still matches the plan content."""

    patch_plan = _patch_plan_from_dict(run.patch_plan)
    recomputed = compute_plan_digest(
        target_path=run.target_file,
        original_hash=run.original_file_hash,
        patch_plan=patch_plan,
        diff=run.generated_diff,
        policy_settings=(run.benchmark_result or {}).get("policy", {}),
    )
    if recomputed != run.plan_digest:
        raise ValueError("Plan digest mismatch; approval is invalid.")


def validate_current_source_matches_run(run: AgenticOptimizationRun) -> None:
    """Reject stale plans when the source file changed after planning."""

    target = _target_from_run(run)
    current_hash = sha256_file(target.target_path)
    if current_hash != run.original_file_hash:
        raise ValueError(
            "Target file changed after planning; approval is invalid. "
            "Re-run prepare."
        )


def validate_source_text(
    text: str,
    target_path: Path,
    *,
    run_commands: list[str],
    timeout_sec: int,
) -> dict[str, Any]:
    """Run built-in source validation against text before replacing a file."""

    stages: list[dict[str, Any]] = []
    if target_path.suffix.lower() == ".py":
        try:
            ast.parse(text, filename=str(target_path))
            stages.append({"name": "ast_parse", "status": "passed"})
        except SyntaxError as error:
            stages.append({"name": "ast_parse", "status": "failed", "error": str(error)})

        with tempfile.TemporaryDirectory(prefix="gpuboost-validate-") as temp_dir:
            temp_file = Path(temp_dir) / target_path.name
            temp_file.write_text(text, encoding="utf-8")
            try:
                py_compile.compile(str(temp_file), doraise=True)
                stages.append({"name": "py_compile", "status": "passed"})
            except py_compile.PyCompileError as error:
                stages.append(
                    {"name": "py_compile", "status": "failed", "error": str(error)}
                )
    else:
        stages.append({"name": "ast_parse", "status": "skipped"})
        stages.append({"name": "py_compile", "status": "skipped"})

    for command in run_commands:
        stages.append(run_explicit_validation_command(command, target_path.parent, timeout_sec))

    status = "failed" if any(stage["status"] == "failed" for stage in stages) else "passed"
    return {"status": status, "stages": stages}


def run_post_application_validation(
    target_path: Path,
    *,
    validation_command: str | None,
    test_command: str | None,
    timeout_sec: int,
) -> dict[str, Any]:
    """Run built-in and explicitly configured validators after application."""

    commands = [
        command
        for command in (validation_command, test_command)
        if command is not None and command.strip()
    ]
    text = target_path.read_text(encoding="utf-8")
    return validate_source_text(
        text,
        target_path,
        run_commands=commands,
        timeout_sec=timeout_sec,
    )


def run_explicit_validation_command(
    command: str,
    cwd: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    """Run a user-provided command without a shell."""

    argv = shlex.split(command)
    if not argv:
        return {"name": "explicit_command", "status": "skipped"}
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "name": "explicit_command",
            "command": argv[0],
            "status": "failed",
            "error": str(error),
        }
    return {
        "name": "explicit_command",
        "command": argv[0],
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "stdout": redact_output(completed.stdout),
        "stderr": redact_output(completed.stderr),
    }


def run_benchmark_command(
    command: str | None,
    *,
    cwd: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    """Run an optional benchmark command and parse JSON metrics when present."""

    if command is None or not command.strip():
        return {"status": "not_run", "metrics": None}
    result = run_explicit_validation_command(command, cwd, timeout_sec)
    result["name"] = "benchmark_command"
    metrics: dict[str, Any] | None = None
    stdout = result.get("stdout")
    if isinstance(stdout, str) and stdout.strip():
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            metrics = parsed
    result["metrics"] = metrics
    return result


def redact_output(text: str, limit: int = 4000) -> str:
    """Redact common secret-looking markers from captured command output."""

    redacted = text
    for marker in ("TOKEN=", "SECRET=", "PASSWORD=", "API_KEY=", "DATABASE_URL="):
        redacted = redacted.replace(marker, f"{marker}<redacted>")
    return redacted[:limit]


def create_backup(run: AgenticOptimizationRun, *, backup_dir: str | None) -> Path:
    """Create and return the backup path for a run."""

    target = _target_from_run(run)
    base = (target.repo_root / (backup_dir or DEFAULT_BACKUP_DIR) / run.run_id).resolve()
    _ensure_inside_root(base, target.repo_root)
    base.mkdir(parents=True, exist_ok=True)
    backup_path = base / target.target_path.name
    shutil.copy2(target.target_path, backup_path)
    manifest = {
        "run_id": run.run_id,
        "target_file": run.target_file,
        "original_file_hash": run.original_file_hash,
        "plan_id": run.plan_id,
        "plan_digest": run.plan_digest,
        "created_at": create_timestamp(),
    }
    (base / "backup-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return backup_path


def restore_backup(
    target_path: Path,
    backup_path: Path,
    *,
    expected_backup_hash: str,
) -> dict[str, Any]:
    """Atomically restore a backup to the target path."""

    if not backup_path.exists() or not backup_path.is_file():
        return {"status": "failed", "error": f"Backup missing: {backup_path}"}
    backup_hash = sha256_file(backup_path)
    if backup_hash != expected_backup_hash:
        return {"status": "failed", "error": "Backup hash mismatch."}
    temp_path = target_path.with_name(f".{target_path.name}.gpuboost-rollback.tmp")
    shutil.copy2(backup_path, temp_path)
    os.replace(temp_path, target_path)
    restored_hash = sha256_file(target_path)
    return {
        "status": "passed" if restored_hash == expected_backup_hash else "failed",
        "backup_path": str(backup_path),
        "restored_hash": restored_hash,
    }


def select_patch_plan_actions(
    patch_plan: PatchPlan,
    *,
    include_action_ids: list[str] | None,
    exclude_action_ids: list[str] | None,
) -> PatchPlan:
    """Return a patch plan filtered by action IDs."""

    include = set(include_action_ids or [])
    exclude = set(exclude_action_ids or [])
    suggestions = [
        suggestion
        for suggestion in patch_plan.suggestions
        if (not include or suggestion.id in include) and suggestion.id not in exclude
    ]
    return PatchPlan(
        generated_at=patch_plan.generated_at,
        filepath=patch_plan.filepath,
        status=patch_plan.status,
        suggestions=suggestions,
        warnings=list(patch_plan.warnings),
        error=patch_plan.error,
    )


def _require_valid_approval(run: AgenticOptimizationRun) -> OptimizationApproval:
    if run.approval_state != ApprovalState.APPROVED or run.approval is None:
        raise ValueError("Cannot apply without a persisted approval.")
    approval = OptimizationApproval.from_dict(run.approval)
    if approval.run_id != run.run_id:
        raise ValueError("Approval run ID mismatch.")
    if approval.plan_id != run.plan_id or approval.plan_digest != run.plan_digest:
        raise ValueError("Approval does not match the current plan.")
    if approval.target_file_hash != run.original_file_hash:
        raise ValueError("Approval target hash mismatch.")
    if sorted(run.approved_action_ids) != sorted(approval.approved_action_ids):
        raise ValueError(
            "Approved action IDs do not match the approval record; "
            "approval is invalid."
        )
    return approval


def _collect_approved_edits(
    patch_plan: PatchPlan,
    approved_action_ids: list[str],
) -> list[PatchEdit]:
    approved = set(approved_action_ids)
    return [
        edit
        for suggestion in patch_plan.suggestions
        if suggestion.id in approved
        for edit in suggestion.edits
    ]


def _preflight_edits(source_text: str, edits: list[PatchEdit]) -> list[str]:
    warnings: list[str] = []
    for edit in edits:
        count = source_text.count(edit.original_text)
        if count != 1:
            warnings.append(
                f"Edit {edit.description!r} expected source occurrence count 1; got {count}."
            )
    return warnings


def _write_text_safely(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _approved_policy_settings(
    run: AgenticOptimizationRun,
    *,
    acceptance_policy: str | None,
    min_speedup_percent: float | None,
    max_regression_percent: float | None,
    benchmark_command: str | None,
) -> dict[str, Any]:
    policy = dict((run.benchmark_result or {}).get("policy", {}))
    policy.setdefault("acceptance_policy", AcceptancePolicy.VALIDATION_ONLY.value)
    policy.setdefault("min_speedup_percent", 0.0)
    policy.setdefault("max_regression_percent", 0.0)
    policy.setdefault("benchmark_command", None)

    requested = {
        "acceptance_policy": acceptance_policy,
        "min_speedup_percent": min_speedup_percent,
        "max_regression_percent": max_regression_percent,
        "benchmark_command": benchmark_command,
    }
    for key, value in requested.items():
        if value is None:
            continue
        if str(value) != str(policy.get(key)):
            raise ValueError(
                f"{key} differs from the approved plan; re-run prepare and approve."
            )
    return policy


def _evaluate_acceptance_policy(
    policy: AcceptancePolicy,
    *,
    benchmark_result: dict[str, Any],
    min_speedup_percent: float,
    max_regression_percent: float,
) -> dict[str, Any]:
    if benchmark_result.get("status") == "failed":
        return {
            "status": "failed",
            "policy": policy.value,
            "benchmark": benchmark_result,
            "message": "Benchmark command failed.",
        }
    if policy == AcceptancePolicy.VALIDATION_ONLY:
        return {
            "status": "passed",
            "policy": policy.value,
            "benchmark": benchmark_result,
            "message": "Validation-only policy passed after validation succeeded.",
        }
    if policy == AcceptancePolicy.MANUAL_REVIEW:
        return {
            "status": "manual_review",
            "policy": policy.value,
            "benchmark": benchmark_result,
            "message": "Manual review policy selected; no benchmark gate enforced.",
        }
    metrics = benchmark_result.get("metrics")
    if not isinstance(metrics, dict):
        return {
            "status": "failed",
            "policy": policy.value,
            "benchmark": benchmark_result,
            "message": (
                "Benchmark threshold policy requires JSON stdout with "
                "speedup_percent or regression_percent."
            ),
        }
    speedup = _extract_float_metric(metrics, "speedup_percent")
    regression = _extract_float_metric(metrics, "regression_percent")
    if regression is None and speedup is not None:
        regression = max(0.0, -speedup)
    if speedup is None and regression is not None:
        speedup = -regression
    if policy == AcceptancePolicy.NO_REGRESSION:
        if regression is None:
            return {
                "status": "failed",
                "policy": policy.value,
                "benchmark": benchmark_result,
                "message": "No-regression policy requires regression_percent.",
            }
        passed = regression <= max_regression_percent
        return {
            "status": "passed" if passed else "failed",
            "policy": policy.value,
            "benchmark": benchmark_result,
            "message": (
                f"Regression {regression:.3f}% "
                f"{'is within' if passed else 'exceeds'} "
                f"the {max_regression_percent:.3f}% limit."
            ),
        }
    if policy == AcceptancePolicy.MINIMUM_SPEEDUP:
        if speedup is None:
            return {
                "status": "failed",
                "policy": policy.value,
                "benchmark": benchmark_result,
                "message": "Minimum-speedup policy requires speedup_percent.",
            }
        passed = speedup >= min_speedup_percent
        return {
            "status": "passed" if passed else "failed",
            "policy": policy.value,
            "benchmark": benchmark_result,
            "message": (
                f"Speedup {speedup:.3f}% "
                f"{'meets' if passed else 'misses'} "
                f"the {min_speedup_percent:.3f}% minimum."
            ),
        }
    return {
        "status": "failed",
        "policy": policy.value,
        "benchmark": benchmark_result,
        "message": "Unsupported acceptance policy.",
    }


def _policy_failed(result: dict[str, Any] | None) -> bool:
    return result is not None and result.get("status") == "failed"


def _extract_float_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _build_proposed_actions(patch_plan: PatchPlan) -> list[dict[str, Any]]:
    return [
        {
            "action_id": suggestion.id,
            "title": suggestion.title,
            "category": suggestion.category,
            "confidence": suggestion.confidence,
            "risk": _risk_for_suggestion(suggestion).value,
            "expected_benefit": suggestion.summary,
            "rationale": suggestion.rationale,
            "edit_count": len(suggestion.edits),
            "warnings": list(suggestion.warnings),
        }
        for suggestion in patch_plan.suggestions
    ]


def _build_proposed_edits(patch_plan: PatchPlan) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    for suggestion in patch_plan.suggestions:
        risk = _risk_for_suggestion(suggestion)
        for index, edit in enumerate(suggestion.edits, start=1):
            edits.append(
                ProposedEdit(
                    edit_id=f"{suggestion.id}:edit-{index}",
                    action_id=suggestion.id,
                    path=edit.filepath,
                    start_line=edit.start_line,
                    end_line=edit.end_line,
                    expected_before=edit.original_text,
                    replacement=edit.replacement_text,
                    rationale=suggestion.rationale,
                    risk=risk,
                ).to_dict()
            )
    return edits


def _risk_for_suggestion(suggestion: PatchSuggestion) -> RiskLevel:
    if suggestion.severity == "error":
        return RiskLevel.HIGH
    if suggestion.confidence == "low" or suggestion.warnings:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _patch_plan_from_dict(data: dict[str, Any]) -> PatchPlan:
    suggestions = []
    for suggestion_data in data.get("suggestions", []):
        edits = [
            PatchEdit(
                filepath=str(edit["filepath"]),
                start_line=int(edit["start_line"]),
                end_line=int(edit["end_line"]),
                original_text=str(edit["original_text"]),
                replacement_text=str(edit["replacement_text"]),
                description=str(edit["description"]),
            )
            for edit in suggestion_data.get("edits", [])
        ]
        suggestions.append(
            PatchSuggestion(
                id=str(suggestion_data["id"]),
                title=str(suggestion_data["title"]),
                category=str(suggestion_data["category"]),
                severity=str(suggestion_data["severity"]),
                confidence=str(suggestion_data["confidence"]),
                filepath=str(suggestion_data["filepath"]),
                finding_ids=[str(item) for item in suggestion_data.get("finding_ids", [])],
                summary=str(suggestion_data.get("summary", "")),
                rationale=str(suggestion_data.get("rationale", "")),
                edits=edits,
                warnings=[str(item) for item in suggestion_data.get("warnings", [])],
            )
        )
    return PatchPlan(
        generated_at=str(data["generated_at"]),
        filepath=str(data["filepath"]),
        status=str(data["status"]),
        suggestions=suggestions,
        warnings=[str(item) for item in data.get("warnings", [])],
        error=data.get("error"),
    )


def _target_from_run(run: AgenticOptimizationRun) -> ProjectTarget:
    root = Path(run.target_repository_root).resolve()
    target = (root / run.target_file).resolve()
    _ensure_inside_root(target, root)
    return ProjectTarget(
        repo_root=root,
        target_path=target,
        relative_target=run.target_file,
    )


def _run_path(
    run_id: str,
    *,
    repo_root: str | None,
    runs_dir: str | None,
) -> Path:
    root = _resolve_root(repo_root)
    base = (root / (runs_dir or DEFAULT_RUNS_DIR)).resolve()
    _ensure_inside_root(base, root)
    return base / f"{run_id}.json"


def _resolve_root(repo_root: str | None) -> Path:
    return Path(repo_root).expanduser().resolve() if repo_root else Path.cwd().resolve()


def _ensure_inside_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes repository root: {path}") from error


def _new_run_id() -> str:
    return f"opt-{uuid.uuid4().hex[:12]}"


def _new_plan_id() -> str:
    return f"plan-{uuid.uuid4().hex[:8]}"
