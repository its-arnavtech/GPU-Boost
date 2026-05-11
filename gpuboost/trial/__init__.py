"""Safe trial workspace helpers for GPUBoost."""

from gpuboost.trial.patch_apply import (
    apply_patch_plan_to_trial_file,
    collect_patch_edits,
    trial_file_is_inside_workspace,
)
from gpuboost.trial.engine import run_patch_trial
from gpuboost.trial.syntax_check import (
    is_python_file,
    run_python_syntax_check,
)
from gpuboost.trial.test_command import (
    run_trial_test_command,
    should_run_test_command,
)
from gpuboost.trial.workspace import (
    build_step,
    cleanup_trial_workspace,
    create_trial_workspace,
    hash_file,
    verify_original_unchanged,
    workspace_is_separate_from_original,
)

__all__ = [
    "apply_patch_plan_to_trial_file",
    "build_step",
    "cleanup_trial_workspace",
    "collect_patch_edits",
    "create_trial_workspace",
    "hash_file",
    "is_python_file",
    "run_patch_trial",
    "run_python_syntax_check",
    "run_trial_test_command",
    "should_run_test_command",
    "trial_file_is_inside_workspace",
    "verify_original_unchanged",
    "workspace_is_separate_from_original",
]
