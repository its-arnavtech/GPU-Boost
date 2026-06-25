"""GPUBoost command-line interface.

The CLI exposes Phase 1 inspection and Phase 2 benchmark commands.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

from gpuboost import __version__
from gpuboost.agent.approved_apply import (
    apply_approved_optimization_run,
    approve_optimization_run,
    load_run as load_agentic_run,
    prepare_optimization_run,
    reject_optimization_run,
    rollback_optimization_run,
    short_plan_id,
)
from gpuboost.agent.workflow import run_optimize_script_workflow
from gpuboost.advisor.engine import generate_advisor_result
from gpuboost.benchmarks.runner import run_full_benchmark, run_quick_benchmark
from gpuboost.code_analysis.runner import analyze_python_file
from gpuboost.comparison.engine import compare_benchmarks
from gpuboost.dataset.outcome_collection import (
    collect_outcomes_from_pairs_file,
)
from gpuboost.demo.real_world import (
    build_real_world_demo_pairs,
    write_real_world_pairs_file,
)
from gpuboost.history.compare import compare_history_runs
from gpuboost.history.store import list_history_runs, load_history_run
from gpuboost.inspector.profile import collect_profile
from gpuboost.model.artifacts import (
    DEFAULT_ARTIFACT_DIR,
    list_model_artifacts,
    save_neural_model_artifact,
    summarize_model_artifact,
    validate_model_artifact,
)
from gpuboost.model.feature_encoding import build_encoded_training_dataset
from gpuboost.model.neural_reports import write_neural_training_reports
from gpuboost.model.provider import TrainedLocalModelProvider
from gpuboost.model.safety import (
    verify_model_workflow_safety,
)
from gpuboost.model.training_data import load_training_rows_jsonl
from gpuboost.model.training_pipeline import (
    run_baseline_model_comparison,
)
from gpuboost.model.training_reports import (
    DEFAULT_BASELINE_REPORT_DIR,
    write_baseline_comparison_reports,
)
from gpuboost.repository import resolve_repository_context
from gpuboost.schemas.agentic import AcceptancePolicy, AgenticOptimizationRun
from gpuboost.schemas.model import ModelFeatureSet, ModelInput
from gpuboost.schemas.training import NeuralSearchResult, NeuralTrainingConfig
from gpuboost.utils.formatting import format_benchmark_suite, format_profile
from gpuboost.cli.rendering import (
    _create_patch_cli_output,
    _format_advisor_result,
    _format_code_analysis_result,
    _format_patch_output,
    build_agent_optimize_error_json_payload,
    build_agent_optimize_json_payload,
    build_compare_error_payload,
    build_compare_json_payload,
    build_dataset_collect_outcomes_error_payload,
    build_dataset_collect_outcomes_json_payload,
    build_demo_real_world_info_payload,
    build_history_compare_error_payload,
    build_history_compare_json_payload,
    build_history_list_error_payload,
    build_history_list_json_payload,
    build_history_show_error_payload,
    build_history_show_json_payload,
    build_model_check_artifact_json_payload,
    build_model_evaluate_baselines_error_payload,
    build_model_evaluate_baselines_json_payload,
    build_model_list_artifacts_json_payload,
    build_model_predict_artifact_error_payload,
    build_model_predict_artifact_json_payload,
    build_model_safety_check_json_payload,
    build_model_show_artifact_json_payload,
    build_model_train_neural_error_payload,
    build_model_train_neural_json_payload,
    build_model_validate_artifact_json_payload,
    render_agent_report_human,
    render_agent_unexpected_error_human,
    render_compare_error_human,
    render_comparison_human,
    render_dataset_collect_outcomes_error_human,
    render_dataset_collect_outcomes_human,
    render_demo_real_world_info_human,
    render_demo_real_world_pairs_human,
    render_history_compare_human,
    render_history_error_human,
    render_history_list_human,
    render_history_show_human,
    render_model_check_artifact_human,
    render_model_evaluate_baselines_error_human,
    render_model_evaluate_baselines_human,
    render_model_list_artifacts_human,
    render_model_predict_artifact_error_human,
    render_model_predict_artifact_human,
    render_model_safety_check_human,
    render_model_show_artifact_human,
    render_model_train_neural_error_human,
    render_model_train_neural_human,
    render_model_validate_artifact_human,
)

SETUP_DOCTOR_SCHEMA_VERSION = "setup.doctor.v1"
MIN_PYTHON_VERSION = (3, 9)


def run_neural_config_search(*args, **kwargs):
    from gpuboost.model.neural_training import run_neural_config_search as implementation

    return implementation(*args, **kwargs)


def run_neural_hyperparameter_search(*args, **kwargs):
    from gpuboost.model.neural_training import (
        run_neural_hyperparameter_search as implementation,
    )

    return implementation(*args, **kwargs)


def train_best_neural_model_for_artifact(*args, **kwargs):
    from gpuboost.model.neural_training import (
        train_best_neural_model_for_artifact as implementation,
    )

    return implementation(*args, **kwargs)


def train_neural_model_for_artifact_config(*args, **kwargs):
    from gpuboost.model.neural_training import (
        train_neural_model_for_artifact_config as implementation,
    )

    return implementation(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Build the GPUBoost argument parser."""

    parser = argparse.ArgumentParser(
        prog="gpuboost",
        description="Inspect NVIDIA GPU and system information.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run lightweight local setup checks.",
        description=(
            "Run lightweight local setup checks without requiring CUDA, "
            "running benchmarks, training models, or calling network services."
        ),
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    doctor_parser.add_argument(
        "--repo-root",
        help="Optional GPUBoost source repository root for repository-only checks.",
    )

    info_parser = subparsers.add_parser(
        "info",
        help="Show GPU and system information.",
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run Phase 2 synthetic GPU benchmarks.",
    )
    benchmark_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the quick benchmark subset.",
    )
    benchmark_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    benchmark_parser.add_argument(
        "--recommend",
        action="store_true",
        help="Generate optimization recommendations from benchmark results.",
    )
    benchmark_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to benchmark.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run static code analysis on a Python file.",
    )
    analyze_parser.add_argument(
        "filepath",
        help="Python source file to analyze.",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    analyze_parser.add_argument(
        "--patch",
        action="store_true",
        help="Print review-only unified patch suggestions.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two GPUBoost benchmark JSON files.",
    )
    compare_parser.add_argument(
        "baseline_json",
        help="Baseline benchmark JSON file.",
    )
    compare_parser.add_argument(
        "optimized_json",
        help="Optimized benchmark JSON file.",
    )
    compare_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    agent_parser = subparsers.add_parser(
        "agent",
        help="Run GPUBoost agent workflows.",
        description="Run GPUBoost agent workflows.",
    )
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command")

    agent_optimize_parser = agent_subparsers.add_parser(
        "optimize",
        help="Prepare an agent optimization workflow.",
        description=(
            "Prepare a deterministic optimization workflow. "
            "With --model-artifact, local generated model predictions are "
            "advisory-only and cannot apply patches. With --prepare, "
            "GPUBoost writes an approval-gated plan record but never mutates "
            "source before explicit approval."
        ),
    )
    agent_optimize_parser.add_argument(
        "script_path",
        nargs="?",
        help="Optional training script path.",
    )
    agent_optimize_parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare a human-approved apply run and show its immutable plan.",
    )
    agent_optimize_parser.add_argument(
        "--repo-root",
        help="Repository root used to constrain prepared/applyable source paths.",
    )
    agent_optimize_parser.add_argument(
        "--action",
        dest="action_ids",
        action="append",
        help="Only include this deterministic action ID in a prepared plan.",
    )
    agent_optimize_parser.add_argument(
        "--exclude-action",
        dest="exclude_action_ids",
        action="append",
        help="Exclude this deterministic action ID from a prepared plan.",
    )
    agent_optimize_parser.add_argument(
        "--acceptance-policy",
        choices=[policy.value for policy in AcceptancePolicy],
        default=AcceptancePolicy.VALIDATION_ONLY.value,
        help="Approved post-apply acceptance policy.",
    )
    agent_optimize_parser.add_argument(
        "--min-speedup-percent",
        type=float,
        default=0.0,
        help="Minimum speedup required by the minimum-speedup policy.",
    )
    agent_optimize_parser.add_argument(
        "--max-regression-percent",
        type=float,
        default=0.0,
        help="Maximum regression allowed by the no-regression policy.",
    )
    agent_optimize_parser.add_argument(
        "--benchmark",
        dest="benchmark_command",
        help=(
            "Optional benchmark command approved with the plan. Threshold "
            "policies read speedup_percent or regression_percent from JSON stdout."
        ),
    )
    agent_optimize_parser.add_argument(
        "--validation-command",
        help="Optional post-apply validation command for interactive apply.",
    )
    agent_optimize_parser.add_argument(
        "--backup-dir",
        help="Optional backup directory for interactive apply.",
    )
    agent_optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --interactive-apply, validate the approved apply without writing.",
    )
    agent_optimize_parser.add_argument(
        "--interactive-apply",
        action="store_true",
        help="Prepare, prompt for approval, then apply in one interactive session.",
    )
    agent_optimize_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    agent_optimize_parser.add_argument(
        "--include-raw-artifacts",
        action="store_true",
        help="Include raw diff and trial stdout/stderr in JSON output.",
    )
    agent_optimize_parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Accept quick-mode placeholder for future workflow integration.",
    )
    agent_optimize_parser.add_argument(
        "--trial",
        action="store_true",
        help="Validate generated patch suggestions in a temporary workspace.",
    )
    agent_optimize_parser.add_argument(
        "--model",
        action="store_true",
        help="Include optional model inference metadata in the workflow result.",
    )
    agent_optimize_parser.add_argument(
        "--model-artifact",
        dest="model_artifact_path",
        help=(
            "Use a local/generated model artifact for advisory-only inference; "
            "model predictions cannot apply patches."
        ),
    )
    agent_optimize_parser.add_argument(
        "--test",
        dest="test_command",
        help="Explicit test command to run inside the trial workspace.",
    )
    agent_optimize_parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save a safe local history record for this run.",
    )
    agent_optimize_parser.add_argument(
        "--history-db-path",
        help="Optional history database path for development and testing.",
    )

    agent_show_plan_parser = agent_subparsers.add_parser(
        "show-plan",
        help="Show an approval-gated optimization plan.",
    )
    agent_show_plan_parser.add_argument("run_id", help="Prepared run ID.")
    agent_show_plan_parser.add_argument("--repo-root", help="Repository root.")
    agent_show_plan_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    agent_approve_parser = agent_subparsers.add_parser(
        "approve",
        help="Approve a prepared optimization plan.",
    )
    agent_approve_parser.add_argument("run_id", help="Prepared run ID.")
    agent_approve_parser.add_argument("--repo-root", help="Repository root.")
    agent_approve_parser.add_argument(
        "--action",
        dest="action_ids",
        action="append",
        help="Approve only this action ID. Repeat to approve multiple actions.",
    )
    agent_approve_parser.add_argument(
        "--approved-by",
        default="local-user",
        help="Human approver identifier recorded in the audit trail.",
    )
    agent_approve_parser.add_argument(
        "--confirm",
        help='Exact confirmation phrase, for example: "APPLY 1234abcd".',
    )
    agent_approve_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    agent_reject_parser = agent_subparsers.add_parser(
        "reject",
        help="Reject a prepared optimization plan.",
    )
    agent_reject_parser.add_argument("run_id", help="Prepared run ID.")
    agent_reject_parser.add_argument("--repo-root", help="Repository root.")
    agent_reject_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    agent_apply_parser = agent_subparsers.add_parser(
        "apply",
        help="Apply an approved deterministic optimization plan.",
    )
    agent_apply_parser.add_argument("run_id", help="Approved run ID.")
    agent_apply_parser.add_argument("--repo-root", help="Repository root.")
    agent_apply_parser.add_argument(
        "--backup-dir",
        help="Optional backup directory under the repository root.",
    )
    agent_apply_parser.add_argument(
        "--validation-command",
        help="Optional explicit validation command to run after application.",
    )
    agent_apply_parser.add_argument(
        "--test",
        dest="test_command",
        help="Optional explicit test command to run after application.",
    )
    agent_apply_parser.add_argument(
        "--benchmark",
        dest="benchmark_command",
        help="Benchmark command; must match the approved plan when supplied.",
    )
    agent_apply_parser.add_argument(
        "--acceptance-policy",
        choices=[policy.value for policy in AcceptancePolicy],
        help="Acceptance policy; must match the approved plan when supplied.",
    )
    agent_apply_parser.add_argument(
        "--min-speedup-percent",
        type=float,
        help="Minimum speedup; must match the approved plan when supplied.",
    )
    agent_apply_parser.add_argument(
        "--max-regression-percent",
        type=float,
        help="Maximum regression; must match the approved plan when supplied.",
    )
    agent_apply_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the approved apply without modifying source.",
    )
    agent_apply_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    agent_rollback_parser = agent_subparsers.add_parser(
        "rollback",
        help="Rollback an applied optimization run from its backup.",
    )
    agent_rollback_parser.add_argument("run_id", help="Applied run ID.")
    agent_rollback_parser.add_argument("--repo-root", help="Repository root.")
    agent_rollback_parser.add_argument(
        "--force",
        action="store_true",
        help="Rollback even if the target changed after application.",
    )
    agent_rollback_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    agent_status_parser = agent_subparsers.add_parser(
        "status",
        help="Show the lifecycle status for an optimization run.",
    )
    agent_status_parser.add_argument("run_id", help="Prepared run ID.")
    agent_status_parser.add_argument("--repo-root", help="Repository root.")
    agent_status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    history_parser = subparsers.add_parser(
        "history",
        help="Inspect local GPUBoost run history.",
    )
    history_subparsers = history_parser.add_subparsers(dest="history_command")

    history_list_parser = history_subparsers.add_parser(
        "list",
        help="List local history runs.",
    )
    history_list_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    history_list_parser.add_argument(
        "--db-path",
        help="Optional history database path for development and testing.",
    )

    history_show_parser = history_subparsers.add_parser(
        "show",
        help="Show one local history run.",
    )
    history_show_parser.add_argument("run_id", help="History run ID to show.")
    history_show_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    history_show_parser.add_argument(
        "--db-path",
        help="Optional history database path for development and testing.",
    )

    history_compare_parser = history_subparsers.add_parser(
        "compare",
        help="Compare two local history runs.",
    )
    history_compare_parser.add_argument("left_run_id", help="Left history run ID.")
    history_compare_parser.add_argument("right_run_id", help="Right history run ID.")
    history_compare_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    history_compare_parser.add_argument(
        "--db-path",
        help="Optional history database path for development and testing.",
    )

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Run local dataset workflows.",
    )
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")

    collect_outcomes_parser = dataset_subparsers.add_parser(
        "collect-outcomes",
        help="Collect labeled outcome rows from benchmark JSON pairs.",
    )
    collect_outcomes_parser.add_argument(
        "pairs_json",
        help="Local JSON file describing baseline/optimized benchmark pairs.",
    )
    collect_outcomes_parser.add_argument(
        "--output-dir",
        default="data/gpuboost/generated/outcomes",
        help="Directory for outcome dataset and reports.",
    )
    collect_outcomes_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    model_parser = subparsers.add_parser(
        "model",
        help="Run local model training/evaluation workflows.",
        description=(
            "Run local model lifecycle workflows. Model predictions are "
            "advisory-only, artifacts are local/generated files, and models "
            "cannot apply patches."
        ),
        epilog=(
            "Lifecycle commands: evaluate-baselines, train-neural, "
            "train-neural --save-artifact, list-artifacts, show-artifact, "
            "check-artifact, validate-artifact, predict-artifact. "
            "--save-artifact is explicit. check-artifact is a read-only "
            "quality gate before optional advisory agent use."
        ),
    )
    model_subparsers = model_parser.add_subparsers(dest="model_command")

    evaluate_baselines_parser = model_subparsers.add_parser(
        "evaluate-baselines",
        help="Evaluate safe structured baseline models without saving artifacts.",
        description=(
            "Evaluate dependency-free baselines on safe encoded training data. "
            "This command does not train a neural model or save model artifacts. "
            "Model predictions are advisory only and cannot apply patches."
        ),
    )
    evaluate_baselines_parser.add_argument(
        "--dataset",
        default="data/gpuboost/generated/training_dataset.jsonl",
        help="DatasetRow JSONL file to evaluate.",
    )
    evaluate_baselines_parser.add_argument(
        "--output-dir",
        default=DEFAULT_BASELINE_REPORT_DIR,
        help="Directory for baseline comparison reports.",
    )
    evaluate_baselines_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    train_neural_parser = model_subparsers.add_parser(
        "train-neural",
        help="Train a small local neural model from scratch.",
        description=(
            "Train a small local MLP from scratch on safe encoded features. "
            "Artifacts are local generated files and are saved only with "
            "the explicit --save-artifact flag. Any later model predictions "
            "are advisory only and cannot apply patches."
        ),
    )
    train_neural_parser.add_argument(
        "--dataset",
        default="data/gpuboost/generated/training_dataset.jsonl",
        help="DatasetRow JSONL file to train on.",
    )
    train_neural_parser.add_argument(
        "--output-dir",
        default=DEFAULT_BASELINE_REPORT_DIR,
        help="Directory for neural training reports.",
    )
    train_neural_parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum epochs per neural candidate.",
    )
    train_neural_parser.add_argument(
        "--hidden-sizes",
        help="Comma-separated hidden layer sizes for a single config, e.g. 32,16.",
    )
    train_neural_parser.add_argument(
        "--target-macro-f1",
        type=float,
        default=0.85,
        help="Aspirational validation macro F1 target.",
    )
    train_neural_parser.add_argument(
        "--max-candidates",
        type=int,
        default=12,
        help="Maximum search candidates to train.",
    )
    train_neural_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    train_neural_parser.add_argument(
        "--save-artifact",
        action="store_true",
        help=(
            "Explicitly save a local generated model artifact; without this "
            "flag, training writes reports only."
        ),
    )
    train_neural_parser.add_argument(
        "--artifact-dir",
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory for generated model artifacts.",
    )
    train_neural_parser.add_argument(
        "--artifact-name",
        help="Optional artifact directory name.",
    )

    list_artifacts_parser = model_subparsers.add_parser(
        "list-artifacts",
        help="List local generated model artifacts.",
        description=(
            "List local/generated model artifact manifests. This command only "
            "reads manifests and does not inspect model weights."
        ),
    )
    list_artifacts_parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing generated model artifacts.",
    )
    list_artifacts_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    show_artifact_parser = model_subparsers.add_parser(
        "show-artifact",
        help="Show a safe summary for one local model artifact.",
        description=(
            "Show manifest metrics and validation status without printing model "
            "weights, raw source, raw diffs, stdout, or stderr. Artifact "
            "predictions are advisory only and cannot apply patches."
        ),
    )
    show_artifact_parser.add_argument("manifest_path")
    show_artifact_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    check_artifact_parser = model_subparsers.add_parser(
        "check-artifact",
        help="Run read-only quality gates for a local model artifact.",
        description=(
            "Validate a local model artifact and run read-only quality gates. "
            "This command does not train, mutate files, or select artifacts "
            "for the agent automatically."
        ),
    )
    check_artifact_parser.add_argument(
        "manifest_path",
        help="Artifact manifest to check; does not train or mutate files.",
    )
    check_artifact_parser.add_argument(
        "--min-test-macro-f1",
        type=float,
        default=None,
        help="Quality gates: require test macro F1 to be at least this value.",
    )
    check_artifact_parser.add_argument(
        "--require-beats-baseline",
        action="store_true",
        help="Require the artifact to beat the best baseline.",
    )
    check_artifact_parser.add_argument(
        "--require-target-met",
        action="store_true",
        help="Require the artifact target macro F1 gate to be met.",
    )
    check_artifact_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    validate_artifact_parser = model_subparsers.add_parser(
        "validate-artifact",
        help="Validate a saved local/generated model artifact manifest.",
        description=(
            "Validate one local/generated artifact manifest and referenced "
            "files. This command does not train, mutate files, or apply patches."
        ),
    )
    validate_artifact_parser.add_argument(
        "manifest_path",
        help="Local/generated artifact manifest to validate.",
    )
    validate_artifact_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    predict_artifact_parser = model_subparsers.add_parser(
        "predict-artifact",
        help="Run advisory local prediction from a saved model artifact.",
        description=(
            "Run local prediction from safe feature JSON. Predictions are "
            "advisory-only and cannot apply patches."
        ),
    )
    predict_artifact_parser.add_argument(
        "manifest_path",
        help="Local/generated artifact manifest to load for advisory prediction.",
    )
    predict_artifact_parser.add_argument(
        "--features-json",
        help="JSON object of safe feature values.",
    )
    predict_artifact_parser.add_argument(
        "--features-file",
        help="Path to a JSON object of safe feature values.",
    )
    predict_artifact_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    safety_check_parser = model_subparsers.add_parser(
        "safety-check",
        help="Verify Phase 12 local model workflow safety guardrails.",
        description=(
            "Inspect docs, ignore rules, and provider metadata for lightweight "
            "release-readiness checks. This command does not train or load "
            "model weights."
        ),
    )
    safety_check_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    safety_check_parser.add_argument(
        "--repo-root",
        help="Optional GPUBoost source repository root for repository-only checks.",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Discover safe GPUBoost demo workflows.",
        description=(
            "Discover Phase 14 demo workflows without running benchmarks, "
            "training models, calling network services, or applying patches."
        ),
        epilog=(
            "Demo commands are informational by default. Model output remains "
            "advisory-only, generated artifacts are ignored, and there is no "
            "automatic patch application."
        ),
    )
    demo_subparsers = demo_parser.add_subparsers(dest="demo_command")

    real_world_parser = demo_subparsers.add_parser(
        "real-world",
        help="Show the real-world validation demo workflow.",
        description=(
            "Show available Phase 14 real-world demo workloads and commands. "
            "This does not execute benchmarks or train models."
        ),
    )
    real_world_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    real_world_info_parser = demo_subparsers.add_parser(
        "real-world-info",
        help="Print real-world demo workflow information.",
    )
    real_world_info_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )

    real_world_pairs_parser = demo_subparsers.add_parser(
        "real-world-pairs",
        help="Print real-world demo pair specs without running benchmarks.",
        description=(
            "Print collect-outcomes-compatible real-world demo pair metadata. "
            "Use --write to write the pairs file; no benchmarks are executed."
        ),
    )
    real_world_pairs_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    real_world_pairs_parser.add_argument(
        "--write",
        action="store_true",
        help="Write data/gpuboost/generated/demo_real_world/pairs.json.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the GPUBoost CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        result = build_setup_doctor_report(repo_root=args.repo_root)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(render_setup_doctor_human(result))
        return setup_doctor_status_to_exit_code(str(result["status"]))

    if args.command == "info":
        profile = collect_profile()
        if args.json:
            print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
        else:
            try:
                from rich.console import Console
            except Exception:
                print(format_profile(profile))
            else:
                Console().print(format_profile(profile))
        return 0

    if args.command == "benchmark":
        suite = (
            run_quick_benchmark(args.device)
            if args.quick
            else run_full_benchmark(args.device)
        )
        advisor_result = generate_advisor_result(suite) if args.recommend else None
        if args.json:
            output = (
                {
                    "benchmark": suite.to_dict(),
                    "advisor": advisor_result.to_dict(),
                }
                if advisor_result is not None
                else suite.to_dict()
            )
            print(json.dumps(output, indent=2, sort_keys=True))
        else:
            output = format_benchmark_suite(suite)
            if advisor_result is None:
                output = (
                    f"{output}\n\n"
                    "Run with --recommend to generate optimization advice."
                )
            else:
                output = f"{output}\n\n{_format_advisor_result(advisor_result)}"
            try:
                from rich.console import Console
            except Exception:
                print(output)
            else:
                Console().print(output, markup=False)
        return 0

    if args.command == "analyze":
        result = analyze_python_file(args.filepath)
        if result.status != "ok":
            if args.json and args.patch:
                print(
                    json.dumps(
                        {
                            "analysis": result.to_dict(),
                            "patch_plan": None,
                            "diff": "",
                            "patch_warnings": [],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            elif args.json:
                print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            else:
                print(_format_code_analysis_result(result))
            return 1

        patch_output = None
        if args.patch:
            patch_output = _create_patch_cli_output(args.filepath, result)

        if args.json:
            if args.patch and patch_output is not None:
                print(
                    json.dumps(
                        {
                            "analysis": result.to_dict(),
                            "patch_plan": patch_output["patch_plan"],
                            "diff": patch_output["diff"],
                            "patch_warnings": patch_output["patch_warnings"],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            output = _format_code_analysis_result(result)
            if args.patch and patch_output is not None:
                output = f"{output}\n\n{_format_patch_output(patch_output)}"
            print(output)

        return 0

    if args.command == "compare":
        try:
            baseline = load_json_file(args.baseline_json)
            optimized = load_json_file(args.optimized_json)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            error_message = _format_json_file_error(error)
            if args.json:
                print(
                    json.dumps(
                        build_compare_error_payload(error_message),
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(render_compare_error_human(error_message))
            return 1

        result = compare_benchmarks(
            baseline=baseline,
            optimized=optimized,
            baseline_label=args.baseline_json,
            optimized_label=args.optimized_json,
        )
        if args.json:
            print(
                json.dumps(
                    build_compare_json_payload(result),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_comparison_human(result))
        return comparison_status_to_exit_code(result.status)

    if args.command == "agent":
        if args.agent_command == "optimize":
            if args.prepare or args.interactive_apply:
                return _run_agent_prepare(args)

            validation_error = _validate_agent_optimize_args(args)
            if validation_error is not None:
                if args.json:
                    print(
                        json.dumps(
                            build_agent_optimize_error_json_payload(
                                validation_error,
                            ),
                            indent=2,
                            sort_keys=True,
                        )
                    )
                else:
                    print(render_agent_unexpected_error_human(validation_error))
                return 1

            try:
                workflow_kwargs = {
                    "script_path": args.script_path,
                    "quick": args.quick,
                    "model": args.model or bool(args.model_artifact_path),
                }
                if args.model_artifact_path is not None:
                    workflow_kwargs["model_artifact_path"] = args.model_artifact_path
                if args.trial:
                    workflow_kwargs["trial"] = True
                if args.test_command is not None:
                    workflow_kwargs["test_command"] = args.test_command
                if args.save_history:
                    workflow_kwargs["save_history"] = True
                    workflow_kwargs["history_db_path"] = args.history_db_path
                result, report = run_optimize_script_workflow(**workflow_kwargs)
            except Exception as error:  # noqa: BLE001 - CLI boundary
                error_message = _format_exception_message(error)
                if args.json:
                    print(
                        json.dumps(
                            build_agent_optimize_error_json_payload(error_message),
                            indent=2,
                            sort_keys=True,
                        )
                    )
                else:
                    print(render_agent_unexpected_error_human(error_message))
                return 1

            if args.json:
                print(
                    json.dumps(
                        build_agent_optimize_json_payload(
                            result,
                            report,
                            include_raw_artifacts=args.include_raw_artifacts,
                        ),
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                output = render_agent_report_human(
                    report=report,
                    result=result,
                    script_path=args.script_path,
                    trial_requested=args.trial,
                )
                try:
                    from rich.console import Console
                except Exception:
                    print(output)
                else:
                    Console().print(output, markup=False, soft_wrap=True)
            return agent_status_to_exit_code(result.status)

        if args.agent_command == "show-plan":
            return _run_agent_show_plan(args)
        if args.agent_command == "approve":
            return _run_agent_approve(args)
        if args.agent_command == "reject":
            return _run_agent_reject(args)
        if args.agent_command == "apply":
            return _run_agent_apply(args)
        if args.agent_command == "rollback":
            return _run_agent_rollback(args)
        if args.agent_command == "status":
            return _run_agent_status(args)

        print(
            "GPUBoost Agent\n"
            "Available commands: optimize, show-plan, approve, reject, "
            "apply, rollback, status"
        )
        return 0

    if args.command == "history":
        if args.history_command == "list":
            return _run_history_list(args)
        if args.history_command == "show":
            return _run_history_show(args)
        if args.history_command == "compare":
            return _run_history_compare(args)

        print("GPUBoost History\nAvailable commands: list, show, compare")
        return 0

    if args.command == "dataset":
        if args.dataset_command == "collect-outcomes":
            return _run_dataset_collect_outcomes(args)

        print("GPUBoost Dataset\nAvailable commands: collect-outcomes")
        return 0

    if args.command == "model":
        if args.model_command == "evaluate-baselines":
            return _run_model_evaluate_baselines(args)
        if args.model_command == "train-neural":
            return _run_model_train_neural(args)
        if args.model_command == "list-artifacts":
            return _run_model_list_artifacts(args)
        if args.model_command == "show-artifact":
            return _run_model_show_artifact(args)
        if args.model_command == "check-artifact":
            return _run_model_check_artifact(args)
        if args.model_command == "validate-artifact":
            return _run_model_validate_artifact(args)
        if args.model_command == "predict-artifact":
            return _run_model_predict_artifact(args)
        if args.model_command == "safety-check":
            return _run_model_safety_check(args)

        print(
            "GPUBoost Model\nAvailable commands: evaluate-baselines, train-neural, "
            "list-artifacts, show-artifact, check-artifact, validate-artifact, "
            "predict-artifact, safety-check"
        )
        return 0

    if args.command == "demo":
        if args.demo_command in {"real-world", "real-world-info"}:
            return _run_demo_real_world_info(args)
        if args.demo_command == "real-world-pairs":
            return _run_demo_real_world_pairs(args)

        print(
            "GPUBoost Demo\n"
            "Available commands: real-world, real-world-info, real-world-pairs\n"
            "Demo commands are lightweight and do not run benchmarks or train models."
        )
        return 0

    parser.print_help()
    return 0


def build_setup_doctor_report(repo_root: str | None = None) -> dict[str, object]:
    """Return lightweight setup checks for local development."""

    repository = resolve_repository_context(repo_root)
    checks = [
        _check_python_version(),
        _check_import("gpuboost", required=True),
        _check_import("psutil", required=True),
        _check_import("rich", required=True),
        _check_import("pynvml", required=True, label="nvidia-ml-py / pynvml"),
        _check_import("pytest", required=False),
        _check_import("ruff", required=False),
        _check_torch_availability(),
        _check_gitignore_safety(repo_root=repo_root),
    ]
    failed = [check for check in checks if check["status"] == "failed"]
    warnings = [
        check for check in checks if check["status"] in {"warning", "skipped"}
    ]
    status = "error" if failed else "warning" if warnings else "ok"
    return {
        "schema_version": SETUP_DOCTOR_SCHEMA_VERSION,
        "status": status,
        "python_version": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "repo_root": str(repository.root) if repository.root is not None else None,
        "repo_root_status": repository.status,
        "repo_root_message": repository.message,
        "checks": checks,
        "cuda_required": False,
    }


def render_setup_doctor_human(result: dict[str, object]) -> str:
    """Render setup doctor checks for terminal output."""

    lines = [
        "GPUBoost Doctor",
        f"Status: {result['status']}",
        f"Python: {result['python_version']}",
        "",
        "Checks:",
    ]
    checks = result.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            name = check.get("name", "unknown")
            status = check.get("status", "unknown")
            message = check.get("message", "")
            lines.append(f"- {name}: {status} - {message}")
    lines.extend(
        [
            "",
            "CUDA: not required for setup verification.",
            "Doctor does not run benchmarks, train models, call external APIs, "
            "or write generated artifacts.",
        ]
    )
    return "\n".join(lines)


def setup_doctor_status_to_exit_code(status: str) -> int:
    """Return the CLI exit code for setup doctor status."""

    return 1 if status == "error" else 0


def _check_python_version() -> dict[str, object]:
    current = sys.version_info[:3]
    minimum = ".".join(str(part) for part in MIN_PYTHON_VERSION)
    current_text = ".".join(str(part) for part in current)
    passed = current >= MIN_PYTHON_VERSION
    return {
        "name": "python_version",
        "status": "passed" if passed else "failed",
        "required": True,
        "message": f"Python {current_text}; requires >= {minimum}.",
    }


def _check_import(
    module_name: str,
    *,
    required: bool,
    label: str | None = None,
) -> dict[str, object]:
    display = label or module_name
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {
            "name": f"import:{display}",
            "status": "failed" if required else "warning",
            "required": required,
            "message": f"{display} is not importable.",
        }

    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # noqa: BLE001 - import diagnostics boundary
        return {
            "name": f"import:{display}",
            "status": "failed" if required else "warning",
            "required": required,
            "message": f"{display} import failed: {_format_exception_message(error)}",
        }

    version = getattr(module, "__version__", None)
    message = f"{display} is importable."
    if isinstance(version, str) and version:
        message = f"{message} version={version}."
    return {
        "name": f"import:{display}",
        "status": "passed",
        "required": required,
        "message": message,
    }


def _check_torch_availability() -> dict[str, object]:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return {
            "name": "optional:torch",
            "status": "warning",
            "required": False,
            "message": (
                "PyTorch is not importable; CUDA benchmarks and model training "
                "will be unavailable, but CUDA is not required for setup checks."
            ),
        }

    try:
        torch = importlib.import_module("torch")
    except Exception as error:  # noqa: BLE001 - import diagnostics boundary
        return {
            "name": "optional:torch",
            "status": "warning",
            "required": False,
            "message": f"PyTorch import failed: {_format_exception_message(error)}",
        }

    cuda_available: bool | str
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as error:  # noqa: BLE001 - import diagnostics boundary
        cuda_available = f"unknown: {_format_exception_message(error)}"

    return {
        "name": "optional:torch",
        "status": "passed",
        "required": False,
        "message": (
            f"PyTorch is importable. version={getattr(torch, '__version__', 'unknown')}; "
            f"cuda_available={cuda_available}; CUDA is not required."
        ),
    }


def _check_gitignore_safety(repo_root: str | None = None) -> dict[str, object]:
    repository = resolve_repository_context(repo_root)
    if repository.root is None:
        return {
            "name": "gitignore_generated_artifacts",
            "status": "warning" if repository.status == "invalid" else "skipped",
            "required": False,
            "applicable": False,
            "repo_root": None,
            "message": f"{repository.message} Repository artifact-policy checks skipped.",
        }

    gitignore = _read_optional_text(repository.root / ".gitignore")
    expected_patterns = [
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
    active_patterns = {
        line.strip()
        for line in gitignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = [pattern for pattern in expected_patterns if pattern not in active_patterns]
    return {
        "name": "gitignore_generated_artifacts",
        "status": "failed" if missing else "passed",
        "required": True,
        "applicable": True,
        "repo_root": str(repository.root),
        "message": "All generated data, model artifact, and local DB patterns present."
        if not missing
        else f"Missing .gitignore patterns: {', '.join(missing)}.",
    }


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def load_json_file(filepath: str) -> dict:
    """Load a UTF-8 JSON object from disk, accepting an optional BOM."""

    with Path(filepath).open(encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in file: {filepath}")
    return data


def _run_dataset_collect_outcomes(args: argparse.Namespace) -> int:
    try:
        summary = collect_outcomes_from_pairs_file(
            pairs_file=args.pairs_json,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as error:
        error_message = _format_exception_message(error)
        if args.json:
            print(
                json.dumps(
                    build_dataset_collect_outcomes_error_payload(error_message),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_dataset_collect_outcomes_error_human(error_message))
        return 1

    if args.json:
        print(
            json.dumps(
                build_dataset_collect_outcomes_json_payload(summary),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_dataset_collect_outcomes_human(summary))

    # Treat "no rows collected" (e.g., every pair failed, or an empty pairs
    # file) as a failure so callers and CI don't mistake an empty dataset for
    # success.
    if summary.get("collected_row_count", 0) == 0:
        return 1
    return 0


def _run_demo_real_world_info(args: argparse.Namespace) -> int:
    payload = build_demo_real_world_info_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_demo_real_world_info_human(payload))
    return 0


def _run_demo_real_world_pairs(args: argparse.Namespace) -> int:
    pairs = build_real_world_demo_pairs()
    output_path = None
    if args.write:
        output_path = write_real_world_pairs_file(pairs)

    payload = build_demo_real_world_info_payload()
    payload["pairs"] = pairs
    payload["pairs_file_written"] = output_path
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_demo_real_world_pairs_human(payload))
    return 0


def _run_model_evaluate_baselines(args: argparse.Namespace) -> int:
    try:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
        rows = load_training_rows_jsonl(str(dataset_path))
        dataset = build_encoded_training_dataset(rows)
        comparison = run_baseline_model_comparison(dataset)
        output_files = write_baseline_comparison_reports(comparison, args.output_dir)
    except (OSError, ValueError) as error:
        error_message = _format_exception_message(error)
        if args.json:
            print(
                json.dumps(
                    build_model_evaluate_baselines_error_payload(error_message),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_model_evaluate_baselines_error_human(error_message))
        return 1

    result = dict(comparison)
    result["output_files"] = output_files
    if args.json:
        print(
            json.dumps(
                build_model_evaluate_baselines_json_payload(result),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_model_evaluate_baselines_human(result))
    return 0 if comparison.get("status") == "ok" else 1


def _run_model_train_neural(args: argparse.Namespace) -> int:
    try:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
        rows = load_training_rows_jsonl(str(dataset_path))
        dataset = build_encoded_training_dataset(rows)
        baseline = run_baseline_model_comparison(dataset)
        baseline_macro_f1 = _optional_float(baseline.get("best_macro_f1"))
        baseline_model_name = _optional_string(baseline.get("best_model_name"))
        hidden_sizes = _parse_hidden_sizes(args.hidden_sizes)
        artifact_manifest = None
        if args.save_artifact and hidden_sizes is None:
            model, feature_spec, label_mapping, search = train_best_neural_model_for_artifact(
                dataset,
                baseline_macro_f1=baseline_macro_f1,
                target_macro_f1=args.target_macro_f1,
                max_epochs=args.max_epochs,
                max_candidates=args.max_candidates,
            )
            _attach_baseline_metadata(search, baseline_model_name, baseline_macro_f1)
            artifact_manifest = save_neural_model_artifact(
                model,
                feature_spec,
                label_mapping,
                search.best_config,
                search,
                output_dir=args.artifact_dir,
                artifact_name=args.artifact_name,
            )
        elif args.save_artifact and hidden_sizes is not None:
            config = NeuralTrainingConfig(
                hidden_sizes=hidden_sizes,
                max_epochs=args.max_epochs,
            )
            model, feature_spec, label_mapping, single_result = (
                train_neural_model_for_artifact_config(
                    dataset,
                    config=config,
                    baseline_macro_f1=baseline_macro_f1,
                )
            )
            search = run_neural_config_search(
                dataset,
                [config],
                baseline_macro_f1=baseline_macro_f1,
                target_macro_f1=args.target_macro_f1,
            )
            search = replace(
                search,
                best_result=single_result,
                candidates=[single_result],
                best_config=single_result.config,
                best_validation_macro_f1=(
                    single_result.validation_evaluation.macro_f1
                    if single_result.validation_evaluation is not None
                    else None
                ),
                best_test_macro_f1=(
                    single_result.test_evaluation.macro_f1
                    if single_result.test_evaluation is not None
                    else None
                ),
            )
            _attach_baseline_metadata(search, baseline_model_name, baseline_macro_f1)
            artifact_manifest = save_neural_model_artifact(
                model,
                feature_spec,
                label_mapping,
                search.best_config,
                search,
                output_dir=args.artifact_dir,
                artifact_name=args.artifact_name,
            )
        elif hidden_sizes is None:
            search = run_neural_hyperparameter_search(
                dataset,
                baseline_macro_f1=baseline_macro_f1,
                target_macro_f1=args.target_macro_f1,
                max_epochs=args.max_epochs,
                max_candidates=args.max_candidates,
            )
        else:
            search = run_neural_config_search(
                dataset,
                [
                    NeuralTrainingConfig(
                        hidden_sizes=hidden_sizes,
                        max_epochs=args.max_epochs,
                    )
                ],
                baseline_macro_f1=baseline_macro_f1,
                target_macro_f1=args.target_macro_f1,
            )
        _attach_baseline_metadata(search, baseline_model_name, baseline_macro_f1)
        output_files = write_neural_training_reports(search, args.output_dir)
    except (OSError, ValueError) as error:
        error_message = _format_exception_message(error)
        if args.json:
            print(
                json.dumps(
                    build_model_train_neural_error_payload(error_message),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_model_train_neural_error_human(error_message))
        return 1

    result = search.to_dict()
    result["output_files"] = output_files
    result["baseline_comparison"] = baseline
    result["patch_application_allowed"] = False
    if artifact_manifest is not None:
        manifest_path = artifact_manifest.metadata.get("manifest_path")
        if isinstance(manifest_path, str):
            portable_manifest_path = Path(manifest_path).as_posix()
            result["artifact_manifest"] = portable_manifest_path
            result["artifact_manifest_path"] = portable_manifest_path
            validation = validate_model_artifact(portable_manifest_path)
            result["artifact_validation_status"] = validation.get("status")
    if args.json:
        print(
            json.dumps(
                build_model_train_neural_json_payload(result),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_model_train_neural_human(result))
    return 0 if search.status == "ok" else 1


def _run_model_list_artifacts(args: argparse.Namespace) -> int:
    artifacts = list_model_artifacts(args.artifacts_dir)
    result = {
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    if args.json:
        print(
            json.dumps(
                build_model_list_artifacts_json_payload(result),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_model_list_artifacts_human(result))
    return 0


def _run_model_show_artifact(args: argparse.Namespace) -> int:
    summary = summarize_model_artifact(args.manifest_path)
    if args.json:
        print(
            json.dumps(
                build_model_show_artifact_json_payload(summary),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_model_show_artifact_human(summary))
    return 0 if summary.get("validation_status") == "ok" else 1


def _run_model_check_artifact(args: argparse.Namespace) -> int:
    result = check_model_artifact_quality(
        args.manifest_path,
        min_test_macro_f1=args.min_test_macro_f1,
        require_beats_baseline=args.require_beats_baseline,
        require_target_met=args.require_target_met,
    )
    if args.json:
        print(
            json.dumps(
                build_model_check_artifact_json_payload(result),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_model_check_artifact_human(result))
    return 0 if result.get("status") == "passed" else 1


def _run_model_validate_artifact(args: argparse.Namespace) -> int:
    result = validate_model_artifact(args.manifest_path)
    result["manifest_path"] = args.manifest_path
    if args.json:
        print(
            json.dumps(
                build_model_validate_artifact_json_payload(result),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_model_validate_artifact_human(result))
    return 0 if result.get("status") == "ok" else 1


def _run_model_predict_artifact(args: argparse.Namespace) -> int:
    try:
        features = _load_cli_features(args.features_json, args.features_file)
        provider = TrainedLocalModelProvider(args.manifest_path, device="cpu")
        model_input = ModelInput(
            goal="model predict-artifact",
            features=ModelFeatureSet(),
            context={"features": features, "command": "model predict-artifact"},
        )
        result = provider.predict(model_input)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        error_message = _format_exception_message(error)
        if args.json:
            print(
                json.dumps(
                    build_model_predict_artifact_error_payload(error_message),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_model_predict_artifact_error_human(error_message))
        return 1

    payload = result.to_dict()
    if args.json:
        print(
            json.dumps(
                build_model_predict_artifact_json_payload(payload),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_model_predict_artifact_human(payload))
    return 0 if payload.get("status") == "ok" else 1


def _run_model_safety_check(args: argparse.Namespace) -> int:
    result = verify_model_workflow_safety(repo_root=args.repo_root)
    if args.json:
        print(
            json.dumps(
                build_model_safety_check_json_payload(result),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_model_safety_check_human(result))
    return 1 if result.get("status") == "error" else 0


def _run_history_list(args: argparse.Namespace) -> int:
    try:
        history = list_history_runs(db_path=args.db_path)
    except Exception as error:  # noqa: BLE001 - CLI boundary
        error_message = _format_exception_message(error)
        if args.json:
            print(
                json.dumps(
                    build_history_list_error_payload(error_message),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_history_error_human("GPUBoost History", error_message))
        return 1

    if args.json:
        print(
            json.dumps(
                build_history_list_json_payload(history),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_history_list_human(history))
    return 0


def _run_history_show(args: argparse.Namespace) -> int:
    try:
        record = load_history_run(args.run_id, db_path=args.db_path)
    except Exception as error:  # noqa: BLE001 - CLI boundary
        error_message = _format_exception_message(error)
        if args.json:
            print(
                json.dumps(
                    build_history_show_error_payload(error_message),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_history_error_human("GPUBoost History Run", error_message))
        return 1

    if record is None:
        error_message = f"History run not found: {args.run_id}"
        if args.json:
            print(
                json.dumps(
                    build_history_show_error_payload(error_message),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(error_message)
        return 1

    if args.json:
        print(
            json.dumps(
                build_history_show_json_payload(record),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_history_show_human(record))
    return 0


def _run_history_compare(args: argparse.Namespace) -> int:
    try:
        left = load_history_run(args.left_run_id, db_path=args.db_path)
        right = load_history_run(args.right_run_id, db_path=args.db_path)
        if left is None:
            raise ValueError(f"History run not found: {args.left_run_id}")
        if right is None:
            raise ValueError(f"History run not found: {args.right_run_id}")
        comparison = compare_history_runs(left, right)
    except Exception as error:  # noqa: BLE001 - CLI boundary
        error_message = _format_exception_message(error)
        if args.json:
            print(
                json.dumps(
                    build_history_compare_error_payload(error_message),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(render_history_error_human("GPUBoost History Compare", error_message))
        return 1

    if args.json:
        print(
            json.dumps(
                build_history_compare_json_payload(comparison),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_history_compare_human(comparison))
    return 0
























































































def comparison_status_to_exit_code(status: str) -> int:
    """Return the CLI exit code for a comparison status."""

    if status in {"ok", "partial"}:
        return 0
    return 1








def _parse_hidden_sizes(value: str | None) -> list[int] | None:
    if value is None:
        return None
    hidden_sizes: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            hidden_size = int(stripped)
        except ValueError as error:
            raise ValueError(f"Invalid --hidden-sizes value: {value}") from error
        if hidden_size <= 0:
            raise ValueError("--hidden-sizes values must be positive integers.")
        hidden_sizes.append(hidden_size)
    if not hidden_sizes:
        raise ValueError("--hidden-sizes must include at least one integer.")
    return hidden_sizes


def _attach_baseline_metadata(
    search: NeuralSearchResult,
    baseline_model_name: str | None,
    baseline_macro_f1: float | None,
) -> None:
    search.metadata["best_baseline_model_name"] = baseline_model_name
    search.metadata["best_baseline_macro_f1"] = baseline_macro_f1
    for candidate in search.candidates:
        candidate.baseline_comparison["best_baseline_model_name"] = baseline_model_name
        candidate.baseline_comparison["best_baseline_macro_f1"] = baseline_macro_f1
        candidate.baseline_comparison["beats_baseline"] = (
            candidate.validation_evaluation is not None
            and candidate.validation_evaluation.macro_f1 is not None
            and baseline_macro_f1 is not None
            and candidate.validation_evaluation.macro_f1 > baseline_macro_f1
        )


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None




def _load_cli_features(
    features_json: str | None,
    features_file: str | None,
) -> dict[str, object]:
    if features_json and features_file:
        raise ValueError("Use either --features-json or --features-file, not both.")
    if features_file:
        data = json.loads(Path(features_file).read_text(encoding="utf-8"))
    elif features_json:
        data = json.loads(features_json)
    else:
        raise ValueError("predict-artifact requires --features-json or --features-file.")
    if not isinstance(data, dict):
        raise ValueError("Artifact prediction features must be a JSON object.")
    return data


def check_model_artifact_quality(
    manifest_path: str,
    *,
    min_test_macro_f1: float | None = None,
    require_beats_baseline: bool = False,
    require_target_met: bool = False,
) -> dict[str, object]:
    summary = summarize_model_artifact(manifest_path)
    checks: list[dict[str, object]] = []

    validation_ok = summary.get("validation_status") == "ok"
    checks.append(
        {
            "name": "valid_artifact",
            "status": "passed" if validation_ok else "failed",
            "message": "artifact validation passed"
            if validation_ok
            else "artifact validation failed",
        }
    )

    if min_test_macro_f1 is not None:
        test_macro_f1 = summary.get("test_macro_f1")
        metric_ok = isinstance(test_macro_f1, int | float) and (
            float(test_macro_f1) >= min_test_macro_f1
        )
        checks.append(
            {
                "name": "min_test_macro_f1",
                "status": "passed" if metric_ok else "failed",
                "threshold": min_test_macro_f1,
                "value": test_macro_f1,
                "message": "test macro F1 meets threshold"
                if metric_ok
                else "test macro F1 does not meet threshold",
            }
        )

    if require_beats_baseline:
        beats_baseline = summary.get("beats_baseline") is True
        checks.append(
            {
                "name": "beats_baseline",
                "status": "passed" if beats_baseline else "failed",
                "message": "artifact beats baseline"
                if beats_baseline
                else "artifact does not beat baseline",
            }
        )

    if require_target_met:
        target_met = summary.get("target_met") is True
        checks.append(
            {
                "name": "target_met",
                "status": "passed" if target_met else "failed",
                "message": "artifact target was met"
                if target_met
                else "artifact target was not met",
            }
        )

    passed = all(check.get("status") == "passed" for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "summary": summary,
    }




def _format_json_file_error(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return f"File not found: {error.filename}"
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON: {error.msg} at line {error.lineno} column {error.colno}"
    return _format_exception_message(error)


















def _run_agent_prepare(args: argparse.Namespace) -> int:
    validation_error = _validate_agent_optimize_args(args)
    if validation_error is not None:
        return _emit_agentic_error(
            validation_error,
            json_output=args.json,
            command="agent optimize --prepare",
        )
    try:
        run = prepare_optimization_run(
            args.script_path,
            repo_root=args.repo_root,
            trial=args.trial,
            action_ids=args.action_ids,
            exclude_action_ids=args.exclude_action_ids,
            acceptance_policy=args.acceptance_policy,
            min_speedup_percent=args.min_speedup_percent,
            max_regression_percent=args.max_regression_percent,
            benchmark_command=args.benchmark_command,
        )
        if args.interactive_apply:
            if args.json:
                return _emit_agentic_error(
                    "--interactive-apply cannot be used with --json.",
                    json_output=True,
                    command="agent optimize --prepare",
                )
            print(_render_agentic_run_human(run, command="optimize --prepare"))
            confirmation = input(
                f'Type "{_confirmation_phrase(run)}" to approve this plan: '
            )
            approved = approve_optimization_run(
                run.run_id,
                approved_by="local-user",
                confirmation_phrase=confirmation,
                repo_root=args.repo_root,
            )
            applied = apply_approved_optimization_run(
                approved.run_id,
                repo_root=args.repo_root,
                backup_dir=args.backup_dir,
                dry_run=args.dry_run,
                validation_command=args.validation_command,
                test_command=args.test_command,
                benchmark_command=args.benchmark_command,
                acceptance_policy=args.acceptance_policy,
                min_speedup_percent=args.min_speedup_percent,
                max_regression_percent=args.max_regression_percent,
            )
            print(_render_agentic_run_human(applied, command="apply"))
            return _agentic_apply_exit_code(applied)
    except Exception as error:  # noqa: BLE001 - CLI boundary
        return _emit_agentic_error(
            _format_exception_message(error),
            json_output=args.json,
            command="agent optimize --prepare",
        )

    if args.json:
        print(
            json.dumps(
                _agentic_payload(
                    command="agent optimize --prepare",
                    run=run,
                    include_confirmation=True,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_render_agentic_run_human(run, command="optimize --prepare"))
    return 0 if run.lifecycle_status.value != "FAILED" else 1


def _run_agent_show_plan(args: argparse.Namespace) -> int:
    try:
        run = load_agentic_run(args.run_id, repo_root=args.repo_root)
    except Exception as error:  # noqa: BLE001 - CLI boundary
        return _emit_agentic_error(
            _format_exception_message(error),
            json_output=args.json,
            command="agent show-plan",
        )
    if args.json:
        print(
            json.dumps(
                _agentic_payload(
                    command="agent show-plan",
                    run=run,
                    include_confirmation=True,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_render_agentic_run_human(run, command="show-plan"))
    return 0


def _run_agent_approve(args: argparse.Namespace) -> int:
    try:
        confirmation = args.confirm
        if confirmation is None:
            if args.json:
                return _emit_agentic_error(
                    "--confirm is required with --json.",
                    json_output=True,
                    command="agent approve",
                )
            run = load_agentic_run(args.run_id, repo_root=args.repo_root)
            confirmation = input(
                f'Type "{_confirmation_phrase(run)}" to approve this plan: '
            )
        approved = approve_optimization_run(
            args.run_id,
            approved_by=args.approved_by,
            confirmation_phrase=confirmation,
            approved_action_ids=args.action_ids,
            repo_root=args.repo_root,
        )
    except Exception as error:  # noqa: BLE001 - CLI boundary
        return _emit_agentic_error(
            _format_exception_message(error),
            json_output=args.json,
            command="agent approve",
        )
    return _emit_agentic_success(approved, args.json, "agent approve")


def _run_agent_reject(args: argparse.Namespace) -> int:
    try:
        run = reject_optimization_run(args.run_id, repo_root=args.repo_root)
    except Exception as error:  # noqa: BLE001 - CLI boundary
        return _emit_agentic_error(
            _format_exception_message(error),
            json_output=args.json,
            command="agent reject",
        )
    return _emit_agentic_success(run, args.json, "agent reject")


def _run_agent_apply(args: argparse.Namespace) -> int:
    try:
        run = apply_approved_optimization_run(
            args.run_id,
            repo_root=args.repo_root,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
            validation_command=args.validation_command,
            test_command=args.test_command,
            benchmark_command=args.benchmark_command,
            acceptance_policy=args.acceptance_policy,
            min_speedup_percent=args.min_speedup_percent,
            max_regression_percent=args.max_regression_percent,
        )
    except Exception as error:  # noqa: BLE001 - CLI boundary
        return _emit_agentic_error(
            _format_exception_message(error),
            json_output=args.json,
            command="agent apply",
        )
    if args.json:
        print(
            json.dumps(
                _agentic_payload(command="agent apply", run=run),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_render_agentic_run_human(run, command="apply", include_diff=False))
    return _agentic_apply_exit_code(run)


def _run_agent_rollback(args: argparse.Namespace) -> int:
    try:
        run = rollback_optimization_run(
            args.run_id,
            repo_root=args.repo_root,
            force=args.force,
        )
    except Exception as error:  # noqa: BLE001 - CLI boundary
        return _emit_agentic_error(
            _format_exception_message(error),
            json_output=args.json,
            command="agent rollback",
        )
    return _emit_agentic_success(run, args.json, "agent rollback", include_diff=False)


def _run_agent_status(args: argparse.Namespace) -> int:
    try:
        run = load_agentic_run(args.run_id, repo_root=args.repo_root)
    except Exception as error:  # noqa: BLE001 - CLI boundary
        return _emit_agentic_error(
            _format_exception_message(error),
            json_output=args.json,
            command="agent status",
        )
    return _emit_agentic_success(run, args.json, "agent status", include_diff=False)


def _emit_agentic_success(
    run: AgenticOptimizationRun,
    json_output: bool,
    command: str,
    *,
    include_diff: bool = True,
) -> int:
    if json_output:
        print(
            json.dumps(
                _agentic_payload(command=command, run=run),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            _render_agentic_run_human(
                run,
                command=command.replace("agent ", ""),
                include_diff=include_diff,
            )
        )
    return 0


def _emit_agentic_error(
    message: str,
    *,
    json_output: bool,
    command: str,
) -> int:
    if json_output:
        print(
            json.dumps(
                {
                    "schema_version": "agentic.optimization.cli.v1",
                    "command": command,
                    "run": None,
                    "error": message,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"GPUBoost Agentic Optimization\nCommand: {command}\nError: {message}")
    return 1


def _agentic_payload(
    *,
    command: str,
    run: AgenticOptimizationRun,
    include_confirmation: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "agentic.optimization.cli.v1",
        "command": command,
        "run": run.to_dict(),
    }
    if include_confirmation:
        payload["confirmation_phrase"] = _confirmation_phrase(run)
    return payload


def _render_agentic_run_human(
    run: AgenticOptimizationRun,
    *,
    command: str,
    include_diff: bool = True,
) -> str:
    policy = dict((run.benchmark_result or {}).get("policy", {}))
    lines = [
        "GPUBoost Agentic Optimization",
        f"Command: {command}",
        f"Run ID: {run.run_id}",
        f"Target: {run.target_file}",
        f"Lifecycle: {run.lifecycle_status.value}",
        f"Approval: {run.approval_state.value}",
        f"Plan ID: {run.plan_id}",
        f"Plan digest: {run.plan_digest}",
        f"Original SHA256: {run.original_file_hash}",
        "Source mutation before approval: none",
    ]
    if policy:
        lines.append(
            "Acceptance: "
            f"{policy.get('acceptance_policy', AcceptancePolicy.VALIDATION_ONLY.value)}"
        )
    if run.error:
        lines.append(f"Error: {run.error}")
    if run.application_result:
        lines.append(f"Application: {run.application_result.get('status')}")
    if run.validation_result:
        lines.append(f"Validation: {run.validation_result.get('status')}")
    if run.benchmark_result and run.benchmark_result.get("status") != "not_run":
        lines.append(f"Benchmark: {run.benchmark_result.get('status')}")
    if run.rollback_result:
        lines.append(f"Rollback: {run.rollback_result.get('status')}")

    lines.append("")
    lines.append("Actions:")
    for action in run.proposed_actions:
        lines.append(
            "- "
            f"{action.get('action_id')}: {action.get('title')} "
            f"(risk={action.get('risk')}, confidence={action.get('confidence')}, "
            f"edits={action.get('edit_count')})"
        )
    if not run.proposed_actions:
        lines.append("- none")

    if run.approval_state.value == "awaiting_approval":
        lines.extend(
            [
                "",
                "Approval Commands:",
                f'- approve: gpuboost agent approve {run.run_id} --confirm "{_confirmation_phrase(run)}"',
                f"- apply: gpuboost agent apply {run.run_id}",
            ]
        )

    if include_diff and run.generated_diff:
        lines.extend(["", "Reviewable Diff:", run.generated_diff])
    return "\n".join(lines)


def _confirmation_phrase(run: AgenticOptimizationRun) -> str:
    return f"APPLY {short_plan_id(run.plan_id)}"


def _agentic_apply_exit_code(run: AgenticOptimizationRun) -> int:
    if run.final_status in {"completed", "dry_run"}:
        return 0
    if run.lifecycle_status.value == "COMPLETED":
        return 0
    return 1


def _validate_agent_optimize_args(args: argparse.Namespace) -> str | None:
    if (args.prepare or args.interactive_apply) and not args.script_path:
        return "--prepare requires a script_path."
    if args.interactive_apply and args.json:
        return "--interactive-apply cannot be used with --json."
    if args.trial and not args.script_path:
        return "--trial requires a script_path."
    if (
        args.test_command is not None
        and not args.trial
        and not (args.prepare or args.interactive_apply)
    ):
        return "--test requires --trial."
    if args.test_command is not None and not args.script_path:
        return "--test requires a script_path."
    return None




def agent_status_to_exit_code(status: str) -> int:
    """Return the CLI exit code for an agent result status."""

    if status in {"ok", "partial"}:
        return 0
    return 1


def _format_exception_message(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON: {error.msg} at line {error.lineno} column {error.colno}"
    if isinstance(error, FileNotFoundError) and error.filename:
        return f"File not found: {error.filename}"
    message = str(error)
    if message:
        return message
    return error.__class__.__name__










































if __name__ == "__main__":
    raise SystemExit(main())
