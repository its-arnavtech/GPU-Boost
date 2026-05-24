"""GPUBoost command-line interface.

The CLI exposes Phase 1 inspection and Phase 2 benchmark commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpuboost.agent.report import AgentReport
from gpuboost.agent.workflow import run_optimize_script_workflow
from gpuboost.advisor.engine import generate_advisor_result
from gpuboost.advisor.utils import format_speedup
from gpuboost.benchmarks.runner import run_full_benchmark, run_quick_benchmark
from gpuboost.code_analysis.runner import analyze_python_file
from gpuboost.comparison.engine import compare_benchmarks
from gpuboost.dataset.outcome_collection import (
    OUTCOME_COLLECTION_SCHEMA_VERSION,
    collect_outcomes_from_pairs_file,
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
from gpuboost.model.neural_training import (
    run_neural_config_search,
    run_neural_hyperparameter_search,
    train_best_neural_model_for_artifact,
    train_neural_model_for_artifact_config,
)
from gpuboost.model.provider import TrainedLocalModelProvider
from gpuboost.model.safety import (
    MODEL_WORKFLOW_SAFETY_SCHEMA_VERSION,
    verify_model_workflow_safety,
)
from gpuboost.model.training_data import load_training_rows_jsonl
from gpuboost.model.training_pipeline import (
    BASELINE_COMPARISON_SCHEMA_VERSION,
    run_baseline_model_comparison,
)
from gpuboost.model.training_reports import (
    DEFAULT_BASELINE_REPORT_DIR,
    write_baseline_comparison_reports,
)
from gpuboost.patching.diff import generate_patch_plan_diff
from gpuboost.patching.planner import create_patch_plan_from_analysis
from gpuboost.schemas.agent import AgentRunResult
from gpuboost.schemas.code_analysis import CodeAnalysisResult, CodeFinding
from gpuboost.schemas.comparison import BenchmarkMetricDelta, ComparisonResult
from gpuboost.schemas.history import (
    HistoryCompareResult,
    HistoryRunRecord,
    HistorySummary,
    HistoryValue,
)
from gpuboost.schemas.model import ModelFeatureSet, ModelInput
from gpuboost.schemas.recommendation import AdvisorResult
from gpuboost.schemas.training import NeuralSearchResult, NeuralTrainingConfig
from gpuboost.utils.formatting import format_benchmark_suite, format_profile


def build_parser() -> argparse.ArgumentParser:
    """Build the GPUBoost argument parser."""

    parser = argparse.ArgumentParser(
        prog="gpuboost",
        description="Inspect NVIDIA GPU and system information.",
    )
    subparsers = parser.add_subparsers(dest="command")

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
    )
    agent_optimize_parser.add_argument(
        "script_path",
        nargs="?",
        help="Optional training script path.",
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
            "Use a local trained model artifact for advisory-only inference; "
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
    )
    model_subparsers = model_parser.add_subparsers(dest="model_command")

    evaluate_baselines_parser = model_subparsers.add_parser(
        "evaluate-baselines",
        help="Evaluate safe structured baseline models without saving artifacts.",
        description=(
            "Evaluate dependency-free baselines on safe encoded training data. "
            "This command does not train a neural model or save model artifacts."
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
            "the explicit --save-artifact flag."
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
        help="Explicitly save a local generated model artifact.",
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
            "List local generated model artifact manifests. This command only "
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
            "weights, raw source, raw diffs, stdout, or stderr."
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
            "Validate a local model artifact and check quality gates. "
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
        help="Require test macro F1 to be at least this value when available.",
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
        help="Validate a saved local generated model artifact manifest.",
    )
    validate_artifact_parser.add_argument("manifest_path")
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
            "advisory only and cannot apply patches."
        ),
    )
    predict_artifact_parser.add_argument("manifest_path")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the GPUBoost CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

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

        print("GPUBoost Agent\nAvailable commands: optimize")
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

    parser.print_help()
    return 0


def load_json_file(filepath: str) -> dict:
    """Load a UTF-8 JSON object from disk."""

    with Path(filepath).open(encoding="utf-8") as file:
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
            search.best_result = single_result
            search.candidates = [single_result]
            search.best_config = single_result.config
            search.best_validation_macro_f1 = (
                single_result.validation_evaluation.macro_f1
                if single_result.validation_evaluation is not None
                else None
            )
            search.best_test_macro_f1 = (
                single_result.test_evaluation.macro_f1
                if single_result.test_evaluation is not None
                else None
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
        result["artifact_manifest"] = manifest_path
        result["artifact_manifest_path"] = manifest_path
        if isinstance(manifest_path, str):
            validation = validate_model_artifact(manifest_path)
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
    result = verify_model_workflow_safety()
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


def build_history_list_json_payload(history: HistorySummary) -> dict[str, object]:
    return {
        "schema_version": "history.list.v1",
        "command": "history list",
        "history": history.to_dict(),
    }


def build_history_list_error_payload(error: str) -> dict[str, object]:
    return {
        "schema_version": "history.list.v1",
        "command": "history list",
        "history": None,
        "error": error,
    }


def build_history_show_json_payload(record: HistoryRunRecord) -> dict[str, object]:
    return {
        "schema_version": "history.show.v1",
        "command": "history show",
        "run": record.to_dict(),
    }


def build_history_show_error_payload(error: str) -> dict[str, object]:
    return {
        "schema_version": "history.show.v1",
        "command": "history show",
        "run": None,
        "error": error,
    }


def build_history_compare_json_payload(
    comparison: HistoryCompareResult,
) -> dict[str, object]:
    return {
        "schema_version": "history.compare.v1",
        "command": "history compare",
        "comparison": comparison.to_dict(),
    }


def build_history_compare_error_payload(error: str) -> dict[str, object]:
    return {
        "schema_version": "history.compare.v1",
        "command": "history compare",
        "comparison": None,
        "error": error,
    }


def build_dataset_collect_outcomes_json_payload(
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": OUTCOME_COLLECTION_SCHEMA_VERSION,
        "command": "dataset collect-outcomes",
        "result": summary,
    }


def build_dataset_collect_outcomes_error_payload(error: str) -> dict[str, object]:
    return {
        "schema_version": OUTCOME_COLLECTION_SCHEMA_VERSION,
        "command": "dataset collect-outcomes",
        "result": None,
        "error": error,
    }


def build_model_evaluate_baselines_json_payload(
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": BASELINE_COMPARISON_SCHEMA_VERSION,
        "command": "model evaluate-baselines",
        "result": result,
    }


def build_model_evaluate_baselines_error_payload(error: str) -> dict[str, object]:
    return {
        "schema_version": BASELINE_COMPARISON_SCHEMA_VERSION,
        "command": "model evaluate-baselines",
        "result": None,
        "error": error,
    }


def build_model_train_neural_json_payload(
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "training.neural_search_result.v1",
        "command": "model train-neural",
        "result": result,
    }


def build_model_train_neural_error_payload(error: str) -> dict[str, object]:
    return {
        "schema_version": "training.neural_search_result.v1",
        "command": "model train-neural",
        "result": None,
        "error": error,
    }


def build_model_list_artifacts_json_payload(
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "training.model_artifacts.list.v1",
        "command": "model list-artifacts",
        "result": result,
    }


def build_model_show_artifact_json_payload(
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "training.model_artifact.show.v1",
        "command": "model show-artifact",
        "result": result,
    }


def build_model_check_artifact_json_payload(
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "training.model_artifact.check.v1",
        "command": "model check-artifact",
        "result": result,
    }


def build_model_safety_check_json_payload(
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": MODEL_WORKFLOW_SAFETY_SCHEMA_VERSION,
        "command": "model safety-check",
        "result": result,
    }


def build_model_validate_artifact_json_payload(
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "training.model_artifact_validation.v1",
        "command": "model validate-artifact",
        "result": result,
    }


def build_model_predict_artifact_json_payload(
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "training.model_artifact_prediction.v1",
        "command": "model predict-artifact",
        "result": result,
    }


def build_model_predict_artifact_error_payload(error: str) -> dict[str, object]:
    return {
        "schema_version": "training.model_artifact_prediction.v1",
        "command": "model predict-artifact",
        "result": None,
        "error": error,
    }


def render_dataset_collect_outcomes_human(summary: dict[str, object]) -> str:
    lines = [
        "GPUBoost Outcome Collection",
        f"Pairs: {summary['pair_count']}",
        f"Collected rows: {summary['collected_row_count']}",
        f"Validation: {summary['validation_status']}",
        "",
        "Labels:",
    ]
    label_counts = summary.get("label_counts")
    if isinstance(label_counts, dict) and label_counts:
        for label in ("improved", "regressed", "neutral", "failed", "unknown"):
            lines.append(f"- {label}: {label_counts.get(label, 0)}")
    else:
        lines.append("- none")

    output_files = summary.get("output_files")
    lines.extend(["", "Output:"])
    if isinstance(output_files, dict):
        for key, value in output_files.items():
            if value:
                lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")

    errors = summary.get("errors")
    if isinstance(errors, list) and errors:
        lines.extend(["", "Errors:"])
        for error in errors:
            if isinstance(error, dict):
                lines.append(
                    "- "
                    f"pair_index={error.get('pair_index')} "
                    f"row_id={error.get('row_id') or 'none'} "
                    f"error={error.get('error')}"
                )
            else:
                lines.append(f"- {error}")

    warnings = summary.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines)


def render_dataset_collect_outcomes_error_human(error: str) -> str:
    return "\n".join(
        [
            "GPUBoost Outcome Collection",
            "Status: error",
            "",
            "Error:",
            f"- {error}",
        ]
    )


def render_model_evaluate_baselines_human(result: dict[str, object]) -> str:
    summary = result.get("dataset_summary")
    summary = summary if isinstance(summary, dict) else {}
    lines = [
        "GPUBoost Baseline Model Evaluation",
        f"Rows: {summary.get('encoded_row_count', 0)} encoded "
        f"of {summary.get('row_count', 0)} total",
        f"Labels: {_format_training_counts(summary.get('label_counts'))}",
        f"Eval split: {result.get('eval_split_used') or 'none'}",
        f"Best model: {result.get('best_model_name') or 'none'}",
        "",
        "Model scores:",
    ]

    models = result.get("models")
    if isinstance(models, list) and models:
        for model in models:
            if not isinstance(model, dict):
                continue
            evaluation = model.get("evaluation")
            evaluation = evaluation if isinstance(evaluation, dict) else {}
            lines.append(
                "- "
                f"{model.get('model_name')}: "
                f"accuracy={_format_optional_score(evaluation.get('accuracy'))}, "
                f"macro_f1={_format_optional_score(evaluation.get('macro_f1'))}, "
                f"status={evaluation.get('status') or 'unknown'}"
            )
    else:
        lines.append("- none")

    output_files = result.get("output_files")
    if isinstance(output_files, dict):
        lines.extend(["", "Reports:"])
        lines.extend(f"- {key}: {value}" for key, value in output_files.items())

    warnings = result.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(
        [
            "",
            "Safety: no production model artifact was saved and no agent "
            "integration was changed.",
        ]
    )
    return "\n".join(lines)


def render_model_evaluate_baselines_error_human(error: str) -> str:
    return "\n".join(
        [
            "GPUBoost Baseline Model Evaluation",
            "Status: error",
            "",
            "Error:",
            f"- {error}",
        ]
    )


def render_model_train_neural_human(result: dict[str, object]) -> str:
    best_result = result.get("best_result")
    best_result = best_result if isinstance(best_result, dict) else {}
    validation = best_result.get("validation_evaluation")
    validation = validation if isinstance(validation, dict) else {}
    test = best_result.get("test_evaluation")
    test = test if isinstance(test, dict) else {}
    baseline = best_result.get("baseline_comparison")
    baseline = baseline if isinstance(baseline, dict) else {}
    dataset_summary = _baseline_dataset_summary(result.get("baseline_comparison"))
    lines = [
        "GPUBoost Neural Model Training",
        f"Status: {result.get('status') or 'unknown'}",
        f"Dataset rows: {dataset_summary.get('encoded_row_count', 0)} encoded "
        f"of {dataset_summary.get('row_count', 0)} total",
        f"Classes: {dataset_summary.get('encoded_class_count', 0)}",
        f"Best validation macro F1: "
        f"{_format_optional_score(result.get('best_validation_macro_f1'))}",
        f"Test macro F1: {_format_optional_score(result.get('best_test_macro_f1'))}",
        f"Best baseline macro F1: "
        f"{_format_optional_score(baseline.get('best_baseline_macro_f1'))}",
        f"Beats baseline: {_format_yes_no(result.get('beats_baseline') is True)}",
        f"Target macro F1: {_format_optional_score(result.get('target_macro_f1'))}",
        f"Target met: {_format_yes_no(result.get('target_met') is True)}",
        f"Validation accuracy: {_format_optional_score(validation.get('accuracy'))}",
        f"Test accuracy: {_format_optional_score(test.get('accuracy'))}",
    ]

    output_files = result.get("output_files")
    lines.extend(["", "Output files:"])
    if isinstance(output_files, dict) and output_files:
        lines.extend(f"- {key}: {value}" for key, value in output_files.items())
    else:
        lines.append("- none")

    manifest_path = result.get("artifact_manifest_path") or result.get(
        "artifact_manifest"
    )
    if isinstance(manifest_path, str) and manifest_path:
        lines.extend(
            [
                "",
                "Artifact:",
                f"- Manifest: {manifest_path}",
                f"- Validation status: "
                f"{result.get('artifact_validation_status') or 'unknown'}",
                "- Patch application allowed: no",
                "- Validate: "
                f"python -m gpuboost model validate-artifact {manifest_path}",
                "- Use in agent: "
                "python -m gpuboost agent optimize <script> "
                f"--model-artifact {manifest_path}",
            ]
        )

    warnings = result.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)

    if isinstance(manifest_path, str) and manifest_path:
        safety = (
            "Safety: artifact saved only because --save-artifact was provided; "
            "model predictions remain advisory and cannot apply patches."
        )
    else:
        safety = (
            "Safety: no production model artifact was saved and no agent "
            "integration was changed."
        )
    lines.extend(["", safety])
    return "\n".join(lines)


def render_model_train_neural_error_human(error: str) -> str:
    return "\n".join(
        [
            "GPUBoost Neural Model Training",
            "Status: error",
            "",
            "Error:",
            f"- {error}",
        ]
    )


def render_model_list_artifacts_human(result: dict[str, object]) -> str:
    artifacts = result.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    lines = [
        "GPUBoost Model Artifacts",
        f"Found: {result.get('artifact_count', len(artifacts))}",
    ]
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        lines.extend(
            [
                "",
                f"- path: {artifact.get('manifest_path') or 'unknown'}",
                f"  model name: {artifact.get('model_name') or 'unknown'}",
                "  validation macro F1: "
                f"{_format_optional_score(artifact.get('validation_macro_f1'))}",
                "  test macro F1: "
                f"{_format_optional_score(artifact.get('test_macro_f1'))}",
                "  beats baseline: "
                f"{_format_yes_no(artifact.get('beats_baseline') is True)}",
                "  target met: "
                f"{_format_yes_no(artifact.get('target_met') is True)}",
                "  validation status: "
                f"{artifact.get('validation_status') or 'unknown'}",
            ]
        )
    return "\n".join(lines)


def render_model_show_artifact_human(summary: dict[str, object]) -> str:
    lines = [
        "GPUBoost Model Artifact",
        f"Path: {summary.get('manifest_path') or 'unknown'}",
        f"Status: {summary.get('validation_status') or 'unknown'}",
        f"Model: {summary.get('model_name') or 'unknown'}",
        f"Labels: {_format_label_list(summary.get('labels'))}",
        f"Feature count: {summary.get('feature_count') or 0}",
        "Validation macro F1: "
        f"{_format_optional_score(summary.get('validation_macro_f1'))}",
        f"Test macro F1: {_format_optional_score(summary.get('test_macro_f1'))}",
        "Baseline macro F1: "
        f"{_format_optional_score(summary.get('baseline_macro_f1'))}",
        f"Beats baseline: {_format_yes_no(summary.get('beats_baseline') is True)}",
        f"Target met: {_format_yes_no(summary.get('target_met') is True)}",
    ]
    warnings = summary.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)
    errors = summary.get("validation_errors")
    if isinstance(errors, list) and errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines)


def render_model_check_artifact_human(result: dict[str, object]) -> str:
    lines = [
        "GPUBoost Model Artifact Check",
        f"Status: {result.get('status') or 'unknown'}",
    ]
    checks = result.get("checks")
    if isinstance(checks, list) and checks:
        lines.extend(["", "Checks:"])
        for check in checks:
            if isinstance(check, dict):
                lines.append(
                    "- "
                    f"{check.get('name')}: {check.get('status')} "
                    f"({check.get('message')})"
                )
    return "\n".join(lines)


def render_model_safety_check_human(result: dict[str, object]) -> str:
    lines = [
        "GPUBoost Model Workflow Safety Check",
        f"Status: {result.get('status') or 'unknown'}",
        "",
        "Checks:",
    ]
    for key in (
        "generated_dir_ignored",
        "artifact_extensions_ignored",
        "raw_data_ignored",
        "model_patch_application_allowed_false_documented",
        "provider_patch_application_allowed_false",
        "no_default_artifact_path_required",
    ):
        lines.append(f"- {key}: {_format_yes_no(result.get(key) is True)}")
    warnings = result.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def render_model_validate_artifact_human(result: dict[str, object]) -> str:
    lines = [
        "GPUBoost Model Artifact Validation",
        f"Status: {result.get('status') or 'unknown'}",
    ]
    summary = result.get("manifest_summary")
    if isinstance(summary, dict) and summary:
        lines.extend(
            [
                f"Model: {summary.get('model_name') or 'unknown'}",
                f"Input size: {summary.get('input_size') or 0}",
                f"Output size: {summary.get('output_size') or 0}",
            ]
        )
    errors = result.get("errors")
    if isinstance(errors, list) and errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in errors)
    warnings = result.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def render_model_predict_artifact_human(result: dict[str, object]) -> str:
    lines = [
        "GPUBoost Model Artifact Prediction",
        f"Status: {result.get('status') or 'unknown'}",
    ]
    predictions = result.get("predictions")
    if isinstance(predictions, list) and predictions:
        prediction = predictions[0]
        if isinstance(prediction, dict):
            lines.extend(
                [
                    f"Label: {prediction.get('label') or 'unknown'}",
                    f"Confidence: "
                    f"{_format_optional_score(prediction.get('confidence'))}",
                ]
            )
    errors = result.get("error")
    if errors:
        lines.extend(["", "Error:", f"- {errors}"])
    return "\n".join(lines)


def render_model_predict_artifact_error_human(error: str) -> str:
    return "\n".join(
        [
            "GPUBoost Model Artifact Prediction",
            "Status: error",
            "",
            "Error:",
            f"- {error}",
        ]
    )


def render_history_list_human(history: HistorySummary) -> str:
    lines = [
        "GPUBoost History",
        f"Total runs: {history.total_runs}",
    ]
    if not history.runs:
        lines.extend(["", "No history runs found."])
        return "\n".join(lines)

    lines.append("")
    for record in history.runs:
        lines.append(
            "- "
            f"{record.run_id} | {record.status} | {record.command} | "
            f"{record.created_at} | script={record.script_path or 'none'} | "
            f"trial={record.trial_summary.get('test_status') or record.trial_summary.get('status') or 'none'}"
        )
    return "\n".join(lines)


def render_history_show_human(record: HistoryRunRecord) -> str:
    lines = [
        "GPUBoost History Run",
        f"Run ID: {record.run_id}",
        f"Status: {record.status}",
        f"Command: {record.command}",
        f"Created: {record.created_at}",
        f"Goal: {record.goal_kind} - {record.goal_description}",
        f"Script: {record.script_path or 'none'}",
        f"Script SHA256: {record.script_sha256 or 'none'}",
        f"GPU: {record.gpu_name or 'unknown'}",
        f"CUDA available: {_format_optional_bool(record.cuda_available)}",
        "",
        "Actions:",
    ]
    if record.action_statuses:
        lines.extend(
            f"- {action_name}: {status}"
            for action_name, status in record.action_statuses.items()
        )
    else:
        lines.append("- none")

    summary_sections = [
        ("benchmark", record.benchmark_summary),
        ("advisor", record.advisor_summary),
        ("code", record.code_summary),
        ("patch", record.patch_summary),
        ("trial", record.trial_summary),
        ("comparison", record.comparison_summary),
    ]
    non_empty_sections = [(name, data) for name, data in summary_sections if data]
    if non_empty_sections:
        lines.extend(["", "Summaries:"])
        for name, data in non_empty_sections:
            lines.append(f"- {name}: {_format_history_summary_dict(data)}")

    if record.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in record.warnings)

    if record.error:
        lines.extend(["", "Error:", f"- {record.error}"])

    return "\n".join(lines)


def render_history_compare_human(comparison: HistoryCompareResult) -> str:
    lines = [
        "GPUBoost History Compare",
        f"Left: {comparison.left_run_id}",
        f"Right: {comparison.right_run_id}",
        f"Status: {comparison.status}",
        f"Summary: {comparison.summary}",
    ]
    if comparison.changed_fields:
        lines.extend(["", "Changed fields:"])
        lines.extend(
            f"- {field_name}: {value}"
            for field_name, value in comparison.changed_fields.items()
        )
    else:
        lines.extend(["", "No tracked fields changed."])

    if comparison.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in comparison.warnings)
    if comparison.error:
        lines.extend(["", "Error:", f"- {comparison.error}"])

    return "\n".join(lines)


def render_history_error_human(title: str, error: str) -> str:
    return "\n".join([title, "Status: error", "", "Error:", f"- {error}"])


def build_compare_json_payload(result: ComparisonResult) -> dict[str, object]:
    """Build the stable JSON payload for benchmark comparisons."""

    return {
        "schema_version": "comparison.v1",
        "command": "compare",
        "comparison": result.to_dict(),
    }


def build_compare_error_payload(message: str) -> dict[str, object]:
    """Build the stable JSON payload for compare input errors."""

    return {
        "schema_version": "comparison.v1",
        "command": "compare",
        "comparison": None,
        "error": message,
    }


def render_comparison_human(result: ComparisonResult) -> str:
    """Render a concise human-readable benchmark comparison."""

    lines = [
        "GPUBoost Comparison",
        f"Status: {result.status}",
        f"Baseline: {result.baseline_label}",
        f"Optimized: {result.optimized_label}",
        f"Overall verdict: {result.overall_verdict}",
    ]

    for section in result.sections:
        lines.extend(["", f"{section.title}:"])
        if section.metrics:
            lines.extend(
                f"- {metric.name}: {_format_metric_delta_line(metric)}"
                for metric in section.metrics
            )
        else:
            lines.append("- no comparable metrics")

    if result.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in result.warnings)

    if result.error:
        lines.extend(["", "Error:", f"- {result.error}"])

    return "\n".join(lines)


def render_compare_error_human(error: str) -> str:
    """Render a clean human-readable compare input error."""

    return "\n".join(
        [
            "GPUBoost Comparison",
            "Status: error",
            "",
            "Error:",
            f"- {error}",
        ]
    )


def comparison_status_to_exit_code(status: str) -> int:
    """Return the CLI exit code for a comparison status."""

    if status in {"ok", "partial"}:
        return 0
    return 1


def _format_metric_delta_line(metric: BenchmarkMetricDelta) -> str:
    unit_text = f" {metric.unit}" if metric.unit else ""
    percent_text = ""
    if metric.percent_delta is not None:
        percent_text = f" ({metric.percent_delta:+.2f}%)"

    return (
        f"{metric.before} -> {metric.after}{unit_text}"
        f"{percent_text} [{metric.direction}]"
    )


def _format_training_counts(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}={item}" for key, item in sorted(value.items()))


def _format_label_list(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(str(item) for item in value)


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


def _baseline_dataset_summary(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    summary = value.get("dataset_summary")
    return summary if isinstance(summary, dict) else {}


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


def _format_optional_score(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


def _format_json_file_error(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return f"File not found: {error.filename}"
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON: {error.msg} at line {error.lineno} column {error.colno}"
    return _format_exception_message(error)


def _format_advisor_result(advisor_result: AdvisorResult) -> str:
    lines = ["Recommendations:"]

    if not advisor_result.recommendations:
        lines.extend(f"- {warning}" for warning in advisor_result.warnings)
        return "\n".join(lines)

    for recommendation in advisor_result.recommendations:
        lines.extend(
            [
                "",
                f"[{recommendation.priority}] {recommendation.title}",
                "    Impact: "
                f"{recommendation.impact} | Confidence: "
                f"{recommendation.confidence} | Effort: {recommendation.effort}",
                "    Estimated speedup: "
                f"{format_speedup(recommendation.estimated_speedup)}",
                f"    Why: {recommendation.summary}",
                f"    Do: {recommendation.suggested_action}",
            ],
        )

    if advisor_result.warnings:
        lines.extend(["", "Advisor warnings:"])
        lines.extend(f"- {warning}" for warning in advisor_result.warnings)

    return "\n".join(lines)


def _format_code_analysis_result(result: CodeAnalysisResult) -> str:
    lines = [
        "GPUBoost Code Analysis",
        f"File: {result.filepath}",
        f"Status: {result.status}",
    ]

    if result.error:
        lines.extend(["", f"Error: {result.error}"])
        if result.warnings:
            _append_code_analysis_warnings(lines, result.warnings)
        return "\n".join(lines)

    if not result.findings:
        lines.extend(["", "No performance findings detected."])
    else:
        for finding in result.findings:
            lines.extend(
                [
                    "",
                    f"[{finding.severity}] {finding.title}",
                    f"  Location: {_format_code_finding_location(finding)}",
                    "  Category: "
                    f"{finding.category} | Confidence: {finding.confidence}",
                    f"  Why: {finding.rationale}",
                    f"  Do: {finding.suggested_action}",
                ]
            )

    if result.warnings:
        _append_code_analysis_warnings(lines, result.warnings)

    return "\n".join(lines)


def _format_code_finding_location(finding: CodeFinding) -> str:
    if finding.line is None:
        return finding.filepath

    return f"{finding.filepath}:{finding.line}"


def _create_patch_cli_output(
    filepath: str,
    analysis: CodeAnalysisResult,
) -> dict[str, object]:
    source_text = Path(filepath).read_text(encoding="utf-8")
    patch_plan = create_patch_plan_from_analysis(source_text, analysis)
    diff, patch_warnings = generate_patch_plan_diff(source_text, patch_plan)
    return {
        "patch_plan": patch_plan.to_dict(),
        "diff": diff,
        "patch_warnings": patch_warnings,
    }


def _format_patch_output(patch_output: dict[str, object]) -> str:
    diff = patch_output["diff"]
    patch_warnings = patch_output["patch_warnings"]
    lines = [
        "Patch Suggestions:",
        "GPUBoost does not apply patches automatically. "
        "Review the diff before applying changes.",
        "",
    ]

    if isinstance(diff, str) and diff:
        lines.append(diff)
    else:
        lines.append("No safe automatic patch suggestions were generated.")

    if isinstance(patch_warnings, list) and patch_warnings:
        lines.extend(["", "Patch Warnings:"])
        lines.extend(f"- {warning}" for warning in patch_warnings)

    return "\n".join(lines)


def render_agent_report_human(
    report: AgentReport,
    result: AgentRunResult,
    script_path: str | None,
    command: str = "optimize",
    trial_requested: bool = False,
) -> str:
    """Render a concise human-readable agent report."""

    script_display = script_path if script_path is not None else "none"
    lines = [
        "GPUBoost Agent",
        f"Command: {command}",
        f"Status: {report.status}",
        f"Script: {script_display}",
        "",
        "Summary:",
        report.summary,
        "",
        "Plan:",
    ]

    if result.plan.actions:
        lines.extend(
            f"- {action.id}: {action.status}"
            for action in result.plan.actions
        )
    else:
        lines.append("- none")

    event_items: list[str] = []
    warning_items = list(report.warnings)
    diff = _get_agent_diff_artifact(result)
    trial = _get_agent_trial_artifact(result)
    model = _get_agent_model_artifact(result)
    comparison = _get_agent_comparison_artifact(result)
    error_items = [
        error
        for error in [result.error, report.error]
        if error
    ]

    report_lines: list[str] = []
    for section in report.sections:
        title_key = section.title.lower()
        if title_key == "events":
            event_items.extend(section.items[-5:])
            continue
        if title_key == "warnings":
            warning_items.extend(section.items)
            continue
        if title_key == "errors":
            error_items.extend(section.items)
            continue

        report_lines.extend([section.title])
        report_lines.extend(f"- {item}" for item in section.items)
        report_lines.append("")

    if report_lines:
        lines.extend(["", "Report:"])
        lines.extend(report_lines)
        if lines[-1] == "":
            lines.pop()

    if warning_items:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in _deduplicate_lines(warning_items))

    if error_items:
        lines.extend(["", "Error:"])
        lines.extend(f"- {error}" for error in _deduplicate_lines(error_items))

    if event_items:
        lines.extend(["", "Recent Events:"])
        lines.extend(f"- {event}" for event in event_items[-5:])

    if diff:
        lines.extend(
            [
                "",
                "Reviewable Patch Diff:",
                "GPUBoost does not apply patches automatically. "
                "Review the diff before applying changes.",
                "",
                diff,
            ]
        )

    if trial_requested or trial is not None:
        lines.extend(["", _format_trial_output(trial)])

    if model is not None:
        lines.extend(["", _format_model_output(model)])

    if comparison is not None:
        lines.extend(["", _format_agent_comparison_output(comparison)])

    history_run_id = _get_agent_history_run_id(result)
    if history_run_id is not None:
        lines.extend(["", "History:", f"- Saved run: {history_run_id}"])

    lines.extend(
        [
            "",
            "Safety:",
            "GPUBoost does not apply patches automatically. "
            "Review generated diffs before applying changes.",
        ]
    )

    return "\n".join(lines)


def build_agent_optimize_json_payload(
    result: AgentRunResult,
    report: AgentReport,
    include_raw_artifacts: bool = False,
) -> dict[str, object]:
    """Build the stable JSON payload for agent optimize."""

    result_dict = result.to_dict()
    if not include_raw_artifacts:
        _redact_agent_result_artifacts(result_dict)

    return {
        "schema_version": "agent.optimize.v1",
        "command": "agent optimize",
        "result": result_dict,
        "report": report.to_dict(),
        "artifacts": _build_agent_json_artifacts(
            result,
            include_raw_artifacts=include_raw_artifacts,
        ),
    }


def build_agent_optimize_error_json_payload(error: str) -> dict[str, object]:
    """Build the stable JSON payload for unexpected agent optimize errors."""

    return {
        "schema_version": "agent.optimize.v1",
        "command": "agent optimize",
        "result": None,
        "report": None,
        "artifacts": {
            "diff": None,
            "trial": None,
            "comparison": None,
            "history_run_id": None,
            "model": None,
        },
        "error": error,
    }


def _validate_agent_optimize_args(args: argparse.Namespace) -> str | None:
    if args.trial and not args.script_path:
        return "--trial requires a script_path."
    if args.test_command is not None and not args.trial:
        return "--test requires --trial."
    if args.test_command is not None and not args.script_path:
        return "--test requires a script_path."
    return None


def render_agent_unexpected_error_human(error: str) -> str:
    """Render a clean human-readable unexpected agent error."""

    return "\n".join(
        [
            "GPUBoost Agent",
            "Command: optimize",
            "Status: error",
            "",
            "Error:",
            error,
        ]
    )


def agent_status_to_exit_code(status: str) -> int:
    """Return the CLI exit code for an agent result status."""

    if status in {"ok", "partial"}:
        return 0
    return 1


def _format_exception_message(error: Exception) -> str:
    message = str(error)
    if message:
        return message
    return error.__class__.__name__


def _get_agent_diff_artifact(result: AgentRunResult) -> str | None:
    diff = result.artifacts.get("diff")
    if isinstance(diff, str) and diff:
        return diff
    return None


def _build_agent_json_artifacts(
    result: AgentRunResult,
    *,
    include_raw_artifacts: bool,
) -> dict[str, object]:
    raw_diff = _get_agent_diff_artifact(result)
    raw_trial = _get_agent_trial_artifact(result)
    if include_raw_artifacts:
        diff = raw_diff
        trial = raw_trial
        diff_redacted = False
    else:
        diff = None
        trial = _redacted_trial_artifact(raw_trial)
        diff_redacted = raw_diff is not None

    return {
        "diff": diff,
        "diff_redacted": diff_redacted,
        "trial": trial,
        "comparison": _get_agent_comparison_artifact(result),
        "history_run_id": _get_agent_history_run_id(result),
        "model": _get_agent_model_artifact(result),
        "raw_artifacts_included": include_raw_artifacts,
    }


def _redact_agent_result_artifacts(result_dict: dict[str, object]) -> None:
    artifacts = result_dict.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    raw_diff = artifacts.get("diff")
    artifacts["diff"] = None
    artifacts["diff_redacted"] = isinstance(raw_diff, str) and bool(raw_diff)
    artifacts["trial"] = _redacted_trial_artifact(artifacts.get("trial"))
    if isinstance(artifacts.get("model"), dict):
        artifacts["model"] = _summarize_agent_model_artifact(artifacts["model"])
    artifacts["raw_artifacts_included"] = False


def _redacted_trial_artifact(trial: object) -> dict[str, object] | None:
    if not isinstance(trial, dict):
        return None

    redacted = dict(trial)
    steps = redacted.get("steps")
    if isinstance(steps, list):
        redacted_steps = []
        for step in steps:
            if not isinstance(step, dict):
                redacted_steps.append(step)
                continue
            redacted_step = dict(step)
            for stream_name in ("stdout", "stderr"):
                value = redacted_step.get(stream_name)
                was_redacted = isinstance(value, str) and bool(value)
                if was_redacted:
                    redacted_step[stream_name] = None
                redacted_step[f"{stream_name}_redacted"] = was_redacted
            redacted_steps.append(redacted_step)
        redacted["steps"] = redacted_steps
    return redacted


def _get_agent_trial_artifact(result: AgentRunResult) -> dict[str, object] | None:
    trial = result.artifacts.get("trial")
    if isinstance(trial, dict):
        return trial
    return None


def _get_agent_model_artifact(result: AgentRunResult) -> dict[str, object] | None:
    model = result.artifacts.get("model")
    if isinstance(model, dict):
        return _summarize_agent_model_artifact(model)
    return None


def _summarize_agent_model_artifact(model: dict[str, object]) -> dict[str, object]:
    prediction = _first_model_prediction(model)
    metadata = model.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    probabilities = None
    if prediction is not None:
        prediction_metadata = prediction.get("metadata")
        if isinstance(prediction_metadata, dict):
            raw_probabilities = prediction_metadata.get("probabilities")
            if isinstance(raw_probabilities, dict):
                probabilities = raw_probabilities
    return {
        "status": model.get("status"),
        "provider": metadata.get("provider") or "unknown",
        "prediction": {
            "label": prediction.get("label") if prediction else None,
            "confidence": prediction.get("confidence") if prediction else None,
        },
        "probabilities": probabilities,
        "patch_application_allowed": bool(
            metadata.get("patch_application_allowed", False)
        ),
        "warnings": model.get("warnings") if isinstance(model.get("warnings"), list) else [],
        "model_available": model.get("model_available"),
        "fallback_used": model.get("fallback_used"),
        "model_name": model.get("model_name"),
        "model_version": model.get("model_version"),
        "error": model.get("error"),
    }


def _first_model_prediction(model: dict[str, object]) -> dict[str, object] | None:
    predictions = model.get("predictions")
    if isinstance(predictions, list) and predictions:
        first = predictions[0]
        if isinstance(first, dict):
            return first
    return None


def _get_agent_comparison_artifact(
    result: AgentRunResult,
) -> dict[str, object] | None:
    comparison = result.artifacts.get("comparison")
    if isinstance(comparison, dict):
        return comparison
    return None


def _get_agent_history_run_id(result: AgentRunResult) -> str | None:
    history_run_id = result.artifacts.get("history_run_id")
    if isinstance(history_run_id, str) and history_run_id:
        return history_run_id
    return None


def _format_agent_comparison_output(comparison: dict[str, object]) -> str:
    return "\n".join(
        [
            "Comparison:",
            f"- Status: {comparison.get('status') or 'unknown'}",
            f"- Overall verdict: {comparison.get('overall_verdict') or 'unknown'}",
        ]
    )


def _format_model_output(model: dict[str, object]) -> str:
    model_name = model.get("model") or model.get("model_name") or model.get("name")
    model_version = model.get("version") or model.get("model_version")
    if model_name and model_version:
        model_display = f"{model_name}/{model_version}"
    elif model_name or model_version:
        model_display = str(model_name or model_version)
    else:
        model_display = "none"

    return "\n".join(
        [
            "Model:",
            f"- Provider: {model.get('provider') or 'unknown'}",
            f"- Status: {model.get('status') or 'unknown'}",
            f"- Available: {_format_yes_no(bool(model.get('model_available')))}",
            f"- Fallback used: {_format_yes_no(bool(model.get('fallback_used')))}",
            f"- Model: {model_display}",
            f"- Prediction: {_model_prediction_label(model)}",
            f"- Confidence: {_format_optional_score(_model_prediction_confidence(model))}",
            "- Patch application allowed: "
            f"{_format_yes_no(bool(model.get('patch_application_allowed')))}",
            "- Safety: model prediction is advisory only; deterministic checks "
            "remain authoritative.",
        ]
    )


def _model_prediction_label(model: dict[str, object]) -> str:
    prediction = model.get("prediction")
    if isinstance(prediction, dict) and prediction.get("label"):
        return str(prediction["label"])
    return "none"


def _model_prediction_confidence(model: dict[str, object]) -> object:
    prediction = model.get("prediction")
    if isinstance(prediction, dict):
        return prediction.get("confidence")
    return None


def _format_trial_output(trial: dict[str, object] | None) -> str:
    lines = [
        "Trial Workspace:",
        "Trial mode applies patches only to a temporary copy. "
        "The original file is not modified.",
    ]
    if trial is None:
        lines.extend(
            [
                "- Status: none",
                "- Patch applied: no",
                "- Syntax check: none",
                "- Test command: none",
                "- Test status: none",
                "- Original file unchanged: yes",
            ]
        )
        return "\n".join(lines)

    test_command = trial.get("test_command")
    test_status = trial.get("test_status")
    lines.extend(
        [
            f"- Status: {trial.get('status')}",
            f"- Patch applied: {_format_yes_no(bool(trial.get('patch_applied')))}",
            f"- Syntax check: {trial.get('syntax_check_status') or 'none'}",
            f"- Test command: {test_command or 'none'}",
            f"- Test status: {test_status or 'none'}",
            "- Original file unchanged: "
            f"{_format_yes_no(bool(trial.get('original_file_unchanged')))}",
        ]
    )
    error = trial.get("error")
    if error:
        lines.append(f"- Error: {error}")
    return "\n".join(lines)


def _format_yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _format_history_summary_dict(data: dict[str, HistoryValue]) -> str:
    return ", ".join(f"{key}={value}" for key, value in data.items())


def _deduplicate_lines(items: list[str]) -> list[str]:
    unique_items = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        unique_items.append(item)
        seen.add(item)
    return unique_items


def _append_code_analysis_warnings(lines: list[str], warnings: list[str]) -> None:
    lines.extend(["", "Warnings:"])
    lines.extend(f"- {warning}" for warning in warnings)


if __name__ == "__main__":
    raise SystemExit(main())
