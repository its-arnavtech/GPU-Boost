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
from gpuboost.schemas.recommendation import AdvisorResult
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
                    "model": args.model,
                }
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
                        build_agent_optimize_json_payload(result, report),
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
) -> dict[str, object]:
    """Build the stable JSON payload for agent optimize."""

    return {
        "schema_version": "agent.optimize.v1",
        "command": "agent optimize",
        "result": result.to_dict(),
        "report": report.to_dict(),
        "artifacts": {
            "diff": _get_agent_diff_artifact(result),
            "trial": _get_agent_trial_artifact(result),
            "comparison": _get_agent_comparison_artifact(result),
            "history_run_id": _get_agent_history_run_id(result),
            "model": _get_agent_model_artifact(result),
        },
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


def _get_agent_trial_artifact(result: AgentRunResult) -> dict[str, object] | None:
    trial = result.artifacts.get("trial")
    if isinstance(trial, dict):
        return trial
    return None


def _get_agent_model_artifact(result: AgentRunResult) -> dict[str, object] | None:
    model = result.artifacts.get("model")
    if isinstance(model, dict):
        return model
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
            f"- Status: {model.get('status') or 'unknown'}",
            f"- Available: {_format_yes_no(bool(model.get('model_available')))}",
            f"- Fallback used: {_format_yes_no(bool(model.get('fallback_used')))}",
            f"- Model: {model_display}",
        ]
    )


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
