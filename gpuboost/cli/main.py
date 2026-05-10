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
from gpuboost.inspector.profile import collect_profile
from gpuboost.patching.diff import generate_patch_plan_diff
from gpuboost.patching.planner import create_patch_plan_from_analysis
from gpuboost.schemas.agent import AgentRunResult
from gpuboost.schemas.code_analysis import CodeAnalysisResult, CodeFinding
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

    if args.command == "agent":
        if args.agent_command == "optimize":
            result, report = run_optimize_script_workflow(
                script_path=args.script_path,
                quick=args.quick,
            )
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
                )
                try:
                    from rich.console import Console
                except Exception:
                    print(output)
                else:
                    Console().print(output, markup=False, soft_wrap=True)
            return 1 if result.status == "error" else 0

        print("GPUBoost Agent\nAvailable commands: optimize")
        return 0

    parser.print_help()
    return 0


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
        lines.extend(f"- {action.id}: {action.status}" for action in result.plan.actions)
    else:
        lines.append("- none")

    event_items: list[str] = []
    warning_items = list(report.warnings)
    diff = _get_agent_diff_artifact(result)
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
        },
    }


def _get_agent_diff_artifact(result: AgentRunResult) -> str | None:
    diff = result.artifacts.get("diff")
    if isinstance(diff, str) and diff:
        return diff
    return None


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
