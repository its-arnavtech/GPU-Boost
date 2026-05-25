"""Demo validation report generation for GPUBoost Phase 14."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gpuboost.schemas.comparison import create_timestamp


DEFAULT_OUTPUT_DIR = "data/gpuboost/generated/demo_real_world"
REPORT_JSON_NAME = "demo_validation_report.json"
REPORT_MD_NAME = "demo_validation_report.md"
REPORT_SCHEMA_VERSION = "demo.validation_report.v1"

SAFETY_NOTES = {
    "model_advisory_only": True,
    "patch_application_allowed": False,
    "deterministic_checks_authoritative": True,
    "automatic_patch_application": False,
}
LIMITATIONS = [
    "Synthetic data is used for demo repeatability.",
    "Demo workloads are lightweight approximations of user workloads.",
    "Results vary by hardware, driver, PyTorch version, and system load.",
]
_UNSAFE_KEYS = {
    "raw_source",
    "source",
    "source_code",
    "raw_diff",
    "diff",
    "stdout",
    "stderr",
    "traceback",
    "model_weights",
    "weights",
}


def create_demo_validation_report(
    comparison_results: list[dict],
    model_results: list[dict] | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Create JSON and Markdown demo validation reports from sanitized inputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    workload_summaries = [
        _summarize_comparison(result, index)
        for index, result in enumerate(comparison_results)
        if isinstance(result, dict)
    ]
    advisory_summaries = [
        _summarize_model_result(result, index)
        for index, result in enumerate(model_results or [])
        if isinstance(result, dict)
    ]
    warnings = []
    if not workload_summaries:
        warnings.append("No comparison results were provided.")
    if comparison_results and len(workload_summaries) != len(comparison_results):
        warnings.append("Some comparison results were skipped because they were invalid.")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": create_timestamp(),
        "summary": {
            "workload_count": len(workload_summaries),
            "model_advisory_count": len(advisory_summaries),
            "verdict_counts": _verdict_counts(workload_summaries),
        },
        "workloads": workload_summaries,
        "model_advisory_predictions": advisory_summaries,
        "safety_notes": dict(SAFETY_NOTES),
        "limitations": list(LIMITATIONS),
        "warnings": warnings,
    }

    json_path = output_path / REPORT_JSON_NAME
    md_path = output_path / REPORT_MD_NAME
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    report["output_files"] = {
        "json": str(json_path),
        "markdown": str(md_path),
    }
    return report


def _summarize_comparison(result: dict, index: int) -> dict:
    sections = result.get("sections") if isinstance(result.get("sections"), list) else []
    metric_deltas = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        for metric in section.get("metrics", []):
            if isinstance(metric, dict):
                metric_deltas.append(_summarize_metric(metric))

    return {
        "name": _safe_text(
            result.get("workload_name")
            or result.get("name")
            or result.get("baseline_label")
            or f"workload_{index + 1}"
        ),
        "status": _safe_text(result.get("status") or "unknown"),
        "overall_verdict": _safe_text(result.get("overall_verdict") or "unknown"),
        "baseline_label": _safe_text(result.get("baseline_label") or "baseline"),
        "optimized_label": _safe_text(result.get("optimized_label") or "optimized"),
        "sections": [_summarize_section(section) for section in sections],
        "metric_deltas": metric_deltas,
        "warnings": _safe_text_list(result.get("warnings")),
        "error": _safe_text(result.get("error")) if result.get("error") else None,
    }


def _summarize_section(section: dict) -> dict:
    metrics = section.get("metrics") if isinstance(section.get("metrics"), list) else []
    return {
        "title": _safe_text(section.get("title") or "Benchmark Metrics"),
        "verdict": _safe_text(section.get("verdict") or "unknown"),
        "metric_count": len([metric for metric in metrics if isinstance(metric, dict)]),
    }


def _summarize_metric(metric: dict) -> dict:
    return {
        "name": _safe_text(metric.get("name") or "metric"),
        "unit": _safe_text(metric.get("unit")) if metric.get("unit") is not None else None,
        "before": _safe_scalar(metric.get("before")),
        "after": _safe_scalar(metric.get("after")),
        "absolute_delta": _safe_scalar(metric.get("absolute_delta")),
        "percent_delta": _safe_scalar(metric.get("percent_delta")),
        "direction": _safe_text(metric.get("direction") or "unknown"),
        "summary": _safe_text(metric.get("summary") or ""),
    }


def _summarize_model_result(result: dict, index: int) -> dict:
    recommendation = (
        result.get("recommendation")
        or result.get("prediction")
        or result.get("advice")
        or result.get("title")
        or "model advisory prediction"
    )
    return {
        "name": _safe_text(result.get("name") or f"model_advisory_{index + 1}"),
        "recommendation": _safe_text(recommendation),
        "confidence": _safe_scalar(result.get("confidence")),
        "estimated_speedup": _safe_text(result.get("estimated_speedup"))
        if result.get("estimated_speedup") is not None
        else None,
        "impact": _safe_text(result.get("impact"))
        if result.get("impact") is not None
        else None,
    }


def _render_markdown(report: dict) -> str:
    lines = [
        "# GPUBoost Demo Validation Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        f"- Workloads: {report['summary']['workload_count']}",
        f"- Model advisory predictions: {report['summary']['model_advisory_count']}",
        "",
        "## Workloads",
    ]

    if report["workloads"]:
        for workload in report["workloads"]:
            lines.extend(
                [
                    f"### {workload['name']}",
                    f"- Status: {workload['status']}",
                    f"- Verdict: {workload['overall_verdict']}",
                    f"- Baseline: {workload['baseline_label']}",
                    f"- Optimized: {workload['optimized_label']}",
                ]
            )
            if workload["metric_deltas"]:
                lines.append("- Metric deltas:")
                for metric in workload["metric_deltas"]:
                    lines.append(f"  - {_metric_markdown(metric)}")
            else:
                lines.append("- Metric deltas: none available")
            lines.append("")
    else:
        lines.extend(["No comparison results were provided.", ""])

    lines.extend(["## Model Advisory Predictions"])
    if report["model_advisory_predictions"]:
        for advisory in report["model_advisory_predictions"]:
            lines.append(
                "- "
                f"{advisory['name']}: {advisory['recommendation']} "
                f"(confidence: {_format_optional(advisory['confidence'])})"
            )
    else:
        lines.append("- none provided")

    lines.extend(
        [
            "",
            "## Safety Notes",
            "- Model advisory only: true",
            "- Patch application allowed: false",
            "- Deterministic checks authoritative: true",
            "- No automatic patch application",
            "",
            "## Limitations",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in report["limitations"])

    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])

    lines.append("")
    return "\n".join(lines)


def _metric_markdown(metric: dict) -> str:
    return (
        f"{metric['name']}: {metric['before']} -> {metric['after']} "
        f"({metric['direction']}, delta={_format_optional(metric['absolute_delta'])}, "
        f"pct={_format_optional(metric['percent_delta'])})"
    )


def _verdict_counts(workloads: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for workload in workloads:
        verdict = str(workload.get("overall_verdict") or "unknown")
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def _safe_text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_safe_text(item) for item in value if _safe_text(item)]


def _safe_scalar(value: object) -> str | int | float | bool | None:
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, str):
        return _safe_text(value)
    return None


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if _is_unsafe_key(text):
        return "[omitted]"
    return _remove_private_paths(text)


def _remove_private_paths(text: str) -> str:
    text = re.sub(r"[A-Za-z]:[\\/][^\s,;:]+", "[path]", text)
    text = re.sub(r"(?<!\w)/(?:Users|home|tmp|var|etc)/[^\s,;:]+", "[path]", text)
    return text


def _is_unsafe_key(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in _UNSAFE_KEYS


def _format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
