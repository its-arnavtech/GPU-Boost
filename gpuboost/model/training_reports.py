"""Report writing for Phase 12.2 baseline comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_BASELINE_REPORT_DIR = "data/gpuboost/generated/model_training"


def write_baseline_comparison_reports(
    comparison: dict,
    output_dir: str = DEFAULT_BASELINE_REPORT_DIR,
) -> dict[str, str]:
    """Write JSON and Markdown baseline comparison reports."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "baseline_comparison_report.json"
    markdown_path = directory / "baseline_comparison_report.md"

    json_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown_report(comparison), encoding="utf-8")

    return {
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }


def _render_markdown_report(comparison: dict[str, Any]) -> str:
    summary = comparison.get("dataset_summary")
    summary = summary if isinstance(summary, dict) else {}
    warnings = comparison.get("warnings")
    warnings = warnings if isinstance(warnings, list) else []

    lines = [
        "# GPUBoost Baseline Model Evaluation",
        "",
        "This report compares dependency-free structured baselines only. It is "
        "baseline evaluation, not production model integration.",
        "",
        "## Dataset Summary",
        "",
        f"- Rows: {summary.get('row_count', 0)}",
        f"- Encoded rows: {summary.get('encoded_row_count', 0)}",
        f"- Encoded features: {summary.get('encoded_feature_count', 0)}",
        f"- Encoded classes: {summary.get('encoded_class_count', 0)}",
        f"- Labels: {_format_counts(summary.get('label_counts'))}",
        f"- Splits: {_format_counts(summary.get('split_counts'))}",
        "",
        "## Evaluation",
        "",
        f"- Status: {comparison.get('status') or 'unknown'}",
        f"- Eval split used: {comparison.get('eval_split_used') or 'none'}",
        f"- Best model: {comparison.get('best_model_name') or 'none'}",
        f"- Best macro F1: {_format_float(comparison.get('best_macro_f1'))}",
        "",
        "## Model Scores",
        "",
        "| Model | Status | Accuracy | Macro F1 |",
        "| --- | --- | ---: | ---: |",
    ]

    models = comparison.get("models")
    if isinstance(models, list) and models:
        for model in models:
            evaluation = model.get("evaluation") if isinstance(model, dict) else {}
            evaluation = evaluation if isinstance(evaluation, dict) else {}
            lines.append(
                "| "
                f"{model.get('model_name', 'unknown')} | "
                f"{evaluation.get('status', 'unknown')} | "
                f"{_format_float(evaluation.get('accuracy'))} | "
                f"{_format_float(evaluation.get('macro_f1'))} |"
            )
    else:
        lines.append("| none | error | n/a | n/a |")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Safety Reminder",
            "",
            "No production model artifacts are saved by this report, and no model "
            "is integrated into the GPUBoost agent in Phase 12.2.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_counts(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}={item}" for key, item in sorted(value.items()))


def _format_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"
