"""Report writing for Phase 12.3 neural training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpuboost.model.training_reports import DEFAULT_BASELINE_REPORT_DIR
from gpuboost.schemas.training import NeuralSearchResult, NeuralTrainingResult


def write_neural_training_reports(
    result: NeuralSearchResult | NeuralTrainingResult,
    output_dir: str = DEFAULT_BASELINE_REPORT_DIR,
) -> dict[str, str]:
    """Write JSON and Markdown neural training reports."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "neural_training_report.json"
    markdown_path = directory / "neural_training_report.md"
    payload = result.to_dict()

    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown_report(payload), encoding="utf-8")

    return {
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }


def _render_markdown_report(payload: dict[str, Any]) -> str:
    best_result = _best_result_payload(payload)
    validation = _evaluation(best_result, "validation_evaluation")
    test = _evaluation(best_result, "test_evaluation")
    history = best_result.get("history") if isinstance(best_result, dict) else {}
    history = history if isinstance(history, dict) else {}
    baseline = _baseline_comparison(payload, best_result)
    warnings = _warnings(payload, best_result)

    lines = [
        "# GPUBoost Neural Model Training",
        "",
        "This is not a production model artifact and is not integrated with the agent.",
        "",
        "## Summary",
        "",
        f"- Status: {payload.get('status') or 'unknown'}",
        f"- Target macro F1: {_format_float(payload.get('target_macro_f1'))}",
        f"- Target met: {_format_yes_no(payload.get('target_met'))}",
        f"- Neural beats baseline: {_format_yes_no(payload.get('beats_baseline'))}",
        f"- Candidates tried: {_candidate_count(payload)}",
        f"- Best baseline model: {baseline.get('best_baseline_model_name') or 'none'}",
        f"- Best baseline macro F1: "
        f"{_format_float(baseline.get('best_baseline_macro_f1'))}",
        "",
        "## Best Config",
        "",
        f"```json\n{json.dumps(_best_config(payload), indent=2, sort_keys=True)}\n```",
        "",
        "## Best Run",
        "",
        f"- Epochs ran: {history.get('epochs_ran', 0)}",
        f"- Best epoch: {history.get('best_epoch') or 'none'}",
        f"- Validation accuracy: {_format_float(validation.get('accuracy'))}",
        f"- Validation macro F1: {_format_float(validation.get('macro_f1'))}",
        f"- Test accuracy: {_format_float(test.get('accuracy'))}",
        f"- Test macro F1: {_format_float(test.get('macro_f1'))}",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _best_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    best_result = payload.get("best_result")
    if isinstance(best_result, dict):
        return best_result
    return payload


def _best_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("best_config")
    if isinstance(config, dict):
        return config
    best_result = _best_result_payload(payload)
    result_config = best_result.get("config")
    return result_config if isinstance(result_config, dict) else {}


def _evaluation(best_result: dict[str, Any], key: str) -> dict[str, Any]:
    evaluation = best_result.get(key)
    return evaluation if isinstance(evaluation, dict) else {}


def _baseline_comparison(
    payload: dict[str, Any],
    best_result: dict[str, Any],
) -> dict[str, Any]:
    comparison = best_result.get("baseline_comparison")
    if isinstance(comparison, dict):
        return comparison
    return {
        "best_baseline_macro_f1": payload.get("baseline_macro_f1"),
        "best_baseline_model_name": payload.get("best_baseline_model_name"),
    }


def _warnings(
    payload: dict[str, Any],
    best_result: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    for source in (payload.get("warnings"), best_result.get("warnings")):
        if isinstance(source, list):
            warnings.extend(str(item) for item in source)
    return sorted(set(warnings))


def _candidate_count(payload: dict[str, Any]) -> int:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return len(candidates)
    return 1


def _format_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


def _format_yes_no(value: object) -> str:
    return "yes" if value is True else "no"
