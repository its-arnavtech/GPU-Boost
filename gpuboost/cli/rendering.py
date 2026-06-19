"""Human-readable and JSON rendering for the GPUBoost CLI.

Pure presentation/formatting helpers split out of cli/main.py so command
dispatch and argument parsing stay separate from output rendering.
"""

from __future__ import annotations

from pathlib import Path

from gpuboost.agent.report import AgentReport
from gpuboost.advisor.utils import format_speedup
from gpuboost.dataset.outcome_collection import (
    OUTCOME_COLLECTION_SCHEMA_VERSION,
)
from gpuboost.demo.real_world import (
    DEFAULT_OUTPUT_ROOT as REAL_WORLD_DEMO_OUTPUT_ROOT,
    DEFAULT_PAIRS_PATH as REAL_WORLD_DEMO_PAIRS_PATH,
    build_real_world_demo_pairs,
)
from gpuboost.model.safety import (
    MODEL_WORKFLOW_SAFETY_SCHEMA_VERSION,
)
from gpuboost.model.training_pipeline import (
    BASELINE_COMPARISON_SCHEMA_VERSION,
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
from gpuboost.schemas.recommendation import AdvisorResult


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


def build_demo_real_world_info_payload() -> dict[str, object]:
    """Build a lightweight Phase 14 real-world demo discovery payload."""

    pairs = build_real_world_demo_pairs()
    workloads = [
        {
            "row_id": pair["row_id"],
            "workload_name": pair["workload_name"],
            "baseline_script": pair["baseline_script"],
            "optimized_script": pair["optimized_script"],
            "workload_family": pair["metadata"]["workload_family"],
        }
        for pair in pairs
    ]
    first_pair = pairs[0]
    return {
        "schema_version": "demo.real_world_cli.v1",
        "available_workloads": workloads,
        "commands": {
            "run_benchmarks": (
                "powershell -ExecutionPolicy Bypass -File "
                ".\\scripts\\run_real_world_demo_benchmarks.ps1"
            ),
            "collect_outcomes": (
                "python -m gpuboost dataset collect-outcomes "
                f"{REAL_WORLD_DEMO_PAIRS_PATH} --output-dir "
                f"{REAL_WORLD_DEMO_OUTPUT_ROOT}/outcomes"
            ),
            "compare_example": (
                "python -m gpuboost compare "
                f"{first_pair['baseline_json_path']} "
                f"{first_pair['optimized_json_path']}"
            ),
            "model_advisory_example": (
                "python -m gpuboost agent optimize <script> "
                "--model-artifact "
                "data/gpuboost/generated/model_training/artifacts/<id>/manifest.json "
                "--json"
            ),
        },
        "output_paths": {
            "output_root": REAL_WORLD_DEMO_OUTPUT_ROOT,
            "pairs_json": REAL_WORLD_DEMO_PAIRS_PATH,
            "outcomes": f"{REAL_WORLD_DEMO_OUTPUT_ROOT}/outcomes",
            "demo_report_json": (
                f"{REAL_WORLD_DEMO_OUTPUT_ROOT}/demo_validation_report.json"
            ),
            "demo_report_md": f"{REAL_WORLD_DEMO_OUTPUT_ROOT}/demo_validation_report.md",
        },
        "safety_notes": {
            "model_behavior": "advisory-only",
            "generated_artifacts": "ignored",
            "automatic_patch_application": False,
            "patch_application_allowed": False,
            "runs_heavy_commands": False,
            "trains_models": False,
            "calls_network": False,
        },
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


def render_demo_real_world_info_human(payload: dict[str, object]) -> str:
    workloads = payload["available_workloads"]
    commands = payload["commands"]
    output_paths = payload["output_paths"]
    lines = [
        "GPUBoost Real-World Demo",
        "",
        "Available workloads:",
    ]
    if isinstance(workloads, list):
        for workload in workloads:
            if isinstance(workload, dict):
                lines.append(
                    "- "
                    f"{workload['row_id']}: {workload['workload_family']} "
                    f"({workload['baseline_script']} -> "
                    f"{workload['optimized_script']})"
                )

    lines.extend(["", "Commands:"])
    if isinstance(commands, dict):
        for label in (
            "run_benchmarks",
            "compare_example",
            "collect_outcomes",
            "model_advisory_example",
        ):
            lines.append(f"- {label}: {commands[label]}")

    lines.extend(["", "Output paths:"])
    if isinstance(output_paths, dict):
        for label, path in output_paths.items():
            lines.append(f"- {label}: {path}")

    lines.extend(
        [
            "",
            "Safety:",
            "- Demo CLI commands are lightweight and do not run benchmarks by default.",
            "- Model behavior is advisory-only.",
            "- Generated artifacts are ignored under data/gpuboost/generated/.",
            "- No automatic patch application; patch_application_allowed=false.",
            "- Deterministic GPUBoost checks remain authoritative.",
            "- No hidden training and no network calls.",
        ]
    )
    return "\n".join(lines)


def render_demo_real_world_pairs_human(payload: dict[str, object]) -> str:
    output = render_demo_real_world_info_human(payload)
    pairs_file_written = payload.get("pairs_file_written")
    if pairs_file_written:
        output = f"{output}\n\nPairs file written: {pairs_file_written}"
    else:
        output = (
            f"{output}\n\n"
            "Pairs file not written. Re-run with --write to create "
            f"{REAL_WORLD_DEMO_PAIRS_PATH}."
        )
    return output


def render_model_evaluate_baselines_human(result: dict[str, object]) -> str:
    summary = result.get("dataset_summary")
    summary = summary if isinstance(summary, dict) else {}
    lines = [
        "GPUBoost Baseline Model Evaluation",
        f"Status: {result.get('status') or 'unknown'}",
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
            "integration was changed. Model predictions are advisory only "
            "and cannot apply patches.",
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
            "model predictions remain advisory-only and cannot apply patches."
        )
    else:
        safety = (
            "Safety: no production model artifact was saved because "
            "--save-artifact was not provided; no agent integration was "
            "changed. Model predictions are advisory-only and cannot apply "
            "patches."
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
        "Status: ok",
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
                "  next: python -m gpuboost model show-artifact "
                f"{artifact.get('manifest_path') or '<manifest>'}",
            ]
        )
    lines.extend(
        [
            "",
            "Safety: artifacts are local/generated files; model predictions "
            "are advisory-only and cannot apply patches.",
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
    manifest_path = summary.get("manifest_path") or "<manifest>"
    lines.extend(
        [
            "",
            "Next steps:",
            f"- Validate: python -m gpuboost model validate-artifact {manifest_path}",
            f"- Quality gate: python -m gpuboost model check-artifact {manifest_path}",
            "",
            "Safety: artifacts are local/generated files; model predictions "
            "are advisory-only and cannot apply patches.",
        ]
    )
    return "\n".join(lines)


def render_model_check_artifact_human(result: dict[str, object]) -> str:
    summary = result.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    manifest_path = summary.get("manifest_path") or "<manifest>"
    lines = [
        "GPUBoost Model Artifact Check",
        f"Status: {result.get('status') or 'unknown'}",
        f"Manifest: {manifest_path}",
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
    lines.extend(
        [
            "",
            "Next steps:",
            f"- Inspect: python -m gpuboost model show-artifact {manifest_path}",
            f"- Validate: python -m gpuboost model validate-artifact {manifest_path}",
            "",
            "Safety: check-artifact is a read-only quality gate; model "
            "predictions are advisory-only and cannot apply patches.",
        ]
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
        "local_db_artifacts_ignored",
        "cache_dirs_ignored",
        "env_secret_patterns_ignored",
        "raw_data_ignored",
        "patch_application_allowed",
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
        f"Manifest: {result.get('manifest_path') or 'unknown'}",
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
    lines.extend(
        [
            "",
            "Next steps:",
            "- Quality gate: python -m gpuboost model check-artifact "
            f"{result.get('manifest_path') or '<manifest>'}",
            "",
            "Safety: validation is read-only; model predictions are "
            "advisory-only and cannot apply patches.",
        ]
    )
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
    lines.extend(
        [
            "",
            "Safety: artifact predictions are advisory-only and cannot apply "
            "patches or override deterministic GPUBoost checks.",
        ]
    )
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


def _baseline_dataset_summary(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    summary = value.get("dataset_summary")
    return summary if isinstance(summary, dict) else {}


def _format_optional_score(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


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
            "- Safety: model prediction is advisory only, cannot apply patches, "
            "and deterministic checks remain authoritative.",
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
