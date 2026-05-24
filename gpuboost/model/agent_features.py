"""Safe feature extraction for advisory agent model predictions."""

from __future__ import annotations

from pathlib import Path
from gpuboost.dataset.training_features import (
    is_safe_training_feature_name,
    is_target_derived_feature_name,
)
from gpuboost.schemas.model import ModelFeatureSet, ModelInput, ModelValue

_FORBIDDEN_PARTS = (
    "raw_source",
    "raw_diff",
    "stdout",
    "stderr",
    "source",
    "file_contents",
    "patch_contents",
    "diff",
)
_TARGET_DERIVED_PARTS = (
    "overall_verdict",
    "improved_metric_count",
    "regressed_metric_count",
    "unchanged_metric_count",
    "before_",
    "after_",
    "delta_",
    "percent_delta_",
    "comparison_",
    "label",
    "target",
)


def build_model_input_features_from_agent_state(
    state: object,
) -> dict[str, str | int | float | bool | None]:
    """Build safe scalar model features from live agent state."""

    features: dict[str, ModelValue] = {}
    goal = getattr(state, "goal", None)
    script_path = getattr(goal, "script_path", None)
    options = getattr(goal, "options", {}) if goal is not None else {}
    if isinstance(script_path, str) and script_path:
        _add_feature(features, "metadata.script_extension", Path(script_path).suffix)
        _add_feature(features, "metadata.has_script_path", True)
    else:
        _add_feature(features, "metadata.has_script_path", False)
    if isinstance(options, dict):
        _add_feature(features, "metadata.quick", bool(options.get("quick")))
        _add_feature(features, "metadata.model", bool(options.get("model")))
        _add_feature(features, "metadata.trial_requested", bool(options.get("trial")))

    _add_count_features(features, state)
    _add_gpu_features(features, getattr(state, "gpu_profile", None))
    _add_code_features(features, getattr(state, "code_analysis", None))
    _add_patch_features(features, getattr(state, "patch_plan", None))
    _add_trial_features(features, getattr(state, "metadata", {}))
    return dict(sorted(features.items()))


def build_model_input_from_agent_state(state: object) -> ModelInput:
    """Build a provider-compatible ModelInput from safe agent state features."""

    features = build_model_input_features_from_agent_state(state)
    warnings: list[str] = []
    if len(features) < 3:
        warnings.append("Sparse safe model features extracted from agent state.")
    return ModelInput(
        goal="agent optimize",
        features=ModelFeatureSet(metadata=features),
        context={
            "command": "agent optimize",
            "features": features,
            "feature_source": "safe_agent_state",
        },
        warnings=warnings + list(getattr(state, "warnings", [])),
    )


def _add_count_features(features: dict[str, ModelValue], state: object) -> None:
    _add_feature(features, "metadata.warning_count", len(getattr(state, "warnings", [])))
    _add_feature(
        features,
        "metadata.completed_action_count",
        len(getattr(state, "completed_actions", [])),
    )
    _add_feature(
        features,
        "metadata.failed_action_count",
        len(getattr(state, "failed_actions", [])),
    )
    _add_feature(features, "metadata.event_count", len(getattr(state, "events", [])))
    _add_feature(features, "metadata.has_diff", bool(getattr(state, "diff", None)))


def _add_gpu_features(features: dict[str, ModelValue], gpu_profile: object) -> None:
    profile = gpu_profile if isinstance(gpu_profile, dict) else {}
    torch_env = profile.get("torch_env") if isinstance(profile.get("torch_env"), dict) else {}
    _add_feature(features, "hardware.cuda_available", torch_env.get("cuda_available"))
    _add_feature(features, "hardware.device_count", torch_env.get("device_count"))
    gpus = profile.get("gpus")
    if isinstance(gpus, list):
        _add_feature(features, "hardware.gpu_count", len(gpus))


def _add_code_features(features: dict[str, ModelValue], code_analysis: object) -> None:
    analysis = code_analysis if isinstance(code_analysis, dict) else {}
    findings = analysis.get("findings")
    if isinstance(findings, list):
        _add_feature(features, "metadata.code_finding_count", len(findings))
        severity_counts: dict[str, int] = {}
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            severity = finding.get("severity")
            if isinstance(severity, str):
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        for severity, count in severity_counts.items():
            _add_feature(features, f"metadata.code_finding_{severity}_count", count)
    _add_feature(features, "metadata.has_code_analysis", bool(analysis))


def _add_patch_features(features: dict[str, ModelValue], patch_plan: object) -> None:
    plan = patch_plan if isinstance(patch_plan, dict) else {}
    suggestions = plan.get("suggestions")
    if isinstance(suggestions, list):
        _add_feature(features, "metadata.patch_suggestion_count", len(suggestions))
        kinds: dict[str, int] = {}
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                continue
            category = suggestion.get("category") or suggestion.get("kind")
            if isinstance(category, str):
                kinds[category] = kinds.get(category, 0) + 1
        for kind, count in kinds.items():
            _add_feature(features, f"metadata.patch_suggestion_{kind}_count", count)
    _add_feature(features, "metadata.has_patch_plan", bool(plan))


def _add_trial_features(features: dict[str, ModelValue], metadata: object) -> None:
    data = metadata if isinstance(metadata, dict) else {}
    trial = data.get("trial_result")
    trial = trial if isinstance(trial, dict) else {}
    _add_feature(features, "metadata.has_trial", bool(trial))
    _add_feature(features, "metadata.trial_status", trial.get("status"))
    _add_feature(features, "metadata.syntax_check_status", trial.get("syntax_check_status"))
    _add_feature(features, "metadata.test_status", trial.get("test_status"))


def _add_feature(
    features: dict[str, ModelValue],
    name: str,
    value: object,
) -> None:
    if value is None:
        return
    if not _is_allowed_feature_name(name):
        return
    if isinstance(value, bool | int | float | str):
        features[name] = value


def _is_allowed_feature_name(name: str) -> bool:
    normalized = name.lower()
    if any(part in normalized for part in _FORBIDDEN_PARTS):
        return normalized == "metadata.has_diff"
    if any(part in normalized for part in _TARGET_DERIVED_PARTS):
        return False
    if is_target_derived_feature_name(name):
        return False
    return is_safe_training_feature_name(name)
