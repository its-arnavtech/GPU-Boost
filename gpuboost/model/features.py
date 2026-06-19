"""Build safe, derived features for future local model inputs."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

from gpuboost.schemas.agent import AgentRunResult
from gpuboost.schemas.history import HistoryRunRecord
from gpuboost.schemas.model import ModelFeatureSet, ModelFeatureValue


_MAX_STRING_LENGTH = 500
_UNSAFE_KEY_PARTS = (
    "source",
    "diff",
    "stdout",
    "stderr",
    "snippet",
    "original_text",
    "replacement_text",
)
_TRIAL_SAFE_KEYS = (
    "status",
    "patch_applied",
    "syntax_check_status",
    "test_status",
    "original_file_unchanged",
)
_COMPARISON_SAFE_KEYS = ("status", "overall_verdict")


def extract_model_features_from_agent_result(
    result: AgentRunResult,
) -> ModelFeatureSet:
    """Extract safe derived model features from an agent run result."""

    artifacts = result.artifacts
    trial = _as_mapping(artifacts.get("trial"))
    comparison = _as_mapping(artifacts.get("comparison"))
    has_diff = bool(artifacts.get("diff"))

    return ModelFeatureSet(
        hardware=_extract_hardware_features(artifacts),
        benchmarks=_extract_benchmark_features(artifacts),
        advisor=_extract_advisor_features(artifacts),
        code=_extract_code_features(result, artifacts),
        patches=_extract_patch_features(artifacts, has_diff=has_diff),
        trial=_extract_trial_features(trial),
        comparison=_pick_safe_scalars(comparison, _COMPARISON_SAFE_KEYS),
        history=_extract_agent_history_features(artifacts),
        metadata=_extract_agent_metadata(result, trial, comparison, has_diff),
    )


def extract_model_features_from_history_record(
    record: HistoryRunRecord,
) -> ModelFeatureSet:
    """Extract safe derived model features from a local history record."""

    return ModelFeatureSet(
        hardware=_compact_features(
            {
                "gpu_name": record.gpu_name,
                "cuda_available": record.cuda_available,
            }
        ),
        benchmarks=_copy_safe_summary(record.benchmark_summary),
        advisor=_copy_safe_summary(record.advisor_summary),
        code=_copy_safe_summary(record.code_summary),
        patches=_copy_safe_summary(record.patch_summary),
        trial=_copy_safe_summary(record.trial_summary),
        comparison=_copy_safe_summary(record.comparison_summary),
        history=_compact_features(
            {
                "status": record.status,
                "command": record.command,
                "goal_kind": record.goal_kind,
                "has_trial": record.has_trial(),
                "has_comparison": record.has_comparison(),
            }
        ),
    )


def sanitize_feature_value(value: Any) -> ModelFeatureValue:
    """Return a scalar feature value, omitting nested or unsafe content."""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        if _looks_like_raw_content(value):
            return None
        return value[:_MAX_STRING_LENGTH]
    if isinstance(value, dict | list | tuple | set):
        return None

    text = str(value)
    if len(text) > _MAX_STRING_LENGTH or _looks_like_raw_content(text):
        return None
    return text


def safe_count(value: Any) -> int | None:
    """Return a length for countable values, otherwise None."""

    if isinstance(value, list | dict | str):
        return len(value)
    return None


def _extract_hardware_features(
    artifacts: dict[str, Any],
) -> dict[str, ModelFeatureValue]:
    hardware_sources = (
        artifacts.get("gpu"),
        artifacts.get("gpu_profile"),
        artifacts.get("profile"),
        artifacts.get("metadata"),
    )
    for source in hardware_sources:
        data = _as_mapping(source)
        if not data:
            continue

        features = _compact_features(
            {
                "gpu_name": data.get("gpu_name") or data.get("name"),
                "cuda_available": data.get("cuda_available"),
            }
        )
        gpus = data.get("gpus")
        if "gpu_name" not in features and isinstance(gpus, list) and gpus:
            first_gpu = _as_mapping(gpus[0])
            features.update(_compact_features({"gpu_name": first_gpu.get("name")}))
        if features:
            return features
    return {}


def _extract_benchmark_features(
    artifacts: dict[str, Any],
) -> dict[str, ModelFeatureValue]:
    data = _as_mapping(
        artifacts.get("benchmarks")
        or artifacts.get("benchmark")
        or artifacts.get("benchmark_result")
    )
    if not data:
        data = _as_mapping(artifacts.get("metadata")).get("benchmarks", {})
        data = _as_mapping(data)
    if not data:
        return {}

    features = _copy_safe_summary(data)
    count = safe_count(data.get("results")) or safe_count(data.get("metrics"))
    if count is not None:
        features.setdefault("metric_count", count)

    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            _add_named_metric(features, key, value)
    elif isinstance(metrics, list):
        for metric in metrics:
            metric_data = _as_mapping(metric)
            _add_named_metric(
                features,
                metric_data.get("name"),
                metric_data.get("value"),
            )

    results = data.get("results")
    if isinstance(results, list):
        for result in results:
            result_data = _as_mapping(result)
            result_metrics = result_data.get("metrics")
            if isinstance(result_metrics, list):
                for metric in result_metrics:
                    metric_data = _as_mapping(metric)
                    _add_named_metric(
                        features,
                        metric_data.get("name"),
                        metric_data.get("value"),
                    )
    return features


def _extract_advisor_features(
    artifacts: dict[str, Any],
) -> dict[str, ModelFeatureValue]:
    data = _as_mapping(artifacts.get("advisor") or artifacts.get("advisor_result"))
    if not data:
        return {}

    return _compact_features(
        {
            "recommendation_count": safe_count(data.get("recommendations")),
            "warning_count": safe_count(data.get("warnings")),
        }
    )


def _extract_code_features(
    result: AgentRunResult,
    artifacts: dict[str, Any],
) -> dict[str, ModelFeatureValue]:
    data = _as_mapping(artifacts.get("code") or artifacts.get("code_analysis"))
    features = _compact_features(
        {
            "finding_count": safe_count(data.get("findings")),
            "warning_count": safe_count(data.get("warnings")),
        }
    )

    code_actions = [
        action
        for action in result.plan.actions
        if "code" in action.name or "analy" in action.name
    ]
    if code_actions:
        features["code_action_count"] = len(code_actions)
        features["completed_code_action_count"] = sum(
            1 for action in code_actions if action.status == "completed"
        )
        features["failed_code_action_count"] = sum(
            1 for action in code_actions if action.status == "failed"
        )
    return features


def _extract_patch_features(
    artifacts: dict[str, Any],
    *,
    has_diff: bool,
) -> dict[str, ModelFeatureValue]:
    data = _as_mapping(artifacts.get("patch") or artifacts.get("patch_plan"))
    return _compact_features(
        {
            "has_diff": has_diff,
            "patch_suggestion_count": safe_count(data.get("suggestions")),
        }
    )


def _extract_trial_features(
    trial: dict[str, Any],
) -> dict[str, ModelFeatureValue]:
    if not trial:
        return {}

    features = _pick_safe_scalars(trial, _TRIAL_SAFE_KEYS)
    step_count = safe_count(trial.get("steps"))
    if step_count is not None:
        features["step_count"] = step_count
    return features


def _extract_agent_history_features(
    artifacts: dict[str, Any],
) -> dict[str, ModelFeatureValue]:
    if artifacts.get("history_run_id"):
        return {"has_history_run": True}
    return {}


def _extract_agent_metadata(
    result: AgentRunResult,
    trial: dict[str, Any],
    comparison: dict[str, Any],
    has_diff: bool,
) -> dict[str, ModelFeatureValue]:
    actions = result.plan.actions
    return {
        "action_count": len(actions),
        "completed_action_count": sum(
            1 for action in actions if action.status == "completed"
        ),
        "failed_action_count": sum(1 for action in actions if action.status == "failed"),
        "warning_count": len(result.warnings) + len(result.plan.warnings),
        "event_count": len(result.events),
        "has_diff": has_diff,
        "has_trial": bool(trial),
        "has_comparison": bool(comparison),
    }


def _copy_safe_summary(data: dict[str, Any]) -> dict[str, ModelFeatureValue]:
    features: dict[str, ModelFeatureValue] = {}
    for key, value in data.items():
        if _is_unsafe_key(key):
            continue
        safe_value = sanitize_feature_value(value)
        if safe_value is not None:
            features[key] = safe_value
    return features


def _pick_safe_scalars(
    data: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, ModelFeatureValue]:
    return _compact_features({key: data.get(key) for key in keys})


def _compact_features(data: dict[str, Any]) -> dict[str, ModelFeatureValue]:
    features: dict[str, ModelFeatureValue] = {}
    for key, value in data.items():
        if _is_unsafe_key(key):
            continue
        safe_value = sanitize_feature_value(value)
        if safe_value is not None:
            features[key] = safe_value
    return features


def _add_named_metric(
    features: dict[str, ModelFeatureValue],
    name: Any,
    value: Any,
) -> None:
    safe_name = sanitize_feature_value(name)
    if not isinstance(safe_name, str) or _is_unsafe_key(safe_name):
        return
    safe_value = sanitize_feature_value(value)
    if safe_value is not None:
        features[f"metric_{safe_name}"] = safe_value


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return converted
    return {}


def _is_unsafe_key(key: Any) -> bool:
    key_text = str(key).lower()
    if key_text == "has_diff":
        return False
    return any(part in key_text for part in _UNSAFE_KEY_PARTS)


_CODE_LINE_RE = re.compile(
    r"^(?:(?:async\s+)?def\s+\w+\s*\(|class\s+\w+[\s(:]|"
    r"import\s+[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+)*\s*$|"
    r"from\s+[\w.]+\s+import\s+|@\w[\w.]*)"
)


def _looks_like_raw_content(value: str) -> bool:
    """Heuristically detect pasted source code, diffs, or fenced code blocks.

    Uses line-anchored signals instead of plain substring matching so that
    ordinary multi-line descriptions that merely mention words like ``import``
    or ``class`` are not misclassified as raw content and silently dropped.
    """

    if "\n" not in value:
        return False

    stripped = [line.strip() for line in value.splitlines()]

    # Unified-diff / git-diff signatures (line-anchored and corroborated).
    has_diff_headers = any(s.startswith("--- ") for s in stripped) and any(
        s.startswith("+++ ") for s in stripped
    )
    has_hunk_header = any(
        s.startswith("@@ ") and s.count("@@") >= 2 for s in stripped
    )
    has_git_diff = any(s.startswith("diff --git ") for s in stripped)
    if has_diff_headers or has_hunk_header or has_git_diff:
        return True

    # A fenced code block is a strong signal on its own.
    if any(s.startswith("```") for s in stripped):
        return True

    # Otherwise require a line that clearly begins a Python construct.
    return any(_CODE_LINE_RE.match(s) for s in stripped)
