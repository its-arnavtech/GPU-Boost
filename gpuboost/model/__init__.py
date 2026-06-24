"""Model-layer helpers for GPUBoost."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str | None]] = {
    "BaseModelProvider": ("gpuboost.model.provider", "BaseModelProvider"),
    "FailingModelProvider": ("gpuboost.model.provider", "FailingModelProvider"),
    "MajorityClassBaseline": ("gpuboost.model.baseline", "MajorityClassBaseline"),
    "NullModelProvider": ("gpuboost.model.provider", "NullModelProvider"),
    "StaticModelProvider": ("gpuboost.model.provider", "StaticModelProvider"),
    "build_encoded_training_dataset": (
        "gpuboost.model.feature_encoding",
        "build_encoded_training_dataset",
    ),
    "build_model_input_from_agent_result": (
        "gpuboost.model.inference",
        "build_model_input_from_agent_result",
    ),
    "encode_feature_dicts": ("gpuboost.model.feature_encoding", "encode_feature_dicts"),
    "encode_labels": ("gpuboost.model.feature_encoding", "encode_labels"),
    "extract_model_features_from_agent_result": (
        "gpuboost.model.features",
        "extract_model_features_from_agent_result",
    ),
    "extract_model_features_from_history_record": (
        "gpuboost.model.features",
        "extract_model_features_from_history_record",
    ),
    "infer_feature_spec": ("gpuboost.model.feature_encoding", "infer_feature_spec"),
    "load_default_training_rows": (
        "gpuboost.model.training_data",
        "load_default_training_rows",
    ),
    "load_training_rows_jsonl": (
        "gpuboost.model.training_data",
        "load_training_rows_jsonl",
    ),
    "model_result_to_artifact": ("gpuboost.model.inference", "model_result_to_artifact"),
    "run_model_inference": ("gpuboost.model.inference", "run_model_inference"),
    "summarize_training_rows": (
        "gpuboost.model.training_data",
        "summarize_training_rows",
    ),
    "train_majority_baseline": ("gpuboost.model.baseline", "train_majority_baseline"),
    "artifacts": ("gpuboost.model.artifacts", None),
    "inference": ("gpuboost.model.inference", None),
    "neural": ("gpuboost.model.neural", None),
    "neural_training": ("gpuboost.model.neural_training", None),
    "provider": ("gpuboost.model.provider", None),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
