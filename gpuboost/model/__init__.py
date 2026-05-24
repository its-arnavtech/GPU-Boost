"""Model-layer helpers for GPUBoost."""

from gpuboost.model.baseline import MajorityClassBaseline, train_majority_baseline
from gpuboost.model.feature_encoding import (
    build_encoded_training_dataset,
    encode_feature_dicts,
    encode_labels,
    infer_feature_spec,
)
from gpuboost.model.features import (
    extract_model_features_from_agent_result,
    extract_model_features_from_history_record,
)
from gpuboost.model.inference import (
    build_model_input_from_agent_result,
    model_result_to_artifact,
    run_model_inference,
)
from gpuboost.model.provider import (
    BaseModelProvider,
    FailingModelProvider,
    NullModelProvider,
    StaticModelProvider,
)
from gpuboost.model.training_data import (
    load_default_training_rows,
    load_training_rows_jsonl,
    summarize_training_rows,
)

__all__ = [
    "BaseModelProvider",
    "FailingModelProvider",
    "MajorityClassBaseline",
    "NullModelProvider",
    "StaticModelProvider",
    "build_encoded_training_dataset",
    "build_model_input_from_agent_result",
    "encode_feature_dicts",
    "encode_labels",
    "extract_model_features_from_agent_result",
    "extract_model_features_from_history_record",
    "infer_feature_spec",
    "load_default_training_rows",
    "load_training_rows_jsonl",
    "model_result_to_artifact",
    "run_model_inference",
    "summarize_training_rows",
    "train_majority_baseline",
]
