"""Save, load, and validate local Phase 12 model artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from gpuboost.model.neural import MLPClassifier, select_torch_device, torch, torch_available
from gpuboost.schemas.training import (
    ModelArtifactManifest,
    NeuralSearchResult,
    NeuralTrainingConfig,
    NeuralTrainingResult,
    TrainingFeatureSpec,
    create_timestamp,
)

DEFAULT_ARTIFACT_DIR = "data/gpuboost/generated/model_training/artifacts"
MANIFEST_SCHEMA_VERSION = "training.model_artifact.v1"
_RELATIVE_FILES = (
    "model_file",
    "feature_spec_file",
    "label_mapping_file",
    "training_config_file",
    "evaluation_report_file",
)


def create_model_artifact_dir(
    output_dir: str = DEFAULT_ARTIFACT_DIR,
    artifact_name: str | None = None,
) -> str:
    """Create and return a safe model artifact directory under output_dir."""

    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_artifact_name(artifact_name or create_timestamp())
    artifact_dir = base_dir / safe_name
    counter = 1
    while artifact_dir.exists():
        artifact_dir = base_dir / f"{safe_name}-{counter}"
        counter += 1
    artifact_dir.mkdir(parents=True, exist_ok=False)
    if base_dir.resolve() not in artifact_dir.resolve().parents:
        raise ValueError("Refusing to create artifact outside output_dir.")
    return str(artifact_dir)


def save_neural_model_artifact(
    model: object,
    feature_spec: TrainingFeatureSpec,
    label_to_index: dict[str, int],
    config: NeuralTrainingConfig,
    result: NeuralSearchResult | NeuralTrainingResult,
    output_dir: str = DEFAULT_ARTIFACT_DIR,
    artifact_name: str | None = None,
) -> ModelArtifactManifest:
    """Save a local neural model artifact without raw training rows."""

    if not torch_available() or torch is None:
        raise RuntimeError("PyTorch is unavailable; cannot save neural artifact.")

    artifact_dir = Path(create_model_artifact_dir(output_dir, artifact_name))
    model_file = artifact_dir / "model.pt"
    feature_spec_file = artifact_dir / "feature_spec.json"
    label_mapping_file = artifact_dir / "label_mapping.json"
    training_config_file = artifact_dir / "training_config.json"
    evaluation_report_file = artifact_dir / "evaluation_report.json"
    manifest_file = artifact_dir / "manifest.json"

    torch.save(model.state_dict(), model_file)
    _write_json(feature_spec_file, feature_spec.to_dict())
    _write_json(label_mapping_file, dict(sorted(label_to_index.items())))
    _write_json(
        training_config_file,
        {
            "config": config.to_dict(),
            "normalization": _normalization_payload(model),
        },
    )
    _write_json(evaluation_report_file, result.to_dict())

    labels = _labels_from_mapping(label_to_index)
    metrics = _artifact_metrics(result)
    manifest = ModelArtifactManifest(
        created_at=create_timestamp(),
        model_name=config.model_name,
        model_file=model_file.name,
        feature_spec_file=feature_spec_file.name,
        label_mapping_file=label_mapping_file.name,
        training_config_file=training_config_file.name,
        evaluation_report_file=evaluation_report_file.name,
        input_size=len(feature_spec.feature_names),
        output_size=len(labels),
        labels=labels,
        feature_names=list(feature_spec.feature_names),
        validation_macro_f1=metrics["validation_macro_f1"],
        test_macro_f1=metrics["test_macro_f1"],
        baseline_macro_f1=metrics["baseline_macro_f1"],
        beats_baseline=bool(metrics["beats_baseline"]),
        target_macro_f1=metrics["target_macro_f1"],
        target_met=bool(metrics["target_met"]),
        warnings=_artifact_warnings(result),
        metadata={
            "model_artifact_saved": True,
            "agent_integration_changed": False,
        },
    )
    _write_json(manifest_file, manifest.to_dict())
    manifest.metadata["manifest_path"] = str(manifest_file)
    return manifest


def load_model_artifact_manifest(path: str) -> ModelArtifactManifest:
    """Load and validate a model artifact manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model artifact manifest not found: {path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Model artifact manifest must be a JSON object.")
    manifest = ModelArtifactManifest(**data)
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported model artifact schema: {manifest.schema_version}")
    _validate_relative_file_names(manifest)
    for filename in _manifest_files(manifest):
        file_path = manifest_path.parent / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Model artifact file not found: {filename}")
    return manifest


def load_neural_model_artifact(
    manifest_path: str,
    device: str = "auto",
) -> tuple[MLPClassifier, TrainingFeatureSpec, dict[str, int], ModelArtifactManifest]:
    """Load a local neural model artifact without training."""

    if not torch_available() or torch is None:
        raise RuntimeError("PyTorch is unavailable; cannot load neural artifact.")
    manifest = load_model_artifact_manifest(manifest_path)
    directory = Path(manifest_path).parent
    feature_spec = _load_feature_spec(directory / manifest.feature_spec_file)
    label_to_index = _load_label_mapping(directory / manifest.label_mapping_file)
    config_payload = _load_json(directory / manifest.training_config_file)
    config_data = config_payload.get("config", config_payload)
    config = NeuralTrainingConfig(**config_data)
    selected_device = select_torch_device(device)
    model = MLPClassifier(
        input_size=manifest.input_size,
        output_size=manifest.output_size,
        hidden_sizes=list(config.hidden_sizes),
        dropout=config.dropout,
    ).to(selected_device)
    state = torch.load(
        directory / manifest.model_file,
        map_location=selected_device,
        weights_only=True,
    )
    model.load_state_dict(state)
    _attach_normalization(model, config_payload, selected_device)
    model.eval()
    return model, feature_spec, label_to_index, manifest


def validate_model_artifact(manifest_path: str) -> dict[str, Any]:
    """Validate a local model artifact and return JSON-serializable status."""

    warnings: list[str] = []
    errors: list[str] = []
    summary: dict[str, Any] = {}
    try:
        manifest = load_model_artifact_manifest(manifest_path)
        directory = Path(manifest_path).parent
        feature_spec = _load_feature_spec(directory / manifest.feature_spec_file)
        label_to_index = _load_label_mapping(directory / manifest.label_mapping_file)
        summary = {
            "model_name": manifest.model_name,
            "artifact_type": manifest.artifact_type,
            "input_size": manifest.input_size,
            "output_size": manifest.output_size,
            "validation_macro_f1": manifest.validation_macro_f1,
            "test_macro_f1": manifest.test_macro_f1,
            "beats_baseline": manifest.beats_baseline,
            "target_met": manifest.target_met,
        }
        if manifest.input_size != len(feature_spec.feature_names):
            errors.append("Manifest input_size does not match feature spec.")
        if manifest.output_size != len(label_to_index):
            errors.append("Manifest output_size does not match label mapping.")
        if manifest.labels != _labels_from_mapping(label_to_index):
            errors.append("Manifest labels do not match label mapping.")
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError) as error:
        errors.append(str(error))

    return {
        "schema_version": "training.model_artifact_validation.v1",
        "status": "error" if errors else "ok",
        "warnings": warnings,
        "errors": errors,
        "manifest_summary": summary,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return data


def _load_feature_spec(path: Path) -> TrainingFeatureSpec:
    return TrainingFeatureSpec(**_load_json(path))


def _load_label_mapping(path: Path) -> dict[str, int]:
    data = _load_json(path)
    return {str(key): int(value) for key, value in data.items()}


def _validate_relative_file_names(manifest: ModelArtifactManifest) -> None:
    for field_name in _RELATIVE_FILES:
        filename = getattr(manifest, field_name)
        if filename is None:
            continue
        if Path(filename).name != filename:
            raise ValueError(f"Manifest {field_name} must be a relative file name.")


def _manifest_files(manifest: ModelArtifactManifest) -> list[str]:
    return [
        filename
        for filename in [
            manifest.model_file,
            manifest.feature_spec_file,
            manifest.label_mapping_file,
            manifest.training_config_file,
            manifest.evaluation_report_file,
        ]
        if filename is not None
    ]


def _safe_artifact_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return name or "artifact"


def _normalization_payload(model: object) -> dict[str, list[float]]:
    mean = getattr(model, "_gpuboost_feature_mean", None)
    std = getattr(model, "_gpuboost_feature_std", None)
    return {
        "feature_mean": _tensor_to_list(mean),
        "feature_std": _tensor_to_list(std),
    }


def _tensor_to_list(value: object) -> list[float]:
    if value is None:
        return []
    data = value.detach().cpu().reshape(-1).tolist()
    return [float(item) for item in data]


def _attach_normalization(
    model: object,
    config_payload: dict[str, Any],
    selected_device: str,
) -> None:
    if torch is None:
        return
    normalization = config_payload.get("normalization")
    if not isinstance(normalization, dict):
        return
    mean = normalization.get("feature_mean")
    std = normalization.get("feature_std")
    if isinstance(mean, list) and isinstance(std, list) and mean and std:
        setattr(
            model,
            "_gpuboost_feature_mean",
            torch.tensor([mean], dtype=torch.float32),
        )
        setattr(
            model,
            "_gpuboost_feature_std",
            torch.tensor([std], dtype=torch.float32),
        )
    setattr(model, "_gpuboost_device", selected_device)


def _labels_from_mapping(label_to_index: dict[str, int]) -> list[str]:
    return [
        label
        for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])
    ]


def _artifact_metrics(
    result: NeuralSearchResult | NeuralTrainingResult,
) -> dict[str, float | bool | None]:
    best = result.best_result if isinstance(result, NeuralSearchResult) else result
    validation = best.validation_evaluation if best is not None else None
    test = best.test_evaluation if best is not None else None
    baseline_macro = (
        result.baseline_macro_f1
        if isinstance(result, NeuralSearchResult)
        else best.baseline_comparison.get("best_baseline_macro_f1")
    )
    target_macro = (
        result.target_macro_f1 if isinstance(result, NeuralSearchResult) else None
    )
    target_met = result.target_met if isinstance(result, NeuralSearchResult) else False
    beats_baseline = (
        result.beats_baseline
        if isinstance(result, NeuralSearchResult)
        else bool(best.baseline_comparison.get("beats_baseline"))
    )
    return {
        "validation_macro_f1": validation.macro_f1 if validation else None,
        "test_macro_f1": test.macro_f1 if test else None,
        "baseline_macro_f1": baseline_macro if isinstance(baseline_macro, float) else None,
        "beats_baseline": beats_baseline,
        "target_macro_f1": target_macro,
        "target_met": target_met,
    }


def _artifact_warnings(result: NeuralSearchResult | NeuralTrainingResult) -> list[str]:
    return list(result.warnings)
