"""Tests for Phase 12.4 local model artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuboost.model.artifacts import (
    create_model_artifact_dir,
    load_model_artifact_manifest,
    load_neural_model_artifact,
    save_neural_model_artifact,
    validate_model_artifact,
)
from gpuboost.model.neural import torch_available
from gpuboost.model.neural_training import train_best_neural_model_for_artifact
from gpuboost.schemas.training import (
    EncodedTrainingDataset,
    TrainingDatasetSummary,
    TrainingFeatureSpec,
)


def test_create_model_artifact_dir_stays_under_output_dir(tmp_path) -> None:
    artifact_dir = Path(
        create_model_artifact_dir(str(tmp_path), artifact_name="../unsafe:name")
    )

    assert artifact_dir.exists()
    assert tmp_path.resolve() in artifact_dir.resolve().parents
    assert ".." not in artifact_dir.name


def test_save_and_load_neural_model_artifact(tmp_path) -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    dataset = _artifact_dataset()
    model, feature_spec, label_mapping, result = train_best_neural_model_for_artifact(
        dataset,
        max_epochs=3,
        max_candidates=1,
    )

    manifest = save_neural_model_artifact(
        model,
        feature_spec,
        label_mapping,
        result.best_config,
        result,
        output_dir=str(tmp_path),
        artifact_name="fixture",
    )
    manifest_path = tmp_path / "fixture" / "manifest.json"

    assert manifest.model_file == "model.pt"
    assert not Path(manifest.model_file).is_absolute()
    assert manifest_path.exists()
    assert (tmp_path / "fixture" / "model.pt").exists()

    loaded_manifest = load_model_artifact_manifest(str(manifest_path))
    loaded_model, loaded_spec, loaded_mapping, _ = load_neural_model_artifact(
        str(manifest_path),
        device="cpu",
    )

    assert loaded_manifest.input_size == len(feature_spec.feature_names)
    assert loaded_spec.feature_names == feature_spec.feature_names
    assert loaded_mapping == label_mapping
    assert loaded_model.training is False


def test_validate_model_artifact_catches_missing_file(tmp_path) -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    manifest_path = _write_artifact(tmp_path)
    (manifest_path.parent / "feature_spec.json").unlink()

    result = validate_model_artifact(str(manifest_path))

    assert result["status"] == "error"
    assert any("feature_spec.json" in error for error in result["errors"])


def test_artifact_json_files_do_not_save_raw_payloads(tmp_path) -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    manifest_path = _write_artifact(tmp_path)

    combined_json = "\n".join(
        path.read_text(encoding="utf-8")
        for path in manifest_path.parent.glob("*.json")
    )

    for forbidden in ("raw_source", "raw_diff", "stdout", "stderr", "def train"):
        assert forbidden not in combined_json


def _write_artifact(tmp_path) -> Path:
    dataset = _artifact_dataset()
    model, feature_spec, label_mapping, result = train_best_neural_model_for_artifact(
        dataset,
        max_epochs=3,
        max_candidates=1,
    )
    save_neural_model_artifact(
        model,
        feature_spec,
        label_mapping,
        result.best_config,
        result,
        output_dir=str(tmp_path),
        artifact_name="fixture",
    )
    return tmp_path / "fixture" / "manifest.json"


def _artifact_dataset() -> EncodedTrainingDataset:
    feature_spec = TrainingFeatureSpec(
        feature_names=["features.safe_signal"],
        categorical_features=[],
        numeric_features=["features.safe_signal"],
        boolean_features=[],
    )
    X = [[0.0], [0.1], [1.0], [1.1], [0.05], [1.05], [0.2], [1.2]]
    y = [0, 0, 1, 1, 0, 1, 0, 1]
    split = [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]
    return EncodedTrainingDataset(
        row_ids=[f"row-{index}" for index in range(len(y))],
        feature_names=feature_spec.feature_names,
        X=X,
        y=y,
        labels=["improved", "regressed"],
        label_to_index={"improved": 0, "regressed": 1},
        split=split,
        summary=TrainingDatasetSummary(
            row_count=len(y),
            labeled_count=len(y),
            skipped_count=0,
            feature_count=1,
            label_counts={"improved": 4, "regressed": 4},
            split_counts={name: split.count(name) for name in set(split)},
            warnings=[],
            metadata={},
        ),
        feature_spec=feature_spec,
        warnings=[],
        metadata={},
    )
