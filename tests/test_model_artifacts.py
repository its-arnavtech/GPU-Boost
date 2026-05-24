"""Tests for Phase 12.4 local model artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuboost.model.artifacts import (
    create_model_artifact_dir,
    find_model_artifact_manifests,
    list_model_artifacts,
    load_model_artifact_manifest,
    load_neural_model_artifact,
    save_neural_model_artifact,
    summarize_model_artifact,
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


def test_find_model_artifact_manifests_returns_sorted_paths(tmp_path) -> None:
    second = _write_manifest_fixture(tmp_path / "b-artifact", created_at="2026-01-02T00:00:00+00:00")
    first = _write_manifest_fixture(tmp_path / "a-artifact", created_at="2026-01-01T00:00:00+00:00")

    manifests = find_model_artifact_manifests(str(tmp_path))

    assert manifests == sorted([str(first), str(second)])


def test_list_model_artifacts_handles_empty_dir(tmp_path) -> None:
    assert list_model_artifacts(str(tmp_path / "missing")) == []


def test_list_model_artifacts_handles_invalid_manifest_without_crashing(tmp_path) -> None:
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    artifacts = list_model_artifacts(str(tmp_path))

    assert len(artifacts) == 1
    assert artifacts[0]["validation_status"] == "error"


def test_summarize_model_artifact_uses_safe_summary_without_weights(tmp_path) -> None:
    manifest_path = _write_manifest_fixture(tmp_path / "fixture")

    summary = summarize_model_artifact(str(manifest_path))
    serialized = str(summary)

    assert summary["validation_status"] == "ok"
    assert summary["feature_count"] == 1
    assert summary["validation_macro_f1"] == 0.8
    assert "state_dict" not in serialized
    assert "model.pt" not in serialized
    assert str(tmp_path) not in serialized


def test_generated_model_artifacts_are_ignored_by_git() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "data/gpuboost/generated/" in gitignore
    assert "*.pt" in gitignore
    assert "*.pth" in gitignore
    assert "*.safetensors" in gitignore


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


def _write_manifest_fixture(
    artifact_dir: Path,
    *,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> Path:
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.pt").write_bytes(b"not inspected by summaries")
    (artifact_dir / "feature_spec.json").write_text(
        (
            '{'
            '"feature_names":["features.safe_signal"],'
            '"categorical_features":[],'
            '"numeric_features":["features.safe_signal"],'
            '"boolean_features":[]'
            '}'
        ),
        encoding="utf-8",
    )
    (artifact_dir / "label_mapping.json").write_text(
        '{"improved":0,"regressed":1}',
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "evaluation_report.json").write_text("{}", encoding="utf-8")
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        (
            "{"
            '"schema_version":"training.model_artifact.v1",'
            '"artifact_type":"mlp_classifier",'
            f'"created_at":"{created_at}",'
            '"model_name":"mlp_classifier",'
            '"model_format":"torch_state_dict",'
            '"model_file":"model.pt",'
            '"feature_spec_file":"feature_spec.json",'
            '"label_mapping_file":"label_mapping.json",'
            '"training_config_file":"training_config.json",'
            '"evaluation_report_file":"evaluation_report.json",'
            '"input_size":1,'
            '"output_size":2,'
            '"labels":["improved","regressed"],'
            '"feature_names":["features.safe_signal"],'
            '"validation_macro_f1":0.8,'
            '"test_macro_f1":0.76,'
            '"baseline_macro_f1":0.7,'
            '"beats_baseline":true,'
            '"target_macro_f1":0.85,'
            '"target_met":false,'
            '"warnings":[],'
            '"metadata":{}'
            "}"
        ),
        encoding="utf-8",
    )
    return manifest_path


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
