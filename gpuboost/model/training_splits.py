"""Split helpers for encoded Phase 12 datasets."""

from __future__ import annotations

from typing import Any

from gpuboost.schemas.training import EncodedTrainingDataset

SPLIT_NAMES = ("train", "validation", "test", "unassigned")


def split_encoded_dataset(dataset: EncodedTrainingDataset) -> dict[str, dict[str, Any]]:
    """Group an encoded dataset by split name."""

    _validate_encoded_lengths(dataset)
    splits: dict[str, dict[str, Any]] = {
        split_name: {"X": [], "y": [], "row_ids": []} for split_name in SPLIT_NAMES
    }

    for index, row in enumerate(dataset.X):
        split_name = dataset.split[index] or "unassigned"
        if split_name not in splits:
            split_name = "unassigned"
        splits[split_name]["X"].append(row)
        splits[split_name]["y"].append(dataset.y[index])
        splits[split_name]["row_ids"].append(dataset.row_ids[index])

    return splits


def get_eval_split(splits: dict, preferred: str = "validation") -> str:
    """Return the best available evaluation split name."""

    for split_name in _eval_split_order(preferred):
        split = splits.get(split_name)
        if isinstance(split, dict) and split.get("y"):
            return split_name
    raise ValueError("No non-empty train, validation, or test split is available.")


def _validate_encoded_lengths(dataset: EncodedTrainingDataset) -> None:
    lengths = {len(dataset.X), len(dataset.y), len(dataset.row_ids), len(dataset.split)}
    if len(lengths) != 1:
        raise ValueError("Encoded dataset lengths do not match.")


def _eval_split_order(preferred: str) -> list[str]:
    order = [preferred, "validation", "test", "train"]
    unique_order: list[str] = []
    for split_name in order:
        if split_name in unique_order:
            continue
        unique_order.append(split_name)
    return unique_order
