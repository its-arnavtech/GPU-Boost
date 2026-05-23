"""Deterministic train/validation/test split assignment for dataset rows."""

from __future__ import annotations

import random
from dataclasses import replace
from math import isclose
from collections import defaultdict

from gpuboost.schemas.dataset import DatasetRow, DatasetSplitSummary, create_timestamp


VALID_SPLIT_NAMES = {"train", "validation", "test"}


def assign_dataset_splits(
    rows: list[DatasetRow],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[list[DatasetRow], DatasetSplitSummary]:
    """Return copied rows with deterministic train/validation/test splits."""

    validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    copied_rows = [replace(row) for row in rows]

    if overwrite:
        assignment_indexes = list(range(len(copied_rows)))
    else:
        assignment_indexes = [
            index for index, row in enumerate(copied_rows) if row.split is None
        ]

    shuffled_indexes = list(assignment_indexes)
    random.Random(seed).shuffle(shuffled_indexes)
    assignments = _assignment_sequence(
        len(shuffled_indexes),
        train_ratio,
        validation_ratio,
    )

    for index, split in zip(shuffled_indexes, assignments):
        copied_rows[index] = replace(copied_rows[index], split=split)

    counts = split_counts(copied_rows)
    summary = DatasetSplitSummary(
        generated_at=create_timestamp(),
        train_count=counts["train"],
        validation_count=counts["validation"],
        test_count=counts["test"],
        unassigned_count=counts["unassigned"],
        strategy="deterministic_random",
        seed=seed,
        metadata={
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "overwrite": overwrite,
        },
    )
    return copied_rows, summary


def assign_grouped_stratified_splits(
    rows: list[DatasetRow],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[list[DatasetRow], DatasetSplitSummary]:
    """Assign deterministic splits while keeping related row groups together."""

    validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    copied_rows = [replace(row) for row in rows]

    grouped_indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(copied_rows):
        if not overwrite and row.split is not None:
            continue
        grouped_indexes[_split_group_key(row)].append(index)

    group_items = list(grouped_indexes.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    groups_by_label: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for group_key, indexes in group_items:
        label = _primary_group_label(copied_rows[index] for index in indexes)
        groups_by_label[label].append((group_key, indexes))

    for label in sorted(groups_by_label):
        groups = groups_by_label[label]
        assignments = _assignment_sequence(
            len(groups),
            train_ratio,
            validation_ratio,
        )
        for (_group_key, indexes), split in zip(groups, assignments):
            for index in indexes:
                copied_rows[index] = replace(copied_rows[index], split=split)

    counts = split_counts(copied_rows)
    summary = DatasetSplitSummary(
        generated_at=create_timestamp(),
        train_count=counts["train"],
        validation_count=counts["validation"],
        test_count=counts["test"],
        unassigned_count=counts["unassigned"],
        strategy="grouped_stratified",
        seed=seed,
        metadata={
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "overwrite": overwrite,
            "group_count": len(grouped_indexes),
        },
    )
    return copied_rows, summary


def validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """Validate split ratios."""

    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    if not isclose(sum(ratios), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0.")


def split_counts(rows: list[DatasetRow]) -> dict[str, int]:
    """Return train, validation, test, and unassigned split counts."""

    counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
        "unassigned": 0,
    }
    for row in rows:
        if row.split in VALID_SPLIT_NAMES:
            counts[row.split] += 1
        else:
            counts["unassigned"] += 1
    return counts


def split_group_key(row: DatasetRow) -> str:
    """Return the stable grouping key used by grouped split assignment."""

    return _split_group_key(row)


def _assignment_sequence(
    count: int,
    train_ratio: float,
    validation_ratio: float,
) -> list[str]:
    if count == 0:
        return []

    train_count = max(1, int(count * train_ratio))
    validation_count = int(count * validation_ratio)
    if train_count + validation_count > count:
        validation_count = max(0, count - train_count)
    test_count = count - train_count - validation_count

    return (
        ["train"] * train_count
        + ["validation"] * validation_count
        + ["test"] * test_count
    )


def _split_group_key(row: DatasetRow) -> str:
    metadata_family = row.metadata.get("grid_family") or row.metadata.get(
        "workload_family"
    )
    feature_family = row.features.get("workload_family")
    workload_name = row.workload.get("workload_name")

    if isinstance(metadata_family, str) and metadata_family:
        return f"metadata.grid_family:{metadata_family}"
    if isinstance(feature_family, str) and feature_family:
        return f"features.workload_family:{feature_family}"
    if isinstance(workload_name, str) and workload_name:
        return f"workload.workload_name:{workload_name}"

    prefix = _controlled_row_prefix(row.row_id)
    if prefix:
        return f"row_id_prefix:{prefix}"
    return f"row_id:{row.row_id}"


def _controlled_row_prefix(row_id: str) -> str | None:
    parts = row_id.split("_")
    if len(parts) >= 3 and parts[0] == "controlled" and parts[1] == "grid":
        return "_".join(parts[:3])
    return None


def _primary_group_label(rows: object) -> str:
    labels: dict[str, int] = {}
    for row in rows:
        label = row.label.value if row.label.is_known() else "unknown"
        labels[label] = labels.get(label, 0) + 1
    if not labels:
        return "unknown"
    return sorted(labels.items(), key=lambda item: (-item[1], item[0]))[0][0]
