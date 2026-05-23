"""Deterministic train/validation/test split assignment for dataset rows."""

from __future__ import annotations

import random
from dataclasses import replace
from math import isclose

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
