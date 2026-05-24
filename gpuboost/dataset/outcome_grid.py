"""Controlled outcome experiment grid generation for GPUBoost Phase 11.10."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from gpuboost.schemas.dataset import create_timestamp


GRID_SCHEMA_VERSION = "dataset.outcome_grid.v1"
DEFAULT_OUTPUT_ROOT = "data/gpuboost/experiments/grid"
DEFAULT_PAIRS_PATH = "data/gpuboost/experiments/grid_pairs.json"
DEFAULT_RUNNER_MANIFEST_PATH = (
    "data/gpuboost/experiments/grid_runner_manifest.json"
)
WORKLOAD_ROOT = "examples/outcome_collection/workloads"


def build_controlled_outcome_grid(
    output_root: str = DEFAULT_OUTPUT_ROOT,
    max_pairs: int | None = None,
) -> list[dict]:
    """Return deterministic controlled outcome pair specs without executing them."""

    output_path = _path_from_text(output_root)
    pairs = _interleave_pair_groups(
        [
            _dataloader_pairs(output_path),
            _amp_pairs(output_path),
            _batch_size_pairs(output_path),
            _neutral_control_pairs(output_path),
        ]
    )

    if max_pairs is not None:
        if max_pairs < 0:
            raise ValueError("max_pairs must be non-negative.")
        return pairs[:max_pairs]
    return pairs


def write_grid_pairs_file(
    pairs: list[dict],
    output_path: str = DEFAULT_PAIRS_PATH,
) -> str:
    """Write a collect-outcomes-compatible pairs JSON file."""

    path = _path_from_text(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "row_id": str(pair["row_id"]),
            "workload_name": str(pair["workload_name"]),
            "baseline_json_path": _relative_path(
                str(pair["baseline_json_path"]),
                path.parent,
            ),
            "optimized_json_path": _relative_path(
                str(pair["optimized_json_path"]),
                path.parent,
            ),
            "hardware": _safe_scalar_dict(pair.get("hardware")),
            "features": _safe_scalar_dict(pair.get("features")),
            "metadata": _safe_scalar_dict(pair.get("metadata")),
        }
        for pair in pairs
    ]
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def write_grid_runner_manifest(
    pairs: list[dict],
    output_path: str = DEFAULT_RUNNER_MANIFEST_PATH,
) -> str:
    """Write a manifest with scripts and arguments for the PowerShell runner."""

    path = _path_from_text(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": GRID_SCHEMA_VERSION,
        "generated_at": create_timestamp(),
        "pair_count": len(pairs),
        "pairs": pairs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for writing the default controlled grid."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-default", action="store_true")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pairs-output", default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--manifest-output", default=DEFAULT_RUNNER_MANIFEST_PATH)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.write_default:
        parser.print_help()
        return 0

    pairs = build_controlled_outcome_grid(
        output_root=args.output_root,
        max_pairs=args.max_pairs,
    )
    pairs_path = write_grid_pairs_file(pairs, output_path=args.pairs_output)
    manifest_path = write_grid_runner_manifest(
        pairs,
        output_path=args.manifest_output,
    )
    print(
        json.dumps(
            {
                "pair_count": len(pairs),
                "pairs_file": pairs_path,
                "runner_manifest": manifest_path,
            },
            sort_keys=True,
        )
    )
    return 0


def _dataloader_pairs(output_root: Path) -> list[dict]:
    variants = _dataloader_variants()
    pairs = []
    for index, variant in enumerate(variants, start=1):
        row_id = f"controlled_grid_dataloader_{index:03d}"
        workload_name = (
            "dataloader_grid_"
            f"b{variant['batch_size']}_"
            f"f{variant['feature_size']}_"
            f"n{variant['num_batches']}_"
            f"{variant['device']}"
        )
        common_args = [
            "--workload-id",
            workload_name,
            "--batch-size",
            str(variant["batch_size"]),
            "--feature-size",
            str(variant["feature_size"]),
            "--num-batches",
            str(variant["num_batches"]),
            "--warmup",
            str(variant["warmup"]),
            "--num-workers",
            str(variant["num_workers"]),
            "--device",
            str(variant["device"]),
        ]
        optimized_args = [
            *common_args,
            "--pin-memory",
            _bool_arg(bool(variant["optimized_pin_memory"])),
        ]
        pairs.append(
            _pair(
                row_id=row_id,
                workload_name=workload_name,
                family="dataloader",
                output_root=output_root,
                baseline_script=f"{WORKLOAD_ROOT}/dataloader_baseline.py",
                optimized_script=f"{WORKLOAD_ROOT}/dataloader_optimized.py",
                baseline_args=[*common_args, "--pin-memory", "false"],
                optimized_args=optimized_args,
                metadata=variant,
            )
        )
    return pairs


def _amp_pairs(output_root: Path) -> list[dict]:
    variants = _amp_variants()
    pairs = []
    for index, variant in enumerate(variants, start=1):
        row_id = f"controlled_grid_amp_{index:03d}"
        workload_name = (
            "amp_grid_"
            f"b{variant['batch_size']}_"
            f"f{variant['feature_size']}_"
            f"h{variant['hidden_size']}_"
            f"{variant['device']}"
        )
        common_args = [
            "--workload-id",
            workload_name,
            "--batch-size",
            str(variant["batch_size"]),
            "--feature-size",
            str(variant["feature_size"]),
            "--hidden-size",
            str(variant["hidden_size"]),
            "--num-batches",
            str(variant["num_batches"]),
            "--warmup",
            str(variant["warmup"]),
            "--device",
            str(variant["device"]),
        ]
        pairs.append(
            _pair(
                row_id=row_id,
                workload_name=workload_name,
                family="amp",
                output_root=output_root,
                baseline_script=f"{WORKLOAD_ROOT}/amp_baseline.py",
                optimized_script=f"{WORKLOAD_ROOT}/amp_optimized.py",
                baseline_args=[*common_args, "--amp", "false"],
                optimized_args=[*common_args, "--amp", "true"],
                metadata=variant,
            )
        )
    return pairs


def _batch_size_pairs(output_root: Path) -> list[dict]:
    variants = _batch_size_variants()
    pairs = []
    for index, variant in enumerate(variants, start=1):
        row_id = f"controlled_grid_batch_{index:03d}"
        workload_name = (
            "batch_size_grid_"
            f"b{variant['baseline_batch_size']}_"
            f"to{variant['optimized_batch_size']}_"
            f"f{variant['feature_size']}_"
            f"{variant['device']}"
        )
        common_args = [
            "--workload-id",
            workload_name,
            "--feature-size",
            str(variant["feature_size"]),
            "--num-batches",
            str(variant["num_batches"]),
            "--warmup",
            str(variant["warmup"]),
            "--device",
            str(variant["device"]),
        ]
        pairs.append(
            _pair(
                row_id=row_id,
                workload_name=workload_name,
                family="batch",
                output_root=output_root,
                baseline_script=f"{WORKLOAD_ROOT}/batch_small_baseline.py",
                optimized_script=f"{WORKLOAD_ROOT}/batch_small_optimized.py",
                baseline_args=[
                    *common_args,
                    "--batch-size",
                    str(variant["baseline_batch_size"]),
                ],
                optimized_args=[
                    *common_args,
                    "--batch-size",
                    str(variant["optimized_batch_size"]),
                ],
                metadata=variant,
            )
        )
    return pairs


def _neutral_control_pairs(output_root: Path) -> list[dict]:
    variants = _neutral_control_variants()
    pairs = []
    for index, variant in enumerate(variants, start=1):
        row_id = f"controlled_grid_neutral_{index:03d}"
        workload_name = (
            "neutral_control_grid_"
            f"b{variant['batch_size']}_"
            f"f{variant['feature_size']}_"
            f"h{variant['hidden_size']}_"
            f"{variant['device']}"
        )
        common_args = [
            "--workload-id",
            workload_name,
            "--batch-size",
            str(variant["batch_size"]),
            "--feature-size",
            str(variant["feature_size"]),
            "--hidden-size",
            str(variant["hidden_size"]),
            "--num-batches",
            str(variant["num_batches"]),
            "--warmup",
            str(variant["warmup"]),
            "--device",
            str(variant["device"]),
        ]
        pairs.append(
            _pair(
                row_id=row_id,
                workload_name=workload_name,
                family="neutral_control",
                output_root=output_root,
                baseline_script=f"{WORKLOAD_ROOT}/neutral_baseline.py",
                optimized_script=f"{WORKLOAD_ROOT}/neutral_optimized.py",
                baseline_args=common_args,
                optimized_args=common_args,
                metadata=variant,
            )
        )
    return pairs


def _pair(
    *,
    row_id: str,
    workload_name: str,
    family: str,
    output_root: Path,
    baseline_script: str,
    optimized_script: str,
    baseline_args: list[str],
    optimized_args: list[str],
    metadata: dict[str, str | int | bool],
) -> dict:
    pair_root = output_root / row_id
    features = {
        "controlled_grid": True,
        "workload_family": family,
        "phase": "11.10",
    }
    safe_metadata = {
        "grid_schema_version": GRID_SCHEMA_VERSION,
        "workload_family": family,
        **metadata,
    }
    return {
        "row_id": row_id,
        "workload_name": workload_name,
        "baseline_script": baseline_script,
        "optimized_script": optimized_script,
        "baseline_args": list(baseline_args),
        "optimized_args": list(optimized_args),
        "baseline_json_path": _posix(pair_root / "baseline.json"),
        "optimized_json_path": _posix(pair_root / "optimized.json"),
        "hardware": {"gpu_name": "auto"},
        "features": features,
        "metadata": safe_metadata,
    }


def _dataloader_variants() -> list[dict[str, str | int | bool]]:
    variants: list[dict[str, str | int | bool]] = []
    devices = ["auto", "auto", "cpu", "cuda", "auto"]
    for batch_size in (8, 16, 24, 32, 48, 64):
        for feature_size in (64, 128, 192, 256, 384):
            index = len(variants)
            variants.append(
                {
                    "batch_size": batch_size,
                    "feature_size": feature_size,
                    "num_batches": (8, 10, 12, 14, 16)[index % 5],
                    "warmup": (1, 2, 2)[index % 3],
                    "num_workers": (0, 0, 1)[index % 3],
                    "optimized_pin_memory": index % 2 == 0,
                    "device": devices[index % len(devices)],
                }
            )
    return variants


def _amp_variants() -> list[dict[str, str | int | bool]]:
    variants: list[dict[str, str | int | bool]] = []
    devices = ["auto", "auto", "cpu", "cuda", "auto"]
    for batch_size in (16, 32, 64, 96, 128):
        for feature_size in (128, 256, 384):
            for hidden_multiplier in (1, 2):
                index = len(variants)
                variants.append(
                    {
                        "batch_size": batch_size,
                        "feature_size": feature_size,
                        "hidden_size": feature_size * hidden_multiplier,
                        "num_batches": (8, 10, 12, 16, 20)[index % 5],
                        "warmup": (1, 2, 2)[index % 3],
                        "device": devices[index % len(devices)],
                    }
                )
    return variants


def _batch_size_variants() -> list[dict[str, str | int | bool]]:
    variants: list[dict[str, str | int | bool]] = []
    batch_pairs = ((2, 4), (2, 8), (2, 16), (4, 16), (4, 32), (8, 32))
    devices = ["auto", "auto", "cpu", "cuda", "auto"]
    for feature_size in (128, 192, 256, 384, 512):
        for baseline_batch_size, optimized_batch_size in batch_pairs:
            index = len(variants)
            variants.append(
                {
                    "baseline_batch_size": baseline_batch_size,
                    "optimized_batch_size": optimized_batch_size,
                    "feature_size": feature_size,
                    "num_batches": (8, 10, 12, 16, 20)[index % 5],
                    "warmup": (1, 2, 2)[index % 3],
                    "device": devices[index % len(devices)],
                }
            )
    return variants


def _neutral_control_variants() -> list[dict[str, str | int | bool]]:
    variants: list[dict[str, str | int | bool]] = []
    devices = ["auto", "cpu", "auto", "cuda", "auto"]
    for batch_size in (8, 16, 32, 64, 96):
        for feature_size in (64, 128, 192, 256, 320, 384):
            index = len(variants)
            variants.append(
                {
                    "batch_size": batch_size,
                    "feature_size": feature_size,
                    "hidden_size": feature_size,
                    "num_batches": (8, 10, 12, 16, 20)[index % 5],
                    "warmup": (1, 2, 2)[index % 3],
                    "device": devices[index % len(devices)],
                }
            )
    return variants


def _interleave_pair_groups(groups: list[list[dict]]) -> list[dict]:
    pairs: list[dict] = []
    max_length = max((len(group) for group in groups), default=0)
    for index in range(max_length):
        for group in groups:
            if index < len(group):
                pairs.append(group[index])
    return pairs


def _relative_path(path: str, base_dir: Path) -> str:
    try:
        relative = os.path.relpath(_path_from_text(path), base_dir)
    except ValueError:
        return _posix(_path_from_text(path))
    return _posix(_path_from_text(relative))


def _safe_scalar_dict(value: Any) -> dict[str, str | int | float | bool | None]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, str | int | float | bool) or item is None
    }


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def _posix(path: Path) -> str:
    return path.as_posix().replace("\\", "/")


def _path_from_text(value: str) -> Path:
    return Path(value.replace("\\", "/"))


if __name__ == "__main__":
    raise SystemExit(main())
