"""Phase 14 real-world demo benchmark pair helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = "data/gpuboost/generated/demo_real_world"
DEFAULT_PAIRS_PATH = "data/gpuboost/generated/demo_real_world/pairs.json"
EXAMPLE_ROOT = "examples/real_world"
DEMO_SCHEMA_VERSION = "demo.real_world.v1"


def build_real_world_demo_pairs(
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> list[dict]:
    """Return deterministic before/after pair specs for real-world demos."""

    output_path = _path_from_text(output_root)
    return [
        _pair(
            row_id="cnn_real_world",
            workload_name="cnn_real_world",
            workload_family="cnn_image_classification",
            output_root=output_path,
            baseline_script=f"{EXAMPLE_ROOT}/pytorch_cnn_baseline.py",
            optimized_script=f"{EXAMPLE_ROOT}/pytorch_cnn_optimized.py",
            features={
                "phase": "14B",
                "real_world_demo": True,
                "uses_synthetic_data": True,
                "optimization_amp": True,
                "optimization_non_blocking": True,
            },
        ),
        _pair(
            row_id="transformer_toy_real_world",
            workload_name="transformer_toy_real_world",
            workload_family="toy_transformer_text_classification",
            output_root=output_path,
            baseline_script=f"{EXAMPLE_ROOT}/transformer_toy_baseline.py",
            optimized_script=f"{EXAMPLE_ROOT}/transformer_toy_optimized.py",
            features={
                "phase": "14B",
                "real_world_demo": True,
                "uses_synthetic_data": True,
                "optimization_amp": True,
                "optimization_batching": True,
            },
        ),
        _pair(
            row_id="dataloader_real_world",
            workload_name="dataloader_real_world",
            workload_family="dataloader_training",
            output_root=output_path,
            baseline_script=f"{EXAMPLE_ROOT}/dataloader_training_baseline.py",
            optimized_script=f"{EXAMPLE_ROOT}/dataloader_training_optimized.py",
            features={
                "phase": "14B",
                "real_world_demo": True,
                "uses_synthetic_data": True,
                "optimization_dataloader": True,
                "optimization_non_blocking": True,
            },
        ),
    ]


def write_real_world_pairs_file(
    pairs: list[dict],
    output_path: str = DEFAULT_PAIRS_PATH,
) -> str:
    """Write a collect-outcomes-compatible real-world demo pairs file."""

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


def _pair(
    *,
    row_id: str,
    workload_name: str,
    workload_family: str,
    output_root: Path,
    baseline_script: str,
    optimized_script: str,
    features: dict[str, str | int | float | bool | None],
) -> dict:
    pair_root = output_root / row_id
    metadata = {
        "demo_schema_version": DEMO_SCHEMA_VERSION,
        "example": "real_world",
        "workload_family": workload_family,
        "variant_pair": "baseline_vs_optimized",
        "quick_mode": True,
        "benchmark_json": True,
        "no_downloads": True,
        "no_external_apis": True,
    }
    return {
        "row_id": row_id,
        "workload_name": workload_name,
        "baseline_script": baseline_script,
        "optimized_script": optimized_script,
        "baseline_args": ["--quick", "--benchmark-json"],
        "optimized_args": ["--quick", "--benchmark-json"],
        "baseline_json_path": _posix(pair_root / "baseline.json"),
        "optimized_json_path": _posix(pair_root / "optimized.json"),
        "hardware": {"gpu_name": "auto"},
        "features": features,
        "metadata": metadata,
    }


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


def _path_from_text(value: str) -> Path:
    return Path(value.replace("\\", "/"))


def _posix(path: Path) -> str:
    return path.as_posix().replace("\\", "/")
