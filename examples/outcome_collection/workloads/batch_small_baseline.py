"""Controlled intentionally small-batch workload."""

from __future__ import annotations

import argparse

from _common import (
    emit_result,
    fallback_measurement,
    load_torch,
    positive_int,
    resolve_device,
    smoke_mode,
    timed_iterations,
)


def main() -> None:
    args = _parse_args()
    torch = load_torch()
    batch_size = 2 if smoke_mode() else args.batch_size
    feature_size = 64 if smoke_mode() else args.feature_size
    warmup_iterations = 1 if smoke_mode() else args.warmup
    measured_iterations = 2 if smoke_mode() else args.num_batches
    metadata = _metadata(args, batch_size, feature_size)

    if torch is None:
        samples_per_sec, median_step_ms, samples = fallback_measurement(
            batch_size=batch_size,
            iterations=measured_iterations,
        )
        emit_result(
            workload="batch_size",
            variant="baseline",
            cuda_available=False,
            gpu_name=None,
            metrics=_metrics(samples_per_sec, median_step_ms, samples, batch_size),
            extra_metadata={**metadata, "torch_available": False},
        )
        return

    torch.manual_seed(33)
    device, cuda_available, gpu_name = resolve_device(torch, args.device)
    weight = torch.randn(feature_size, feature_size, device=device)
    inputs = torch.randn(batch_size, feature_size, device=device)

    def step() -> int:
        output = inputs @ weight
        value = output.relu().mean().item()
        if value < 0:
            raise RuntimeError("unreachable")
        return batch_size

    samples_per_sec, median_step_ms, samples = timed_iterations(
        step,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        torch_module=torch,
        cuda_available=cuda_available,
    )
    emit_result(
        workload="batch_size",
        variant="baseline",
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        metrics=_metrics(samples_per_sec, median_step_ms, samples, batch_size),
        extra_metadata={
            **metadata,
            "torch_available": True,
            "actual_device": device,
        },
    )


def _metrics(
    samples_per_sec: float,
    median_step_ms: float,
    samples: int,
    batch_size: int,
) -> list[dict]:
    return [
        {"name": "samples_per_sec", "value": samples_per_sec, "unit": "samples/sec"},
        {
            "name": "fp32_samples_per_sec",
            "value": samples_per_sec,
            "unit": "samples/sec",
        },
        {"name": "best_images_per_sec", "value": samples_per_sec, "unit": "samples/sec"},
        {"name": "median_step_ms", "value": median_step_ms, "unit": "ms"},
        {"name": "best_batch_size", "value": batch_size, "unit": None},
        {"name": "measured_samples", "value": samples, "unit": "samples"},
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-id", default="batch_size")
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--feature-size", type=positive_int, default=256)
    parser.add_argument("--num-batches", type=positive_int, default=16)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def _metadata(
    args: argparse.Namespace,
    batch_size: int,
    feature_size: int,
) -> dict:
    return {
        "workload_id": args.workload_id,
        "batch_size": batch_size,
        "feature_size": feature_size,
        "num_batches": 2 if smoke_mode() else args.num_batches,
        "warmup": 1 if smoke_mode() else args.warmup,
        "requested_device": args.device,
        "smoke_mode": smoke_mode(),
    }


if __name__ == "__main__":
    main()
