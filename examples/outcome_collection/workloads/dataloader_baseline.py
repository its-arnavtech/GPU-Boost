"""Controlled DataLoader baseline workload."""

from __future__ import annotations

import argparse

from _common import (
    emit_result,
    fallback_measurement,
    load_torch,
    parse_bool,
    positive_int,
    resolve_device,
    smoke_mode,
    timed_iterations,
)


def main() -> None:
    args = _parse_args()
    torch = load_torch()
    batch_size = 4 if smoke_mode() else args.batch_size
    feature_size = 32 if smoke_mode() else args.feature_size
    batches = 2 if smoke_mode() else args.num_batches
    warmup_iterations = 1 if smoke_mode() else args.warmup
    measured_iterations = 2 if smoke_mode() else args.num_batches
    num_workers = 0 if smoke_mode() else args.num_workers
    pin_memory = False if args.pin_memory is None else args.pin_memory
    metadata = _metadata(args, batch_size, feature_size, batches, num_workers, pin_memory)

    if torch is None:
        samples_per_sec, median_step_ms, samples = fallback_measurement(
            batch_size=batch_size,
            iterations=measured_iterations,
        )
        emit_result(
            workload="dataloader",
            variant="baseline",
            cuda_available=False,
            gpu_name=None,
            metrics=_metrics(samples_per_sec, median_step_ms, samples),
            extra_metadata={**metadata, "torch_available": False},
        )
        return

    torch.manual_seed(11)
    device, cuda_available, gpu_name = resolve_device(torch, args.device)
    sample_count = batch_size * batches
    features = torch.arange(sample_count * feature_size, dtype=torch.float32).reshape(
        sample_count,
        feature_size,
    )
    targets = torch.arange(sample_count, dtype=torch.float32).reshape(sample_count, 1)
    dataset = torch.utils.data.TensorDataset(features, targets)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    iterator = iter(loader)

    def step() -> int:
        nonlocal iterator
        try:
            batch, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch, target = next(iterator)

        # Baseline intentionally does extra host-side copies before transfer.
        batch = batch.clone()
        target = target.clone()
        batch = batch.to(device)
        target = target.to(device)
        value = (batch.mean() + target.mean()).item()
        if value < 0:
            raise RuntimeError("unreachable")
        return int(batch.shape[0])

    samples_per_sec, median_step_ms, samples = timed_iterations(
        step,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        torch_module=torch,
        cuda_available=cuda_available,
    )
    emit_result(
        workload="dataloader",
        variant="baseline",
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        metrics=_metrics(samples_per_sec, median_step_ms, samples),
        extra_metadata={
            **metadata,
            "torch_available": True,
            "actual_device": device,
        },
    )


def _metrics(samples_per_sec: float, median_step_ms: float, samples: int) -> list[dict]:
    return [
        {"name": "samples_per_sec", "value": samples_per_sec, "unit": "samples/sec"},
        {
            "name": "fp32_samples_per_sec",
            "value": samples_per_sec,
            "unit": "samples/sec",
        },
        {"name": "best_images_per_sec", "value": samples_per_sec, "unit": "samples/sec"},
        {"name": "median_step_ms", "value": median_step_ms, "unit": "ms"},
        {"name": "measured_samples", "value": samples, "unit": "samples"},
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-id", default="dataloader")
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--feature-size", type=positive_int, default=64)
    parser.add_argument("--num-batches", type=positive_int, default=12)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", type=parse_bool, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def _metadata(
    args: argparse.Namespace,
    batch_size: int,
    feature_size: int,
    batches: int,
    num_workers: int,
    pin_memory: bool,
) -> dict:
    return {
        "workload_id": args.workload_id,
        "batch_size": batch_size,
        "feature_size": feature_size,
        "num_batches": batches,
        "warmup": 1 if smoke_mode() else args.warmup,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "requested_device": args.device,
        "smoke_mode": smoke_mode(),
    }


if __name__ == "__main__":
    main()
