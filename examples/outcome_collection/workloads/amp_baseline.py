"""Controlled FP32 baseline workload for AMP comparison."""

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
    batch_size = 8 if smoke_mode() else args.batch_size
    feature_size = 64 if smoke_mode() else args.feature_size
    hidden_size = 64 if smoke_mode() else args.hidden_size
    warmup_iterations = 1 if smoke_mode() else args.warmup
    measured_iterations = 2 if smoke_mode() else args.num_batches
    metadata = _metadata(args, batch_size, feature_size, hidden_size)

    if torch is None:
        samples_per_sec, median_step_ms, samples = fallback_measurement(
            batch_size=batch_size,
            iterations=measured_iterations,
        )
        emit_result(
            workload="amp",
            variant="baseline",
            cuda_available=False,
            gpu_name=None,
            metrics=_metrics(samples_per_sec, median_step_ms, samples, batch_size),
            extra_metadata={**metadata, "torch_available": False, "amp_used": False},
        )
        return

    torch.manual_seed(22)
    device, cuda_available, gpu_name = resolve_device(torch, args.device)
    model = torch.nn.Sequential(
        torch.nn.Linear(feature_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 16),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    inputs = torch.randn(batch_size, feature_size, device=device)
    targets = torch.randn(batch_size, 16, device=device)
    loss_fn = torch.nn.MSELoss()

    def step() -> int:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        return batch_size

    samples_per_sec, median_step_ms, samples = timed_iterations(
        step,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        torch_module=torch,
        cuda_available=cuda_available,
    )
    emit_result(
        workload="amp",
        variant="baseline",
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        metrics=_metrics(samples_per_sec, median_step_ms, samples, batch_size),
        extra_metadata={
            **metadata,
            "torch_available": True,
            "actual_device": device,
            "amp_used": False,
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
    parser.add_argument("--workload-id", default="amp")
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--feature-size", type=positive_int, default=256)
    parser.add_argument("--hidden-size", type=positive_int, default=256)
    parser.add_argument("--num-batches", type=positive_int, default=16)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--amp", type=parse_bool, default=False)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def _metadata(
    args: argparse.Namespace,
    batch_size: int,
    feature_size: int,
    hidden_size: int,
) -> dict:
    return {
        "workload_id": args.workload_id,
        "batch_size": batch_size,
        "feature_size": feature_size,
        "hidden_size": hidden_size,
        "num_batches": 2 if smoke_mode() else args.num_batches,
        "warmup": 1 if smoke_mode() else args.warmup,
        "amp_requested": args.amp,
        "requested_device": args.device,
        "smoke_mode": smoke_mode(),
    }


if __name__ == "__main__":
    main()
