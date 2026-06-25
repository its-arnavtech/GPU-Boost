"""Realistic lightweight CNN image-classification optimized workload."""

from __future__ import annotations

import argparse

from _common import (
    autocast_context,
    emit_result,
    load_torch,
    make_grad_scaler,
    positive_int,
    resolve_device,
    set_deterministic_seed,
    timed_steps,
)


WORKLOAD_FAMILY = "cnn_image_classification"
VARIANT = "optimized"


def main() -> None:
    args = _parse_args()
    torch = load_torch()
    set_deterministic_seed(torch, 1401)
    device, cuda_available = resolve_device(torch, args.device)
    batch_size = 8 if args.quick else args.batch_size
    warmup_steps = 3 if args.quick else args.warmup_steps
    measured_steps = 10 if args.quick else args.steps

    model = SmallCnn(torch).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    images = torch.randn(batch_size, 3, 32, 32, pin_memory=cuda_available)
    labels = torch.arange(batch_size, dtype=torch.long, pin_memory=cuda_available) % 10
    scaler = make_grad_scaler(torch, cuda_available, enabled=True)

    def step() -> int:
        optimizer.zero_grad(set_to_none=True)
        batch = images.to(device, non_blocking=cuda_available)
        target = labels.to(device, non_blocking=cuda_available)
        with autocast_context(torch, cuda_available, enabled=True):
            logits = model(batch)
            loss = loss_fn(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return batch_size

    samples_per_sec, median_step_ms, _ = timed_steps(
        step,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        torch_module=torch,
        cuda_available=cuda_available,
    )
    emit_result(
        benchmark_json=args.benchmark_json,
        workload_family=WORKLOAD_FAMILY,
        variant=VARIANT,
        cuda_available=cuda_available,
        samples_per_sec=samples_per_sec,
        median_step_ms=median_step_ms,
        extra_metadata={
            "batch_size": batch_size,
            "amp_used": cuda_available,
            "device": device,
            "non_blocking": cuda_available,
            "quick": args.quick,
        },
    )


class SmallCnn:
    """A tiny user-style CNN without external model dependencies."""

    def __new__(cls, torch_module):
        return torch_module.nn.Sequential(
            torch_module.nn.Conv2d(3, 12, kernel_size=3, padding=1),
            torch_module.nn.ReLU(),
            torch_module.nn.MaxPool2d(2),
            torch_module.nn.Conv2d(12, 24, kernel_size=3, padding=1),
            torch_module.nn.ReLU(),
            torch_module.nn.AdaptiveAvgPool2d((4, 4)),
            torch_module.nn.Flatten(),
            torch_module.nn.Linear(24 * 4 * 4, 10),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--steps", type=positive_int, default=6)
    parser.add_argument("--warmup-steps", type=positive_int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
