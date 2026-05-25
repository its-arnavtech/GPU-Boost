"""Realistic lightweight CNN image-classification baseline workload."""

from __future__ import annotations

import argparse

from _common import (
    emit_result,
    load_torch,
    positive_int,
    resolve_device,
    set_deterministic_seed,
    timed_steps,
)


WORKLOAD_FAMILY = "cnn_image_classification"
VARIANT = "baseline"


def main() -> None:
    args = _parse_args()
    torch = load_torch()
    set_deterministic_seed(torch, 1401)
    device, cuda_available = resolve_device(torch, args.device)
    batch_size = 4 if args.quick else args.batch_size
    warmup_steps = 1 if args.quick else args.warmup_steps
    measured_steps = 2 if args.quick else args.steps

    model = SmallCnn(torch).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.arange(batch_size, dtype=torch.long) % 10

    def step() -> int:
        optimizer.zero_grad()
        batch = images.to(device)
        target = labels.to(device)
        logits = model(batch)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
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
            "amp_used": False,
            "device": device,
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
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--steps", type=positive_int, default=6)
    parser.add_argument("--warmup-steps", type=positive_int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
