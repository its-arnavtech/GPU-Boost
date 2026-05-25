"""Realistic lightweight DataLoader training optimized workload."""

from __future__ import annotations

import argparse
import os

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


WORKLOAD_FAMILY = "dataloader_training"
VARIANT = "optimized"


def main() -> None:
    args = _parse_args()
    torch = load_torch()
    set_deterministic_seed(torch, 1403)
    device, cuda_available = resolve_device(torch, args.device)
    batch_size = 16 if args.quick else args.batch_size
    warmup_steps = 1 if args.quick else args.warmup_steps
    measured_steps = 2 if args.quick else args.steps
    num_workers = 0 if args.quick else min(2, os.cpu_count() or 1)
    dataset_size = batch_size * (warmup_steps + measured_steps + 2)

    dataset = SyntheticTabularDataset(torch, dataset_size, args.feature_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cuda_available,
        persistent_workers=num_workers > 0,
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(args.feature_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = make_grad_scaler(torch, cuda_available, enabled=True)
    iterator = iter(loader)

    def step() -> int:
        nonlocal iterator
        try:
            features, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            features, labels = next(iterator)

        optimizer.zero_grad(set_to_none=True)
        features = features.to(device, non_blocking=cuda_available)
        labels = labels.to(device, non_blocking=cuda_available)
        with autocast_context(torch, cuda_available, enabled=True):
            logits = model(features)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return int(features.shape[0])

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
            "num_workers": num_workers,
            "pin_memory": cuda_available,
            "quick": args.quick,
        },
    )


class SyntheticTabularDataset:
    """Factory for a deterministic synthetic Dataset."""

    def __new__(cls, torch_module, size: int, feature_size: int):
        class _Dataset(torch_module.utils.data.Dataset):
            def __init__(self) -> None:
                self.features = torch_module.randn(size, feature_size)
                self.labels = torch_module.arange(size, dtype=torch_module.long) % 4

            def __len__(self) -> int:
                return size

            def __getitem__(self, index: int):
                return self.features[index], self.labels[index]

        return _Dataset()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--feature-size", type=positive_int, default=64)
    parser.add_argument("--steps", type=positive_int, default=8)
    parser.add_argument("--warmup-steps", type=positive_int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
