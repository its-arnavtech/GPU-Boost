"""Realistic lightweight DataLoader training baseline workload."""

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


WORKLOAD_FAMILY = "dataloader_training"
VARIANT = "baseline"


def main() -> None:
    args = _parse_args()
    torch = load_torch()
    set_deterministic_seed(torch, 1403)
    device, cuda_available = resolve_device(torch, args.device)
    batch_size = 8 if args.quick else args.batch_size
    warmup_steps = 3 if args.quick else args.warmup_steps
    measured_steps = 10 if args.quick else args.steps
    dataset_size = batch_size * (warmup_steps + measured_steps + 2)

    features = torch.randn(dataset_size, args.feature_size)
    labels = torch.arange(dataset_size, dtype=torch.long) % 4
    dataset = SyntheticTabularDataset(features, labels)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(args.feature_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    iterator = iter(loader)

    def step() -> int:
        nonlocal iterator
        try:
            features, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            features, labels = next(iterator)

        optimizer.zero_grad()
        features = features.clone().to(device)
        labels = labels.clone().to(device)
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        if float(loss.item()) < 0.0:
            raise RuntimeError("unreachable")
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
            "device": device,
            "num_workers": 0,
            "pin_memory": False,
            "quick": args.quick,
        },
    )


class SyntheticTabularDataset:
    """Module-level map-style dataset.

    Defined at module scope (not as a closure) so it is picklable for
    DataLoader worker processes under the Windows/macOS ``spawn`` start method.
    """

    def __init__(self, features, labels) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--feature-size", type=positive_int, default=64)
    parser.add_argument("--steps", type=positive_int, default=8)
    parser.add_argument("--warmup-steps", type=positive_int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
