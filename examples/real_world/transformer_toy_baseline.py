"""Realistic lightweight toy transformer text-classification baseline workload."""

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


WORKLOAD_FAMILY = "toy_transformer_text_classification"
VARIANT = "baseline"


def main() -> None:
    args = _parse_args()
    torch = load_torch()
    set_deterministic_seed(torch, 1402)
    device, cuda_available = resolve_device(torch, args.device)
    batch_size = 4 if args.quick else args.batch_size
    sequence_length = 16 if args.quick else args.sequence_length
    warmup_steps = 3 if args.quick else args.warmup_steps
    measured_steps = 10 if args.quick else args.steps

    model = TinyTransformerClassifier(torch, args.vocab_size, sequence_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    tokens = torch.randint(0, args.vocab_size, (batch_size, sequence_length))
    labels = torch.arange(batch_size, dtype=torch.long) % args.num_classes

    def step() -> int:
        optimizer.zero_grad()
        batch = tokens.to(device)
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
            "sequence_length": sequence_length,
            "amp_used": False,
            "device": device,
            "quick": args.quick,
        },
    )


class TinyTransformerClassifier:
    """Factory for a compact TransformerEncoder classifier."""

    def __new__(cls, torch_module, vocab_size: int, sequence_length: int):
        return torch_module.nn.Sequential(
            TokenAndPositionEmbedding(torch_module, vocab_size, sequence_length),
            torch_module.nn.TransformerEncoder(
                torch_module.nn.TransformerEncoderLayer(
                    d_model=32,
                    nhead=4,
                    dim_feedforward=64,
                    batch_first=True,
                    dropout=0.0,
                ),
                num_layers=1,
            ),
            MeanPool(torch_module),
            torch_module.nn.Linear(32, 2),
        )


class TokenAndPositionEmbedding:
    """Small embedding module created without a global torch import."""

    def __new__(cls, torch_module, vocab_size: int, sequence_length: int):
        class _Embedding(torch_module.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.token = torch_module.nn.Embedding(vocab_size, 32)
                self.position = torch_module.nn.Embedding(sequence_length, 32)
                self.register_buffer(
                    "positions",
                    torch_module.arange(sequence_length).unsqueeze(0),
                    persistent=False,
                )

            def forward(self, input_ids):
                return self.token(input_ids) + self.position(self.positions)

        return _Embedding()


class MeanPool:
    """Mean-pool encoded token states."""

    def __new__(cls, torch_module):
        class _MeanPool(torch_module.nn.Module):
            def forward(self, hidden_states):
                return hidden_states.mean(dim=1)

        return _MeanPool()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=12)
    parser.add_argument("--sequence-length", type=positive_int, default=32)
    parser.add_argument("--vocab-size", type=positive_int, default=128)
    parser.add_argument("--num-classes", type=positive_int, default=2)
    parser.add_argument("--steps", type=positive_int, default=5)
    parser.add_argument("--warmup-steps", type=positive_int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
