# Real-World PyTorch Demo Workloads

These scripts are realistic, lightweight PyTorch workloads for GPUBoost Phase 14
validation and demos. They model common user training loops while staying small
enough for local smoke tests and normal CI.

The examples use synthetic tensors only. They do not download datasets, call
external APIs, scrape data, or write files unless a future caller explicitly adds
an output path. Every workload can run on CPU, and it will use CUDA safely when
PyTorch reports that CUDA is available.

Workload pairs:

- `pytorch_cnn_baseline.py` and `pytorch_cnn_optimized.py`: image
  classification-style CNN training.
- `transformer_toy_baseline.py` and `transformer_toy_optimized.py`: small
  transformer-like text classification with synthetic token IDs.
- `dataloader_training_baseline.py` and `dataloader_training_optimized.py`:
  synthetic `Dataset` and `DataLoader` training loops.

Baseline scripts intentionally keep conservative settings such as smaller
batches, blocking transfers, no AMP, or simple `DataLoader` options. Optimized
scripts apply safe improvements where available, including CUDA AMP,
`non_blocking` transfers, improved batch sizing, `set_to_none=True`, and
CUDA-aware `pin_memory` behavior.

Run a quick benchmark JSON smoke test:

```bash
python examples/real_world/pytorch_cnn_baseline.py --quick --benchmark-json
python examples/real_world/pytorch_cnn_optimized.py --quick --benchmark-json
```

**Measurement caveat.** These workloads are intentionally tiny, so their per-step
time is dominated by launch overhead and is noisy run-to-run. `--quick` runs only
a handful of steps and is a smoke test, not a measurement — do not treat a single
`--quick` baseline-vs-optimized delta as a real speedup. For actual numbers, pass
`--steps 50` (or more) and compare the median of several runs, and prefer
`python -m gpuboost benchmark --recommend`, whose larger, steadier workloads give
a far more reliable signal.

When `--benchmark-json` is passed, stdout contains only benchmark-compatible
JSON:

```json
{
  "results": [
    {
      "name": "cnn_image_classification:baseline",
      "status": "ok",
      "metrics": [
        {"name": "samples_per_sec", "value": 123.4, "unit": "samples/sec"},
        {"name": "median_step_ms", "value": 12.3, "unit": "ms"}
      ]
    }
  ],
  "metadata": {
    "example": "real_world",
    "workload_family": "cnn_image_classification",
    "variant": "baseline",
    "cuda_available": false
  }
}
```

The JSON shape is intended to feed GPUBoost comparison, outcome, analysis,
trial, and demo tooling without requiring CUDA or any external data.
