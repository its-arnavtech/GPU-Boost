# Real-World Validation

Phase 14 moves GPUBoost beyond controlled training rows into realistic,
demo-friendly PyTorch workloads. The goal is to show that GPUBoost can analyze,
benchmark, compare, collect outcomes from, and report on user-style before/after
workloads while keeping the workflow local, deterministic, and safe.

The Phase 14 demos prove that the pipeline can run realistic lightweight
examples end to end. They do not prove universal speedups, production model
quality, or correctness for every user workload.

## Realistic Workload Examples

The real-world examples live under `examples/real_world/`:

- CNN image classification-style training with synthetic image tensors.
- Toy transformer text classification with synthetic token IDs.
- Synthetic `Dataset` and `DataLoader` training loops.

Each workload has a baseline and optimized script. Baselines intentionally use
conservative settings. Optimized scripts apply safe improvements such as CUDA
AMP when available, non-blocking transfers when CUDA is active, better loop
handling, and safer DataLoader settings.

The examples use synthetic data only. They do not download datasets, call
external APIs, scrape websites, or require CUDA. CPU fallback is supported, but
CPU fallback does not validate CUDA-specific behavior such as AMP or pinned
memory transfer benefits.

## Run Demo Benchmark Pairs

Run the Phase 14 real-world demo benchmark runner:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_real_world_demo_benchmarks.ps1
```

The runner writes generated benchmark JSON and `pairs.json` under:

```text
data/gpuboost/generated/demo_real_world/
```

This directory is ignored by git. Generated artifacts are ignored and should not
be committed.

## Compare Baseline And Optimized JSON

The runner prints comparison commands for each pair, for example:

```bash
python -m gpuboost compare data/gpuboost/generated/demo_real_world/cnn_real_world/baseline.json data/gpuboost/generated/demo_real_world/cnn_real_world/optimized.json
```

Comparison verdicts should be interpreted as local evidence for that run:

- `improved`: comparable metrics moved in the expected direction.
- `regressed`: comparable metrics moved in the wrong direction.
- `neutral` or `unchanged`: comparable metrics did not move meaningfully.
- `mixed`: some metrics improved while others regressed.
- `unknown` or `error`: comparable metrics were missing or invalid.

Hardware variability matters. Results can change with GPU model, CPU, memory
bandwidth, driver version, CUDA version, PyTorch build, laptop power mode,
thermals, background load, and whether the system is plugged in.

## Collect Outcomes

After running the demo benchmark pairs, collect outcome rows:

```bash
python -m gpuboost dataset collect-outcomes data/gpuboost/generated/demo_real_world/pairs.json --output-dir data/gpuboost/generated/demo_real_world/outcomes
```

Outcome collection compares saved local JSON files only. It does not execute
arbitrary benchmark commands, apply patches, call network services, or upload
data.

## Model Artifact Advisory Mode

If a local Phase 12 model artifact exists, it can be used as an advisory signal:

```bash
python -m gpuboost agent optimize train.py --model-artifact data/gpuboost/generated/model_training/artifacts/<id>/manifest.json --json
```

Model predictions remain advisory only. They may help rank or score likely
outcomes, but they cannot apply patches, edit files, override benchmark
evidence, or replace deterministic checks. Reports should continue to include
`patch_application_allowed=false`.

Deterministic GPUBoost checks remain authoritative. Static analysis, generated
diffs, trial workspace validation, syntax checks, explicit tests, and measured
before/after benchmark comparisons are the evidence that should decide whether
an optimization is accepted.

## Avoid Overclaiming

Use Phase 14 demo results to say:

- GPUBoost can run and compare realistic lightweight PyTorch examples.
- GPUBoost can prepare outcome pairs for validation and future training data.
- GPUBoost can produce advisory model output while preserving safety
  boundaries.
- GPUBoost can generate reports without committing generated artifacts.

Do not use Phase 14 demo results to claim:

- All user workloads will improve.
- A CPU-only demo proves CUDA-specific optimizations.
- Synthetic data represents real dataset performance.
- Model advisory predictions are authoritative.
- GPUBoost can safely apply patches automatically.

GPUBoost has no automatic patch application. Reviewable diffs and model
advisory output are not permission to modify original source files.
