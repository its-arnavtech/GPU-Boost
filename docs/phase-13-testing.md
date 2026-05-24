# Phase 13 Testing

Phase 13 is testing and release hardening. It is not new model development, not
large-model training, and not an LLM fine-tuning phase. The goal is to prove
that the Phase 11 dataset readiness work and Phase 12 local model workflow are
safe, repeatable, documented, and demo-ready.

Normal verification should run without CUDA, network access, external APIs,
scraping, downloads, committed generated artifacts, or heavy training.

## Required Checks

Run these from the repository root:

```bash
python -m ruff check .
python -m pytest
python -m gpuboost model safety-check --json
```

Expected results:

- Ruff exits cleanly.
- Pytest passes on CPU-only machines.
- `model safety-check --json` returns `ok` or a clearly explained non-blocking
  warning.
- No raw data, generated data, local databases, model weights, or generated
  model artifacts are tracked.

## Optional Manual Smoke

The manual Phase 12 workflow smoke script is optional for release verification:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_phase_12_model_workflow.ps1
```

This script is intended for a human release pass, not normal CI. It exercises
the local model workflow with bounded settings and writes generated outputs
under ignored paths.

## Safety Boundaries

- Generated artifacts remain ignored under `data/gpuboost/generated/`.
- Raw third-party or local data remains ignored under `data/gpuboost/raw/`.
- Model artifacts, checkpoints, local databases, caches, and secrets must not
  be committed.
- Model predictions are advisory-only.
- Deterministic GPUBoost checks, syntax checks, explicit tests, trial results,
  and benchmark evidence remain authoritative.
- GPUBoost does not apply patches automatically.
- Phase 13 does not fine-tune an LLM, call external APIs, download datasets, or
  scrape websites.
