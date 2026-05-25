# Release Checklist

Use this checklist before tagging or announcing a GPUBoost release after Phase
15 final polish.

## Required

- [ ] CI is green.
- [ ] `python -m ruff check .` is green.
- [ ] `python -m pytest` is green.
- [ ] `python -m gpuboost model safety-check --json` returns `ok` or an
  understood non-blocking warning.
- [ ] No raw data or generated data is tracked.
- [ ] No generated model artifacts, checkpoints, or model weights are tracked.
- [ ] No local databases, caches, or secrets are tracked.
- [ ] Documentation is updated, including setup, quickstart, model training,
  agent CLI, demo workflow, release notes, final project summary, and this
  release checklist.
- [ ] README links to the current workflow and release docs.
- [ ] `python -m gpuboost --version` reports the expected package version.
- [ ] `python -m gpuboost doctor --json` returns `ok` or understood optional
  dependency/CUDA warnings.

## Optional Manual Smoke

- [ ] `powershell -ExecutionPolicy Bypass -File .\scripts\smoke_phase_12_model_workflow.ps1`
  passed.
- [ ] The smoke script was not run, and the release notes say it was skipped.

## Safety Sign-Off

- [ ] Model predictions are advisory-only.
- [ ] `patch_application_allowed=false` appears wherever trained artifact
  predictions are exposed.
- [ ] Deterministic GPUBoost checks remain authoritative.
- [ ] Trial mode modifies only a temporary copy.
- [ ] GPUBoost does not apply patches automatically.
- [ ] No LLM fine-tuning is part of the release.
- [ ] No external APIs, scraping, or downloads are required for normal tests.
- [ ] CUDA is not required for normal tests.
- [ ] Generated artifacts remain ignored under `data/gpuboost/generated/`.
- [ ] Real-world demo docs state that synthetic demos and hardware-specific
  results should not be overclaimed.

## Useful Git Checks

```bash
git ls-files data/gpuboost/generated
git ls-files data/gpuboost/raw
git ls-files "*.pt" "*.pth" "*.onnx" "*.safetensors" "*.pkl" "*.joblib"
git ls-files "*.db" "*.sqlite" "*.sqlite3"
git status --short
```

The first four commands should not show tracked generated artifacts, raw data,
model weights, or local databases. Review `git status --short` before release
so only intentional source, docs, tests, and manifest changes are included.
