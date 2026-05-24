# Release Checklist

Use this checklist before tagging or announcing a GPUBoost release after Phase
12 and Phase 13 hardening.

## Required

- [ ] CI is green.
- [ ] `python -m ruff check .` is green.
- [ ] `python -m pytest` is green.
- [ ] `python -m gpuboost model safety-check --json` returns `ok` or an
  understood non-blocking warning.
- [ ] No raw data or generated data is tracked.
- [ ] No generated model artifacts, checkpoints, or model weights are tracked.
- [ ] No local databases, caches, or secrets are tracked.
- [ ] Documentation is updated, including model training, agent CLI, demo
  workflow, Phase 13 testing, and this release checklist.
- [ ] README links to the current workflow and release docs.

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
- [ ] Generated artifacts remain ignored under `data/gpuboost/generated/`.

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
