# Phase 13 Release Readiness

Phase 13 integrates the release-hardening work after Phase 12 local model
training, artifact packaging, and advisory agent integration. It adds tests,
cross-platform script hardening, security/data leak checks, CLI polish, and
release documentation. It does not add a new model architecture, train a large
model, fine-tune an LLM, call external APIs, scrape websites, or download data.

## Completed Capabilities

- End-to-end subprocess CLI smoke tests for model safety checks, baseline
  evaluation, neural training, artifact save/validate/predict, and agent
  `--model-artifact` use.
- Cross-platform PowerShell and path hardening for generated outcome scripts
  and workflow smoke tooling.
- Security, artifact, and data leak audit coverage for CLI JSON, tracked files,
  ignored generated paths, model weights, local databases, caches, and secrets.
- CLI UX and error-message polish for model artifact lifecycle commands and
  advisory model output.
- Documentation for Phase 13 testing, the demo workflow, and the release
  checklist.

## Validation Results

Latest local validation:

```bash
python -m ruff check .
python -m pytest
python -m gpuboost model safety-check --json
git ls-files data/gpuboost/generated
git ls-files data/gpuboost/raw
git ls-files "*.pt" "*.pth" "*.onnx" "*.safetensors" "*.pkl" "*.joblib"
```

Results:

- Ruff: passed.
- Pytest: `937 passed`.
- Pytest warning: `.pytest_cache` could not be written in this workspace due
  to permissions; tests still passed.
- Safety check: `ok`.
- Safety warnings: none.
- Tracked generated data: none.
- Tracked raw data: none.
- Tracked model artifacts/checkpoints: none.

Safety-check summary:

- `generated_dir_ignored=true`
- `raw_data_ignored=true`
- `artifact_extensions_ignored=true`
- `local_db_artifacts_ignored=true`
- `cache_dirs_ignored=true`
- `env_secret_patterns_ignored=true`
- `provider_patch_application_allowed_false=true`
- `patch_application_allowed=false`

## Covered Workflows

- `python -m gpuboost model safety-check --json`
- `python -m gpuboost model evaluate-baselines --json`
- `python -m gpuboost model train-neural --json`
- `python -m gpuboost model train-neural --save-artifact --json`
- `python -m gpuboost model validate-artifact <manifest> --json`
- `python -m gpuboost model check-artifact <manifest> --json`
- `python -m gpuboost model list-artifacts`
- `python -m gpuboost model show-artifact <manifest>`
- `python -m gpuboost model predict-artifact <manifest> --json`
- `python -m gpuboost agent optimize examples/bad_train_sample.txt --json`
- `python -m gpuboost agent optimize examples/bad_train_sample.txt --trial`
- `python -m gpuboost agent optimize examples/bad_train_sample.txt --model-artifact <manifest> --json`

The optional manual smoke script exists:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_phase_12_model_workflow.ps1
```

No `scripts/smoke_phase_13_release_workflow.ps1` script is currently present.
No heavy manual smoke or benchmark grid was run for this readiness pass.

## Safety Position

- Model predictions are advisory only.
- The model cannot apply patches.
- Trained artifact output reports `patch_application_allowed=false`.
- Deterministic GPUBoost checks remain authoritative.
- Trial workspaces apply patches only to temporary copies.
- GPUBoost does not apply patches automatically.
- CLI JSON redacts raw source, raw diffs, stdout, and stderr by default.
- Runtime artifact manifest paths remain usable for follow-up CLI commands.
- Summaries and reports avoid private absolute path leakage where practical.
- No generated artifacts should be committed.

## Known Limitations

- No bundled/default trained GPUBoost model is included.
- Artifact selection remains explicit and manual.
- Current model quality depends on safe feature coverage and available measured
  outcome rows.
- The `0.85` macro F1 target is aspirational, not guaranteed.
- Full benchmark agent mode and automatic benchmark-command comparison remain
  future work.
- No automatic patch application exists.
- No external LLM provider integration exists.

## Remaining Risks

- CPU-only and CUDA-enabled systems may produce different benchmark skip or
  performance details, though tests exercise CPU-safe behavior.
- Real-world model usefulness depends on broader measured user-script outcome
  data, not just controlled workflow rows.
- Shell and path behavior can still vary across older PowerShell versions, but
  Phase 13 coverage exercises current path normalization and UTF-8 JSON writes.
- Private path redaction is best effort for summaries and reports; runtime CLI
  arguments may still include user-provided paths when needed to perform local
  commands.

## Release Recommendation

Phase 13 is release-ready from the local validation perspective. The repo is
merge-ready once the existing uncommitted Phase 13 source, docs, scripts, and
tests are reviewed. Do not commit generated artifacts, raw data, local
databases, model weights, secrets, or cache files.
