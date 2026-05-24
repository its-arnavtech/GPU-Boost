# Demo Workflow

This demo shows the end-to-end GPUBoost workflow after Phase 12. It is designed
to be local-first, review-only, and safe to run without CUDA or network access.
Commands that train use small local settings. Generated artifacts should stay
under ignored `data/gpuboost/generated/` paths.

## 1. Run Agent Optimize

```bash
python -m gpuboost agent optimize examples/bad_train_sample.txt
python -m gpuboost agent optimize examples/bad_train_sample.txt --json
```

The agent inspects the system, runs the quick benchmark path, generates
deterministic recommendations, analyzes the sample script, and emits a
reviewable diff when safe patch suggestions exist.

GPUBoost does not apply patches automatically. Diffs are for human review.

## 2. Run A Trial Workspace

```bash
python -m gpuboost agent optimize examples/bad_train_sample.txt --trial
python -m gpuboost agent optimize examples/bad_train_sample.txt --trial --json
```

Trial mode copies the target file to a temporary workspace, applies suggested
patches only to that copy, and runs syntax validation. The original
`examples/bad_train_sample.txt` file is not modified.

## 3. Evaluate Baselines

```bash
python -m gpuboost model evaluate-baselines --json
```

This reads the configured safe training dataset, evaluates dependency-free
baselines, and writes reports under `data/gpuboost/generated/model_training/`
by default. It does not save a neural model artifact.

## 4. Train Neural Reports Only

```bash
python -m gpuboost model train-neural --max-epochs 50 --max-candidates 12 --json
```

This trains a small local MLP from scratch on safe structured features and
writes reports only. It does not fine-tune an LLM, call external APIs, download
data, or save a production artifact unless `--save-artifact` is explicitly
passed.

## 5. Train And Save An Artifact

```bash
python -m gpuboost model train-neural --max-epochs 50 --max-candidates 12 --save-artifact --json
```

The JSON output includes an artifact manifest path. Use that path as
`<manifest>` in the next commands. The artifact is a local generated artifact
and should remain ignored by Git.

## 6. Validate, Check, List, And Show The Artifact

```bash
python -m gpuboost model validate-artifact <manifest> --json
python -m gpuboost model check-artifact <manifest> --min-test-macro-f1 0.75 --require-beats-baseline --json
python -m gpuboost model list-artifacts
python -m gpuboost model show-artifact <manifest>
```

`validate-artifact` checks that the manifest and referenced files are present
and coherent. `check-artifact` is a read-only quality gate. `list-artifacts`
and `show-artifact` print safe summaries and do not print model weights.

## 7. Predict Directly From The Artifact

```bash
python -m gpuboost model predict-artifact <manifest> --features-json "{\"features.safe_signal\": 1.0}" --json
```

The prediction is local and advisory-only. It cannot apply patches, edit files,
or override deterministic GPUBoost checks.

## 8. Use The Artifact In The Agent

```bash
python -m gpuboost agent optimize examples/bad_train_sample.txt --model-artifact <manifest> --json
```

`--model-artifact` enables local trained artifact inference for the agent
workflow. The agent report includes a model prediction summary and
`patch_application_allowed=false`.

The model is advisory-only. Deterministic recommendations, patch planning,
trial workspace validation, explicit tests, and benchmark evidence remain
authoritative. GPUBoost still does not apply patches automatically.

## Demo Boundaries

- No CUDA is required for the normal demo path; CUDA benchmarks may be skipped
  cleanly on CPU-only machines.
- No network access is required.
- No external APIs are called.
- No scraping or downloading is performed.
- No LLM fine-tuning is performed.
- Generated artifacts remain ignored and should not be committed.
