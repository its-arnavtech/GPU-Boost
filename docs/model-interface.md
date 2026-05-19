# Local Model Interface

Phase 10 adds a local model interface and model-ready agent layer. It does not
include a trained GPUBoost model, load a real model, train a model, collect
datasets, export data, or call external LLM APIs.

Today, `--model` routes the agent workflow through the local model interface.
Because no provider is configured yet, GPUBoost uses `NullModelProvider` and
returns an explicit fallback artifact:

```json
{
  "model_available": false,
  "fallback_used": true,
  "status": "fallback",
  "warnings": ["No local model provider configured; skipped model inference."]
}
```

The deterministic GPUBoost workflow remains the source of truth. The model
layer may later rank recommendations, score confidence, or predict likely
outcomes, but it cannot apply patches, override measured benchmark data, or
replace deterministic advisor decisions.

## Safe Features

Model inputs are safe summaries only. They may include counts, statuses,
hardware summaries, benchmark metrics, advisor counts, trial status,
comparison status, and history summary flags.

They do not include:

- raw source code
- raw unified diffs
- stdout
- stderr

Model outputs are advisory metadata only. They are stored under
`artifacts.model` in `agent.optimize.v1` JSON and shown in a human `Model`
section when `--model` is used.

## Commands

```bash
python -m gpuboost agent optimize --model
python -m gpuboost agent optimize --model --json
python -m gpuboost agent optimize .\examples\bad_train_sample.txt --model --trial --json
```

Without `--model`, `artifacts.model` is `null`. With `--model`,
`artifacts.model` contains a `ModelInferenceResult` dictionary. `--model` can
be combined with `--trial` and `--save-history`; the trial and history
artifacts remain present alongside the model artifact.

## Roadmap

- Phase 11 will add data collection and validation.
- Phase 12 will train and integrate GPUBoost's own model.
- Phase 13 will test the full production system.
