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

## Phase 12 Training Plan

Phase 12 should start with a baseline structured model. It should not begin
with fine-tuning, model loading, or patch application.

Training code must use the dataset training feature extraction layer and never
train from raw `DatasetRow.to_dict()` output. Reporting fields can remain on
dataset rows, but target-derived comparison fields such as `overall_verdict`,
before/after metric values, deltas, percent deltas, labels, split names, raw
diffs, stdout, and stderr must not become model features.

Controlled outcome rows are useful baseline training examples, but they are
limited. They come from controlled synthetic workloads and measured benchmark
pairs, not from arbitrary real user scripts. Third-party benchmark data should
be treated as context and provenance, not as direct GPUBoost labels.

The deterministic GPUBoost workflow remains authoritative. A trained model may
rank, score, or annotate recommendations, but it must not apply patches
directly or override measured benchmark comparisons.

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

- Phase 11 adds data collection and validation.
- Phase 12 will train and integrate GPUBoost's own baseline structured model
  after safe feature extraction and grouped validation splits are in place.
- Phase 13 will test the full production system.
