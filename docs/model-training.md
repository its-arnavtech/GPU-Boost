# Model Training

Phase 12 adds GPUBoost's local model workflow: safe training data loading, safe
feature extraction, baseline model comparison, small neural MLP training from
scratch, artifact save/load/validation, direct artifact prediction, and
advisory-only agent integration. It starts with local structured models, not
LLM fine-tuning, and does not call external LLM APIs, download datasets, scrape
websites, train a large production model, guarantee optimization success, or
make model predictions authoritative.

The training path must load `DatasetRow` records and convert them through the
safe training feature extraction layer in `gpuboost.dataset.training_features`.
Do not train directly on `DatasetRow.to_dict()`: that raw schema includes
labels, split metadata, provenance, privacy flags, and fields that are useful
for reporting but unsafe as model inputs.

Training must not use target-derived comparison fields such as labels,
before/after metrics, deltas, verdicts, raw source code, raw diffs, stdout, or
stderr. Phase 12.1 encodes only the safe scalar feature dictionaries emitted by
the Phase 11 extraction layer.

Phase 12.2 adds a baseline comparison command:

```bash
python -m gpuboost model evaluate-baselines
python -m gpuboost model evaluate-baselines --dataset data/gpuboost/generated/training_dataset.jsonl
python -m gpuboost model evaluate-baselines --output-dir data/gpuboost/generated/model_training
python -m gpuboost model evaluate-baselines --json
```

The command loads `DatasetRow` JSONL data, builds an
`EncodedTrainingDataset`, trains only on rows assigned to the `train` split,
evaluates on `validation` with fallback to `test` and then `train`, and writes:

- `baseline_comparison_report.json`
- `baseline_comparison_report.md`

The Phase 12.2 baselines are dependency-free and intentionally small:

- `MajorityClassBaseline`: predicts the most common train label and proves the
  trivial baseline level.
- `RandomBaseline`: predicts seeded random train labels and catches accidental
  evaluation optimism.
- `NearestCentroidBaseline`: computes one numeric feature centroid per class
  and predicts the nearest centroid.
- `SimpleKNNBaseline`: stores train rows and votes among the nearest small
  neighborhood.

Baselines matter because they answer a simple gate question before any neural
training: do the safe features contain enough signal to beat trivial behavior?
If nearest-centroid or KNN cannot improve on majority/random baselines, a neural
model is unlikely to be worth adding yet.

Phase 12.2 still does not save production model artifacts, checkpoints, pickle
files, or provider state. It also does not integrate predictions into the agent.
The generated reports are evaluation artifacts only and live under the ignored
`data/gpuboost/generated/` tree by default.

Phase 12.3 adds local neural training from scratch for safe structured features:

```bash
python -m gpuboost model train-neural
python -m gpuboost model train-neural --dataset data/gpuboost/generated/training_dataset.jsonl
python -m gpuboost model train-neural --output-dir data/gpuboost/generated/model_training
python -m gpuboost model train-neural --max-epochs 50
python -m gpuboost model train-neural --hidden-sizes 32,16
python -m gpuboost model train-neural --target-macro-f1 0.85
python -m gpuboost model train-neural --max-candidates 12
python -m gpuboost model train-neural --json
```

The Phase 12.3 trainer uses a small PyTorch MLP classifier with ReLU layers,
optional dropout, AdamW, class weighting, and early stopping on validation macro
F1. It runs the Phase 12.2 baseline comparison first, then compares the neural
validation result against the best baseline. Hyperparameter search is modest and
validation-selected: test scores are reported only after the best validation
candidate is selected.

The `0.85` macro F1 target is aspirational, not guaranteed. A result is reported
honestly when the target is missed, and the report warns when validation is much
better than test performance. Strong evaluation requires both meaningful
validation performance and no serious validation/test gap.

Phase 12.3 still does not fine-tune an LLM, call external services, save a
production model artifact, or integrate predictions into the agent. Reports are
written as:

- `neural_training_report.json`
- `neural_training_report.md`

Future Phase 12 work may package and integrate a local model only if evaluation
is strong. Even then, deterministic GPUBoost logic remains authoritative: model
predictions may rank or score recommendations, but they must not directly apply
patches or override measured benchmark evidence.

Phase 12.4 adds an explicit local artifact format and provider. Artifacts are
saved only when requested:

```bash
python -m gpuboost model train-neural --save-artifact --json
python -m gpuboost model train-neural --save-artifact --artifact-dir data/gpuboost/generated/model_training/artifacts --artifact-name local_mlp
```

A saved artifact contains local generated files only:

- `model.pt`
- `feature_spec.json`
- `label_mapping.json`
- `training_config.json`
- `evaluation_report.json`
- `manifest.json`

The manifest references relative file names and records validation/test macro
F1, baseline macro F1, target status, feature names, labels, and warnings. It
does not store raw `DatasetRow` records, raw source, raw diffs, stdout, or
stderr. The generated artifact directory is ignored by git.

Artifacts can be validated and used for a local standalone prediction:

```bash
python -m gpuboost model validate-artifact data/gpuboost/generated/model_training/artifacts/local_mlp/manifest.json --json
python -m gpuboost model predict-artifact data/gpuboost/generated/model_training/artifacts/local_mlp/manifest.json --features-json '{"features.workload_family":"amp","features.batch_size":16}' --json
```

Standalone artifact predictions are local signals only: they must not apply
patches, call external APIs, or override deterministic GPUBoost checks.

Phase 12.5 wires the saved artifact provider into the agent report path:

```bash
python -m gpuboost agent optimize train.py --model-artifact data/gpuboost/generated/model_training/artifacts/local_mlp/manifest.json
python -m gpuboost agent optimize train.py --model --model-artifact data/gpuboost/generated/model_training/artifacts/local_mlp/manifest.json --json
```

`--model-artifact` automatically enables model inference. The agent extracts
only safe scalar features from the current workflow state and excludes raw
source, raw diffs, stdout, stderr, patch contents, final labels, comparison
verdicts, and target-derived before/after/delta fields. The JSON and human
reports include the model prediction, confidence, probabilities when available,
and `patch_application_allowed=false`.

The trained model remains advisory only. It does not apply patches, edit files,
override deterministic safety checks, or change trial behavior.

Phase 12.6 adds lifecycle commands for local generated artifacts. The full local
workflow is:

```bash
# 1. Train reports only
python -m gpuboost model train-neural --json

# 2. Train and explicitly save an artifact
python -m gpuboost model train-neural --save-artifact --json

# 3. List generated artifacts
python -m gpuboost model list-artifacts
python -m gpuboost model list-artifacts --json

# 4. Show a safe artifact summary
python -m gpuboost model show-artifact data/gpuboost/generated/model_training/artifacts/local_mlp/manifest.json

# 5. Validate and quality-check an artifact
python -m gpuboost model validate-artifact data/gpuboost/generated/model_training/artifacts/local_mlp/manifest.json
python -m gpuboost model check-artifact data/gpuboost/generated/model_training/artifacts/local_mlp/manifest.json --min-test-macro-f1 0.75 --require-beats-baseline

# 6. Predict directly from safe feature JSON
python -m gpuboost model predict-artifact data/gpuboost/generated/model_training/artifacts/local_mlp/manifest.json --features-json '{"features.workload_family":"amp","features.batch_size":16}' --json

# 7. Use the artifact as an advisory agent signal
python -m gpuboost agent optimize train.py --model-artifact data/gpuboost/generated/model_training/artifacts/local_mlp/manifest.json
```

`list-artifacts` recursively finds `manifest.json` files under
`data/gpuboost/generated/model_training/artifacts/` by default. `show-artifact`
prints only manifest-level metadata and validation status; it does not load or
print model weights. `check-artifact` is a read-only quality gate for local
automation. It can require a valid artifact, a minimum test macro F1, beating
the best baseline, and target-met status.

Artifact files remain local generated files and are ignored by Git. They do not
involve external APIs, scraping, or LLM fine-tuning. Model predictions are
advisory only, cannot apply patches, and cannot override deterministic
GPUBoost checks, trials, syntax checks, tests, or benchmark evidence.

The final Phase 12 release-readiness command set is:

```bash
python -m gpuboost model evaluate-baselines --json
python -m gpuboost model train-neural --max-epochs 50 --max-candidates 12 --target-macro-f1 0.85 --json
python -m gpuboost model train-neural --max-epochs 50 --max-candidates 12 --target-macro-f1 0.85 --save-artifact --json
python -m gpuboost model list-artifacts
python -m gpuboost model show-artifact <manifest>
python -m gpuboost model check-artifact <manifest> --min-test-macro-f1 0.75 --require-beats-baseline
python -m gpuboost model validate-artifact <manifest>
python -m gpuboost model predict-artifact <manifest> --features-json '{"features.workload_family":"amp","features.batch_size":16}' --json
python -m gpuboost agent optimize <script> --model-artifact <manifest> --json
python -m gpuboost model safety-check --json
```

Agent JSON redacts raw source, raw diffs, stdout, and stderr by default. Trained
artifact predictions report `patch_application_allowed=false`. Generated model
artifacts remain under ignored `data/gpuboost/generated/` paths and must not be
committed.
