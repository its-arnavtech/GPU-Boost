# Phase 12 Release Readiness

Phase 12 completes GPUBoost's first local model training and artifact workflow.
It remains local-first, deterministic-first, and review-only.

## Completed Capabilities

- Safe training row loading from `DatasetRow` JSONL.
- Safe scalar feature extraction and deterministic encoding.
- Dependency-free baseline comparison for majority, random, nearest-centroid,
  and KNN baselines.
- Small PyTorch MLP training from scratch on safe structured features.
- Modest validation-selected hyperparameter search with honest validation/test
  reporting.
- Local model artifact save, load, validation, listing, showing, and quality
  checking.
- Direct local artifact prediction from safe feature JSON.
- Advisory-only agent integration through
  `agent optimize --model-artifact <manifest>`.

## Model Lifecycle Commands

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

For a manual end-to-end smoke run on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_phase_12_model_workflow.ps1
```

## Safety Guarantees

- Model predictions are advisory only.
- `patch_application_allowed=false` is reported for trained artifact providers.
- Models cannot apply patches or edit files.
- Deterministic GPUBoost recommendations, trial workspaces, syntax checks,
  explicit tests, and benchmark evidence remain authoritative.
- Raw source, raw diffs, stdout, and stderr are redacted from agent JSON by
  default.
- Training features exclude target-derived fields, labels, comparison verdicts,
  before/after metrics, deltas, and raw artifacts.
- Generated artifacts live under ignored `data/gpuboost/generated/` paths by
  default.
- Model weights and generated artifacts must not be committed.
- Phase 12 does not fine-tune an LLM, call external APIs, scrape websites, or
  download external datasets.

## Latest Local Verification Checklist

- `python -m ruff check .`
- `python -m pytest`
- `python -m gpuboost model safety-check --json`
- `git ls-files data/gpuboost/generated`
- `git ls-files "*.pt" "*.pth" "*.onnx" "*.safetensors" "*.pkl" "*.joblib"`
- `git status --short`

Expected verification state:

- Ruff passes.
- Pytest passes without CUDA.
- Safety check returns `ok` or an explainable non-blocking `warning`.
- No generated artifacts are tracked.
- No model checkpoint files are tracked.
- Model artifact usage in the agent remains advisory only.

## Known Limitations

- The current model is trained on controlled outcome data; real-world
  generalization needs more measured user-script outcomes.
- Artifact selection remains explicit and manual; there is no global default
  artifact or automatic model selection.
- Model quality depends on safe feature coverage and dataset coverage.
- The `0.85` macro F1 target is aspirational and not guaranteed.
- No automatic patch application is implemented.
- No LLM fine-tuning is performed.
- No external model APIs are called.
- Generated artifacts are local files and are not committed.

## Future Phases

- Broaden dataset coverage with more real, measured user-script outcomes.
- Add explicit artifact selection/configuration only after quality and safety
  gates are stronger.
- Continue keeping deterministic GPUBoost logic authoritative.
- Consider production packaging only for artifacts that beat baselines and show
  healthy validation/test behavior.
- Preserve review-only patch behavior unless a future phase introduces a
  separate, explicit, user-confirmed apply workflow.
