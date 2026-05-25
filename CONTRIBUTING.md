# Contributing To GPUBoost

Thanks for helping keep GPUBoost maintainable. Phase 15 is final polish, so
changes should stay small, reviewable, and aligned with the existing safety
model.

## Setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m gpuboost doctor
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\activate
```

CUDA is useful for real benchmark exploration, but it is not required for the
test suite.

## Running Tests

Run the standard checks before opening a pull request:

```bash
python -m ruff check .
python -m pytest
```

Avoid heavy workflows in routine PR validation. Do not train models unless the
change explicitly requires model workflow work.

## Coding Style

- Keep changes narrowly scoped to the issue or feature.
- Prefer deterministic logic and existing project patterns.
- Keep CLI JSON stable and versioned when changing command output.
- Add focused tests for behavior, safety boundaries, and user-facing docs.
- Use clear names and avoid unrelated refactors.

## Generated Data And Artifact Policy

Do not commit generated/raw data, model artifacts, checkpoints, weights,
serialized models, local databases, cache directories, local reports, secrets,
or private data. These are ignored by `.gitignore`, including:

- `data/gpuboost/generated/`
- `data/gpuboost/raw/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `*.joblib`, `*.safetensors`, `*.onnx`
- `*.db`, `*.sqlite`, `*.sqlite3`
- `.env`, tokens, private keys, and credential files

If a test needs data, use small synthetic fixtures that are safe to commit.

## Safety Model

GPUBoost is local-first and review-oriented:

- Patch suggestions are review-only.
- There is no automatic patch application to original source files.
- Trial mode modifies only temporary copies.
- Static analysis must not import or execute user scripts.
- Model predictions remain advisory-only.
- Deterministic checks, tests, trials, and measured benchmark evidence remain
  authoritative.
- User code, raw diffs, trial stdout/stderr, secrets, and raw private data
  should not be stored or exposed by default.

## Pull Request Expectations

- Explain the problem and the approach.
- Include tests for changed behavior.
- Run `python -m ruff check .` and `python -m pytest`.
- Confirm no generated/raw data, model artifacts, or secrets are committed.
- Update docs when CLI behavior, user-facing behavior, workflow steps, or
  safety boundaries change.
- Keep model-related changes advisory-only and subordinate to deterministic
  checks.

