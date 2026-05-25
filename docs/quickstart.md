# Quickstart

This is the shortest path for trying GPUBoost from a fresh checkout. These
commands avoid heavy workflows and do not train models.

## Install

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment first if desired:

```powershell
.\.venv\Scripts\activate
```

## Check The CLI

```bash
python -m gpuboost --help
```

## Run Agent Optimize On The Demo Sample

```bash
python -m gpuboost agent optimize examples/bad_train_sample.txt --json
```

This runs the deterministic optimize workflow and emits JSON. Patch suggestions
are review-only; GPUBoost does not apply patches automatically.

## Run The Trial Example

```bash
python -m gpuboost agent optimize examples/bad_train_sample.txt --trial --json
```

Trial mode copies the sample into a temporary workspace, applies generated
suggestions only to the copy, and syntax-checks the copied file. The original
file is not modified.

## Run The Safety Check

```bash
python -m gpuboost model safety-check --json
```

The safety check verifies key guardrails such as ignored generated artifacts,
ignored raw data, ignored model-weight extensions, and disabled model patch
application.

## Optional Next Steps

- Model artifact workflow: [Model Training](model-training.md)
- Real-world demo workflow: [Real-World Validation](real-world-validation.md)
- Agent JSON and trial details: [Agent CLI](agent-cli.md)

