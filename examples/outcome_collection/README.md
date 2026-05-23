# Controlled Outcome Collection Templates

This folder is for controlled local GPUBoost experiments that produce real
optimization outcome labels from measured benchmark results.

The `workloads/` scripts generate real local measured benchmark JSON from tiny
synthetic workloads. They do not download datasets, call external APIs, scrape
websites, write output files, or train the final GPUBoost model.

Use real measured benchmark JSON from controlled experiments. Do not use
AI-generated benchmark data as production training data. Labels are derived from
measured comparison results, not guessed manually.

The normal workflow is:

1. Run the controlled workload scripts with the PowerShell runner.
2. Collect labeled outcome rows from `data/gpuboost/experiments/pairs.json`.
3. Rerun dataset assembly/readiness.

Run controlled local experiments:

```powershell
.\scripts\run_outcome_experiments.ps1
```

Collect the measured outcomes:

```powershell
python -m gpuboost dataset collect-outcomes data/gpuboost/experiments/pairs.json --output-dir data/gpuboost/generated/outcomes
```

Rerun readiness:

```powershell
python -c "from gpuboost.dataset.assembly import assemble_training_dataset; print(assemble_training_dataset())"
```

Current Phase 11.9 tooling consumes existing benchmark JSON files only. The
PowerShell runner is an explicit local helper for these controlled scripts; the
dataset collector itself does not execute benchmark commands.

## Controlled Grid Collection

Phase 11.10 adds a controlled grid with 120+ local measured baseline/optimized
JSON pairs. The grid uses configurable lightweight workloads and writes
benchmark JSON locally; it does not train a model, download data, scrape
websites, or call external APIs.

Generate and run a small grid sample:

```powershell
.\scripts\run_outcome_experiment_grid.ps1 -MaxPairs 20
```

Run a medium validation collection:

```powershell
.\scripts\run_outcome_experiment_grid.ps1 -MaxPairs 50
```

Run a readiness-scale collection:

```powershell
.\scripts\run_outcome_experiment_grid.ps1 -MaxPairs 100
```

Run the full grid:

```powershell
.\scripts\run_outcome_experiment_grid.ps1 -MaxPairs 120
```

Collect labels from the measured grid JSON:

```powershell
python -m gpuboost dataset collect-outcomes data/gpuboost/experiments/grid_pairs.json --output-dir data/gpuboost/generated/outcomes_grid
```

Assemble the original controlled outcomes plus the grid outcomes:

```powershell
python -c "from gpuboost.dataset.assembly import assemble_training_dataset; print(assemble_training_dataset(outcome_dataset_paths=['data/gpuboost/generated/outcomes/outcome_dataset.jsonl','data/gpuboost/generated/outcomes_grid/outcome_dataset.jsonl']))"
```

The resulting labels are comparison-derived from real measured local JSON.
More rows improve training readiness, but Phase 12 training still requires
enough labeled rows and enough label diversity. Neutral-oriented pairs are
included in the grid, but their labels still come from measured comparison
results rather than guessed labels. Phase 12 should start only after readiness
no longer reports hard blockers.

The `pairs.example.json` file remains a template with placeholder paths. Replace
those paths with real local benchmark JSON files before collecting training rows.
