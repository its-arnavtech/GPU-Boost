# Pre-Phase-12 GitHub Issue Drafts

GitHub issue creation was skipped because `gh auth status` reported that the active GitHub token is invalid. These drafts are written in GitHub-ready form and should be deduplicated against open issues after GitHub CLI access is restored.

## [high][dataset][phase-12] Remove target-derived comparison fields from Phase 12 training features

### Summary
Controlled outcome and history-derived dataset rows can expose the target label back into training inputs. Labels are derived from comparison verdicts, while the same rows retain comparison verdicts, metric direction counts, and before/after delta fields in feature or metric payloads.

### Evidence
- `gpuboost/dataset/outcome_collection.py:55` through `gpuboost/dataset/outcome_collection.py:70` derives labels from `ComparisonResult.overall_verdict`.
- `gpuboost/dataset/outcome_collection.py:84` through `gpuboost/dataset/outcome_collection.py:106` stores comparison-derived features and metric deltas on the dataset row.
- `gpuboost/dataset/outcome_collection.py:310` through `gpuboost/dataset/outcome_collection.py:340` includes `overall_verdict`, metric direction counts, before values, after values, absolute deltas, and percent deltas.
- `gpuboost/dataset/history_converter.py:53` through `gpuboost/dataset/history_converter.py:62` adds comparison summaries to metrics, while `gpuboost/dataset/history_converter.py:98` through `gpuboost/dataset/history_converter.py:125` derives labels from the same comparison verdict.
- `data/gpuboost/manifests/training_readiness_report.json` reports `overall_verdict` present in 103 rows and also reports before/after/delta metric coverage.

### Why it matters
If Phase 12 trains on these fields as-is, the model can learn the label directly from post-comparison artifacts instead of learning from pre-outcome workload, hardware, code, and recommendation features. That would make training and validation metrics invalid.

### Suggested fix
Define an explicit Phase 12 training feature allowlist or denylist. Exclude target-derived fields such as `overall_verdict`, comparison verdict aliases, label values, metric direction counts, and before/after/delta/percent-delta metrics from model inputs unless a separate post-hoc analysis model is intentionally being trained. Move these fields to label provenance or non-training metadata. Add readiness validation that warns or blocks when known target-derived fields are present in trainable features.

### Acceptance criteria
- [ ] Phase 12 training inputs exclude direct labels and post-comparison target-derived fields.
- [ ] Controlled outcome rows no longer expose `overall_verdict` or direction counts as trainable features.
- [ ] History-derived rows no longer expose comparison verdict aliases as trainable metrics or metadata used for training.
- [ ] Readiness analysis reports a blocker or warning when target-derived fields are present in trainable inputs.
- [ ] Tests cover leakage detection for controlled outcome rows and history-derived rows.

### Phase 12 impact
Blocks Phase 12: yes, if the current assembled rows are used as training inputs without filtering.

## [medium][dataset][phase-12] Use grouped and stratified splits for controlled outcome rows

### Summary
Dataset split assignment is random at the row level. The controlled grid contains many closely related variants from the same workload families, so near-duplicate or highly correlated rows can land in train, validation, and test splits at the same time.

### Evidence
- `gpuboost/dataset/splitting.py:25` through `gpuboost/dataset/splitting.py:43` shuffles row indexes and assigns splits without grouping.
- `gpuboost/dataset/outcome_grid.py:316` through `gpuboost/dataset/outcome_grid.py:352` creates controlled-grid rows with shared `workload_family` metadata.
- `data/gpuboost/manifests/training_readiness_report.json` reports 100 `controlled_grid` rows and split counts of 83 train, 10 validation, and 11 test rows.

### Why it matters
Validation and test metrics can be overly optimistic if related controlled workload variants are split across train and holdout sets. This is especially risky before Phase 12 because the current dataset is dominated by controlled synthetic workload families.

### Suggested fix
Add grouped split support for Phase 12 datasets. Group by stable provenance fields such as workload family, workload template, script pair, row-id prefix, and hardware when available. Stratify labels within those constraints where possible. Add readiness reporting for group leakage across splits.

### Acceptance criteria
- [ ] Split assignment can keep related controlled-grid rows in a single split group.
- [ ] Label balance is reported after grouped splitting.
- [ ] Readiness reports whether any grouping key appears in more than one split.
- [ ] Tests cover grouped split behavior for controlled-grid rows.

### Phase 12 impact
Blocks Phase 12: needs verification. It blocks trustworthy validation/test reporting if Phase 12 relies on the current row-level splits.

## [medium][security] Stop unignoring `.env.staging`

### Summary
The repository ignores `.env.*` files but explicitly unignores `.env.staging`. That makes a real staging env file visible to Git and easier to commit by accident.

### Evidence
- `.gitignore:55` and `.gitignore:56` ignore `.env` and `.env.*`.
- `.gitignore:58` unignores `.env.staging`.
- `git ls-files ".env" ".env.*" ...` returned no tracked env or credential files during this audit.

### Why it matters
Staging env files often contain credentials, API keys, database URLs, or service endpoints. Even if no such file is tracked today, the ignore exception creates an avoidable future secret-leak path.

### Suggested fix
Remove the `.env.staging` exception. If staging defaults need to be shared, track a non-secret template such as `.env.staging.example` and keep real `.env.staging` files ignored.

### Acceptance criteria
- [ ] `.env.staging` remains ignored by Git.
- [ ] Any committed staging example file is clearly named as an example and contains no secrets.
- [ ] Documentation points developers to the example file rather than a real env file.

### Phase 12 impact
Blocks Phase 12: no, but should be fixed before broader data/model workflows increase the chance of local credentials being present.

## [medium][dataset][windows] Normalize PowerShell-generated JSON encoding in Phase 11 scripts and manifests

### Summary
The legacy controlled outcome runner writes benchmark JSON through PowerShell redirection, which can produce UTF-16 output in Windows PowerShell. A tracked third-party inventory JSON also contains a UTF-8 BOM that the readiness parser does not handle, causing the readiness report to say the inventory could not be parsed.

### Evidence
- `scripts/run_outcome_experiments.ps1:18` writes workload JSON with `>` redirection.
- `scripts/run_outcome_experiment_grid.ps1:50` through `scripts/run_outcome_experiment_grid.ps1:54` validates JSON before writing, and later writes with an explicit UTF-8 encoder.
- `gpuboost/dataset/assembly.py:469` through `gpuboost/dataset/assembly.py:478` reads `third_party_raw_inventory.json` as UTF-8 and reports a parse warning on `JSONDecodeError`.
- `data/gpuboost/manifests/training_readiness_report.json` reports `mlcommons_raw_inventory` as unparseable.

### Why it matters
Encoding drift makes the Phase 11 pipeline harder to reproduce on Windows and can hide valid provenance data from readiness reports. That is exactly the kind of brittle data-prep behavior Phase 12 should avoid.

### Suggested fix
Update the legacy PowerShell runner to capture stdout, validate JSON, and write with explicit UTF-8, matching the grid runner pattern. Either normalize tracked JSON manifests to UTF-8 without BOM or read legacy manifest files with `utf-8-sig` where appropriate.

### Acceptance criteria
- [ ] `scripts/run_outcome_experiments.ps1` writes collector-compatible JSON on Windows PowerShell and PowerShell 7.
- [ ] Readiness can parse `third_party_raw_inventory.json` without a warning.
- [ ] Tests or a lightweight script check cover BOM/encoding compatibility.

### Phase 12 impact
Blocks Phase 12: no for the already assembled dataset, but yes for reproducible regeneration if the legacy runner is used.

## [medium][security][dataset] Remove local absolute paths from tracked raw-inventory manifests

### Summary
The tracked raw-inventory manifest records local absolute workspace paths. Manifests are allowed to be tracked, but tracked provenance should be portable and should avoid local-only paths.

### Evidence
- `data/gpuboost/manifests/third_party_raw_inventory.json` contains `local_path` fields with drive-rooted Windows workspace paths.
- `data/gpuboost/manifests/third_party_raw_inventory.md` contains corresponding "Saved under" entries.
- A repo-wide absolute-path pattern scan found matches only in this raw-inventory manifest JSON, with matching Markdown entries in the same manifest pair.

### Why it matters
Absolute local paths reduce portability and can expose private workstation layout if the repo is shared. They also make manifest diffs less stable across machines.

### Suggested fix
Store repo-relative paths in tracked manifests. If absolute paths are useful locally, keep them only in ignored generated files or derive them at runtime.

### Acceptance criteria
- [ ] Tracked manifests use repo-relative paths for collected local data locations.
- [ ] No tracked files contain drive-rooted or home-directory local workspace paths unless explicitly documented as examples.
- [ ] Manifest generation preserves enough provenance without local absolute paths.

### Phase 12 impact
Blocks Phase 12: no, but this should be fixed before publishing or sharing Phase 12 artifacts.

## [medium][cli][security] Redact or opt in to raw diff and trial stdout/stderr in JSON artifacts

### Summary
`agent optimize --json` includes full result artifacts and selected artifacts, including raw diffs and trial results. Trial test commands capture stdout/stderr and store those strings in `TrialStep`, which can then be serialized in JSON output.

### Evidence
- `gpuboost/trial/test_command.py:51` through `gpuboost/trial/test_command.py:58` captures stdout and stderr from explicit test commands.
- `gpuboost/trial/test_command.py:153` through `gpuboost/trial/test_command.py:164` stores captured stdout/stderr on `TrialStep`.
- `gpuboost/schemas/trial.py:46` and `gpuboost/schemas/trial.py:47` include stdout/stderr fields in the schema.
- `gpuboost/agent/handlers.py:172` and `gpuboost/agent/handlers.py:173` store `trial_result.to_dict()` in agent metadata.
- `gpuboost/cli/main.py:1190` through `gpuboost/cli/main.py:1201` serializes `result.to_dict()` and selected artifacts into the stable JSON payload.

### Why it matters
The command is explicit opt-in, but test output can contain secrets, paths, logs, or raw data. JSON output is often redirected to files or downstream tooling, so raw artifacts should either be redacted/truncated by default or clearly require an opt-in flag.

### Suggested fix
Redact or truncate trial stdout/stderr by default in CLI JSON, with an explicit flag for raw output if needed. Consider emitting only diff metadata by default or documenting and gating raw diff output similarly. Ensure history and model-training paths continue to use safe summaries only.

### Acceptance criteria
- [ ] Default `agent optimize --json` output does not include raw trial stdout/stderr.
- [ ] Raw diff output behavior is documented and, if possible, gated behind explicit opt-in for machine output.
- [ ] Tests cover redaction and any raw-output opt-in flag.

### Phase 12 impact
Blocks Phase 12: no, but it protects users and downstream tools before Phase 12 data/model workflows expand.

## [low][docs][phase-12] Document the Phase 12 baseline-first training plan and controlled-data limitations

### Summary
The current docs explain that Phase 12 will train and integrate GPUBoost's model, and the controlled outcome docs mention synthetic workloads. They do not clearly say that Phase 12 should start with a simple baseline model and explicitly treat controlled synthetic rows as limited, non-user-script training data.

### Evidence
- `docs/model-interface.md` describes Phase 12 as future training and integration work but does not describe a baseline-first training plan.
- `README.md` describes readiness gating but does not spell out that `ready` is not the same as production-quality data.
- `examples/outcome_collection/README.md` correctly says the workloads are synthetic, but that limitation is not surfaced in the main Phase 12 readiness narrative.

### Why it matters
Clear docs reduce the chance that Phase 12 starts with an overly complex model or overclaims readiness from controlled synthetic rows. The first model should be treated as a baseline with explicit leakage checks, grouped validation, and limitations.

### Suggested fix
Add a short Phase 12 training note covering baseline-first modeling, controlled-row limitations, target leakage exclusions, grouped validation, and criteria for moving beyond the baseline.

### Acceptance criteria
- [ ] Main docs say Phase 12 starts with a baseline model before more complex training.
- [ ] Docs distinguish controlled synthetic workloads from real user-script outcomes.
- [ ] Docs explain that readiness means no hard data-pipeline blockers, not production data quality.

### Phase 12 impact
Blocks Phase 12: no, but should be fixed before reporting Phase 12 results.
