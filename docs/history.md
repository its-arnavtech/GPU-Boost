# Local History

GPUBoost history is local-only run memory for deterministic agent runs. It is
stored in SQLite at:

```text
~/.gpuboost/gpuboost.db
```

History is saved only when requested:

```bash
gpuboost agent optimize train.py --save-history
```

For development or tests, use a temporary database:

```bash
gpuboost agent optimize train.py --save-history --history-db-path ./pytest_tmp/history/gpuboost.db
gpuboost history list --db-path ./pytest_tmp/history/gpuboost.db
```

## Privacy And Safety

History records store safe summaries only:

- run ID, timestamps, command, status, and schema version
- goal kind and description
- script path and script SHA256
- GPU name and CUDA availability when available
- action statuses, warnings, errors, counts, and summary metrics
- trial and comparison statuses when available

History records do not store raw source code by default. They also do not store
raw diffs, trial stdout, or trial stderr by default. GPUBoost does not upload
history records or perform cloud sync.

## Commands

List local history:

```bash
gpuboost history list
gpuboost history list --json
gpuboost history list --db-path ./pytest_tmp/history/gpuboost.db
```

Show one run:

```bash
gpuboost history show <run_id>
gpuboost history show <run_id> --json
gpuboost history show <run_id> --db-path ./pytest_tmp/history/gpuboost.db
```

Compare two history records:

```bash
gpuboost history compare <left_run_id> <right_run_id>
gpuboost history compare <left_run_id> <right_run_id> --json
gpuboost history compare <left_run_id> <right_run_id> --db-path ./pytest_tmp/history/gpuboost.db
```

History comparison checks safe tracked fields only, such as status, command,
goal kind, script hash, GPU identity, CUDA availability, trial status,
comparison verdict, and presence flags for diff/trial/comparison data. It does
not compare raw source, raw diffs, stdout, or stderr.

To manually delete local history, remove the database file:

```bash
rm ~/.gpuboost/gpuboost.db
```

On Windows PowerShell:

```powershell
Remove-Item $HOME\.gpuboost\gpuboost.db
```

Future Phase 11 data collection/validation and Phase 12 model training may
build on this local history layer. Those features are not implemented yet.
