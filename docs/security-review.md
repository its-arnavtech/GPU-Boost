# Security Review

GPUBoost `0.2.0` is local-first release tooling. The normal workflow does not
require secrets, external APIs, cloud credentials, scraping, or dataset
downloads.

## Ignored Sensitive Files

The `.gitignore` excludes common local and sensitive files, including:

- `.env`, `.env.*`, tokens, secret files, and `secrets/`
- credential files such as `credentials.json`, service-account JSON files, and
  `token.json`
- private key and certificate material such as `*.pem`, `*.key`, `*.p12`,
  `*.pfx`, `*.crt`, and `*.cer`
- SSH private key names such as `id_rsa`, `id_dsa`, `id_ecdsa`, and
  `id_ed25519`
- caches, local virtual environments, logs, reports, temporary directories, and
  local databases

## Generated, Raw, And Model Artifacts

Generated artifacts are ignored under `data/gpuboost/generated/`. Raw intake
data is ignored under `data/gpuboost/raw/`; raw intake data is ignored for the
release checkpoint. Model artifacts, checkpoints, and
weights are ignored through paths and extensions including `checkpoints/`,
`models/`, `weights/`, `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `*.joblib`,
`*.safetensors`, and `*.onnx`.

The release should not track generated datasets, raw data, model weights, local
history databases, caches, or local reports.

## Secrets And External APIs

No secrets are required for normal GPUBoost use. Normal setup, docs validation,
linting, and tests do not require external API credentials, cloud credentials,
external LLM APIs, scraping, or network dataset downloads.

## Advisory Model Behavior

Model predictions are advisory-only. Advisory model output cannot apply
patches, edit files, approve changes, override deterministic GPUBoost checks,
replace tests, or replace measured benchmark evidence.

Saved model artifacts are local/generated files and must be explicitly selected
for advisory prediction. Trained artifact metadata is expected to expose
`patch_application_allowed=false`.

## Human-Approved Agentic Apply Boundary

GPUBoost supports human-approved agentic optimization for deterministic patch
plans. It does not support unapproved patch application. `agent optimize
--prepare` is non-mutating and writes an ignored local run record. `agent
approve` requires an exact confirmation phrase and binds approval to the run
ID, plan ID, immutable plan digest, approved action IDs, approver timestamp,
and original target file hash. `agent apply` repeats the hash and digest checks,
creates a backup, validates the modified source, and automatically restores the
backup if validation or acceptance policy fails.

User-provided validation, test, and benchmark commands are explicit opt-in and
execute local code. Benchmark threshold policies require user-provided JSON
metrics such as `speedup_percent` or `regression_percent`.

## Raw Output Redaction Policy

CLI JSON redacts raw diffs and trial stdout/stderr by default for agent
artifacts. Raw source, raw diffs, stdout, and stderr should not be stored in
local history or model feature datasets by default. Users can opt into raw
agent artifacts with explicit CLI flags where supported, but release docs and
tests treat redacted output as the safe default.

## Known Remaining Risks

- User-provided test commands can execute arbitrary local code when explicitly
  passed with trial mode or the approved apply workflow.
- Benchmark results and model signals can be misleading if interpreted without
  hardware, driver, thermal, and workload context.
- Redaction is best effort for summaries and structured artifacts; user-supplied
  paths and command strings may still appear where needed for local execution.
- Local model artifacts are experimental advisory aids and should not be
  treated as production security boundaries.
- Contributors must continue to review `git status` and tracked files before a
  release to avoid committing generated artifacts, raw data, local databases,
  secrets, or model weights.
