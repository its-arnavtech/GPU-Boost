# Security Policy

GPUBoost is a local-first development tool. It analyzes local files, creates
reviewable suggestions, and may generate local reports or artifacts. Treat any
security or data exposure report with care.

## Reporting Sensitive Issues

Do not paste secrets, tokens, credentials, private keys, raw private data,
private source code, raw diffs, model weights, or generated datasets in public
issues, pull requests, comments, logs, or screenshots. Public issues should
contain only redacted summaries.

For sensitive reports, use GitHub private vulnerability reporting if it is
enabled for this repository. If private reporting is not available, contact the
maintainers out-of-band before sharing details. Public reports should include
only a redacted, minimal summary.

## What To Redact

- API keys, tokens, credentials, private keys, and account identifiers
- Private paths, hostnames, usernames, and organization names
- Raw private source code and raw diffs
- Raw/generated datasets and private benchmark outputs
- Model weights, checkpoints, serialized models, and local databases
- Trial stdout/stderr or logs that include private content

## Generated Data And Model Artifacts

Generated artifacts and raw data are not intended to be committed. The repo
ignores generated outputs and common artifact formats such as:

- `data/gpuboost/generated/`
- `data/gpuboost/raw/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `*.joblib`, `*.safetensors`, `*.onnx`
- `*.db`, `*.sqlite`, `*.sqlite3`

Model artifacts are local/generated files. Model predictions are advisory-only
and must not apply patches, edit files, override deterministic checks, or
replace tests, trials, or benchmark evidence.

## Optimization Results

GPUBoost does not guarantee speedups or production readiness. Benchmark results
depend on hardware, drivers, CUDA/PyTorch versions, power mode, thermals,
background load, and workload shape. Synthetic demos are useful validation
coverage, not universal proof.
