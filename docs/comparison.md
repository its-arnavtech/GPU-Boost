# GPUBoost Comparison

`gpuboost compare` compares two GPUBoost benchmark JSON files and reports
whether selected benchmark metrics improved, regressed, stayed unchanged, or
could not be compared.

It only reads JSON files. It does not run workloads, execute benchmark
commands, apply patches, or modify source files.

## Save Benchmark Outputs

Create a baseline benchmark JSON file:

```bash
gpuboost benchmark --quick --json | tee baseline.json
```

After making and reviewing your own optimization changes, create an optimized
benchmark JSON file:

```bash
gpuboost benchmark --quick --json | tee optimized.json
```

On Windows PowerShell, avoid plain redirection for JSON files. Capture stdout
and write UTF-8 without BOM:

```powershell
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$json = gpuboost benchmark --quick --json
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "baseline.json"), $json + [Environment]::NewLine, $utf8NoBom)
```

## Compare Results

Print a concise human-readable report:

```bash
gpuboost compare baseline.json optimized.json
```

Print stable JSON:

```bash
gpuboost compare baseline.json optimized.json --json
```

The same command is available through the module entry point:

```bash
python -m gpuboost compare baseline.json optimized.json
```

## JSON Shape

JSON output uses this top-level shape:

```json
{
  "schema_version": "comparison.v1",
  "command": "compare",
  "comparison": {
    "...": "ComparisonResult"
  }
}
```

For input errors, `comparison` is `null` and `error` contains a clean message:

```json
{
  "schema_version": "comparison.v1",
  "command": "compare",
  "comparison": null,
  "error": "File not found: baseline.json"
}
```

`comparison` is a `ComparisonResult` with labels, sections, metric deltas,
warnings, status, and overall verdict.

## Verdicts

- `improved`: at least one compared metric improved and none regressed
- `regressed`: at least one compared metric regressed and none improved
- `mixed`: both improvements and regressions were found
- `unchanged`: all compared metrics stayed within tolerance
- `unknown`: no clear improvement, regression, or unchanged verdict is available

## Limitations

- Compares benchmark JSON files only.
- Does not run before/after workloads automatically yet.
- Does not execute benchmark commands yet.
- Does not apply patches.
- Missing metrics are skipped and reported as warnings.

## Future Benchmark-Command Design

A future phase may support explicit benchmark-command integration such as:

```bash
gpuboost agent optimize train.py --trial --benchmark-command "python train.py --epochs 1"
```

That future behavior must be explicit user opt-in. It must run in a controlled
workspace, capture stdout, stderr, and exit code, and avoid modifying original
files by default. It must remain separate from the current `--test` option,
which is for explicit trial-workspace validation commands.
