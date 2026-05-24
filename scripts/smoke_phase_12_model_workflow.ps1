[CmdletBinding()]
param(
    [string]$Dataset = "data/gpuboost/generated/training_dataset.jsonl",
    [string]$OutputDir = "data/gpuboost/generated/model_training",
    [string]$AgentScript = "examples/bad_train_sample.txt",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

function Write-Section {
    param([string]$Name)
    Write-Host ""
    Write-Host "== $Name =="
}

function Invoke-JsonCommand {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Section $Name
    $output = & $PythonExe -m gpuboost @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL: $Name"
        Write-Host $output
        exit $LASTEXITCODE
    }
    Write-Host "PASS: $Name"
    return ($output | Out-String | ConvertFrom-Json -ErrorAction Stop)
}

Write-Section "Phase 12 Model Workflow Smoke"
Write-Host "Dataset: $Dataset"
Write-Host "Output dir: $OutputDir"
Write-Host "Agent script: $AgentScript"
Write-Host "Safety: model predictions are advisory only; patch_application_allowed must be false."

$baseline = Invoke-JsonCommand "Evaluate baselines" @(
    "model", "evaluate-baselines",
    "--dataset", $Dataset,
    "--output-dir", $OutputDir,
    "--json"
)

$trainReports = Invoke-JsonCommand "Train neural reports only" @(
    "model", "train-neural",
    "--dataset", $Dataset,
    "--output-dir", $OutputDir,
    "--max-epochs", "20",
    "--max-candidates", "4",
    "--target-macro-f1", "0.85",
    "--json"
)

$trainArtifact = Invoke-JsonCommand "Train neural and save artifact" @(
    "model", "train-neural",
    "--dataset", $Dataset,
    "--output-dir", $OutputDir,
    "--max-epochs", "20",
    "--max-candidates", "4",
    "--target-macro-f1", "0.85",
    "--save-artifact",
    "--json"
)

$manifest = $trainArtifact.result.artifact_manifest_path
if ([string]::IsNullOrWhiteSpace($manifest)) {
    Write-Host "FAIL: artifact manifest path was not returned"
    exit 1
}
Write-Host "Manifest: $manifest"

$validation = Invoke-JsonCommand "Validate artifact" @(
    "model", "validate-artifact", $manifest, "--json"
)
if ($validation.result.status -ne "ok") {
    Write-Host "FAIL: artifact validation status was $($validation.result.status)"
    exit 1
}

$check = Invoke-JsonCommand "Check artifact quality gates" @(
    "model", "check-artifact", $manifest,
    "--min-test-macro-f1", "0.75",
    "--require-beats-baseline",
    "--json"
)
if ($check.result.status -ne "passed") {
    Write-Host "FAIL: artifact quality gate status was $($check.result.status)"
    exit 1
}

$featuresJson = '{"features.workload_family":"amp","features.batch_size":16,"features.cuda_available":true}'
$prediction = Invoke-JsonCommand "Predict artifact" @(
    "model", "predict-artifact", $manifest,
    "--features-json", $featuresJson,
    "--json"
)
if ($prediction.result.metadata.patch_application_allowed -ne $false) {
    Write-Host "FAIL: direct prediction did not report patch_application_allowed=false"
    exit 1
}

$agent = Invoke-JsonCommand "Agent advisory artifact use" @(
    "agent", "optimize", $AgentScript,
    "--model-artifact", $manifest,
    "--json"
)
if ($agent.artifacts.model.patch_application_allowed -ne $false) {
    Write-Host "FAIL: agent model artifact did not report patch_application_allowed=false"
    exit 1
}

Write-Section "PASS"
Write-Host "Phase 12 model workflow smoke completed."
Write-Host "Generated artifacts are under ignored data/gpuboost/generated/ and should not be committed."
