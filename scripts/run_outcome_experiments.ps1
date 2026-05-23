param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Run-Workload {
    param(
        [string]$Name,
        [string]$ScriptPath,
        [string]$OutputPath
    )

    Write-Host "Running $Name..."
    $outputDirectory = Split-Path -Parent $OutputPath
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

    $captured = & $PythonExe $ScriptPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Workload failed: $Name"
    }

    $jsonText = (($captured | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()
    if ([string]::IsNullOrWhiteSpace($jsonText)) {
        throw "Workload emitted empty JSON: $Name"
    }

    try {
        $null = $jsonText | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Workload did not emit valid JSON: $Name"
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
    [System.IO.File]::WriteAllText(
        $OutputPath,
        $jsonText + [Environment]::NewLine,
        $utf8NoBom
    )
    if (-not (Test-Path -LiteralPath $OutputPath)) {
        throw "Missing workload output: $OutputPath"
    }
    if ((Get-Item -LiteralPath $OutputPath).Length -le 0) {
        throw "Empty workload output: $OutputPath"
    }
}

$root = (Resolve-Path -LiteralPath ".").Path
$workloadDir = Join-Path $root "examples/outcome_collection/workloads"
$experimentDir = Join-Path $root "data/gpuboost/experiments"

Write-Host "GPUBoost controlled outcome experiments"
Write-Host "Python: $PythonExe"
Write-Host "Output: $experimentDir"

Run-Workload `
    -Name "dataloader baseline" `
    -ScriptPath (Join-Path $workloadDir "dataloader_baseline.py") `
    -OutputPath (Join-Path $experimentDir "dataloader_001/baseline.json")
Run-Workload `
    -Name "dataloader optimized" `
    -ScriptPath (Join-Path $workloadDir "dataloader_optimized.py") `
    -OutputPath (Join-Path $experimentDir "dataloader_001/optimized.json")

Run-Workload `
    -Name "AMP baseline" `
    -ScriptPath (Join-Path $workloadDir "amp_baseline.py") `
    -OutputPath (Join-Path $experimentDir "amp_001/baseline.json")
Run-Workload `
    -Name "AMP optimized" `
    -ScriptPath (Join-Path $workloadDir "amp_optimized.py") `
    -OutputPath (Join-Path $experimentDir "amp_001/optimized.json")

Run-Workload `
    -Name "small batch baseline" `
    -ScriptPath (Join-Path $workloadDir "batch_small_baseline.py") `
    -OutputPath (Join-Path $experimentDir "batch_001/baseline.json")
Run-Workload `
    -Name "small batch optimized" `
    -ScriptPath (Join-Path $workloadDir "batch_small_optimized.py") `
    -OutputPath (Join-Path $experimentDir "batch_001/optimized.json")

Write-Host "Outcome experiment JSON files written successfully."
Write-Host "Next:"
Write-Host "python -m gpuboost dataset collect-outcomes data/gpuboost/experiments/pairs.json --output-dir data/gpuboost/generated/outcomes"
