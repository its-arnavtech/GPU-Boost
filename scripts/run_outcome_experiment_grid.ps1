[CmdletBinding()]
param(
    [int]$MaxPairs = 0,
    [switch]$Smoke,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

function Join-RepoPath {
    param(
        [string]$BasePath,
        [string[]]$Segments
    )

    $resolvedPath = $BasePath
    foreach ($segment in $Segments) {
        if ([string]::IsNullOrWhiteSpace($segment)) {
            continue
        }
        $resolvedPath = Join-Path -Path $resolvedPath -ChildPath $segment
    }
    return $resolvedPath
}

function Resolve-GridPath {
    param([string]$PathValue)

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-RepoPath -BasePath $root -Segments ($PathValue -split "[\\/]+"))
}

function ConvertTo-StringArray {
    param($Value)

    $items = @()
    if ($null -eq $Value) {
        return $items
    }
    foreach ($item in @($Value)) {
        $items += [string]$item
    }
    return $items
}

function Invoke-WorkloadJson {
    param(
        [string]$ScriptPath,
        $Arguments,
        [string]$OutputPath
    )

    if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
        throw "Workload script not found: $ScriptPath"
    }

    $argArray = ConvertTo-StringArray $Arguments
    $captured = & $PythonExe $ScriptPath @argArray 2>&1
    $exitCode = $LASTEXITCODE
    $jsonText = (($captured | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()

    if ($exitCode -ne 0) {
        throw "Workload failed with exit code $exitCode`: $ScriptPath"
    }
    if ([string]::IsNullOrWhiteSpace($jsonText)) {
        throw "Workload emitted empty JSON: $ScriptPath"
    }

    $outputDirectory = Split-Path -Parent $OutputPath
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    Write-Utf8JsonFile -Name $ScriptPath -OutputPath $OutputPath -JsonText $jsonText
}

function Write-Utf8JsonFile {
    param(
        [string]$Name,
        [string]$OutputPath,
        [string]$JsonText
    )

    try {
        $null = $JsonText | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Workload did not emit valid JSON: $Name"
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding -ArgumentList $false
    [System.IO.File]::WriteAllText(
        $OutputPath,
        $JsonText + [Environment]::NewLine,
        $utf8NoBom
    )
}

$root = (Resolve-Path -LiteralPath ".").Path
$manifestPath = Join-RepoPath -BasePath $root -Segments @("data", "gpuboost", "experiments", "grid_runner_manifest.json")
$pairsPath = Join-RepoPath -BasePath $root -Segments @("data", "gpuboost", "experiments", "grid_pairs.json")
$previousSmoke = $env:GPUBOOST_OUTCOME_SMOKE

try {
    if ($Smoke) {
        $env:GPUBOOST_OUTCOME_SMOKE = "1"
        if ($MaxPairs -le 0) {
            $MaxPairs = 2
        }
    }

    $shouldGenerate = -not (Test-Path -LiteralPath $manifestPath) -or
        -not (Test-Path -LiteralPath $pairsPath) -or
        $MaxPairs -gt 0

    if ($shouldGenerate) {
        $generatorArgs = @("-m", "gpuboost.dataset.outcome_grid", "--write-default")
        if ($MaxPairs -gt 0) {
            $generatorArgs += @("--max-pairs", [string]$MaxPairs)
        }

        Write-Host "Generating controlled outcome grid manifest..."
        $generationOutput = & $PythonExe @generatorArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Grid manifest generation failed."
        }
        if ($generationOutput) {
            Write-Host (($generationOutput | ForEach-Object { [string]$_ }) -join [Environment]::NewLine)
        }
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 |
        ConvertFrom-Json -ErrorAction Stop
    $pairs = @($manifest.pairs)
    $total = $pairs.Count

    for ($index = 0; $index -lt $total; $index++) {
        $pair = $pairs[$index]
        $ordinal = $index + 1
        $rowId = [string]$pair.row_id

        Write-Host "[$ordinal/$total] $rowId baseline"
        Invoke-WorkloadJson `
            -ScriptPath (Resolve-GridPath ([string]$pair.baseline_script)) `
            -Arguments $pair.baseline_args `
            -OutputPath (Resolve-GridPath ([string]$pair.baseline_json_path))

        Write-Host "[$ordinal/$total] $rowId optimized"
        Invoke-WorkloadJson `
            -ScriptPath (Resolve-GridPath ([string]$pair.optimized_script)) `
            -Arguments $pair.optimized_args `
            -OutputPath (Resolve-GridPath ([string]$pair.optimized_json_path))
    }

    Write-Host "Outcome experiment grid JSON files written successfully."
    Write-Host "Pairs file: data/gpuboost/experiments/grid_pairs.json"
    Write-Host "Next:"
    Write-Host "python -m gpuboost dataset collect-outcomes data/gpuboost/experiments/grid_pairs.json --output-dir data/gpuboost/generated/outcomes_grid"
}
finally {
    if ($null -eq $previousSmoke) {
        Remove-Item Env:\GPUBOOST_OUTCOME_SMOKE -ErrorAction SilentlyContinue
    }
    else {
        $env:GPUBOOST_OUTCOME_SMOKE = $previousSmoke
    }
}
