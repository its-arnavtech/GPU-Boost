$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$OutputRoot = Join-Path $RepoRoot "data\gpuboost\generated\demo_real_world"
$PairsPath = Join-Path $OutputRoot "pairs.json"
$PythonExe = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Invoke-PythonProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $PythonExe
    $startInfo.WorkingDirectory = $RepoRoot
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    foreach ($argument in $Arguments) {
        [void] $startInfo.ArgumentList.Add($argument)
    }

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    [void] $process.Start()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    if ($process.ExitCode -ne 0) {
        throw "Python command failed with exit code $($process.ExitCode): $stderr"
    }

    return @{
        Stdout = $stdout
        Stderr = $stderr
    }
}

function ConvertFrom-RequiredJson {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Text,
        [Parameter(Mandatory = $true)]
        [string] $Description
    )

    try {
        return $Text | ConvertFrom-Json
    } catch {
        throw "$Description did not emit valid JSON. $($_.Exception.Message)"
    }
}

function Write-Utf8Json {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,
        [Parameter(Mandatory = $true)]
        [string] $Json
    )

    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    [System.IO.File]::WriteAllText($Path, $Json, $Utf8NoBom)
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$pairSpecResult = Invoke-PythonProcess -Arguments @(
    "-c",
    "import json; from gpuboost.demo.real_world import build_real_world_demo_pairs; print(json.dumps(build_real_world_demo_pairs(), sort_keys=True))"
)
$pairSpecs = ConvertFrom-RequiredJson -Text $pairSpecResult.Stdout -Description "Pair spec builder"

foreach ($pair in $pairSpecs) {
    $baselineScript = Join-Path $RepoRoot $pair.baseline_script
    $optimizedScript = Join-Path $RepoRoot $pair.optimized_script
    $baselineJson = Join-Path $RepoRoot $pair.baseline_json_path
    $optimizedJson = Join-Path $RepoRoot $pair.optimized_json_path

    Write-Host "Running $($pair.row_id) baseline..."
    $baseline = Invoke-PythonProcess -Arguments @(
        $baselineScript,
        "--quick",
        "--benchmark-json"
    )
    [void] (ConvertFrom-RequiredJson -Text $baseline.Stdout -Description "$($pair.row_id) baseline")
    Write-Utf8Json -Path $baselineJson -Json $baseline.Stdout

    Write-Host "Running $($pair.row_id) optimized..."
    $optimized = Invoke-PythonProcess -Arguments @(
        $optimizedScript,
        "--quick",
        "--benchmark-json"
    )
    [void] (ConvertFrom-RequiredJson -Text $optimized.Stdout -Description "$($pair.row_id) optimized")
    Write-Utf8Json -Path $optimizedJson -Json $optimized.Stdout
}

$pairsFileResult = Invoke-PythonProcess -Arguments @(
    "-c",
    "from gpuboost.demo.real_world import build_real_world_demo_pairs, write_real_world_pairs_file; print(write_real_world_pairs_file(build_real_world_demo_pairs()))"
)
[void] $pairsFileResult

Write-Host ""
Write-Host "Wrote demo benchmark JSON under data/gpuboost/generated/demo_real_world/"
Write-Host "Wrote pairs file: data/gpuboost/generated/demo_real_world/pairs.json"
Write-Host ""
Write-Host "Next commands:"
Write-Host "python -m gpuboost dataset collect-outcomes data/gpuboost/generated/demo_real_world/pairs.json --output-dir data/gpuboost/generated/demo_real_world/outcomes"

foreach ($pair in $pairSpecs) {
    Write-Host "python -m gpuboost compare $($pair.baseline_json_path) $($pair.optimized_json_path)"
}
