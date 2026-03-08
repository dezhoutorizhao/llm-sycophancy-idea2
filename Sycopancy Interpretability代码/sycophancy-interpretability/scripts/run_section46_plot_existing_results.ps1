param(
    [string]$RepoRoot = "",
    [string]$InRoot = "",
    [string]$OutDir = "",
    [string]$Models = "llama3-exp,llama3-exp_maxscore,llama3-exp_maxscore_finiteonly,Qwen,Mistral",
    [string]$PythonExe = "python",
    [string]$Cmap = "viridis",
    [string]$RobustPercentiles = "1,99",
    [int]$Dpi = 300
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($InRoot)) {
    $resultsDir = Get-ChildItem -LiteralPath $RepoRoot -Directory |
        Where-Object { $_.Name -like "results-pt*" } |
        Select-Object -First 1
    if ($null -eq $resultsDir) {
        throw "Cannot infer results dir under repo root: $RepoRoot"
    }
    $InRoot = $resultsDir.FullName
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $paperDir = Get-ChildItem -LiteralPath $RepoRoot -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "figures") } |
        Select-Object -First 1
    if ($null -eq $paperDir) {
        $OutDir = Join-Path $RepoRoot "figures"
    }
    else {
        $OutDir = Join-Path $paperDir.FullName "figures"
    }
}

$plotScript = Join-Path $RepoRoot "path_patching\plot_results_pt_heatmaps.py"
if (!(Test-Path -LiteralPath $plotScript)) {
    throw "Missing plot script: $plotScript"
}
if (!(Test-Path -LiteralPath $InRoot)) {
    throw "Missing input dir: $InRoot"
}

$modelList = $Models.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
if ($modelList.Count -eq 0) {
    throw "No models parsed from --Models"
}

foreach ($m in $modelList) {
    $pt = Join-Path (Join-Path $InRoot $m) "results.pt"
    if (!(Test-Path -LiteralPath $pt)) {
        throw "Missing results.pt for model '$m': $pt"
    }
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

Write-Host "[PLOT] in_root  = $InRoot"
Write-Host "[PLOT] out_dir  = $OutDir"
Write-Host "[PLOT] models   = $($modelList -join ',')"
Write-Host "[PLOT] style    = cmap=$Cmap, origin=upper, robust=$RobustPercentiles, dpi=$Dpi, highlight_k=0, no_html"

& $PythonExe -u $plotScript `
    --in_root $InRoot `
    --out_dir $OutDir `
    --models ($modelList -join ",") `
    --highlight_k 0 `
    --origin upper `
    --cmap $Cmap `
    --robust_percentiles $RobustPercentiles `
    --dpi $Dpi `
    --no_html

if ($LASTEXITCODE -ne 0) {
    throw "Plot command failed with exit code $LASTEXITCODE"
}

Write-Host "[DONE] paper-style heatmaps generated."
foreach ($m in $modelList) {
    Write-Host ("[FIG] " + (Join-Path $OutDir ("R_heatmap_{0}.png" -f $m)))
}
Write-Host ("[MANIFEST] " + (Join-Path $OutDir "R_heatmap_manifest.json"))
