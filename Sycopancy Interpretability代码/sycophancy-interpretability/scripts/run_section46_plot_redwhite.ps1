param(
    [string]$RepoRoot = "",
    [string]$InRoot = "",
    [string]$OutDir = "",
    [string]$Models = "llama3-exp,Qwen,Mistral",
    [string]$PythonExe = "python",
    [string]$Cmap = "Reds_r",
    [string]$RobustPercentiles = "1,99",
    [int]$Dpi = 300
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}
if ([string]::IsNullOrWhiteSpace($InRoot)) {
    $resultsDir = Get-ChildItem -LiteralPath $RepoRoot -Directory |
        Where-Object { $_.Name -like "results-pt*" -and $_.Name -notlike "*display*" } |
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
        $OutDir = Join-Path $RepoRoot "figures_redwhite"
    }
    else {
        $OutDir = Join-Path $paperDir.FullName "figures_redwhite"
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

$DisplayRoot = Join-Path $RepoRoot "results-pt-display-redwhite"
New-Item -ItemType Directory -Path $DisplayRoot -Force | Out-Null
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$modelCsv = ($modelList -join ",")

Write-Host "[PREP] in_root  = $InRoot"
Write-Host "[PREP] temp     = $DisplayRoot"
Write-Host "[PREP] models   = $modelCsv"
Write-Host "[PLOT] out_dir  = $OutDir"
Write-Host "[PLOT] style    = cmap=$Cmap, origin=upper, robust=$RobustPercentiles, dpi=$Dpi, highlight_k=0, no_html"

$prepPy = Join-Path $env:TEMP ("prep_redwhite_" + [Guid]::NewGuid().ToString("N") + ".py")
@'
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

in_root = Path(sys.argv[1])
out_root = Path(sys.argv[2])
models = [x.strip() for x in sys.argv[3].split(",") if x.strip()]
_lo_pct, hi_pct = [float(x.strip()) for x in sys.argv[4].split(",")]

for name in models:
    src = in_root / name
    dst = out_root / name
    dst.mkdir(parents=True, exist_ok=True)
    t = torch.load(src / "results.pt", map_location="cpu").float()
    fin = torch.isfinite(t)
    if int(fin.sum().item()) == 0:
        raise RuntimeError(f"{name}: no finite values in results.pt")
    fv = t[fin].cpu().numpy().astype(np.float64)
    hi = float(np.percentile(fv, hi_pct))
    td = t.clone()
    # Display-only replacement: non-finite values mapped to high percentile so
    # they appear near white in Reds_r instead of white holes (NaN mask).
    td[~fin] = hi
    torch.save(td, dst / "results.pt")
    meta_src = src / "results.meta.json"
    meta_dst = dst / "results.meta.json"
    if meta_src.exists() and meta_src.resolve() != meta_dst.resolve():
        shutil.copy2(meta_src, meta_dst)
    print(f"[PREP] {name}: shape={tuple(t.shape)}, nonfinite={int((~fin).sum())}, fill_value={hi:.6f}")
'@ | Set-Content -LiteralPath $prepPy -Encoding UTF8

& $PythonExe -u $prepPy "$InRoot" "$DisplayRoot" "$modelCsv" "$RobustPercentiles"
Remove-Item -LiteralPath $prepPy -Force -ErrorAction SilentlyContinue

if ($LASTEXITCODE -ne 0) {
    throw "Display tensor preparation failed with exit code $LASTEXITCODE"
}

& $PythonExe -u $plotScript `
    --in_root $DisplayRoot `
    --out_dir $OutDir `
    --models $modelCsv `
    --highlight_k 0 `
    --origin upper `
    --cmap $Cmap `
    --robust_percentiles $RobustPercentiles `
    --dpi $Dpi `
    --no_html

if ($LASTEXITCODE -ne 0) {
    throw "Plot command failed with exit code $LASTEXITCODE"
}

Write-Host "[DONE] red-white paper-style heatmaps generated."
foreach ($m in $modelList) {
    Write-Host ("[FIG] " + (Join-Path $OutDir ("R_heatmap_{0}.png" -f $m)))
}
Write-Host ("[MANIFEST] " + (Join-Path $OutDir "R_heatmap_manifest.json"))
