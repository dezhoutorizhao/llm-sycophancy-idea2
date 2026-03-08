#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot layer x head heatmaps from a `results.pt` head-score tensor.

This follows the repo's existing visualization style:
  - plotly.express.imshow + plotly.offline.plot (like `path_patching_hf.py`)
  - matplotlib for paper-ready static PNG/PDF

Notes on score direction:
  - In this repo, downstream pinpoint tuning selects *min-k* entries of `results.pt`.
    Therefore, *smaller* values correspond to more "sensitive" heads under the current
    scoring construction (e.g., neg-delta or rank-normalized neg-delta).
"""

from __future__ import annotations

import os
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# Windows + (torch/numpy/mkl) can trigger duplicate OpenMP runtime loading.
# This script is visualization-only; allow the process to continue.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _repo_root() -> Path:
    # .../path_patching/plot_results_pt_heatmaps.py -> repo root
    return Path(__file__).resolve().parents[1]


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")


def _load_results_tensor(results_pt: Path) -> torch.Tensor:
    t = torch.load(results_pt, map_location="cpu")
    if isinstance(t, dict):
        # Future-proofing: allow dict checkpoints, but keep it strict.
        for k in ("results", "R", "scores", "W"):
            if k in t and isinstance(t[k], torch.Tensor):
                t = t[k]
                break
        else:
            raise TypeError(f"{results_pt}: unsupported dict checkpoint keys={list(t.keys())}")
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{results_pt}: expected torch.Tensor, got {type(t)}")
    if t.ndim != 2:
        raise ValueError(f"{results_pt}: expected 2D tensor [L,H], got shape={tuple(t.shape)}")
    return t.detach().cpu().float()


def _finite_stats(t: torch.Tensor) -> Dict[str, float]:
    fin = torch.isfinite(t)
    fv = t[fin]
    out: Dict[str, float] = {
        "L": int(t.shape[0]),
        "H": int(t.shape[1]),
        "numel": int(t.numel()),
        "nan_count": int(torch.isnan(t).sum().item()),
        "inf_count": int(torch.isinf(t).sum().item()),
        "finite_count": int(fv.numel()),
    }
    if fv.numel() > 0:
        out.update(
            {
                "finite_min": float(fv.min().item()),
                "finite_max": float(fv.max().item()),
                "finite_mean": float(fv.mean().item()),
                "finite_median": float(fv.median().item()),
            }
        )
    return out


def _min_k_indices(t: torch.Tensor, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (layers, heads) for the min-k entries, excluding non-finite."""
    L, H = t.shape
    v = t.clone()
    v[~torch.isfinite(v)] = float("inf")
    k = int(min(max(k, 0), v.numel()))
    if k == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    idx = torch.topk(v.flatten(), k=k, largest=False).indices
    layers = (idx // H).cpu().numpy()
    heads = (idx % H).cpu().numpy()
    return layers, heads


def _robust_vmin_vmax(a: np.ndarray, lo_pct: float, hi_pct: float) -> Tuple[float, float]:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    # Avoid degenerate scales.
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1e-6
    return lo, hi


def plot_matplotlib_heatmap(
    t: torch.Tensor,
    out_path: Path,
    *,
    title: str,
    cmap: str,
    origin: str,
    highlight: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    robust_percentiles: Tuple[float, float] = (1.0, 99.0),
    dpi: int = 300,
) -> None:
    import matplotlib.pyplot as plt

    # Accept plotly-style names (e.g., "Viridis") while keeping matplotlib happy.
    cmap_mpl = cmap
    try:
        if cmap_mpl not in plt.colormaps():
            cmap_mpl = cmap_mpl.lower()
    except Exception:
        cmap_mpl = cmap_mpl.lower()

    a = t.cpu().numpy().astype(np.float64, copy=True)
    a[~np.isfinite(a)] = np.nan
    vmin, vmax = _robust_vmin_vmax(a, robust_percentiles[0], robust_percentiles[1])

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    im = ax.imshow(a, cmap=cmap_mpl, aspect="auto", origin=origin, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    if highlight is not None and highlight[0].size:
        layers, heads = highlight
        ax.scatter(heads, layers, s=10, facecolors="none", edgecolors="lime", linewidths=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Head score (lower = more selected by min-k)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_plotly_html(
    t: torch.Tensor,
    out_path: Path,
    *,
    title: str,
    cmap: str,
    origin: str,
    highlight: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    robust_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> None:
    # Keep plotly optional at runtime.
    import plotly.express as px
    import plotly.offline

    a = t.cpu().numpy().astype(np.float64, copy=True)
    a[~np.isfinite(a)] = np.nan
    vmin, vmax = _robust_vmin_vmax(a, robust_percentiles[0], robust_percentiles[1])

    # px.imshow defaults to origin="upper"; keep consistent with matplotlib via `origin`.
    fig = px.imshow(a, title=title, color_continuous_scale=cmap, zmin=vmin, zmax=vmax)
    fig.update_layout(xaxis_title="Head", yaxis_title="Layer")
    if origin == "upper":
        fig.update_yaxes(autorange="reversed")

    if highlight is not None and highlight[0].size:
        layers, heads = highlight
        fig.add_scatter(
            x=heads,
            y=layers,
            mode="markers",
            marker=dict(color="lime", size=6, opacity=0.7),
            name="min-k heads",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotly.offline.plot(fig, filename=str(out_path), auto_open=False)


def _iter_model_dirs(in_root: Path, models: Optional[List[str]]) -> Iterable[Tuple[str, Path]]:
    if models:
        for m in models:
            yield m, in_root / m
        return
    for d in sorted(in_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "results.pt").exists():
            yield d.name, d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_root",
        type=str,
        default=str(_repo_root() / ("results-pt" + "\u6587\u4ef6")),
        help="Directory containing <model_name>/results.pt (default: repo_root/results-pt<Chinese>).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(_repo_root() / ("\u8bba\u6587\u7ec8\u7a3f") / "figures"),
        help="Output directory for plots (default: repo_root/<paper_dir>/figures).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model subdir names. Empty = all subdirs under --in_root containing results.pt.",
    )
    parser.add_argument(
        "--highlight_k",
        type=int,
        default=64,
        help="Highlight min-k heads (same direction as training selection). Set 0 to disable.",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default="upper",
        choices=["upper", "lower"],
        help="Heatmap origin (upper keeps layer 0 at the top, matching px.imshow default).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap name (matplotlib + plotly). For paper, sequential maps (Viridis/Plasma) usually work best.",
    )
    parser.add_argument(
        "--robust_percentiles",
        type=str,
        default="1,99",
        help="Percentile range for vmin/vmax computed on finite entries (e.g., '1,99').",
    )
    parser.add_argument(
        "--no_html",
        action="store_true",
        help="Do not write interactive HTML plots.",
    )
    parser.add_argument(
        "--no_png",
        action="store_true",
        help="Do not write PNG plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG DPI (default: 300).",
    )
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None

    try:
        lo_pct, hi_pct = [float(x.strip()) for x in args.robust_percentiles.split(",")]
        robust = (lo_pct, hi_pct)
    except Exception as e:
        raise SystemExit(f"Invalid --robust_percentiles='{args.robust_percentiles}': {e}")

    if not in_root.exists():
        raise SystemExit(f"--in_root not found: {in_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    wrote: List[Dict[str, object]] = []
    for name, d in _iter_model_dirs(in_root, models):
        pt = d / "results.pt"
        t = _load_results_tensor(pt)
        stats = _finite_stats(t)
        layers, heads = _min_k_indices(t, args.highlight_k)
        highlight = (layers, heads) if args.highlight_k > 0 else None

        tag = _sanitize(name)
        title = f"R heatmap: {name}  (L={t.shape[0]}, H={t.shape[1]})"

        if not args.no_png:
            plot_matplotlib_heatmap(
                t,
                out_dir / f"R_heatmap_{tag}.png",
                title=title,
                cmap=args.cmap,
                origin=args.origin,
                highlight=highlight,
                robust_percentiles=robust,
                dpi=args.dpi,
            )

        if not args.no_html:
            plot_plotly_html(
                t,
                out_dir / f"R_heatmap_{tag}.html",
                title=title,
                cmap=args.cmap,
                origin=args.origin,
                highlight=highlight,
                robust_percentiles=robust,
            )

        # Copy meta if present to keep the figure auditable.
        meta_path = d / "results.meta.json"
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None

        wrote.append(
            {
                "name": name,
                "dir": str(d),
                "results_pt": str(pt),
                "stats": stats,
                "highlight_k": int(args.highlight_k),
                "meta": meta,
            }
        )

    (out_dir / "R_heatmap_manifest.json").write_text(
        json.dumps(wrote, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] wrote {len(wrote)} heatmaps to: {out_dir}")


if __name__ == "__main__":
    main()
