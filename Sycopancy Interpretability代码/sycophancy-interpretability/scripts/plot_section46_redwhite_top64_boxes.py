#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

TITLE_MAP = {
    "llama3-exp": "Llama 3.1-8B-Instruct",
    "Qwen": "Qwen2.5-7B-Instruct",
    "Mistral": "Mistral-7B-Instruct-v0.3",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in_root",
        type=str,
        default="results-pt文件",
        help="Directory containing <model>/results.pt",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="论文终稿/figures_redwhite",
        help="Output directory for figures",
    )
    p.add_argument(
        "--models",
        type=str,
        default="llama3-exp,Qwen,Mistral",
        help="Comma-separated model subdirs",
    )
    p.add_argument("--highlight_k", type=int, default=64)
    p.add_argument("--cmap", type=str, default="Reds")
    p.add_argument("--robust_percentiles", type=str, default="1,99")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def robust_vrange(arr: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1e-6
    return lo, hi


def top_k_pairs_on_importance(score: torch.Tensor, k: int) -> list[tuple[int, int]]:
    # Training score in this repo is "lower is more sensitive".
    # For presentation, convert to non-negative importance:
    #   importance = max_finite(score) - score
    v = score.clone()
    finite = torch.isfinite(v)
    if int(finite.sum().item()) == 0:
        return []
    smax = v[finite].max()
    imp = torch.full_like(v, float("-inf"))
    imp[finite] = smax - v[finite]
    k = int(min(max(k, 0), v.numel()))
    if k == 0:
        return []
    idx = torch.topk(imp.flatten(), k=k, largest=True).indices
    H = int(score.shape[1])
    return [(int(i // H), int(i % H)) for i in idx]


def main() -> None:
    args = parse_args()
    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lo_pct, hi_pct = [float(x.strip()) for x in args.robust_percentiles.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    manifest: list[dict] = []

    for name in models:
        pt = in_root / name / "results.pt"
        if not pt.exists():
            raise FileNotFoundError(f"Missing results.pt: {pt}")

        t = torch.load(pt, map_location="cpu").float()
        fin = torch.isfinite(t)
        if int(fin.sum().item()) == 0:
            raise RuntimeError(f"{name}: no finite value in tensor")

        arr_score = t.numpy().astype(np.float64, copy=True)
        finite_arr = np.isfinite(arr_score)
        smax = float(np.max(arr_score[finite_arr]))
        imp = np.full_like(arr_score, np.nan, dtype=np.float64)
        imp[finite_arr] = smax - arr_score[finite_arr]
        arr = imp
        finite_vals = arr[np.isfinite(arr)]
        # Display-only fill for non-finite cells, avoids white holes.
        fill_lo = float(np.percentile(finite_vals, lo_pct))
        arr_disp = arr.copy()
        arr_disp[~np.isfinite(arr_disp)] = fill_lo

        vmin, vmax = robust_vrange(arr_disp, lo_pct, hi_pct)
        topk = top_k_pairs_on_importance(t, args.highlight_k)

        fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
        im = ax.imshow(
            arr_disp,
            cmap=args.cmap,
            origin="upper",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        title = TITLE_MAP.get(name, name)
        ax.set_title(title, fontsize=12, pad=6)

        # Box-highlight top-k heads.
        for l, h in topk:
            ax.add_patch(
                Rectangle(
                    (h - 0.5, l - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.75,
                )
            )

        ax.text(
            0.015,
            0.985,
            f"Top-{args.highlight_k} by score",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative Head Score", fontsize=9)

        out_png = out_dir / f"R_heatmap_{name}.png"
        fig.savefig(out_png, dpi=args.dpi)
        plt.close(fig)

        manifest.append(
            {
                "name": name,
                "results_pt": str(pt),
                "output_png": str(out_png),
                "shape": [int(t.shape[0]), int(t.shape[1])],
                "nonfinite_count": int((~fin).sum().item()),
                "highlight_k": int(args.highlight_k),
                "note": "Relative score = max(score_finite) - score; Top-k uses highest relative score.",
            }
        )

    (out_dir / "R_heatmap_manifest_boxes.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] wrote {len(manifest)} boxed heatmaps to: {out_dir}")


if __name__ == "__main__":
    main()
