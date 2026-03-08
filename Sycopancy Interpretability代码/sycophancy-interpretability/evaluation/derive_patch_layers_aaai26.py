#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derive AAAI'26-style "critical layers" from logit-lens outputs.

This script is meant to operationalize the paper's mechanistic localization:
- Stage 1: late-layer output preference shift (decision score cross-over)
- Stage 2: deeper representational divergence (KL peak)

Inputs are the `.pkl` files produced by `LLM-sycophancy/experiments/mechanistic_analysis/run_syco_logit_cot.py`
with `--inference_mode logit_only --inference_layer all` for:
- plain questions
- opinion_only questions

Output:
Prints a comma-separated PATCH_LAYERS list suitable for the HF patching evaluators,
plus diagnostic shift/peak layers.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LETTERS = ["A", "B", "C", "D"]


def _ensure_synthetic_id(df: pd.DataFrame, col: str = "synthetic_id") -> pd.DataFrame:
    if col in df.columns:
        return df
    out = df.copy()
    out[col] = out.index
    return out


def align_plain_opinion(df_plain: pd.DataFrame, df_op: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_plain = _ensure_synthetic_id(df_plain)
    df_op = _ensure_synthetic_id(df_op)
    merged = pd.merge(df_plain, df_op, on="synthetic_id", suffixes=("_plain", "_op"))
    # Recover original column names per side.
    P = merged[[c for c in merged.columns if c.endswith("_plain")]].rename(columns=lambda x: x[:-6])
    O = merged[[c for c in merged.columns if c.endswith("_op")]].rename(columns=lambda x: x[:-3])
    return P, O


def _layer_key(layer_idx: int, keys: List[str]) -> str | None:
    # Support both "layer_12" and "Layer_12" (repo scripts vary).
    k1 = f"layer_{layer_idx}"
    k2 = f"Layer_{layer_idx}"
    if k1 in keys:
        return k1
    if k2 in keys:
        return k2
    return None


def logits_vec(layer_logits: Dict, layer_idx: int) -> np.ndarray | None:
    if not isinstance(layer_logits, dict):
        return None
    k = _layer_key(layer_idx, list(layer_logits.keys()))
    if k is None:
        return None
    d = layer_logits.get(k)
    if not isinstance(d, dict):
        return None
    return np.array([float(d.get(x, -1e9)) for x in LETTERS], dtype=np.float64)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def decision_score_mean(dfP: pd.DataFrame, dfO: pd.DataFrame, layer: int, opinion_col: str) -> float:
    scores = []
    for i in range(len(dfP)):
        lc = str(dfP.at[i, "answer"])
        lo = str(dfO.at[i, opinion_col])
        if lc not in LETTERS or lo not in LETTERS:
            continue
        vo = logits_vec(dfO.at[i, "layer_logits"], layer)  # opinion-condition logits
        if vo is None:
            continue
        ic, io = LETTERS.index(lc), LETTERS.index(lo)
        a, b = vo[ic], vo[io]
        scores.append((a - b) / (abs(a) + abs(b) + 1e-8))
    return float(np.mean(scores)) if scores else float("nan")


def kl_mean(dfP: pd.DataFrame, dfO: pd.DataFrame, layer: int) -> float:
    kls = []
    for i in range(len(dfP)):
        vp = logits_vec(dfP.at[i, "layer_logits"], layer)
        vo = logits_vec(dfO.at[i, "layer_logits"], layer)
        if vp is None or vo is None:
            continue
        pp = softmax(vp)
        po = softmax(vo)
        kls.append(float(np.sum(pp * (np.log(pp + 1e-12) - np.log(po + 1e-12)))))
    return float(np.mean(kls)) if kls else float("nan")


def infer_num_layers(layer_logits: Dict) -> int:
    if not isinstance(layer_logits, dict):
        raise ValueError("layer_logits is not a dict; cannot infer layers")
    idxs = []
    for k in layer_logits.keys():
        if k.lower().startswith("layer_"):
            try:
                idxs.append(int(k.split("_")[1]))
            except Exception:
                pass
    if not idxs:
        raise ValueError("No layer_* keys found in layer_logits")
    return max(idxs) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain_pkl", required=True)
    ap.add_argument("--opinion_pkl", required=True)
    ap.add_argument("--opinion_col", default="", help="Column name for opinion label (default: auto).")
    ap.add_argument("--budget_k", type=int, default=5, help="How many layers to output (default: 5 via +/- windows).")
    ap.add_argument("--window", type=int, default=1, help="Include +/- window around shift and KL peak.")
    args = ap.parse_args()

    df_plain = pd.read_pickle(args.plain_pkl)
    df_op = pd.read_pickle(args.opinion_pkl)
    P, O = align_plain_opinion(df_plain, df_op)

    # Choose opinion label column
    opinion_col = args.opinion_col.strip()
    if not opinion_col:
        for c in ["opinion", "human_opinion", "opinion_answer", "human_preference"]:
            if c in O.columns:
                opinion_col = c
                break
        if not opinion_col:
            opinion_col = "model_answer"  # fallback proxy

    # Truth basis: only questions where plain is correct.
    P = P.reset_index(drop=True)
    O = O.reset_index(drop=True)
    mask = (
        P.get("answer", "").astype(str).isin(LETTERS)
        & P.get("model_answer", "").astype(str).isin(LETTERS)
        & (P["model_answer"] == P["answer"])
    )
    P = P[mask].reset_index(drop=True)
    O = O.loc[P.index].reset_index(drop=True)

    if len(P) == 0:
        raise SystemExit("No truth-basis samples after filtering (plain correct subset is empty).")

    L = infer_num_layers(P.iloc[0]["layer_logits"])
    ds = [decision_score_mean(P, O, l, opinion_col) for l in range(L)]
    kl = [kl_mean(P, O, l) for l in range(L)]

    # Stage 1: decision score cross-over (+ -> -); fallback to argmin.
    shift = None
    for l in range(1, L):
        if math.isfinite(ds[l - 1]) and math.isfinite(ds[l]) and ds[l - 1] > 0 and ds[l] < 0:
            shift = l
            break
    if shift is None:
        shift = int(np.nanargmin(ds))

    # Stage 2: KL peak
    kl_peak = int(np.nanargmax(kl))

    cand = set()
    for center in [shift, kl_peak]:
        for x in range(center - args.window, center + args.window + 1):
            if 0 <= x < L:
                cand.add(x)

    layers = ",".join(str(x) for x in sorted(cand))
    print(layers)
    print(f"shift_layer={shift} kl_peak_layer={kl_peak} opinion_col={opinion_col} truth_n={len(P)} total_layers={L}")


if __name__ == "__main__":
    main()

