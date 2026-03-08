#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-transition pilot:
Estimate per-layer R_l and test whether max_l R_l predicts wrong-follow.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_in_path(path_arg: str, repo_root: Path) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root / p).resolve()


def resolve_out_path(path_arg: str, repo_root: Path) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.parent.exists():
        return cwd_candidate
    return (repo_root / p).resolve()


def extract_question_from_user_content(user_content: str) -> str:
    # Dataset format: "question\n\nbias block"
    if "\n\n" in user_content:
        return user_content.split("\n\n", 1)[0].strip()
    return user_content.strip()


def build_letter_token_ids(tok) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for letter in "ABCDE":
        variants = [
            letter,
            f" {letter}",
            f"{letter}.",
            f" {letter}.",
            f"({letter})",
            f" ({letter})",
            f"{letter}\n",
            f" {letter}\n",
        ]
        ids = set()
        for v in variants:
            token_ids = tok.encode(v, add_special_tokens=False)
            if token_ids:
                ids.add(token_ids[0])
        mapping[letter] = sorted(ids)
    return mapping


def mean_token_vector(weight: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
    vecs = [weight[i] for i in token_ids if 0 <= i < weight.shape[0]]
    if not vecs:
        raise ValueError("No valid token ids for letter vector.")
    return torch.stack(vecs, dim=0).mean(dim=0)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = average_ranks(scores)
    sum_pos_ranks = ranks[pos].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    # Maximize Youden's J = TPR - FPR.
    uniq = np.unique(scores)
    best_tau = float(uniq[0]) if len(uniq) else 0.0
    best_j = -1.0
    best_tpr = 0.0
    best_fpr = 0.0
    pos = labels == 1
    neg = labels == 0
    n_pos = max(1, int(pos.sum()))
    n_neg = max(1, int(neg.sum()))
    for tau in uniq:
        pred = scores >= tau
        tp = int(np.logical_and(pred, pos).sum())
        fp = int(np.logical_and(pred, neg).sum())
        tpr = tp / n_pos
        fpr = fp / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_tau = float(tau)
            best_tpr = float(tpr)
            best_fpr = float(fpr)
    return best_tau, best_tpr, best_fpr


@torch.no_grad()
def get_last_hidden_by_layer(model, tok, messages: List[Dict[str, str]]) -> torch.Tensor:
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt_text, return_tensors="pt")
    out = model(**inp, output_hidden_states=True, return_dict=True)
    # hidden_states: tuple(layer0_embedding, layer1, ..., layerN)
    # Use actual transformer layers only (exclude embedding at index 0).
    hs = out.hidden_states[1:]
    last = torch.stack([h[0, -1, :].detach().cpu() for h in hs], dim=0)
    return last  # [num_layers, hidden_dim]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Local model directory.")
    ap.add_argument(
        "--pred_jsonl",
        default="isb_experiments/isb_predictions_qwen05b_logits_50_allwrong.jsonl",
        help="Prediction jsonl path (relative to repo root or absolute).",
    )
    ap.add_argument(
        "--output_json",
        default="isb_experiments/r_phase_pilot_summary.json",
        help="Output summary json path.",
    )
    ap.add_argument(
        "--output_md",
        default="isb_experiments/r_phase_pilot_report.md",
        help="Output markdown path.",
    )
    ap.add_argument("--max_rows", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--print_every", type=int, default=20)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pred_path = resolve_in_path(args.pred_jsonl, repo_root)
    out_json_path = resolve_out_path(args.output_json, repo_root)
    out_md_path = resolve_out_path(args.output_md, repo_root)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [r for r in read_jsonl(pred_path) if r.get("parsed_ok", False)]
    rng = random.Random(args.seed)
    if len(rows) > args.max_rows:
        rows = rng.sample(rows, args.max_rows)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    letter_token_ids = build_letter_token_ids(tok)
    lm_w = model.get_output_embeddings().weight.detach().cpu()
    letter_vecs = {
        k: mean_token_vector(lm_w, ids) for k, ids in letter_token_ids.items()
    }

    delta_h_list: List[np.ndarray] = []
    h_op_list: List[np.ndarray] = []
    v_truth_rows: List[np.ndarray] = []
    labels: List[int] = []
    style_support: List[int] = []

    for i, r in enumerate(rows, start=1):
        msgs = r["prompt_messages"]
        user_content = msgs[-1]["content"]
        question = extract_question_from_user_content(user_content)
        neutral_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Please answer with exactly one capital letter among A, B, C, D, E."
                ),
            },
        ]

        h_op = get_last_hidden_by_layer(model, tok, msgs)
        h_neu = get_last_hidden_by_layer(model, tok, neutral_msgs)
        h_op_np = h_op.numpy().astype(np.float32)
        h_op_list.append(h_op_np)
        delta = (h_op - h_neu).numpy().astype(np.float32)  # [L, D]
        delta_h_list.append(delta)

        c = r["correct_letter"]
        w = r["wrong_letter"]
        v_truth = (letter_vecs[c] - letter_vecs[w]).numpy().astype(np.float32)  # [D]
        v_truth_rows.append(v_truth)

        labels.append(1 if r.get("is_wrong_follow", False) else 0)
        style_support.append(1 if r.get("factors", {}).get("opinion_polarity") == "support" else 0)

        if args.print_every > 0 and i % args.print_every == 0:
            print(f"[progress] {i}/{len(rows)}")

    delta_h = np.stack(delta_h_list, axis=0)  # [N, L, D]
    h_op_mat = np.stack(h_op_list, axis=0)  # [N, L, D]
    v_truth_mat = np.stack(v_truth_rows, axis=0)  # [N, D]
    y = np.array(labels, dtype=np.int64)  # [N]
    s = np.array(style_support, dtype=np.int64)  # [N]

    v_pref = delta_h.mean(axis=0)  # [L, D]
    v_truth_global = v_truth_mat.mean(axis=0)  # [D]
    if int((s == 1).sum()) > 0 and int((s == 0).sum()) > 0:
        v_pref_style = h_op_mat[s == 1].mean(axis=0) - h_op_mat[s == 0].mean(axis=0)  # [L, D]
    else:
        v_pref_style = v_pref.copy()

    def l2_normalize_last(x: np.ndarray, eps: float) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(n, eps)

    # Variant 1: raw projections + global truth vector (original baseline)
    num_raw_global = np.abs(np.einsum("nld,ld->nl", delta_h, v_pref))
    den_raw_global = np.abs(np.einsum("nld,d->nl", delta_h, v_truth_global)) + args.eps
    r_mat_raw_global = num_raw_global / den_raw_global
    max_r_raw_global = r_mat_raw_global.max(axis=1)

    # Variant 2: cosine projections + global truth vector
    delta_n = l2_normalize_last(delta_h, args.eps)
    v_pref_n = l2_normalize_last(v_pref, args.eps)
    v_truth_global_n = v_truth_global / max(np.linalg.norm(v_truth_global), args.eps)
    num_cos_global = np.abs(np.einsum("nld,ld->nl", delta_n, v_pref_n))
    den_cos_global = np.abs(np.einsum("nld,d->nl", delta_n, v_truth_global_n)) + args.eps
    r_mat_cos_global = num_cos_global / den_cos_global
    max_r_cos_global = r_mat_cos_global.max(axis=1)

    # Variant 3: raw projections + row-specific truth vectors
    num_raw_rowtruth = np.abs(np.einsum("nld,ld->nl", delta_h, v_pref))
    den_raw_rowtruth = np.abs(np.einsum("nld,nd->nl", delta_h, v_truth_mat)) + args.eps
    r_mat_raw_rowtruth = num_raw_rowtruth / den_raw_rowtruth
    max_r_raw_rowtruth = r_mat_raw_rowtruth.max(axis=1)

    # Variant 4: cosine projections + row-specific truth vectors
    v_truth_rows_n = l2_normalize_last(v_truth_mat, args.eps)
    num_cos_rowtruth = np.abs(np.einsum("nld,ld->nl", delta_n, v_pref_n))
    den_cos_rowtruth = np.abs(np.einsum("nld,nd->nl", delta_n, v_truth_rows_n)) + args.eps
    r_mat_cos_rowtruth = num_cos_rowtruth / den_cos_rowtruth
    max_r_cos_rowtruth = r_mat_cos_rowtruth.max(axis=1)

    variants = {
        "raw_global": (r_mat_raw_global, max_r_raw_global),
        "cos_global": (r_mat_cos_global, max_r_cos_global),
        "raw_rowtruth": (r_mat_raw_rowtruth, max_r_raw_rowtruth),
        "cos_rowtruth": (r_mat_cos_rowtruth, max_r_cos_rowtruth),
    }

    # Additional variants using support-vs-oppose style direction estimated from h_op.
    num_raw_stylepref = np.abs(np.einsum("nld,ld->nl", h_op_mat, v_pref_style))
    den_raw_stylepref = np.abs(np.einsum("nld,d->nl", h_op_mat, v_truth_global)) + args.eps
    r_mat_raw_stylepref = num_raw_stylepref / den_raw_stylepref
    max_r_raw_stylepref = r_mat_raw_stylepref.max(axis=1)

    h_op_n = l2_normalize_last(h_op_mat, args.eps)
    v_pref_style_n = l2_normalize_last(v_pref_style, args.eps)
    num_cos_stylepref = np.abs(np.einsum("nld,ld->nl", h_op_n, v_pref_style_n))
    den_cos_stylepref = np.abs(np.einsum("nld,d->nl", h_op_n, v_truth_global_n)) + args.eps
    r_mat_cos_stylepref = num_cos_stylepref / den_cos_stylepref
    max_r_cos_stylepref = r_mat_cos_stylepref.max(axis=1)

    variants["raw_stylepref_global"] = (r_mat_raw_stylepref, max_r_raw_stylepref)
    variants["cos_stylepref_global"] = (r_mat_cos_stylepref, max_r_cos_stylepref)

    variant_stats: Dict[str, Dict[str, float]] = {}
    best_name = ""
    best_auc = -1.0
    for name, (_, max_r) in variants.items():
        auc = auc_binary(max_r, y)
        tau, tpr, fpr = best_threshold(max_r, y)
        pos = y == 1
        neg = y == 0
        variant_stats[name] = {
            "auc": float(auc),
            "best_tau": float(tau),
            "best_tpr": float(tpr),
            "best_fpr": float(fpr),
            "mean_max_r_wrong_follow": float(max_r[pos].mean()) if pos.any() else 0.0,
            "mean_max_r_non_wrong_follow": float(max_r[neg].mean()) if neg.any() else 0.0,
        }
        if auc > best_auc:
            best_auc = auc
            best_name = name

    r_mat, max_r = variants[best_name]
    tau = variant_stats[best_name]["best_tau"]
    tpr = variant_stats[best_name]["best_tpr"]
    fpr = variant_stats[best_name]["best_fpr"]
    auc = variant_stats[best_name]["auc"]

    pos = y == 1
    neg = y == 0
    mean_max_r_pos = float(max_r[pos].mean()) if pos.any() else 0.0
    mean_max_r_neg = float(max_r[neg].mean()) if neg.any() else 0.0
    per_layer_mean_pos = r_mat[pos].mean(axis=0).tolist() if pos.any() else [0.0] * r_mat.shape[1]
    per_layer_mean_neg = r_mat[neg].mean(axis=0).tolist() if neg.any() else [0.0] * r_mat.shape[1]

    summary = {
        "pred_path": str(pred_path),
        "rows_used": int(len(rows)),
        "wrong_follow_rate_used": float(y.mean()) if len(y) else 0.0,
        "num_layers": int(r_mat.shape[1]),
        "selected_variant": best_name,
        "variant_stats": variant_stats,
        "r_auc_for_wrong_follow": float(auc),
        "best_tau_by_youden_j": float(tau),
        "best_tau_tpr": float(tpr),
        "best_tau_fpr": float(fpr),
        "mean_max_r_wrong_follow": mean_max_r_pos,
        "mean_max_r_non_wrong_follow": mean_max_r_neg,
        "layer_index_1based": list(range(1, r_mat.shape[1] + 1)),
        "per_layer_mean_r_wrong_follow": per_layer_mean_pos,
        "per_layer_mean_r_non_wrong_follow": per_layer_mean_neg,
    }

    out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {out_json_path}")

    lines = []
    lines.append("# R_l Phase Transition Pilot")
    lines.append("")
    lines.append(f"- Input prediction file: `{pred_path}`")
    lines.append(f"- Rows used: `{summary['rows_used']}`")
    lines.append(f"- Wrong-follow rate (subset): `{summary['wrong_follow_rate_used']:.4f}`")
    lines.append(f"- Num layers: `{summary['num_layers']}`")
    lines.append("")
    lines.append("## Core Results")
    lines.append("")
    lines.append(f"- Selected variant: `{summary['selected_variant']}`")
    lines.append(f"- `AUC(max_l R_l -> wrong_follow)`: `{summary['r_auc_for_wrong_follow']:.4f}`")
    lines.append(f"- `tau*` (Youden-J): `{summary['best_tau_by_youden_j']:.4f}`")
    lines.append(f"- `TPR@tau*`: `{summary['best_tau_tpr']:.4f}`")
    lines.append(f"- `FPR@tau*`: `{summary['best_tau_fpr']:.4f}`")
    lines.append(f"- Mean `max_l R_l` (wrong-follow): `{summary['mean_max_r_wrong_follow']:.4f}`")
    lines.append(f"- Mean `max_l R_l` (non-wrong-follow): `{summary['mean_max_r_non_wrong_follow']:.4f}`")
    lines.append("")
    lines.append("## Variant Comparison")
    lines.append("")
    lines.append("| variant | auc | tau* | tpr | fpr | mean_max_r_wrong_follow | mean_max_r_non_wrong_follow |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, v in summary["variant_stats"].items():
        lines.append(
            f"| {name} | {v['auc']:.4f} | {v['best_tau']:.4f} | {v['best_tpr']:.4f} | {v['best_fpr']:.4f} | {v['mean_max_r_wrong_follow']:.4f} | {v['mean_max_r_non_wrong_follow']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-Layer Means")
    lines.append("")
    lines.append("| layer | mean_R_wrong_follow | mean_R_non_wrong_follow |")
    lines.append("|---:|---:|---:|")
    for idx, (a, b) in enumerate(
        zip(summary["per_layer_mean_r_wrong_follow"], summary["per_layer_mean_r_non_wrong_follow"]),
        start=1,
    ):
        lines.append(f"| {idx} | {a:.4f} | {b:.4f} |")

    out_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_md_path}")


if __name__ == "__main__":
    main()
