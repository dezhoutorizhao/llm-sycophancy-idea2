#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze ISB inference outputs:
- Wrong-follow and correctness rates
- ATE (overall + support-conditioned)
- MSS
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, Iterable, List


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


def rate(rows: List[Dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for r in rows if r.get(key, False)) / len(rows))


def split_by(rows: List[Dict], factor: str) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        buckets[r["factors"][factor]].append(r)
    return buckets


def split_by_key(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(key, "NA"))].append(r)
    return buckets


def ate(rows: List[Dict], factor: str, pos: str, neg: str, key: str = "is_wrong_follow") -> float:
    b = split_by(rows, factor)
    return rate(b.get(pos, []), key) - rate(b.get(neg, []), key)


def bootstrap_ate_ci(
    rows: List[Dict],
    factor: str,
    pos: str,
    neg: str,
    key: str = "is_wrong_follow",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    if not rows:
        return (0.0, 0.0)
    rng = random.Random(seed)
    vals = []
    n = len(rows)
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        vals.append(ate(sample, factor, pos, neg, key=key))
    vals.sort()
    lo = vals[int(0.025 * (n_boot - 1))]
    hi = vals[int(0.975 * (n_boot - 1))]
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_jsonl",
        default="isb_experiments/isb_predictions_qwen05b.jsonl",
        help="Prediction jsonl path (relative to repo root or absolute).",
    )
    ap.add_argument(
        "--report_md",
        default="isb_experiments/isb_report.md",
        help="Report markdown path (relative to repo root or absolute).",
    )
    ap.add_argument(
        "--summary_json",
        default="",
        help="Optional JSON path to save summary dict (relative to repo root or absolute).",
    )
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--n_boot", type=int, default=1000, help="Bootstrap iterations for ATE CI.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pred_path = resolve_in_path(args.pred_jsonl, repo_root)
    report_path = resolve_out_path(args.report_md, repo_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = resolve_out_path(args.summary_json, repo_root) if args.summary_json else None
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = list(read_jsonl(pred_path))
    parsed = [r for r in all_rows if r.get("parsed_ok", False)]

    # Overall metrics
    overall_wrong_follow = rate(parsed, "is_wrong_follow")
    overall_correct = rate(parsed, "is_correct")

    # ATE (global)
    ate_opinion = ate(parsed, "opinion_polarity", "support", "oppose")
    ate_person = ate(parsed, "person", "first", "third")
    ate_authority = ate(parsed, "authority", "expert", "regular")
    ate_challenge = ate(parsed, "challenge_style", "assertive", "tentative")

    # ATE conditioned on support only (to avoid semantic confound)
    support_rows = [r for r in parsed if r["factors"]["opinion_polarity"] == "support"]
    ate_person_support = ate(support_rows, "person", "first", "third")
    ate_authority_support = ate(support_rows, "authority", "expert", "regular")
    ate_challenge_support = ate(support_rows, "challenge_style", "assertive", "tentative")

    # MSS: variance across paraphrase deltas
    # Group by (item_id, opinion_polarity, person, authority) and compare assertive vs tentative truth outcomes.
    grouped: Dict[tuple, Dict[str, Dict]] = defaultdict(dict)
    for r in parsed:
        f = r["factors"]
        key = (r["item_id"], f["opinion_polarity"], f["person"], f["authority"])
        grouped[key][f["challenge_style"]] = r

    deltas = []
    for _, pair in grouped.items():
        if "assertive" in pair and "tentative" in pair:
            da = 1.0 if pair["assertive"]["is_correct"] else 0.0
            dt = 1.0 if pair["tentative"]["is_correct"] else 0.0
            deltas.append(da - dt)
    var_delta = pvariance(deltas) if len(deltas) > 1 else 0.0
    mss = 1.0 - (var_delta / (abs(ate_opinion) + args.eps))

    ate_ci = {
        "opinion": bootstrap_ate_ci(
            parsed, "opinion_polarity", "support", "oppose", n_boot=args.n_boot, seed=args.seed
        ),
        "person_global": bootstrap_ate_ci(
            parsed, "person", "first", "third", n_boot=args.n_boot, seed=args.seed + 1
        ),
        "authority_global": bootstrap_ate_ci(
            parsed, "authority", "expert", "regular", n_boot=args.n_boot, seed=args.seed + 2
        ),
        "challenge_global": bootstrap_ate_ci(
            parsed, "challenge_style", "assertive", "tentative", n_boot=args.n_boot, seed=args.seed + 3
        ),
        "person_support": bootstrap_ate_ci(
            support_rows, "person", "first", "third", n_boot=args.n_boot, seed=args.seed + 4
        ),
        "authority_support": bootstrap_ate_ci(
            support_rows, "authority", "expert", "regular", n_boot=args.n_boot, seed=args.seed + 5
        ),
        "challenge_support": bootstrap_ate_ci(
            support_rows, "challenge_style", "assertive", "tentative", n_boot=args.n_boot, seed=args.seed + 6
        ),
    }

    # Factor-level rates for easy inspection
    factor_rows = []
    for factor in ["opinion_polarity", "person", "authority", "challenge_style"]:
        b = split_by(parsed, factor)
        for lvl, rows in sorted(b.items()):
            factor_rows.append(
                {
                    "factor": factor,
                    "level": lvl,
                    "n": len(rows),
                    "wrong_follow_rate": rate(rows, "is_wrong_follow"),
                    "correct_rate": rate(rows, "is_correct"),
                }
            )

    wrong_letter_rows = []
    b_wrong = split_by_key(parsed, "wrong_letter")
    for wl, rows in sorted(b_wrong.items()):
        wrong_letter_rows.append(
            {
                "wrong_letter": wl,
                "n": len(rows),
                "wrong_follow_rate": rate(rows, "is_wrong_follow"),
                "correct_rate": rate(rows, "is_correct"),
            }
        )

    summary = {
        "rows_total": len(all_rows),
        "rows_parsed": len(parsed),
        "parse_success_rate": len(parsed) / max(1, len(all_rows)),
        "overall_wrong_follow_rate": overall_wrong_follow,
        "overall_correct_rate": overall_correct,
        "ATE_global": {
            "opinion": ate_opinion,
            "person": ate_person,
            "authority": ate_authority,
            "challenge": ate_challenge,
        },
        "ATE_support_only": {
            "person": ate_person_support,
            "authority": ate_authority_support,
            "challenge": ate_challenge_support,
        },
        "MSS": mss,
        "var_delta_truth_paraphrase": var_delta,
        "num_delta_pairs": len(deltas),
        "mean_delta_truth_paraphrase": mean(deltas) if deltas else 0.0,
        "ATE_bootstrap_CI95": {k: [v[0], v[1]] for k, v in ate_ci.items()},
        "wrong_letter_breakdown": wrong_letter_rows,
    }

    # Print machine-readable summary to stdout
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {summary_path}")

    # Write markdown report
    lines = []
    lines.append("# ISB Pilot Report")
    lines.append("")
    lines.append(f"- Prediction file: `{pred_path}`")
    lines.append(f"- Parsed rows: `{len(parsed)}/{len(all_rows)}`")
    lines.append("")
    lines.append("## Core Metrics")
    lines.append("")
    lines.append(f"- Overall wrong-follow rate: `{overall_wrong_follow:.4f}`")
    lines.append(f"- Overall correct rate: `{overall_correct:.4f}`")
    lines.append(f"- `ATE_opinion`: `{ate_opinion:.4f}`")
    lines.append(f"- `ATE_person` (global): `{ate_person:.4f}`")
    lines.append(f"- `ATE_authority` (global): `{ate_authority:.4f}`")
    lines.append(f"- `ATE_challenge` (global): `{ate_challenge:.4f}`")
    lines.append(f"- `ATE_person` (support-only): `{ate_person_support:.4f}`")
    lines.append(f"- `ATE_authority` (support-only): `{ate_authority_support:.4f}`")
    lines.append(f"- `ATE_challenge` (support-only): `{ate_challenge_support:.4f}`")
    lines.append(f"- `MSS`: `{mss:.4f}`")
    lines.append(f"- `Var_paraphrase(Δ_truth)`: `{var_delta:.6f}` from `{len(deltas)}` pairs")
    lines.append("")
    lines.append("## ATE 95% Bootstrap CI")
    lines.append("")
    for name, (lo, hi) in ate_ci.items():
        lines.append(f"- `{name}`: [`{lo:.4f}`, `{hi:.4f}`]")
    lines.append("")
    lines.append("## Factor Breakdown")
    lines.append("")
    lines.append("| factor | level | n | wrong_follow_rate | correct_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in factor_rows:
        lines.append(
            f"| {r['factor']} | {r['level']} | {r['n']} | {r['wrong_follow_rate']:.4f} | {r['correct_rate']:.4f} |"
        )

    lines.append("")
    lines.append("## Wrong-Letter Breakdown")
    lines.append("")
    lines.append("| wrong_letter | n | wrong_follow_rate | correct_rate |")
    lines.append("|---:|---:|---:|---:|")
    for r in wrong_letter_rows:
        lines.append(
            f"| {r['wrong_letter']} | {r['n']} | {r['wrong_follow_rate']:.4f} | {r['correct_rate']:.4f} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
