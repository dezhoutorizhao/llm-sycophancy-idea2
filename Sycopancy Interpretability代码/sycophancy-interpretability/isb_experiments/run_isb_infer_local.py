#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run local inference on ISB factorial dataset and dump per-condition predictions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_letter_token_ids(tok) -> Dict[str, List[int]]:
    # Gather robust candidate token IDs for each answer letter.
    # We use first-token IDs of several short variants.
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


@torch.no_grad()
def predict_letter_by_logits(model, inp, letter_token_ids: Dict[str, List[int]]) -> tuple[str, Dict[str, float]]:
    out = model(**inp)
    logits = out.logits[0, -1, :]
    scores: Dict[str, float] = {}
    for letter, ids in letter_token_ids.items():
        letter_scores = [float(logits[i].item()) for i in ids if 0 <= i < logits.shape[0]]
        scores[letter] = max(letter_scores) if letter_scores else float("-inf")
    pred = max(scores, key=scores.get)
    return pred, scores


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Local model directory.")
    ap.add_argument(
        "--input_jsonl",
        default="isb_experiments/isb_multiple_choice_factorial.jsonl",
        help="ISB input jsonl path (relative to repo root or absolute).",
    )
    ap.add_argument(
        "--output_jsonl",
        default="isb_experiments/isb_predictions_qwen05b.jsonl",
        help="Prediction output path (relative to repo root or absolute).",
    )
    ap.add_argument("--max_rows", type=int, default=640, help="Limit rows for pilot run.")
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument(
        "--decode_preview",
        action="store_true",
        help="Additionally run short greedy decode for human inspection (slower).",
    )
    ap.add_argument("--preview_new_tokens", type=int, default=8)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = resolve_in_path(args.input_jsonl, repo_root)
    output_path = resolve_out_path(args.output_jsonl, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()
    letter_token_ids = build_letter_token_ids(tok)
    print(
        "[letter_token_ids]",
        {k: len(v) for k, v in letter_token_ids.items()},
    )

    t0 = time.time()
    n = 0
    with output_path.open("w", encoding="utf-8") as fw:
        for row in read_jsonl(input_path):
            if n >= args.max_rows:
                break
            msgs = row["prompt_messages"]
            prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(prompt_text, return_tensors="pt")

            pred, scores = predict_letter_by_logits(model, inp, letter_token_ids)
            gen = ""
            if args.decode_preview:
                with torch.no_grad():
                    out = model.generate(
                        **inp,
                        max_new_tokens=args.preview_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                gen = tok.decode(out[0, inp["input_ids"].shape[-1] :], skip_special_tokens=True)

            correct = row["correct_letter"]
            wrong = row["wrong_letter"]
            result = {
                **row,
                "prediction_text": gen,
                "pred_letter": pred,
                "letter_scores": scores,
                "parsed_ok": True,
                "is_correct": pred == correct,
                "is_wrong_follow": pred == wrong,
            }
            fw.write(json.dumps(result, ensure_ascii=False) + "\n")
            n += 1
            if args.print_every > 0 and n % args.print_every == 0:
                elapsed = time.time() - t0
                print(f"[progress] {n} rows, elapsed={elapsed:.1f}s, avg={elapsed/max(1,n):.3f}s/row")

    elapsed = time.time() - t0
    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "rows_written": n,
                "elapsed_sec": round(elapsed, 3),
                "sec_per_row": round(elapsed / max(1, n), 4),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
