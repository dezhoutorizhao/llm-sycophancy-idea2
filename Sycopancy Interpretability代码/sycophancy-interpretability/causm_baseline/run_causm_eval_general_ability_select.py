#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate General Ability under CAUSM baseline, with selectable tasks.

This file is intentionally a non-breaking alternative to
`run_causm_eval_general_ability.py`, which always runs CSQA+GSM8K+StrategyQA due to
default paths and unconditional execution.

Usage examples:
  # Only StrategyQA
  CUDA_VISIBLE_DEVICES=0,1 python3 -u causm_baseline/run_causm_eval_general_ability_select.py \\
    --tasks strategyqa --model_path /root/shared-nvme/llama3-exp__causm \\
    --w_path causm_baseline/outputs/llama3-exp__causm/W.pt --device_map auto

  # CSQA + StrategyQA
  python3 -u causm_baseline/run_causm_eval_general_ability_select.py \\
    --tasks csqa,strategyqa --model_path ... --w_path ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, List

import torch
from tqdm import tqdm

from causm_common import construct_text_from_messages, greedy_generate_with_causm, load_hf

# Import few-shot prompts from the repo's evaluation utilities without modifying them.
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EVAL = os.path.join(_ROOT, "evaluation")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

from utils_fewshot import FEWSHOT_CSQA, FEWSHOT_STRATEGYQA  # type: ignore


def read_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]


def read_strategyqa_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # The repo uses a StrategyQA JSON with {"examples": [...]}.
    return obj["examples"]


def _extract_last_match(pattern: str, text: str) -> str | None:
    if not text:
        return None
    ms = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not ms:
        return None
    return ms[-1].group(1)


def extract_answer_multiple_choice(text: str) -> str | None:
    # Prefer the last A-D-like token; this is more robust than the first match when the model
    # includes "A:" or other intermediate tokens.
    x = _extract_last_match(r"\b([A-H])\b", text)
    return x.upper() if x else None


def extract_answer_last_number(text: str) -> str | None:
    if not text:
        return None
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text.replace(",", ""))
    if not nums:
        return None
    return nums[-1]


def extract_answer_yes_no(text: str) -> str | None:
    # Use the last yes/no token to avoid early "yes/no" appearing in explanations.
    x = _extract_last_match(r"\b(yes|no)\b", text)
    return x.lower() if x else None


def is_correct_csqa(completion: str, answer: str) -> bool:
    pred = extract_answer_multiple_choice(completion)
    return pred is not None and pred.upper() == answer.upper()


def is_correct_gsm8k(completion: str, answer: str) -> bool:
    gold = extract_answer_last_number(answer)
    pred = extract_answer_last_number(completion)
    if gold is None or pred is None:
        return False
    try:
        return math.isclose(float(eval(gold)), float(eval(pred)), rel_tol=0, abs_tol=1e-4)
    except Exception:
        return False


def is_correct_strategyqa(completion: str, scores: Dict) -> bool:
    gold = "yes" if scores.get("Yes", 0) == 1 else "no"
    pred = extract_answer_yes_no(completion)
    return pred is not None and pred.lower() == gold.lower()


def msgs_csqa(dp: Dict) -> List[Dict]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for q, a in zip(FEWSHOT_CSQA[::2], FEWSHOT_CSQA[1::2]):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    prompt = "Q: " + dp["question"].strip() + "\nAnswer Choices:\n"
    for label, text in zip(dp["choices"]["label"], dp["choices"]["text"]):
        prompt += f"({label}) {text.strip()}\n"
    # Strongly constrain format.
    prompt += "\nPlease answer with a single letter (A, B, C, D, or E)."
    messages.append({"role": "user", "content": prompt})
    return messages


def msgs_gsm8k(dp: Dict) -> List[Dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": dp["question"]},
    ]


def msgs_strategyqa(dp: Dict) -> List[Dict]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for q, a in zip(FEWSHOT_STRATEGYQA[::2], FEWSHOT_STRATEGYQA[1::2]):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    # Strongly constrain format.
    messages.append(
        {
            "role": "user",
            "content": "Q: " + dp["input"].strip() + "\nAnswer with Yes or No only.",
        }
    )
    return messages


def _parse_tasks(s: str) -> List[str]:
    tasks = []
    for t in (s or "").split(","):
        t = t.strip().lower()
        if not t:
            continue
        tasks.append(t)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="csqa,gsm8k,strategyqa", help="Comma-separated: csqa,gsm8k,strategyqa")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--w_path", required=True, help="Path to W.pt produced by train_causm_w.py")
    ap.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="cuda", choices=["cuda", "auto"])
    ap.add_argument("--patch_layers", default="", help="Optional comma-separated subset of layers; empty = all layers.")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--lamb", type=float, default=0.0, help="CAC strength. Default 0 for General Ability.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--max_new_tokens_csqa", type=int, default=32)
    ap.add_argument("--max_new_tokens_gsm8k", type=int, default=256)
    ap.add_argument("--max_new_tokens_strategyqa", type=int, default=128)
    ap.add_argument("--csqa_path", default="evaluation/datasets/csqa_validation.jsonl")
    ap.add_argument("--gsm8k_path", default="evaluation/datasets/gsm8k_test.jsonl")
    ap.add_argument("--strategyqa_path", default="evaluation/datasets/strategyqa_test.json")
    ap.add_argument("--out_dir", default="evaluation/results")
    args = ap.parse_args()

    tasks = set(_parse_tasks(args.tasks))
    valid = {"csqa", "gsm8k", "strategyqa"}
    if not tasks.issubset(valid):
        raise SystemExit(f"--tasks must be subset of {sorted(valid)}; got {sorted(tasks)}")

    bundle = load_hf(args.model_path, torch_dtype=args.torch_dtype, device_map=args.device_map)
    model, tok = bundle.model, bundle.tok

    ckpt = torch.load(args.w_path, map_location="cpu")
    W = ckpt["W"]

    model_name = os.path.basename(args.model_path.rstrip("/"))
    os.makedirs(args.out_dir, exist_ok=True)

    csqa_out = gsm_out = st_out = None

    if "csqa" in tasks:
        csqa_out_dir = os.path.join(args.out_dir, "csqa_causm")
        os.makedirs(csqa_out_dir, exist_ok=True)
        csqa_out = os.path.join(csqa_out_dir, model_name + ".jsonl")
        if os.path.exists(csqa_out):
            os.remove(csqa_out)
        csqa = read_jsonl(args.csqa_path)
        acc = []
        pbar = tqdm(range(0, len(csqa), args.batch_size), desc="CSQA (CAUSM)")
        for i in pbar:
            batch = csqa[i : i + args.batch_size]
            prompts = [construct_text_from_messages(msgs_csqa(dp), tok, continue_generation=True) for dp in batch]
            comps = greedy_generate_with_causm(
                model,
                tok,
                prompts,
                ref_texts=prompts,
                W=W,
                patch_layers=None,
                topk=args.topk,
                lamb=args.lamb,
                max_new_tokens=args.max_new_tokens_csqa,
                max_length=args.max_length,
            )
            with open(csqa_out, "a", encoding="utf-8") as f:
                for dp, c in zip(batch, comps):
                    item = dict(dp)
                    item["completion"] = c
                    item["is_correct"] = is_correct_csqa(c, dp["answerKey"])
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    acc.append(item["is_correct"])
            pbar.set_description(f"CSQA (CAUSM) AVG ACC: {sum(acc)/len(acc)*100:.2f}%")

    if "gsm8k" in tasks:
        gsm_out_dir = os.path.join(args.out_dir, "gsm8k_causm")
        os.makedirs(gsm_out_dir, exist_ok=True)
        gsm_out = os.path.join(gsm_out_dir, model_name + ".jsonl")
        if os.path.exists(gsm_out):
            os.remove(gsm_out)
        gsm = read_jsonl(args.gsm8k_path)
        acc = []
        pbar = tqdm(range(0, len(gsm), args.batch_size), desc="GSM8K (CAUSM)")
        for i in pbar:
            batch = gsm[i : i + args.batch_size]
            prompts = [construct_text_from_messages(msgs_gsm8k(dp), tok, continue_generation=True) for dp in batch]
            comps = greedy_generate_with_causm(
                model,
                tok,
                prompts,
                ref_texts=prompts,
                W=W,
                patch_layers=None,
                topk=args.topk,
                lamb=args.lamb,
                max_new_tokens=args.max_new_tokens_gsm8k,
                max_length=args.max_length,
            )
            with open(gsm_out, "a", encoding="utf-8") as f:
                for dp, c in zip(batch, comps):
                    item = dict(dp)
                    item["completion"] = c
                    item["is_correct"] = is_correct_gsm8k(c, dp["answer"])
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    acc.append(item["is_correct"])
            pbar.set_description(f"GSM8K (CAUSM) AVG ACC: {sum(acc)/len(acc)*100:.2f}%")

    if "strategyqa" in tasks:
        st_out_dir = os.path.join(args.out_dir, "strategyqa_causm")
        os.makedirs(st_out_dir, exist_ok=True)
        st_out = os.path.join(st_out_dir, model_name + ".jsonl")
        if os.path.exists(st_out):
            os.remove(st_out)
        st = read_strategyqa_json(args.strategyqa_path)
        acc = []
        pbar = tqdm(range(0, len(st), args.batch_size), desc="StrategyQA (CAUSM)")
        for i in pbar:
            batch = st[i : i + args.batch_size]
            prompts = [construct_text_from_messages(msgs_strategyqa(dp), tok, continue_generation=True) for dp in batch]
            comps = greedy_generate_with_causm(
                model,
                tok,
                prompts,
                ref_texts=prompts,
                W=W,
                patch_layers=None,
                topk=args.topk,
                lamb=args.lamb,
                max_new_tokens=args.max_new_tokens_strategyqa,
                max_length=args.max_length,
            )
            with open(st_out, "a", encoding="utf-8") as f:
                for dp, c in zip(batch, comps):
                    item = dict(dp)
                    item["completion"] = c
                    item["is_correct"] = is_correct_strategyqa(c, dp["target_scores"])
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    acc.append(item["is_correct"])
            pbar.set_description(f"StrategyQA (CAUSM) AVG ACC: {sum(acc)/len(acc)*100:.2f}%")

    print("Done.")
    if csqa_out:
        print("CSQA:", csqa_out)
    if gsm_out:
        print("GSM8K:", gsm_out)
    if st_out:
        print("StrategyQA:", st_out)


if __name__ == "__main__":
    main()

