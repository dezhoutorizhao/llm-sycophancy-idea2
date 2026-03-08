#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Ability (GSM8K) evaluation with AAAI'26-style activation patching (HF/Transformers).

Protocol matches `evaluate_gsm8k_chat_vllm.py` (single-turn Q -> model answer),
but uses HF generation and optional AAAI-style patching.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, List

from tqdm import tqdm

from patching_hf_common import (
    construct_text_from_messages,
    generate_batch_with_patching,
    load_hf_model,
    parse_layers,
)


def read_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]


def construct_messages(datapoint: Dict) -> List[Dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": datapoint["question"]},
    ]


def extract_answer_last_number(text: str) -> str | None:
    if not text:
        return None
    # GSM8K answers often contain "#### 42" or end with a number.
    # Take the last occurrence of a number (integer/decimal/fraction).
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text.replace(",", ""))
    if not nums:
        return None
    return nums[-1]


def is_correct(completion: str, answer: str) -> bool:
    gold = extract_answer_last_number(answer)
    pred = extract_answer_last_number(completion)
    if gold is None or pred is None:
        return False
    try:
        return math.isclose(float(eval(gold)), float(eval(pred)), rel_tol=0, abs_tol=1e-4)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", default="results/gsm8k_aaai_patching")
    ap.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="cuda", choices=["cuda", "auto"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--patch_layers", default="", help="Comma-separated layer indices. Empty disables patching.")
    ap.add_argument("--ref_mode", default="identity", choices=["identity"])
    ap.add_argument("--max_length", type=int, default=4096)
    args = ap.parse_args()

    bundle = load_hf_model(args.model_path, torch_dtype=args.torch_dtype, device_map=args.device_map)
    model, tok = bundle.model, bundle.tokenizer
    patch_layers = parse_layers(args.patch_layers)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, os.path.basename(args.model_path.rstrip("/")) + ".jsonl")

    data = read_jsonl(args.data_path)
    all_acc: List[bool] = []

    pbar = tqdm(range(0, len(data), args.batch_size), desc="GSM8K (HF+patching)")
    for start in pbar:
        batch = data[start : start + args.batch_size]
        input_texts = []
        for dp in batch:
            msgs = construct_messages(dp)
            input_texts.append(construct_text_from_messages(msgs, tok, continue_generation=True))

        completions = generate_batch_with_patching(
            model,
            tok,
            input_texts,
            patch_layers=patch_layers,
            ref_texts=input_texts,  # identity ref
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
        )

        with open(out_path, "a", encoding="utf-8") as f:
            for dp, comp in zip(batch, completions):
                dp = dict(dp)
                dp["completion"] = comp
                dp["is_correct"] = is_correct(comp, dp["answer"])
                f.write(json.dumps(dp, ensure_ascii=False) + "\n")
                all_acc.append(dp["is_correct"])

        pbar.set_description(f"GSM8K (HF+patching) AVG ACC: {sum(all_acc)/len(all_acc)*100:.2f}%")

    print("Finished!")
    print(f"Final AVG ACC: {sum(all_acc)/len(all_acc)*100:.2f}%")
    print("Output:", out_path)


if __name__ == "__main__":
    main()

