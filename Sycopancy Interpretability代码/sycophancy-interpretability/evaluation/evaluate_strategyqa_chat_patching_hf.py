#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Ability (StrategyQA) evaluation with AAAI'26-style activation patching (HF/Transformers).

Protocol matches `evaluate_strategyqa_chat_vllm.py` (few-shot + yes/no).
Input format matches the repo dataset: `datasets/strategyqa_test.json` with {"examples": [...]}.
"""

from __future__ import annotations

import argparse
import json
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
from utils_fewshot import FEWSHOT_STRATEGYQA


def extract_answer_yes_no(text: str) -> str | None:
    if not text:
        return None
    # Common patterns: "Yes", "No", "Answer: yes"
    m = re.search(r"\b(yes|no)\b", text.lower())
    if m:
        return m.group(1)
    return None


def construct_messages(datapoint: Dict) -> List[Dict]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for q, a in zip(FEWSHOT_STRATEGYQA[::2], FEWSHOT_STRATEGYQA[1::2]):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    prompt = "Q: " + datapoint["input"].strip()
    messages.append({"role": "user", "content": prompt})
    return messages


def is_correct(completion: str, answer_scores: Dict) -> bool:
    gold = "yes" if answer_scores.get("Yes", 0) == 1 else "no"
    pred = extract_answer_yes_no(completion)
    return pred is not None and pred.lower() == gold.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", default="results/strategyqa_aaai_patching")
    ap.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="cuda", choices=["cuda", "auto"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--patch_layers", default="", help="Comma-separated layer indices. Empty disables patching.")
    ap.add_argument("--ref_mode", default="identity", choices=["identity"])
    ap.add_argument("--max_length", type=int, default=4096)
    args = ap.parse_args()

    bundle = load_hf_model(args.model_path, torch_dtype=args.torch_dtype, device_map=args.device_map)
    model, tok = bundle.model, bundle.tokenizer
    patch_layers = parse_layers(args.patch_layers)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, os.path.basename(args.model_path.rstrip("/")) + ".jsonl")

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)["examples"]

    all_acc: List[bool] = []

    pbar = tqdm(range(0, len(data), args.batch_size), desc="StrategyQA (HF+patching)")
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
                dp["is_correct"] = is_correct(comp, dp["target_scores"])
                f.write(json.dumps(dp, ensure_ascii=False) + "\n")
                all_acc.append(dp["is_correct"])

        pbar.set_description(f"StrategyQA (HF+patching) AVG ACC: {sum(all_acc)/len(all_acc)*100:.2f}%")

    print("Finished!")
    print(f"Final AVG ACC: {sum(all_acc)/len(all_acc)*100:.2f}%")
    print("Output:", out_path)


if __name__ == "__main__":
    main()

