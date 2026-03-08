#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Ability (CSQA) evaluation with AAAI'26-style activation patching (HF/Transformers).

This is a *baseline for comparison*: we keep the evaluation protocol identical to
`evaluate_csqa_chat_vllm.py`, but replace vLLM generation with deterministic HF generation
and optionally apply the AAAI-style patching hook on selected layers.

Important:
- Patching is applied only at the first decoding step at the last prompt token.
- By default, `--ref_mode identity` uses the prompt itself as the reference. This is the
  least opinionated way to measure whether *enabling* AAAI patching harms General Ability.
  If you want a different reference construction, extend `build_ref_texts`.
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
from utils_fewshot import FEWSHOT_CSQA


def read_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]


def extract_answer_multiple_choice(text: str) -> str | None:
    if not text:
        return None
    # 1) (A)
    m = re.search(r"\(([A-Za-z])\)", text)
    if m:
        return m.group(1).upper()
    # 2) A)
    m = re.search(r"([A-Za-z])\)", text)
    if m:
        return m.group(1).upper()
    # 3) first uppercase letter among A-H
    m = re.search(r"\b([A-H])\b", text)
    if m:
        return m.group(1).upper()
    return None


def construct_messages(datapoint: Dict) -> List[Dict]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    # few-shot
    for q, a in zip(FEWSHOT_CSQA[::2], FEWSHOT_CSQA[1::2]):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    # question
    prompt = "Q: " + datapoint["question"].strip() + "\nAnswer Choices:\n"
    for label, text in zip(datapoint["choices"]["label"], datapoint["choices"]["text"]):
        prompt += f"({label}) {text.strip()}\n"
    messages.append({"role": "user", "content": prompt})
    return messages


def is_correct(completion: str, answer: str) -> bool:
    pred = extract_answer_multiple_choice(completion)
    return pred is not None and pred == answer


def build_ref_texts(input_texts: List[str], ref_mode: str) -> List[str]:
    if ref_mode == "identity":
        return input_texts
    raise ValueError(f"Unknown ref_mode={ref_mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", default="results/csqa_aaai_patching")
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

    data = read_jsonl(args.data_path)
    all_acc: List[bool] = []

    pbar = tqdm(range(0, len(data), args.batch_size), desc="CSQA (HF+patching)")
    for start in pbar:
        batch = data[start : start + args.batch_size]
        input_texts = []
        for dp in batch:
            msgs = construct_messages(dp)
            input_texts.append(construct_text_from_messages(msgs, tok, continue_generation=True))

        ref_texts = build_ref_texts(input_texts, args.ref_mode)
        completions = generate_batch_with_patching(
            model,
            tok,
            input_texts,
            patch_layers=patch_layers,
            ref_texts=ref_texts,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
        )

        with open(out_path, "a", encoding="utf-8") as f:
            for dp, comp in zip(batch, completions):
                dp = dict(dp)
                dp["completion"] = comp
                dp["is_correct"] = is_correct(comp, dp["answerKey"])
                f.write(json.dumps(dp, ensure_ascii=False) + "\n")
                all_acc.append(dp["is_correct"])

        pbar.set_description(f"CSQA (HF+patching) AVG ACC: {sum(all_acc)/len(all_acc)*100:.2f}%")

    print("Finished!")
    print(f"Final AVG ACC: {sum(all_acc)/len(all_acc)*100:.2f}%")
    print("Output:", out_path)


if __name__ == "__main__":
    main()

