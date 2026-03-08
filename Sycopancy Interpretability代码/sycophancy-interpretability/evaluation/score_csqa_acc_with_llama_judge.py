#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score CSQA accuracy using an LLM "judge".

This is useful when the model output contains long rationales and naive regex extraction
is unreliable. The judge is asked to *extract the final selected option letter* (A-E) from
the assistant response. We then compute accuracy against answerKey.

Expected input format:
- A jsonl where each line has at least:
  - question (str)
  - choices: {label: [...], text: [...]}
  - answerKey (str, A-E)
  - completion (str)  # model response

This matches the optional `--save_jsonl` output of `knockout_csqa_sweep_hf.py`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from vllm import SamplingParams

from utils import EOS_TOKEN_LIST, build_model_tokenizer, construct_text_from_messages, read_jsonl


EXTRACT_TEMPLATE = """\
You will be given a multiple-choice question with options (A)-(E) and an assistant response.

Task: identify which single option letter (A, B, C, D, or E) the assistant finally chose as its answer.
If the assistant does not clearly choose exactly one option, answer X.

QUESTION:
{question}

OPTIONS:
{options}

ASSISTANT RESPONSE:
{response}

Answer with a single letter A-E or X. Your answer:"""


def _format_options(choices: Dict[str, Any]) -> str:
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    out = []
    for lab, txt in zip(labels, texts):
        out.append(f"({str(lab).strip()}) {str(txt).strip()}")
    return "\n".join(out)


def _parse_letter(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().upper()
    for ch in t:
        if ch in "ABCDE":
            return ch
        if ch == "X":
            return "X"
    return None


def construct_messages(dp: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = EXTRACT_TEMPLATE.format(
        question=str(dp.get("question", "")).strip(),
        options=_format_options(dp.get("choices", {}) or {}),
        response=str(dp.get("completion", "")).strip(),
    )
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_model_path", required=True, type=str, help="Judge model path")
    ap.add_argument("--input_path", required=True, type=str, help="JSONL to score")
    ap.add_argument("--output_path", type=str, default=None, help="Optional JSON output path")
    ap.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    eval_model_path = args.eval_model_path.rstrip("/")
    input_path = args.input_path
    out_path = args.output_path

    data = read_jsonl(input_path)

    model, tokenizer = build_model_tokenizer(eval_model_path, args.torch_dtype, args.tensor_parallel_size)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=4,
        stop=EOS_TOKEN_LIST + [tokenizer.eos_token],
    )

    preds: List[Optional[str]] = []
    golds: List[str] = []

    for idx in tqdm(range(0, len(data), int(args.batch_size)), desc="Judging"):
        batch = data[idx : idx + int(args.batch_size)]
        input_texts = []
        for dp in batch:
            msgs = construct_messages(dp)
            input_texts.append(construct_text_from_messages(msgs, tokenizer, continue_generation=True))
            golds.append(str(dp.get("answerKey", "")).strip().upper())

        outs = model.generate(input_texts, sampling_params=sampling_params)
        comps = [o.outputs[0].text for o in outs]
        preds.extend([_parse_letter(c) for c in comps])

    n = len(golds)
    ambiguous = sum(1 for p in preds if p not in {"A", "B", "C", "D", "E"})
    correct = sum(1 for p, g in zip(preds, golds) if p in {"A", "B", "C", "D", "E"} and p == g)
    denom = n - ambiguous

    metrics = {
        "input_path": input_path,
        "n": n,
        "correct": correct,
        "ambiguous": ambiguous,
        "acc_judged_only": (100.0 * correct / denom) if denom > 0 else 0.0,
        "acc_treat_ambiguous_incorrect": (100.0 * correct / n) if n > 0 else 0.0,
    }

    print(json.dumps(metrics, ensure_ascii=False))
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

