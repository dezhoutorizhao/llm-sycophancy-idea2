#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate HumanEval completions under top-k head knockout (HF backend).

Notes:
- This script only generates completions; evaluate pass@1 with:
    pip install human-eval
    evaluate_functional_correctness <output_jsonl>
- We keep the prompt format from `evaluation/evaluate_humaneval_chat_vllm.py` for comparability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from utils import build_model_tokenizer_hf, construct_text_from_messages, read_jsonl


def _infer_head_dim(model) -> int:
    cfg = model.config
    if getattr(cfg, "head_dim", None) is not None:
        return int(cfg.head_dim)
    return int(cfg.hidden_size // cfg.num_attention_heads)


def _ranked_heads_from_results_pt(results_pt: str, *, score_order: str = "asc") -> List[Tuple[int, int]]:
    scores = torch.load(results_pt, map_location="cpu")
    if hasattr(scores, "numpy"):
        scores = scores.numpy()
    n_layers, n_heads = scores.shape
    flat = scores.reshape(-1)
    order = flat.argsort()
    if score_order == "desc":
        order = order[::-1]
    return [(int(i // n_heads), int(i % n_heads)) for i in order.tolist()]


def _select_topk(
    ranked: Sequence[Tuple[int, int]],
    *,
    k: int,
    per_layer_cap: Optional[int] = None,
) -> List[Tuple[int, int]]:
    if k <= 0:
        return []
    if per_layer_cap is None:
        return list(ranked[:k])

    selected: List[Tuple[int, int]] = []
    used_per_layer: Dict[int, int] = {}
    for layer, head in ranked:
        if used_per_layer.get(layer, 0) >= per_layer_cap:
            continue
        selected.append((int(layer), int(head)))
        used_per_layer[layer] = used_per_layer.get(layer, 0) + 1
        if len(selected) >= k:
            break
    return selected


def _install_knockout_hooks(model, heads_by_layer: Dict[int, Sequence[int]]):
    head_dim = _infer_head_dim(model)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Unsupported model structure: expected model.model.layers (LLaMA-like).")

    hooks = []
    for layer_idx, head_list in heads_by_layer.items():
        if not head_list:
            continue
        o_proj = layers[int(layer_idx)].self_attn.o_proj
        head_list = list(sorted(set(int(h) for h in head_list)))

        def _hook(module, inputs, head_list=head_list):
            x = inputs[0]
            if x is None:
                return inputs
            x = x.clone()
            for h in head_list:
                s = h * head_dim
                e = (h + 1) * head_dim
                x[..., s:e] = 0
            return (x,) + inputs[1:]

        hooks.append(o_proj.register_forward_pre_hook(_hook))
    return hooks


def _construct_messages(datapoint: Dict[str, Any]) -> List[Dict[str, str]]:
    # Keep the exact behavior from evaluate_humaneval_chat_vllm.py.
    signature = re.search(rf"def\s+({datapoint['entry_point']}.*?):\s*\n", datapoint["prompt"]).group(1)
    description = "\n".join(
        [
            line.strip()
            for line in re.search(r"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", datapoint["prompt"], re.DOTALL)
            .group(1)
            .split("\n")
        ]
    )
    prompt = (
        f"Write a Python function `{signature}` to solve the following problem:\n"
        f"{description}\n"
        f"{datapoint['prompt']}"
    )

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


def _extract_code(text: str, entry_point: str) -> str:
    # Extract the code block (copied from evaluate_humaneval_chat_vllm.py).
    code_block_pattern = re.compile(rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL)
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL)
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL)
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        code_str = code_block.group(1)
        # Legacy artifact from some encodings; keep behavior but in ASCII.
        code_str = code_str.replace("ã€?, ", "")
        return code_str

    # If no code block is found, assume the LM is simply filling the code.
    return textwrap.indent(text, " " * 4)


@torch.no_grad()
def _generate(model, tokenizer, prompts: List[str], *, batch_size: int, max_new_tokens: int) -> List[str]:
    out: List[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        tok = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        tok = {k: v.to(device) for k, v in tok.items()}
        in_width = int(tok["input_ids"].shape[1])
        gen = model.generate(
            **tok,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for row in range(gen.shape[0]):
            out.append(tokenizer.decode(gen[row, in_width:], skip_special_tokens=True))
    return out


def _iter_k_values(args) -> List[int]:
    if args.k_values is not None and len(args.k_values) > 0:
        return [int(k) for k in args.k_values]
    return [int(args.k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--results_pt", required=True, type=str)
    ap.add_argument("--data_path", required=True, type=str)

    ap.add_argument("--output_dir", required=True, type=str)

    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", type=str, default="auto")

    ap.add_argument("--score_order", type=str, default="asc", choices=["asc", "desc"])
    ap.add_argument("--per_layer_cap", type=int, default=None)

    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--k_values", type=int, nargs="*", default=None)
    ap.add_argument("--resume", action="store_true", help="Skip task_ids already in the output file.")
    args = ap.parse_args()

    data = read_jsonl(args.data_path)
    if args.max_samples is not None:
        data = data[: int(args.max_samples)]

    ranked = _ranked_heads_from_results_pt(args.results_pt, score_order=args.score_order)

    model, tokenizer = build_model_tokenizer_hf(args.model_path, args.dtype, device=args.device_map)
    model.eval()

    model_name = os.path.basename(args.model_path.rstrip("/\\"))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for k in _iter_k_values(args):
        out_dir = out_root / f"k_{int(k)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}.jsonl"

        to_run = data
        if args.resume and out_path.exists():
            done = set()
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["task_id"])
                    except Exception:
                        continue
            to_run = [dp for dp in data if dp.get("task_id") not in done]

        selected = _select_topk(ranked, k=int(k), per_layer_cap=args.per_layer_cap)
        heads_by_layer: Dict[int, List[int]] = {}
        for layer, head in selected:
            heads_by_layer.setdefault(int(layer), []).append(int(head))

        hooks = _install_knockout_hooks(model, heads_by_layer) if k > 0 else []
        try:
            prompts = [
                construct_text_from_messages(_construct_messages(dp), tokenizer, continue_generation=True) for dp in to_run
            ]
            responses = _generate(
                model,
                tokenizer,
                prompts,
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.max_new_tokens),
            )

            with open(out_path, "a", encoding="utf-8") as f:
                for dp, resp in zip(to_run, responses):
                    completion = _extract_code(resp, dp["entry_point"])
                    item = {"task_id": dp["task_id"], "completion": completion}
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"[DONE] k={k} wrote {len(to_run)} to {out_path}")
        finally:
            for h in hooks:
                h.remove()


if __name__ == "__main__":
    main()

