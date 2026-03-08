#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-k head knockout sweep on CommonsenseQA (CSQA) using Hugging Face generation.

Why this script exists:
- The repo's CSQA evaluator uses vLLM (fast) but vLLM does not easily support per-head
  masking hooks. For head knockout, we need HF so we can attach hooks and zero specific
  attention-head slices at the input of each layer's self_attn.o_proj.

This script mirrors the prompting in `evaluation/evaluate_csqa_chat_vllm.py` (few-shot CoT),
but runs a sweep over k values and reports accuracy for each k.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from utils import build_model_tokenizer_hf, construct_text_from_messages, read_jsonl
from utils_fewshot import FEWSHOT_CSQA


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
            # x: [batch, seq, hidden]. Heads are contiguous head_dim blocks.
            x = x.clone()
            for h in head_list:
                s = h * head_dim
                e = (h + 1) * head_dim
                x[..., s:e] = 0
            return (x,) + inputs[1:]

        hooks.append(o_proj.register_forward_pre_hook(_hook))
    return hooks


_MC_LETTER_RE = re.compile(r"\(([A-Za-z])\)")


def _extract_letter(text: str, *, letters: str = "ABCDE") -> Optional[str]:
    if not text:
        return None
    t = text.strip()

    # 1) "(C)" style.
    m = _MC_LETTER_RE.search(t)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()

    # 2) Leading letter.
    m = re.match(r"^\s*([A-Za-z])\b", t)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()

    # 3) "answer is C" / "So the answer is (C)".
    m = re.search(rf"\b([{re.escape(letters)}])\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def _construct_messages(dp: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant."}]

    # Few-shot (copied from evaluate_csqa_chat_vllm.py).
    for question, answer in zip(FEWSHOT_CSQA[::2], FEWSHOT_CSQA[1::2]):
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    prompt = "Q: " + str(dp["question"]).strip() + "\nAnswer Choices:\n"
    labels = dp["choices"]["label"]
    texts = dp["choices"]["text"]
    for lab, txt in zip(labels, texts):
        prompt += f"({str(lab).strip()}) {str(txt).strip()}\n"

    messages.append({"role": "user", "content": prompt})
    return messages


@torch.no_grad()
def _generate(model, tokenizer, prompts: List[str], *, batch_size: int, max_new_tokens: int) -> List[str]:
    out: List[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        tok = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        tok = {k: v.to(device) for k, v in tok.items()}
        # Generated tokens start after the padded input width (not after per-row nonpad length).
        in_width = int(tok["input_ids"].shape[1])
        gen = model.generate(
            **tok,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for row in range(gen.shape[0]):
            out.append(tokenizer.decode(gen[row, in_width:], skip_special_tokens=True).strip())
    return out


def _iter_k_values(args) -> List[int]:
    if args.k_values is not None and len(args.k_values) > 0:
        return [int(k) for k in args.k_values]
    return list(range(int(args.k_min), int(args.k_max) + 1, int(args.k_step)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--results_pt", required=True, type=str)
    ap.add_argument("--data_path", required=True, type=str)

    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--summary_path", type=str, default=None)
    ap.add_argument(
        "--save_jsonl",
        action="store_true",
        help="Also save per-example generations to output_dir/k_<k>/<model>.jsonl",
    )

    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", type=str, default="auto")

    ap.add_argument("--score_order", type=str, default="asc", choices=["asc", "desc"])
    ap.add_argument("--per_layer_cap", type=int, default=None)

    ap.add_argument("--k_values", type=int, nargs="*", default=None)
    ap.add_argument("--k_min", type=int, default=0)
    ap.add_argument("--k_max", type=int, default=100)
    ap.add_argument("--k_step", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path) if args.summary_path else out_dir / "summary.json"

    data = read_jsonl(args.data_path)
    if args.max_samples is not None:
        data = data[: int(args.max_samples)]

    ranked = _ranked_heads_from_results_pt(args.results_pt, score_order=args.score_order)

    model, tokenizer = build_model_tokenizer_hf(args.model_path, args.dtype, device=args.device_map)
    model.eval()
    model_name = os.path.basename(args.model_path.rstrip("/\\"))

    k_values = _iter_k_values(args)
    results: List[Dict[str, Any]] = []

    for k in k_values:
        selected = _select_topk(ranked, k=int(k), per_layer_cap=args.per_layer_cap)
        heads_by_layer: Dict[int, List[int]] = {}
        for layer, head in selected:
            heads_by_layer.setdefault(int(layer), []).append(int(head))

        hooks = _install_knockout_hooks(model, heads_by_layer) if k > 0 else []
        try:
            prompts = [
                construct_text_from_messages(_construct_messages(dp), tokenizer, continue_generation=True) for dp in data
            ]
            completions = _generate(
                model,
                tokenizer,
                prompts,
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.max_new_tokens),
            )

            correct = 0
            n = len(data)
            per_item_out: List[Dict[str, Any]] = []
            for dp, comp in zip(data, completions):
                pred = _extract_letter(comp, letters="ABCDE")
                gold = str(dp.get("answerKey", "")).strip().upper()
                if pred is not None and pred.upper() == gold:
                    correct += 1
                if args.save_jsonl:
                    per_item_out.append(
                        {
                            "id": dp.get("id"),
                            "question": dp.get("question"),
                            "choices": dp.get("choices"),
                            "answerKey": gold,
                            "completion": comp,
                            "pred_letter": pred,
                            "is_correct": bool(pred is not None and pred.upper() == gold),
                        }
                    )

            acc = 100.0 * correct / max(1, n)
            rec = {
                "dataset": "csqa_validation",
                "model_path": args.model_path,
                "results_pt": args.results_pt,
                "k": int(k),
                "n": int(n),
                "correct": int(correct),
                "acc": acc,
                "per_layer_cap": args.per_layer_cap,
                "score_order": args.score_order,
            }
            results.append(rec)
            print(f"[DONE] k={k} n={n} acc={acc:.2f}%")

            if args.save_jsonl:
                k_dir = out_dir / f"k_{int(k)}"
                k_dir.mkdir(parents=True, exist_ok=True)
                out_path = k_dir / f"{model_name}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for row in per_item_out:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            for h in hooks:
                h.remove()

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
