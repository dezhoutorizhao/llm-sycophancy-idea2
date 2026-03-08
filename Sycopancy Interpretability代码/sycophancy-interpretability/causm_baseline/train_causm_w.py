#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CAUSM head-weight matrix W on labeled "General Ability" tasks with preference interventions.

This is a clean-room implementation inspired by CAUSM (ICLR 2025):
- Freeze backbone parameters.
- Learn per-(layer, head) weights W that reweight attention head outputs (pre-o_proj).
- Optimize a causal objective proxy:
    L = CE(Y | X_pref; W) + gamma * max_{X_bar in S} | CE(Y | X_pref; W) - CE(Y | X_bar; W) |
  where X_bar is an intervention on the preference statement (objective / subjective / random).

We intentionally keep this training lightweight and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from causm_common import HFBundle, construct_text_from_messages, load_hf

# Import few-shot prompts from the repo's evaluation utilities without modifying them.
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EVAL = os.path.join(_ROOT, "evaluation")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

from utils_fewshot import FEWSHOT_CSQA, FEWSHOT_STRATEGYQA  # type: ignore


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]


def read_strategyqa_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["examples"]


def _ce_loss_on_target(model, tok, prompt_text: str, target_text: str, max_length: int) -> torch.Tensor:
    """
    Teacher-forced negative log-likelihood over target_text tokens.
    """
    device = next(model.parameters()).device
    full = prompt_text + target_text
    enc = tok(full, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        # We need gradients w.r.t. W hooks only, so model params can remain frozen;
        # but we still require autograd through the forward for W.
        pass
    out = model(**enc, use_cache=False)
    logits = out.logits  # [B,T,V]
    # Compute loss only on target token positions.
    prompt_ids = tok(prompt_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)["input_ids"]
    t0 = prompt_ids.shape[1]
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = enc["input_ids"][:, 1:].contiguous()
    # positions in shift space that correspond to predicting target tokens:
    # predicting token at index i uses shift position i-1.
    start = max(t0 - 1, 0)
    end = shift_labels.shape[1]
    ce = nn.CrossEntropyLoss(reduction="none")
    losses = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view_as(shift_labels)
    tgt_losses = losses[:, start:end]
    return tgt_losses.mean()


def _mc_messages_csqa(dp: Dict) -> List[Dict]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for q, a in zip(FEWSHOT_CSQA[::2], FEWSHOT_CSQA[1::2]):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    prompt = "Q: " + dp["question"].strip() + "\nAnswer Choices:\n"
    for label, text in zip(dp["choices"]["label"], dp["choices"]["text"]):
        prompt += f"({label}) {text.strip()}\n"
    messages.append({"role": "user", "content": prompt})
    return messages


def _messages_gsm8k(dp: Dict) -> List[Dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": dp["question"]},
    ]


def _messages_strategyqa(dp: Dict) -> List[Dict]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for q, a in zip(FEWSHOT_STRATEGYQA[::2], FEWSHOT_STRATEGYQA[1::2]):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": "Q: " + dp["input"].strip()})
    return messages


def _pref_stmt_mc(wrong_letter: str) -> str:
    return f"\nI think the answer is ({wrong_letter})."


def _pref_stmt_yesno(wrong: str) -> str:
    return f"\nI think the correct answer is {wrong}."


INTERVENTIONS = {
    "objective": "\nPlease provide an objective response based only on facts.",
    "subjective": "\nPlease provide a subjective response that aligns with my opinion.",
    "random": "\nRandom.",
}


class HeadReweighting(nn.Module):
    """
    Learnable W that reweights pre-o_proj concatenated head outputs at all layers.

    Implemented via forward pre-hooks on each layer's self_attn.o_proj.
    """

    def __init__(self, model, num_layers: int, num_heads: int, init_scale: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        # W raw; we use tanh to keep scaling bounded.
        self.W = nn.Parameter(torch.zeros(num_layers, num_heads) + init_scale)
        self._hooks = []
        self._install(model)

    def _install(self, model):
        layers = model.model.layers

        def make_hook(layer_id: int):
            def _pre_hook(module, inp):
                x = inp[0]  # [B,T,hidden]
                b, t, hidden = x.shape
                h = self.num_heads
                d = hidden // h
                xh = x.view(b, t, h, d)
                scale = 1.0 + 0.5 * torch.tanh(self.W[layer_id]).to(device=x.device, dtype=x.dtype)  # [H]
                xh = xh * scale.view(1, 1, h, 1)
                x2 = xh.reshape_as(x)
                return (x2,) + inp[1:]

            return _pre_hook

        for l in range(self.num_layers):
            self._hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l)))

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", default="causm_baseline/outputs")
    ap.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="cuda", choices=["cuda", "auto"])
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csqa_path", default="evaluation/datasets/csqa_validation.jsonl")
    ap.add_argument("--gsm8k_path", default="evaluation/datasets/gsm8k_test.jsonl")
    ap.add_argument("--strategyqa_path", default="evaluation/datasets/strategyqa_test.json")
    args = ap.parse_args()

    set_seed(args.seed)

    bundle: HFBundle = load_hf(args.model_path, torch_dtype=args.torch_dtype, device_map=args.device_map)
    model, tok = bundle.model, bundle.tok

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad_(False)

    num_layers = int(getattr(model.config, "num_hidden_layers"))
    num_heads = int(getattr(model.config, "num_attention_heads"))

    reweighter = HeadReweighting(model, num_layers=num_layers, num_heads=num_heads)
    opt = AdamW([reweighter.W], lr=args.lr)

    # Build a mixed calibration set (labeled).
    # We subsample to keep runtime bounded and deterministic.
    csqa = read_jsonl(args.csqa_path)[: args.max_examples]
    gsm8k = read_jsonl(args.gsm8k_path)[: args.max_examples]
    strat = read_strategyqa_json(args.strategyqa_path)[: args.max_examples]

    examples = []
    for dp in csqa:
        messages = _mc_messages_csqa(dp)
        correct = dp["answerKey"]
        wrong = random.choice([x for x in dp["choices"]["label"] if x != correct])
        examples.append(("mc", messages, " " + correct, wrong))
    for dp in gsm8k:
        messages = _messages_gsm8k(dp)
        # Use the gold answer string as target; teacher-forced over it.
        # Preference statement uses an obviously wrong answer to induce "preference".
        examples.append(("gsm8k", messages, "\n" + dp["answer"].strip(), "0"))
    for dp in strat:
        messages = _messages_strategyqa(dp)
        gold = "yes" if dp["target_scores"]["Yes"] == 1 else "no"
        wrong = "no" if gold == "yes" else "yes"
        examples.append(("yn", messages, " " + gold, wrong))

    random.shuffle(examples)
    examples = examples[: args.max_examples]

    model_name = os.path.basename(args.model_path.rstrip("/"))
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "W.pt")

    pbar = tqdm(range(args.epochs), desc="CAUSM W training (epochs)")
    for _ in pbar:
        random.shuffle(examples)
        for kind, messages, target_text, wrong in tqdm(examples, desc="steps", leave=False):
            # Build preference prompt XP
            base_text = construct_text_from_messages(messages, tok, continue_generation=True)
            if kind == "mc":
                xp = base_text + _pref_stmt_mc(wrong)
            elif kind == "yn":
                xp = base_text + _pref_stmt_yesno(wrong)
            else:
                xp = base_text + "\nI think the answer is 0."

            # Evaluate candidate interventions X_bar and pick worst-case diff.
            losses = {}
            ce_xp = _ce_loss_on_target(model, tok, xp, target_text, args.max_length)
            for name, stmt in INTERVENTIONS.items():
                xb = base_text + stmt
                ce_xb = _ce_loss_on_target(model, tok, xb, target_text, args.max_length)
                losses[name] = ce_xb

            # Choose X_bar that maximizes |CE(XP)-CE(Xb)|
            diffs = {k: (ce_xp - v).abs() for k, v in losses.items()}
            worst = max(diffs, key=lambda k: float(diffs[k].detach().cpu()))
            reg = diffs[worst]

            loss = ce_xp + args.gamma * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        pbar.set_description(f"CAUSM W training (saved -> {out_path})")

    torch.save({"W": reweighter.W.detach().cpu()}, out_path)
    reweighter.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
