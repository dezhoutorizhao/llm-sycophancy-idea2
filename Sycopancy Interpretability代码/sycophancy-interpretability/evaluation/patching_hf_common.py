#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for AAAI'26-style activation patching baselines (HF/Transformers).

Design goals:
- Deterministic decoding (no sampling).
- Patch at *the last non-pad prompt token* only on the first decoding step.
  This matches the causal "decision point" spirit in AAAI'26 (logit preference shift),
  and avoids over-intervening during the entire continuation.
- Batchable (padding-aware) to speed up General Ability evaluation.

This module is intentionally self-contained to avoid importing vLLM-only helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_layers(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def construct_text_from_messages(messages, tokenizer, continue_generation: bool = True) -> str:
    add_generation_prompt = True if continue_generation and messages[-1]["role"] != "assistant" else False
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    if continue_generation and messages[-1]["role"] == "assistant":
        text = text.strip().rstrip(tokenizer.eos_token or "")
    return text


@dataclass
class HFModelBundle:
    model: torch.nn.Module
    tokenizer: object


def load_hf_model(model_path: str, torch_dtype: str = "float16", device_map: str = "cuda") -> HFModelBundle:
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[torch_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return HFModelBundle(model=model, tokenizer=tok)


def _last_nonpad_index(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: [B, T] of 0/1
    # last idx = sum(mask)-1
    lengths = attention_mask.long().sum(dim=1)
    return torch.clamp(lengths - 1, min=0)


@torch.no_grad()
def get_ref_hidden_last_token(
    model,
    tokenizer,
    ref_texts: Sequence[str],
    layers: Sequence[int],
    *,
    max_length: int | None = None,
) -> Dict[int, torch.Tensor]:
    """
    Returns a map: layer_idx -> hidden_state_at_last_prompt_token (shape [B, D]).
    """
    device = next(model.parameters()).device
    enc = tokenizer(
        list(ref_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # len = n_layers + 1
    last_idx = _last_nonpad_index(enc["attention_mask"])  # [B]

    ref: Dict[int, torch.Tensor] = {}
    for l in layers:
        # hs[l+1]: [B, T, D]
        h = hs[l + 1]
        ref[l] = h[torch.arange(h.shape[0], device=h.device), last_idx, :].clone()
    return ref, enc, last_idx


def make_layer_hooks_patch_last_token_first_step(
    model,
    ref_map: Dict[int, torch.Tensor],
    layers: Sequence[int],
    last_indices: torch.Tensor,
    prompt_seq_len: int,
):
    """
    Patch the layer output hidden state at each sample's last prompt token.
    Only applies on the first decoding step (sequence length == prompt_seq_len).
    """
    hooks = []
    layers_mod = model.model.layers

    def hook_fn(layer_idx: int):
        def _hook(module, inp, out):
            # LlamaDecoderLayer forward returns tuple: (hidden_states, ...)
            if isinstance(out, tuple):
                hs = out[0]
            else:
                hs = out

            # Only patch on the first step where full prompt is processed.
            if hs.dim() != 3 or hs.shape[1] != prompt_seq_len:
                return out

            if layer_idx not in ref_map:
                return out

            hs2 = hs.clone()
            bsz = hs2.shape[0]
            idx = last_indices.to(device=hs2.device)
            hs2[torch.arange(bsz, device=hs2.device), idx, :] = ref_map[layer_idx].to(
                device=hs2.device, dtype=hs2.dtype
            )

            if isinstance(out, tuple):
                return (hs2,) + out[1:]
            return hs2

        return _hook

    for l in layers:
        hooks.append(layers_mod[l].register_forward_hook(hook_fn(l)))
    return hooks


@torch.no_grad()
def generate_batch_with_patching(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    patch_layers: Sequence[int],
    max_new_tokens: int,
    ref_texts: Sequence[str] | None = None,
    max_length: int | None = None,
) -> List[str]:
    """
    Deterministic greedy generation with optional AAAI-style patching.
    Returns decoded continuations (including special tokens).
    """
    device = next(model.parameters()).device
    enc = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # prompt lengths for "first step" detection
    prompt_seq_len = enc["input_ids"].shape[1]

    hooks = []
    if patch_layers:
        if ref_texts is None:
            # Default: identity reference (measures the side-effect of enabling patching machinery).
            ref_texts = prompts
        ref_map, ref_enc, last_idx = get_ref_hidden_last_token(
            model, tokenizer, ref_texts, patch_layers, max_length=max_length
        )
        # Ensure prompt_seq_len matches (padding length) between prompt and ref.
        if ref_enc["input_ids"].shape[1] != prompt_seq_len:
            # If they differ (due to truncation differences), fall back to using prompt padding length.
            # This keeps the "first step" gate stable.
            prompt_seq_len = min(prompt_seq_len, ref_enc["input_ids"].shape[1])
        hooks = make_layer_hooks_patch_last_token_first_step(model, ref_map, patch_layers, last_idx, prompt_seq_len)

    try:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        for h in hooks:
            h.remove()

    # Decode per-sample continuation (strip the prompt prefix length per sample)
    # Use attention_mask lengths rather than padded length.
    last = _last_nonpad_index(enc["attention_mask"]) + 1
    texts: List[str] = []
    for i in range(out.shape[0]):
        gen = out[i, last[i].item() :]
        texts.append(tokenizer.decode(gen, skip_special_tokens=False))
    return texts

