#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean-room CAUSM baseline utilities.

Core idea implemented here:
- Learn per-(layer, head) weights W (head reweighting).
- Apply Causal Activation Calibration (CAC): subtract lambda * |W_lh| * d_lh from the
  *pre-o_proj* concatenated head outputs, at the last prompt token, on the first decoding step.

This is designed to be architecture-compatible with Llama/Qwen-style HF models where:
  model.model.layers[i].self_attn.o_proj exists
and the input to o_proj is the concatenation of per-head outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
class HFBundle:
    model: torch.nn.Module
    tok: object


def load_hf(model_path: str, torch_dtype: str = "float16", device_map: str = "cuda") -> HFBundle:
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[torch_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Decoder-only models should use left padding for batched generation; right padding can
    # degrade quality due to position/attention alignment issues.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return HFBundle(model=model, tok=tok)


def _last_nonpad_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Return the index (0-based) of the last non-padding token in the *padded* sequence.

    Important: this must work for BOTH left-padding and right-padding. Using
    `sum(mask)-1` is only correct for right-padding. In this repo we use
    left-padding (tok.padding_side="left") for decoder-only models, so we
    compute the last 1-position robustly.
    """
    am = attention_mask.long()
    # Find last index where am == 1 for each row.
    # Works for both:
    # - right padding: 11110000 -> flip -> 00001111 -> argmax -> 4 -> last=3
    # - left  padding: 00001111 -> flip -> 11110000 -> argmax -> 0 -> last=T-1
    rev = torch.flip(am, dims=[1])
    first_one_from_end = rev.argmax(dim=1)
    has_any = am.sum(dim=1) > 0
    last = am.size(1) - 1 - first_one_from_end
    last = torch.where(has_any, last, torch.zeros_like(last))
    return last


@torch.no_grad()
def capture_o_proj_inputs(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int = 4096,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Runs a forward pass and captures, for each layer, the o_proj input at the
    last non-padding token ONLY:
      shape [B, hidden] where hidden = num_heads * head_dim.

    Returns:
      o_inps: list of length n_layers with tensors [B,H*D]
      attention_mask: [B,T]
      last_idx: [B] last non-pad index
    """
    device = next(model.parameters()).device
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Compute indices once so the hook can gather only the last non-pad token.
    # This avoids storing [B,T,*] activations per layer (which OOMs for long prompts).
    last_idx = _last_nonpad_index(enc["attention_mask"]).to(device=device)

    o_inps: List[torch.Tensor] = []
    hooks = []

    def make_hook():
        def _pre_hook(module, inp):
            # inp is a tuple; inp[0] is [B,T,hidden]
            x = inp[0]
            bsz = x.shape[0]
            idx = last_idx
            if idx.device != x.device:
                idx = idx.to(device=x.device)
            x_last = x[torch.arange(bsz, device=x.device), idx, :]  # [B, hidden]
            o_inps.append(x_last.detach().clone())
            return inp

        return _pre_hook

    # Register hooks in order of layers
    for layer in model.model.layers:
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook()))

    try:
        # IMPORTANT: call the base model to avoid allocating full-sequence logits
        # (e.g., Qwen2's lm_head over long prompts), which can OOM at later rounds.
        _ = model.model(**enc, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return o_inps, enc["attention_mask"], last_idx


def _reshape_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Reshape concatenated heads to [B, H, D] or [B, T, H, D].

    Inputs:
      x:
        - [B, hidden]  OR
        - [B, T, hidden]
      where hidden = num_heads * head_dim.
    """
    if x.dim() == 2:
        b, h = x.shape
        assert h % num_heads == 0
        d = h // num_heads
        return x.view(b, num_heads, d)
    if x.dim() == 3:
        b, t, h = x.shape
        assert h % num_heads == 0
        d = h // num_heads
        return x.view(b, t, num_heads, d)
    raise ValueError(f"unexpected shape for head reshape: {tuple(x.shape)}")


def make_causm_cac_hooks(
    model,
    *,
    W: torch.Tensor,
    d_list: List[torch.Tensor],
    layer_indices: Optional[Sequence[int]] = None,
    topk: int = 64,
    lamb: float = 1.0,
    prompt_len: int,
    last_idx: torch.Tensor,
):
    """
    Create pre-hooks on o_proj that apply CAC:
      x_h := x_h - lamb * |W_lh| * d_lh
    at the last prompt token, ONLY on the first decoding step (seq_len == prompt_len).

    Inputs:
      W: [L, H] head weights
      d_list: list length L, each [B, H, D] (direction for each head at last token)
      layer_indices: subset of layers to apply; default all layers
    """
    hooks = []
    layers = model.model.layers

    L, H = W.shape
    if layer_indices is None:
        layer_indices = list(range(L))

    # Precompute topk mask per layer from |W|
    absW = W.abs()
    topk = min(topk, H)
    top_idx = torch.topk(absW, k=topk, dim=1).indices  # [L, topk]

    def make_pre_hook(layer_id: int):
        def _hook(module, inp):
            x = inp[0]
            # only first step: full prompt processed
            if x.dim() != 3 or x.shape[1] != prompt_len:
                return inp
            bsz = x.shape[0]
            nh = H
            xh = _reshape_heads(x, nh)  # [B,T,H,D]
            # patch only at last token per sample
            idx = last_idx.to(device=x.device)

            # gather per-head direction for this layer: [B,H,D]
            d = d_list[layer_id].to(device=x.device, dtype=x.dtype)

            # build per-head scale: |W_lh|, but zero for non-topk heads
            w = absW[layer_id].to(device=x.device, dtype=x.dtype)  # [H]
            m = torch.zeros_like(w)
            m[top_idx[layer_id].to(device=x.device)] = 1.0
            w = w * m  # [H]

            # CAUSM head reweighting (W Att): apply bounded multiplicative scaling to selected heads.
            # We follow the paper's intent (head selection signal) while keeping the magnitude stable.
            scale = 1.0 + 0.5 * torch.tanh(W[layer_id].to(device=x.device, dtype=x.dtype))  # [H]
            scale = 1.0 + (scale - 1.0) * m  # only top-k heads are scaled
            for b in range(bsz):
                xh[b, idx[b], :, :] = xh[b, idx[b], :, :] * scale.view(1, -1, 1)

            # apply: xh[b, idx[b], h, :] -= lamb * w[h] * d[b,h,:]
            for b in range(bsz):
                xh[b, idx[b], :, :] = xh[b, idx[b], :, :] - (lamb * w.view(1, -1, 1) * d[b : b + 1, :, :])

            x2 = xh.reshape_as(x)
            return (x2,) + inp[1:]

        return _hook

    for l in layer_indices:
        hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(make_pre_hook(l)))
    return hooks


@torch.no_grad()
def greedy_generate_with_causm(
    model,
    tokenizer,
    prompt_texts: Sequence[str],
    *,
    ref_texts: Sequence[str],
    W: torch.Tensor,
    patch_layers: Optional[Sequence[int]] = None,
    topk: int = 64,
    lamb: float = 1.0,
    max_new_tokens: int = 64,
    max_length: int = 4096,
) -> List[str]:
    """
    For each prompt, compute per-layer per-head causal direction d = x(XP) - x(Xbar)
    at the last prompt token (pre-o_proj head outputs), then apply CAC during greedy generation.

    Returns: decoded continuations (skip_special_tokens=True).
    """
    device = next(model.parameters()).device

    # Encode prompt and ref; use prompt padding length for first-step detection.
    enc = tokenizer(
        list(prompt_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]
    last_idx = _last_nonpad_index(enc["attention_mask"])

    # Capture o_proj inputs for prompt and ref at the same padded length.
    # NOTE: prompt and ref texts can have different lengths. We must index each side using
    # its own last-nonpad token to avoid out-of-bounds indexing (which can surface as a
    # device-side assert on CUDA).
    o_prompt, _, last_idx_p = capture_o_proj_inputs(model, tokenizer, prompt_texts, max_length=max_length)
    o_ref, _, last_idx_r = capture_o_proj_inputs(model, tokenizer, ref_texts, max_length=max_length)

    # Build d_list: [B,H,D] for each layer, at last token positions.
    L = len(o_prompt)
    H = W.shape[1]
    d_list: List[torch.Tensor] = []
    for l in range(L):
        # o_prompt/o_ref already contain ONLY the last token: [B, hidden]
        xp_last = _reshape_heads(o_prompt[l], H)  # [B,H,D]
        xr_last = _reshape_heads(o_ref[l], H)
        d_list.append((xp_last - xr_last).detach())

    hooks = make_causm_cac_hooks(
        model,
        W=W.to(device=device),
        d_list=d_list,
        layer_indices=patch_layers,
        topk=topk,
        lamb=lamb,
        prompt_len=prompt_len,
        # Use the prompt-side last indices for deciding where to patch during generation.
        last_idx=last_idx_p,
    )
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

    # Decode continuation per sample using true lengths.
    conts: List[str] = []
    start = _last_nonpad_index(enc["attention_mask"]) + 1
    for i in range(out.shape[0]):
        gen = out[i, start[i].item() :]
        conts.append(tokenizer.decode(gen, skip_special_tokens=True))
    return conts
