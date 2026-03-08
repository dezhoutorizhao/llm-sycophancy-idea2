#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for Trigger-Conditioned Δ-head scoring (HF).

Design goals:
- Deterministic / no generation: score fixed positions via logits / teacher-forced logprobs.
- Efficient per-head attribution: gradient×activation at each layer's o_proj input, sliced by head.
- Minimal coupling: do not import vLLM; only torch/transformers/tqdm/stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def safe_model_max_length(tokenizer) -> Optional[int]:
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None:
        return None
    try:
        max_len = int(max_len)
    except Exception:
        return None
    # Some tokenizers set this to a sentinel value.
    if max_len > 1_000_000:
        return None
    return max_len


def build_chat_text(tokenizer, messages: Sequence[Mapping[str, str]], add_generation_prompt: bool) -> str:
    """
    Best-effort conversion of chat messages -> text, compatible with HF chat templates.
    Falls back to a simple role-tagged format if no template is available.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except TypeError:
            # Older transformers may not support add_generation_prompt.
            return tokenizer.apply_chat_template(list(messages), tokenize=False)

    # Fallback: simple transcript.
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("assistant:")
    return "\n".join(parts)


def load_model_and_tokenizer(
    model_path: str,
    dtype: str,
    device_map: str,
) -> Tuple[torch.nn.Module, Any, torch.device]:
    if hasattr(torch, dtype):
        torch_dtype = getattr(torch, dtype)
    else:
        raise ValueError(f"Invalid dtype: {dtype} (expected torch dtype name, e.g. float16/bfloat16/float32)")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    # Avoid kv-cache allocations; we do backprop.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    return model, tokenizer, model_device


def load_path_patching_config(repo_path_patching_dir: str, model_type: str) -> Dict[str, str]:
    cfg_path = os.path.join(repo_path_patching_dir, "configs", f"{model_type}.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def head_shape_from_config(model) -> Tuple[int, int, int]:
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    if hasattr(model.config, "head_dim"):
        head_dim = int(model.config.head_dim)
    else:
        head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    return num_layers, num_heads, head_dim


def attn_weight_fingerprint(model) -> Dict[str, Any]:
    """
    Save a cheap "fingerprint" (shape/dtype/sum/l2) of attention projection weights.

    This is intentionally lightweight (no full tensor dumps) while still detecting
    accidental in-place mutations during head scoring.
    """
    fp: Dict[str, Any] = {}
    for name, p in model.named_parameters():
        if not any(k in name for k in ("self_attn", "attn")):
            continue
        if not any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
            continue
        with torch.no_grad():
            t = p.detach()
            # Use float32 reductions for stability across dtypes/devices.
            s = t.float().sum().item()
            l2 = (t.float().pow(2).sum().sqrt()).item()
        fp[name] = {"shape": list(t.shape), "dtype": str(t.dtype), "sum": float(s), "l2": float(l2)}
    return fp


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


@dataclass
class OProjInputCapturer:
    """
    Captures (and retains grads for) the input to each layer's attention o_proj.

    By default, captures the last K positions of the sequence (K is set per forward).
    """

    model: torch.nn.Module
    module_output_name: str
    num_layers: int

    capture_k: int = 1  # number of final positions to slice: x[:, -capture_k:, :]
    _hooks: Optional[List[Any]] = None
    _acts: Optional[Dict[int, torch.Tensor]] = None

    def install(self) -> None:
        if self._hooks is not None:
            return
        self._hooks = []
        self._acts = {}

        for i in range(self.num_layers):
            name = self.module_output_name.format(i=i)
            module = self.model.get_submodule(name)

            def _make_hook(layer_idx: int):
                def hook(_module, inputs, _output):
                    x = inputs[0]
                    # Capture a view of last K positions to keep memory bounded.
                    xk = x[:, -self.capture_k :, :]
                    xk.retain_grad()
                    self._acts[layer_idx] = xk

                return hook

            self._hooks.append(module.register_forward_hook(_make_hook(i)))

    def remove(self) -> None:
        if not self._hooks:
            return
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = None
        self._acts = None

    def clear(self) -> None:
        if self._acts is not None:
            self._acts.clear()

    @property
    def acts(self) -> Dict[int, torch.Tensor]:
        if self._acts is None:
            raise RuntimeError("Capturer not installed")
        return self._acts


def compute_gradxact_scores(
    capturer: OProjInputCapturer,
    num_heads: int,
    head_dim: int,
    pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert captured o_proj inputs + grads into per-layer, per-head scalar scores.

    Args:
      capturer: installed OProjInputCapturer with acts containing tensors shaped [B, K, D]
      pos_mask: optional float mask shaped [B, K] selecting which positions contribute.
                If None, all captured positions are used.

    Returns:
      scores: [num_layers, num_heads] float32 tensor (batch-mean).
    """
    layers = sorted(capturer.acts.keys())
    if not layers:
        raise RuntimeError("No activations captured (did the model run forward?)")

    # Infer batch/K from first layer.
    first = capturer.acts[layers[0]]
    bsz, kpos, dim = first.shape
    if dim != num_heads * head_dim:
        raise ValueError(f"Unexpected o_proj input dim={dim}, expected num_heads*head_dim={num_heads*head_dim}")

    if pos_mask is None:
        pos_mask = torch.ones((bsz, kpos), device=first.device, dtype=torch.float32)
    else:
        pos_mask = pos_mask.to(device=first.device, dtype=torch.float32)
        if pos_mask.shape != (bsz, kpos):
            raise ValueError(f"pos_mask shape {tuple(pos_mask.shape)} != (B,K)={(bsz,kpos)}")

    # Normalize by number of selected positions per sample (avoid length bias).
    denom = pos_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
    pos_mask = pos_mask / denom  # per-sample average over selected positions

    out = torch.zeros((len(layers), num_heads), dtype=torch.float32)

    for li, layer in enumerate(layers):
        act = capturer.acts[layer]  # [B,K,D]
        grad = act.grad
        if grad is None:
            raise RuntimeError("Missing gradients on captured activations (did you call backward?)")
        # [B,K,H,HD]
        act4 = act.reshape(bsz, kpos, num_heads, head_dim)
        grad4 = grad.reshape(bsz, kpos, num_heads, head_dim)
        # grad×act -> sum over head_dim -> [B,K,H]
        gx = (grad4 * act4).sum(dim=-1).float()
        # Weighted mean over positions -> [B,H]
        gx = (gx * pos_mask[:, :, None]).sum(dim=1)
        out[li] = gx.mean(dim=0).detach().cpu()

    return out


def stable_args_seed(seed: int) -> None:
    # Ensure deterministic-ish behavior for scoring.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <this_dir>/results_delta_xx/<model_name>/",
    )

