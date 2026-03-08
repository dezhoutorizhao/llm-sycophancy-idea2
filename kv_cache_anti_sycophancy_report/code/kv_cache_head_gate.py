import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GateConfig:
    challenge_window: int = 64
    risk_threshold: float = 0.0
    gamma: float = 0.35
    topk_heads: int = 64


class RiskHeadGate:
    def __init__(self, head_scores: torch.Tensor, config: GateConfig):
        self.head_scores = head_scores.float()
        self.config = config
        flat = self.head_scores.flatten()
        k = min(config.topk_heads, flat.numel())
        idx = torch.topk(flat, k=k, largest=True).indices
        self.selected = set((int(i // self.head_scores.size(1)), int(i % self.head_scores.size(1))) for i in idx)

    def layer_head_gamma(self, layer_idx: int, head_idx: int, in_window: bool) -> float:
        if not in_window:
            return 1.0
        if (layer_idx, head_idx) not in self.selected:
            return 1.0
        score = float(self.head_scores[layer_idx, head_idx].item())
        return self.config.gamma if score >= self.config.risk_threshold else 1.0


def gate_value_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    gate: RiskHeadGate,
    step_idx: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    in_window = step_idx < gate.config.challenge_window
    new_past = []
    for l, (k_cache, v_cache) in enumerate(past_key_values):
        v = v_cache
        bsz, n_heads, seqlen, d_head = v.shape
        gammas = []
        for h in range(n_heads):
            gammas.append(gate.layer_head_gamma(l, h, in_window))
        g = torch.tensor(gammas, device=v.device, dtype=v.dtype).view(1, n_heads, 1, 1)
        v_new = v * g
        new_past.append((k_cache, v_new))
    return tuple(new_past)


def build_head_scores_from_results(results_pt: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "results" in results_pt:
        return results_pt["results"].float()
    if "R" in results_pt:
        return results_pt["R"].float()
    if "scores" in results_pt:
        return results_pt["scores"].float()
    raise ValueError("No valid score tensor key found in results dict")
