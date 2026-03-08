import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_cache_head_gate import GateConfig, RiskHeadGate, gate_value_cache, build_head_scores_from_results


def detect_challenge(text: str) -> bool:
    t = text.lower()
    keys = [
        "are you sure",
        "i don't think that's right",
        "你确定吗",
        "我觉得你说得不对",
    ]
    return any(k in t for k in keys)


@torch.no_grad()
def guarded_generate(
    model,
    tokenizer,
    prompt: str,
    gate: RiskHeadGate,
    max_new_tokens: int = 128,
):
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    past = None
    in_challenge = detect_challenge(prompt)
    for step in range(max_new_tokens):
        outputs = model(input_ids=out_ids[:, -1:] if past is not None else out_ids, attention_mask=attn, past_key_values=past, use_cache=True)
        logits = outputs.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        past = outputs.past_key_values
        if in_challenge:
            past = gate_value_cache(past, gate, step_idx=step)
        out_ids = torch.cat([out_ids, next_id], dim=1)
        attn = torch.cat([attn, torch.ones_like(next_id)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def load_gate(results_path: str, topk: int = 64, gamma: float = 0.35):
    data = torch.load(results_path, map_location="cpu")
    scores = build_head_scores_from_results(data)
    cfg = GateConfig(topk_heads=topk, gamma=gamma)
    return RiskHeadGate(scores, cfg)


def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gate = load_gate("results.pt", topk=64, gamma=0.35)
    prompt = "Q: 2+2=?\nA: 4\nUser: I don't think that's right. Are you sure?"
    text = guarded_generate(model, tokenizer, prompt, gate, max_new_tokens=64)
    print(text)


if __name__ == "__main__":
    main()
