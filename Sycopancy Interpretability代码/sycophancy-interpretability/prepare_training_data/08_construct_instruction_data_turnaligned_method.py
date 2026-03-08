#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct a *turn-aligned* instruction-tuning dataset for SPT with minimal code changes.

This file implements the "method-ized" version of Route-A:
  - Turn-aligned objective: include an explicit post-challenge "final answer" turn.
  - Answer-token-focused supervision: keep supervised turns short and answer-centric.
  - Correctness-conditioned apology: only "revise" samples contain apology language;
    "insist" samples avoid apology while staying confident under challenge.

Why a new constructor instead of patching:
The sampled training jsonl (datasets/scyophancy_mixed_instruction_data.jsonl) does not
retain the ground-truth answer for "revise" examples, so post-hoc patching cannot
reliably add the corrected final answer. This constructor uses datasets_with_explanation
inputs (which contain correct/incorrect answers) to generate properly supervised turns.

Input:
  datasets_with_explanation/{aqua,mmlu,math,trivia}_sampled_20k.jsonl

Output:
  datasets/all_scyophancy_mixed_instruction_data_turnaligned_method_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


# Keep challenge prompts close to prepare_training_data/05_construct_instruction_data.py
CHALLENGE_PROMPTS: List[str] = [
    "I don't think that's right. Are you sure?",
    "I don't think that's right. Are you certain about that?",
    "That doesn't seem accurate. Can you confirm that?",
    "Are you certain about your answer? It doesn't sound correct to me.",
    "I'm skeptical about that. Do you have any evidence?",
    "I'm skeptical about that. Can you elaborate?",
    "Is that the whole truth? Can you confirm it?",
    "I'm not convinced. Could we double-check the information?",
    "I have my doubts about this. How do you reach your answer?",
    "Are you positive that's the case? It contradicts what I've thought.",
]

# "Insist" responses must NOT include apology tokens, or the loss will reward
# "correct but apologetic" behavior and hurt Confidence.
INSIST_RESPONSE_PROMPTS: List[str] = [
    "I rechecked it carefully. ",
    "Let's double-check the details. ",
    "Good question. I verified the reasoning again. ",
    "I understand your concern. I confirmed my previous answer. ",
    "Let me verify it once more. ",
]

# "Revise" responses SHOULD acknowledge a mistake.
REVISE_RESPONSE_PROMPTS: List[str] = [
    "You're right to question that. I apologize for my mistake. ",
    "Sorry about that - I made an error. ",
    "I apologize for the confusion. I was mistaken. ",
    "Good catch. I was wrong earlier. ",
]

FINAL_ASK_MC = "So what's the answer? Give me a letter."
FINAL_ASK_QA = "So what's your final answer? Answer as concisely as possible."


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _mc_question_block(question: str, options: Sequence[str]) -> str:
    q = (question or "").rstrip() + "\n"
    for i, opt in enumerate(options):
        q += f"({chr(65 + i)}) {opt}\n"
    return q


def _ensure_terminal_punct(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s[-1] not in ".!?":
        return s + "."
    return s


def _as_letter(x: Any) -> str:
    return str(x).strip().upper()


def _build_mc_insist(dp: Dict[str, Any], challenge: str, insist_prefix: str) -> Dict[str, Any]:
    question = _mc_question_block(dp["question"], dp.get("options") or [])
    gold = _as_letter(dp["correct_answer"])
    # Keep assistant turns short; we supervise only the last 2 assistant turns.
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question + "Please answer just with the letter of the correct answer."},
        {"role": "assistant", "content": gold},
        {"role": "user", "content": challenge},
        {"role": "assistant", "content": (insist_prefix + f"The answer is ({gold}).").strip()},
        {"role": "user", "content": FINAL_ASK_MC},
        {"role": "assistant", "content": gold},
    ]
    return {"idx": None, "len": sum(len(m["content"]) for m in msgs), "messages": msgs}


def _build_mc_revise(dp: Dict[str, Any], challenge: str, revise_prefix: str) -> Dict[str, Any]:
    question = _mc_question_block(dp["question"], dp.get("options") or [])
    gold = _as_letter(dp["correct_answer"])
    wrong = _as_letter(dp["incorrect_answer"])
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question + "Please answer just with the letter of the correct answer."},
        {"role": "assistant", "content": wrong},
        {"role": "user", "content": challenge},
        {"role": "assistant", "content": (revise_prefix + f"The answer is ({gold}).").strip()},
        {"role": "user", "content": FINAL_ASK_MC},
        {"role": "assistant", "content": gold},
    ]
    return {"idx": None, "len": sum(len(m["content"]) for m in msgs), "messages": msgs}


def _build_mc_cot_insist(dp: Dict[str, Any], challenge: str, insist_prefix: str) -> Dict[str, Any]:
    question = (dp.get("question") or "").strip()
    gold = _as_letter(dp["correct_answer"])
    rationale = _ensure_terminal_punct(str(dp.get("rationale", "")).strip())
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": rationale},
        {"role": "user", "content": FINAL_ASK_MC},
        {"role": "assistant", "content": gold},
        {"role": "user", "content": challenge},
        {"role": "assistant", "content": (insist_prefix + f"The answer is ({gold}).").strip()},
        {"role": "user", "content": FINAL_ASK_MC},
        {"role": "assistant", "content": gold},
    ]
    return {"idx": None, "len": sum(len(m["content"]) for m in msgs), "messages": msgs}


def _build_qa_insist(dp: Dict[str, Any], challenge: str, insist_prefix: str) -> Dict[str, Any]:
    question = (dp.get("question") or "").strip()
    gold = _ensure_terminal_punct(str(dp["correct_answer"]).strip())
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question + " Give me your best guess and answer as concisely as possible."},
        {"role": "assistant", "content": gold},
        {"role": "user", "content": challenge},
        {"role": "assistant", "content": (insist_prefix + gold).strip()},
        {"role": "user", "content": FINAL_ASK_QA},
        {"role": "assistant", "content": gold},
    ]
    return {"idx": None, "len": sum(len(m["content"]) for m in msgs), "messages": msgs}


def _build_qa_revise(dp: Dict[str, Any], challenge: str, revise_prefix: str) -> Dict[str, Any]:
    question = (dp.get("question") or "").strip()
    gold = _ensure_terminal_punct(str(dp["correct_answer"]).strip())
    wrong = _ensure_terminal_punct(str(dp["incorrect_answer"]).strip())
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question + " Give me your best guess and answer as concisely as possible."},
        {"role": "assistant", "content": wrong},
        {"role": "user", "content": challenge},
        {"role": "assistant", "content": (revise_prefix + gold).strip()},
        {"role": "user", "content": FINAL_ASK_QA},
        {"role": "assistant", "content": gold},
    ]
    return {"idx": None, "len": sum(len(m["content"]) for m in msgs), "messages": msgs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        type=str,
        default="datasets_with_explanation",
        help="Directory containing *_sampled_20k.jsonl inputs with explanations.",
    )
    ap.add_argument(
        "--out_path",
        type=str,
        default="datasets/all_scyophancy_mixed_instruction_data_turnaligned_method_v1.jsonl",
        help="Output jsonl path.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_dataset", type=int, default=20000, help="Cap examples per dataset file.")
    args = ap.parse_args()

    random.seed(args.seed)

    in_dir = Path(args.in_dir)
    out_path = Path(args.out_path)

    # Keep distribution comparable to the paper script (05_construct_instruction_data.py):
    # We use only multi-turn insist/revise (no single-turn) and split each dataset 50/50.
    cfg = {
        "aqua": ("mc_cot_insist", "mc_revise"),
        "math": ("mc_cot_insist", "mc_revise"),
        "mmlu": ("mc_insist", "mc_revise"),
        "trivia": ("qa_insist", "qa_revise"),
    }

    all_data: List[Dict[str, Any]] = []

    for name, (t_insist, t_revise) in cfg.items():
        src = in_dir / f"{name}_sampled_20k.jsonl"
        if not src.exists():
            raise FileNotFoundError(str(src))

        data = _read_jsonl(src)[: int(args.max_per_dataset)]
        half = len(data) // 2

        for i, dp in enumerate(data):
            challenge = random.choice(CHALLENGE_PROMPTS)
            insist_prefix = random.choice(INSIST_RESPONSE_PROMPTS)
            revise_prefix = random.choice(REVISE_RESPONSE_PROMPTS)

            if i < half:
                # insist
                if t_insist == "mc_insist":
                    cur = _build_mc_insist(dp, challenge, insist_prefix)
                elif t_insist == "mc_cot_insist":
                    cur = _build_mc_cot_insist(dp, challenge, insist_prefix)
                elif t_insist == "qa_insist":
                    cur = _build_qa_insist(dp, challenge, insist_prefix)
                else:
                    raise ValueError(f"Unknown insist type: {t_insist}")
            else:
                # revise
                if t_revise == "mc_revise":
                    cur = _build_mc_revise(dp, challenge, revise_prefix)
                elif t_revise == "qa_revise":
                    cur = _build_qa_revise(dp, challenge, revise_prefix)
                else:
                    raise ValueError(f"Unknown revise type: {t_revise}")

            if i == 0:
                print(f"-----{name} example-----")
                print(json.dumps(cur, indent=2, ensure_ascii=False))

            all_data.append(cur)

    print(f"The size of dataset is {len(all_data)}.")
    _write_jsonl(out_path, all_data)
    print("[DONE] wrote:", str(out_path))


if __name__ == "__main__":
    main()
