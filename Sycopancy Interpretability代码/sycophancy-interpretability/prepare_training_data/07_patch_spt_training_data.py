#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch the SPT instruction-tuning dataset (jsonl) WITHOUT changing training code.

Goal:
1) Remove "apology / sorry" language from *insist* samples (i.e., cases where the
   model was correct and should stay confident under challenge), so the loss no
   longer rewards "correct but apologetic" behavior.
2) Make the "final answer letter" explicit in the *challenge-response* target
   for MC insist samples, so the model is trained to restate the same answer
   after being challenged.

This script:
- copies the input jsonl to a new file (to avoid confusion),
- edits the copy in-place,
- prints a small stats report.

It is intentionally conservative:
- only edits datapoints whose last assistant message contains the insist marker
  ("I'm sure that my previous answer is correct."),
- does NOT touch "revise" samples (where apologizing is appropriate).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


# Keep these in sync with prepare_training_data/05_construct_instruction_data.py
RESPONSE_PROMPT_PREFIXES = [
    "I apologize for the confusion. Let me double-check my answer. ",
    "I apologize for the confusion. Let's double-check my previous answer again to be absolutely certain. ",
    "Let's go through the facts once more to address any concerns. ",
    "Let me go over my previous answer once again to provide a precise response. ",
    "Good question. Let's review the information together to clear up any doubts. ",
    "My apologies if there's been a misunderstanding. Let me clarify that for you. ",
    "My apologies for not being clear enough. Let me explain my previous answer again. ",
    "I understand your concern. Allow me to verify the details of my previous answer. ",
    "Sorry if there's been a misunderstanding. I will recheck everything in detail. ",
    "Sorry for any ambiguity. Allow me to explain my answer further. ",
]

_RE_INSIST_MARKER = re.compile(r"I['\u2019]m sure that my previous answer is correct\.")

# A coarse filter for MC samples in this dataset.
MC_USER_HINTS = [
    "Please answer just with the letter of the correct answer.",
    "Give me a letter.",
]


_RE_PARENS_LETTER = re.compile(r"\(([A-Za-z])\)")
_RE_ANSWER_IS_LETTER = re.compile(r"(?i)\banswer\s+is\s+([A-Za-z])\b")

# Remove apology tokens if they remain after stripping the known prefixes.
_RE_APOLOGY_WORD = re.compile(r"(?i)\b(apolog(?:y|ies|ize|ise|ized|ised|izing|ising)|sorry)\b")


@dataclass
class Stats:
    total: int = 0
    insist_seen: int = 0
    insist_modified: int = 0
    apology_prefix_stripped: int = 0
    apology_words_removed: int = 0
    mc_answer_appended: int = 0
    mc_answer_missing: int = 0


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _is_insist_last_assistant(text: str) -> bool:
    return _RE_INSIST_MARKER.search(text) is not None


def _is_multiple_choice(dp: Dict[str, Any]) -> bool:
    msgs = dp.get("messages") or []
    for m in msgs:
        if m.get("role") != "user":
            continue
        c = str(m.get("content", ""))
        if any(hint in c for hint in MC_USER_HINTS):
            return True
    return False


def _extract_mc_letter_from_messages(dp: Dict[str, Any]) -> Optional[str]:
    msgs = dp.get("messages") or []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        t = str(m.get("content", ""))
        mo = _RE_PARENS_LETTER.search(t)
        if mo:
            return mo.group(1).upper()
        mo = _RE_ANSWER_IS_LETTER.search(t)
        if mo:
            return mo.group(1).upper()
    return None


def _strip_known_response_prefix(text: str) -> Tuple[str, bool]:
    s = text.lstrip()
    for p in RESPONSE_PROMPT_PREFIXES:
        if s.startswith(p):
            return s[len(p):].lstrip(), True
    return text, False


def _remove_apology_words(text: str) -> Tuple[str, int]:
    # Replace apology words with nothing; keep spacing readable.
    new_text, n = _RE_APOLOGY_WORD.subn("", text)
    # Normalize repeated spaces created by deletions.
    new_text = re.sub(r"[ \t]{2,}", " ", new_text)
    new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    return new_text.strip(), n


def _contains_mc_letter(text: str, letter: str) -> bool:
    # Avoid overfitting to a single template; accept either "(C)" or "answer is C".
    if re.search(rf"\(\s*{re.escape(letter)}\s*\)", text):
        return True
    if re.search(rf"(?i)\banswer\s+is\s+{re.escape(letter)}\b", text):
        return True
    return False


def _recompute_len(dp: Dict[str, Any]) -> None:
    msgs = dp.get("messages") or []
    try:
        dp["len"] = sum(len(str(m.get("content", ""))) for m in msgs)
    except Exception:
        # Be resilient; "len" is not required by training.
        pass


def patch_dataset(dp: Dict[str, Any], stats: Stats) -> Dict[str, Any]:
    stats.total += 1

    msgs = dp.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return dp
    if msgs[-1].get("role") != "assistant":
        return dp

    last = str(msgs[-1].get("content", ""))
    if not _is_insist_last_assistant(last):
        return dp

    stats.insist_seen += 1
    changed = False

    # 1) Remove apology-style prefix + any residual apology words.
    cleaned, stripped = _strip_known_response_prefix(last)
    if stripped:
        stats.apology_prefix_stripped += 1
        changed = True

    cleaned2, n_removed = _remove_apology_words(cleaned)
    if n_removed:
        stats.apology_words_removed += n_removed
        changed = True

    # 2) For MC insist samples, append the (correct) answer letter at the end.
    if _is_multiple_choice(dp):
        letter = _extract_mc_letter_from_messages(dp)
        if letter is None:
            stats.mc_answer_missing += 1
        else:
            if not _contains_mc_letter(cleaned2, letter):
                cleaned2 = cleaned2.rstrip()
                if cleaned2 and not cleaned2.endswith((".", "!", "?", "\n")):
                    cleaned2 += "."
                cleaned2 += f" The answer is ({letter})."
                stats.mc_answer_appended += 1
                changed = True

    if changed:
        msgs[-1]["content"] = cleaned2
        _recompute_len(dp)
        stats.insist_modified += 1

    return dp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Original training jsonl.")
    ap.add_argument(
        "--out_path",
        default=None,
        help="Path of the copied+patched jsonl. Default: add suffix to in_path.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite out_path if it exists.")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    if args.out_path is None:
        out_path = in_path.with_name(in_path.stem + "_no_sorry_insist_finalanswer.jsonl")
    else:
        out_path = Path(args.out_path)

    if out_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_path} already exists (use --overwrite).")
        out_path.unlink()

    # Step 1: copy to avoid touching the original.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(in_path, out_path)

    # Step 2: patch the copied file in-place via a temp file.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    stats = Stats()

    patched = []
    for dp in _read_jsonl(out_path):
        patched.append(patch_dataset(dp, stats))

    _write_jsonl(tmp_path, patched)
    tmp_path.replace(out_path)

    print("[DONE] Patched dataset written to:", str(out_path))
    print(
        "[STATS]",
        f"total={stats.total}",
        f"insist_seen={stats.insist_seen}",
        f"insist_modified={stats.insist_modified}",
        f"apology_prefix_stripped={stats.apology_prefix_stripped}",
        f"apology_words_removed={stats.apology_words_removed}",
        f"mc_answer_appended={stats.mc_answer_appended}",
        f"mc_answer_missing={stats.mc_answer_missing}",
    )


if __name__ == "__main__":
    main()
