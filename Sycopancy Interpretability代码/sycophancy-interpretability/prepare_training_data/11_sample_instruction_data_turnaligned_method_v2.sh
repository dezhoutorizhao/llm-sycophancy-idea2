#!/usr/bin/env bash
set -euo pipefail

# Sample a subset for SPT training from the Method v2 turn-aligned dataset.
#
# Usage:
#   bash prepare_training_data/11_sample_instruction_data_turnaligned_method_v2.sh \
#     datasets/all_scyophancy_mixed_instruction_data_turnaligned_method_v2.jsonl \
#     datasets/scyophancy_mixed_instruction_data_turnaligned_method_v2.jsonl \
#     3840

IN_PATH="${1:-datasets/all_scyophancy_mixed_instruction_data_turnaligned_method_v2.jsonl}"
OUT_PATH="${2:-datasets/scyophancy_mixed_instruction_data_turnaligned_method_v2.jsonl}"
N="${3:-3840}"

mkdir -p "$(dirname "$OUT_PATH")"

if ! command -v shuf >/dev/null 2>&1; then
  echo "ERROR: shuf not found (need GNU coreutils)." >&2
  exit 1
fi

shuf -n "$N" "$IN_PATH" > "$OUT_PATH"
echo "[DONE] wrote $OUT_PATH (n=$N) from $IN_PATH"

