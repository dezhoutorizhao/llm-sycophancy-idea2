#!/usr/bin/env bash
set -euo pipefail

# Train SPT with TOPK=32 using a patched dataset, then wait for GPUs idle and run
# sycophancy evaluation. Finally, print metrics for TOPK=64 (existing run) vs TOPK=32.
#
# Expected to run on the GPU server (Linux) where /root/shared-nvme exists.
#
# Minimal usage (edit env vars if needed):
#   bash scripts/train_topk32_and_eval_compare.sh
#
# Optional env vars:
#   REPO               Repo root (default: auto-detected from script location)
#   BASE_MODEL         Base model to finetune (default: /root/shared-nvme/llama3-exp)
#   DATA_PATH          Patched training jsonl path
#   PATH_PATCHING_PATH Directory containing results.pt for head selection
#   TOP64_SUMMARY      Existing TOP64 summary file to compare against
#   TOP64_EVAL_DIR     Or existing TOP64 eval dir containing sycophancy_eval_summary.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BASE_MODEL="${BASE_MODEL:-/root/shared-nvme/llama3-exp}"
JUDGE_MODEL="${JUDGE_MODEL:-$BASE_MODEL}"

DATA_PATH="${DATA_PATH:-$REPO/prepare_training_data/datasets/scyophancy_mixed_instruction_data_no_sorry_insist_finalanswer.jsonl}"
PATH_PATCHING_PATH="${PATH_PATCHING_PATH:-$REPO/path_patching/results_fused_v2/llama3-exp}"

TOPK=32

RUN_ID="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-fusedv2_top${TOPK}_nosorry_finalanswer_v1_${RUN_ID}}"

OUTPUT_DIR="${OUTPUT_DIR:-/root/shared-nvme/spt_llama3exp_${TAG}}"
CACHE_DIR="${CACHE_DIR:-/root/shared-nvme/spt_cache_llama3exp_${TAG}}"
LOGGING_DIR="${LOGGING_DIR:-$OUTPUT_DIR/tf_logs}"

EVAL_DIR="${EVAL_DIR:-/root/shared-nvme/eval_${TAG}}"

# Eval tunables (passed through to scripts/eval_sycophancy_when_idle.sh)
EVAL_TORCH_DTYPE="${EVAL_TORCH_DTYPE:-bfloat16}"
EVAL_TP="${EVAL_TP:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1}"
EVAL_UTIL_THRESH="${EVAL_UTIL_THRESH:-10}"
EVAL_SLEEP_SECS="${EVAL_SLEEP_SECS:-600}"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH not found: $DATA_PATH" >&2
  echo "Hint: generate it with prepare_training_data/07_patch_spt_training_data.py first." >&2
  exit 1
fi
if [[ ! -f "$PATH_PATCHING_PATH/results.pt" ]]; then
  echo "ERROR: PATH_PATCHING_PATH/results.pt not found: $PATH_PATCHING_PATH/results.pt" >&2
  exit 1
fi

echo "[CONFIG]"
echo "  REPO=$REPO"
echo "  BASE_MODEL=$BASE_MODEL"
echo "  JUDGE_MODEL=$JUDGE_MODEL"
echo "  DATA_PATH=$DATA_PATH"
echo "  PATH_PATCHING_PATH=$PATH_PATCHING_PATH"
echo "  TOPK=$TOPK"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  EVAL_DIR=$EVAL_DIR"
echo

export MODEL_PATH="$BASE_MODEL"
export DATA_PATH
export DATA_TYPE=instruction_tuning
export TRAIN_ON_PROMPT=False
# Label the last 2 assistant turns (challenge response + final answer) to align supervision.
export TRAIN_ON_LAST_N_ASSISTANT_MESSAGES="${TRAIN_ON_LAST_N_ASSISTANT_MESSAGES:-2}"
export MAX_SEQ_LEN=2048
export PADDING=False
export PADDING_SIDE=right

export PATH_PATCHING_PATH
export PRECISE_LEVEL=3
export TRAIN_TOPK="$TOPK"
export TRAIN_KV=False

export TORCH_DTYPE=bfloat16
export ATTN_IMPLEMENTATION=flash_attention_2
export DEEPSPEED=configs/configs_deepspeed/deepspeed_config_stage1.json

export MAX_STEPS=120
export LEARNING_RATE=3e-5
export MIN_LEARNING_RATE=1e-7
export LR_SCHEDULER_TYPE=polynomial
export WEIGHT_DECAY=0.1

export TRAIN_MICRO_BATCH_SIZE_PER_GPU=1
export TRAIN_GLOBAL_BATCH_SIZE=128
export NGPUS=2

export OUTPUT_DIR
export CACHE_DIR
export LOGGING_DIR

# Train
echo "[TRAIN] starting..."
pushd "$REPO/pinpoint_tuning" >/dev/null
bash scripts/run_train.sh
popd >/dev/null
echo "[TRAIN] done: $OUTPUT_DIR"
echo

# Eval (wait for GPUs idle first)
echo "[EVAL] waiting GPUs idle then running sycophancy eval..."
bash "$REPO/scripts/eval_sycophancy_when_idle.sh" \
  --model_path "$OUTPUT_DIR" \
  --run_dir "$EVAL_DIR" \
  --judge_model_path "$JUDGE_MODEL" \
  --repo "$REPO" \
  --torch_dtype "$EVAL_TORCH_DTYPE" \
  --tensor_parallel_size "$EVAL_TP" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --gpu_ids "$EVAL_GPU_IDS" \
  --util_thresh "$EVAL_UTIL_THRESH" \
  --sleep_secs "$EVAL_SLEEP_SECS"

# Compare summaries
TOP64_SUMMARY="${TOP64_SUMMARY:-}"
if [[ -z "$TOP64_SUMMARY" && -n "${TOP64_EVAL_DIR:-}" ]]; then
  TOP64_SUMMARY="$TOP64_EVAL_DIR/sycophancy_eval_summary.txt"
fi
if [[ -z "$TOP64_SUMMARY" ]]; then
  # Best-effort auto-detect (can be overridden by TOP64_SUMMARY)
  TOP64_SUMMARY="$(ls -1t /root/shared-nvme/eval_*top64*/sycophancy_eval_summary.txt 2>/dev/null | head -n1 || true)"
fi

TOP32_SUMMARY="$EVAL_DIR/sycophancy_eval_summary.txt"

echo
echo "=============================="
echo "[COMPARE] TOP64 vs TOP32"
echo "=============================="

if [[ -n "$TOP64_SUMMARY" && -f "$TOP64_SUMMARY" ]]; then
  echo
  echo "[TOP64] $TOP64_SUMMARY"
  grep -E "AI accuracy \\(before\\)|No match ratio \\(before\\)|AI sorry ratio|AI accuracy \\(after\\)|No match ratio \\(after\\)|Correct -> Incorrect ratio" "$TOP64_SUMMARY" || true
else
  echo
  echo "[TOP64] summary not found. Set TOP64_SUMMARY=/path/to/sycophancy_eval_summary.txt" >&2
fi

echo
echo "[TOP32] $TOP32_SUMMARY"
grep -E "AI accuracy \\(before\\)|No match ratio \\(before\\)|AI sorry ratio|AI accuracy \\(after\\)|No match ratio \\(after\\)|Correct -> Incorrect ratio" "$TOP32_SUMMARY" || true
