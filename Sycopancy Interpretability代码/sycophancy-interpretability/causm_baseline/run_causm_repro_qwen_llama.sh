#!/usr/bin/env bash
set -euo pipefail

# Reproduce CAUSM baseline results:
# - Sycophancy metrics for Qwen + Llama (full SycophancyEval: free_generation / multiple_choice / multiple_choice_cot)
# - General Ability metrics for Qwen (CSQA / GSM8K / StrategyQA)
#
# Run location: this script is intended to be executed from within this folder:
#   /root/Sycopancy Interpretability代码/sycophancy-interpretability/causm_baseline
#
# Notes for rigor:
# - We run deterministic (greedy) generation.
# - We keep all per-round jsonl outputs for audit (saved under evaluation/results/...).
# - For judge scoring, we use a fixed judge model path (override via $JUDGE_MODEL).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${REPO_ROOT}/evaluation"

PY_BIN="${PY_BIN:-python3}"
GPU_GEN="${GPU_GEN:-0,1}"
GPU_JUDGE="${GPU_JUDGE:-0,1}"

#
# Base models (weights). For output tagging / avoiding mixing with base runs, you can
# optionally evaluate via a symlink alias (e.g. /root/shared-nvme/Qwen__causm) which points
# to the same checkpoint directory.
LLAMA_MODEL="${LLAMA_MODEL:-/root/shared-nvme/llama3-exp}"
QWEN_MODEL="${QWEN_MODEL:-/root/shared-nvme/Qwen}"

# Model paths actually used for evaluation (default: the base paths above).
LLAMA_EVAL_MODEL="${LLAMA_EVAL_MODEL:-$LLAMA_MODEL}"
QWEN_EVAL_MODEL="${QWEN_EVAL_MODEL:-$QWEN_MODEL}"

# Model paths used for training W (default: prefer existing __causm aliases if present).
LLAMA_W_MODEL="${LLAMA_W_MODEL:-/root/shared-nvme/llama3-exp__causm}"
QWEN_W_MODEL="${QWEN_W_MODEL:-/root/shared-nvme/Qwen__causm}"

# CAUSM W checkpoints (learned head weights)
LLAMA_W="${LLAMA_W:-${REPO_ROOT}/causm_baseline/outputs/llama3-exp__causm/W.pt}"
QWEN_W="${QWEN_W:-${REPO_ROOT}/causm_baseline/outputs/Qwen__causm/W.pt}"

# Judge model for scoring sycophancy outputs
JUDGE_MODEL="${JUDGE_MODEL:-/root/shared-nvme/llama3-exp}"

# CAUSM inference hyperparams
TOPK="${TOPK:-64}"
LAMB="${LAMB:-1.0}"

# CAUSM W training hyperparams (only used if W.pt is missing)
W_MAX_EXAMPLES="${W_MAX_EXAMPLES:-1500}"
W_MAX_LENGTH="${W_MAX_LENGTH:-2048}"
W_EPOCHS="${W_EPOCHS:-1}"
W_LR="${W_LR:-5e-3}"
W_GAMMA="${W_GAMMA:-1.0}"
W_SEED="${W_SEED:-42}"

# General Ability config (Qwen only)
GA_MAX_LENGTH="${GA_MAX_LENGTH:-2048}"      # input truncation cap
GA_MAX_NEW_TOKENS="${GA_MAX_NEW_TOKENS:-300}"  # output cap (per task)
GA_BATCH_SIZE="${GA_BATCH_SIZE:-12}"

# Sycophancy generation caps
SYCO_MAX_LENGTH="${SYCO_MAX_LENGTH:-4096}"
SYCO_MAX_NEW_FREE="${SYCO_MAX_NEW_FREE:-256}"
SYCO_MAX_NEW_MC="${SYCO_MAX_NEW_MC:-32}"
SYCO_MAX_NEW_MCCOT="${SYCO_MAX_NEW_MCCOT:-128}"

# Judge batching
JUDGE_TP="${JUDGE_TP:-1}"
JUDGE_DTYPE="${JUDGE_DTYPE:-float16}"
JUDGE_BS="${JUDGE_BS:-32}"

log() { echo "[$(date +'%F %T')] $*" >&2; }

require_file() {
  local p="$1"
  if [ ! -f "$p" ]; then
    log "Missing required file: $p"
    return 1
  fi
}

train_w_if_missing() {
  local model_path="$1"
  local out_dir="$2"
  local w_path="$3"
  local tag="$4"
  if [ -f "$w_path" ]; then
    log "Found W.pt for $tag: $w_path"
    return 0
  fi
  log "W.pt not found for $tag; training CAUSM W (this can take a while)."
  CUDA_VISIBLE_DEVICES="$GPU_GEN" "$PY_BIN" -u "${REPO_ROOT}/causm_baseline/train_causm_w.py" \
    --model_path "$model_path" \
    --output_dir "$out_dir" \
    --torch_dtype float16 --device_map auto \
    --max_length "$W_MAX_LENGTH" \
    --epochs "$W_EPOCHS" --lr "$W_LR" --gamma "$W_GAMMA" --seed "$W_SEED" \
    --max_examples "$W_MAX_EXAMPLES"
  require_file "$w_path"
}

run_sycophancy_for_model() {
  local model_path="$1"
  local w_path="$2"
  local tag="$3"

  require_file "$w_path"
  log "SycophancyEval (CAUSM) for $tag"

  cd "$EVAL_DIR"

  # Keep results under evaluation/results/..., compatible with judge scripts.
  # Remove only this model's prior outputs to avoid skipping.
  local model_name
  model_name="$(basename "${model_path%/}")"
  rm -rf "results/sycophancy_eval/free_generation/${model_name}" \
         "results/sycophancy_eval/multiple_choice/${model_name}" \
         "results/sycophancy_eval/multiple_choice_cot/${model_name}" \
         "results/sycophancy_eval_correctness/free_generation/${model_name}" \
         "results/sycophancy_eval_apologies/free_generation/${model_name}" \
         "results/sycophancy_eval_apologies/multiple_choice/${model_name}" \
         "results/sycophancy_eval_apologies/multiple_choice_cot/${model_name}" || true

  CUDA_VISIBLE_DEVICES="$GPU_GEN" "$PY_BIN" -u "${REPO_ROOT}/causm_baseline/run_causm_eval_sycophancy.py" \
    --model_path "$model_path" \
    --w_path "$w_path" \
    --data_path "datasets/sycophancy_eval/free_generation.jsonl" \
    --output_dir "results/sycophancy_eval" \
    --torch_dtype float16 --device_map auto \
    --topk "$TOPK" --lamb "$LAMB" \
    --max_length "$SYCO_MAX_LENGTH" --max_new_tokens "$SYCO_MAX_NEW_FREE" --print_every 50

  CUDA_VISIBLE_DEVICES="$GPU_GEN" "$PY_BIN" -u "${REPO_ROOT}/causm_baseline/run_causm_eval_sycophancy.py" \
    --model_path "$model_path" \
    --w_path "$w_path" \
    --data_path "datasets/sycophancy_eval/multiple_choice.jsonl" \
    --output_dir "results/sycophancy_eval" \
    --torch_dtype float16 --device_map auto \
    --topk "$TOPK" --lamb "$LAMB" \
    --max_length "$SYCO_MAX_LENGTH" --max_new_tokens "$SYCO_MAX_NEW_MC" --print_every 100

  CUDA_VISIBLE_DEVICES="$GPU_GEN" "$PY_BIN" -u "${REPO_ROOT}/causm_baseline/run_causm_eval_sycophancy.py" \
    --model_path "$model_path" \
    --w_path "$w_path" \
    --data_path "datasets/sycophancy_eval/multiple_choice_cot.jsonl" \
    --output_dir "results/sycophancy_eval" \
    --torch_dtype float16 --device_map auto \
    --topk "$TOPK" --lamb "$LAMB" \
    --max_length "$SYCO_MAX_LENGTH" --max_new_tokens "$SYCO_MAX_NEW_MCCOT" --print_every 100

  # Judge scoring
  CUDA_VISIBLE_DEVICES="$GPU_JUDGE" "$PY_BIN" -u score_answer_correctness.py \
    --model_path "$model_path" --eval_model_path "$JUDGE_MODEL" \
    --results_dir results/sycophancy_eval/free_generation \
    --tensor_parallel_size "$JUDGE_TP" --torch_dtype "$JUDGE_DTYPE" --batch_size "$JUDGE_BS"

  for ds in free_generation multiple_choice multiple_choice_cot; do
    CUDA_VISIBLE_DEVICES="$GPU_JUDGE" "$PY_BIN" -u score_answer_apologies.py \
      --model_path "$model_path" --eval_model_path "$JUDGE_MODEL" \
      --results_dir "results/sycophancy_eval/$ds" \
      --tensor_parallel_size "$JUDGE_TP" --torch_dtype "$JUDGE_DTYPE" --batch_size "$JUDGE_BS"
  done

  "$PY_BIN" -u print_sycophancy_eval_results.py --model_path "$model_path"
}

run_general_ability_qwen() {
  local model_path="$1"
  local w_path="$2"
  local tag="$3"

  require_file "$w_path"
  log "General Ability (CAUSM; Qwen) for $tag"

  cd "$REPO_ROOT"

  # Use the selectable-task runner to avoid running unintended tasks.
  CUDA_VISIBLE_DEVICES="$GPU_GEN" "$PY_BIN" -u "${REPO_ROOT}/causm_baseline/run_causm_eval_general_ability_select.py" \
    --tasks csqa,gsm8k,strategyqa \
    --model_path "$model_path" --w_path "$w_path" \
    --torch_dtype float16 --device_map auto \
    --topk "$TOPK" --lamb 0.0 \
    --batch_size "$GA_BATCH_SIZE" --max_length "$GA_MAX_LENGTH" \
    --max_new_tokens_csqa "$GA_MAX_NEW_TOKENS" \
    --max_new_tokens_gsm8k "$GA_MAX_NEW_TOKENS" \
    --max_new_tokens_strategyqa "$GA_MAX_NEW_TOKENS" \
    --csqa_path evaluation/datasets/csqa_validation.jsonl \
    --gsm8k_path evaluation/datasets/gsm8k_test.jsonl \
    --strategyqa_path evaluation/datasets/strategyqa_test.json \
    --out_dir evaluation/results
}

main() {
  export TOKENIZERS_PARALLELISM=false
  export PYTHONHASHSEED=0

  log "Repo: $REPO_ROOT"
  log "PY_BIN=$PY_BIN"
  log "GPU_GEN=$GPU_GEN GPU_JUDGE=$GPU_JUDGE"
  log "TOPK=$TOPK LAMB=$LAMB"
  log "LLAMA_MODEL=$LLAMA_MODEL"
  log "QWEN_MODEL=$QWEN_MODEL"
  log "LLAMA_EVAL_MODEL=$LLAMA_EVAL_MODEL"
  log "QWEN_EVAL_MODEL=$QWEN_EVAL_MODEL"
  log "LLAMA_W_MODEL=$LLAMA_W_MODEL"
  log "QWEN_W_MODEL=$QWEN_W_MODEL"
  log "LLAMA_W=$LLAMA_W"
  log "QWEN_W=$QWEN_W"
  log "JUDGE_MODEL=$JUDGE_MODEL"
  log "W_MAX_EXAMPLES=$W_MAX_EXAMPLES W_MAX_LENGTH=$W_MAX_LENGTH W_EPOCHS=$W_EPOCHS W_LR=$W_LR W_GAMMA=$W_GAMMA W_SEED=$W_SEED"

  # 0) Ensure W checkpoints exist (train if missing). Qwen first.
  # If you want to train W under a stable tag directory, keep QWEN_W_MODEL pointing to a
  # symlink alias (e.g. /root/shared-nvme/Qwen__causm) that resolves to the base weights.
  train_w_if_missing "$QWEN_W_MODEL" "${REPO_ROOT}/causm_baseline/outputs" "$QWEN_W" "Qwen"

  # 1) Qwen sycophancy (CAUSM) + judged metrics
  run_sycophancy_for_model "$QWEN_EVAL_MODEL" "$QWEN_W" "Qwen"

  # 2) Qwen General Ability (CAUSM)
  run_general_ability_qwen "$QWEN_EVAL_MODEL" "$QWEN_W" "Qwen"

  # 3) Llama W (if needed)
  train_w_if_missing "$LLAMA_W_MODEL" "${REPO_ROOT}/causm_baseline/outputs" "$LLAMA_W" "Llama"

  # 4) Llama sycophancy (CAUSM) + judged metrics
  run_sycophancy_for_model "$LLAMA_EVAL_MODEL" "$LLAMA_W" "Llama"

  log "All done."
}

main "$@"
