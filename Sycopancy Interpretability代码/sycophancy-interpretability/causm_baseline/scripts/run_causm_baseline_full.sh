#!/usr/bin/env bash
set -euo pipefail

# End-to-end CAUSM baseline pipeline (clean-room reimplementation).
#
# This script:
#  1) Trains CAUSM head weights W for each target model (frozen backbone).
#  2) Evaluates:
#     - Llama: General Ability under CAUSM (W reweighting, CAC disabled on GA)
#     - Qwen : SycophancyEval under CAUSM (CAC on challenge rounds) + General Ability
#
# It writes outputs to the standard evaluation folders, but disambiguates by model suffix "__causm".
#
# Usage (on server):
#   bash /root/sycint/causm_baseline/scripts/run_causm_baseline_full.sh
#
# Env overrides:
#   GPU=0 / GPU=0,1
#   DTYPE=float16|bfloat16
#   DEVICE_MAP=cuda|auto
#   MAX_EXAMPLES=500
#   TOPK=64
#   LAMB=1.0

ROOT="${ROOT:-/root/sycint}"
AAAI_VENV_PY="${AAAI_VENV_PY:-python3}"
PY_BIN="${AAAI_VENV_PY:-python3}"
if [ ! -x "$PY_BIN" ]; then PY_BIN=python3; fi
PY_BIN="$PY_BIN"
if [ ! -x "$PY_BIN" ]; then PY_BIN=python3; fi

GPU="${GPU:-0}"
DTYPE="${DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
TOPK="${TOPK:-64}"
LAMB="${LAMB:-1.0}"

LLAMA_MODEL="${LLAMA_MODEL:-/root/shared-nvme/llama3-exp}"
QWEN_MODEL="${QWEN_MODEL:-/root/shared-nvme/Qwen}"

JUDGE_QWEN="${JUDGE_QWEN:-$QWEN_MODEL}"

log() { echo "[$(date +'%F %T')] $*" >&2; }

cd "$ROOT"

# Create stable aliases to avoid mixing with base outputs.
if [[ ! -e "${LLAMA_MODEL}__causm" ]]; then ln -s "$LLAMA_MODEL" "${LLAMA_MODEL}__causm"; fi
if [[ ! -e "${QWEN_MODEL}__causm" ]]; then ln -s "$QWEN_MODEL" "${QWEN_MODEL}__causm"; fi

LLAMA_ALIAS="${LLAMA_MODEL}__causm"
QWEN_ALIAS="${QWEN_MODEL}__causm"

mkdir -p causm_baseline/outputs

train_one() {
  local model_path="$1"
  local model_name
  model_name="$(basename "${model_path%/}")"
  local out="causm_baseline/outputs/${model_name}/W.pt"
  if [[ -f "$out" ]]; then
    log "Found existing W: $out (skip training)"
    echo "$out"
    return
  fi
  log "Training CAUSM W for $model_name (max_examples=$MAX_EXAMPLES)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_BIN" -u causm_baseline/train_causm_w.py \
    --model_path "$model_path" \
    --output_dir causm_baseline/outputs \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --max_examples "$MAX_EXAMPLES" --max_length "$MAX_LENGTH"
  echo "$out"
}

eval_ga_one() {
  local model_path="$1"
  local w_path="$2"
  log "General Ability (CAUSM) for $(basename "${model_path%/}")"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_BIN" -u causm_baseline/run_causm_eval_general_ability.py \
    --model_path "$model_path" \
    --w_path "$w_path" \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --topk "$TOPK" --lamb 0.0 \
    --csqa_path evaluation/datasets/csqa_validation.jsonl \
    --gsm8k_path evaluation/datasets/gsm8k_test.jsonl \
    --strategyqa_path evaluation/datasets/strategyqa_test.json \
    --out_dir evaluation/results
}

eval_syc_one() {
  local model_path="$1"
  local w_path="$2"
  local patch_layers="${3:-}" # optional subset
  log "SycophancyEval (CAUSM) for $(basename "${model_path%/}")"
  cd evaluation
  local model_name
  model_name="$(basename "${model_path%/}")"

  rm -rf "results/sycophancy_eval/free_generation/$model_name" \
         "results/sycophancy_eval/multiple_choice/$model_name" \
         "results/sycophancy_eval/multiple_choice_cot/$model_name" \
         "results/sycophancy_eval_correctness/free_generation/$model_name" \
         "results/sycophancy_eval_apologies/free_generation/$model_name" \
         "results/sycophancy_eval_apologies/multiple_choice/$model_name" \
         "results/sycophancy_eval_apologies/multiple_choice_cot/$model_name" || true

  CUDA_VISIBLE_DEVICES="$GPU" "$PY_BIN" -u ../causm_baseline/run_causm_eval_sycophancy.py \
    --model_path "$model_path" \
    --w_path "$w_path" \
    --data_path datasets/sycophancy_eval/free_generation.jsonl \
    --output_dir results/sycophancy_eval \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --patch_layers "$patch_layers" --topk "$TOPK" --lamb "$LAMB" \
    --max_new_tokens 256

  CUDA_VISIBLE_DEVICES="$GPU" "$PY_BIN" -u ../causm_baseline/run_causm_eval_sycophancy.py \
    --model_path "$model_path" \
    --w_path "$w_path" \
    --data_path datasets/sycophancy_eval/multiple_choice.jsonl \
    --output_dir results/sycophancy_eval \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --patch_layers "$patch_layers" --topk "$TOPK" --lamb "$LAMB" \
    --max_new_tokens 16

  CUDA_VISIBLE_DEVICES="$GPU" "$PY_BIN" -u ../causm_baseline/run_causm_eval_sycophancy.py \
    --model_path "$model_path" \
    --w_path "$w_path" \
    --data_path datasets/sycophancy_eval/multiple_choice_cot.jsonl \
    --output_dir results/sycophancy_eval \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --patch_layers "$patch_layers" --topk "$TOPK" --lamb "$LAMB" \
    --max_new_tokens 32

  # judge + summary (uses existing scripts)
  CUDA_VISIBLE_DEVICES="$GPU" "$PY_BIN" -u score_answer_correctness.py \
    --model_path "$model_name" \
    --eval_model_path "$JUDGE_QWEN" \
    --results_dir results/sycophancy_eval/free_generation \
    --tensor_parallel_size 1 --torch_dtype "$DTYPE" --batch_size 32

  for ds in free_generation multiple_choice multiple_choice_cot; do
    CUDA_VISIBLE_DEVICES="$GPU" "$PY_BIN" -u score_answer_apologies.py \
      --model_path "$model_name" \
      --eval_model_path "$JUDGE_QWEN" \
      --results_dir "results/sycophancy_eval/$ds" \
      --tensor_parallel_size 1 --torch_dtype "$DTYPE" --batch_size 32
  done

  "$PY_BIN" -u print_sycophancy_eval_results.py --model_path "$model_name"
  cd ..
}

main() {
  log "GPU=$GPU DTYPE=$DTYPE DEVICE_MAP=$DEVICE_MAP MAX_EXAMPLES=$MAX_EXAMPLES TOPK=$TOPK LAMB=$LAMB"
  log "LLAMA_MODEL=$LLAMA_MODEL"
  log "QWEN_MODEL=$QWEN_MODEL"

  # 1) Llama: train + GA
  LLAMA_W="$(train_one "$LLAMA_ALIAS")"
  eval_ga_one "$LLAMA_ALIAS" "$LLAMA_W"

  # 2) Qwen: train + sycophancy + GA
  QWEN_W="$(train_one "$QWEN_ALIAS")"
  eval_syc_one "$QWEN_ALIAS" "$QWEN_W"
  eval_ga_one "$QWEN_ALIAS" "$QWEN_W"

  log "Done. Outputs:"
  log "  W: $ROOT/causm_baseline/outputs/*/W.pt"
  log "  SycophancyEval: $ROOT/evaluation/results/sycophancy_eval/*/*__causm/round_*.jsonl"
  log "  GA: $ROOT/evaluation/results/{csqa_causm,gsm8k_causm,strategyqa_causm}/*__causm.jsonl"
}

main "$@"
