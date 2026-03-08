#!/usr/bin/env bash
set -euo pipefail

# AAAI'26 baseline full pipeline under *your* evaluation protocol:
# - For each model, localize critical layers via AAAI'26 mechanistic analysis (logit-lens + KL).
# - Evaluate AAAI-style test-time activation patching on:
#   * Llama: General Ability (CSQA/GSM8K/StrategyQA)
#   * Qwen:  SycophancyEval (multi-round) + General Ability
#
# Assumptions on server paths:
#   AAAI repo:  /root/2026AAAI_When_Truth/LLM-sycophancy
#   Eval repo:  /root/sycint/evaluation
#   Models:     /root/shared-nvme/llama3-exp and /root/shared-nvme/Qwen
#
# Usage:
#   bash /root/sycint/evaluation/scripts/run_aaai26_baseline_full.sh
#
# Optional env overrides:
#   GPU=0 (default) or GPU=0,1 for multi-GPU sharding with HF device_map=auto
#   MAX_EXAMPLES=5000
#   DTYPE=float16|bfloat16

AAAI="${AAAI:-/root/2026AAAI_When_Truth/LLM-sycophancy}"
EVAL="${EVAL:-/root/sycint/evaluation}"

PY="${PY:-$AAAI/.venv/bin/python}"

GPU="${GPU:-0}"
DTYPE="${DTYPE:-float16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-5000}"

LLAMA_MODEL="${LLAMA_MODEL:-/root/shared-nvme/llama3-exp}"
QWEN_MODEL="${QWEN_MODEL:-/root/shared-nvme/Qwen}"

JUDGE_LLAMA="${JUDGE_LLAMA:-$LLAMA_MODEL}"
JUDGE_QWEN="${JUDGE_QWEN:-$QWEN_MODEL}"

# HF loading settings
DEVICE_MAP="${DEVICE_MAP:-cuda}" # set to auto for multi-GPU

log() { echo "[$(date +'%F %T')] $*"; }

latest_pkl() {
  local dir="$1"
  # Pick newest logit_all .pkl in dir (avoid mixing with other outputs).
  ls -1t "$dir"/*logit_all*.pkl 2>/dev/null | head -n 1
}

derive_layers_for_model() {
  local model_path="$1"
  local model_name
  model_name="$(basename "${model_path%/}")"

  log "== Localizing AAAI'26 critical layers for $model_name =="
  mkdir -p "$AAAI/output_inference/mmlu/plain" "$AAAI/output_inference/mmlu/opinion_only"

  (cd "$AAAI" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u experiments/mechanistic_analysis/run_syco_logit_cot.py \
    --model_name "$model_path" \
    --question_type plain \
    --input_filename lib/plain/mmlu_plain.pkl \
    --inference_mode logit_only \
    --inference_layer all \
    --torch_dtype "$DTYPE" \
    --device_map "$DEVICE_MAP" \
    --max_examples "$MAX_EXAMPLES")

  (cd "$AAAI" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u experiments/mechanistic_analysis/run_syco_logit_cot.py \
    --model_name "$model_path" \
    --question_type opinion_only \
    --input_filename lib/opinion_only/prefix/mmlu_opinion_only.pkl \
    --inference_mode logit_only \
    --inference_layer all \
    --torch_dtype "$DTYPE" \
    --device_map "$DEVICE_MAP" \
    --max_examples "$MAX_EXAMPLES")

  local plain_pkl opin_pkl
  plain_pkl="$(latest_pkl "$AAAI/output_inference/mmlu/plain")"
  opin_pkl="$(latest_pkl "$AAAI/output_inference/mmlu/opinion_only")"

  if [[ -z "${plain_pkl:-}" || -z "${opin_pkl:-}" ]]; then
    log "ERROR: cannot find logit_all pkl outputs under $AAAI/output_inference/mmlu/{plain,opinion_only}"
    exit 1
  fi

  log "plain_pkl=$plain_pkl"
  log "opinion_pkl=$opin_pkl"

  local layers
  layers="$(CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$EVAL/derive_patch_layers_aaai26.py" \
    --plain_pkl "$plain_pkl" --opinion_pkl "$opin_pkl" | head -n 1)"

  log "PATCH_LAYERS=$layers"
  echo "$layers"
}

eval_general_ability_hf_patching() {
  local model_path="$1"
  local layers="$2"
  local model_name
  model_name="$(basename "${model_path%/}")"

  log "== General Ability (HF+AAAI patching) for $model_name =="

  (cd "$EVAL" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluate_csqa_chat_patching_hf.py \
    --model_path "$model_path" \
    --data_path datasets/csqa_validation.jsonl \
    --output_dir "results/csqa_aaai26" \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --patch_layers "$layers" --batch_size 16 --max_new_tokens 16)

  (cd "$EVAL" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluate_gsm8k_chat_patching_hf.py \
    --model_path "$model_path" \
    --data_path datasets/gsm8k_test.jsonl \
    --output_dir "results/gsm8k_aaai26" \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --patch_layers "$layers" --batch_size 8 --max_new_tokens 256)

  (cd "$EVAL" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluate_strategyqa_chat_patching_hf.py \
    --model_path "$model_path" \
    --data_path datasets/strategyqa_test.json \
    --output_dir "results/strategyqa_aaai26" \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --patch_layers "$layers" --batch_size 16 --max_new_tokens 16)
}

eval_sycophancy_hf_patching() {
  local model_path="$1"
  local judge_path="$2"
  local layers="$3"
  local model_name
  model_name="$(basename "${model_path%/}")"

  log "== SycophancyEval (HF+AAAI patching) for $model_name =="

  rm -rf "results/sycophancy_eval/free_generation/$model_name" \
         "results/sycophancy_eval/multiple_choice/$model_name" \
         "results/sycophancy_eval/multiple_choice_cot/$model_name" \
         "results/sycophancy_eval_correctness/free_generation/$model_name" \
         "results/sycophancy_eval_apologies/free_generation/$model_name" \
         "results/sycophancy_eval_apologies/multiple_choice/$model_name" \
         "results/sycophancy_eval_apologies/multiple_choice_cot/$model_name" || true

  (cd "$EVAL" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluate_sycophancy_chat_patching_hf_aaai26.py \
    --model_path "$model_path" \
    --data_path datasets/sycophancy_eval/free_generation.jsonl \
    --output_dir "results/sycophancy_eval" \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --patch_layers "$layers" --neutral_challenge "Okay." --max_new_tokens 256)

  (cd "$EVAL" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluate_sycophancy_chat_patching_hf_aaai26.py \
    --model_path "$model_path" \
    --data_path datasets/sycophancy_eval/multiple_choice.jsonl \
    --output_dir "results/sycophancy_eval" \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --patch_layers "$layers" --neutral_challenge "Okay." --max_new_tokens 64)

  (cd "$EVAL" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluate_sycophancy_chat_patching_hf_aaai26.py \
    --model_path "$model_path" \
    --data_path datasets/sycophancy_eval/multiple_choice_cot.jsonl \
    --output_dir "results/sycophancy_eval" \
    --torch_dtype "$DTYPE" --device_map "$DEVICE_MAP" \
    --patch_layers "$layers" --neutral_challenge "Okay." --max_new_tokens 128)

  # Judge (vLLM). Uses tensor_parallel_size=1 by default; adjust if you run on multiple GPUs.
  (cd "$EVAL" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u score_answer_correctness.py \
    --model_path "$model_path" --eval_model_path "$judge_path" \
    --results_dir "results/sycophancy_eval/free_generation" \
    --tensor_parallel_size 1 --torch_dtype "$DTYPE" --batch_size 32)

  for ds in free_generation multiple_choice multiple_choice_cot; do
    (cd "$EVAL" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u score_answer_apologies.py \
      --model_path "$model_path" --eval_model_path "$judge_path" \
      --results_dir "results/sycophancy_eval/$ds" \
      --tensor_parallel_size 1 --torch_dtype "$DTYPE" --batch_size 32)
  done

  (cd "$EVAL" && "$PY" -u print_sycophancy_eval_results.py --model_path "$model_path")
}


main() {
  log "GPU=$GPU DTYPE=$DTYPE DEVICE_MAP=$DEVICE_MAP MAX_EXAMPLES=$MAX_EXAMPLES"

  # 1) Llama: derive layers + General Ability
  local llama_alias="${LLAMA_MODEL%/}__aaai26"
  if [[ ! -e "$llama_alias" ]]; then ln -s "$LLAMA_MODEL" "$llama_alias"; fi
  LLAMA_LAYERS="$(derive_layers_for_model "$LLAMA_MODEL")"
  eval_general_ability_hf_patching "$llama_alias" "$LLAMA_LAYERS"

  # 2) Qwen: derive layers + SycophancyEval + General Ability
  local qwen_alias="${QWEN_MODEL%/}__aaai26"
  if [[ ! -e "$qwen_alias" ]]; then ln -s "$QWEN_MODEL" "$qwen_alias"; fi
  QWEN_LAYERS="$(derive_layers_for_model "$QWEN_MODEL")"
  eval_sycophancy_hf_patching "$qwen_alias" "$JUDGE_QWEN" "$QWEN_LAYERS"
  eval_general_ability_hf_patching "$qwen_alias" "$QWEN_LAYERS"

  log "All done."
  log "AAAI baseline results:"
  log "  SycophancyEval: $EVAL/results/sycophancy_eval/*/*__aaai26/"
  log "  General Ability: $EVAL/results/{csqa_aaai26,gsm8k_aaai26,strategyqa_aaai26}/*.jsonl"
}

main "$@"
