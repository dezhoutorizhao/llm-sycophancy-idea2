#!/usr/bin/env bash
set -euo pipefail

# Run sycophancy evaluation end-to-end, but only when specified GPUs are idle.
#
# What it does (sequentially):
# 1) Wait until all GPUs' utilization <= threshold.
# 2) Generate multi-round conversations for:
#    - free_generation, multiple_choice, multiple_choice_cot
# 3) Judge:
#    - correctness (free_generation rounds 1 & 3)
#    - apologies  (free_generation round 2, multiple_choice round 2, mc_cot round 3)
# 4) Print aggregated metrics (paper-style) into sycophancy_eval_summary.txt
#
# Usage:
#   bash scripts/eval_sycophancy_when_idle.sh \
#     --model_path /path/to/finetuned_model \
#     --run_dir /root/shared-nvme/eval_run_xxx \
#     --judge_model_path /root/shared-nvme/llama3-exp
#
# Optional auto-shutdown (useful for rented GPU containers/VMs):
#   --shutdown_cmd "shutdown -h now"        # run after successful evaluation
#   --shutdown_on_exit --shutdown_cmd "..." # run even if the script errors/CTRL-C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"
TORCH_DTYPE_DEFAULT="bfloat16"
TP_DEFAULT="2"
BATCH_SIZE_DEFAULT="64"

GPU_IDS_DEFAULT="0,1"
UTIL_THRESH_DEFAULT="10"
SLEEP_SECS_DEFAULT="600"

model_path=""
run_dir=""
repo="${REPO:-$REPO_DEFAULT}"
judge_model_path=""

torch_dtype="$TORCH_DTYPE_DEFAULT"
tp="$TP_DEFAULT"
batch_size="$BATCH_SIZE_DEFAULT"

gpu_ids_csv="$GPU_IDS_DEFAULT"
util_thresh="$UTIL_THRESH_DEFAULT"
sleep_secs="$SLEEP_SECS_DEFAULT"

shutdown_cmd=""
shutdown_on_exit="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --run_dir) run_dir="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --judge_model_path) judge_model_path="$2"; shift 2 ;;
    --torch_dtype) torch_dtype="$2"; shift 2 ;;
    --tensor_parallel_size) tp="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --gpu_ids) gpu_ids_csv="$2"; shift 2 ;;
    --util_thresh) util_thresh="$2"; shift 2 ;;
    --sleep_secs) sleep_secs="$2"; shift 2 ;;
    --shutdown_cmd) shutdown_cmd="$2"; shift 2 ;;
    --shutdown_on_exit) shutdown_on_exit="1"; shift ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$model_path" || -z "$run_dir" ]]; then
  echo "ERROR: --model_path and --run_dir are required" >&2
  exit 2
fi
if [[ -z "$judge_model_path" ]]; then
  judge_model_path="$model_path"
fi

run_shutdown_cmd() {
  [[ -n "$shutdown_cmd" ]] || return 0
  echo "[INFO] Running shutdown_cmd: $shutdown_cmd"
  bash -lc "$shutdown_cmd"
}

if [[ -n "$shutdown_cmd" && "$shutdown_on_exit" == "1" ]]; then
  # Run even if the script errors/interrupts; helpful for "fire-and-forget" runs.
  trap run_shutdown_cmd EXIT
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; cannot check GPU utilization." >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<<"$gpu_ids_csv"

gpu_util() {
  local id="$1"
  # nounits => just numbers, no '%'
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$id" 2>/dev/null | tr -d '[:space:]'
}

wait_for_idle_gpus() {
  local ts
  while true; do
    local all_idle=1
    local msg=""
    for id in "${GPU_IDS[@]}"; do
      local util
      util="$(gpu_util "$id")"
      if [[ -z "$util" ]]; then
        echo "ERROR: failed to query GPU utilization for GPU $id" >&2
        exit 1
      fi
      msg+="gpu${id}=${util}% "
      # treat util as integer
      if (( util > util_thresh )); then
        all_idle=0
      fi
    done

    ts="$(date '+%F %T')"
    if (( all_idle == 1 )); then
      echo "[$ts] GPUs idle (<=${util_thresh}%): ${msg}"
      return 0
    fi
    echo "[$ts] Waiting for GPUs idle (<=${util_thresh}%): ${msg}(sleep ${sleep_secs}s)"
    sleep "$sleep_secs"
  done
}

run_eval() {
  local model_path="$1"
  local run_dir="$2"
  local repo="$3"
  local judge_model_path="$4"

  mkdir -p "$run_dir/results"

  local data_root="$repo/evaluation/datasets/sycophancy_eval"
  local out_root="$run_dir/results/sycophancy_eval"

  # 1) Generate conversations
  python "$repo/evaluation/evaluate_sycophancy_chat_vllm.py" \
    --model_path "$model_path" \
    --data_path "$data_root/free_generation.jsonl" \
    --output_dir "$out_root" \
    --torch_dtype "$torch_dtype" --tensor_parallel_size "$tp" --batch_size "$batch_size"

  python "$repo/evaluation/evaluate_sycophancy_chat_vllm.py" \
    --model_path "$model_path" \
    --data_path "$data_root/multiple_choice.jsonl" \
    --output_dir "$out_root" \
    --torch_dtype "$torch_dtype" --tensor_parallel_size "$tp" --batch_size "$batch_size"

  python "$repo/evaluation/evaluate_sycophancy_chat_vllm.py" \
    --model_path "$model_path" \
    --data_path "$data_root/multiple_choice_cot.jsonl" \
    --output_dir "$out_root" \
    --torch_dtype "$torch_dtype" --tensor_parallel_size "$tp" --batch_size "$batch_size"

  # 2) Judge correctness (free_generation)
  python "$repo/evaluation/score_answer_correctness.py" \
    --model_path "$model_path" \
    --eval_model_path "$judge_model_path" \
    --results_dir "$out_root/free_generation" \
    --output_dir "$run_dir/results/sycophancy_eval_correctness" \
    --torch_dtype "$torch_dtype" --tensor_parallel_size "$tp" --batch_size "$batch_size"

  # 3) Judge apologies (all three)
  for ds in free_generation multiple_choice multiple_choice_cot; do
    python "$repo/evaluation/score_answer_apologies.py" \
      --model_path "$model_path" \
      --eval_model_path "$judge_model_path" \
      --results_dir "$out_root/$ds" \
      --output_dir "$run_dir/results/sycophancy_eval_apologies" \
      --torch_dtype "$torch_dtype" --tensor_parallel_size "$tp" --batch_size "$batch_size"
  done

  # 4) Aggregate + print metrics.
  #
  # print_sycophancy_eval_results.py expects paths under ./results/..., so run from run_dir.
  local summary="$run_dir/sycophancy_eval_summary.txt"
  pushd "$run_dir" >/dev/null
  set +e
  python "$repo/evaluation/print_sycophancy_eval_results.py" \
    --model_path "$model_path" | tee "$summary"
  local st="${PIPESTATUS[0]}"
  set -e
  popd >/dev/null

  if [[ "$st" -ne 0 ]]; then
    if grep -q "Kaleido requires Google Chrome" "$summary" 2>/dev/null; then
      echo "WARNING: print_sycophancy_eval_results.py failed due to missing Chrome (kaleido). Metrics were still printed." >&2
    else
      echo "ERROR: print_sycophancy_eval_results.py failed (exit=$st). See: $summary" >&2
      exit "$st"
    fi
  fi

  echo
  echo "[RESULT] $summary"
  grep -E "AI accuracy \\(before\\)|No match ratio \\(before\\)|AI sorry ratio|AI accuracy \\(after\\)|No match ratio \\(after\\)|Correct -> Incorrect ratio" "$summary" || true
}

wait_for_idle_gpus
run_eval "$model_path" "$run_dir" "$repo" "$judge_model_path"

if [[ -n "$shutdown_cmd" && "$shutdown_on_exit" != "1" ]]; then
  echo "[INFO] Evaluation finished; running shutdown_cmd (success-only)"
  run_shutdown_cmd
fi
