#!/usr/bin/env bash
set -euo pipefail

# Wait for GPUs to be "memory-idle", then run:
# 1) Path patching (to produce results.pt for head selection)
# 2) SPT sweep for k in {1,2,4,8,16,32} with training + sycophancy eval
#
# This is a thin wrapper around the repo's existing commands:
# - path_patching/path_patching_hf.py
# - scripts/sweep_spt_topk_train_eval_cleanup.sh
#
# Usage:
#   bash scripts/run_mistral_spt_sweep_when_idle.sh
#
# Common overrides (env or args):
#   --model_path /root/shared-nvme/Mistral
#   --run_root /root/shared-nvme/sweep_mistral_topk_1_2_4_8_16_32_methodv2_n2
#   --gpu_ids "0,1" --mem_thresh_pct 10 --sleep_secs 600
#   --ks "1,2,4,8,16,32"
#   TRAIN_ON_LAST_N_ASSISTANT_MESSAGES=2  (recommended; matches method v2 default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
repo="$REPO_DEFAULT"
model_path="/root/shared-nvme/Mistral"
judge_model_path=""   # default: model_path
data_path=""          # default: repo/prepare_training_data/.../method_v2.jsonl
pp_data_path=""       # default: repo/path_patching/datasets/path_patching_data.jsonl
pp_batch_size="4"
pp_sample_num=""      # optional
run_root="/root/shared-nvme/sweep_mistral_topk_1_2_4_8_16_32_methodv2_n2"

ks_csv="1,2,4,8,16,32"
pp_mode="as_is"

gpu_ids_csv="0,1"
mem_thresh_pct="10"
sleep_secs="600"

usage() {
  sed -n '1,120p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --model_path) model_path="$2"; shift 2 ;;
    --judge_model_path) judge_model_path="$2"; shift 2 ;;
    --data_path) data_path="$2"; shift 2 ;;
    --pp_data_path) pp_data_path="$2"; shift 2 ;;
    --pp_batch_size) pp_batch_size="$2"; shift 2 ;;
    --pp_sample_num) pp_sample_num="$2"; shift 2 ;;
    --run_root) run_root="$2"; shift 2 ;;
    --ks) ks_csv="$2"; shift 2 ;;
    --pp_mode) pp_mode="$2"; shift 2 ;;
    --gpu_ids) gpu_ids_csv="$2"; shift 2 ;;
    --mem_thresh_pct) mem_thresh_pct="$2"; shift 2 ;;
    --sleep_secs) sleep_secs="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$judge_model_path" ]]; then
  judge_model_path="$model_path"
fi
if [[ -z "$data_path" ]]; then
  data_path="$repo/prepare_training_data/datasets/scyophancy_mixed_instruction_data_turnaligned_method_v2.jsonl"
fi
if [[ -z "$pp_data_path" ]]; then
  pp_data_path="$repo/path_patching/datasets/path_patching_data.jsonl"
fi

need_cmd() {
  local c="$1"
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $c" >&2
    exit 1
  fi
}

need_cmd bash
need_cmd python
need_cmd nvidia-smi

if [[ ! -d "$repo" ]]; then
  echo "ERROR: repo not found: $repo" >&2
  exit 1
fi
if [[ ! -d "$model_path" ]]; then
  echo "ERROR: model_path not found: $model_path" >&2
  exit 1
fi
if [[ ! -f "$data_path" ]]; then
  echo "ERROR: data_path not found: $data_path" >&2
  exit 1
fi
if [[ ! -f "$pp_data_path" ]]; then
  echo "ERROR: pp_data_path not found: $pp_data_path" >&2
  exit 1
fi

mkdir -p "$run_root"

# Prevent accidental concurrent runs.
lock_dir="$run_root/.lock_run_mistral_spt_sweep"
cleanup_lock() {
  rm -rf "$lock_dir" 2>/dev/null || true
}
if mkdir "$lock_dir" 2>/dev/null; then
  trap cleanup_lock EXIT
else
  echo "ERROR: lock already exists: $lock_dir" >&2
  echo "  Another sweep may be running, or a previous run crashed. If you're sure it's stale, remove it:" >&2
  echo "    rm -rf \"$lock_dir\"" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<<"$(echo "$gpu_ids_csv" | tr -d '[:space:]')"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "ERROR: empty --gpu_ids" >&2
  exit 2
fi

gpu_mem_pct_int() {
  local id="$1"
  local used total out
  out="$(nvidia-smi -i "$id" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
  used="$(echo "$out" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
  total="$(echo "$out" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
  if [[ -z "$used" || -z "$total" || "$total" -le 0 ]]; then
    echo "100"
    return 0
  fi
  echo $(( used * 100 / total ))
}

wait_for_mem_idle_gpus() {
  local ts
  while true; do
    local all_idle=1
    local msg=""
    for id in "${GPU_IDS[@]}"; do
      local pct
      pct="$(gpu_mem_pct_int "$id")"
      msg+="gpu${id}=${pct}% "
      if (( pct >= mem_thresh_pct )); then
        all_idle=0
      fi
    done
    ts="$(date '+%F %T')"
    if (( all_idle == 1 )); then
      echo "[$ts] GPUs memory-idle (<${mem_thresh_pct}%): ${msg}"
      return 0
    fi
    echo "[$ts] Waiting for GPUs memory-idle (<${mem_thresh_pct}%): ${msg}(sleep ${sleep_secs}s)"
    sleep "$sleep_secs"
  done
}

echo "[CONFIG]"
echo "  repo=$repo"
echo "  model_path=$model_path"
echo "  judge_model_path=$judge_model_path"
echo "  data_path=$data_path"
echo "  pp_data_path=$pp_data_path"
echo "  run_root=$run_root"
echo "  ks=$ks_csv"
echo "  pp_mode=$pp_mode"
echo "  gpu_ids=$gpu_ids_csv"
echo "  mem_thresh_pct=$mem_thresh_pct"
echo "  sleep_secs=$sleep_secs"
echo

wait_for_mem_idle_gpus

# Constrain all subprocesses (path patching / training / eval) to the chosen GPUs.
export CUDA_VISIBLE_DEVICES="$(echo "$gpu_ids_csv" | tr -d '[:space:]')"
# Keep downstream defaults consistent with the selected GPUs unless the user overrides them.
export NGPUS="${NGPUS:-${#GPU_IDS[@]}}"
export EVAL_GPU_IDS="${EVAL_GPU_IDS:-$gpu_ids_csv}"
export EVAL_TP="${EVAL_TP:-$NGPUS}"

# If torch CUDA version and nvcc CUDA version obviously mismatch, disable DeepSpeed by default.
# This avoids common "CUDAMismatchException" when DeepSpeed tries to JIT-build ops.
if [[ -z "${DEEPSPEED:-}" ]]; then
  torch_cuda="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
  nvcc_cuda="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\\.[0-9]+' | head -n1 | awk '{print $2}' || true)"
  torch_mm="$(echo "$torch_cuda" | awk -F. 'NF>=2{print $1"."$2}')"
  nvcc_mm="$(echo "$nvcc_cuda" | awk -F. 'NF>=2{print $1"."$2}')"
  if [[ -n "$torch_mm" && -n "$nvcc_mm" && "$torch_mm" != "$nvcc_mm" ]]; then
    export DEEPSPEED="none"
    echo "[WARN] nvcc CUDA $nvcc_mm != torch CUDA $torch_mm; set DEEPSPEED=none to avoid DeepSpeed op compilation."
  fi
fi

# 1) Path patching (head scoring) for this base model.
model_base="$(basename "${model_path%/}")"
pp_out_dir="$repo/path_patching/results/$model_base"

if [[ -f "$pp_out_dir/results.pt" ]]; then
  echo "[PP] found existing: $pp_out_dir/results.pt"
else
  echo "[PP] start (will write: $pp_out_dir/results.pt)"
  pushd "$repo/path_patching" >/dev/null
  pp_args=(--model_path "$model_path" --data_path "$pp_data_path" --batch_size "$pp_batch_size")
  if [[ -n "$pp_sample_num" ]]; then
    pp_args+=(--sample_num "$pp_sample_num")
  fi
  python path_patching_hf.py "${pp_args[@]}"
  popd >/dev/null
  if [[ ! -f "$pp_out_dir/results.pt" ]]; then
    echo "ERROR: path patching did not produce results.pt: $pp_out_dir/results.pt" >&2
    exit 1
  fi
  echo "[PP] done: $pp_out_dir/results.pt"
fi

# 2) Train + eval sweep (uses existing repo script).
unset CACHE_DIR OUTPUT_DIR LOGGING_DIR

echo "[SWEEP] start"
TRAIN_ON_LAST_N_ASSISTANT_MESSAGES="${TRAIN_ON_LAST_N_ASSISTANT_MESSAGES:-2}" \
bash "$repo/scripts/sweep_spt_topk_train_eval_cleanup.sh" \
  --ks "$ks_csv" \
  --base_model "$model_path" \
  --judge_model "$judge_model_path" \
  --data_path "$data_path" \
  --path_patching_path "$pp_out_dir" \
  --pp_mode "$pp_mode" \
  --run_root "$run_root"

echo
echo "[SWEEP] done. Latest TSV:"
ls -t "$run_root"/sycophancy_sweep_*.tsv 2>/dev/null | head -n1 || true
echo
echo "To collect all results:"
echo "  bash \"$repo/scripts/collect_sycophancy_summaries.sh\" --root \"$run_root\""
