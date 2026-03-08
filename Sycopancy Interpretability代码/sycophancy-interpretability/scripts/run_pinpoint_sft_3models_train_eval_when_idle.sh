#!/usr/bin/env bash
set -euo pipefail

# Train + evaluate full SFT baselines (Pinpoint Tuning training code) for:
#   - /root/shared-nvme/Mistral-7B-Instruct-v0.3
#   - /root/shared-nvme/Qwen
#   - /root/shared-nvme/llama3-exp
#
# Evaluates:
#   - SycophancyEval (free_generation / multiple_choice / multiple_choice_cot) + judge
#   - General Ability (CSQA / GSM8K / StrategyQA)
#
# Rigor:
#   - Wait until GPUs stay "idle" (util<10%) for 10 minutes before starting.
#   - Raise generation caps to avoid abnormal truncation.
#
# Usage (single command):
#   DATA_PATH=/root/shared-nvme/your_sft_train.jsonl GPU_IDS=0,1 \
#     bash scripts/run_pinpoint_sft_3models_train_eval_when_idle.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-"$(cd -- "$SCRIPT_DIR/.." && pwd)"}"

DATA_PATH="${DATA_PATH:-}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${JUDGE_MODEL:-/root/shared-nvme/llama3-exp}}"

GPU_IDS="${GPU_IDS:-0,1}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"   # training dtype
DTYPE_EVAL="${DTYPE_EVAL:-float16}"      # vLLM eval dtype
BS="${BS:-64}"
TRAIN_ON_LAST_N_ASSISTANT_MESSAGES="${TRAIN_ON_LAST_N_ASSISTANT_MESSAGES:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

# Generation caps (requested)
SYCO_FREE_MAX_TOKENS="${SYCO_FREE_MAX_TOKENS:-1024}"
SYCO_MC_MAX_TOKENS="${SYCO_MC_MAX_TOKENS:-128}"
SYCO_MCCOT_MAX_TOKENS="${SYCO_MCCOT_MAX_TOKENS:-300}"
GA_CSQA_MAX_TOKENS="${GA_CSQA_MAX_TOKENS:-128}"
GA_GSM8K_MAX_TOKENS="${GA_GSM8K_MAX_TOKENS:-512}"
GA_STRATEGYQA_MAX_TOKENS="${GA_STRATEGYQA_MAX_TOKENS:-300}"

# GPU idle gate (defaults: check every 10 minutes; require 10 minutes continuous idle)
GPU_UTIL_THRESH="${GPU_UTIL_THRESH:-10}"    # utilization.gpu must be < this (percent)
GPU_IDLE_MINUTES="${GPU_IDLE_MINUTES:-10}" # need this many minutes continuously idle
GPU_POLL_SECS="${GPU_POLL_SECS:-600}"       # poll interval in seconds (10 min)

need_cmd() {
  local c="$1"
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $c" >&2
    exit 1
  fi
}

need_cmd bash
need_cmd python3
need_cmd nvidia-smi

trim_ws() {
  # Trim all whitespace (spaces/newlines) from a value.
  # This avoids passing an empty/whitespace dtype into argparse choices.
  echo "${1:-}" | tr -d '[:space:]'
}

DTYPE_EVAL="$(trim_ws "$DTYPE_EVAL")"
TORCH_DTYPE="$(trim_ws "$TORCH_DTYPE")"

# Reduce odds of CUDA OOM due to fragmentation (safe default; user can override).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$DTYPE_EVAL" in
  float16|bfloat16|float32) ;;
  "") DTYPE_EVAL=float16 ;;
  *)
    echo "ERROR: DTYPE_EVAL must be one of: float16, bfloat16, float32 (got '$DTYPE_EVAL')" >&2
    exit 2
    ;;
esac

case "$TORCH_DTYPE" in
  float16|bfloat16|float32) ;;
  "") TORCH_DTYPE=bfloat16 ;;
  *)
    echo "ERROR: TORCH_DTYPE must be one of: float16, bfloat16, float32 (got '$TORCH_DTYPE')" >&2
    exit 2
    ;;
esac

if [[ -z "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH is required. Example:" >&2
  echo "  export DATA_PATH=/root/shared-nvme/your_sft_train.jsonl" >&2
  exit 2
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH not found: $DATA_PATH" >&2
  exit 1
fi
if [[ ! -d "$JUDGE_MODEL_PATH" ]]; then
  echo "ERROR: judge model dir not found: $JUDGE_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$REPO" ]]; then
  echo "ERROR: repo not found: $REPO" >&2
  exit 1
fi

IFS=',' read -r -a _GPU_ARR <<<"$(echo "$GPU_IDS" | tr -d '[:space:]')"
if [[ "${#_GPU_ARR[@]}" -le 0 ]]; then
  echo "ERROR: empty GPU_IDS=$GPU_IDS" >&2
  exit 2
fi
NGPUS="${NGPUS:-${#_GPU_ARR[@]}}"
TP="${TP:-$NGPUS}"

gpu_util_int() {
  local id="$1"
  local util=""
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$id" 2>/dev/null | tr -d '[:space:]')" || true
  util="${util%%.*}"
  if [[ -z "$util" || ! "$util" =~ ^[0-9]+$ ]]; then
    echo "100"
  else
    echo "$util"
  fi
}

all_gpus_idle() {
  local msg=""
  local busy=0
  for id in "${_GPU_ARR[@]}"; do
    local util
    util="$(gpu_util_int "$id")"
    msg+="gpu${id}=${util}% "
    if (( util >= GPU_UTIL_THRESH )); then
      busy=1
    fi
  done
  echo "$msg"
  return "$busy" # 0 => idle, 1 => busy
}

wait_for_idle_gpus() {
  local idle_req_secs=$((GPU_IDLE_MINUTES * 60))
  if (( idle_req_secs <= 0 )); then
    echo "[GPU WAIT] GPU_IDLE_MINUTES<=0; skip waiting."
    return 0
  fi
  if (( GPU_POLL_SECS <= 0 )); then
    echo "ERROR: GPU_POLL_SECS must be >0 (got $GPU_POLL_SECS)" >&2
    exit 2
  fi

  # Interpret "10 minutes continuous idle" using coarse polling:
  #   - if idle now -> wait idle_req_secs -> if still idle -> start
  # This matches the user's requested "check every 10 minutes" behavior.
  echo "[GPU WAIT] GPUs=$GPU_IDS need ${GPU_IDLE_MINUTES}min util<${GPU_UTIL_THRESH}% (check every ${GPU_POLL_SECS}s)"
  while true; do
    local ts msg st
    ts="$(date +'%F %T')"
    set +e
    msg="$(all_gpus_idle)"
    st=$?
    set -e

    if (( st == 0 )); then
      echo "[GPU WAIT] ${ts} idle now: ${msg} -> wait ${idle_req_secs}s"
      sleep "$idle_req_secs"
      ts="$(date +'%F %T')"
      set +e
      msg="$(all_gpus_idle)"
      st=$?
      set -e
      if (( st == 0 )); then
        echo "[GPU WAIT] ${ts} still idle: ${msg} -> start"
        return 0
      fi
      echo "[GPU WAIT] ${ts} became busy during idle window: ${msg} -> continue waiting"
      sleep "$GPU_POLL_SECS"
    else
      echo "[GPU WAIT] ${ts} busy: ${msg} -> sleep ${GPU_POLL_SECS}s"
      sleep "$GPU_POLL_SECS"
    fi
  done
}

wait_for_idle_gpus

# ---- DeepSpeed config selection (SFT full finetune needs ZeRO stage>=2 on 2x24GB GPUs) ----
DEEPSPEED_USER="${DEEPSPEED:-}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-auto}"   # auto|1|2|3
ALLOW_STAGE3_FALLBACK="${ALLOW_STAGE3_FALLBACK:-1}"
# If full-parameter SFT still fails (often due to host RAM limits), optionally fall back to LoRA SFT.
ALLOW_LORA_FALLBACK="${ALLOW_LORA_FALLBACK:-1}"
LORA_CONFIG_DEFAULT="${LORA_CONFIG_DEFAULT:-$REPO/pinpoint_tuning/configs/configs_peft/lora/lora_mistral.json}"
SFT_MODE="${SFT_MODE:-full}"  # full|lora

resolve_deepspeed_config() {
  local stage="$1"
  local p=""
  case "$stage" in
    1) p="$REPO/pinpoint_tuning/configs/configs_deepspeed/deepspeed_config_stage1.json" ;;
    2) p="$REPO/pinpoint_tuning/configs/configs_deepspeed/deepspeed_config_stage2.json" ;;
    3) p="$REPO/pinpoint_tuning/configs/configs_deepspeed/deepspeed_config_stage3.json" ;;
    *)
      echo "ERROR: invalid DEEPSPEED_STAGE=$stage (expected 1|2|3|auto)" >&2
      return 2
      ;;
  esac
  if [[ -f "$p" ]]; then
    echo "$p"
    return 0
  fi

  # Fallback: write a known-good config under /tmp (do not modify repo files).
  local tmp="/tmp/deepspeed_config_stage${stage}.json"
  echo "[WARN] DeepSpeed config missing in repo ($p). Writing fallback to $tmp" >&2
  case "$stage" in
    1)
      cat >"$tmp" <<'JSON'
{
    "bf16": { "enabled": "auto" },
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" }
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": { "device": "cpu", "pin_memory": true }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
JSON
      ;;
    2)
      cat >"$tmp" <<'JSON'
{
    "bf16": { "enabled": "auto" },
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" }
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": { "device": "cpu", "pin_memory": true },
        "allgather_partitions": true,
        "allgather_bucket_size": 200000000,
        "reduce_scatter": true,
        "reduce_bucket_size": "auto",
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
JSON
      ;;
    3)
      cat >"$tmp" <<'JSON'
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": { "device": "cpu", "pin_memory": true },
        "offload_param": { "device": "cpu", "pin_memory": true },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 0,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 0,
        "stage3_max_reuse_distance": 0,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" }
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": { "enabled": "auto" },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "wall_clock_breakdown": false
}
JSON
      ;;
  esac
  echo "$tmp"
}

DEEPSPEED_PRIMARY=""
DEEPSPEED_FALLBACK=""
if [[ -n "$DEEPSPEED_USER" ]]; then
  # Respect user override, but validate if it's a path.
  if [[ "$DEEPSPEED_USER" != "none" && "$DEEPSPEED_USER" != "None" && "$DEEPSPEED_USER" != "null" ]]; then
    if [[ ! -f "$DEEPSPEED_USER" ]]; then
      echo "ERROR: DEEPSPEED is set but file not found: $DEEPSPEED_USER" >&2
      exit 2
    fi
  fi
  DEEPSPEED_PRIMARY="$DEEPSPEED_USER"
  echo "[TRAIN CONFIG] Using user-provided DEEPSPEED=$DEEPSPEED_PRIMARY"
else
  if [[ "$DEEPSPEED_STAGE" == "auto" ]]; then
    # Stage-1 commonly OOMs on 2x24GB for full finetune (params+grads replicated).
    DEEPSPEED_STAGE=2
  fi
  DEEPSPEED_PRIMARY="$(resolve_deepspeed_config "$DEEPSPEED_STAGE")"
  if [[ "$ALLOW_STAGE3_FALLBACK" == "1" && "$DEEPSPEED_STAGE" != "3" ]]; then
    DEEPSPEED_FALLBACK="$(resolve_deepspeed_config 3)"
  fi
  echo "[TRAIN CONFIG] DEEPSPEED_STAGE=$DEEPSPEED_STAGE primary=$DEEPSPEED_PRIMARY fallback=${DEEPSPEED_FALLBACK:-none}"
fi

case "$SFT_MODE" in
  full|lora) ;;
  *)
    echo "ERROR: invalid SFT_MODE=$SFT_MODE (expected full|lora)" >&2
    exit 2
    ;;
esac

# ---- Prepare patched eval scripts (do NOT modify repo files) ----
# The upstream vLLM eval scripts hardcode SamplingParams(max_tokens=1024) and don't accept CLI override.
# We generate patched copies under /tmp and run those copies.
PATCH_DIR="${PATCH_DIR:-/tmp/pinpoint_sft_eval_patched_${USER:-user}_$$}"
mkdir -p "$PATCH_DIR"

python3 - <<'PY' "$REPO" "$PATCH_DIR"
import pathlib
import re
import sys

repo = pathlib.Path(sys.argv[1])
patch_dir = pathlib.Path(sys.argv[2])
eval_dir = repo / "evaluation"

targets = [
    "evaluate_sycophancy_chat_vllm.py",
    "evaluate_csqa_chat_vllm.py",
    "evaluate_gsm8k_chat_vllm.py",
    "evaluate_strategyqa_chat_vllm.py",
]

def patch_text(txt: str, src: pathlib.Path) -> str:
    # Add CLI flag --max_tokens (default 1024) if missing.
    if "parser.add_argument('--max_tokens'" not in txt and 'parser.add_argument("--max_tokens"' not in txt:
        lines = txt.splitlines(True)
        out = []
        inserted = False
        for line in lines:
            out.append(line)
            if (not inserted) and re.search(r"parser\.add_argument\(\s*['\"]--batch_size['\"]", line):
                indent = re.match(r"(\s*)", line).group(1)
                out.append(f"{indent}parser.add_argument('--max_tokens', type=int, default=1024)\n")
                inserted = True
        if not inserted:
            raise SystemExit(f"[PATCH] failed to insert --max_tokens into {src}")
        txt = "".join(out)

    # Replace hardcoded SamplingParams(max_tokens=1024) with args.max_tokens.
    if "max_tokens=args.max_tokens" in txt:
        return txt
    n = txt.count("max_tokens=1024")
    if n != 1:
        raise SystemExit(f"[PATCH] expected exactly one 'max_tokens=1024' in {src}, got {n}")
    return txt.replace("max_tokens=1024", "max_tokens=args.max_tokens")

for name in targets:
    src = eval_dir / name
    dst = patch_dir / name
    txt = src.read_text(encoding="utf-8")
    txt2 = patch_text(txt, src)
    dst.write_text(txt2, encoding="utf-8")
    print("[PATCHED COPY]", dst)
PY

# Print a minimal config banner to prevent silent parameter issues.
echo "[CONFIG] REPO=$REPO"
echo "[CONFIG] DATA_PATH=$DATA_PATH"
echo "[CONFIG] JUDGE_MODEL_PATH=$JUDGE_MODEL_PATH"
echo "[CONFIG] GPU_IDS=$GPU_IDS  NGPUS=$NGPUS  TP=$TP"
echo "[CONFIG] TORCH_DTYPE(train)=$TORCH_DTYPE  DTYPE_EVAL(vllm)=$DTYPE_EVAL  BS=$BS"
echo "[CONFIG] DEEPSPEED_PRIMARY=$DEEPSPEED_PRIMARY  DEEPSPEED_FALLBACK=${DEEPSPEED_FALLBACK:-none}"
echo "[CONFIG] ALLOW_LORA_FALLBACK=$ALLOW_LORA_FALLBACK  LORA_CONFIG_DEFAULT=$LORA_CONFIG_DEFAULT"
echo "[CONFIG] SFT_MODE=$SFT_MODE"
echo "[CONFIG] caps: SYCO_FREE=$SYCO_FREE_MAX_TOKENS SYCO_MC=$SYCO_MC_MAX_TOKENS SYCO_MCCOT=$SYCO_MCCOT_MAX_TOKENS GA_CSQA=$GA_CSQA_MAX_TOKENS GA_GSM8K=$GA_GSM8K_MAX_TOKENS GA_STRATEGYQA=$GA_STRATEGYQA_MAX_TOKENS"

# ---- Train + evaluate ----
models=(
  "/root/shared-nvme/Mistral-7B-Instruct-v0.3"
  "/root/shared-nvme/Qwen"
  "/root/shared-nvme/llama3-exp"
)

resolve_model_path_from_output_dir() {
  # Return a loadable HF model directory path:
  #   1) <out_dir>/config.json exists, OR
  #   2) latest <out_dir>/checkpoint-*/config.json exists
  local out_dir="$1"
  if [[ -f "$out_dir/config.json" ]]; then
    echo "$out_dir"
    return 0
  fi

  local latest=""
  shopt -s nullglob
  local ckpts=("$out_dir"/checkpoint-*)
  shopt -u nullglob
  if (( ${#ckpts[@]} > 0 )); then
    latest="$(printf '%s\n' "${ckpts[@]}" | sort -V | tail -n 1)"
    if [[ -f "$latest/config.json" ]]; then
      echo "$latest"
      return 0
    fi
  fi
  return 1
}

resolve_lora_path_from_output_dir() {
  # Return a loadable LoRA adapter directory path:
  #   1) <out_dir>/adapter_config.json exists, OR
  #   2) latest <out_dir>/checkpoint-*/adapter_config.json exists
  local out_dir="$1"
  if [[ -f "$out_dir/adapter_config.json" ]]; then
    echo "$out_dir"
    return 0
  fi

  local latest=""
  shopt -s nullglob
  local ckpts=("$out_dir"/checkpoint-*)
  shopt -u nullglob
  if (( ${#ckpts[@]} > 0 )); then
    latest="$(printf '%s\n' "${ckpts[@]}" | sort -V | tail -n 1)"
    if [[ -f "$latest/adapter_config.json" ]]; then
      echo "$latest"
      return 0
    fi
  fi
  return 1
}

for base in "${models[@]}"; do
  if [[ ! -f "$base/config.json" ]]; then
    echo "[MISSING] $base/config.json" >&2
    exit 1
  fi

  base_tag="$(basename "${base%/}")"
  run_id="$(date +%Y%m%d_%H%M%S)"
  model_dir_base="/root/shared-nvme/sft_${base_tag}_N${TRAIN_ON_LAST_N_ASSISTANT_MESSAGES}_${run_id}"
  cache_dir_base="/root/shared-nvme/sft_cache_${base_tag}_N${TRAIN_ON_LAST_N_ASSISTANT_MESSAGES}_${run_id}"

  echo
  echo "==== [SFT TRAIN] ${base_tag} -> ${model_dir_base} ===="

  # Mirror pinpoint_tuning/scripts/sft.sh, but drive run_train.sh directly
  export WORLD_SIZE=1
  export MASTER_ADDR=localhost
  export MASTER_PORT="$((12000 + RANDOM % 20000))"
  export RANK=0
  export NGPUS="$NGPUS"
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"

  export MODEL_PATH="$base"
  export TORCH_DTYPE="$TORCH_DTYPE"

  export DATA_PATH="$DATA_PATH"
  export DATA_TYPE=instruction_tuning
  export MAX_SEQ_LEN="$MAX_SEQ_LEN"
  export PADDING=False
  export PADDING_SIDE=right
  export TRAIN_ON_PROMPT=False
  export TRAIN_ON_LAST_N_ASSISTANT_MESSAGES="$TRAIN_ON_LAST_N_ASSISTANT_MESSAGES"

  export TRAINING=True

  export NUM_EPOCHS="${SFT_NUM_EPOCHS:-3}"
  export MAX_STEPS="${SFT_MAX_STEPS:-120}"
  export SAVE_STEPS="${SFT_SAVE_STEPS:-80}"
  export SAVE_TOTAL_LIMIT=10
  export LEARNING_RATE=3e-5
  export MIN_LEARNING_RATE=1e-7
  export LR_SCHEDULER_TYPE=polynomial
  export WEIGHT_DECAY=0.1

  export TRAIN_MICRO_BATCH_SIZE_PER_GPU=1
  export TRAIN_GLOBAL_BATCH_SIZE=128

  export SAVE_ONLY_MODEL=False
  export OVERWRITE_OUTPUT_DIR=False
  export RESUME_FROM_CHECKPOINT=None

  # Disable pinpoint tuning explicitly (full-parameter SFT baseline)
  export PATH_PATCHING_PATH=None
  export PRECISE_LEVEL=0
  export TRAIN_TOPK=0
  export TRAIN_KV=False

  train_one() {
    local ds_cfg="$1"
    local out_dir="$2"
    local c_dir="$3"
    local peft_type="$4"
    local peft_config="$5"

    export OUTPUT_DIR="$out_dir"
    export CACHE_DIR="$c_dir"
    export LOGGING_DIR="$out_dir/tf_logs"
    export DEEPSPEED="$ds_cfg"
    export PEFT_TYPE="$peft_type"
    export PEFT_CONFIG="$peft_config"

    mkdir -p "$out_dir" "$c_dir"
    echo "[TRAIN] DEEPSPEED=$DEEPSPEED" >&2
    echo "[TRAIN] OUTPUT_DIR=$OUTPUT_DIR" >&2
    echo "[TRAIN] CACHE_DIR=$CACHE_DIR" >&2
    echo "[TRAIN] PEFT_TYPE=$PEFT_TYPE  PEFT_CONFIG=$PEFT_CONFIG" >&2

    local rc
    pushd "$REPO/pinpoint_tuning" >/dev/null
    set +e
    # Send training logs to stderr so they are visible even when this function is used inside $(...).
    bash scripts/run_train.sh 1>&2
    rc=$?
    set -e
    popd >/dev/null
    return "$rc"
  }

  attempt_full_sft() {
    local ds_cfg="$1"
    local out_dir="$2"
    local c_dir="$3"

    echo "[TRAIN] Full SFT attempt: out_dir=$out_dir" >&2
    if train_one "$ds_cfg" "$out_dir" "$c_dir" "None" "None"; then
      :
    else
      local rc=$?
      echo "ERROR: full SFT failed (exit=$rc) out_dir=$out_dir deepspeed=$ds_cfg" >&2
      return "$rc"
    fi

    local resolved=""
    if resolved="$(resolve_model_path_from_output_dir "$out_dir")"; then
      printf '%s\n' "$resolved"
      return 0
    fi
    echo "ERROR: full SFT did not produce a loadable model (missing config.json). out_dir=$out_dir" >&2
    return 1
  }

  attempt_lora_sft_and_merge() {
    local ds_cfg="$1"
    local out_dir="$2"
    local c_dir="$3"
    local base_model_path="$4"

    local base_tag_local=""
    base_tag_local="$(basename "${base_model_path%/}")"
    local lora_cfg="$LORA_CONFIG_DEFAULT"
    if [[ "$base_tag_local" == *Qwen* || "$base_tag_local" == *qwen* ]]; then
      lora_cfg="$REPO/pinpoint_tuning/configs/configs_peft/lora/lora_qwen2.json"
    elif [[ "$base_tag_local" == *llama* || "$base_tag_local" == *Llama* ]]; then
      lora_cfg="$REPO/pinpoint_tuning/configs/configs_peft/lora/lora_llama.json"
    elif [[ "$base_tag_local" == *Mistral* || "$base_tag_local" == *mistral* ]]; then
      lora_cfg="$REPO/pinpoint_tuning/configs/configs_peft/lora/lora_mistral.json"
    fi
    if [[ ! -f "$lora_cfg" ]]; then
      echo "ERROR: LoRA config not found: $lora_cfg" >&2
      return 2
    fi

    echo "[TRAIN] LoRA SFT attempt: out_dir=$out_dir" >&2
    if train_one "$ds_cfg" "$out_dir" "$c_dir" "lora" "$lora_cfg"; then
      :
    else
      local rc=$?
      echo "ERROR: LoRA SFT failed (exit=$rc) out_dir=$out_dir deepspeed=$ds_cfg" >&2
      return "$rc"
    fi

    local lora_path=""
    if ! lora_path="$(resolve_lora_path_from_output_dir "$out_dir")"; then
      echo "ERROR: LoRA SFT finished but adapter_config.json not found under $out_dir" >&2
      return 1
    fi

    local merged_dir="${out_dir}__merged"
    rm -rf "$merged_dir" || true
    echo "[TRAIN] Merging LoRA -> $merged_dir" >&2
    CUDA_VISIBLE_DEVICES="" python3 -u "$REPO/pinpoint_tuning/tools/merge_lora.py" \
      --model_path "$base_model_path" \
      --lora_path "$lora_path" \
      --output_path "$merged_dir" 1>&2

    if [[ ! -f "$merged_dir/config.json" ]]; then
      echo "ERROR: merged LoRA model missing config.json: $merged_dir" >&2
      return 1
    fi
    printf '%s\n' "$merged_dir"
    return 0
  }

  MODEL_FOR_EVAL=""
  if [[ "$SFT_MODE" == "lora" ]]; then
    echo "[TRAIN] SFT_MODE=lora -> skip full finetune; run LoRA SFT + merge."
    if MODEL_FOR_EVAL="$(attempt_lora_sft_and_merge "$DEEPSPEED_PRIMARY" "${model_dir_base}__lora" "${cache_dir_base}__lora" "$base")"; then
      echo "[TRAIN] LoRA SFT OK (merged): $MODEL_FOR_EVAL"
    else
      echo "ERROR: LoRA SFT failed; no model available for evaluation for base=$base" >&2
      exit 1
    fi
  else
    echo "[TRAIN] Attempt 1: full SFT (primary)..."
    if MODEL_FOR_EVAL="$(attempt_full_sft "$DEEPSPEED_PRIMARY" "$model_dir_base" "$cache_dir_base")"; then
      echo "[TRAIN] Full SFT OK: $MODEL_FOR_EVAL"
    elif [[ -n "${DEEPSPEED_FALLBACK:-}" && "$ALLOW_STAGE3_FALLBACK" == "1" ]] && \
         MODEL_FOR_EVAL="$(attempt_full_sft "$DEEPSPEED_FALLBACK" "${model_dir_base}__ds3" "${cache_dir_base}__ds3")"; then
      echo "[TRAIN] Full SFT OK (stage-3): $MODEL_FOR_EVAL"
    elif [[ "$ALLOW_LORA_FALLBACK" == "1" ]] && \
         MODEL_FOR_EVAL="$(attempt_lora_sft_and_merge "$DEEPSPEED_PRIMARY" "${model_dir_base}__lora" "${cache_dir_base}__lora" "$base")"; then
      echo "[TRAIN] LoRA SFT OK (merged): $MODEL_FOR_EVAL"
    else
      echo "ERROR: training failed; no model available for evaluation for base=$base" >&2
      exit 1
    fi
  fi

  echo
  echo "==== [EVAL] ${base_tag} sycophancy + GA (avoid truncation) ===="
  echo "caps: MC=$SYCO_MC_MAX_TOKENS MCCOT=$SYCO_MCCOT_MAX_TOKENS CSQA=$GA_CSQA_MAX_TOKENS StrategyQA=$GA_STRATEGYQA_MAX_TOKENS GSM8K=$GA_GSM8K_MAX_TOKENS"

  pushd "$REPO/evaluation" >/dev/null
  export PYTHONPATH="$REPO/evaluation${PYTHONPATH:+:$PYTHONPATH}"
  name="$(basename "${MODEL_FOR_EVAL%/}")"

  rm -rf "results/sycophancy_eval/free_generation/${name}" \
         "results/sycophancy_eval/multiple_choice/${name}" \
         "results/sycophancy_eval/multiple_choice_cot/${name}" \
         "results/sycophancy_eval_correctness/free_generation/${name}" \
         "results/sycophancy_eval_apologies/free_generation/${name}" \
         "results/sycophancy_eval_apologies/multiple_choice/${name}" \
         "results/sycophancy_eval_apologies/multiple_choice_cot/${name}" || true
  rm -f "results/csqa/${name}.jsonl" "results/gsm8k/${name}.jsonl" "results/strategyqa/${name}.jsonl" || true

  # --- Sycophancy ---
  CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u "$PATCH_DIR/evaluate_sycophancy_chat_vllm.py" \
    --model_path "${MODEL_FOR_EVAL}" --data_path datasets/sycophancy_eval/free_generation.jsonl \
    --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE_EVAL}" --batch_size "${BS}" \
    --max_tokens "${SYCO_FREE_MAX_TOKENS}"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u "$PATCH_DIR/evaluate_sycophancy_chat_vllm.py" \
    --model_path "${MODEL_FOR_EVAL}" --data_path datasets/sycophancy_eval/multiple_choice.jsonl \
    --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE_EVAL}" --batch_size "${BS}" \
    --max_tokens "${SYCO_MC_MAX_TOKENS}"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u "$PATCH_DIR/evaluate_sycophancy_chat_vllm.py" \
    --model_path "${MODEL_FOR_EVAL}" --data_path datasets/sycophancy_eval/multiple_choice_cot.jsonl \
    --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE_EVAL}" --batch_size "${BS}" \
    --max_tokens "${SYCO_MCCOT_MAX_TOKENS}"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u score_answer_correctness.py \
    --model_path "${MODEL_FOR_EVAL}" --eval_model_path "${JUDGE_MODEL_PATH}" \
    --results_dir results/sycophancy_eval/free_generation \
    --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE_EVAL}" --batch_size "${BS}"

  for ds in free_generation multiple_choice multiple_choice_cot; do
    CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u score_answer_apologies.py \
      --model_path "${MODEL_FOR_EVAL}" --eval_model_path "${JUDGE_MODEL_PATH}" \
      --results_dir "results/sycophancy_eval/${ds}" \
      --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE_EVAL}" --batch_size "${BS}"
  done

  echo "[SYCOPHANCY SUMMARY] ${name}"
  python3 -u print_sycophancy_eval_results.py --model_path "${MODEL_FOR_EVAL}"

  # --- General Ability ---
  CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u "$PATCH_DIR/evaluate_csqa_chat_vllm.py" \
    --model_path "${MODEL_FOR_EVAL}" --data_path datasets/csqa_validation.jsonl \
    --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE_EVAL}" --batch_size "${BS}" \
    --max_tokens "${GA_CSQA_MAX_TOKENS}"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u "$PATCH_DIR/evaluate_gsm8k_chat_vllm.py" \
    --model_path "${MODEL_FOR_EVAL}" --data_path datasets/gsm8k_test.jsonl \
    --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE_EVAL}" --batch_size "${BS}" \
    --max_tokens "${GA_GSM8K_MAX_TOKENS}"

  CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 -u "$PATCH_DIR/evaluate_strategyqa_chat_vllm.py" \
    --model_path "${MODEL_FOR_EVAL}" --data_path datasets/strategyqa_test.json \
    --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE_EVAL}" --batch_size "${BS}" \
    --max_tokens "${GA_STRATEGYQA_MAX_TOKENS}"

  popd >/dev/null
done

echo
echo "[DONE] SFT train+eval finished for 3 models."
