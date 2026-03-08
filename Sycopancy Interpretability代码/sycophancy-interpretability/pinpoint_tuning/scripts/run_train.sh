#!/bin/bash
###
 # @Author             : 陈蔚 (weichen.cw@zju.edu.cn)
 # @Date               : 2024-10-09 10:41
 # @Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
 # @Last Modified Date : 2024-10-18 18:27
 # @Description        : 
 # -------- 
 # Copyright (c) 2024 Wei Chen. 
### 

export PATH=/usr/local/cuda/bin:$PATH   # using nvcc in cuda for cpu_adam compile
export PYTHONPATH='.'

# Reduce odds of "Input/output error" when writing Triton/torch extension caches on unstable filesystems.
# Users can override these env vars externally; we only set safe defaults when missing.
if [[ -z "${TRITON_CACHE_DIR:-}" ]]; then
    if [[ -d "/root/shared-nvme" ]]; then
        export TRITON_CACHE_DIR="/root/shared-nvme/.triton"
    else
        export TRITON_CACHE_DIR="${HOME:-/tmp}/.triton"
    fi
fi
if [[ -z "${TORCH_EXTENSIONS_DIR:-}" ]]; then
    if [[ -d "/root/shared-nvme" ]]; then
        export TORCH_EXTENSIONS_DIR="/root/shared-nvme/torch_extensions"
    else
        export TORCH_EXTENSIONS_DIR="${HOME:-/tmp}/.cache/torch_extensions"
    fi
fi

# Distributed training arguments
WORLD_SIZE="${WORLD_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-12345}"
RANK="${RANK:-0}"
NGPUS=${NGPUS:-8}

echo "- WORLD_SIZE  : $WORLD_SIZE"
echo "- MASTER_ADDR : $MASTER_ADDR"
echo "- MASTER_PORT : $MASTER_PORT"
echo "- RANK        : $RANK"
echo "- NGPUS       : $NGPUS"

# Model arguments
model_path="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
attn_implementation="${ATTN_IMPLEMENTATION:-flash_attention_2}"
torch_dtype="${TORCH_DTYPE:-bfloat16}"
from_config="${FROM_CONFIG:-False}"

# PEFT arguments
peft_type="${PEFT_TYPE:-None}"
peft_config="${PEFT_CONFIG:-None}"
peft_path="${PEFT_PATH:-None}"

# Dataset / Tokenization arguments
data_path="${DATA_PATH:-data/alpaca_data_cleaned.json}"
data_type="${DATA_TYPE:-instruction_tuning}"
max_seq_len="${MAX_SEQ_LEN:-2048}"
padding="${PADDING:-True}"
padding_side="${PADDING_SIDE:-right}"
cache_dir="${CACHE_DIR:-default_dataset_cache}"
train_on_prompt="${TRAIN_ON_PROMPT:-True}"
train_on_last_n_assistant_messages="${TRAIN_ON_LAST_N_ASSISTANT_MESSAGES:-1}"

# Training arguments
training="${TRAINING:-True}"
profile="${PROFILE:-False}"
# DeepSpeed is optional. Set DEEPSPEED=none to disable and run vanilla DDP + HF Trainer.
deepspeed="${DEEPSPEED:-configs/deepspeed_config_stage1.json}"
deepspeed_lc="$(echo "${deepspeed}" | tr '[:upper:]' '[:lower:]')"
deepspeed_arg=()
if [[ -n "${deepspeed_lc}" && "${deepspeed_lc}" != "none" && "${deepspeed_lc}" != "null" ]]; then
    if [[ ! -f "${deepspeed}" ]]; then
        echo "ERROR: DeepSpeed config not found: ${deepspeed}" >&2
        echo "  Set DEEPSPEED=none to disable DeepSpeed." >&2
        exit 1
    fi
    deepspeed_arg=(--deepspeed "${deepspeed}")
else
    echo "[INFO] DeepSpeed disabled (DEEPSPEED=${deepspeed})"
fi

num_epochs="${NUM_EPOCHS:-3}"
max_steps="${MAX_STEPS:--1}"
save_strategy="${SAVE_STRATEGY:-steps}"
save_steps="${SAVE_STEPS:-200}"
save_total_limit="${SAVE_TOTAL_LIMIT:-10}"

learning_rate="${LEARNING_RATE:-1e-4}"
min_learning_rate="${MIN_LEARNING_RATE:-1e-7}"
lr_scheduler_type="${LR_SCHEDULER_TYPE:-constant}"
weight_decay="${WEIGHT_DECAY:-0.1}"

# Batch size control
# - Memory is primarily determined by micro-batch size per GPU and sequence length.
# - GLOBAL_BATCH_SIZE only affects gradient accumulation (time/updates), not per-step activation memory.
#
# Backward-compatible env vars:
# - BATCH_SIZE_PER_DEVICE, GLOBAL_BATCH_SIZE
# Preferred aliases (clearer):
# - TRAIN_MICRO_BATCH_SIZE_PER_GPU, TRAIN_GLOBAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
batch_size_per_device="${TRAIN_MICRO_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_DEVICE:-1}}"
global_batch_size="${TRAIN_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-128}}"

if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
else
  denom=$(( WORLD_SIZE * NGPUS * batch_size_per_device ))
  if [[ "$denom" -le 0 ]]; then
    echo "ERROR: invalid batch size setup (WORLD_SIZE=$WORLD_SIZE NGPUS=$NGPUS batch_size_per_device=$batch_size_per_device)"
    exit 1
  fi
  if (( global_batch_size % denom != 0 )); then
    echo "ERROR: TRAIN_GLOBAL_BATCH_SIZE/GLOBAL_BATCH_SIZE must be divisible by WORLD_SIZE*NGPUS*TRAIN_MICRO_BATCH_SIZE_PER_GPU."
    echo "  got global_batch_size=$global_batch_size, denom=$denom (WORLD_SIZE=$WORLD_SIZE NGPUS=$NGPUS micro=$batch_size_per_device)"
    exit 1
  fi
  gradient_accumulation_steps=$(( global_batch_size / denom ))
fi

if [[ "$gradient_accumulation_steps" -le 0 ]]; then
  echo "ERROR: gradient_accumulation_steps must be >= 1 (got $gradient_accumulation_steps)"
  exit 1
fi

# Saving arguments
output_dir="${OUTPUT_DIR:-outputs}"
logging_dir="${LOGGING_DIR:-$output_dir/tf_logs}"
save_only_model="${SAVE_ONLY_MODEL:-False}"
save_last_interval="${SAVE_LAST_INTERVAL:-3600}"
overwrite_output_dir="${OVERWRITE_OUTPUT_DIR:-False}"
resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-None}"

# Supervised pinpoint tuning arguments
path_patching_path="${PATH_PATCHING_PATH:-None}"
precise_level="${PRECISE_LEVEL:-0}"
train_topk="${TRAIN_TOPK:-32}"
train_kv="${TRAIN_KV:-False}"

# Logger file name
cur_time=$(date +"%Y%m%d%H%M")
log_name=${logging_dir}/log_${cur_time}_${RANK}.txt

# Determine training precision according to model precision
if [[ $torch_dtype == float16 ]]; then 
    fp16=True 
else 
    fp16=False 
fi
if [[ $torch_dtype == bfloat16 ]]; then 
    bf16=True 
else 
    bf16=False 
fi

mkdir -p "${output_dir}"
mkdir -p "${logging_dir}"
mkdir -p "${cache_dir}"

torchrun --nproc_per_node="$NGPUS" --nnodes="$WORLD_SIZE" --node_rank="$RANK" --rdzv_id=333 --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train.py \
    --ddp_timeout 1800 \
    --seed 42 \
    --trust_remote_code True \
    --model_path "$model_path" \
    --attn_implementation "$attn_implementation" \
    --torch_dtype "$torch_dtype" \
    --fp16 $fp16 \
    --bf16 $bf16 \
    --from_config $from_config \
    --peft_type "$peft_type" \
    --peft_config "$peft_config" \
    --peft_path "$peft_path" \
    --data_path "$data_path" \
    --data_type "$data_type" \
    --max_seq_len "$max_seq_len" \
    --padding "$padding" \
    --padding_side "$padding_side" \
    --cache_dir "$cache_dir" \
    --train_on_prompt "$train_on_prompt" \
    --train_on_last_n_assistant_messages "$train_on_last_n_assistant_messages" \
    --training "$training" \
    --profile "$profile" \
    "${deepspeed_arg[@]}" \
    --gradient_checkpointing True \
    --num_train_epochs "$num_epochs" \
    --max_steps "$max_steps" \
    --save_strategy "$save_strategy" \
    --save_steps "$save_steps" \
    --save_total_limit "$save_total_limit" \
    --learning_rate "$learning_rate" \
    --min_learning_rate "$min_learning_rate" \
    --lr_scheduler_type "$lr_scheduler_type" \
    --weight_decay "$weight_decay" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.02 \
    --per_device_train_batch_size "$batch_size_per_device" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --eval_strategy no \
    --output_dir "$output_dir" \
    --logging_dir "$logging_dir" \
    --logging_steps 1 \
    --disable_tqdm True \
    --report_to tensorboard \
    --save_only_model "$save_only_model" \
    --save_last_interval "$save_last_interval" \
    --overwrite_output_dir "$overwrite_output_dir" \
    --resume_from_checkpoint "$resume_from_checkpoint" \
    --path_patching_path "$path_patching_path" \
    --precise_level "$precise_level" \
    --train_topk "$train_topk" \
    --train_kv "$train_kv" \
    2>&1 | tee "$log_name"

    exit ${PIPESTATUS[0]}
