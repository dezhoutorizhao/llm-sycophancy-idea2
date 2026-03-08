#!/bin/bash
###
 # @Author             : 陈蔚 (weichen.cw@zju.edu.cn)
 # @Date               : 2024-10-18 21:21
 # @Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
 # @Last Modified Date : 2024-10-18 21:22
 # @Description        : 
 # -------- 
 # Copyright (c) 2024 Alibaba Inc. 
### 

# EXAMPLE SCRIPT: WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345 RANK=0 NGPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/spt-lora.sh

# Model arguments
model_path=/path/to/model
torch_dtype=bfloat16

# PEFT arguments
peft_type=lora
peft_config=/path/to/peft_config(lora_spt_<model_name>.json)

# Dataset / Tokenization arguments
data_path=finetuning_data/scyophancy_mixed_instruction_data.jsonl
data_type=instruction_tuning
max_seq_len=2048
padding=False
padding_side=right
train_on_prompt=False
# When TRAIN_ON_PROMPT=False, label the last N assistant messages (default 1).
# For the sycophancy multi-round objective, 2 is usually desired (challenge response + final answer).
train_on_last_n_assistant_messages=2

# Training arguments
training=True
deepspeed=configs/configs_deepspeed/deepspeed_config_stage1.json

num_epochs=3
max_steps=120   # max_steps will override num_epochs
save_steps=80
save_total_limit=10

learning_rate=3e-5
min_learning_rate=1e-7
lr_scheduler_type=polynomial
weight_decay=0.1

batch_size_per_device=1
global_batch_size=128

# Saving arguments
output_dir=outputs/spt_lora
save_only_model=False
overwrite_output_dir=False
resume_from_checkpoint=None

# Pinpoint tuning arguments

path_patching_path=path_patching/results/Qwen2.5-0.5B-Instruct
# Precise level means the type of parameters to be tuned.
# Here is a list of precise level and tunable parameters mapping:
#   - 0: all
#   - 1: qkv_proj + o_proj + mlp + wte/lm_head
#   - 2: qkv_proj + o_proj + mlp
#   - 3: qkv_proj + o_proj
#   - 4: qkv_proj
precise_level=3
train_topk=8
train_kv=False

cmd="WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RANK=$RANK NGPUS=$NGPUS MODEL_PATH=$model_path TORCH_DTYPE=$torch_dtype PEFT_TYPE=$peft_type PEFT_CONFIG=$peft_config DATA_PATH=$data_path DATA_TYPE=$data_type MAX_SEQ_LEN=$max_seq_len PADDING=$padding PADDING_SIDE=$padding_side TRAIN_ON_PROMPT=$train_on_prompt TRAIN_ON_LAST_N_ASSISTANT_MESSAGES=$train_on_last_n_assistant_messages TRAINING=$training DEEPSPEED=$deepspeed NUM_EPOCHS=$num_epochs MAX_STEPS=$max_steps SAVE_STEPS=$save_steps SAVE_TOTAL_LIMIT=$save_total_limit LEARNING_RATE=$learning_rate MIN_LEARNING_RATE=$min_learning_rate LR_SCHEDULER_TYPE=$lr_scheduler_type WEIGHT_DECAY=$weight_decay BATCH_SIZE_PER_DEVICE=$batch_size_per_device GLOBAL_BATCH_SIZE=$global_batch_size OUTPUT_DIR=$output_dir SAVE_ONLY_MODEL=$save_only_model OVERWRITE_OUTPUT_DIR=$overwrite_output_dir RESUME_FROM_CHECKPOINT=$resume_from_checkpoint PATH_PATCHING_PATH=$path_patching_path PRECISE_LEVEL=$precise_level TRAIN_TOPK=$train_topk TRAIN_KV=$train_kv bash scripts/run_train.sh"
eval $cmd
