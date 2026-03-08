#!/bin/bash
###
 # @Author             : 陈蔚 (weichen.cw@zju.edu.cn)
 # @Date               : 2024-10-09 11:03
 # @Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
 # @Last Modified Date : 2024-10-18 18:58
 # @Description        : 
 # -------- 
 # Copyright (c) 2024 Wei Chen. 
### 

# EXAMPLE SCRIPT: WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345 RANK=0 NGPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sft.sh

# Model arguments
model_path=/path/to/model
torch_dtype=bfloat16

# PEFT arguments
peft_type=None
peft_config=None

# Dataset / Tokenization arguments
data_path=finetuning_data/scyophancy_mixed_instruction_data.jsonl
data_type=instruction_tuning
max_seq_len=2048
padding=False
padding_side=right
train_on_prompt=False

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
output_dir=outputs/sft
save_only_model=False
overwrite_output_dir=False
resume_from_checkpoint=None

cmd="WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RANK=$RANK NGPUS=$NGPUS MODEL_PATH=$model_path TORCH_DTYPE=$torch_dtype PEFT_TYPE=$peft_type PEFT_CONFIG=$peft_config DATA_PATH=$data_path DATA_TYPE=$data_type MAX_SEQ_LEN=$max_seq_len PADDING=$padding PADDING_SIDE=$padding_side TRAIN_ON_PROMPT=$train_on_prompt TRAINING=$training DEEPSPEED=$deepspeed NUM_EPOCHS=$num_epochs MAX_STEPS=$max_steps SAVE_STEPS=$save_steps SAVE_TOTAL_LIMIT=$save_total_limit LEARNING_RATE=$learning_rate MIN_LEARNING_RATE=$min_learning_rate LR_SCHEDULER_TYPE=$lr_scheduler_type WEIGHT_DECAY=$weight_decay BATCH_SIZE_PER_DEVICE=$batch_size_per_device GLOBAL_BATCH_SIZE=$global_batch_size OUTPUT_DIR=$output_dir SAVE_ONLY_MODEL=$save_only_model OVERWRITE_OUTPUT_DIR=$overwrite_output_dir RESUME_FROM_CHECKPOINT=$resume_from_checkpoint bash scripts/run_train.sh"
eval $cmd
