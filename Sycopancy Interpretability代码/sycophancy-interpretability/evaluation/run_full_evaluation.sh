#!/bin/bash
###
 # @Author             : 陈蔚 (cy424151@alibaba-inc.com)
 # @Date               : 2024-10-18 11:01
 # @Last Modified By   : 殊理 (husile.hsl@alibaba-inc.com)
 # @Last Modified Date : 2024-10-18 16:14
 # @Description        : 
 # -------- 
 # Copyright (c) 2024 Alibaba Inc. 
### 

model_path=/mnt/llm/model/Qwen-Public/Qwen2.5/Qwen2.5-3B-Instruct
eval_model_path=/mnt/llm/model/Qwen-Public/Qwen2.5/Qwen2.5-3B-Instruct

# 1. Syocphancy Evaluation

## 1.1 Generate multi-round conversations

CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_sycophancy_chat_vllm.py --model_path $model_path --data_path datasets/sycophancy_eval/free_generation.jsonl --tensor_parallel_size 4
CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_sycophancy_chat_vllm.py --model_path $model_path --data_path datasets/sycophancy_eval/multiple_choice.jsonl --tensor_parallel_size 4
CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_sycophancy_chat_vllm.py --model_path $model_path --data_path datasets/sycophancy_eval/multiple_choice_cot.jsonl --tensor_parallel_size 4

## 1.2 Score answer correctness of free generation

CUDA_VISIBLE_DEVICES=4,5,6,7 python score_answer_correctness.py --model_path $model_path --eval_model_path $eval_model_path --results_dir results/sycophancy_eval/free_generation --tensor_parallel_size 4

## 1.3 Score model apologies

CUDA_VISIBLE_DEVICES=4,5,6,7 python score_answer_apologies.py --model_path $model_path --eval_model_path $eval_model_path --results_dir results/sycophancy_eval/free_generation --tensor_parallel_size 4
CUDA_VISIBLE_DEVICES=4,5,6,7 python score_answer_apologies.py --model_path $model_path --eval_model_path $eval_model_path --results_dir results/sycophancy_eval/multiple_choice --tensor_parallel_size 4
CUDA_VISIBLE_DEVICES=4,5,6,7 python score_answer_apologies.py --model_path $model_path --eval_model_path $eval_model_path --results_dir results/sycophancy_eval/multiple_choice_cot --tensor_parallel_size 4

## 1.4 Print final results

python print_sycophancy_eval_results.py --model_path $model_path

# 2. CSQA Evaluation

CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_csqa_chat_vllm.py --model_path $model_path --data_path datasets/csqa_validation.jsonl --tensor_parallel_size 4

# 3. GSM8K Evaluation

CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_gsm8k_chat_vllm.py --model_path $model_path --data_path datasets/gsm8k_test.jsonl --tensor_parallel_size 4

# 4. HumanEval Evaluation

CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_humaneval_chat_vllm.py --model_path $model_path --data_path datasets/HumanEval.jsonl --tensor_parallel_size 4
# pip install human-eval
# evaluate_functional_correctness results/humaneval/<model_name>.jsonl

# 5. MMLU

CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_mmlu.py --model_path $model_path --data_path datasets/mmlu/data --batch_size 4

# 6. StrategyQA

CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_strategyqa_chat_vllm.py --model_path $model_path --data_path datasets/strategyqa_test.json --tensor_parallel_size 4

# 7. KL Divergence

origin_model_path=/path/to/origin_model
CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_kl_divergence.py --model_path_0 $origin_model_path --model_path_1 $model_path

# 8. Sycophancy Evaluation (OOD)

CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_sycophancy_ood.py --model_path $model_path --data_path datasets/sycophancy_eval_ood --batch_size 4
