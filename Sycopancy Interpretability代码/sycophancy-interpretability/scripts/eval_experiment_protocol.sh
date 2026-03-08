#!/usr/bin/env bash
set -euo pipefail

# Run the Experiment protocol used in this repository:
# 1) Multi-round sycophancy_eval generation (vLLM) + judge scoring + summary.
# 2) General ability evaluation on (CSQA, GSM8K, StrategyQA).
#
# Usage:
#   bash scripts/eval_experiment_protocol.sh /path/to/model_dir /path/to/judge_model_dir
#
# Notes:
# - `tensor_parallel_size` must match the number of visible GPUs.
# - Results are written under evaluation/results/ by default (repo-local).

MODEL_PATH="${1:-}"
JUDGE_PATH="${2:-}"
if [[ -z "${MODEL_PATH}" || -z "${JUDGE_PATH}" ]]; then
  echo "Usage: bash scripts/eval_experiment_protocol.sh /path/to/model_dir /path/to/judge_model_dir"
  exit 2
fi

TP="${TP:-2}"
DTYPE="${DTYPE:-float16}"
BS="${BS:-64}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/evaluation"

MODEL_NAME="$(basename "${MODEL_PATH%/}")"

rm -rf "results/sycophancy_eval/free_generation/${MODEL_NAME}" \
       "results/sycophancy_eval/multiple_choice/${MODEL_NAME}" \
       "results/sycophancy_eval/multiple_choice_cot/${MODEL_NAME}" \
       "results/sycophancy_eval_correctness/free_generation/${MODEL_NAME}" \
       "results/sycophancy_eval_apologies/free_generation/${MODEL_NAME}" \
       "results/sycophancy_eval_apologies/multiple_choice/${MODEL_NAME}" \
       "results/sycophancy_eval_apologies/multiple_choice_cot/${MODEL_NAME}"

echo "[1/3] SycophancyEval generation..."
python evaluate_sycophancy_chat_vllm.py --model_path "${MODEL_PATH}" --data_path datasets/sycophancy_eval/free_generation.jsonl     --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE}" --batch_size "${BS}"
python evaluate_sycophancy_chat_vllm.py --model_path "${MODEL_PATH}" --data_path datasets/sycophancy_eval/multiple_choice.jsonl     --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE}" --batch_size "${BS}"
python evaluate_sycophancy_chat_vllm.py --model_path "${MODEL_PATH}" --data_path datasets/sycophancy_eval/multiple_choice_cot.jsonl --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE}" --batch_size "${BS}"

echo "[2/3] Judge scoring..."
python score_answer_correctness.py --model_path "${MODEL_PATH}" --eval_model_path "${JUDGE_PATH}" --results_dir results/sycophancy_eval/free_generation --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE}" --batch_size "${BS}"
for ds in free_generation multiple_choice multiple_choice_cot; do
  python score_answer_apologies.py --model_path "${MODEL_PATH}" --eval_model_path "${JUDGE_PATH}" --results_dir "results/sycophancy_eval/${ds}" --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE}" --batch_size "${BS}"
done

echo "[SYCOPHANCY SUMMARY]"
python print_sycophancy_eval_results.py --model_path "${MODEL_PATH}"

echo "[3/3] General ability..."
python evaluate_csqa_chat_vllm.py --model_path "${MODEL_PATH}" --data_path datasets/csqa_validation.jsonl --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE}" --batch_size "${BS}"
python evaluate_gsm8k_chat_vllm.py --model_path "${MODEL_PATH}" --data_path datasets/gsm8k_test.jsonl     --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE}" --batch_size "${BS}"
python evaluate_strategyqa_chat_vllm.py --model_path "${MODEL_PATH}" --data_path datasets/strategyqa_test.json --tensor_parallel_size "${TP}" --torch_dtype "${DTYPE}" --batch_size "${BS}"

echo "[DONE] Results are under ${ROOT_DIR}/evaluation/results/"

