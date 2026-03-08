#!/usr/bin/env bash
set -euo pipefail

# End-to-end instruction-data preparation for Pinpoint Tuning.
# Paper: "From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning"
#
# What it produces (relative to this directory):
# - datasets_sampled/*_sampled_20k*.jsonl
# - datasets_with_explanation/*_sampled_20k*.jsonl
# - datasets/all_scyophancy_mixed_instruction_data.jsonl
# - datasets/scyophancy_mixed_instruction_data.jsonl  (default: 3840 lines; this is what scripts under pinpoint_tuning/ expect)
#
# Notes:
# - Steps 3/4 use vLLM and require GPU. The repo authors suggest using a >=72B model for better explanations.
# - This script skips downloads if files exist, and generation scripts resume if output exists.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash 00_prepare_training_data.sh [--skip_download]

Environment variables (recommended):
  GEN_MODEL_PATH           Generator model for step (3) and (4).
                           Default: Qwen/Qwen2.5-72B-Instruct (default also used by this repo's evaluation scripts)
  TENSOR_PARALLEL_SIZE     vLLM tensor parallel size. Default: auto (count CUDA_VISIBLE_DEVICES, else 1)
  TORCH_DTYPE              float16|bfloat16|float32 for vLLM. Default: float16
  GEN_BATCH_SIZE           Batch size for generation. Default: 64
  SUBSAMPLE_N              Subsample size for final training data. Default: 3840

Optional (network):
  HF_ENDPOINT              Hugging Face endpoint (e.g., https://hf-mirror.com).
EOF
}

SKIP_DOWNLOAD=0
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--skip_download" ]]; then
  SKIP_DOWNLOAD=1
fi

if [[ -z "${GEN_MODEL_PATH:-}" ]]; then
  # Prefer a local generator model if present (common for reproduction setups).
  # This avoids accidentally trying to pull/load a 72B model and OOM-ing.
  if [[ -d "/root/shared-nvme/llama3-exp" ]]; then
    GEN_MODEL_PATH="/root/shared-nvme/llama3-exp"
  else
    # The repo authors suggest using a >=72B model for higher-quality explanations.
    GEN_MODEL_PATH="Qwen/Qwen2.5-72B-Instruct"
  fi
fi
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
SUBSAMPLE_N="${SUBSAMPLE_N:-3840}"

count_visible_gpus() {
  # 1) If user pinned CUDA_VISIBLE_DEVICES, honor it.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local csv="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -z "$csv" ]]; then
      echo "1"
      return
    fi
    IFS=',' read -r -a devs <<< "$csv"
    echo "${#devs[@]}"
    return
  fi

  # 2) Otherwise, try to detect physical GPUs.
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -n "$n" && "$n" -ge 1 ]]; then
      echo "$n"
      return
    fi
  fi

  # 3) Fallback.
  echo "1"
}

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$(count_visible_gpus)}"

echo "[00_prepare_training_data] GEN_MODEL_PATH=$GEN_MODEL_PATH"
echo "[00_prepare_training_data] TORCH_DTYPE=$TORCH_DTYPE"
echo "[00_prepare_training_data] TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "[00_prepare_training_data] GEN_BATCH_SIZE=$GEN_BATCH_SIZE"
echo "[00_prepare_training_data] SUBSAMPLE_N=$SUBSAMPLE_N"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "[00_prepare_training_data] HF_ENDPOINT=$HF_ENDPOINT"
fi

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  echo "[Step 1/6] Download raw datasets..."

  mkdir -p datasets

  # MMLU (Hugging Face dataset)
  if [[ ! -d "datasets/mmlu" ]]; then
    if command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download --repo-type dataset --resume-download cais/mmlu \
        --local-dir datasets/mmlu --local-dir-use-symlinks False
    elif command -v hf >/dev/null 2>&1; then
      hf download cais/mmlu --repo-type dataset --resume-download --local-dir datasets/mmlu
    else
      echo "ERROR: neither huggingface-cli nor hf found. Please install huggingface_hub." >&2
      exit 1
    fi
  else
    echo "  - datasets/mmlu exists; skip."
  fi

  # MathQA (zip)
  if [[ ! -f "datasets/mathqa/train.json" ]]; then
    curl -fL -o datasets/MathQA.zip https://math-qa.github.io/math-QA/data/MathQA.zip
    mkdir -p datasets/mathqa
    unzip -o datasets/MathQA.zip -d datasets/mathqa
    rm -f datasets/MathQA.zip
  else
    echo "  - datasets/mathqa/train.json exists; skip."
  fi

  # AQuA (train.json from DeepMind repo)
  if [[ ! -f "datasets/aqua/train.json" ]]; then
    mkdir -p datasets/aqua
    curl -fL -o datasets/aqua/train.json https://raw.githubusercontent.com/google-deepmind/AQuA/refs/heads/master/train.json
  else
    echo "  - datasets/aqua/train.json exists; skip."
  fi

  # TriviaQA (Hugging Face dataset)
  if [[ ! -d "datasets/trivia_qa" ]]; then
    if command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download --repo-type dataset --resume-download mandarjoshi/trivia_qa \
        --local-dir datasets/trivia_qa --local-dir-use-symlinks False --include unfiltered.nocontext/*
    elif command -v hf >/dev/null 2>&1; then
      hf download mandarjoshi/trivia_qa --repo-type dataset --resume-download \
        --local-dir datasets/trivia_qa --include unfiltered.nocontext/*
    else
      echo "ERROR: neither huggingface-cli nor hf found. Please install huggingface_hub." >&2
      exit 1
    fi
  else
    echo "  - datasets/trivia_qa exists; skip."
  fi
else
  echo "[Step 1/6] Skip download (--skip_download)."
fi

echo "[Step 2/6] Normalize and sample datasets (20k each)..."
python 02_normalize_sample_datasets.py

echo "[Step 3/6] Generate TriviaQA false answers (requires GPU + vLLM)..."
python 03_generate_false_answer.py \
  --model_path "$GEN_MODEL_PATH" \
  --torch_dtype "$TORCH_DTYPE" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --batch_size "$GEN_BATCH_SIZE"

echo "[Step 4/6] Generate explanations (requires GPU + vLLM)..."
mkdir -p datasets_with_explanation
python 04_generate_explanation.py --model_path "$GEN_MODEL_PATH" --data_path datasets_sampled/aqua_sampled_20k.jsonl   --output_dir datasets_with_explanation --torch_dtype "$TORCH_DTYPE" --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" --batch_size "$GEN_BATCH_SIZE"
python 04_generate_explanation.py --model_path "$GEN_MODEL_PATH" --data_path datasets_sampled/math_sampled_20k.jsonl   --output_dir datasets_with_explanation --torch_dtype "$TORCH_DTYPE" --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" --batch_size "$GEN_BATCH_SIZE"
python 04_generate_explanation.py --model_path "$GEN_MODEL_PATH" --data_path datasets_sampled/mmlu_sampled_20k.jsonl   --output_dir datasets_with_explanation --torch_dtype "$TORCH_DTYPE" --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" --batch_size "$GEN_BATCH_SIZE"
python 04_generate_explanation.py --model_path "$GEN_MODEL_PATH" --data_path datasets_sampled/trivia_sampled_20k.jsonl --output_dir datasets_with_explanation --torch_dtype "$TORCH_DTYPE" --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" --batch_size "$GEN_BATCH_SIZE"

echo "[Step 5/6] Construct mixed multi-turn sycophancy instruction-tuning data..."
python 05_construct_instruction_data.py

echo "[Step 6/6] Subsample training data (paper config: 3840 examples)..."
if [[ ! -f "datasets/all_scyophancy_mixed_instruction_data.jsonl" ]]; then
  echo "ERROR: datasets/all_scyophancy_mixed_instruction_data.jsonl not found. Step 5 failed?" >&2
  exit 1
fi

mkdir -p datasets
shuf -n "$SUBSAMPLE_N" datasets/all_scyophancy_mixed_instruction_data.jsonl > datasets/scyophancy_mixed_instruction_data.jsonl
echo "[DONE] Wrote datasets/scyophancy_mixed_instruction_data.jsonl ($(wc -l < datasets/scyophancy_mixed_instruction_data.jsonl) lines)"
