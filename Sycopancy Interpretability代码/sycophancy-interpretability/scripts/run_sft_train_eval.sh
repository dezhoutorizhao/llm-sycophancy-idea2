#!/usr/bin/env bash
set -euo pipefail

# Train a full-parameter SFT baseline (no pinpoint tuning) using the repo's
# compact training code, then run:
#  - SycophancyEval (multi-round) + judge + sycophancy_eval_summary.txt
#  - General Ability (CSQA / GSM8K / StrategyQA) with vLLM
#
# This is intended as a matched-data/steps baseline to compare against:
#  - Pinpoint Tuning / our head-budget methods (k heads)
#  - CAUSM (test-time calibration)
#
# Typical usage (GPU server):
#   bash scripts/run_sft_train_eval.sh \
#     --base_model /root/shared-nvme/Mistral-7B-Instruct-v0.3 \
#     --judge_model /root/shared-nvme/llama3-exp \
#     --data_path /root/.../scyophancy_mixed_instruction_data_no_sorry_insist_finalanswer.jsonl \
#     --run_root /root/shared-nvme/sft_mistral_baseline
#
# Notes (rigor):
# - Uses the same optimizer/hparams defaults as scripts/sweep_spt_topk_train_eval_cleanup.sh
#   unless overridden via env vars.
# - We default to TRAIN_ON_PROMPT=False and TRAIN_ON_LAST_N_ASSISTANT_MESSAGES=2 to align
#   supervision with the multi-turn evaluation protocol (challenge response + final answer).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"

repo="$REPO_DEFAULT"
base_model=""
judge_model=""
data_path=""
run_root=""
cache_dir_arg=""
cleanup_models="0" # default: keep full SFT model dir unless explicitly asked to cleanup

# Train config (defaults match sweep script)
train_on_prompt="${TRAIN_ON_PROMPT:-False}"
train_on_last_n="${TRAIN_ON_LAST_N_ASSISTANT_MESSAGES:-2}"
max_seq_len="${MAX_SEQ_LEN:-2048}"
padding="${PADDING:-False}"
padding_side="${PADDING_SIDE:-right}"

torch_dtype="${TORCH_DTYPE:-bfloat16}"
attn_impl="${ATTN_IMPLEMENTATION:-flash_attention_2}"
deepspeed_cfg="${DEEPSPEED:-configs/configs_deepspeed/deepspeed_config_stage1.json}"

max_steps="${MAX_STEPS:-120}"
lr="${LEARNING_RATE:-3e-5}"
min_lr="${MIN_LEARNING_RATE:-1e-7}"
lr_sched="${LR_SCHEDULER_TYPE:-polynomial}"
weight_decay="${WEIGHT_DECAY:-0.1}"

micro_bs="${TRAIN_MICRO_BATCH_SIZE_PER_GPU:-1}"
global_bs="${TRAIN_GLOBAL_BATCH_SIZE:-128}"
ngpus="${NGPUS:-2}"

# Eval config (SycophancyEval uses scripts/eval_sycophancy_when_idle.sh)
eval_tp="${EVAL_TP:-$ngpus}"
eval_batch_size="${EVAL_BATCH_SIZE:-64}"
eval_gpu_ids="${EVAL_GPU_IDS:-0,1}"
eval_util_thresh="${EVAL_UTIL_THRESH:-10}"
eval_sleep_secs="${EVAL_SLEEP_SECS:-600}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_sft_train_eval.sh \\
    --base_model /path/to/base_model \\
    --data_path /path/to/train.jsonl \\
    --run_root /root/shared-nvme/sft_xxx

Required:
  --base_model     Base model path to finetune (MODEL_PATH)
  --data_path      Training jsonl (instruction_tuning)
  --run_root       Parent directory to store artifacts (eval outputs, tsv, logs)

Optional:
  --judge_model    Judge model path for sycophancy eval (default: base_model)
  --repo           Repo root (default: auto-detected)
  --cache_dir      Tokenization cache dir (default: unique under /root/shared-nvme)
  --cleanup_models 1=delete finetuned model dir after eval, 0=keep (default: 0)

Env (common overrides):
  TRAIN_ON_PROMPT, TRAIN_ON_LAST_N_ASSISTANT_MESSAGES, MAX_SEQ_LEN
  TORCH_DTYPE, ATTN_IMPLEMENTATION, DEEPSPEED
  MAX_STEPS, LEARNING_RATE, MIN_LEARNING_RATE, LR_SCHEDULER_TYPE, WEIGHT_DECAY
  TRAIN_MICRO_BATCH_SIZE_PER_GPU, TRAIN_GLOBAL_BATCH_SIZE, NGPUS
  EVAL_TP, EVAL_BATCH_SIZE, EVAL_GPU_IDS, EVAL_UTIL_THRESH, EVAL_SLEEP_SECS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --base_model) base_model="$2"; shift 2 ;;
    --judge_model) judge_model="$2"; shift 2 ;;
    --data_path) data_path="$2"; shift 2 ;;
    --run_root) run_root="$2"; shift 2 ;;
    --cache_dir) cache_dir_arg="$2"; shift 2 ;;
    --cleanup_models) cleanup_models="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$base_model" || -z "$data_path" || -z "$run_root" ]]; then
  echo "ERROR: missing required args" >&2
  usage
  exit 2
fi
if [[ -z "$judge_model" ]]; then
  judge_model="$base_model"
fi
if [[ ! -f "$data_path" ]]; then
  echo "ERROR: data_path not found: $data_path" >&2
  exit 1
fi

mkdir -p "$run_root"

run_id="$(date +%Y%m%d_%H%M%S)"
base_tag="$(basename "${base_model%/}")"
tag="sft_${base_tag}_${run_id}"

model_dir="/root/shared-nvme/${tag}"
log_dir="$model_dir/tf_logs"
eval_dir="$run_root/eval_${tag}"
meta_dir="$run_root/train_meta/${tag}"
results_tsv="$run_root/sft_${run_id}.tsv"

# Safer default: avoid stale tokenization caches (which can cause train_dataset_size=0).
cache_dir="${cache_dir_arg:-/root/shared-nvme/sft_cache_${tag}_n${train_on_last_n}}"

parse_summary_val() {
  local summary="$1"
  local key="$2"
  grep -E "$key" "$summary" | head -n1 | sed -E 's/.*:[[:space:]]*//'
}

safe_rm_rf() {
  local p="$1"
  if [[ "$p" != /root/shared-nvme/* || "$p" != *"/sft_"* ]]; then
    echo "ERROR: refusing to rm -rf unexpected path: $p" >&2
    exit 1
  fi
  rm -rf "$p"
}

echo "[CONFIG]"
echo "  repo=$repo"
echo "  base_model=$base_model"
echo "  judge_model=$judge_model"
echo "  data_path=$data_path"
echo "  run_root=$run_root"
echo "  model_dir=$model_dir"
echo "  cache_dir=$cache_dir"
echo "  cleanup_models=$cleanup_models"
echo

# Fast tokenization sanity-check to avoid wasting a full training launch.
echo "[PRECHECK] tokenization (first 50 datapoints)"
python3 - <<'PY' "$repo" "$base_model" "$data_path" "$train_on_prompt" "$train_on_last_n"
import json, os, sys
from transformers import AutoTokenizer

repo, model_path, data_path, train_on_prompt, train_on_last_n = sys.argv[1:]
sys.path.insert(0, os.path.join(repo, "pinpoint_tuning"))

from dataset.dataset_json import JsonDataset, IGNORE_INDEX  # type: ignore

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
ok = 0
total = 0
train_on_prompt = str(train_on_prompt).lower() == "true"
try:
    n_last = int(train_on_last_n)
except Exception:
    n_last = 1
n_last = max(1, n_last)

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if total >= 50:
            break
        line = line.strip()
        if not line:
            continue
        total += 1
        dp = json.loads(line)
        ids, labels = JsonDataset._tokenize_datapoint_instruction_tuning(
            dp,
            tok,
            train_on_prompt=train_on_prompt,
            train_on_last_n_assistant_messages=n_last,
        )
        good = bool(ids) and bool(labels) and any(x != IGNORE_INDEX for x in labels)
        ok += int(good)

print(f"[PRECHECK] ok={ok}/{total} (train_on_prompt={train_on_prompt} n_last={n_last})")
if ok == 0:
    raise SystemExit("PRECHECK failed: no valid tokenized samples in first 50 datapoints.")
PY

# Training env (consumed by pinpoint_tuning/scripts/run_train.sh)
export MODEL_PATH="$base_model"
export DATA_PATH="$data_path"
export DATA_TYPE=instruction_tuning
export TRAIN_ON_PROMPT="$train_on_prompt"
export TRAIN_ON_LAST_N_ASSISTANT_MESSAGES="$train_on_last_n"
export MAX_SEQ_LEN="$max_seq_len"
export PADDING="$padding"
export PADDING_SIDE="$padding_side"

# Disable pinpoint tuning explicitly
export PATH_PATCHING_PATH=None
export PRECISE_LEVEL=0
export TRAIN_TOPK=0
export TRAIN_KV=False

export TORCH_DTYPE="$torch_dtype"
export ATTN_IMPLEMENTATION="$attn_impl"
export DEEPSPEED="$deepspeed_cfg"

export MAX_STEPS="$max_steps"
export LEARNING_RATE="$lr"
export MIN_LEARNING_RATE="$min_lr"
export LR_SCHEDULER_TYPE="$lr_sched"
export WEIGHT_DECAY="$weight_decay"

export TRAIN_MICRO_BATCH_SIZE_PER_GPU="$micro_bs"
export TRAIN_GLOBAL_BATCH_SIZE="$global_bs"
export NGPUS="$ngpus"

export OUTPUT_DIR="$model_dir"
export CACHE_DIR="$cache_dir"
export LOGGING_DIR="$log_dir"

# Reduce checkpoint churn (still saves final model).
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export SAVE_STEPS="${SAVE_STEPS:-999999}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"

# Avoid accidental port conflicts if multiple jobs run on the same node.
export MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 20000))}"

safe_rm_rf "$model_dir" || true

echo "[TRAIN] start"
pushd "$repo/pinpoint_tuning" >/dev/null
bash scripts/run_train.sh
popd >/dev/null
echo "[TRAIN] done"

# Ensure config/tokenizer files exist in model_dir (vLLM requires them).
for f in config.json tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json generation_config.json; do
  if [[ ! -f "$model_dir/$f" && -f "$base_model/$f" ]]; then
    cp -f "$base_model/$f" "$model_dir/$f"
  fi
done

# Save lightweight training metadata.
mkdir -p "$meta_dir"
{
  echo "run_id=$run_id"
  echo "tag=$tag"
  echo "base_model=$base_model"
  echo "judge_model=$judge_model"
  echo "data_path=$data_path"
  echo "train_on_prompt=$train_on_prompt"
  echo "train_on_last_n=$train_on_last_n"
  echo "max_seq_len=$max_seq_len"
  echo "torch_dtype=$torch_dtype"
  echo "attn_impl=$attn_impl"
  echo "deepspeed_cfg=$deepspeed_cfg"
  echo "max_steps=$max_steps"
  echo "lr=$lr"
  echo "min_lr=$min_lr"
  echo "lr_sched=$lr_sched"
  echo "weight_decay=$weight_decay"
  echo "micro_bs=$micro_bs"
  echo "global_bs=$global_bs"
  echo "ngpus=$ngpus"
} >"$meta_dir/run_config.env"
cp -f "$model_dir/training_args.bin" "$meta_dir/" 2>/dev/null || true
cp -f "$model_dir/config.json" "$meta_dir/" 2>/dev/null || true
ls -1t "$log_dir"/log_*_0.txt 2>/dev/null | head -n1 | xargs -I{} cp -f "{}" "$meta_dir/" 2>/dev/null || true

echo "[EVAL] sycophancy (wait for idle GPUs) start"
rm -rf "$eval_dir"
bash "$repo/scripts/eval_sycophancy_when_idle.sh" \
  --model_path "$model_dir" \
  --run_dir "$eval_dir" \
  --repo "$repo" \
  --judge_model_path "$judge_model" \
  --torch_dtype "$torch_dtype" \
  --tensor_parallel_size "$eval_tp" \
  --batch_size "$eval_batch_size" \
  --gpu_ids "$eval_gpu_ids" \
  --util_thresh "$eval_util_thresh" \
  --sleep_secs "$eval_sleep_secs"
echo "[EVAL] sycophancy done"

summary="$eval_dir/sycophancy_eval_summary.txt"
if [[ ! -f "$summary" ]]; then
  echo "ERROR: missing summary: $summary" >&2
  exit 1
fi

echo "[EVAL] general ability start"
mkdir -p "$eval_dir/results"
python3 "$repo/evaluation/evaluate_csqa_chat_vllm.py" \
  --model_path "$model_dir" \
  --data_path "$repo/evaluation/datasets/csqa_validation.jsonl" \
  --output_dir "$eval_dir/results/csqa" \
  --torch_dtype "$torch_dtype" --tensor_parallel_size "$eval_tp" --batch_size "$eval_batch_size"

python3 "$repo/evaluation/evaluate_gsm8k_chat_vllm.py" \
  --model_path "$model_dir" \
  --data_path "$repo/evaluation/datasets/gsm8k_test.jsonl" \
  --output_dir "$eval_dir/results/gsm8k" \
  --torch_dtype "$torch_dtype" --tensor_parallel_size "$eval_tp" --batch_size "$eval_batch_size"

python3 "$repo/evaluation/evaluate_strategyqa_chat_vllm.py" \
  --model_path "$model_dir" \
  --data_path "$repo/evaluation/datasets/strategyqa_test.json" \
  --output_dir "$eval_dir/results/strategyqa" \
  --torch_dtype "$torch_dtype" --tensor_parallel_size "$eval_tp" --batch_size "$eval_batch_size"
echo "[EVAL] general ability done"

acc_b="$(parse_summary_val "$summary" "AI accuracy \\(before\\)")"
acc_a="$(parse_summary_val "$summary" "AI accuracy \\(after\\)")"
sorry="$(parse_summary_val "$summary" "AI sorry ratio")"
c2i="$(parse_summary_val "$summary" "Correct -> Incorrect ratio")"
conf="$(parse_summary_val "$summary" "Confidence \\(derived\\)")"
truth="$(parse_summary_val "$summary" "Truthfulness \\(derived\\)")"

python3 - <<'PY' "$eval_dir" "$model_dir" "$results_tsv" "$run_id" "$acc_b" "$acc_a" "$sorry" "$c2i" "$conf" "$truth"
import json, os, sys

eval_dir, model_dir, tsv, run_id, acc_b, acc_a, sorry, c2i, conf, truth = sys.argv[1:]
model_name = os.path.basename(model_dir.rstrip("/"))

def jsonl_acc(path: str) -> float:
    n = 0
    c = 0
    if not os.path.exists(path):
        return float("nan")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            obj = json.loads(line)
            v = obj.get("is_correct", False)
            if isinstance(v, str):
                v = v.lower() in ("true", "1", "yes")
            if bool(v):
                c += 1
    return (c / n) if n else float("nan")

csqa = os.path.join(eval_dir, "results", "csqa", model_name + ".jsonl")
gsm8k = os.path.join(eval_dir, "results", "gsm8k", model_name + ".jsonl")
strategyqa = os.path.join(eval_dir, "results", "strategyqa", model_name + ".jsonl")

csqa_acc = jsonl_acc(csqa)
gsm8k_acc = jsonl_acc(gsm8k)
strategyqa_acc = jsonl_acc(strategyqa)
ga_avg = (csqa_acc + gsm8k_acc + strategyqa_acc) / 3.0

hdr = (
    "run_id\tvariant\ttrain_on_last_n\tmodel_dir\teval_dir\t"
    "acc_before\tacc_after\tsorry_ratio\tc2i_ratio\tconfidence\ttruthfulness\t"
    "csqa_acc\tgsm8k_acc\tstrategyqa_acc\tga_avg\n"
)
row = (
    f"{run_id}\tSFT\t{os.environ.get('TRAIN_ON_LAST_N_ASSISTANT_MESSAGES','')}\t{model_dir}\t{eval_dir}\t"
    f"{acc_b}\t{acc_a}\t{sorry}\t{c2i}\t{conf}\t{truth}\t"
    f"{csqa_acc}\t{gsm8k_acc}\t{strategyqa_acc}\t{ga_avg}\n"
)

need_hdr = not os.path.exists(tsv)
with open(tsv, "a", encoding="utf-8") as f:
    if need_hdr:
        f.write(hdr)
    f.write(row)

print("[SFT] wrote TSV:", tsv)
print("[SFT] GA accs:", "CSQA", csqa_acc, "GSM8K", gsm8k_acc, "StrategyQA", strategyqa_acc, "AVG", ga_avg)
PY

echo
echo "[RESULT] sycophancy summary: $summary"
grep -E "AI accuracy \\(before\\)|No match ratio \\(before\\)|AI sorry ratio|AI accuracy \\(after\\)|No match ratio \\(after\\)|Correct -> Incorrect ratio|Confidence \\(derived\\)|Truthfulness \\(derived\\)" "$summary" || true
echo "[RESULT] TSV: $results_tsv"

if [[ "$cleanup_models" == "1" ]]; then
  echo "[CLEANUP] deleting model_dir: $model_dir"
  safe_rm_rf "$model_dir"
else
  echo "[CLEANUP] keeping model_dir: $model_dir"
fi

echo "[DONE] SFT baseline finished"
