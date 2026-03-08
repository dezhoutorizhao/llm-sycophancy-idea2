#!/usr/bin/env bash
set -euo pipefail

# Sweep Pinpoint Tuning head counts (TRAIN_TOPK) and run sycophancy_eval for each.
# After each eval, delete the finetuned model directory to save disk space.
#
# Typical usage (on the GPU server):
#   bash scripts/sweep_spt_topk_train_eval_cleanup.sh \
#     --ks "8,16,32,48,80" \
#     --base_model "/root/shared-nvme/llama3-exp" \
#     --judge_model "/root/shared-nvme/llama3-exp" \
#     --data_path "/root/.../scyophancy_mixed_instruction_data_turnaligned_method_v2.jsonl" \
#     --path_patching_path "/root/.../path_patching/results_fused_v2/llama3-exp" \
#     --run_root "/root/shared-nvme/sweep_llama3exp_methodv2_topk"
#
# After it finishes, one command to get all metrics:
#   bash scripts/collect_sycophancy_summaries.sh --root "/root/shared-nvme/sweep_llama3exp_methodv2_topk"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"

repo="$REPO_DEFAULT"
ks_csv="8,16,32,48,80"
base_model=""
judge_model=""
data_path=""
path_patching_path=""
run_root=""
pp_mode="as_is"         # as_is|max_finite|random
pp_seed=""              # used when pp_mode=random
cache_dir_arg=""        # optional override
cleanup_models="1"      # 1=delete finetuned model dirs after eval, 0=keep
keep_topks_csv=""       # optional: comma-separated k values to keep even if cleanup_models=1

# Train config (defaults match your current working setup; override via args/env)
train_on_prompt="${TRAIN_ON_PROMPT:-False}"
train_on_last_n="${TRAIN_ON_LAST_N_ASSISTANT_MESSAGES:-2}"
max_seq_len="${MAX_SEQ_LEN:-2048}"
padding="${PADDING:-False}"
padding_side="${PADDING_SIDE:-right}"

precise_level="${PRECISE_LEVEL:-3}"
train_kv="${TRAIN_KV:-False}"

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

# Eval config
eval_tp="${EVAL_TP:-$ngpus}"
eval_batch_size="${EVAL_BATCH_SIZE:-64}"
eval_gpu_ids="${EVAL_GPU_IDS:-0,1}"
eval_util_thresh="${EVAL_UTIL_THRESH:-10}"
eval_sleep_secs="${EVAL_SLEEP_SECS:-600}"

usage() {
  cat <<EOF
Usage:
  bash scripts/sweep_spt_topk_train_eval_cleanup.sh \\
    --ks "8,16,32,48,80" \\
    --base_model /path/to/base_model \\
    --judge_model /path/to/judge_model \\
    --data_path /path/to/train.jsonl \\
    --path_patching_path /path/to/path_patching/results_dir \\
    --run_root /root/shared-nvme/sweep_xxx

Required:
  --base_model            Base model path to finetune (MODEL_PATH)
  --data_path             Training jsonl (DATA_PATH)
  --path_patching_path    Directory containing results.pt for head selection
  --run_root              Parent directory to store sweep artifacts (eval outputs, tsv)

Optional:
  --judge_model           Judge model path (default: base_model)
  --repo                  Repo root (default: auto-detected)
  --ks                    Comma-separated list of TRAIN_TOPK values (default: $ks_csv)
  --pp_mode               Head-score transform: as_is|max_finite|random (default: $pp_mode)
  --pp_seed               Seed for pp_mode=random
  --cache_dir             Cache dir for tokenization/labels (shared across ks). Default: /root/shared-nvme/spt_cache_methodv2_shared_n\${TRAIN_ON_LAST_N_ASSISTANT_MESSAGES}
  --cleanup_models        1=delete finetuned model dirs after eval (default), 0=keep all model dirs
  --keep_topks            Comma-separated TRAIN_TOPK values to keep (e.g. "8,80"). Only used when --cleanup_models=1.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --ks) ks_csv="$2"; shift 2 ;;
    --base_model) base_model="$2"; shift 2 ;;
    --judge_model) judge_model="$2"; shift 2 ;;
    --data_path) data_path="$2"; shift 2 ;;
    --path_patching_path) path_patching_path="$2"; shift 2 ;;
    --pp_mode) pp_mode="$2"; shift 2 ;;
    --pp_seed) pp_seed="$2"; shift 2 ;;
    --cache_dir) cache_dir_arg="$2"; shift 2 ;;
    --cleanup_models) cleanup_models="$2"; shift 2 ;;
    --keep_topks) keep_topks_csv="$2"; shift 2 ;;
    --run_root) run_root="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$base_model" || -z "$data_path" || -z "$path_patching_path" || -z "$run_root" ]]; then
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
if [[ ! -f "$path_patching_path/results.pt" ]]; then
  echo "ERROR: path_patching_path/results.pt not found: $path_patching_path/results.pt" >&2
  exit 1
fi

mkdir -p "$run_root"

run_id="$(date +%Y%m%d_%H%M%S)"
results_tsv="$run_root/sycophancy_sweep_${run_id}.tsv"

pp_used="$path_patching_path"
pp_tag="$(basename "$pp_used")"

pp_stats_dir="$run_root/pp_transforms_${run_id}"
mkdir -p "$pp_stats_dir"

pp_stats_file="$pp_stats_dir/pp_stats.txt"
python - <<'PY' "$path_patching_path/results.pt" "$pp_stats_file"
import sys, torch
src, out = sys.argv[1], sys.argv[2]
r = torch.load(src, map_location="cpu")
nan = torch.isnan(r).sum().item()
inf = torch.isinf(r).sum().item()
fin = r[torch.isfinite(r)]
with open(out, "w", encoding="utf-8") as f:
    f.write(f"src={src}\n")
    f.write(f"shape={tuple(r.shape)} dtype={r.dtype}\n")
    f.write(f"nan_count={nan}/{r.numel()}\n")
    f.write(f"inf_count={inf}/{r.numel()}\n")
    if fin.numel():
        f.write(f"finite_min={fin.min().item()}\n")
        f.write(f"finite_max={fin.max().item()}\n")
print("[PP] wrote stats:", out)
print(open(out, "r", encoding="utf-8").read().strip())
PY

case "$pp_mode" in
  as_is)
    ;;
  max_finite)
    # Create a transformed results.pt that selects "max finite" heads under a min-k selection rule.
    # We must neutralize +Inf first; otherwise, negation turns +Inf into -Inf and forces those heads into the selection.
    pp_used="$pp_stats_dir/pp_max_finite"
    mkdir -p "$pp_used"
    python - <<'PY' "$path_patching_path/results.pt" "$pp_used/results.pt"
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
r = torch.load(src, map_location="cpu")
fin = torch.isfinite(r)
out = torch.empty_like(r)
out[fin] = -r[fin]
# Exclude non-finite entries from min-k selection by assigning a very large positive score.
out[~fin] = 1e9
torch.save(out, dst)
print("[PP] wrote max_finite transform:", dst)
print("nan:", torch.isnan(out).sum().item(), "inf:", torch.isinf(out).sum().item())
PY
    pp_tag="$(basename "$pp_used")"
    ;;
  random)
    if [[ -z "$pp_seed" ]]; then
      echo "ERROR: --pp_seed is required when --pp_mode=random" >&2
      exit 2
    fi
    pp_used="$pp_stats_dir/pp_random_seed${pp_seed}"
    mkdir -p "$pp_used"
    python - <<'PY' "$path_patching_path/results.pt" "$pp_used/results.pt" "$pp_seed"
import sys, torch
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = torch.load(src, map_location="cpu")
g = torch.Generator().manual_seed(seed)
out = torch.rand(r.shape, generator=g, dtype=r.dtype)
torch.save(out, dst)
print("[PP] wrote random transform:", dst, "seed=", seed)
print("nan:", torch.isnan(out).sum().item(), "inf:", torch.isinf(out).sum().item())
PY
    pp_tag="$(basename "$pp_used")"
    ;;
  *)
    echo "ERROR: unknown --pp_mode: $pp_mode (expected: as_is|max_finite|random)" >&2
    exit 2
    ;;
esac

{
  printf "run_id\ttrain_topk\tpp_mode\tpp_tag\ttrain_on_last_n\tmodel_dir\teval_dir\tacc_before\tacc_after\tsorry_ratio\tc2i_ratio\tconfidence\ttruthfulness\n"
} >"$results_tsv"

IFS=',' read -r -a KS <<<"$ks_csv"
IFS=',' read -r -a KEEP_KS <<<"$keep_topks_csv"

echo "[CONFIG]"
echo "  repo=$repo"
echo "  base_model=$base_model"
echo "  judge_model=$judge_model"
echo "  data_path=$data_path"
echo "  path_patching_path(base)=$path_patching_path"
echo "  pp_mode=$pp_mode"
echo "  path_patching_path(used)=$pp_used"
echo "  ks=$ks_csv"
echo "  run_root=$run_root"
echo "  results_tsv=$results_tsv"
echo "  cleanup_models=$cleanup_models"
echo "  keep_topks=$keep_topks_csv"
echo

parse_summary_val() {
  local summary="$1"
  local key="$2"
  # Print the text after the last ':' on the first matching line.
  grep -E "$key" "$summary" | head -n1 | sed -E 's/.*:[[:space:]]*//'
}

safe_rm_rf() {
  local p="$1"
  # Very defensive: only delete under /root/shared-nvme and only if path contains "/spt_".
  if [[ "$p" != /root/shared-nvme/* || "$p" != *"/spt_"* ]]; then
    echo "ERROR: refusing to rm -rf unexpected path: $p" >&2
    exit 1
  fi
  rm -rf "$p"
}

should_keep_model() {
  local k="$1"
  if [[ "$cleanup_models" != "1" ]]; then
    return 0
  fi
  local kk
  for kk in "${KEEP_KS[@]}"; do
    kk="$(echo "$kk" | tr -d '[:space:]')"
    [[ -n "$kk" ]] || continue
    if [[ "$kk" == "$k" ]]; then
      return 0
    fi
  done
  return 1
}

for k in "${KS[@]}"; do
  k="$(echo "$k" | tr -d '[:space:]')"
  [[ -n "$k" ]] || continue

  tag="spt_llama3exp_sweep_top${k}_${run_id}"
  model_dir="/root/shared-nvme/spt_${tag}"
  cache_dir="${cache_dir_arg:-/root/shared-nvme/spt_cache_methodv2_shared_n${train_on_last_n}}"
  log_dir="$model_dir/tf_logs"
  eval_dir="$run_root/eval_top${k}_${run_id}"
  meta_dir="$run_root/train_meta/top${k}_${run_id}"

  echo "=============================="
  echo "[RUN] TRAIN_TOPK=$k"
  echo "  model_dir=$model_dir"
  echo "  eval_dir=$eval_dir"
  echo "=============================="

  # Training env (consumed by pinpoint_tuning/scripts/run_train.sh)
  export MODEL_PATH="$base_model"
  export DATA_PATH="$data_path"
  export DATA_TYPE=instruction_tuning
  export TRAIN_ON_PROMPT="$train_on_prompt"
  export TRAIN_ON_LAST_N_ASSISTANT_MESSAGES="$train_on_last_n"
  export MAX_SEQ_LEN="$max_seq_len"
  export PADDING="$padding"
  export PADDING_SIDE="$padding_side"

  export PATH_PATCHING_PATH="$pp_used"
  export PRECISE_LEVEL="$precise_level"
  export TRAIN_TOPK="$k"
  export TRAIN_KV="$train_kv"

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

  # Fresh model_dir per run.
  safe_rm_rf "$model_dir" || true

  echo "[TRAIN] start"
  pushd "$repo/pinpoint_tuning" >/dev/null
  bash scripts/run_train.sh
  popd >/dev/null
  echo "[TRAIN] done"

  # vLLM/HF config loading requires config/tokenizer files to exist in model_dir.
  # Depending on trainer save settings, OUTPUT_DIR may only contain weights; copy missing files from base_model.
  for f in config.json tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json generation_config.json; do
    if [[ ! -f "$model_dir/$f" && -f "$base_model/$f" ]]; then
      cp -f "$base_model/$f" "$model_dir/$f"
    fi
  done

  # Save lightweight training metadata before deleting the model weights.
  mkdir -p "$meta_dir"
  {
    echo "run_id=$run_id"
    echo "train_topk=$k"
    echo "base_model=$base_model"
    echo "judge_model=$judge_model"
    echo "data_path=$data_path"
    echo "pp_mode=$pp_mode"
    echo "path_patching_path_base=$path_patching_path"
    echo "path_patching_path_used=$pp_used"
    echo "precise_level=$precise_level"
    echo "train_kv=$train_kv"
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
  # Keep the most recent rank0 log if present.
  ls -1t "$log_dir"/log_*_0.txt 2>/dev/null | head -n1 | xargs -I{} cp -f "{}" "$meta_dir/" 2>/dev/null || true

  # Eval
  echo "[EVAL] start"
  # Safety: ensure eval_dir is fresh so vLLM generation doesn't reuse old round_*.jsonl.
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
  echo "[EVAL] done"

  summary="$eval_dir/sycophancy_eval_summary.txt"
  if [[ ! -f "$summary" ]]; then
    echo "ERROR: missing summary: $summary" >&2
    exit 1
  fi

  acc_b="$(parse_summary_val "$summary" "AI accuracy \\(before\\)")"
  acc_a="$(parse_summary_val "$summary" "AI accuracy \\(after\\)")"
  sorry="$(parse_summary_val "$summary" "AI sorry ratio")"
  c2i="$(parse_summary_val "$summary" "Correct -> Incorrect ratio")"
  conf="$(parse_summary_val "$summary" "Confidence \\(derived\\)")"
  truth="$(parse_summary_val "$summary" "Truthfulness \\(derived\\)")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_id" "$k" "$pp_mode" "$pp_tag" "$train_on_last_n" "$model_dir" "$eval_dir" \
    "$acc_b" "$acc_a" "$sorry" "$c2i" "$conf" "$truth" >>"$results_tsv"

  if should_keep_model "$k"; then
    echo "[CLEANUP] keeping model_dir (per flags): $model_dir"
  else
    # Clean up model weights to save disk.
    echo "[CLEANUP] deleting model_dir: $model_dir"
    safe_rm_rf "$model_dir"
  fi
  echo
done

echo "[DONE] sweep finished"
echo "Results TSV: $results_tsv"
echo "One-command view:"
echo "  column -t -s $'\\t' \"$results_tsv\""
