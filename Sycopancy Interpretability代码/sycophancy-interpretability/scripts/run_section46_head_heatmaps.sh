#!/usr/bin/env bash
set -euo pipefail

# Section 4.6 pipeline:
# 1) run CI/FG head scoring (delta scripts)
# 2) fuse to results.pt
# 3) render paper-style heatmaps
#
# Example:
#   bash scripts/run_section46_head_heatmaps.sh \
#     --model_specs "llama3-exp=/root/shared-nvme/llama3-exp,Qwen=/root/shared-nvme/Qwen,Mistral=/root/shared-nvme/Mistral-7B-Instruct-v0.3" \
#     --data_path "/root/Sycopancy Interpretability代码/sycophancy-interpretability/path_patching/datasets/path_patching_data.jsonl"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-"$(cd -- "$SCRIPT_DIR/.." && pwd)"}"

MODEL_SPECS="${MODEL_SPECS:-}"
DATA_PATH_CI="${DATA_PATH_CI:-$REPO/evaluation/datasets/sycophancy_eval/multiple_choice.jsonl}"
DATA_PATH_FG="${DATA_PATH_FG:-$REPO/evaluation/datasets/sycophancy_eval/free_generation.jsonl}"

DELTA_CI_ROOT="${DELTA_CI_ROOT:-$REPO/path_patching/results_delta_ci}"
DELTA_FG_ROOT="${DELTA_FG_ROOT:-$REPO/path_patching/results_delta_fg}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results-pt文件}"
FIG_OUT="${FIG_OUT:-$REPO/论文终稿/figures}"

BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SEED="${SEED:-0}"
ANSWER_MAX_TOKENS="${ANSWER_MAX_TOKENS:-16}"

WEIGHT_MODE="${WEIGHT_MODE:-by_used_num}"  # by_used_num|equal
NORM="${NORM:-rank}"                       # rank|zscore|none
FUSE_MODE="${FUSE_MODE:-sum}"              # sum|max|rank_product
RANK_PRODUCT_EPS="${RANK_PRODUCT_EPS:-1e-6}"

# Paper-style defaults (aligned with existing repo figures).
CMAP="${CMAP:-viridis}"
ROBUST_PERCENTILES="${ROBUST_PERCENTILES:-1,99}"
HIGHLIGHT_K="${HIGHLIGHT_K:-0}"
DPI="${DPI:-300}"
NO_HTML="${NO_HTML:-1}"                    # 1 => png only
SKIP_SCORING="${SKIP_SCORING:-0}"          # 1 => only fuse+plot existing scores

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_section46_head_heatmaps.sh \
    --model_specs "name1=/path/model1,name2=/path/model2" \
    [--data_path_ci /path/multiple_choice.jsonl] \
    [--data_path_fg /path/free_generation.jsonl] \
    [--batch_size 4 --dtype bfloat16 --device_map auto] \
    [--max_samples 200 --seed 0 --answer_max_tokens 16] \
    [--weight_mode by_used_num --norm rank --fuse_mode sum] \
    [--results_root /repo/results-pt文件 --fig_out /repo/论文终稿/figures] \
    [--skip_scoring 0]

Notes:
  1) model_specs accepts CSV entries in either form:
     - "name=/abs/model_path"
     - "/abs/model_path"   (name = basename(path))
  2) This script always uses your existing head-scoring scripts:
     - path_patching/path_patching_delta_ci_hf.py
     - path_patching/path_patching_delta_fg_hf.py
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --model_specs) MODEL_SPECS="$2"; shift 2 ;;
    --data_path) DATA_PATH_CI="$2"; DATA_PATH_FG="$2"; shift 2 ;;  # backward compatible
    --data_path_ci) DATA_PATH_CI="$2"; shift 2 ;;
    --data_path_fg) DATA_PATH_FG="$2"; shift 2 ;;
    --delta_ci_root) DELTA_CI_ROOT="$2"; shift 2 ;;
    --delta_fg_root) DELTA_FG_ROOT="$2"; shift 2 ;;
    --results_root) RESULTS_ROOT="$2"; shift 2 ;;
    --fig_out) FIG_OUT="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --device_map) DEVICE_MAP="$2"; shift 2 ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --answer_max_tokens) ANSWER_MAX_TOKENS="$2"; shift 2 ;;
    --weight_mode) WEIGHT_MODE="$2"; shift 2 ;;
    --norm) NORM="$2"; shift 2 ;;
    --fuse_mode) FUSE_MODE="$2"; shift 2 ;;
    --rank_product_eps) RANK_PRODUCT_EPS="$2"; shift 2 ;;
    --cmap) CMAP="$2"; shift 2 ;;
    --robust_percentiles) ROBUST_PERCENTILES="$2"; shift 2 ;;
    --highlight_k) HIGHLIGHT_K="$2"; shift 2 ;;
    --dpi) DPI="$2"; shift 2 ;;
    --no_html) NO_HTML="$2"; shift 2 ;;
    --skip_scoring) SKIP_SCORING="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing file: $1" >&2
    exit 1
  fi
}

need_cmd python
need_file "$DATA_PATH_CI"
need_file "$DATA_PATH_FG"
need_file "$REPO/path_patching/path_patching_delta_ci_hf.py"
need_file "$REPO/path_patching/path_patching_delta_fg_hf.py"
need_file "$REPO/path_patching/plot_results_pt_heatmaps.py"

if [[ -z "${MODEL_SPECS// }" ]]; then
  echo "ERROR: --model_specs is required" >&2
  usage
  exit 2
fi

mkdir -p "$DELTA_CI_ROOT" "$DELTA_FG_ROOT" "$RESULTS_ROOT" "$FIG_OUT"

declare -a MODEL_NAMES=()
IFS=',' read -r -a SPECS <<<"$MODEL_SPECS"
for raw_spec in "${SPECS[@]}"; do
  spec="$(echo "$raw_spec" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -n "$spec" ]] || continue

  if [[ "$spec" == *"="* ]]; then
    model_name="${spec%%=*}"
    model_path="${spec#*=}"
  else
    model_path="$spec"
    model_name="$(basename "${model_path%/}")"
  fi
  model_name="$(echo "$model_name" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  model_path="$(echo "$model_path" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

  if [[ -z "$model_name" || -z "$model_path" ]]; then
    echo "ERROR: bad model spec: $raw_spec" >&2
    exit 2
  fi
  if [[ ! -d "$model_path" ]]; then
    echo "ERROR: model path not found for '$model_name': $model_path" >&2
    exit 1
  fi
  if [[ ! -f "$model_path/config.json" ]]; then
    echo "ERROR: missing config.json for '$model_name': $model_path/config.json" >&2
    exit 1
  fi

  MODEL_NAMES+=("$model_name")

  ci_out="$DELTA_CI_ROOT/$model_name"
  fg_out="$DELTA_FG_ROOT/$model_name"
  fused_out="$RESULTS_ROOT/$model_name"
  mkdir -p "$ci_out" "$fg_out" "$fused_out"

  echo "========== [$model_name] scoring =========="
  echo "[MODEL] $model_path"
  echo "[DATA_CI] $DATA_PATH_CI"
  echo "[DATA_FG] $DATA_PATH_FG"

  if [[ "$SKIP_SCORING" != "1" ]]; then
    ci_cmd=(
      python -u "$REPO/path_patching/path_patching_delta_ci_hf.py"
      --model_path "$model_path"
      --data_path "$DATA_PATH_CI"
      --batch_size "$BATCH_SIZE"
      --dtype "$DTYPE"
      --device_map "$DEVICE_MAP"
      --seed "$SEED"
      --out_dir "$ci_out"
    )
    if [[ -n "${MAX_SAMPLES:-}" ]]; then
      ci_cmd+=(--max_samples "$MAX_SAMPLES")
    fi
    "${ci_cmd[@]}"

    fg_cmd=(
      python -u "$REPO/path_patching/path_patching_delta_fg_hf.py"
      --model_path "$model_path"
      --data_path "$DATA_PATH_FG"
      --batch_size "$BATCH_SIZE"
      --dtype "$DTYPE"
      --device_map "$DEVICE_MAP"
      --seed "$SEED"
      --answer_max_tokens "$ANSWER_MAX_TOKENS"
      --out_dir "$fg_out"
    )
    if [[ -n "${MAX_SAMPLES:-}" ]]; then
      fg_cmd+=(--max_samples "$MAX_SAMPLES")
    fi
    "${fg_cmd[@]}"
  else
    echo "[SKIP] scoring skipped, reusing existing score files."
  fi

  need_file "$ci_out/scores_delta.neg.pt"
  need_file "$fg_out/scores_delta.neg.pt"

  python - "$ci_out/scores_delta.neg.pt" "$fg_out/scores_delta.neg.pt" \
    "$ci_out/meta.json" "$fg_out/meta.json" \
    "$fused_out/results.pt" "$fused_out/results.meta.json" \
    "$FUSE_MODE" "$WEIGHT_MODE" "$NORM" "$RANK_PRODUCT_EPS" <<'PY'
import json
import pathlib
import sys
import torch


def read_used_num(p: pathlib.Path) -> float:
    if not p.exists():
        return 1.0
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        v = float(obj.get("used_num", 1.0))
        return v if v > 0 else 1.0
    except Exception:
        return 1.0


def rank_norm(x: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(x, float("inf"))
    finite = torch.isfinite(x)
    vals = x[finite]
    n = int(vals.numel())
    if n == 0:
        return out
    order = torch.argsort(vals, dim=0, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(n, dtype=torch.float32)
    if n > 1:
        ranks = ranks / float(n - 1)
    else:
        ranks.zero_()
    out[finite] = ranks
    return out


def zscore_norm(x: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(x, float("inf"))
    finite = torch.isfinite(x)
    vals = x[finite]
    n = int(vals.numel())
    if n == 0:
        return out
    mean = vals.mean()
    std = vals.std(unbiased=False)
    if torch.isfinite(std).item() and std.item() > 0:
        out[finite] = (vals - mean) / std
    else:
        out[finite] = vals - mean
    return out


def apply_norm(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "rank":
        return rank_norm(x)
    if mode == "zscore":
        return zscore_norm(x)
    raise ValueError(f"unknown norm: {mode}")


def main() -> None:
    a_path = pathlib.Path(sys.argv[1])
    b_path = pathlib.Path(sys.argv[2])
    meta_a_path = pathlib.Path(sys.argv[3])
    meta_b_path = pathlib.Path(sys.argv[4])
    out_pt = pathlib.Path(sys.argv[5])
    out_meta = pathlib.Path(sys.argv[6])
    fuse_mode = sys.argv[7]
    weight_mode = sys.argv[8]
    norm = sys.argv[9]
    rank_product_eps = float(sys.argv[10])

    a = torch.load(a_path, map_location="cpu").float()
    b = torch.load(b_path, map_location="cpu").float()
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {tuple(a.shape)} and {tuple(b.shape)}")
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")

    na = read_used_num(meta_a_path)
    nb = read_used_num(meta_b_path)
    if weight_mode == "by_used_num":
        s = na + nb
        wa = na / s
        wb = nb / s
    elif weight_mode == "equal":
        wa = 0.5
        wb = 0.5
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode}")

    a_n = apply_norm(a, norm)
    b_n = apply_norm(b, norm)

    if fuse_mode == "sum":
        r = wa * a_n + wb * b_n
    elif fuse_mode == "max":
        r = torch.maximum(wa * a_n, wb * b_n)
    elif fuse_mode == "rank_product":
        r = (wa * a_n + rank_product_eps) * (wb * b_n + rank_product_eps)
    else:
        raise ValueError(f"unknown fuse_mode: {fuse_mode}")

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(r.float(), out_pt)

    meta = {
        "scores_a": str(a_path),
        "scores_b": str(b_path),
        "meta_a": str(meta_a_path),
        "meta_b": str(meta_b_path),
        "fuse_mode": fuse_mode,
        "rank_product_eps": rank_product_eps,
        "weight_mode": weight_mode,
        "weight_a": wa,
        "weight_b": wb,
        "norm": norm,
        "shape": [int(a.shape[0]), int(a.shape[1])],
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FUSE] wrote: {out_pt}")
    print(f"[FUSE] meta : {out_meta}")


if __name__ == "__main__":
    main()
PY
done

if [[ "${#MODEL_NAMES[@]}" -eq 0 ]]; then
  echo "ERROR: no valid model specs parsed." >&2
  exit 2
fi

models_csv="$(IFS=,; echo "${MODEL_NAMES[*]}")"
plot_cmd=(
  python -u "$REPO/path_patching/plot_results_pt_heatmaps.py"
  --in_root "$RESULTS_ROOT"
  --out_dir "$FIG_OUT"
  --models "$models_csv"
  --highlight_k "$HIGHLIGHT_K"
  --origin upper
  --cmap "$CMAP"
  --robust_percentiles "$ROBUST_PERCENTILES"
  --dpi "$DPI"
)
if [[ "$NO_HTML" == "1" ]]; then
  plot_cmd+=(--no_html)
fi
"${plot_cmd[@]}"

echo
echo "[DONE] Section 4.6 head heatmaps ready."
echo "[FIG_DIR] $FIG_OUT"
for n in "${MODEL_NAMES[@]}"; do
  echo "[FIG] $FIG_OUT/R_heatmap_${n}.png"
done
echo "[MANIFEST] $FIG_OUT/R_heatmap_manifest.json"
