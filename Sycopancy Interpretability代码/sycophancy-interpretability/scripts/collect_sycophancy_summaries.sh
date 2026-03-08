#!/usr/bin/env bash
set -euo pipefail

# Collect sycophancy_eval_summary.txt files under a sweep root directory and print a TSV.
#
# Usage:
#   bash scripts/collect_sycophancy_summaries.sh --root /root/shared-nvme/sweep_xxx
#
# Output:
#   Prints a TSV to stdout. Tip:
#     bash ... | column -t -s $'\t'

root=""

usage() {
  cat <<EOF
Usage:
  bash scripts/collect_sycophancy_summaries.sh --root /path/to/sweep_root
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) root="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$root" ]]; then
  echo "ERROR: --root is required" >&2
  usage
  exit 2
fi

if [[ ! -d "$root" ]]; then
  echo "ERROR: root dir not found: $root" >&2
  exit 1
fi

parse_val() {
  local summary="$1"
  local key="$2"
  grep -E "$key" "$summary" | head -n1 | sed -E 's/.*:[[:space:]]*//'
}

echo -e "eval_dir\tacc_before\tacc_after\tsorry_ratio\tc2i_ratio\tconfidence\ttruthfulness"

while IFS= read -r -d '' summary; do
  eval_dir="$(dirname "$summary")"
  acc_b="$(parse_val "$summary" "AI accuracy \\(before\\)")"
  acc_a="$(parse_val "$summary" "AI accuracy \\(after\\)")"
  sorry="$(parse_val "$summary" "AI sorry ratio")"
  c2i="$(parse_val "$summary" "Correct -> Incorrect ratio")"
  conf="$(parse_val "$summary" "Confidence \\(derived\\)")"
  truth="$(parse_val "$summary" "Truthfulness \\(derived\\)")"
  echo -e "${eval_dir}\t${acc_b}\t${acc_a}\t${sorry}\t${c2i}\t${conf}\t${truth}"
done < <(find "$root" -type f -name "sycophancy_eval_summary.txt" -print0 | sort -z)

