#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET="${DATASET:-sift1m}"
WORKLOAD=query
source "$REPO_DIR/benchmark/datasets/_load.sh"
source "$REPO_DIR/benchmark/IVF/common.sh"
ivf_require_build_selection
: "${IVF_PQ_PREFIX:?Set IVF_PQ_PREFIX to existing pivots plus full-source PQ codes}"
[[ -f "$DATA_BIN" && -f "$QUERY_BIN" ]] || error "stage the configured .bin inputs first"
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || error "work tree already exists; preserve it or choose a fresh EVAL_TEMPFILES root"
safe_mutable_path "$WORK_DIR" >/dev/null
mkdir -p "$(dirname "$WORK_DIR")"
mkdir "$WORK_DIR" # exclusive ownership; keep partial work on failure
link_dataset_files "$WORK_DIR"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${NPTS_BASE}_gt${RECALL_AT}.bin"
if [[ -n "${IVF_GT_FILE:-}" ]]; then
    [[ -f "$IVF_GT_FILE" ]] || error "missing supplied base GT: $IVF_GT_FILE"
    ln -s "$(readlink -f "$IVF_GT_FILE")" "$GT_FILE"
else
    resolve_incremental_gt "$GT_FILE" "$RECALL_AT" --base_trunc "$NPTS_BASE"
fi
ivf_profile_args
python3 "$REPO_DIR/benchmark/IVF/harness.py" prepare "${PROFILE_ARGS[@]}" \
    --pq-prefix "$IVF_PQ_PREFIX" --query "$WORK_DIR/data/query.bin" --truth "$GT_FILE" \
    --cache-root "$EVAL_CACHED" --recall-at "$RECALL_AT"
note "Prepared IVF query tree at $WORK_DIR (no production qualification implied)"
