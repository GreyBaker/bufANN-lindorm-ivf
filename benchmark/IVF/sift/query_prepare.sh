#!/usr/bin/env bash
set -euo pipefail
WORKLOAD=query
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_profile.sh"
: "${IVF_PQ_PREFIX:?Set IVF_PQ_PREFIX to existing pivots plus full-source PQ codes}"
[[ -f "$DATA_BIN" && -f "$QUERY_BIN" ]] || error "stage the configured .bin inputs first"
ivf_prepare_work_dir
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${NPTS_BASE}_gt${RECALL_AT}.bin"
ivf_stage_prefix_truth "$GT_FILE" "$RECALL_AT" "$NPTS_BASE"
ivf_profile_args
python3 "$REPO_DIR/benchmark/IVF/harness.py" prepare "${PROFILE_ARGS[@]}" \
    --pq-prefix "$IVF_PQ_PREFIX" --query "$WORK_DIR/data/query.bin" --truth "$GT_FILE" \
    --cache-root "$EVAL_CACHED" --recall-at "$RECALL_AT"
note "Prepared IVF query tree at $WORK_DIR (no production qualification implied)"
