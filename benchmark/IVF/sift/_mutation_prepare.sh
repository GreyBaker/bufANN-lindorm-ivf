#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mutation_profile.sh"
ivf_require_build_selection
: "${IVF_PQ_PREFIX:?Set IVF_PQ_PREFIX to pivots plus full-source PQ codes}"
ensure_dataset
ivf_prepare_work_dir
write_counted_ids_range "$WORK_DIR/${WORKLOAD}_ids.bin" "$FIRST_ID" "$UPDATE_POINTS"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${FINAL_POINTS}_gt${RECALL_AT}.bin"
ivf_stage_prefix_truth "$GT_FILE" "$RECALL_AT" "$FINAL_POINTS"
ivf_profile_args
python3 "$REPO_DIR/benchmark/IVF/harness.py" prepare "${PROFILE_ARGS[@]}" \
    --pq-prefix "$IVF_PQ_PREFIX" --query "$WORK_DIR/data/query.bin" --truth "$GT_FILE" \
    --recall-at "$RECALL_AT" --cache-root "$EVAL_CACHED"
