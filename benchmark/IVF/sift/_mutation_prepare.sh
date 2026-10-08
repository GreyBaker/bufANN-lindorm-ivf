#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mutation_profile.sh"
: "${IVF_NLIST:?Set IVF_NLIST explicitly; production training budgets are not qualified}"
: "${IVF_PQ_PREFIX:?Set IVF_PQ_PREFIX to pivots plus full-source PQ codes}"
ensure_dataset
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || error "work tree already exists; choose a fresh EVAL_TEMPFILES root"
safe_mutable_path "$WORK_DIR" >/dev/null
mkdir -p "$(dirname "$WORK_DIR")"
mkdir "$WORK_DIR" # Never reset/delete existing work; preserve partial failures.
link_dataset_files "$WORK_DIR"
write_counted_ids_range "$WORK_DIR/${WORKLOAD}_ids.bin" "$FIRST_ID" "$UPDATE_POINTS"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${FINAL_POINTS}_gt${RECALL_AT}.bin"
if [[ -n "${IVF_GT_FILE:-}" ]]; then
    [[ -f "$IVF_GT_FILE" ]] || error "missing supplied post-update GT: $IVF_GT_FILE"
    ln -s "$(readlink -f "$IVF_GT_FILE")" "$GT_FILE"
else
    resolve_incremental_gt "$GT_FILE" "$RECALL_AT" --base_trunc "$FINAL_POINTS"
fi
ivf_profile_args
python3 "$REPO_DIR/benchmark/IVF/harness.py" prepare "${PROFILE_ARGS[@]}" \
    --pq-prefix "$IVF_PQ_PREFIX" --query "$WORK_DIR/data/query.bin" --truth "$GT_FILE" \
    --recall-at "$RECALL_AT" --cache-root "$EVAL_CACHED"
