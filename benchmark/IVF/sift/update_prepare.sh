#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mixed_profile.sh"
: "${IVF_NLIST:?Set IVF_NLIST explicitly}"
: "${IVF_PQ_PREFIX:?Set IVF_PQ_PREFIX to existing full-source PQ}"
[[ -z "${IVF_GT_FILE:-}" ]] || error "mixed prepare derives per-round GT; IVF_GT_FILE is not a per-round GT family"
ensure_dataset
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || error "work tree exists; preserve it or choose a fresh EVAL_TEMPFILES root"
safe_mutable_path "$WORK_DIR" >/dev/null
mkdir -p "$(dirname "$WORK_DIR")"
mkdir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
write_counted_ids_range "$WORK_DIR/ids/insert_ids.bin" "$BASE_POINTS" "$INSERT_POINTS"
write_counted_ids_range "$WORK_DIR/ids/delete_ids.bin" "$((BASE_POINTS - DELETE_POINTS))" "$DELETE_POINTS"
for (( r=0; r<UPDATE_ROUNDS; ++r )); do
    write_counted_ids_range "$WORK_DIR/ids/round_${r}_insert.bin" "$((BASE_POINTS + r * INSERT_CAP))" "$INSERT_CAP"
    write_counted_ids_range "$WORK_DIR/ids/round_${r}_delete.bin" "$((BASE_POINTS - DELETE_POINTS + r * DELETE_CAP))" "$DELETE_CAP"
done
generate_round_groundtruths "$WORK_DIR/groundtruth/rounds" "$RECALL_AT" "$UPDATE_ROUNDS" "$BASE_POINTS" "$INSERT_CAP" "$((BASE_POINTS - DELETE_POINTS))" "$DELETE_CAP"
python3 "$REPO_DIR/benchmark/IVF/mixed.py" prepare "${PROFILE_ARGS[@]}" \
    --pq-prefix "$IVF_PQ_PREFIX" --query "$WORK_DIR/data/query.bin" \
    --truth "$WORK_DIR/groundtruth/rounds/round_$((UPDATE_ROUNDS - 1))_gt${RECALL_AT}.bin" \
    --recall-at "$RECALL_AT" --cache-root "$EVAL_CACHED"
