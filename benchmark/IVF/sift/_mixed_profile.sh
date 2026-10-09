#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET="${DATASET:-sift1m}"
WORKLOAD=update
source "$REPO_DIR/benchmark/datasets/_load.sh"
source "$REPO_DIR/benchmark/IVF/common.sh"
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
if [[ -n "${UPDATE_BATCH:-}" ]]; then
    INSERT_CAP="${INSERT_CAP:-$UPDATE_BATCH}"
    DELETE_CAP="${DELETE_CAP:-$UPDATE_BATCH}"
fi
compute_update_layout
(( INSERT_CAP == DELETE_CAP )) || error "SPFresh paired mixed layout requires equal insert/delete caps"
(( UPDATE_ROUNDS <= 4096 )) || error "mixed preparation supports at most 4096 rounds"
NPTS_BASE="$BASE_POINTS"
ivf_profile_args
PROFILE_ARGS+=(--insert-count "$INSERT_POINTS" --delete-count "$DELETE_POINTS"
    --insert-cap "$INSERT_CAP" --delete-cap "$DELETE_CAP" --rounds "$UPDATE_ROUNDS")
if [[ -n "${QUERY_POINTS:-}" ]]; then
    (( QUERY_BATCH_PER_ROUND > 0 )) || error "mixed runs require positive concurrent query count"
    PROFILE_ARGS+=(--query-count "$QUERY_BATCH_PER_ROUND")
fi
