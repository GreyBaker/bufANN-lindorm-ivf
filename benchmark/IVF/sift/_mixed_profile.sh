#!/usr/bin/env bash
WORKLOAD=update
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_profile.sh"
if [[ "${IVF_SCHEDULER:-update_driven}" == update_driven ]]; then
    [[ -z "${QUERY_RATIO:-}${QUERY_POINTS:-}${UPDATE_QUERY_BATCH:-}${IVF_SCHEDULER_SEED:-}" ]] || \
        error "update_driven searches until updates finish; unset query ratio/count/seed overrides or select a bounded scheduler"
fi
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
if [[ -n "${UPDATE_BATCH:-}" ]]; then
    INSERT_CAP="${INSERT_CAP:-$UPDATE_BATCH}"
    DELETE_CAP="${DELETE_CAP:-$UPDATE_BATCH}"
fi
compute_update_layout
(( INSERT_CAP == DELETE_CAP )) || error "IVF mixed layout requires equal insert/delete caps"
(( UPDATE_ROUNDS <= 4096 )) || error "mixed preparation supports at most 4096 rounds"
NPTS_BASE="$BASE_POINTS"
ivf_profile_args
PROFILE_ARGS+=(--insert-count "$INSERT_POINTS" --delete-count "$DELETE_POINTS"
    --insert-cap "$INSERT_CAP" --delete-cap "$DELETE_CAP" --rounds "$UPDATE_ROUNDS")
if [[ -n "${QUERY_POINTS:-}" ]]; then
    (( QUERY_BATCH_PER_ROUND > 0 )) || error "mixed runs require positive concurrent query count"
    PROFILE_ARGS+=(--query-count "$QUERY_BATCH_PER_ROUND")
fi
