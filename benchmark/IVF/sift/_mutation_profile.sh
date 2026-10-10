#!/usr/bin/env bash
# Sequential IVF mutation profile arithmetic over source-row IDs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET="${DATASET:-sift1m}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
source "$REPO_DIR/benchmark/IVF/common.sh"
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
compute_split
OVERRIDE="${INSERT_POINTS:-}"
if [[ "$WORKLOAD" == delete ]]; then OVERRIDE="${DELETE_POINTS:-}"; fi
if [[ -n "$OVERRIDE" ]]; then
    [[ "$OVERRIDE" =~ ^[0-9]+$ && "$OVERRIDE" -gt 0 ]] || error "update point override must be a positive integer"
    UPDATE_POINTS="$OVERRIDE"
fi
if [[ "$WORKLOAD" == insert ]]; then
    (( UPDATE_POINTS <= TOTAL_POINTS - BASE_POINTS )) || error "INSERT_POINTS exceeds available tail rows"
    FIRST_ID="$BASE_POINTS"
    FINAL_POINTS=$((BASE_POINTS + UPDATE_POINTS))
else
    (( UPDATE_POINTS < BASE_POINTS )) || error "DELETE_POINTS must be < BASE_POINTS"
    FIRST_ID=$((BASE_POINTS - UPDATE_POINTS))
    FINAL_POINTS="$FIRST_ID"
fi
(( UPDATE_POINTS > 0 )) || error "update count must be positive"
IVF_UPDATE_COUNT="$UPDATE_POINTS"
# Shared split arithmetic establishes the canonical base; overrides change only
# update counts, not the training/base prefix or external row IDs.
NPTS_BASE="$BASE_POINTS"
