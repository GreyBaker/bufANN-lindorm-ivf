#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET="${DATASET:-sift1m}"
WORKLOAD=query
source "$REPO_DIR/benchmark/datasets/_load.sh"
source "$REPO_DIR/benchmark/IVF/common.sh"
[[ -f "$WORK_DIR/prepared.json" ]] || error "missing completed IVF preparation; run query_prepare.sh first"
OUTPUT_DIR="$WORK_DIR"
if [[ -n "${IVF_RUN_TAG:-}" ]]; then
    [[ "$IVF_RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || error "unsafe IVF_RUN_TAG"
    OUTPUT_DIR="$WORK_DIR/runs/$IVF_RUN_TAG"
fi
ivf_profile_args
python3 "$REPO_DIR/benchmark/IVF/harness.py" run "${PROFILE_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" --recall-at "$RECALL_AT"
