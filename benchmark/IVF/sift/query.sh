#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET="${DATASET:-sift1m}"
WORKLOAD=query
source "$REPO_DIR/benchmark/datasets/_load.sh"
source "$REPO_DIR/benchmark/IVF/common.sh"
[[ -f "$WORK_DIR/prepared.json" ]] || error "missing completed IVF preparation; run query_prepare.sh first"
ivf_select_output_dir
ivf_profile_args
python3 "$REPO_DIR/benchmark/IVF/harness.py" run "${PROFILE_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" --recall-at "$RECALL_AT"
