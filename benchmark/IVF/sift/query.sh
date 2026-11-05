#!/usr/bin/env bash
set -euo pipefail
WORKLOAD=query
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_profile.sh"
[[ -f "$WORK_DIR/prepared.json" ]] || error "missing completed IVF preparation; run query_prepare.sh first"
ivf_select_output_dir
ivf_profile_args
python3 "$REPO_DIR/benchmark/IVF/harness.py" run "${PROFILE_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" --recall-at "$RECALL_AT"
