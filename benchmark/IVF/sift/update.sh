#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mixed_profile.sh"
[[ -f "$WORK_DIR/mixed_prepared.json" ]] || error "missing completed mixed preparation"
ivf_select_output_dir
python3 "$REPO_DIR/benchmark/IVF/mixed.py" run "${PROFILE_ARGS[@]}" --output-dir "$OUTPUT_DIR" --recall-at "$RECALL_AT"
