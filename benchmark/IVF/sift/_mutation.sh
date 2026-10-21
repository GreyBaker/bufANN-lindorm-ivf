#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mutation_profile.sh"
[[ -f "$WORK_DIR/prepared.json" && -f "$WORK_DIR/mutation_prepared.json" ]] || error "missing completed mutation preparation"
ivf_select_output_dir
ivf_profile_args
python3 "$REPO_DIR/benchmark/IVF/harness.py" run "${PROFILE_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" --recall-at "$RECALL_AT"
