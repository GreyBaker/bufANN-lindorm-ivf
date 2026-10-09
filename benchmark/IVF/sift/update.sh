#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mixed_profile.sh"
[[ -f "$WORK_DIR/mixed_prepared.json" ]] || error "missing completed mixed preparation"
OUTPUT_DIR="$WORK_DIR"
if [[ -n "${IVF_RUN_TAG:-}" ]]; then
    [[ "$IVF_RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || error "unsafe IVF_RUN_TAG"
    OUTPUT_DIR="$WORK_DIR/runs/$IVF_RUN_TAG"
fi
python3 "$REPO_DIR/benchmark/IVF/mixed.py" run "${PROFILE_ARGS[@]}" --output-dir "$OUTPUT_DIR" --recall-at "$RECALL_AT"
