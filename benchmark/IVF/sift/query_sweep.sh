#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATASET="${DATASET:-sift1m}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
source "$REPO_DIR/benchmark/IVF/common.sh"
python3 "$REPO_DIR/benchmark/IVF/qualify.py" sweep --work-dir "$WORK_DIR" \
  --driver "$IVF_DRIVER_BIN" --recall-at "$RECALL_AT" "$@"
