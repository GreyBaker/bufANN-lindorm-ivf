#!/usr/bin/env bash
set -euo pipefail
WORKLOAD=query
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_profile.sh"
python3 "$REPO_DIR/benchmark/IVF/qualify.py" sweep --work-dir "$WORK_DIR" \
  --driver "$IVF_DRIVER_BIN" --recall-at "$RECALL_AT" "$@"
