#!/usr/bin/env bash
source "$REPO_DIR/benchmark/scripts/utils.sh"

BASELINE="IVF"
WORKLOAD="${WORKLOAD:-query}"
[[ "$DATASET" =~ ^[a-zA-Z0-9_-]+$ ]] || error "unsafe dataset name"
WORK_DIR="$EVAL_TEMPFILES/$DATASET/$BASELINE/$WORKLOAD"
IVF_BUILD_DIR="${IVF_BUILD_DIR:-$REPO_DIR/build/ivf-benchmark}"
IVF_BUILD_BIN="${IVF_BUILD_BIN:-$IVF_BUILD_DIR/tests/ivf_build}"
IVF_DRIVER_BIN="${IVF_DRIVER_BIN:-$IVF_BUILD_DIR/tests/ivf_driver}"
[[ -x "$IVF_BUILD_BIN" && -x "$IVF_DRIVER_BIN" ]] || error \
    "missing IVF tools; configure/build a full routing-enabled tree (benchmark/IVF/build.sh)"
export IVF_BUILD_DIR IVF_BUILD_BIN IVF_DRIVER_BIN

ivf_profile_args() {
    PROFILE_ARGS=(--dataset "$DATASET" --dtype "$DATA_TYPE" --metric "$METRIC" --dim "$DIM"
        --full-count "$NPTS_FULL" --base-count "$NPTS_BASE" --work-dir "$WORK_DIR"
        --workload "$WORKLOAD" --update-count "${IVF_UPDATE_COUNT:-0}")
}
