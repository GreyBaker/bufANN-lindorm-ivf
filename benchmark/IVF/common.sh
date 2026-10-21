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

ivf_require_build_selection() {
    [[ -n "${IVF_NLIST:-}" || -n "${IVF_COST_MODEL_REPORT:-}" ]] || error \
        "Set IVF_NLIST explicitly or provide IVF_COST_MODEL_REPORT; production training budgets are not qualified"
    # Python owns report validation and rejects a conflicting explicit IVF_NLIST.
}

ivf_profile_args() {
    PROFILE_ARGS=(--dataset "$DATASET" --dtype "$DATA_TYPE" --metric "$METRIC" --dim "$DIM"
        --full-count "$NPTS_FULL" --base-count "$NPTS_BASE" --work-dir "$WORK_DIR"
        --workload "$WORKLOAD" --update-count "${IVF_UPDATE_COUNT:-0}")
}

# Claim once, then link inputs. Never reset an existing or partially prepared tree.
ivf_prepare_work_dir() {
    [[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || error \
        "work tree already exists; preserve it or choose a fresh EVAL_TEMPFILES root"
    safe_mutable_path "$WORK_DIR" >/dev/null
    mkdir -p "$(dirname "$WORK_DIR")"
    mkdir "$WORK_DIR"
    link_dataset_files "$WORK_DIR"
}

# Set directly: an error must exit the caller, not a command-substitution subshell.
ivf_select_output_dir() {
    OUTPUT_DIR="$WORK_DIR"
    if [[ -n "${IVF_RUN_TAG:-}" ]]; then
        [[ "$IVF_RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || error "unsafe IVF_RUN_TAG"
        OUTPUT_DIR="$WORK_DIR/runs/$IVF_RUN_TAG"
    fi
}

# Query/sequential GT is a live prefix; mixed workloads retain explicit round GT.
ivf_stage_prefix_truth() {
    local destination="$1" k="$2" prefix_count="$3"
    if [[ -n "${IVF_GT_FILE:-}" ]]; then
        [[ -f "$IVF_GT_FILE" ]] || error "missing supplied GT: $IVF_GT_FILE"
        ln -s "$(readlink -f "$IVF_GT_FILE")" "$destination"
    else
        resolve_incremental_gt "$destination" "$k" --base_trunc "$prefix_count"
    fi
}
