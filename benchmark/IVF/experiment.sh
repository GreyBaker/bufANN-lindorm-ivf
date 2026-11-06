#!/usr/bin/env bash
# The three README recipes; preparation/search policy lives in the workload scripts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_DIR/benchmark/scripts/utils.sh"

mode="${1:-sift100m-wide}"
if [[ "$mode" == --help ]]; then
    echo "Usage: bash benchmark/IVF/experiment.sh [workloads|sift100m-wide|sift100m-refine] [--dry-run]"
    echo "Set DATASET, IVF_PQ_PREFIX and a fresh EVAL_TEMPFILES; refinement also needs IVF_GRID_PREVIOUS_ROOT."
    exit 0
fi
[[ $# -le 2 && ( -z "${2:-}" || "$2" == --dry-run ) ]] || error "see experiment.sh --help"
DATASET="${DATASET:-sift100m}"
case "$mode" in
    workloads) nlists=() ;;
    sift100m-wide) nlists=(100 1000 10000 100000 200000 300000) ;;
    sift100m-refine) nlists=(100000 200000 300000 400000) ;;
    *) error "unknown recipe: $mode" ;;
esac
[[ "$DATASET" == sift100m ]] || error "$mode requires DATASET=sift100m"
[[ "${IVF_MAX_REPS:-15}" == 15 ]] || error "main recipes require IVF_MAX_REPS=15"

schedule() {
    local nlist probe breadth effective
    for nlist in "${nlists[@]}"; do
        probes=(16 32 48 64 80 96 112 128 144 160 192 224 256 384 512)
        if [[ "$mode" == sift100m-wide ]]; then
            probes=()
            for ((probe=1; probe<=8192 && probe<=nlist; probe*=2)); do probes+=("$probe"); done
        fi
        for probe in "${probes[@]}"; do
            breadths=(128)
            if [[ "$mode" == sift100m-refine ]]; then
                case "$probe" in 16|32) breadths+=(64) ;; 48|64) breadths+=(256) ;; 128) breadths+=(512) ;; esac
            fi
            for breadth in "${breadths[@]}"; do
                effective=$((3 * probe < nlist ? 3 * probe : nlist))
                (( effective >= breadth )) || effective="$breadth"
                printf '%s\t%s\t%s\t%s\n' "$nlist" "$probe" "$breadth" "$effective"
            done
        done
    done
}

if [[ "${2:-}" == --dry-run ]]; then
    if [[ "$mode" == workloads ]]; then
        echo "Independent prepare/run pairs: query, insert, delete, update; no execution"
    else
        printf 'nlist\tnprobe\trouter_breadth\teffective_router_breadth\n'
        schedule
        echo "$mode: $(schedule | wc -l) query passes; no execution"
    fi
    exit 0
fi

source "$REPO_DIR/benchmark/datasets/_load.sh"
[[ "$DATA_TYPE" == uint8 && "$DIM" == 128 && "$METRIC" == l2 && "$NPTS_FULL" == 100000000 &&
   "$NPTS_BASE" == 90000000 && "$UPDATE_POINTS" == 10000000 ]] || error "incorrect SIFT100M profile"
: "${IVF_PQ_PREFIX:?Set IVF_PQ_PREFIX to existing full-source PQ64}"
[[ "$EVAL_TEMPFILES" == /* ]] || error "EVAL_TEMPFILES must be absolute"
root="$EVAL_TEMPFILES"
[[ ! -e "$root" && ! -L "$root" ]] || error "preserve the existing root and choose a new one: $root"
[[ -f "$DATA_BIN" && -f "$QUERY_BIN" && -f "${IVF_PQ_PREFIX}_pq_pivots.bin" &&
   -f "${IVF_PQ_PREFIX}_pq_compressed.bin" ]] || error "missing vector/query/PQ inputs"

unset INSERT_POINTS DELETE_POINTS QUERY_POINTS UPDATE_QUERY_BATCH QUERY_RATIO INSERT_CAP DELETE_CAP
unset IVF_RUN_TAG IVF_COST_MODEL_REPORT IVF_SCHEDULER_SEED
export DATASET DATA_BIN QUERY_BIN EVAL_CACHED
export IVF_SAMPLE_RATE=1 IVF_TRAINER_MODE=lv-plus IVF_MAX_REPS=15
export IVF_INIT=random_partition IVF_TRAINING_BREADTH=64 IVF_SEED=0 IVF_BUILD_THREADS=20
export IVF_TRAINING_RERANK_FACTOR=3 IVF_TRAINING_REFINE_BREADTH=128
export IVF_OPTIMIZED_MODES=1 IVF_ASSIGNMENT_ROUTER_MODE=pca-refined IVF_BUILD_ROUTER_RERANK_FACTOR=3
export IVF_PCA_VARIANCE=0.95 IVF_REFINE_EF=128 IVF_QUERY_BREADTH=128 IVF_INSERT_BREADTH=64
export IVF_ROUTER_MODE=pca-refined IVF_ROUTER_RERANK_FACTOR=3 IVF_ROUTER_BREADTH=128 IVF_RERANK_FACTOR=3
export IVF_ROUTING_WORK_LIMIT=3000000000
export IVF_MEMORY_BUDGET_GB="${IVF_MEMORY_BUDGET_GB:-96}" IVF_PAGE_SIZE=4096
export IVF_QUERY_MEMORY_BUDGET_GB="${IVF_QUERY_MEMORY_BUDGET_GB:-$IVF_MEMORY_BUDGET_GB}"
export IVF_BUFFER_FRAMES="${IVF_BUFFER_FRAMES:-262144}" QUERY_THREADS="${QUERY_THREADS:-8}"
export OMP_NUM_THREADS=20 OMP_THREAD_LIMIT=20 MKL_NUM_THREADS=1
export IVF_BUILD_DIR="${IVF_BUILD_DIR:-$REPO_DIR/build/ivf-benchmark}"
export INCREMENTAL_GT_BIN="${INCREMENTAL_GT_BIN:-$root/incremental_groundtruth}"
timeout_s="${IVF_TIMEOUT_S:-86400}"
generate_gt="${IVF_GENERATE_GT:-1}"
if [[ "$mode" == workloads ]]; then
    export IVF_NLIST=200000 IVF_NPROBE=64 UPDATE_PERCENT=10 UPDATE_BATCH="$((UPDATE_POINTS / 10))"
    export IVF_SCHEDULER=update_driven NTHREADS="${NTHREADS:-16}"
    export IVF_INSERT_THREADS="${IVF_INSERT_THREADS:-8}" IVF_INSERT_INPUT_POLICY="${IVF_INSERT_INPUT_POLICY:-staged}"
    export IVF_WARMUP_QUERIES="${IVF_WARMUP_QUERIES:-0}"
else
    unset IVF_NLIST
    export IVF_WARMUP_QUERIES="${IVF_WARMUP_QUERIES:-64}"
fi
[[ -z "${IVF_GT_FILE:-}" ]] || error "unset IVF_GT_FILE; this recipe checks shared GT"
if [[ "$mode" == sift100m-refine ]]; then
    : "${IVF_GRID_PREVIOUS_ROOT:?Set the previous wide-grid root}"
    GT_BIN_KCACHED=$(cat "$IVF_GRID_PREVIOUS_ROOT/selected_gt.txt")
    [[ -f "$GT_BIN_KCACHED" ]] || error "missing previous GT binary"
    for nlist in 100000 200000 300000; do
        [[ -f "$IVF_GRID_PREVIOUS_ROOT/nlist-$nlist/$DATASET/IVF/query/prepared.json" ]] || error "missing previous nlist=$nlist preparation"
        python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["settings"]; assert str(s["max_reps"]) == "15" and str(s["nlist"]) == sys.argv[2], "previous base requires matching nlist and 15-pass training; choose a matching wide run"' \
            "$IVF_GRID_PREVIOUS_ROOT/nlist-$nlist/$DATASET/IVF/query/prepared.json" "$nlist"
    done
    generate_gt=0
fi

mkdir -p "$(dirname "$root")"
mkdir "$root"
gt_args=(--base-file "$DATA_BIN" --query-file "$QUERY_BIN" --full-count "$NPTS_FULL" --base-count "$NPTS_BASE"
    --insert-count "$UPDATE_POINTS" --delete-count "$UPDATE_POINTS" --mixed-rounds 10
    --mixed-insert-cap "$((UPDATE_POINTS / 10))" --mixed-delete-cap "$((UPDATE_POINTS / 10))"
    --k-out 10 --expected-queries 10000 --num-threads 20 --cache-dir "$GT_ARTIFACT_DIR")
[[ "$generate_gt" != 1 ]] || gt_args+=(--generate)
[[ -z "${GT_BIN_KCACHED:-}" ]] || gt_args+=(--cached-gt "$GT_BIN_KCACHED")
# Keep assignment separate from export: GT failure must stop preparation.
GT_BIN_KCACHED=$(timeout -k 30 "$timeout_s" python3 "$REPO_DIR/benchmark/scripts/ensure_gt.py" "${gt_args[@]}" 2> "$root/groundtruth.log")
[[ -f "$GT_BIN_KCACHED" ]] || error "GT preflight did not return a truth file"
export GT_BIN_KCACHED
printf '%s\n' "$GT_BIN_KCACHED" > "$root/selected_gt.txt"
for name in DATASET DATA_BIN QUERY_BIN EVAL_TEMPFILES EVAL_CACHED GT_ARTIFACT_DIR GT_BIN_KCACHED \
    QUERY_THREADS NTHREADS OMP_NUM_THREADS OMP_THREAD_LIMIT MKL_NUM_THREADS UPDATE_PERCENT UPDATE_BATCH ${!IVF_@}; do
    printf '%s=%q\n' "$name" "${!name-}"
done > "$root/settings.env"
printf 'TRAINING_ASSIGNMENT_SEARCH_LIMIT=3000000000\n' >> "$root/settings.env"

run_script() {
    local label="$1" script="$2"
    timeout -k 30 "$timeout_s" bash "$SCRIPT_DIR/sift/$script.sh" 2>&1 | tee "$root/$label.log"
}
if [[ "$mode" == workloads ]]; then
    for workload in query insert delete update; do
        run_script "$workload-prepare" "${workload}_prepare"
        run_script "$workload-run" "$workload"
    done
else
    schedule > "$root/grid.tsv"
    echo 'nlist,nprobe,router_breadth,recall_at_10_percent,query_qps,query_time_s,output_dir' > "$root/results.csv"
    current=""; tag_prefix=""
    if [[ "$mode" == sift100m-refine ]]; then
        tag_prefix="refine_$(printf '%s' "$root" | sha256sum | cut -c1-12)_"
    fi
    while read -r nlist probe breadth effective; do
        export IVF_NLIST="$nlist" EVAL_TEMPFILES="$root/nlist-$nlist" IVF_ROUTING_WORK_LIMIT=3000000000
        [[ "$nlist" != 400000 ]] || export IVF_ROUTING_WORK_LIMIT=5000000000
        if [[ "$mode" == sift100m-refine && "$nlist" != 400000 ]]; then
            export EVAL_TEMPFILES="$IVF_GRID_PREVIOUS_ROOT/nlist-$nlist"
        elif [[ "$current" != "$nlist" ]]; then
            run_script "nl$nlist-prepare" query_prepare
        fi
        current="$nlist"
        export IVF_NPROBE="$probe" IVF_ROUTER_BREADTH="$breadth" IVF_RUN_TAG="${tag_prefix}np${probe}_rb${breadth}"
        run_script "nl$nlist-np${probe}_rb${breadth}" query
        folder="$EVAL_TEMPFILES/$DATASET/IVF/query/runs/$IVF_RUN_TAG"
        python3 -c 'import csv,json,sys; d=json.load(open(sys.argv[1])); csv.writer(sys.stdout).writerow(
            [d[k] for k in ("nlist","nprobe","router_breadth","recall","query_qps","query_time_s")] + [sys.argv[2]])' \
            "$folder/result.json" "$folder" >> "$root/results.csv"
        rm -- "$folder/query.heap" "$folder/query.heap.pq.bin"
        printf '%s\n' "$folder" >> "$root/heap-cleanup.log"
    done < "$root/grid.tsv"
fi
note "Completed $mode: $root"
