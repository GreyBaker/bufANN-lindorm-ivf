#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mutation_profile.sh"
[[ -f "$WORK_DIR/prepared.json" && -f "$WORK_DIR/mutation_prepared.json" ]] || error "missing completed mutation preparation"
OUTPUT_DIR="$WORK_DIR"
if [[ -n "${IVF_RUN_TAG:-}" ]]; then
    [[ "$IVF_RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || error "unsafe IVF_RUN_TAG"
    OUTPUT_DIR="$WORK_DIR/runs/$IVF_RUN_TAG"
fi
ivf_profile_args
python3 "$REPO_DIR/benchmark/IVF/harness.py" run "${PROFILE_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" --recall-at "$RECALL_AT"
write_ann_bench_metrics_json "$OUTPUT_DIR/mutation.log" "$OUTPUT_DIR/mutation_metrics.json"
cmp -s "$OUTPUT_DIR/mutation_result.json" "$OUTPUT_DIR/mutation_metrics.json" || error "canonical mutation metric extraction mismatch"
write_ann_bench_metrics_json "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/result.json"
cmp -s "$OUTPUT_DIR/native_result.json" "$OUTPUT_DIR/result.json" || error "canonical metric extraction mismatch"
