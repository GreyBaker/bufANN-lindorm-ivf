#!/usr/bin/env python3
"""Same-process centroid warmup controls; final population/query oracle unchanged."""
import argparse
import hashlib
from pathlib import Path
import struct

from ivf_test_support import (
    build_tiny_population, read_results, reserve_attempt as attempt,
    run_native_cli as run, write_truth,
)


def fixture(args, root, dtype, fmt):
    population = build_tiny_population(args.build, root, dtype, fmt,
        reverse_insert_ids=False, include_empty_ids=False, run_build=run)
    work, vectors, queries = population.work, population.vectors, population.queries
    data, query, prefix = population.data, population.query, population.prefix
    source_hash, originals = population.source_hash, population.originals
    insert_ids, delete_ids = population.insert_ids, population.delete_ids
    ordinal = 0

    def command(warmups, origin=prefix, live=range(8), update=False):
        nonlocal ordinal
        ordinal += 1
        truth, output, result = work / f"truth-{ordinal}.bin", work / f"neighbors-{ordinal}.bin", work / f"result-{ordinal}.json"
        gold = [sorted([(i, sum((a-b)**2 for a, b in zip(q, vectors[i]))) for i in live],
                       key=lambda hit: (hit[1], hit[0])) for q in queries]
        write_truth(truth, gold)
        private, saved = work / f"private-{ordinal}.heap", work / f"saved-{ordinal}"
        cmd = [args.driver, "--validate_inputs", "0", "--data_type", dtype, "--index_prefix", str(origin),
               "--workload", "search_after_update" if update else "search_only", "--query_file", str(query),
               "--gt_file", str(truth), "--nprobe", "2", "--recall_at", "4", "--rerank_factor", "16",
               "--query_threads", "2", "--buffer_frames", "2", "--result_file", str(result), "--search_result_file", str(output)]
        if warmups is not None:
            cmd += ["--warmup_queries", str(warmups)]
        if update:
            cmd += ["--full_data_file", str(data), "--insert_ids_file", str(insert_ids), "--delete_ids_file", str(delete_ids),
                    "--private_heap_path", str(private), "--checkpoint_prefix", str(saved), "--insert_threads", "8",
                    "--insert_input_policy", "staged"]
        elif origin != prefix:
            cmd += ["--index_format", "mutable", "--private_heap_path", str(private)]
        return cmd, output, result, saved, private, gold

    def check(record, count):
        assert record["query_count"] == 8 and record["logical_fetch_stages"] == 8
        assert record["explicit_warmup_query_count"] == record["warmup_query_count_requested"] == count
        assert record["warmup_policy"] == ("frozen_base_centroid_grid_v1" if count else "none")
        assert record["warmup_row_allowance_bytes"] == (3 * struct.calcsize(fmt) + 64 if count else 0)
        assert (record["warmup_time_s"] > 0) if count else (record["warmup_time_s"] == 0)
        assert "not_final_queries_or_truth" in record["warmup_source_policy"]
        assert "not_in_final_query_stats" in record["warmup_results_policy"]
        assert "no_query_worker_tls_continuity_claim" in record["warmup_cache_scope"]
        assert record["driver_workflow_time_s"] >= record["open_time_s"] + record["warmup_time_s"] + record["query_time_s"]
        assert record["recall"] == 100

    first = None
    base_allowance = None
    for count in (None, 0, 5):
        cmd, output, _, _, _, gold = command(count)
        record = run(cmd, "query")
        check(record, count or 0)
        assert read_results(output) == [row[:4] for row in gold]
        raw = output.read_bytes()
        first = raw if first is None else first
        assert raw == first
        if count is None:
            base_allowance = record["estimated_workspace_bytes"]
    budget = format((base_allowance + 8) / (1 << 30), ".17g")
    for count in (0, 5):
        cmd, output, _, saved, private, gold = command(count)
        record = run(cmd + ["--private_heap_path", str(private)], "query")
        check(record, count)
        assert record["index_format"] == "mutable" and record["input_index_format"] == "immutable"
        assert record["live_count"] == 8
        assert record["query_cache_process_policy"].startswith("private_mutable_copy")
        assert "insert_count" not in record and "delete_count" not in record
        assert output.read_bytes() == first
        assert not Path(str(saved) + ".manifest").exists()
    cmd, _, _, _, _, _ = command(0)
    run(cmd + ["--memory_budget_gb", budget], "query")
    cmd, output, result, _, _, _ = command(5)
    failed = run(cmd + ["--memory_budget_gb", budget])
    assert "composed search exceeds workspace" in failed.stderr and not output.exists() and not result.exists()
    updated_first = None
    for count in (0, 7):
        cmd, output, _, saved, _, gold = command(count, live=range(1, 64), update=True)
        record = run(cmd, "query")
        check(record, count)
        assert record["insert_count"] == 56 and record["delete_count"] == 1
        assert "same_process_live_mutable_session_after_checkpoint_no_reopen" in record["query_cache_process_policy"]
        assert read_results(output) == [row[:4] for row in gold]
        updated_first = output.read_bytes() if updated_first is None else updated_first
        assert output.read_bytes() == updated_first
        reopen, output, _, _, _, gold = command(count, origin=saved, live=range(1, 64))
        record = run(reopen, "query")
        check(record, count)
        assert record["query_cache_process_policy"].startswith("private_mutable_copy")
        assert read_results(output) == [row[:4] for row in gold]
    for count in (-1, 100001, "invalid"):
        cmd, output, result, _, private, _ = command(count)
        run(cmd)
        assert not output.exists() and not result.exists() and not private.exists()
    for workload in ("insert_only", "delete_only"):
        cmd, output, result, _, private, _ = command(0)
        cmd[cmd.index("--workload") + 1] = workload
        failed = run(cmd)
        assert "only applies to query-producing" in failed.stderr
        assert not output.exists() and not result.exists() and not private.exists()
    assert hashlib.sha256(data.read_bytes()).hexdigest() == source_hash
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in originals.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    root = attempt(args.scratch)
    for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
        fixture(args, root, dtype, fmt)
    print("Warmup zero/positive controls, full exact query population, checkpoint/reopen, timing and memory refusals passed")


if __name__ == "__main__":
    main()
