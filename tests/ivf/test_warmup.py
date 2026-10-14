#!/usr/bin/env python3
"""Same-process centroid warmup controls; final population/query oracle unchanged."""
import argparse
import hashlib
from pathlib import Path
import struct

from test_cli import write_pivots
from test_cli_execution import run, read_results, write_truth
from test_parallel_insert import attempt, ids


def fixture(args, root, dtype, fmt):
    work = root / dtype
    work.mkdir()
    vectors = [tuple((i * 7 + d * 3) % 31 - (16 if dtype == "int8" else 0)
                     for d in range(3)) for i in range(64)]
    queries = vectors[::8]
    data, query, prefix = work / "source.bin", work / "queries.bin", work / "base"
    data.write_bytes(struct.pack("<II", 64, 3) + struct.pack("<192" + fmt, *(v for row in vectors for v in row)))
    query.write_bytes(struct.pack("<II", 8, 3) + struct.pack("<24" + fmt, *(v for row in queries for v in row)))
    pq = work / "pq"
    write_pivots(Path(str(pq) + "_pq_pivots.bin"))
    Path(str(pq) + "_pq_compressed.bin").write_bytes(struct.pack("<II", 64, 2) + bytes(128))
    run([args.build, "--validate_inputs", "0", "--data_type", dtype, "--data_file", str(data),
         "--pq_prefix", str(pq), "--index_prefix", str(prefix), "--dim", "3", "--base_count", "8",
         "--nlist", "2", "--max_reps", "2", "--build_threads", "2", "--query_breadth", "2", "--insert_breadth", "2"], "build")
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in work.glob("base.*")}
    source_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    insert_ids, delete_ids = work / "insert.ids", work / "delete.ids"
    ids(insert_ids, range(8, 64))
    ids(delete_ids, [0])
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
