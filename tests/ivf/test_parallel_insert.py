#!/usr/bin/env python3
"""Fixed tiny population: parallel insertion, full exact oracle, fresh artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

from ivf_test_support import (
    build_tiny_population, read_results, reserve_attempt as attempt,
    run_native_cli as run, write_ids as ids, write_truth,
)


def fixture(args, root, dtype, fmt):
    population = build_tiny_population(args.build, root, dtype, fmt,
        reverse_insert_ids=True, include_empty_ids=True, run_build=run)
    work, vectors, queries = population.work, population.vectors, population.queries
    data, query, prefix = population.data, population.query, population.prefix
    source_hash, originals = population.source_hash, population.originals
    insert_ids, delete_ids, empty = population.insert_ids, population.delete_ids, population.empty_ids
    ordinal = 0

    def update(workload, origin=prefix, extra=(), workers=None, frames=64):
        nonlocal ordinal
        ordinal += 1
        saved, private, result = (work / f"saved-{ordinal}", work / f"private-{ordinal}.heap",
                                  work / f"result-{ordinal}.json")
        command = [args.driver, "--validate_inputs", "0", "--data_type", dtype,
                   "--index_prefix", str(origin), "--workload", workload,
                   "--full_data_file", str(data), "--private_heap_path", str(private),
                   "--checkpoint_prefix", str(saved), "--buffer_frames", str(frames),
                   "--result_file", str(result)]
        if origin != prefix:
            command += ["--index_format", "mutable"]
        if workers is not None:
            command += ["--insert_threads", str(workers)]
        return command + list(extra), saved, private, result

    def search(saved, live):
        nonlocal ordinal
        ordinal += 1
        truth, output = work / f"truth-{ordinal}.bin", work / f"neighbors-{ordinal}.bin"
        oracle = [sorted([(i, sum((a-b)**2 for a, b in zip(q, vectors[i]))) for i in live],
                         key=lambda hit: (hit[1], hit[0])) for q in queries]
        write_truth(truth, oracle)
        record = run([args.driver, "--validate_inputs", "0", "--data_type", dtype,
                      "--index_prefix", str(saved), "--index_format", "mutable",
                      "--private_heap_path", str(work / f"query-{ordinal}.heap"),
                      "--workload", "search_only", "--query_file", str(query), "--gt_file", str(truth),
                      "--nprobe", "2", "--recall_at", "4", "--rerank_factor", "16",
                      "--query_threads", "2", "--buffer_frames", "2", "--search_result_file", str(output)], "query")
        assert record["recall"] == 100 and record["live_count"] == len(live)
        assert read_results(output) == [row[:4] for row in oracle]
        return output.read_bytes()

    first = None
    for frames in (2, 64):
        for workers in (1, 2, 8):
            cmd, saved, _, output = update("insert_only", workers=workers, frames=frames,
                                          extra=["--insert_ids_file", str(insert_ids)])
            record = run(cmd, "insert")
            assert json.loads(output.read_text()) == record
            assert record["insert_count"] == 56 and record["live_count"] == record["retained_heap_records"] == 64
            assert record["insert_threads_requested"] == record["insert_threads_effective"] == workers
            assert record["mutation_workers"] == workers and 1 <= record["insert_peak_active"] <= workers
            assert record["insert_time_s"] > 0 and record["insert_qps"] > 0
            assert "no_gt" in record["insert_time_scope"] and "distance_then_id" in record["insert_order_policy"]
            assert record["insert_worker_allowance_bytes"] > 0 and record["insert_list_growth_allowance_bytes"] > 0
            observed = search(saved, range(64))
            if first is None:
                first = observed
            assert observed == first  # Logical oracle, not byte-identical list/checkpoint layout.
    cmd, saved, _, _ = update("insert_only", workers=8, extra=["--insert_ids_file", str(empty)])
    record = run(cmd, "insert")
    assert record["insert_threads_effective"] == record["insert_count"] == record["insert_peak_active"] == 0
    search(saved, range(8))
    cmd, deleted, _, _ = update("delete_only", extra=["--delete_ids_file", str(delete_ids)])
    record = run(cmd, "delete")
    assert record["mutation_workers"] == 1 and record["insert_threads_effective"] == 0
    search(deleted, range(1, 8))
    # A retired ID passes 'not live' preflight, then reserve_insert rejects it.
    # Earlier valid inserts can finish, but failure must publish no checkpoint/result.
    bad = work / "partial-failure.ids"
    ids(bad, list(range(8, 64)) + [0])
    cmd, saved, private, result = update("insert_only", origin=deleted, workers=2,
                                        extra=["--insert_ids_file", str(bad)])
    failed = run(cmd)
    assert "event=failed" in failed.stderr and private.exists()
    assert not result.exists() and not list(work.glob(saved.name + ".*"))
    for workers in (0, 257):
        cmd, saved, private, result = update("insert_only", workers=workers,
                                            extra=["--insert_ids_file", str(insert_ids)])
        run(cmd)
        assert not private.exists() and not result.exists() and not list(work.glob(saved.name + ".*"))
    for workload in ("delete_only", "search_only", "mixed"):
        cmd, saved, private, result = update(workload, workers=2)
        failed = run(cmd)
        assert "only applies" in failed.stderr and not private.exists() and not result.exists()
    # Existing small serial allowance accepts one worker but refuses eight stacks
    # before mutation and publishes no success. No low-memory default is promoted.
    cmd, saved, _, _ = update("insert_only", workers=1,
                              extra=["--insert_ids_file", str(insert_ids), "--memory_budget_gb", "0.08"])
    run(cmd, "insert")
    cmd, saved, _, output = update("insert_only", workers=8,
                                   extra=["--insert_ids_file", str(insert_ids), "--memory_budget_gb", "0.08"])
    failed = run(cmd)
    assert "workers exceed workspace" in failed.stderr and not output.exists()
    assert not list(work.glob(saved.name + ".*"))
    # Sequential delete then joined parallel inserts, queries only after checkpoint.
    cmd, saved, _, _ = update("search_after_update", workers=8,
        extra=["--insert_ids_file", str(insert_ids), "--delete_ids_file", str(delete_ids),
               "--query_file", str(query), "--nprobe", "2", "--recall_at", "4", "--rerank_factor", "16"])
    record = run(cmd, "query")
    assert record["insert_count"] == 56 and record["delete_count"] == 1 and record["live_count"] == 63
    search(saved, range(1, 64))
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
    print("Parallel insertion native full-population, reopen, budgets and failure publication passed")


if __name__ == "__main__":
    main()
