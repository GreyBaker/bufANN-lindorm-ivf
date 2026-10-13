#!/usr/bin/env python3
"""Tiny complete populations: explicit input policies and unchanged query timing."""
import argparse
import hashlib
import json
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
    build = [args.build, "--validate_inputs", "0", "--data_type", dtype,
             "--data_file", str(data), "--pq_prefix", str(pq), "--index_prefix", str(prefix),
             "--dim", "3", "--base_count", "8", "--nlist", "2", "--max_reps", "2",
             "--build_threads", "2", "--query_breadth", "2", "--insert_breadth", "2"]
    run(build, "build")
    source_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in work.glob("base.*")}
    insert_ids, delete_ids, empty = work / "insert.ids", work / "delete.ids", work / "empty.ids"
    ids(insert_ids, reversed(range(8, 64)))
    ids(delete_ids, [0])
    ids(empty, [])
    ordinal = 0

    def update(policy, workers=1, frames=2, workload="insert_only", extra=()):
        nonlocal ordinal
        ordinal += 1
        saved, private, result = work / f"saved-{ordinal}", work / f"private-{ordinal}.heap", work / f"result-{ordinal}.json"
        cmd = [args.driver, "--validate_inputs", "0", "--data_type", dtype,
               "--index_prefix", str(prefix), "--workload", workload, "--full_data_file", str(data),
               "--private_heap_path", str(private), "--checkpoint_prefix", str(saved),
               "--buffer_frames", str(frames), "--insert_threads", str(workers), "--result_file", str(result)]
        if policy is not None:
            cmd += ["--insert_input_policy", policy]
        return cmd + list(extra), saved, private, result

    def oracle(live):
        return [sorted([(i, sum((a-b)**2 for a, b in zip(q, vectors[i]))) for i in live],
                       key=lambda hit: (hit[1], hit[0])) for q in queries]

    def check_query(record):
        assert record["explicit_warmup_query_count"] == 0
        assert "no_explicit_warmup" in record["query_cache_process_policy"]
        assert record["cache_state_claim"] == "open_validation_may_touch_pages_os_cache_not_controlled"
        assert "excludes_gt_and_export" in record["query_timing_scope"]
        assert record["driver_workflow_time_s"] >= record["open_time_s"] + record["query_time_s"]

    def search(saved, live):
        nonlocal ordinal
        ordinal += 1
        truth, output = work / f"truth-{ordinal}.bin", work / f"neighbors-{ordinal}.bin"
        gold = oracle(live)
        write_truth(truth, gold)
        record = run([args.driver, "--validate_inputs", "0", "--data_type", dtype,
                      "--index_prefix", str(saved), "--index_format", "mutable",
                      "--private_heap_path", str(work / f"query-{ordinal}.heap"),
                      "--workload", "search_only", "--query_file", str(query), "--gt_file", str(truth),
                      "--nprobe", "2", "--recall_at", "4", "--rerank_factor", "16", "--query_threads", "2",
                      "--buffer_frames", "2", "--search_result_file", str(output)], "query")
        check_query(record)
        assert record["query_cache_process_policy"].startswith("private_mutable_copy")
        assert record["recall"] == 100 and record["live_count"] == len(live)
        assert read_results(output) == [row[:4] for row in gold]
        return output.read_bytes()

    def check_update(record, policy, count=56):
        assert record["insert_count"] == count and record["insert_input_policy"] == policy
        assert record["insert_timing_contract_version"] == (2 if policy == "staged" else 1)
        assert record["insert_staged_coordinate_bytes"] == (count * 3 * struct.calcsize(fmt) if policy == "staged" else 0)
        assert record["insert_input_preparation_time_s"] == record["coordinate_validation_or_staging_time_s"]
        assert "not_additive" in record["insert_input_preparation_time_scope"]
        assert ("no_source_reads" in record["insert_time_scope"]) == (policy == "staged")
        assert record["mutation_completion_policy"].startswith("joined_publications")
        assert "not_wal_or_crash_durability" in record["checkpoint_completion_scope"]
        assert record["driver_workflow_time_s"] >= (
            record["open_time_s"] + record["insert_time_s"] + record["checkpoint_time_s"] + record["insert_input_preparation_time_s"])
        assert "before_canonical_record_write_not_process_lifetime" in record["driver_workflow_time_scope"]

    first = None
    stream_allowance = None
    for policy in (None, "staged"):
        actual = policy or "streaming"
        for frames in (2, 64):
            for workers in (1, 8):
                cmd, saved, _, output = update(policy, workers, frames, extra=["--insert_ids_file", str(insert_ids)])
                record = run(cmd, "insert")
                assert json.loads(output.read_text()) == record
                check_update(record, actual)
                assert record["insert_threads_effective"] == workers
                assert record["live_count"] == record["retained_heap_records"] == 64
                if policy is None and frames == 2 and workers == 1:
                    stream_allowance = record["estimated_update_workspace_bytes"]
                observed = search(saved, range(64))
                first = observed if first is None else first
                assert observed == first
    # Synthetic budget control: hold the streaming allowance fixed, then refuse
    # the additional staged payload before loading/mutation, without reducing rows.
    budget = format((stream_allowance + 64) / (1 << 30), ".17g")
    cmd, _, _, _ = update("streaming", extra=["--insert_ids_file", str(insert_ids), "--memory_budget_gb", budget])
    run(cmd, "insert")
    cmd, saved, _, output = update("staged", extra=["--insert_ids_file", str(insert_ids), "--memory_budget_gb", budget])
    failed = run(cmd)
    assert "staged insertion exceeds workspace allowance" in failed.stderr
    assert "phase=update_input_validation event=failed" in failed.stderr
    assert not output.exists() and not list(work.glob(saved.name + ".*"))
    for policy in ("streaming", "staged"):
        cmd, saved, _, _ = update(policy, workers=8, extra=["--insert_ids_file", str(empty)])
        record = run(cmd, "insert")
        check_update(record, policy, 0)
        assert record["insert_threads_effective"] == 0
        search(saved, range(8))
        truth, result = work / f"combined-{policy}.gt", work / f"combined-{policy}.bin"
        gold = oracle(range(1, 64))
        write_truth(truth, gold)
        cmd, saved, _, _ = update(policy, workers=8, workload="search_after_update",
            extra=["--insert_ids_file", str(insert_ids), "--delete_ids_file", str(delete_ids),
                   "--query_file", str(query), "--gt_file", str(truth), "--nprobe", "2", "--recall_at", "4",
                   "--rerank_factor", "16", "--search_result_file", str(result)])
        record = run(cmd, "query")
        check_update(record, policy)
        check_query(record)
        assert "same_process_live_mutable_session_after_checkpoint_no_reopen" in record["query_cache_process_policy"]
        assert record["recall"] == 100 and record["live_count"] == 63
        assert read_results(result) == [row[:4] for row in gold]
        search(saved, range(1, 64))
    for policy in ("auto", "unknown"):
        cmd, saved, private, output = update(policy)
        failed = run(cmd)
        assert "streaming or staged" in failed.stderr and not private.exists() and not output.exists()
    for workload in ("delete_only", "mixed", "search_only"):
        cmd, _, private, output = update("staged", workload=workload)
        # Remove the unrelated worker option so the input-policy refusal is tested.
        pos = cmd.index("--insert_threads")
        del cmd[pos:pos+2]
        failed = run(cmd)
        assert "insert_input_policy only applies" in failed.stderr and not private.exists() and not output.exists()
    assert hashlib.sha256(data.read_bytes()).hexdigest() == source_hash
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in originals.items())
    if dtype == "float":
        bad_source = work / "nonfinite-source.bin"
        raw = bytearray(data.read_bytes())
        struct.pack_into("<f", raw, 8 + 63 * 3 * 4, float("nan"))
        bad_source.write_bytes(raw)
        bad_base = work / "nonfinite-base"
        bad_build = build.copy()
        bad_build[bad_build.index("--data_file") + 1] = str(bad_source)
        bad_build[bad_build.index("--index_prefix") + 1] = str(bad_base)
        run(bad_build, "build")  # Unindexed tail; training/indexed rows are unchanged.
        for policy in ("streaming", "staged"):
            cmd, saved, _, output = update(policy, extra=["--insert_ids_file", str(insert_ids)])
            cmd[cmd.index("--index_prefix") + 1] = str(bad_base)
            cmd[cmd.index("--full_data_file") + 1] = str(bad_source)
            failed = run(cmd)
            assert "Nonfinite insert coordinate" in failed.stderr
            assert ("phase=coordinate_staging event=failed" if policy == "staged"
                    else "phase=coordinate_validation event=failed") in failed.stderr
            assert "phase=insert event=begin" not in failed.stderr
            assert not output.exists() and not list(work.glob(saved.name + ".*"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    root = attempt(args.scratch)
    for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
        fixture(args, root, dtype, fmt)
    print("Insertion input policies, full exact recall, budgets, process/timing scopes and failed staging passed")


if __name__ == "__main__":
    main()
