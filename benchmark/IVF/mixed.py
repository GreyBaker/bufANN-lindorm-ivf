#!/usr/bin/env python3
"""One persistent native mixed session; final stable-state recall only."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time
from common import (
    command_log, counted_range_identity, digest, header, is_query_record,
    query_truth_shapes, read_json, require, require_absent, write_json,
)
from harness import ROUTER_BACKEND_ID, component_digests, driver_open_command, prepare as prepare_base
from settings import (
    prepared_selection, profile, query_policy_flags, query_router_flags, query_search_flags,
    runtime_resources, warmup_queries,
)


def counted_identity(path, first, count):
    return counted_range_identity(
        path, first, count,
        framing_message="mixed counted-ID length mismatch",
        count_message="mixed counted-ID count mismatch",
        range_message="mixed IDs differ from prepared IVF source-row ranges",
    )


def identity(args):
    w = args.work_dir.resolve()
    require(
        args.metric == "l2"
        and 0 < args.rounds <= 4096
        and args.insert_cap == args.delete_cap
        and args.insert_count == args.rounds * args.insert_cap
        and args.delete_count == args.rounds * args.delete_cap
        and 0 < args.insert_count <= args.full_count - args.base_count
        and 0 < args.delete_count < args.base_count,
        "invalid IVF mixed layout/equal-cap round contract",
    )
    header(w / "data/full.bin", args.dtype, args.full_count, args.dim)
    result = {
        "version": 1,
        "profile": profile(args),
        "rounds": args.rounds,
        "insert_count": args.insert_count,
        "delete_count": args.delete_count,
        "insert_cap": args.insert_cap,
        "delete_cap": args.delete_cap,
        "source": digest(w / "data/full.bin"),
        "inputs": [],
    }
    for r in range(args.rounds):
        gt = w / f"groundtruth/rounds/round_{r}_gt{args.recall_at}.bin"
        query_truth_shapes(w / "data/query.bin", gt, args.dtype, args.dim, args.recall_at)
        result["inputs"].append(
            {
                "insert": counted_identity(
                    w / f"ids/round_{r}_insert.bin",
                    args.base_count + r * args.insert_cap,
                    args.insert_cap,
                ),
                "delete": counted_identity(
                    w / f"ids/round_{r}_delete.bin",
                    args.base_count - args.delete_count + r * args.delete_cap,
                    args.delete_cap,
                ),
                "truth": digest(gt),
            }
        )
    return result


def prepare(args):
    expected = identity(args)
    base = copy.copy(args)
    base.workload = "query"
    base.update_count = 0
    prepare_base(base)  # Existing immutable base-only trainer/PQ/cache contract.
    write_json(args.work_dir / "mixed_prepared.json", expected)


def run(args):
    workflow_start = time.monotonic()
    w = args.work_dir.resolve()
    base = read_json(w / "prepared.json")
    resources = runtime_resources(base["settings"]["page_size"])
    query_warmups = warmup_queries()
    path = w / "mixed_prepared.json"
    prepared = read_json(path, max_bytes=8 << 20)
    require(prepared == identity(args), "mixed layout/source/IDs/GT changed since prepare")
    selection = prepared_selection(base)
    require(base.get("router_backend") == ROUTER_BACKEND_ID,
            "prepared router backend is incompatible; prepare a fresh FAISS base")
    require(
        base["profile"] == profile(args) and digest(base["query"]) == base["query_identity"],
        "mixed query/profile changed",
    )
    prefix = w / "index" / args.dataset
    require(
        digest(Path(str(prefix) + ".manifest")) == base["manifest_identity"],
        "mixed base manifest changed",
    )
    base_components = component_digests(w / "index", args.dataset)
    output = args.output_dir.resolve()
    require(output == w or w in output.parents, "mixed outputs must stay under prepared work")
    for name in ("rounds", "result.json", "summary.json", "nlist_selection.json",
                 "round_schedule.txt", "private.heap", "private.heap.pq.bin", "neighbors.bin", "run.log"):
        dest = output / name
        require_absent(dest, f"mixed output exists: {dest}")
    scheduler = os.environ.get("IVF_SCHEDULER", "update_driven")
    threads = os.environ.get("NTHREADS", "16")
    require(
        scheduler in ("update_driven", "weighted", "fixed_8_8")
        and threads.isdigit()
        and 0 < int(threads) <= 256
        and (scheduler != "fixed_8_8" or int(threads) == 16),
        "invalid mixed scheduler/foreground allocation",
    )
    if scheduler == "update_driven":
        require(
            int(threads) >= 2
            and args.query_count is None
            and not any(os.environ.get(key) for key in ("QUERY_RATIO", "IVF_SCHEDULER_SEED")),
            "update_driven needs at least two workers and no query ratio/count/seed overrides",
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "nlist_selection.json", selection)
    def quoted(path):
        text = str(path)
        require(not any(c in text for c in "\n\r\0"), "invalid schedule path")
        return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'

    schedule = output / "round_schedule.txt"
    schedule.write_text("".join(
        f'{quoted(w / f"ids/round_{r}_insert.bin")} {quoted(w / f"ids/round_{r}_delete.bin")}\n'
        for r in range(args.rounds)), encoding="utf-8")
    command = driver_open_command(args, base["settings"]["nlist"], prefix) + [
        "--index_format", "immutable", "--workload", "mixed",
        "--scheduler", scheduler, "--pool_threads", threads,
        "--full_data_file", str(w / "data/full.bin"),
        "--query_file", str(w / "data/query.bin"),
        "--round_schedule", str(schedule), "--round_results_dir", str(output / "rounds"),
        "--gt_file", str(w / f"groundtruth/rounds/round_{args.rounds - 1}_gt{args.recall_at}.bin"),
        "--recall_at", str(args.recall_at),
    ] + query_search_flags() + [
        "--private_heap_path", str(output / "private.heap"),
        "--buffer_frames", resources["buffer_frames"],
        "--memory_budget_gb", resources["memory_budget_gb"],
        "--result_file", str(output / "result.json"),
        "--search_result_file", str(output / "neighbors.bin"),
    ]
    if scheduler != "update_driven":
        command += ["--query_ratio", os.environ.get("QUERY_RATIO", "80"),
                    "--scheduler_seed", os.environ.get("IVF_SCHEDULER_SEED", str(0xA5A5A5A5))]
    if args.query_count is not None:
        command += ["--query_count", str(args.query_count)]  # finite quota per round
    command += query_policy_flags(query_warmups)
    process_start = time.monotonic()
    command_log(command, output / "run.log")
    process_wall_s = time.monotonic() - process_start
    record = read_json(output / "result.json")
    require(record["explicit_warmup_query_count"] == int(query_warmups),
            "native mixed warmup completion count mismatch")
    require(is_query_record(record, "mixed") and record["mixed_protocol_version"] == 3
            and record["mixed_round_count"] == args.rounds
            and record["insert_count"] == args.insert_count
            and record["delete_count"] == args.delete_count
             and record["live_count"] == args.base_count + args.insert_count - args.delete_count
             and record["update_protocol_version"] == 4
             and record["writeback_dirty_pages_after"] == 0
            and "recall" in record,
             "mixed execution count/live-set/scoring mismatch")
    round_results = []
    totals = {"concurrent_query_count": 0, "overlapping_query_count": 0}
    for r in range(args.rounds):
        path = f"rounds/round_{r}.json"
        part = read_json(output / path)
        require(part["version"] == 3 and part["round"] == r
                and part["insert_count"] == args.insert_cap
                and part["delete_count"] == args.delete_cap
                and part["recall_available"] is False,
                "mixed round accounting mismatch")
        round_results.append(path)
        for key in totals:
            totals[key] += part[key]
    require(all(totals[key] == record[key] for key in totals), "mixed query accounting mismatch")
    require(
        component_digests(w / "index", args.dataset) == base_components
        and prepared == identity(args),
        "mixed source/base/input identity changed during run",
    )
    write_json(
        output / "summary.json",
        {
            "version": 3,
            "production_qualified": False,
            "round_results": round_results,
            "final_recall": record["recall"],
            "mixed_wall_s": record["mixed_wall_s"],
            "writeback_time_s": record["writeback_time_s"],
            "writeback_heap_page_writes": record["writeback_heap_page_writes"],
            "writeback_heap_flush_bytes": record["writeback_heap_flush_bytes"],
            "writeback_dirty_pages_after": record["writeback_dirty_pages_after"],
            "native_driver_process_count": 1,
            "native_process_wall_s": process_wall_s,
            "harness_workflow_wall_s": time.monotonic() - workflow_start,
            **totals,
            "scope": "one_process_one_session_persistent_workers_final_scoring_only_no_checkpoints",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--dtype", choices=("float", "int8", "uint8"), required=True)
    p.add_argument("--metric", required=True)
    for field in (
        "dim",
        "full-count",
        "base-count",
        "insert-count",
        "delete-count",
        "insert-cap",
        "delete-cap",
        "rounds",
        "recall-at",
    ):
        p.add_argument("--" + field, type=int, required=True)
    p.add_argument("--query-count", type=int)
    p.add_argument("--workload", choices=("update",), default="update")
    p.add_argument("--update-count", type=int, default=0)
    for field in ("work-dir", "output-dir", "pq-prefix", "query", "truth", "cache-root"):
        p.add_argument("--" + field, type=Path, required=field == "work-dir")
    args = p.parse_args()
    try:
        require(
            re.fullmatch(r"[a-zA-Z0-9_-]+", args.dataset)
            and (args.query_count is None or args.query_count > 0),
            "invalid dataset/query count",
        )
        (prepare if args.command == "prepare" else run)(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
