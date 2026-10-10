#!/usr/bin/env python3
"""Bounded mixed-round preparation/reopen orchestration; no production acceptance."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import struct
import sys
from common import command_log, digest, header, query_truth_shapes, read_json, require, write_json
from harness import component_digests, prepare as prepare_base
from settings import prepared_selection, profile, query_router_flags


def counted_identity(path, first, count):
    require(path.stat().st_size == 4 + count * 4, "mixed counted-ID length mismatch")
    with path.open("rb") as source:
        require(source.read(4) == struct.pack("<I", count), "mixed counted-ID count mismatch")
        for offset in range(0, count, 4096):
            n = min(4096, count - offset)
            require(
                source.read(n * 4)
                == struct.pack("<" + "I" * n, *range(first + offset, first + offset + n)),
                "mixed IDs differ from canonical SPFresh round ranges",
            )
    return digest(path)


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
        "invalid mixed layout/equal-cap SPFresh comparison contract",
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


def extract_round(dest):
    # Reuse the unchanged shared canonical extractor before admitting the next
    # round or publishing a successful workload summary. Preserve failed output.
    target = dest / "result.json"
    require(not target.exists() and not target.is_symlink(), f"mixed round result exists: {target}")
    repo = Path(__file__).resolve().parents[2]
    command_log(
        [
            "bash",
            "-c",
            'set -euo pipefail\nREPO_DIR="$1"\n'
            'source "$REPO_DIR/benchmark/scripts/utils.sh"\n'
            'write_ann_bench_metrics_json "$2" "$3"',
            "ivf-mixed-extract",
            str(repo),
            str(dest / "run.log"),
            str(target),
        ],
        dest / "extract.log",
    )
    require(
        target.read_bytes() == (dest / "native_result.json").read_bytes(),
        "mixed canonical extraction mismatch",
    )


def run(args):
    w = args.work_dir.resolve()
    path = w / "mixed_prepared.json"
    require(path.stat().st_size <= 8 << 20, "oversized mixed preparation metadata")
    prepared = json.loads(path.read_text())
    require(prepared == identity(args), "mixed layout/source/IDs/GT changed since prepare")
    base = read_json(w / "prepared.json")
    selection = prepared_selection(base)
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
    for name in ("rounds", "result.json", "summary.json", "nlist_selection.json"):
        dest = output / name
        require(not dest.exists() and not dest.is_symlink(), f"mixed output exists: {dest}")
    scheduler = os.environ.get("IVF_SCHEDULER", "weighted")
    threads = os.environ.get("NTHREADS", "16")
    require(
        scheduler in ("weighted", "fixed_8_8")
        and threads.isdigit()
        and 0 < int(threads) <= 256
        and (scheduler != "fixed_8_8" or int(threads) == 16),
        "invalid mixed scheduler/foreground allocation",
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "nlist_selection.json", selection)
    (output / "rounds").mkdir()
    records = []
    for r in range(args.rounds):
        dest = output / f"rounds/round_{r}"
        dest.mkdir()
        (dest / "checkpoint").mkdir()
        saved = dest / "checkpoint" / args.dataset
        command = [
            os.environ["IVF_DRIVER_BIN"],
            "--validate_inputs",
            "0",
            "--data_type",
            args.dtype,
            "--metric",
            args.metric,
            "--dim",
            str(args.dim),
            "--nlist",
            base["settings"]["nlist"],
            "--index_prefix",
            str(prefix),
            "--index_format",
            "mutable" if r else "immutable",
            "--workload",
            "mixed",
            "--scheduler",
            scheduler,
            "--pool_threads",
            threads,
            "--query_ratio",
            os.environ.get("QUERY_RATIO", "80"),
            "--scheduler_seed",
            os.environ.get("IVF_SCHEDULER_SEED", str(0xA5A5A5A5)),
            "--full_data_file",
            str(w / "data/full.bin"),
            "--query_file",
            str(w / "data/query.bin"),
            "--insert_ids_file",
            str(w / f"ids/round_{r}_insert.bin"),
            "--delete_ids_file",
            str(w / f"ids/round_{r}_delete.bin"),
            "--gt_file",
            str(w / f"groundtruth/rounds/round_{r}_gt{args.recall_at}.bin"),
            "--recall_at",
            str(args.recall_at),
            "--nprobe",
            os.environ.get("IVF_NPROBE", "1"),
            "--rerank_factor",
            os.environ.get("IVF_RERANK_FACTOR", "2"),
            "--query_threads",
            os.environ.get("QUERY_THREADS", "1"),
            "--private_heap_path",
            str(dest / "private.heap"),
            "--checkpoint_prefix",
            str(saved),
            "--buffer_frames",
            os.environ.get("IVF_BUFFER_FRAMES", "64"),
            "--memory_budget_gb",
            os.environ.get("IVF_MEMORY_BUDGET_GB", "1"),
            "--result_file",
            str(dest / "native_result.json"),
            "--search_result_file",
            str(dest / "neighbors.bin"),
        ]
        if args.query_count is not None:
            command += ["--query_count", str(args.query_count)]
        command += query_router_flags()
        command_log(command, dest / "run.log")
        record = read_json(dest / "native_result.json")
        require(
            record["workload"] == "mixed"
            and record["insert_count"] == args.insert_cap
            and record["delete_count"] == args.delete_cap
            and "recall" in record
            and record["live_count"]
            == args.base_count + (r + 1) * (args.insert_cap - args.delete_cap),
            "mixed round execution count/live-set/scoring mismatch",
        )
        extract_round(dest)
        records.append(record)
        prefix = saved
    require(
        component_digests(w / "index", args.dataset) == base_components
        and prepared == identity(args),
        "mixed source/base/input identity changed during run",
    )
    with (output / "result.json").open("xb") as final:
        final.write((output / f"rounds/round_{args.rounds - 1}/native_result.json").read_bytes())
    write_json(
        output / "summary.json",
        {
            "version": 1,
            "production_qualified": False,
            "round_records": records,
            "final_recall": records[-1]["recall"],
            "mixed_wall_s": sum(r["mixed_wall_s"] for r in records),
            "concurrent_query_count": sum(r["concurrent_query_count"] for r in records),
            "overlapping_query_count": sum(r["overlapping_query_count"] for r in records),
            "scope": "separate_process_rounds_checkpoint_reopen_and_post_barrier_scoring_excluded_from_foreground_time",
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
