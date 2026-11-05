#!/usr/bin/env python3
"""Bounded SIFT100M all-lists/vector-rerank diagnostic; no training or GT generation."""

import argparse
import json
import os
from pathlib import Path
import signal
import struct
import sys

from audit_recall import audit
from common import digest, execute, flags, header, positive, read_json, require, require_absent, write_json


def query_prefix(source, output, count):
    require(1 <= count <= 10000, "diagnostic query count must be in [1, 10000]")
    header(source, "uint8", 10000, 128)
    with source.open("rb") as file:
        file.seek(8)
        data = file.read(count * 128)
    require(len(data) == count * 128, "truncated diagnostic queries")
    with output.open("xb") as file:
        file.write(struct.pack("<II", count, 128))
        file.write(data)


def run(args):
    require_absent(args.output, f"preserve the existing diagnostic root: {args.output}")
    previous = args.previous_root.resolve(strict=True)
    work = previous / "nlist-100/sift100m/IVF/query"
    prepared = read_json(work / "prepared.json")
    require(prepared["profile"]["base_count"] == 90000000 and int(prepared["settings"]["nlist"]) == 100,
            "diagnostic requires the unchanged SIFT100M 90M-live, 100-list base")
    require(digest(args.query) == prepared["query_identity"], "diagnostic queries differ from prepared inputs")
    require(args.driver.is_file() and os.access(args.driver, os.X_OK), "missing native IVF driver")
    require(1 <= args.query_count <= 10000, "diagnostic query count must be in [1, 10000]")
    header(args.query, "uint8", 10000, 128)
    truth = Path((previous / "selected_gt.txt").read_text().strip())
    require(truth.is_file(), "missing selected GT binary")
    args.output.mkdir(parents=True, exist_ok=False)
    subset = args.output / f"query{args.query_count}.bin"
    query_prefix(args.query, subset, args.query_count)
    write_json(args.output / "plan.json", dict(query=str(args.query.resolve()), query_identity=digest(args.query),
               index_prefix=str(work / "index/sift100m"), deep_gt=str(truth), query_count=args.query_count,
               rerank_factors=[3, 10, 30], nprobe=100, timeout_s_per_pass=args.timeout_s))
    for factor in (3, 10, 30):
        folder = args.output / f"rf{factor}"
        folder.mkdir()
        command = [str(args.driver.resolve())] + flags(dict(
            validate_inputs=0, data_type="uint8", metric="l2", workload="search_only",
            index_prefix=work / "index/sift100m", private_heap_path=folder / "query.heap",
            query_file=subset, nprobe=100, recall_at=10, rerank_factor=factor,
            router_mode="pca-refined", router_rerank_factor=3, router_breadth=128,
            query_threads=8, buffer_frames=262144, memory_budget_gb=96, warmup_queries=64,
            result_file=folder / "result.json", search_result_file=folder / "neighbors.bin"))
        execute(command, folder / "run.log", args.timeout_s, None)
        result = read_json(folder / "result.json")
        require(result["query_count"] == args.query_count and result["nprobe"] == 100 and
                result["rerank_factor"] == factor and result["live_count"] == 90000000 and
                result["scanned_codes"] == args.query_count * 90000000,
                "diagnostic did not scan the complete live base at the declared settings")
        report = audit(truth, folder / "neighbors.bin", 90000000, k=10, query_offset=0)
        report.update(deep_gt=str(truth.resolve()), results=str((folder / "neighbors.bin").resolve()))
        write_json(folder / "audit.json", report)
        for name in ("query.heap", "query.heap.pq.bin"):
            heap = folder / name
            size = heap.stat().st_size
            heap.unlink()
            with (args.output / "heap-cleanup.jsonl").open("a") as log:
                log.write(json.dumps(dict(phase=f"rf{factor}", path=str(heap), bytes=size, deleted=True)) + "\n")
        print(f"rf{factor}: {folder / 'audit.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--query", type=Path, default=os.environ.get("QUERY_BIN"))
    build = Path(os.environ.get("IVF_BUILD_DIR", Path(__file__).resolve().parents[2] / "build/ivf-benchmark"))
    p.add_argument("--driver", type=Path, default=os.environ.get("IVF_DRIVER_BIN", str(build / "tests/ivf_driver")))
    p.add_argument("--query-count", type=positive, default=256)
    p.add_argument("--timeout-s", type=positive, default=21600)
    args = p.parse_args()
    try:
        require(args.query is not None, "pass --query or export QUERY_BIN")
        run(args)
    except (OSError, ValueError) as error:
        p.exit(1, f"ERROR: {error}\n")


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
