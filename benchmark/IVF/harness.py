#!/usr/bin/env python3
"""IVF-only immutable query preparation/cache and single-pass run orchestration.

Trusted local files, SHA-256 identity/integrity (not authentication), no deletes,
no atomic publication/recovery. Native bundle validation remains authoritative.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

SUFFIXES = (".centroids.bin", ".hnsw", ".router", ".lists", ".heap", ".heap.meta",
            ".heap.meta.ivf", ".pq.bin", ".manifest")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        while chunk := source.read(8 << 20):
            h.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": h.hexdigest()}


def header(path, dtype, expected_rows, expected_dim):
    size = {"float": 4, "int8": 1, "uint8": 1}[dtype]
    with Path(path).open("rb") as source:
        raw = source.read(8)
    require(len(raw) == 8, f"invalid input header: {path}")
    rows, dim = struct.unpack("<II", raw)
    require((rows, dim) == (expected_rows, expected_dim), f"profile/input shape mismatch: {path}")
    require(Path(path).stat().st_size == 8 + rows * dim * size, f"input length mismatch: {path}")


def query_truth_shapes(query, truth, dtype, dim, recall_at):
    with Path(query).open("rb") as source:
        raw = source.read(8)
    require(len(raw) == 8, "invalid query header")
    rows, cols = struct.unpack("<II", raw)
    require(0 < rows <= 0x7FFFFFFF and cols == dim, "invalid query shape")
    header(query, dtype, rows, dim)
    with Path(truth).open("rb") as source:
        raw = source.read(8)
    require(len(raw) == 8, "invalid GT header")
    gt_rows, gt_cols = struct.unpack("<II", raw)
    require(gt_rows == rows and 0 < recall_at <= gt_cols <= 0x7FFFFFFF, "GT/query/recall shape mismatch")
    require(Path(truth).stat().st_size in (8 + rows * gt_cols * 8, 8 + rows * gt_cols * 12), "invalid GT length")


def read_json(path):
    require(Path(path).stat().st_size <= 65536, f"oversized harness metadata: {path}")
    return json.loads(Path(path).read_text())


def write_json(path, record):
    with Path(path).open("x") as out:
        json.dump(record, out, sort_keys=True, indent=2)
        out.write("\n")


def profile(args):
    return {key: getattr(args, key) for key in ("dataset", "dtype", "metric", "dim", "full_count", "base_count")}


def build_settings():
    defaults = {"nlist": None, "sample_rate": "1", "max_reps": "15", "seed": "0", "build_threads": "1",
                "memory_budget_gb": "1", "page_size": "4096", "buffer_frames": "64", "router_degree": "16",
                "construction_breadth": "200", "query_breadth": "128", "insert_breadth": "64", "router_seed": "100"}
    result = {key: os.environ.get("IVF_" + key.upper(), value) for key, value in defaults.items()}
    require(result["nlist"] is not None, "set IVF_NLIST explicitly")
    return result


def flags(settings):
    return [part for key, value in settings.items() for part in ("--" + key, str(value))]


def command_log(command, path):
    with Path(path).open("x") as log:
        log.write("Command: " + repr(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with process:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
        require(process.returncode == 0, f"command failed ({process.returncode}); inspect {path}")


def component_digests(directory, dataset):
    require(directory.is_dir() and not directory.is_symlink(), f"bundle directory must be real: {directory}")
    result = {}
    for suffix in SUFFIXES:
        name = dataset + suffix
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"bundle must contain real files: {path}")
        result[name] = digest(path)
    return result


def prepare(args):
    work = args.work_dir.resolve()
    data = work / "data/full.bin"
    require(args.metric == "l2" and 0 < args.base_count < args.full_count, "invalid profile metric/base split")
    header(data, args.dtype, args.full_count, args.dim)
    query_truth_shapes(args.query, args.truth, args.dtype, args.dim, args.recall_at)
    # Query row count is arbitrary, but dimension, framing and GT shape are
    # checked by the native driver before timed queries.
    settings = build_settings()
    pivots, codes = (Path(str(args.pq_prefix) + suffix) for suffix in ("_pq_pivots.bin", "_pq_compressed.bin"))
    builder = Path(os.environ["IVF_BUILD_BIN"]).resolve()
    libraries = [Path(os.environ["IVF_BUILD_DIR"]) / "src" / name for name in ("libBufANN.a", "libBufANN.so")]
    libraries = [path for path in libraries if path.is_file()]
    require(libraries, "IVF_BUILD_DIR must be a full Linux BufANN/routing build, not isolated contracts")
    identity = {"version": 1, "profile": profile(args), "settings": settings,
                "sources": [digest(data), digest(pivots), digest(codes)],
                "builder": digest(builder), "bufann_libraries": {path.name: digest(path) for path in libraries},
                "cache_tag": os.environ.get("IVF_CACHE_TAG", "")}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = args.cache_root.resolve() / args.dataset / "IVF" / key
    index = work / "index"
    index.mkdir()  # Do not overwrite partial or user work.
    prefix = index / args.dataset
    hit = cache.exists()
    if hit:
        require(cache.is_dir() and not cache.is_symlink(), "IVF cache entry must be a real directory")
        metadata = read_json(cache / "cache.json")
        require(metadata["identity"] == identity, "IVF cache identity mismatch")
        require(component_digests(cache / "index", args.dataset) == metadata["components"], "IVF cache integrity mismatch")
        for suffix in SUFFIXES:
            shutil.copyfile(cache / "index" / (args.dataset + suffix), Path(str(prefix) + suffix))
        require(component_digests(index, args.dataset) == metadata["components"], "IVF restore integrity mismatch")
        print(f"IVF cache restored: {key} (real copies, not fresh build metrics)")
    else:
        command = [str(builder), "--validate_inputs", "0", "--data_type", args.dtype, "--metric", args.metric,
                   "--dim", str(args.dim), "--base_count", str(args.base_count), "--data_file", str(data),
                   "--pq_prefix", str(args.pq_prefix), "--index_prefix", str(prefix)] + flags(settings)
        command_log(command, work / "build.log")
        components = component_digests(index, args.dataset)
        # Recheck identities after construction; native build additionally does
        # its own before/after source checks. No recovery of partial cache dirs.
        require(identity["sources"] == [digest(data), digest(pivots), digest(codes)], "source changed during prepare")
        if not os.environ.get("EVAL_CACHE_NO_STORE"):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.mkdir()
            (cache / "index").mkdir()
            for suffix in SUFFIXES:
                shutil.copyfile(Path(str(prefix) + suffix), cache / "index" / (args.dataset + suffix))
            require(component_digests(cache / "index", args.dataset) == components, "IVF cache copy integrity mismatch")
            write_json(cache / "cache.json", {"identity": identity, "components": components}) # last; not atomic/fsync
    write_json(work / "prepared.json", {"version": 1, "profile": profile(args), "settings": settings,
        "cache_key": key, "cache_hit": hit, "cache_policy": "sha256_inputs_tools_settings_real_bundle_copies",
        "query": str(args.query.resolve()), "query_identity": digest(args.query),
        "truth": str(args.truth.resolve()), "truth_identity": digest(args.truth),
        "manifest_identity": digest(Path(str(prefix) + ".manifest"))})


def run(args):
    work = args.work_dir.resolve()
    prepared = read_json(work / "prepared.json")
    require(prepared["version"] == 1 and prepared["profile"] == profile(args), "profile changed since prepare")
    require(digest(prepared["query"]) == prepared["query_identity"], "query changed since prepare")
    require(digest(prepared["truth"]) == prepared["truth_identity"], "GT changed since prepare")
    query_truth_shapes(prepared["query"], prepared["truth"], args.dtype, args.dim, args.recall_at)
    prefix = work / "index" / args.dataset
    require((work / "index").is_dir() and not (work / "index").is_symlink(), "working index directory must not symlink into cache")
    require(digest(Path(str(prefix) + ".manifest")) == prepared["manifest_identity"], "manifest changed since prepare")
    for suffix in SUFFIXES:
        require(not Path(str(prefix) + suffix).is_symlink(), "working bundle must not symlink into cache")
    output = args.output_dir.resolve()
    require(output == work or work in output.parents, "run output must stay under prepared work tree")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("run.log", "native_result.json", "result.json", "neighbors.bin"):
        path = output / name
        require(not path.exists() and not path.is_symlink(), f"run output exists: {path}; use a fresh IVF_RUN_TAG")
    command = [os.environ["IVF_DRIVER_BIN"], "--validate_inputs", "0", "--data_type", args.dtype,
        "--metric", args.metric, "--dim", str(args.dim), "--nlist", prepared["settings"]["nlist"],
        "--index_prefix", str(prefix), "--workload", "search_only", "--query_file", prepared["query"],
        "--gt_file", prepared["truth"], "--recall_at", str(args.recall_at),
        "--nprobe", os.environ.get("IVF_NPROBE", "1"), "--rerank_factor", os.environ.get("IVF_RERANK_FACTOR", "2"),
        "--query_threads", os.environ.get("QUERY_THREADS", "1"),
        "--buffer_frames", os.environ.get("IVF_BUFFER_FRAMES", "64"),
        "--memory_budget_gb", os.environ.get("IVF_QUERY_MEMORY_BUDGET_GB", os.environ.get("IVF_MEMORY_BUDGET_GB", "1")),
        "--result_file", str(output / "native_result.json"), "--search_result_file", str(output / "neighbors.bin")]
    command_log(command, output / "run.log")
    record = read_json(output / "native_result.json")
    require(record["baseline"] == "IVF" and record["phase"] == "query" and record["workload"] == "query" and "recall" in record,
            "missing canonical native query record")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True)
        p.add_argument("--dtype", required=True, choices=("float", "int8", "uint8"))
        p.add_argument("--metric", required=True)
        for field in ("dim", "full-count", "base-count"):
            p.add_argument("--" + field, required=True, type=int)
        p.add_argument("--work-dir", required=True, type=Path)
        p.add_argument("--recall-at", required=True, type=int)
        if name == "prepare":
            for field in ("pq-prefix", "query", "truth", "cache-root"):
                p.add_argument("--" + field, required=True, type=Path)
        else:
            p.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.dataset), "unsafe dataset name")
        (prepare if args.command == "prepare" else run)(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
