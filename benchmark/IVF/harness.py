#!/usr/bin/env python3
"""IVF-only query and sequential insert/delete preparation/cache/run orchestration.

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
import sys
import time

# Re-export shared helpers for existing local callers; workload ownership stays here.
from common import (
    command_log,
    digest,
    flags,
    header,
    query_truth_shapes,
    read_json,
    require,
    write_json,
)
from settings import (
    build_settings,
    nlist_selection,
    prepared_selection,
    profile,
    query_router_flags,
    read_cost_model_report,
    runtime_resources,
    insertion_threads,
    insertion_input_policy,
    validate_insertion_staging,
    warmup_queries,
)

SUFFIXES = (
    ".centroids.bin",
    ".hnsw",
    ".router",
    ".lists",
    ".heap",
    ".heap.meta",
    ".heap.meta.ivf",
    ".pq.bin",
    ".manifest",
)
MUTABLE_SUFFIXES = (
    ".centroids.bin",
    ".hnsw",
    ".router",
    ".mlists",
    ".heap",
    ".heap.meta",
    ".heap.meta.ivf",
    ".pq.bin",
    ".heap.state",
    ".origin",
    ".manifest",
)
NATIVE_SUFFIXES = (".patched.hnsw", ".refined.hnsw", ".pca")


def bundle_suffixes(prefix, mutable=False):
    manifest = Path(str(prefix) + ".manifest")
    require(
        manifest.is_file() and not manifest.is_symlink() and manifest.stat().st_size <= 16384,
        "missing/symlink/oversized IVF manifest",
    )
    version = manifest.read_text().split(maxsplit=1)[0]
    family = "IVFMUTABLE" if mutable else "IVFINDEX"
    require(
        version in (family + "1", family + "2")
        or (not mutable and version in ("IVFINDEX3", "IVFINDEX4")),
        "unsupported IVF bundle version",
    )
    native = version.endswith("2")
    if version in ("IVFINDEX3", "IVFINDEX4"):
        fields = [
            line.split()
            for line in manifest.read_text().splitlines()
            if line.startswith("optimized_modes ")
        ]
        require(
            len(fields) == 1 and fields[0] in (["optimized_modes", "0"], ["optimized_modes", "1"]),
            "invalid optimized-mode flag",
        )
        native = fields[0][1] == "1"
    return (MUTABLE_SUFFIXES if mutable else SUFFIXES) + (NATIVE_SUFFIXES if native else ())


def mutation_identity(args, work):
    if args.workload == "query":
        require(args.update_count == 0, "query cannot have update IDs")
        return None
    count = args.update_count
    require(
        0
        < count
        <= (
            args.full_count - args.base_count if args.workload == "insert" else args.base_count - 1
        ),
        "invalid mutation count for canonical base split",
    )
    first = args.base_count if args.workload == "insert" else args.base_count - count
    ids = work / (args.workload + "_ids.bin")
    require(ids.stat().st_size == 4 + 4 * count, "invalid prepared counted-ID framing")
    with ids.open("rb") as source:
        require(source.read(4) == struct.pack("<I", count), "counted-ID count mismatch")
        for offset in range(0, count, 4096):
            length = min(4096, count - offset)
            require(
                source.read(4 * length)
                == struct.pack("<" + "I" * length, *range(first + offset, first + offset + length)),
                "prepared update IDs differ from IVF source-row range",
            )
    return {
        "workload": args.workload,
        "count": count,
        "first_id": first,
        "ids_identity": digest(ids),
        "source_identity": digest(work / "data/full.bin"),
    }


def component_digests(directory, dataset):
    require(
        directory.is_dir() and not directory.is_symlink(),
        f"bundle directory must be real: {directory}",
    )
    result = {}
    for suffix in bundle_suffixes(directory / dataset):
        name = dataset + suffix
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"bundle must contain real files: {path}")
        result[name] = digest(path)
    return result


def prepare(args):
    settings = build_settings()  # Reject incompatible frame budgets before hashing/copying.
    work = args.work_dir.resolve()
    data = work / "data/full.bin"
    require(
        args.metric == "l2" and 0 < args.base_count < args.full_count,
        "invalid profile metric/base split",
    )
    header(data, args.dtype, args.full_count, args.dim)
    mutation = mutation_identity(args, work)
    query_truth_shapes(args.query, args.truth, args.dtype, args.dim, args.recall_at)
    # Query row count is arbitrary, but dimension, framing and GT shape are
    # checked by the native driver before timed queries.
    pivots, codes = (
        Path(str(args.pq_prefix) + suffix) for suffix in ("_pq_pivots.bin", "_pq_compressed.bin")
    )
    builder = Path(os.environ["IVF_BUILD_BIN"]).resolve()
    libraries = [
        Path(os.environ["IVF_BUILD_DIR"]) / "src" / name for name in ("libBufANN.a", "libBufANN.so")
    ]
    libraries = [path for path in libraries if path.is_file()]
    require(
        libraries, "IVF_BUILD_DIR must be a full Linux BufANN/routing build, not isolated contracts"
    )
    identity = {
        "version": 1,
        "profile": profile(args),
        "settings": settings,
        "sources": [digest(data), digest(pivots), digest(codes)],
        "builder": digest(builder),
        "bufann_libraries": {path.name: digest(path) for path in libraries},
        "cache_tag": os.environ.get("IVF_CACHE_TAG", ""),
    }
    selection = nlist_selection(args, settings, identity["sources"])
    identity["nlist_selection"] = selection
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = args.cache_root.resolve() / args.dataset / "IVF" / key
    index = work / "index"
    index.mkdir()  # Do not overwrite partial or user work.
    prefix = index / args.dataset
    hit = cache.exists()
    if hit:
        require(
            cache.is_dir() and not cache.is_symlink(), "IVF cache entry must be a real directory"
        )
        metadata = read_json(cache / "cache.json")
        require(metadata["identity"] == identity, "IVF cache identity mismatch")
        require(
            component_digests(cache / "index", args.dataset) == metadata["components"],
            "IVF cache integrity mismatch",
        )
        for suffix in bundle_suffixes(cache / "index" / args.dataset):
            shutil.copyfile(cache / "index" / (args.dataset + suffix), Path(str(prefix) + suffix))
        require(
            component_digests(index, args.dataset) == metadata["components"],
            "IVF restore integrity mismatch",
        )
        print(f"IVF cache restored: {key} (real copies, not fresh build metrics)")
    else:
        command = [
            str(builder),
            "--validate_inputs",
            "0",
            "--data_type",
            args.dtype,
            "--metric",
            args.metric,
            "--dim",
            str(args.dim),
            "--base_count",
            str(args.base_count),
            "--data_file",
            str(data),
            "--pq_prefix",
            str(args.pq_prefix),
            "--index_prefix",
            str(prefix),
        ] + flags(settings)
        command_log(command, work / "build.log")
        components = component_digests(index, args.dataset)
        # Recheck identities after construction; native build additionally does
        # its own before/after source checks. No recovery of partial cache dirs.
        require(
            identity["sources"] == [digest(data), digest(pivots), digest(codes)],
            "source changed during prepare",
        )
        if not os.environ.get("EVAL_CACHE_NO_STORE"):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.mkdir()
            (cache / "index").mkdir()
            for suffix in bundle_suffixes(prefix):
                shutil.copyfile(
                    Path(str(prefix) + suffix), cache / "index" / (args.dataset + suffix)
                )
            require(
                component_digests(cache / "index", args.dataset) == components,
                "IVF cache copy integrity mismatch",
            )
            write_json(
                cache / "cache.json", {"identity": identity, "components": components}
            )  # last; not atomic/fsync
    if selection["source"] == "cost_model":
        require(
            digest(selection["report_path"]) == selection["report_identity"],
            "cost-model report changed during prepare",
        )
    write_json(
        work / "prepared.json",
        {
            "version": 1,
            "profile": profile(args),
            "settings": settings,
            "nlist_selection": selection,
            "cache_key": key,
            "cache_hit": hit,
            "cache_policy": "sha256_inputs_tools_settings_real_bundle_copies",
            "query": str(args.query.resolve()),
            "query_identity": digest(args.query),
            "truth": str(args.truth.resolve()),
            "truth_identity": digest(args.truth),
            "manifest_identity": digest(Path(str(prefix) + ".manifest")),
        },
    )
    if mutation is not None:
        write_json(work / "mutation_prepared.json", mutation)


def run(args):
    workflow_start = time.monotonic()
    mutation_process_wall_s = None
    work = args.work_dir.resolve()
    prepared = read_json(work / "prepared.json")
    query_resources = runtime_resources(prepared["settings"]["page_size"], query=True)
    mutation_resources = (runtime_resources(prepared["settings"]["page_size"])
                           if args.workload != "query" else None)
    insert_workers = insertion_threads() if args.workload == "insert" else None
    query_warmups = warmup_queries()
    insert_policy = insertion_input_policy() if args.workload == "insert" else None
    if insert_policy == "staged":
        validate_insertion_staging(mutation_resources, args.update_count, args.dim, args.dtype)
    require(
        prepared["version"] == 1 and prepared["profile"] == profile(args),
        "profile changed since prepare",
    )
    selection = prepared_selection(prepared)
    mutation = mutation_identity(args, work)
    if mutation is not None:
        require(
            read_json(work / "mutation_prepared.json") == mutation,
            "mutation/source identity changed since prepare",
        )
    else:
        require(not (work / "mutation_prepared.json").exists(), "prepared workload is not query")
    require(digest(prepared["query"]) == prepared["query_identity"], "query changed since prepare")
    require(digest(prepared["truth"]) == prepared["truth_identity"], "GT changed since prepare")
    query_truth_shapes(prepared["query"], prepared["truth"], args.dtype, args.dim, args.recall_at)
    prefix = work / "index" / args.dataset
    require(
        (work / "index").is_dir() and not (work / "index").is_symlink(),
        "working index directory must not symlink into cache",
    )
    require(
        digest(Path(str(prefix) + ".manifest")) == prepared["manifest_identity"],
        "manifest changed since prepare",
    )
    for suffix in bundle_suffixes(prefix):
        require(
            not Path(str(prefix) + suffix).is_symlink(),
            "working bundle must not symlink into cache",
        )
    output = args.output_dir.resolve()
    require(
        output == work or work in output.parents, "run output must stay under prepared work tree"
    )
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "run.log",
        "native_result.json",
        "result.json",
        "neighbors.bin",
        "nlist_selection.json",
        "execution_protocol.json",
    ):
        path = output / name
        require(
            not path.exists() and not path.is_symlink(),
            f"run output exists: {path}; use a fresh IVF_RUN_TAG",
        )
    # Keep native result.json byte-identical to native_result.json; provenance is
    # an adjacent experiment record, never a fabricated native query metric.
    write_json(output / "nlist_selection.json", selection)
    if mutation is not None:
        for name in (
            "mutation.log",
            "mutation_result.json",
            "mutation_metrics.json",
            "private.heap",
            "private.heap.pq.bin",
            "query.heap",
            "query.heap.pq.bin",
            "checkpoint",
        ):
            path = output / name
            require(not path.exists() and not path.is_symlink(), f"mutation output exists: {path}")
        checkpoint = output / "checkpoint"
        checkpoint.mkdir()
        saved = checkpoint / args.dataset
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
            prepared["settings"]["nlist"],
            "--index_prefix",
            str(prefix),
            "--workload",
            args.workload + "_only",
            "--full_data_file",
            str(work / "data/full.bin"),
            "--" + args.workload + "_ids_file",
            str(work / (args.workload + "_ids.bin")),
            "--private_heap_path",
            str(output / "private.heap"),
            "--checkpoint_prefix",
            str(saved),
            "--buffer_frames",
            mutation_resources["buffer_frames"],
            "--memory_budget_gb",
            mutation_resources["memory_budget_gb"],
            "--result_file",
            str(output / "mutation_result.json"),
        ]
        if insert_workers is not None:
            command += ["--insert_threads", insert_workers, "--insert_input_policy", insert_policy]
        mutation_start = time.monotonic()
        command_log(command, output / "mutation.log")
        mutation_process_wall_s = time.monotonic() - mutation_start
        record = read_json(output / "mutation_result.json")
        require(
            record["phase"] == args.workload
            and record[args.workload + "_count"] == args.update_count
            and record["live_count"]
            == args.base_count
            + (args.update_count if args.workload == "insert" else -args.update_count),
            "mutation execution count/live set mismatch",
        )
        require(
            mutation_identity(args, work) == mutation, "source/update IDs changed during mutation"
        )
        require(
            all(
                Path(str(saved) + suffix).is_file() and not Path(str(saved) + suffix).is_symlink()
                for suffix in bundle_suffixes(saved, mutable=True)
            ),
            "missing mutable checkpoint family",
        )
        prefix = saved
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
        prepared["settings"]["nlist"],
        "--index_prefix",
        str(prefix),
        "--workload",
        "search_only",
        "--query_file",
        prepared["query"],
        "--gt_file",
        prepared["truth"],
        "--recall_at",
        str(args.recall_at),
        "--nprobe",
        os.environ.get("IVF_NPROBE", "1"),
        "--rerank_factor",
        os.environ.get("IVF_RERANK_FACTOR", "2"),
        "--query_threads",
        os.environ.get("QUERY_THREADS", "1"),
        "--buffer_frames",
        query_resources["buffer_frames"],
        "--memory_budget_gb",
        query_resources["memory_budget_gb"],
        "--result_file",
        str(output / "native_result.json"),
        "--search_result_file",
        str(output / "neighbors.bin"),
    ]
    if mutation is not None:
        command += ["--index_format", "mutable", "--private_heap_path", str(output / "query.heap")]
    command += query_router_flags()
    command += ["--warmup_queries", query_warmups]
    query_start = time.monotonic()
    command_log(command, output / "run.log")
    query_process_wall_s = time.monotonic() - query_start
    record = read_json(output / "native_result.json")
    require(
        record["baseline"] == "IVF"
        and record["phase"] == "query"
        and record["workload"] == "query"
        and "recall" in record,
        "missing canonical native query record",
    )
    require(record["explicit_warmup_query_count"] == int(query_warmups),
            "native warmup completion count differs from requested policy")
    write_json(output / "execution_protocol.json", {
        "version": 1,
        "kind": "ivf_sequential_harness_execution_protocol",
        "completed": True,
        "insert_input_policy": insert_policy,
        "insert_threads_requested": int(insert_workers) if insert_workers is not None else None,
        "native_driver_process_count": 2 if mutation is not None else 1,
        "process_boundary_policy": (
            "mutation_driver_verified_checkpoint_then_separate_query_driver_private_copy"
            if mutation is not None else "standalone_query_driver"),
        "application_cache_transfer_between_driver_processes": False,
        "warmup_query_count_requested": int(query_warmups),
        "explicit_warmup_query_count": record["explicit_warmup_query_count"],
        "warmup_policy": record["warmup_policy"],
        "warmup_process_policy": "inside_query_driver_same_index_object_frozen_base_centroids_not_transferred_from_mutation_driver",
        "os_cache_policy": "not_controlled_validation_hash_and_copy_may_touch_pages",
        "mutation_process_wall_s": mutation_process_wall_s,
        "query_process_wall_s": query_process_wall_s,
        "process_wall_scope": "parent_spawn_wait_includes_native_record_write_and_log_streaming_not_cpu_or_device_time",
        "harness_workflow_wall_s": time.monotonic() - workflow_start,
        "harness_workflow_scope": "run_entry_through_native_record_validation_pre_protocol_write_excludes_prepare_script_and_outer_shell_exports",
        "canonical_native_query_metrics_rewritten": False,
    })


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
        p.add_argument("--workload", choices=("query", "insert", "delete"), default="query")
        p.add_argument("--update-count", type=int, default=0)
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
