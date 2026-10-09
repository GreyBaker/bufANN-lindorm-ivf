#!/usr/bin/env python3
"""IVF-only query and sequential insert/delete preparation/cache/run orchestration.

Trusted local files, SHA-256 identity/integrity (not authentication), no deletes,
no atomic publication/recovery. Native bundle validation remains authoritative.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

SUFFIXES = (".centroids.bin", ".hnsw", ".router", ".lists", ".heap", ".heap.meta",
            ".heap.meta.ivf", ".pq.bin", ".manifest")
MUTABLE_SUFFIXES = (".centroids.bin", ".hnsw", ".router", ".mlists", ".heap", ".heap.meta",
                     ".heap.meta.ivf", ".pq.bin", ".heap.state", ".origin", ".manifest")
NATIVE_SUFFIXES = (".patched.hnsw", ".refined.hnsw", ".pca")


def bundle_suffixes(prefix, mutable=False):
    manifest = Path(str(prefix) + ".manifest")
    require(manifest.is_file() and not manifest.is_symlink() and manifest.stat().st_size <= 16384,
            "missing/symlink/oversized IVF manifest")
    version = manifest.read_text().split(maxsplit=1)[0]
    family = "IVFMUTABLE" if mutable else "IVFINDEX"
    require(version in (family + "1", family + "2") or (not mutable and version in ("IVFINDEX3", "IVFINDEX4")),
            "unsupported IVF bundle version")
    native = version.endswith("2")
    if version in ("IVFINDEX3", "IVFINDEX4"):
        fields = [line.split() for line in manifest.read_text().splitlines() if line.startswith("optimized_modes ")]
        require(len(fields) == 1 and fields[0] in (["optimized_modes", "0"], ["optimized_modes", "1"]),
                "invalid optimized-mode flag")
        native = fields[0][1] == "1"
    return (MUTABLE_SUFFIXES if mutable else SUFFIXES) + (NATIVE_SUFFIXES if native else ())


def query_router_flags():
    return [part for key in ("router_mode", "router_rerank_factor", "router_breadth")
            if (value := os.environ.get("IVF_" + key.upper())) is not None
            for part in ("--" + key, value)]


def mutation_identity(args, work):
    if args.workload == "query":
        require(args.update_count == 0, "query cannot have update IDs")
        return None
    count = args.update_count
    require(0 < count <= (args.full_count - args.base_count if args.workload == "insert" else args.base_count - 1),
            "invalid mutation count for canonical base split")
    first = args.base_count if args.workload == "insert" else args.base_count - count
    ids = work / (args.workload + "_ids.bin")
    require(ids.stat().st_size == 4 + 4 * count, "invalid prepared counted-ID framing")
    with ids.open("rb") as source:
        require(source.read(4) == struct.pack("<I", count), "counted-ID count mismatch")
        for offset in range(0, count, 4096):
            length = min(4096, count - offset)
            require(source.read(4 * length) == struct.pack("<" + "I" * length, *range(first + offset, first + offset + length)),
                    "prepared update IDs differ from SPFresh source-row range")
    return {"workload": args.workload, "count": count, "first_id": first, "ids_identity": digest(ids),
            "source_identity": digest(work / "data/full.bin")}


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


def read_cost_model_report(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 << 20,
            "missing/symlink/oversized cost-model report")
    def unique(pairs):
        require(len(dict(pairs)) == len(pairs), "duplicate cost-model report field")
        return dict(pairs)
    record = json.loads(path.read_text(), object_pairs_hook=unique)
    require(record.get("version") in (1, 2) and record.get("kind") == "ivf_cost_model" and record.get("completed") is True,
            "incomplete/unsupported cost-model report")
    chosen = record.get("selected_nlist")
    require(type(chosen) is int and chosen > 0, "invalid model-selected nlist")
    minimum = 32 if record["version"] == 1 else record.get("min_sample_per_centroid")
    require(type(minimum) is int and 1 <= minimum <= 0x7fffffff, "invalid declared sample adequacy minimum")
    if record["version"] == 2:
        require(record.get("candidate_eligibility_policy") == "declared_project_minimum_not_paper_requirement",
                "unsupported candidate eligibility policy")
        mode = record.get("trainer_mode")
        legacy_modes = ("exact", "ann-rebuild", "ann-reuse", "ann-reuse-seeded")
        count_modes = ("ann-reuse-count", "ann-reuse-seeded-count")
        require(mode in legacy_modes + count_modes + ("lv", "lv-plus"), "unsupported report trainer")
        expected = ("pca95_refit_each_pass_initial_refinement" if mode in ("lv", "lv-plus") else
                    "full_count_based" if mode in count_modes else "exact_full_assignment" if mode == "exact" else
                    "legacy_full_breadth_rerank")
        require(record.get("training_search_policy") == expected, "inconsistent training search policy")
    require(type(record["sample_count"]) is int and record["sample_count"] > 0, "invalid calibration sample count")
    for candidate in record["candidates"]:
        n = candidate["nlist"]
        require(type(n) is int and 0 < n <= 0x7fffffff and type(candidate["eligible"]) is bool and
                candidate["eligible"] == (n * minimum <= record["sample_count"]), "incorrect candidate eligibility")
    candidates = [candidate for candidate in record["candidates"] if candidate["eligible"]]
    require(candidates and len({c["nlist"] for c in record["candidates"]}) == len(record["candidates"]),
            "empty/duplicate cost-model candidates")
    for candidate in candidates:
        require(candidate["nlist"] * minimum <= record["sample_count"], "ineligible model candidate")
        a, b = candidate["t1_us"], candidate["t2_us"]
        require(math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0 and
                candidate["total_us"] == a + b, "invalid cost-model candidate estimate")
    require(chosen == min(candidates, key=lambda c: (c["total_us"], c["nlist"]))["nlist"],
            "cost-model selection is not minimum T1+T2")
    return record


def nlist_selection(args, settings, sources):
    path = os.environ.get("IVF_COST_MODEL_REPORT")
    if not path:
        return {"source": "manual", "nlist": int(settings["nlist"])}
    record = read_cost_model_report(path)
    require(record["data_type"] == args.dtype and record["dim"] == args.dim and
            record["source_rows"] == args.full_count and record["base_count"] == args.base_count,
            "cost-model profile does not match build")
    require([record["identities"][key] for key in ("data", "pq_pivots", "pq_codes")] == sources,
            "cost-model source/PQ identity does not match build")
    require(record["objective"] == "mean_T1_plus_T2_excludes_fetch_and_exact_rerank" and
            record["calibration_query_policy"] == "disjoint_base_rows_no_official_queries",
            "unsupported cost-model objective/calibration policy")
    aliases = {"init": "trainer_init", "sample_rate": "target_sample_rate", "nlist": "selected_nlist"}
    keys = ("nlist", "sample_rate", "max_reps", "seed", "build_threads", "trainer_mode", "init", "training_breadth",
            "memory_budget_gb", "router_mode", "optimized_modes", "router_degree", "construction_breadth",
            "query_breadth", "insert_breadth", "router_seed", "refine_ef", "pca_variance",
            "router_rerank_factor", "routing_work_limit")
    for key in keys:
        observed = record[aliases.get(key, key)]
        same = settings[key] == str(observed) if isinstance(observed, str) else float(settings[key]) == observed
        require(same, "cost-model/build setting mismatch: " + key)
    for key, default in (("training_rerank_factor", 3), ("training_refine_breadth", 128)):
        require(float(settings.get(key, default)) == record.get(key, default), "cost-model/build setting mismatch: " + key)
    selection = {"source": "cost_model", "nlist": record["selected_nlist"], "report_path": str(Path(path).resolve()),
            "report_identity": digest(path), "objective": record["objective"],
            "topk": record["topk"], "target_recall": record["target_recall"], "rerank_factor": record["rerank_factor"],
             "validation_completed": record.get("validation_completed", False), "production_qualified": False}
    if record["version"] == 2:
        selection.update(min_sample_per_centroid=record["min_sample_per_centroid"],
                         candidate_eligibility_policy=record["candidate_eligibility_policy"])
    return selection


def prepared_selection(prepared):
    selection = prepared.get("nlist_selection", {"source": "manual", "nlist": int(prepared["settings"]["nlist"])})
    if selection["source"] == "cost_model":
        require(digest(selection["report_path"]) == selection["report_identity"], "prepared cost-model report changed")
    return selection


def profile(args):
    return {key: getattr(args, key) for key in ("dataset", "dtype", "metric", "dim", "full_count", "base_count")}


def build_settings():
    defaults = {"nlist": None, "sample_rate": "1", "max_reps": "15", "seed": "0", "build_threads": "1",
                "memory_budget_gb": "1", "page_size": "4096", "buffer_frames": "64", "router_degree": "16",
                 "construction_breadth": "200", "query_breadth": "128", "insert_breadth": "64", "router_seed": "100"}
    defaults.update(optimized_modes="0", refine_ef="128", pca_variance="0.95", routing_work_limit="1000000000")
    defaults.update(trainer_mode="ann-reuse-seeded", init="random_partition", training_breadth="64")
    result = {key: os.environ.get("IVF_" + key.upper(), value) for key, value in defaults.items()}
    # Preserve legacy cache keys unless the new trainer/settings are requested.
    for key, default in (("training_rerank_factor", "3"), ("training_refine_breadth", "128")):
        value = os.environ.get("IVF_" + key.upper())
        if result["trainer_mode"] in ("lv", "lv-plus") or value is not None:
            result[key] = default if value is None else value
    # Query-only mode/factor overrides do not alter the build/cache identity.
    result["router_mode"] = os.environ.get("IVF_ASSIGNMENT_ROUTER_MODE", "hnswlib")
    result["router_rerank_factor"] = os.environ.get("IVF_BUILD_ROUTER_RERANK_FACTOR", "3")
    if os.environ.get("IVF_COST_MODEL_REPORT"):
        chosen = read_cost_model_report(os.environ["IVF_COST_MODEL_REPORT"])["selected_nlist"]
        require(result["nlist"] is None or int(result["nlist"]) == chosen, "IVF_NLIST disagrees with cost-model selection")
        result["nlist"] = str(chosen)
    require(result["nlist"] is not None, "set IVF_NLIST explicitly or provide IVF_COST_MODEL_REPORT")
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
    for suffix in bundle_suffixes(directory / dataset):
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
    mutation = mutation_identity(args, work)
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
    selection = nlist_selection(args, settings, identity["sources"])
    identity["nlist_selection"] = selection
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
        for suffix in bundle_suffixes(cache / "index" / args.dataset):
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
            for suffix in bundle_suffixes(prefix):
                shutil.copyfile(Path(str(prefix) + suffix), cache / "index" / (args.dataset + suffix))
            require(component_digests(cache / "index", args.dataset) == components, "IVF cache copy integrity mismatch")
            write_json(cache / "cache.json", {"identity": identity, "components": components}) # last; not atomic/fsync
    if selection["source"] == "cost_model":
        require(digest(selection["report_path"]) == selection["report_identity"], "cost-model report changed during prepare")
    write_json(work / "prepared.json", {"version": 1, "profile": profile(args), "settings": settings,
        "nlist_selection": selection,
        "cache_key": key, "cache_hit": hit, "cache_policy": "sha256_inputs_tools_settings_real_bundle_copies",
        "query": str(args.query.resolve()), "query_identity": digest(args.query),
        "truth": str(args.truth.resolve()), "truth_identity": digest(args.truth),
        "manifest_identity": digest(Path(str(prefix) + ".manifest"))})
    if mutation is not None:
        write_json(work / "mutation_prepared.json", mutation)


def run(args):
    work = args.work_dir.resolve()
    prepared = read_json(work / "prepared.json")
    require(prepared["version"] == 1 and prepared["profile"] == profile(args), "profile changed since prepare")
    selection = prepared_selection(prepared)
    mutation = mutation_identity(args, work)
    if mutation is not None:
        require(read_json(work / "mutation_prepared.json") == mutation, "mutation/source identity changed since prepare")
    else:
        require(not (work / "mutation_prepared.json").exists(), "prepared workload is not query")
    require(digest(prepared["query"]) == prepared["query_identity"], "query changed since prepare")
    require(digest(prepared["truth"]) == prepared["truth_identity"], "GT changed since prepare")
    query_truth_shapes(prepared["query"], prepared["truth"], args.dtype, args.dim, args.recall_at)
    prefix = work / "index" / args.dataset
    require((work / "index").is_dir() and not (work / "index").is_symlink(), "working index directory must not symlink into cache")
    require(digest(Path(str(prefix) + ".manifest")) == prepared["manifest_identity"], "manifest changed since prepare")
    for suffix in bundle_suffixes(prefix):
        require(not Path(str(prefix) + suffix).is_symlink(), "working bundle must not symlink into cache")
    output = args.output_dir.resolve()
    require(output == work or work in output.parents, "run output must stay under prepared work tree")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("run.log", "native_result.json", "result.json", "neighbors.bin", "nlist_selection.json"):
        path = output / name
        require(not path.exists() and not path.is_symlink(), f"run output exists: {path}; use a fresh IVF_RUN_TAG")
    # Keep native result.json byte-identical to native_result.json; provenance is
    # an adjacent experiment record, never a fabricated native query metric.
    write_json(output / "nlist_selection.json", selection)
    if mutation is not None:
        for name in ("mutation.log", "mutation_result.json", "mutation_metrics.json", "private.heap", "private.heap.pq.bin", "query.heap", "query.heap.pq.bin", "checkpoint"):
            path = output / name
            require(not path.exists() and not path.is_symlink(), f"mutation output exists: {path}")
        checkpoint = output / "checkpoint"
        checkpoint.mkdir()
        saved = checkpoint / args.dataset
        command = [os.environ["IVF_DRIVER_BIN"], "--validate_inputs", "0", "--data_type", args.dtype,
            "--metric", args.metric, "--dim", str(args.dim), "--nlist", prepared["settings"]["nlist"],
            "--index_prefix", str(prefix), "--workload", args.workload + "_only",
            "--full_data_file", str(work / "data/full.bin"), "--" + args.workload + "_ids_file", str(work / (args.workload + "_ids.bin")),
            "--private_heap_path", str(output / "private.heap"), "--checkpoint_prefix", str(saved),
            "--buffer_frames", os.environ.get("IVF_BUFFER_FRAMES", "64"),
            "--memory_budget_gb", os.environ.get("IVF_MEMORY_BUDGET_GB", "1"),
            "--result_file", str(output / "mutation_result.json")]
        command_log(command, output / "mutation.log")
        record = read_json(output / "mutation_result.json")
        require(record["phase"] == args.workload and record[args.workload + "_count"] == args.update_count and
                record["live_count"] == args.base_count + (args.update_count if args.workload == "insert" else -args.update_count),
                "mutation execution count/live set mismatch")
        require(mutation_identity(args, work) == mutation, "source/update IDs changed during mutation")
        require(all(Path(str(saved) + suffix).is_file() and not Path(str(saved) + suffix).is_symlink()
                    for suffix in bundle_suffixes(saved, mutable=True)), "missing mutable checkpoint family")
        prefix = saved
    command = [os.environ["IVF_DRIVER_BIN"], "--validate_inputs", "0", "--data_type", args.dtype,
        "--metric", args.metric, "--dim", str(args.dim), "--nlist", prepared["settings"]["nlist"],
        "--index_prefix", str(prefix), "--workload", "search_only", "--query_file", prepared["query"],
        "--gt_file", prepared["truth"], "--recall_at", str(args.recall_at),
        "--nprobe", os.environ.get("IVF_NPROBE", "1"), "--rerank_factor", os.environ.get("IVF_RERANK_FACTOR", "2"),
        "--query_threads", os.environ.get("QUERY_THREADS", "1"),
        "--buffer_frames", os.environ.get("IVF_BUFFER_FRAMES", "64"),
        "--memory_budget_gb", os.environ.get("IVF_QUERY_MEMORY_BUDGET_GB", os.environ.get("IVF_MEMORY_BUDGET_GB", "1")),
        "--result_file", str(output / "native_result.json"), "--search_result_file", str(output / "neighbors.bin")]
    if mutation is not None:
        command += ["--index_format", "mutable", "--private_heap_path", str(output / "query.heap")]
    command += query_router_flags()
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
