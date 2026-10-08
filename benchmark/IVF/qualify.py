#!/usr/bin/env python3
"""Explicit, non-destructive IVF training comparisons and repeated query sweeps.

Results describe the supplied inputs, not automatic production qualification.
Completion summaries are written last; failed/partial evidence is never deleted.
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

from harness import digest, flags, header, read_json, require, write_json

REPO = Path(__file__).resolve().parents[2]


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def finite(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def values(text, convert, minimum):
    result = [convert(part) for part in text.split(",")]
    require(result and all(math.isfinite(v) and v >= minimum for v in result), "invalid sweep values")
    require(len(set(result)) == len(result), "duplicate sweep values")
    return result


def affinity(cpus):
    available = sorted(os.sched_getaffinity(0))
    selected = values(cpus, int, 0) if cpus else available
    require(set(selected) <= set(available), "requested CPUs outside available affinity")
    return {"available": available, "selected": selected, "explicitly_pinned": bool(cpus)}


def execute(command, log, timeout, cpus, env=None):
    # taskset applies before exec; all descendants inherit the selected affinity.
    if cpus:
        command = ["taskset", "--cpu-list", ",".join(str(v) for v in values(cpus, int, 0))] + command
    start = time.monotonic()
    with Path(log).open("x") as out:
        out.write("Command: " + repr(command) + "\n")
        out.flush()
        with subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # Kill the entire script/driver group, not only its shell parent.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise ValueError(f"command timed out/interrupted; inspect {log}")
    require(code == 0, f"command failed ({code}); inspect {log}")
    return time.monotonic() - start


def record_from_log(path):
    records = []
    with Path(path).open() as source:
        for line in source:
            if line.startswith('{'):
                record = json.loads(line)
                if record.get("kind") == "ivf_training_diagnostics":
                    records.append(record)
    require(len(records) == 1, "missing/duplicate training diagnostic record")
    return records[0]


def aggregate(records, fields):
    result = {}
    for field in fields:
        nums = [r[field] for r in records]
        require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in nums), f"invalid {field} measurement")
        result[field] = {"min": min(nums), "median": statistics.median(nums), "max": max(nums)}
    return result


def runtime_environment():
    return {name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "LD_LIBRARY_PATH", "ASAN_OPTIONS", "UBSAN_OPTIONS")}


def train(args):
    data, trainer, output = args.data.resolve(), args.trainer.resolve(), args.output.resolve()
    require(not output.exists(), f"training output already exists: {output}")
    require(trainer.is_file() and os.access(trainer, os.X_OK), "missing routing-enabled training diagnostic tool")
    header(data, args.dtype, args.full_count, args.dim)
    require(0 < args.base_count <= args.full_count <= 0x7FFFFFFF, "invalid source/base row count")
    require(0 < args.sample_rate <= 1 and args.nlist <= math.ceil(args.base_count * args.sample_rate), "invalid training sample/nlist")
    count = math.ceil(args.base_count * args.sample_rate)
    searches = count * (args.iterations + 1)
    require(searches <= args.assignment_work_limit, "training exceeds declared assignment search limit")
    require(searches * args.nlist <= args.exact_work_limit, "exact training exceeds declared work limit; choose a smaller comparison")
    require(min(count, args.evaluation_rows) * args.nlist <= args.evaluation_work_limit, "evaluation exceeds declared work limit")
    require(args.max_sse_ratio is None or args.max_sse_ratio >= 1, "SSE ratio gate must be >= 1")
    require(0 <= args.seed <= 0xFFFFFFFFFFFFFFFF and 0 <= args.router_seed <= 0xFFFFFFFF, "invalid training/router seed")
    settings = {"data_file": str(data), "data_type": args.dtype, "dim": args.dim, "base_count": args.base_count,
        "nlist": args.nlist, "sample_rate": args.sample_rate, "max_reps": args.iterations, "seed": args.seed,
        "build_threads": args.threads, "memory_budget_gb": args.memory_budget_gb,
        "exact_work_limit": args.exact_work_limit, "assignment_work_limit": args.assignment_work_limit,
        "evaluation_work_limit": args.evaluation_work_limit, "evaluation_rows": args.evaluation_rows,
        "router_degree": args.router_degree, "construction_breadth": args.construction_breadth,
        "insert_breadth": args.insert_breadth, "router_seed": args.router_seed}
    plan = {"version": 1, "kind": "ivf_training_comparison", "settings": settings,
        "source_identity": digest(data), "trainer_identity": digest(trainer), "orchestrator_identity": digest(__file__),
        "affinity": affinity(args.cpus), "runtime_environment": runtime_environment(), "timeout_s_per_mode": args.timeout,
        "process_wall_scope": "entire_diagnostic_child_including_evaluation_export_excludes_python_hashes",
        "gates": {"max_sse_ratio": args.max_sse_ratio, "max_peak_rss_mb": args.max_peak_rss_mb},
        "scope": "same_base_only_training_sample_not_holdout_or_production_acceptance"}
    output.mkdir(parents=True)  # existing destinations, including partial runs, refused
    write_json(output / "plan.json", plan)
    records = {}
    artifacts = {}
    for mode in ("exact", "hnsw"):
        prefix = output / mode
        command = [str(trainer)] + flags(settings) + ["--mode", mode, "--output_prefix", str(prefix)]
        wall = execute(command, output / f"{mode}.log", args.timeout, args.cpus)
        record = record_from_log(output / f"{mode}.log")
        require(record["mode"] == mode and record["sample_count"] == count and record["base_count"] == args.base_count,
                "training diagnostic configuration mismatch")
        aggregate([record], ("exact_evaluation_sse", "training_time_s", "peak_process_rss_mb"))
        record["process_wall_s"] = wall
        records[mode] = record
        artifacts[mode] = {suffix: digest(Path(str(prefix) + suffix)) for suffix in (".sample_ids.bin", ".centroids.bin")}
        require(digest(data) == plan["source_identity"] and digest(trainer) == plan["trainer_identity"],
                "source/trainer changed during comparison")
    require(artifacts["exact"][".sample_ids.bin"] == artifacts["hnsw"][".sample_ids.bin"], "training sample IDs differ")
    require(records["exact"]["evaluation_rows"] == records["hnsw"]["evaluation_rows"], "evaluation samples differ")
    exact, approximate = (records[mode]["exact_evaluation_sse"] for mode in ("exact", "hnsw"))
    require(exact >= 0 and approximate >= 0, "negative SSE")
    ratio = approximate / exact if exact else (1.0 if approximate == 0 else None)
    passed = (args.max_sse_ratio is None or (ratio is not None and ratio <= args.max_sse_ratio)) and (
        args.max_peak_rss_mb is None or max(r["peak_process_rss_mb"] for r in records.values()) <= args.max_peak_rss_mb)
    write_json(output / "summary.json", {"kind": plan["kind"], "completed": True, "gates_passed": passed,
        "plan_identity": digest(output / "plan.json"), "records": records, "artifacts": artifacts,
        "hnsw_over_exact_evaluation_sse_ratio": ratio, "evaluation_sse_delta": approximate - exact,
        "zero_reference_policy": "ratio_1_for_both_zero_null_for_nonzero_hnsw", "production_qualified": False})
    require(passed, f"declared training gates failed; inspect {output / 'summary.json'}")
    print(f"Training comparison complete: {output / 'summary.json'}")


def sweep(args):
    work = args.work_dir.resolve()
    require(work.name == "query" and work.parent.name == "IVF", "expected prepared <root>/<dataset>/IVF/query work tree")
    prepared_path = work / "prepared.json"
    prepared = read_json(prepared_path)
    dataset = prepared["profile"]["dataset"]
    require(work.parent.parent.name == dataset and re.fullmatch(r"[a-zA-Z0-9_-]+", dataset), "work/profile dataset mismatch")
    require(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.tag), "unsafe sweep tag")
    probes = values(args.nprobes, int, 1)
    factors = values(args.rerank_factors, float, 1)
    threads = values(args.threads, int, 1)
    require(max(probes) <= int(prepared["settings"]["nlist"]), "sweep nprobe exceeds prepared nlist")
    require(args.warmups >= 0 and 0 <= args.min_recall <= 100, "invalid warmup/recall gate")
    require(len(probes) * len(factors) * len(threads) * (args.repeats + args.warmups) <= 4096,
            "sweep exceeds 4096-trial safety limit")
    configs = [{"nprobe": p, "rerank_factor": f, "query_threads": t} for p, f, t in itertools.product(probes, factors, threads)]
    script = REPO / "benchmark/IVF/sift/query.sh"
    driver = args.driver.resolve()
    require(driver.is_file() and os.access(driver, os.X_OK), "missing native IVF driver")
    profile_path = Path(os.environ.get("BENCH_DATASETS_DIR", REPO / "benchmark/datasets")) / (dataset + ".sh")
    inputs = {"prepared": prepared_path, "profile": profile_path, "driver": driver, "query_script": script,
        "harness": REPO / "benchmark/IVF/harness.py", "common": REPO / "benchmark/IVF/common.sh",
        "utils": REPO / "benchmark/scripts/utils.sh", "profile_loader": REPO / "benchmark/datasets/_load.sh",
        "orchestrator": Path(__file__), "query": Path(prepared["query"]), "truth": Path(prepared["truth"]),
        "manifest": work / "index" / (dataset + ".manifest")}
    identities = {name: digest(path) for name, path in inputs.items()}
    require(identities["query"] == prepared["query_identity"] and identities["truth"] == prepared["truth_identity"] and
            identities["manifest"] == prepared["manifest_identity"], "prepared input identity changed")
    env = dict(os.environ, DATASET=dataset, EVAL_TEMPFILES=str(work.parents[2]), IVF_DRIVER_BIN=str(driver),
               IVF_QUERY_MEMORY_BUDGET_GB=str(args.memory_budget_gb), IVF_BUFFER_FRAMES=str(args.buffer_frames),
               RECALL_AT=str(args.recall_at))
    trials = []
    for c, config in enumerate(configs):
        for trial in range(args.warmups + args.repeats):
            warmup = trial < args.warmups
            tag = f"{args.tag}_c{c}_{'w' if warmup else 'r'}{trial if warmup else trial - args.warmups}"
            path = work / "runs" / tag
            require(not path.exists() and not path.is_symlink(), f"sweep run destination exists: {path}")
            trials.append({"config": c, "warmup": warmup, "run_tag": tag})
    for directory in (work / "sweeps", work / "runs"):
        require(not directory.is_symlink(), "sweep/run parent must not be a symlink")
    root = work / "sweeps" / args.tag
    require(not root.exists() and not root.is_symlink(), f"sweep destination exists: {root}")
    plan = {"version": 1, "kind": "ivf_query_sweep", "inputs": {k: str(p) for k, p in inputs.items()},
        "identities": identities, "configs": configs, "trials": trials, "repeats": args.repeats, "warmups": args.warmups,
        "warmup_policy": "separate_discarded_processes_os_cache_only_no_inprocess_warmup", "affinity": affinity(args.cpus),
        "runtime_environment": runtime_environment(), "timeout_s_per_trial": args.timeout, "recall_at": args.recall_at,
        "process_wall_scope": "entire_query_script_including_hashes_open_gt_scoring_export",
        "memory_budget_gb": args.memory_budget_gb, "buffer_frames": args.buffer_frames,
        "gates": {"min_recall_percent": args.min_recall, "max_peak_rss_mb": args.max_peak_rss_mb},
        "cache_policy": "uncontrolled_after_full_bundle_hash_no_cache_drop", "order_policy": "config_order_then_warmups_then_repeats"}
    root.mkdir(parents=True)
    write_json(root / "plan.json", plan)
    records = []
    for trial in trials:
        config = configs[trial["config"]]
        env.update(IVF_RUN_TAG=trial["run_tag"], IVF_NPROBE=str(config["nprobe"]),
                   IVF_RERANK_FACTOR=str(config["rerank_factor"]), QUERY_THREADS=str(config["query_threads"]))
        wall = execute(["bash", str(script)], root / (trial["run_tag"] + ".log"), args.timeout, args.cpus, env)
        path = work / "runs" / trial["run_tag"]
        record = read_json(path / "result.json")
        require(digest(path / "result.json") == digest(path / "native_result.json"), "canonical metric extraction mismatch")
        require(record["baseline"] == "IVF" and record["phase"] == "query" and record["workload"] == "query", "unexpected metrics")
        require(all(record[k] == v for k, v in config.items()) and record["recall_at"] == args.recall_at,
                "sweep native configuration mismatch")
        require(record["buffer_pool_frames"] == args.buffer_frames, "sweep buffer frame mismatch")
        require(0 <= record["recall"] <= 100, "invalid recall")
        aggregate([record], ("query_qps", "query_lat_p99_us", "peak_process_rss_mb", "recall"))
        require({name: digest(path) for name, path in inputs.items()} == identities, "sweep input/tool identity changed")
        records.append(dict(trial, metrics=record, process_wall_s=wall, output_dir=str(path),
                            result_identity=digest(path / "result.json"), neighbors_identity=digest(path / "neighbors.bin")))
        # Per-trial completion record preserves successful observations even if
        # a later trial fails. Only the final summary signifies a complete sweep.
        write_json(root / (trial["run_tag"] + ".json"), records[-1])
    groups = []
    passed = True
    for c, config in enumerate(configs):
        measured = [r["metrics"] for r in records if r["config"] == c and not r["warmup"]]
        stats = aggregate(measured, ("query_qps", "recall", "query_time_s", "open_time_s", "query_lat_avg_us",
            "query_lat_p50_us", "query_lat_p95_us", "query_lat_p99_us", "peak_process_rss_mb", "heap_pages_read",
            "estimated_workspace_bytes", "active_query_threads"))
        gate = stats["recall"]["min"] >= args.min_recall and (args.max_peak_rss_mb is None or
               stats["peak_process_rss_mb"]["max"] <= args.max_peak_rss_mb)
        passed = passed and gate
        groups.append({"config": config, "measured_repeats": len(measured), "stats": stats, "gates_passed": gate})
    write_json(root / "summary.json", {"kind": plan["kind"], "completed": True, "gates_passed": passed,
        "plan_identity": digest(root / "plan.json"), "groups": groups, "trial_records": records,
        "aggregation_policy": "min_median_max_across_runs_not_pooled_query_percentiles", "production_qualified": False})
    require(passed, f"declared sweep gates failed; inspect {root / 'summary.json'}")
    print(f"Query sweep complete: {root / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--memory-budget-gb", type=finite, required=True)
        p.add_argument("--timeout", type=finite, required=True, help="whole child process wall seconds, per mode/trial")
        p.add_argument("--cpus", help="optional comma-separated CPU IDs, validated against current affinity")
        p.add_argument("--max-peak-rss-mb", type=finite, help="post-run gate, not a hard RSS allocation limit")
        if name == "train":
            for field in ("data", "trainer", "output"):
                p.add_argument("--" + field, type=Path, required=True)
            p.add_argument("--dtype", choices=("float", "int8", "uint8"), required=True)
            for field in ("dim", "full-count", "base-count", "nlist", "iterations", "threads", "evaluation-rows",
                          "exact-work-limit", "assignment-work-limit", "evaluation-work-limit"):
                p.add_argument("--" + field, type=positive, required=True)
            p.add_argument("--sample-rate", type=finite, required=True)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--router-seed", type=int, default=100)
            p.add_argument("--router-degree", type=positive, default=16)
            p.add_argument("--construction-breadth", type=positive, default=200)
            p.add_argument("--insert-breadth", type=positive, default=64)
            p.add_argument("--max-sse-ratio", type=finite, help="optional HNSW/exact exact-evaluation SSE gate")
        else:
            p.add_argument("--work-dir", type=Path, required=True)
            p.add_argument("--driver", type=Path, required=True)
            for field in ("tag", "nprobes", "rerank-factors", "threads"):
                p.add_argument("--" + field, required=True)
            p.add_argument("--repeats", type=positive, required=True)
            p.add_argument("--warmups", type=int, default=0)
            p.add_argument("--buffer-frames", type=positive, default=64)
            p.add_argument("--recall-at", type=positive, required=True)
            p.add_argument("--min-recall", type=float, default=0, help="minimum percent recall across measured trials")
    args = parser.parse_args()
    try:
        require(sys.platform == "linux", "qualification orchestration targets Linux/WSL")
        (train if args.command == "train" else sweep)(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
