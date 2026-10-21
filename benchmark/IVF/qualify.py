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
import statistics
import sys

from common import (
    affinity,
    digest,
    execute,
    finite,
    flags,
    header,
    positive,
    read_json,
    require,
    runtime_environment,
    shared_module_paths,
    values,
    write_json,
)

REPO = Path(__file__).resolve().parents[2]


def record_from_log(path):
    records = []
    with Path(path).open() as source:
        for line in source:
            if line.startswith("{"):
                record = json.loads(line)
                if record.get("kind") == "ivf_training_diagnostics":
                    records.append(record)
    require(len(records) == 1, "missing/duplicate training diagnostic record")
    return records[0]


def aggregate(records, fields):
    result = {}
    for field in fields:
        nums = [r[field] for r in records]
        require(
            all(isinstance(v, (int, float)) and math.isfinite(v) for v in nums),
            f"invalid {field} measurement",
        )
        result[field] = {"min": min(nums), "median": statistics.median(nums), "max": max(nums)}
    return result


def train(args):
    data, trainer, output = args.data.resolve(), args.trainer.resolve(), args.output.resolve()
    require(not output.exists(), f"training output already exists: {output}")
    require(
        trainer.is_file() and os.access(trainer, os.X_OK),
        "missing routing-enabled training diagnostic tool",
    )
    header(data, args.dtype, args.full_count, args.dim)
    require(0 < args.base_count <= args.full_count <= 0x7FFFFFFF, "invalid source/base row count")
    require(
        0 < args.sample_rate <= 1 and args.nlist <= math.ceil(args.base_count * args.sample_rate),
        "invalid training sample/nlist",
    )
    count = math.ceil(args.base_count * args.sample_rate)
    modes = args.modes.split(",")
    require(
        modes
        and len(modes) == len(set(modes))
        and set(modes)
        <= {
            "exact",
            "hnsw",
            "ann-rebuild",
            "ann-reuse",
            "ann-reuse-seeded",
            "ann-reuse-count",
            "ann-reuse-seeded-count",
            "lv",
            "lv-plus",
        },
        "invalid trainer modes",
    )
    require(
        args.max_sse_ratio is None or "exact" in modes,
        "SSE ratio gate requires an exact-trained reference",
    )
    require(
        not set(modes) & {"lv", "lv-plus"} or args.init == "random_partition",
        "LV/LV+ requires random_partition",
    )
    require(args.training_rerank_factor >= 1, "training rerank factor must be >=1")
    searches = count * (args.iterations + 1)
    require(
        searches <= args.assignment_work_limit, "training exceeds declared assignment search limit"
    )
    if "exact" in modes:
        require(
            searches * args.nlist <= args.exact_work_limit,
            "exact training exceeds declared work limit; choose a smaller comparison",
        )
    require(
        min(count, args.evaluation_rows) * args.nlist <= args.evaluation_work_limit,
        "evaluation exceeds declared work limit",
    )
    require(args.max_sse_ratio is None or args.max_sse_ratio >= 1, "SSE ratio gate must be >= 1")
    require(
        0 <= args.seed <= 0xFFFFFFFFFFFFFFFF and 0 <= args.router_seed <= 0xFFFFFFFF,
        "invalid training/router seed",
    )
    settings = {
        "data_file": str(data),
        "data_type": args.dtype,
        "dim": args.dim,
        "base_count": args.base_count,
        "nlist": args.nlist,
        "sample_rate": args.sample_rate,
        "max_reps": args.iterations,
        "seed": args.seed,
        "build_threads": args.threads,
        "memory_budget_gb": args.memory_budget_gb,
        "exact_work_limit": args.exact_work_limit,
        "assignment_work_limit": args.assignment_work_limit,
        "evaluation_work_limit": args.evaluation_work_limit,
        "evaluation_rows": args.evaluation_rows,
        "router_degree": args.router_degree,
        "construction_breadth": args.construction_breadth,
        "insert_breadth": args.insert_breadth,
        "router_seed": args.router_seed,
        "init": args.init,
        "training_breadth": (
            args.training_breadth if args.training_breadth is not None else args.insert_breadth
        ),
        "routing_work_limit": args.routing_work_limit,
    }
    plan = {
        "version": 1,
        "kind": "ivf_training_comparison",
        "settings": settings,
        "optimized_training_settings": {
            "training_rerank_factor": args.training_rerank_factor,
            "training_refine_breadth": args.training_refine_breadth,
            "search_policy": "pca95_refit_each_pass_initial_refinement",
        },
        "source_identity": digest(data),
        "trainer_identity": digest(trainer),
        "orchestrator_identity": digest(__file__),
        "shared_module_identities": {
            key: digest(path) for key, path in shared_module_paths().items()
        },
        "affinity": affinity(args.cpus),
        "runtime_environment": runtime_environment(),
        "timeout_s_per_mode": args.timeout,
        "process_wall_scope": "entire_diagnostic_child_including_evaluation_export_excludes_python_hashes",
        "gates": {"max_sse_ratio": args.max_sse_ratio, "max_peak_rss_mb": args.max_peak_rss_mb},
        "scope": "same_base_only_training_sample_not_holdout_or_production_acceptance",
        "modes": modes,
    }
    output.mkdir(parents=True)  # existing destinations, including partial runs, refused
    write_json(output / "plan.json", plan)
    records = {}
    artifacts = {}
    for mode in modes:
        prefix = output / mode
        command = (
            [str(trainer)] + flags(settings) + ["--mode", mode, "--output_prefix", str(prefix)]
        )
        if mode in ("lv", "lv-plus"):
            command += flags(
                {
                    "training_rerank_factor": args.training_rerank_factor,
                    "training_refine_breadth": args.training_refine_breadth,
                }
            )
        wall = execute(command, output / f"{mode}.log", args.timeout, args.cpus)
        record = record_from_log(output / f"{mode}.log")
        require(
            record["mode"] == mode
            and record["sample_count"] == count
            and record["base_count"] == args.base_count,
            "training diagnostic configuration mismatch",
        )
        aggregate([record], ("exact_evaluation_sse", "training_time_s", "peak_process_rss_mb"))
        record["process_wall_s"] = wall
        records[mode] = record
        artifacts[mode] = {
            suffix: digest(Path(str(prefix) + suffix))
            for suffix in (
                ".sample_ids.bin",
                ".centroids.bin",
                ".assignments.bin",
                ".occupancy.bin",
            )
        }
        require(
            digest(data) == plan["source_identity"] and digest(trainer) == plan["trainer_identity"],
            "source/trainer changed during comparison",
        )
        require(
            {key: digest(path) for key, path in shared_module_paths().items()}
            == plan["shared_module_identities"],
            "shared module changed during comparison",
        )
    require(
        len({artifacts[mode][".sample_ids.bin"]["sha256"] for mode in modes}) == 1,
        "training sample IDs differ",
    )
    require(
        len({records[mode]["evaluation_rows"] for mode in modes}) == 1, "evaluation samples differ"
    )
    exact = records["exact"]["exact_evaluation_sse"] if "exact" in records else None
    ratios = {}
    for mode in modes:
        value = records[mode]["exact_evaluation_sse"]
        require(value >= 0, "negative SSE")
        ratios[mode] = (
            None if exact is None else (value / exact if exact else (1.0 if value == 0 else None))
        )
    ratio = ratios.get("hnsw")
    passed = (
        args.max_sse_ratio is None
        or all(value is not None and value <= args.max_sse_ratio for value in ratios.values())
    ) and (
        args.max_peak_rss_mb is None
        or max(r["peak_process_rss_mb"] for r in records.values()) <= args.max_peak_rss_mb
    )
    write_json(
        output / "summary.json",
        {
            "kind": plan["kind"],
            "completed": True,
            "gates_passed": passed,
            "plan_identity": digest(output / "plan.json"),
            "records": records,
            "artifacts": artifacts,
            "hnsw_over_exact_evaluation_sse_ratio": ratio,
            "evaluation_sse_delta": (
                records["hnsw"]["exact_evaluation_sse"] - exact
                if exact is not None and "hnsw" in records
                else None
            ),
            "exact_trained_reference_available": exact is not None,
            "evaluation_sse_over_exact_by_mode": ratios,
            "zero_reference_policy": "ratio_1_for_both_zero_null_for_nonzero_approximate_or_absent_reference",
            "production_qualified": False,
        },
    )
    require(passed, f"declared training gates failed; inspect {output / 'summary.json'}")
    print(f"Training comparison complete: {output / 'summary.json'}")


def sweep(args):
    work = args.work_dir.resolve()
    require(
        work.name == "query" and work.parent.name == "IVF",
        "expected prepared <root>/<dataset>/IVF/query work tree",
    )
    prepared_path = work / "prepared.json"
    prepared = read_json(prepared_path)
    dataset = prepared["profile"]["dataset"]
    require(
        work.parent.parent.name == dataset and re.fullmatch(r"[a-zA-Z0-9_-]+", dataset),
        "work/profile dataset mismatch",
    )
    require(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.tag), "unsafe sweep tag")
    probes = values(args.nprobes, int, 1)
    factors = values(args.rerank_factors, float, 1)
    threads = values(args.threads, int, 1)
    require(
        max(probes) <= int(prepared["settings"]["nlist"]), "sweep nprobe exceeds prepared nlist"
    )
    require(args.warmups >= 0 and 0 <= args.min_recall <= 100, "invalid warmup/recall gate")
    require(
        len(probes) * len(factors) * len(threads) * (args.repeats + args.warmups) <= 4096,
        "sweep exceeds 4096-trial safety limit",
    )
    configs = [
        {"nprobe": p, "rerank_factor": f, "query_threads": t}
        for p, f, t in itertools.product(probes, factors, threads)
    ]
    script = REPO / "benchmark/IVF/sift/query.sh"
    driver = args.driver.resolve()
    require(driver.is_file() and os.access(driver, os.X_OK), "missing native IVF driver")
    profile_path = Path(os.environ.get("BENCH_DATASETS_DIR", REPO / "benchmark/datasets")) / (
        dataset + ".sh"
    )
    inputs = {
        "prepared": prepared_path,
        "profile": profile_path,
        "driver": driver,
        "query_script": script,
        "harness": REPO / "benchmark/IVF/harness.py",
        "common": REPO / "benchmark/IVF/common.sh",
        "utils": REPO / "benchmark/scripts/utils.sh",
        "profile_loader": REPO / "benchmark/datasets/_load.sh",
        "orchestrator": Path(__file__),
        "query": Path(prepared["query"]),
        "truth": Path(prepared["truth"]),
        "manifest": work / "index" / (dataset + ".manifest"),
        **shared_module_paths(),
    }
    identities = {name: digest(path) for name, path in inputs.items()}
    require(
        identities["query"] == prepared["query_identity"]
        and identities["truth"] == prepared["truth_identity"]
        and identities["manifest"] == prepared["manifest_identity"],
        "prepared input identity changed",
    )
    env = dict(
        os.environ,
        DATASET=dataset,
        EVAL_TEMPFILES=str(work.parents[2]),
        IVF_DRIVER_BIN=str(driver),
        IVF_QUERY_MEMORY_BUDGET_GB=str(args.memory_budget_gb),
        IVF_BUFFER_FRAMES=str(args.buffer_frames),
        RECALL_AT=str(args.recall_at),
    )
    trials = []
    for c, config in enumerate(configs):
        for trial in range(args.warmups + args.repeats):
            warmup = trial < args.warmups
            tag = f"{args.tag}_c{c}_{'w' if warmup else 'r'}{trial if warmup else trial - args.warmups}"
            path = work / "runs" / tag
            require(
                not path.exists() and not path.is_symlink(), f"sweep run destination exists: {path}"
            )
            trials.append({"config": c, "warmup": warmup, "run_tag": tag})
    for directory in (work / "sweeps", work / "runs"):
        require(not directory.is_symlink(), "sweep/run parent must not be a symlink")
    root = work / "sweeps" / args.tag
    require(not root.exists() and not root.is_symlink(), f"sweep destination exists: {root}")
    plan = {
        "version": 1,
        "kind": "ivf_query_sweep",
        "inputs": {k: str(p) for k, p in inputs.items()},
        "identities": identities,
        "configs": configs,
        "trials": trials,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "warmup_policy": "separate_discarded_processes_os_cache_only_no_inprocess_warmup",
        "affinity": affinity(args.cpus),
        "runtime_environment": runtime_environment(),
        "timeout_s_per_trial": args.timeout,
        "recall_at": args.recall_at,
        "process_wall_scope": "entire_query_script_including_hashes_open_gt_scoring_export",
        "memory_budget_gb": args.memory_budget_gb,
        "buffer_frames": args.buffer_frames,
        "gates": {"min_recall_percent": args.min_recall, "max_peak_rss_mb": args.max_peak_rss_mb},
        "cache_policy": "uncontrolled_after_full_bundle_hash_no_cache_drop",
        "order_policy": "config_order_then_warmups_then_repeats",
    }
    root.mkdir(parents=True)
    write_json(root / "plan.json", plan)
    records = []
    for trial in trials:
        config = configs[trial["config"]]
        env.update(
            IVF_RUN_TAG=trial["run_tag"],
            IVF_NPROBE=str(config["nprobe"]),
            IVF_RERANK_FACTOR=str(config["rerank_factor"]),
            QUERY_THREADS=str(config["query_threads"]),
        )
        wall = execute(
            ["bash", str(script)], root / (trial["run_tag"] + ".log"), args.timeout, args.cpus, env
        )
        path = work / "runs" / trial["run_tag"]
        record = read_json(path / "result.json")
        require(
            record["baseline"] == "IVF"
            and record["phase"] == "query"
            and record["workload"] == "query",
            "unexpected metrics",
        )
        require(
            all(record[k] == v for k, v in config.items())
            and record["recall_at"] == args.recall_at,
            "sweep native configuration mismatch",
        )
        require(record["buffer_pool_frames"] == args.buffer_frames, "sweep buffer frame mismatch")
        require(0 <= record["recall"] <= 100, "invalid recall")
        aggregate([record], ("query_qps", "query_lat_p99_us", "peak_process_rss_mb", "recall"))
        require(
            {name: digest(path) for name, path in inputs.items()} == identities,
            "sweep input/tool identity changed",
        )
        records.append(
            dict(
                trial,
                metrics=record,
                process_wall_s=wall,
                output_dir=str(path),
                result_identity=digest(path / "result.json"),
                neighbors_identity=digest(path / "neighbors.bin"),
            )
        )
        # Per-trial completion record preserves successful observations even if
        # a later trial fails. Only the final summary signifies a complete sweep.
        write_json(root / (trial["run_tag"] + ".json"), records[-1])
    groups = []
    passed = True
    for c, config in enumerate(configs):
        measured = [r["metrics"] for r in records if r["config"] == c and not r["warmup"]]
        stats = aggregate(
            measured,
            (
                "query_qps",
                "recall",
                "query_time_s",
                "open_time_s",
                "query_lat_avg_us",
                "query_lat_p50_us",
                "query_lat_p95_us",
                "query_lat_p99_us",
                "peak_process_rss_mb",
                "heap_pages_read",
                "estimated_workspace_bytes",
                "active_query_threads",
            ),
        )
        gate = stats["recall"]["min"] >= args.min_recall and (
            args.max_peak_rss_mb is None
            or stats["peak_process_rss_mb"]["max"] <= args.max_peak_rss_mb
        )
        passed = passed and gate
        groups.append(
            {
                "config": config,
                "measured_repeats": len(measured),
                "stats": stats,
                "gates_passed": gate,
            }
        )
    write_json(
        root / "summary.json",
        {
            "kind": plan["kind"],
            "completed": True,
            "gates_passed": passed,
            "plan_identity": digest(root / "plan.json"),
            "groups": groups,
            "trial_records": records,
            "aggregation_policy": "min_median_max_across_runs_not_pooled_query_percentiles",
            "production_qualified": False,
        },
    )
    require(passed, f"declared sweep gates failed; inspect {root / 'summary.json'}")
    print(f"Query sweep complete: {root / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--memory-budget-gb", type=finite, required=True)
        p.add_argument(
            "--timeout",
            type=finite,
            required=True,
            help="whole child process wall seconds, per mode/trial",
        )
        p.add_argument(
            "--cpus", help="optional comma-separated CPU IDs, validated against current affinity"
        )
        p.add_argument(
            "--max-peak-rss-mb", type=finite, help="post-run gate, not a hard RSS allocation limit"
        )
        if name == "train":
            for field in ("data", "trainer", "output"):
                p.add_argument("--" + field, type=Path, required=True)
            p.add_argument("--dtype", choices=("float", "int8", "uint8"), required=True)
            for field in (
                "dim",
                "full-count",
                "base-count",
                "nlist",
                "iterations",
                "threads",
                "evaluation-rows",
                "exact-work-limit",
                "assignment-work-limit",
                "evaluation-work-limit",
            ):
                p.add_argument("--" + field, type=positive, required=True)
            p.add_argument("--sample-rate", type=finite, required=True)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--router-seed", type=int, default=100)
            p.add_argument("--router-degree", type=positive, default=16)
            p.add_argument("--construction-breadth", type=positive, default=200)
            p.add_argument("--insert-breadth", type=positive, default=64)
            p.add_argument(
                "--training-breadth",
                type=positive,
                help="independent training assignment breadth (defaults to legacy insert-breadth)",
            )
            p.add_argument("--init", choices=("sample", "random_partition"), default="sample")
            p.add_argument(
                "--modes",
                default="exact,hnsw",
                help="comma-separated exact,hnsw,ann-rebuild,ann-reuse,ann-reuse-seeded,ann-reuse-count,ann-reuse-seeded-count,lv,lv-plus",
            )
            p.add_argument("--training-rerank-factor", type=finite, default=3)
            p.add_argument("--training-refine-breadth", type=positive, default=128)
            p.add_argument(
                "--routing-work-limit",
                type=positive,
                default=1000000000,
                help="per-setup refinement/PCA work allowance",
            )
            p.add_argument(
                "--max-sse-ratio", type=finite, help="optional HNSW/exact exact-evaluation SSE gate"
            )
        else:
            p.add_argument("--work-dir", type=Path, required=True)
            p.add_argument("--driver", type=Path, required=True)
            for field in ("tag", "nprobes", "rerank-factors", "threads"):
                p.add_argument("--" + field, required=True)
            p.add_argument("--repeats", type=positive, required=True)
            p.add_argument("--warmups", type=int, default=0)
            p.add_argument("--buffer-frames", type=positive, default=64)
            p.add_argument("--recall-at", type=positive, required=True)
            p.add_argument(
                "--min-recall",
                type=float,
                default=0,
                help="minimum percent recall across measured trials",
            )
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
