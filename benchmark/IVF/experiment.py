#!/usr/bin/env python3
"""Full-scale SIFT workload and query-grid recipes. See README.md for setup.

This is orchestration, not a benchmark implementation: the existing workload
scripts retain preparation, mutable-session ownership and native timing policy.
Importing this module performs no I/O and never starts an experiment.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from common import digest, execute, header, positive, read_json, require, require_absent, write_json

REPO = Path(__file__).resolve().parents[2]
MODES = ("workloads", "coarse", "expanded", "sift100m-wide", "sift100m-refine")
PROFILES = {"sift10m": (10000000, 9000000, 1000000, 15),
            "sift100m": (100000000, 90000000, 10000000, 10)}
NLISTS = (100, 1000, 10000, 100000, 200000, 300000)
REFINE_NLISTS = (100000, 200000, 300000, 400000)
REFINE_PROBES = (16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 384, 512)
WORKLOADS = ("query", "insert", "delete", "update")
OVERRIDES_TO_CLEAR = ("INSERT_POINTS", "DELETE_POINTS", "QUERY_POINTS", "UPDATE_QUERY_BATCH",
                      "QUERY_RATIO", "INSERT_CAP", "DELETE_CAP", "IVF_RUN_TAG",
                      "IVF_COST_MODEL_REPORT", "IVF_SCHEDULER_SEED")


def schedule(mode):
    """Return ordered points and documented omissions, without touching inputs."""
    require(mode in MODES and mode != "workloads", "unknown grid mode")
    nlists = REFINE_NLISTS if mode == "sift100m-refine" else NLISTS
    coarse = {100: (1, 2, 4, 8), 1000: (1, 4, 16, 64), 10000: (4, 16, 64, 256)}
    points, omissions = [], []
    for nlist in nlists:
        if mode == "sift100m-refine":
            probes = REFINE_PROBES
        elif mode == "sift100m-wide":
            probes = [2 ** i for i in range(14) if 2 ** i <= nlist]
        elif mode == "coarse":
            probes = coarse.get(nlist, (16, 64, 256, 1024))
        else:
            probes = [2 ** i for i in range(0 if nlist <= 10000 else 4, 11) if 2 ** i < nlist]
        for probe in probes:
            breadths = (128,)
            if mode == "sift100m-refine":
                breadths += {16: (64,), 32: (64,), 48: (256,), 64: (256,), 128: (512,)}.get(probe, ())
            elif mode == "expanded" and probe <= 128:
                breadths += (256, 512)
            seen = set()
            for breadth in breadths:
                effective = max(breadth, min(nlist, 3 * probe))
                point = dict(nlist=nlist, nprobe=probe, router_breadth=breadth,
                             effective_router_breadth=effective, tag=f"np{probe}_rb{breadth}")
                reason = ("extra breadth exceeds nlist" if breadth != 128 and breadth > nlist else
                          "duplicate effective breadth" if effective in seen else None)
                if reason:
                    omissions.append(dict(point, reason=reason))
                else:
                    seen.add(effective)
                    points.append(point)
    return points, omissions


def full_scale_environment(source, dataset, mode):
    """One owner for the README recipe, not changes to library/harness defaults."""
    require(dataset in PROFILES, "full-scale recipes require sift10m or sift100m")
    env = dict(source)
    for key in OVERRIDES_TO_CLEAR:
        env.pop(key, None)
    data_root = Path(env.get("DATA_ROOT") or Path(env.get("ANN_DATA_ROOT") or
                     Path.home() / "project/ann-data") / dataset)
    native = Path(env.get("IVF_BUILD_DIR") or REPO / "build/ivf-benchmark").resolve()
    defaults = {
        "DATA_BIN": data_root / "base.bin", "QUERY_BIN": data_root / "query.10K.bin",
        "GT_ARTIFACT_DIR": data_root / "groundtruth",
        "EVAL_CACHED": Path.home() / "project/ann-eval/cache", "IVF_BUILD_DIR": native,
        "IVF_BUILD_BIN": native / "tests/ivf_build", "IVF_DRIVER_BIN": native / "tests/ivf_driver",
        "IVF_MEMORY_BUDGET_GB": "96", "IVF_BUFFER_FRAMES": "262144", "QUERY_THREADS": "8",
        "IVF_WARMUP_QUERIES": "0" if mode == "workloads" else "64",
    }
    if mode == "workloads":
        defaults.update(IVF_INSERT_THREADS="8", IVF_INSERT_INPUT_POLICY="staged", NTHREADS="16")
    for key, value in defaults.items():
        if not env.get(key):
            env[key] = str(value)
    if not env.get("IVF_QUERY_MEMORY_BUDGET_GB"):
        env["IVF_QUERY_MEMORY_BUDGET_GB"] = env["IVF_MEMORY_BUDGET_GB"]
    env.update(DATASET=dataset, REPO_DIR=str(REPO),
               IVF_SAMPLE_RATE="1", IVF_MAX_REPS=str(PROFILES[dataset][3]),
               IVF_TRAINER_MODE="lv-plus", IVF_INIT="random_partition", IVF_TRAINING_BREADTH="64",
               IVF_SEED="0", IVF_BUILD_THREADS="20", IVF_TRAINING_RERANK_FACTOR="3",
               IVF_TRAINING_REFINE_BREADTH="128", IVF_PAGE_SIZE="4096",
               OMP_NUM_THREADS="20", OMP_THREAD_LIMIT="20", MKL_NUM_THREADS="1", RECALL_AT="10",
               IVF_OPTIMIZED_MODES="1", IVF_ASSIGNMENT_ROUTER_MODE="pca-refined",
               IVF_BUILD_ROUTER_RERANK_FACTOR="3", IVF_PCA_VARIANCE="0.95", IVF_REFINE_EF="128",
               IVF_QUERY_BREADTH="128", IVF_INSERT_BREADTH="64", IVF_ROUTING_WORK_LIMIT="3000000000",
               IVF_ROUTER_MODE="pca-refined", IVF_ROUTER_RERANK_FACTOR="3", IVF_RERANK_FACTOR="3",
               IVF_ROUTER_BREADTH="128", IVF_NPROBE="64" if mode == "workloads" else "1")
    if mode == "workloads":
        env.update(IVF_NLIST="200000", UPDATE_PERCENT="10", IVF_SCHEDULER="update_driven",
                   UPDATE_BATCH=str(PROFILES[dataset][2] // 10))
    else:
        env.pop("IVF_NLIST", None)  # Each grid preparation declares its own count.
    require(Path(env["EVAL_CACHED"]).is_absolute(), "EVAL_CACHED must be absolute")
    for key in ("DATA_BIN", "QUERY_BIN", "IVF_PQ_PREFIX", "IVF_BUILD_DIR", "IVF_BUILD_BIN",
                "IVF_DRIVER_BIN", "EVAL_CACHED", "GT_ARTIFACT_DIR"):
        if env.get(key):
            env[key] = str(Path(env[key]).resolve())
    return env


def validate_inputs(env, dataset):
    require(not env.get("IVF_GT_FILE"), "unset IVF_GT_FILE; full-scale recipes coverage-check GT_BIN_KCACHED")
    require(env.get("IVF_PQ_PREFIX"), "export IVF_PQ_PREFIX to verified full-source PQ64 inputs")
    for key in ("IVF_BUILD_BIN", "IVF_DRIVER_BIN"):
        require(Path(env[key]).is_file() and os.access(env[key], os.X_OK), f"missing executable {key}")
    full = PROFILES[dataset][0]
    header(env["DATA_BIN"], "uint8", full, 128)
    header(env["QUERY_BIN"], "uint8", 10000, 128)
    header(env["IVF_PQ_PREFIX"] + "_pq_compressed.bin", "uint8", full, 64)
    require(Path(env["IVF_PQ_PREFIX"] + "_pq_pivots.bin").is_file(), "missing PQ pivots")


def selected_gt(path):
    selected = path.read_text().strip()
    require(selected and "\n" not in selected and Path(selected).is_file(),
            "GT selection is missing or invalid; retain the actual binary, not just selected_gt.txt")
    return selected


def validate_previous(previous, env, nlists, base):
    """Validate every reused base before claiming any new output root."""
    previous = previous.resolve(strict=True)
    old_plan = read_json(previous / "plan.json")
    for key in ("DATA_BIN", "QUERY_BIN", "IVF_PQ_PREFIX"):
        require(Path(old_plan["settings"][key]).resolve() == Path(env[key]).resolve(),
                f"{key} differs from previous grid; do not mix input populations")
    env["GT_BIN_KCACHED"] = selected_gt(previous / "selected_gt.txt")
    work_dirs = {}
    for nlist in nlists:
        parent = previous / f"nlist-{nlist}"
        work = parent / env["DATASET"] / "IVF/query"
        prepared_path = work / "prepared.json"
        old = read_json(prepared_path)
        for key in ("sample_rate", "max_reps", "trainer_mode", "init", "training_breadth",
                    "training_rerank_factor", "training_refine_breadth", "seed", "optimized_modes",
                    "pca_variance", "refine_ef", "query_breadth", "router_mode", "router_rerank_factor"):
            variable = {"router_mode": "IVF_ASSIGNMENT_ROUTER_MODE",
                        "router_rerank_factor": "IVF_BUILD_ROUTER_RERANK_FACTOR"}.get(key, "IVF_" + key.upper())
            require(str(old["settings"].get(key)) == env[variable],
                    f"{prepared_path}: {key} differs from follow-up plan")
        require(int(old["settings"]["nlist"]) == nlist and old["profile"]["base_count"] == base,
                f"{prepared_path}: incorrect prepared population/nlist")
        work_dirs[nlist] = work
    return work_dirs


def groundtruth_command(env, dataset, generate):
    full, base, updates, _ = PROFILES[dataset]
    command = [sys.executable, "benchmark/scripts/ensure_gt.py", "--base-file", env["DATA_BIN"],
               "--query-file", env["QUERY_BIN"], "--full-count", str(full), "--base-count", str(base),
               "--insert-count", str(updates), "--delete-count", str(updates), "--mixed-rounds", "10",
               "--mixed-insert-cap", str(updates // 10), "--mixed-delete-cap", str(updates // 10),
               "--k-out", "10", "--expected-queries", "10000", "--num-threads", "20",
               "--cache-dir", env["GT_ARTIFACT_DIR"]]
    if generate:
        command.append("--generate")
    if env.get("GT_BIN_KCACHED"):
        command += ["--cached-gt", env["GT_BIN_KCACHED"]]
    return command


class Phases:
    """Deadlines and append-only phase receipts; native timers remain untouched."""

    def __init__(self, root, limits):
        self.root, self.limits = root, limits
        self.started = time.monotonic()
        self.deadline = self.started + limits["total"]
        self.records = []

    def run(self, name, command, env, deadline, stdout_path=None):
        remaining = min(deadline, self.deadline) - time.monotonic()
        require(remaining > 0, f"budget exhausted before {name}; phase not started")
        start = time.monotonic()
        record = dict(name=name, command=command, timeout_s=remaining, completed=False)
        try:
            execute(command, self.root / (name + ".log"), remaining, None, env,
                    cwd=REPO, stdout_path=stdout_path)
            record.update(completed=True, returncode=0)
        except BaseException as error:
            record["error"] = repr(error)
            raise
        finally:
            record["process_wall_s"] = time.monotonic() - start
            self.records.append(record)
            with (self.root / "phases.jsonl").open("a") as out:
                out.write(json.dumps(record) + "\n")


def preflight(phases, env, dataset, generate):
    full, base, updates, _ = PROFILES[dataset]
    check = ('source benchmark/datasets/_load.sh; '
             '[[ "$DATA_TYPE" = uint8 && "$DIM" = 128 && "$METRIC" = l2 && '
             f'"$NPTS_FULL" = {full} && "$NPTS_BASE" = {base} && "$UPDATE_POINTS" = {updates} ]]')
    phases.run("profile", ["bash", "-euc", check], env, phases.deadline)
    path = phases.root / "selected_gt.txt"
    phases.run("groundtruth", groundtruth_command(env, dataset, generate), env, phases.deadline, path)
    env["GT_BIN_KCACHED"] = selected_gt(path)


def cleanup_private_query_heaps(root, folder, name):
    heaps = [folder / "query.heap", folder / "query.heap.pq.bin"]
    require(all(p.is_file() and not p.is_symlink() for p in heaps),
            f"{name}: expected regular private query heaps for cleanup")
    for heap in heaps:
        size = heap.stat().st_size
        heap.unlink()
        with (root / "heap-cleanup.jsonl").open("a") as out:
            out.write(json.dumps(dict(phase=name, path=str(heap), bytes=size, deleted=True)) + "\n")


def result_row(folder, point, env, base, wall_s):
    native, protocol = (read_json(folder / name) for name in ("result.json", "execution_protocol.json"))
    expected = dict(baseline="IVF", phase="query", workload="query", query_count=10000,
                    base_count=base, nlist=point["nlist"], pq_chunks=64, recall_at=10,
                    nprobe=point["nprobe"], router_breadth=point["router_breadth"],
                    router_mode="pca-refined", rerank_factor=3, router_rerank_factor=3,
                    query_threads=int(env["QUERY_THREADS"]),
                    explicit_warmup_query_count=int(env["IVF_WARMUP_QUERIES"]),
                    index_format="mutable", input_index_format="immutable", live_count=base)
    require(all(native.get(k) == v for k, v in expected.items()), "native result does not match frozen plan")
    require(math.isfinite(native["recall"]) and 0 <= native["recall"] <= 100 and
            all(math.isfinite(native[k]) and native[k] > 0 for k in ("query_qps", "query_time_s")),
            "invalid native query metrics")
    require(protocol["completed"] and protocol["native_driver_process_count"] == 1,
            "expected one completed standalone query process")
    return dict(point, recall_at_10_percent=native["recall"], query_qps=native["query_qps"],
                query_time_s=native["query_time_s"], query_process_wall_s=protocol["query_process_wall_s"],
                harness_workflow_wall_s=protocol["harness_workflow_wall_s"],
                query_script_wall_s=wall_s, output_dir=str(folder))


def run_grid(phases, env, mode, points, previous, previous_work, rows):
    root, dataset = phases.root, env["DATASET"]
    refine = mode == "sift100m-refine"
    prefix = "refine_" + hashlib.sha256(str(root).encode()).hexdigest()[:12] + "_" if refine else ""
    with (root / "results.csv").open("x", newline="") as table:
        writer = None
        for nlist in dict.fromkeys(p["nlist"] for p in points):
            deadline = min(phases.deadline, time.monotonic() + phases.limits["nlist"])
            child = dict(env, IVF_NLIST=str(nlist), EVAL_TEMPFILES=str(root / f"nlist-{nlist}"))
            work = Path(child["EVAL_TEMPFILES"]) / dataset / "IVF/query"
            if nlist in previous_work:
                child["EVAL_TEMPFILES"] = str(previous / f"nlist-{nlist}")
                work = previous_work[nlist]
                print(f"nlist={nlist}: reuse previous prepared base (no build)", flush=True)
            else:
                child["IVF_ROUTING_WORK_LIMIT"] = "5000000000" if refine else "3000000000"
                print(f"nlist={nlist}: prepare/restore full base", flush=True)
                phases.run(f"nl{nlist}-prepare", ["bash", "benchmark/IVF/sift/query_prepare.sh"], child, deadline)
            for point in (p for p in points if p["nlist"] == nlist):
                child.update(IVF_NPROBE=str(point["nprobe"]), IVF_ROUTER_BREADTH=str(point["router_breadth"]),
                             IVF_RUN_TAG=prefix + point["tag"])
                name = f'nl{nlist}-{point["tag"]}'
                print(name, flush=True)
                phases.run(name, ["bash", "benchmark/IVF/sift/query.sh"], child,
                           min(deadline, time.monotonic() + phases.limits["query"]))
                folder = work / "runs" / child["IVF_RUN_TAG"]
                row = result_row(folder, point, env, PROFILES[dataset][1], phases.records[-1]["process_wall_s"])
                if writer is None:
                    writer = csv.DictWriter(table, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                table.flush()
                rows.append(row)
                if mode in ("sift100m-wide", "sift100m-refine"):
                    cleanup_private_query_heaps(root, folder, name)
    require(len(rows) == len(points), "grid incomplete")


def run_workloads(phases, env):
    for workload in WORKLOADS:
        deadline = min(phases.deadline, time.monotonic() + phases.limits["nlist"])
        for suffix in ("_prepare", ""):
            name = workload + ("-prepare" if suffix else "-run")
            phases.run(name, ["bash", f"benchmark/IVF/sift/{workload}{suffix}.sh"], env, deadline)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, default=os.environ.get("IVF_GRID_MODE", "sift100m-wide"))
    p.add_argument("--dataset", choices=PROFILES, default=os.environ.get("DATASET", "sift100m"))
    p.add_argument("--output", type=Path, default=os.environ.get("EVAL_TEMPFILES"), help="fresh absolute eval root")
    p.add_argument("--previous-root", type=Path, default=os.environ.get("IVF_GRID_PREVIOUS_ROOT"))
    p.add_argument("--dry-run", action="store_true", help="preview only; no inputs read or files created")
    p.add_argument("--no-generate-gt", action="store_true", help="check existing GT only (always true in refinement)")
    for key in ("total", "nlist", "query"):
        p.add_argument(f"--{key}-timeout-s", type=positive, default=os.environ.get(f"IVF_GRID_{key.upper()}_TIMEOUT_S"))
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        require(args.mode in MODES and args.dataset in PROFILES, "unknown experiment mode/dataset")
        if args.mode in ("sift100m-wide", "sift100m-refine"):
            require(args.dataset == "sift100m", f"{args.mode} requires DATASET=sift100m")
        dry = os.environ.get("IVF_GRID_DRY_RUN", "0")
        require(dry in ("0", "1"), "IVF_GRID_DRY_RUN must be 0 or 1")
        points, omissions = ([], []) if args.mode == "workloads" else schedule(args.mode)
        nlists = list(dict.fromkeys(point["nlist"] for point in points))
        if args.dry_run or dry == "1":
            if args.mode == "workloads":
                print("Independent prepare/run pairs: " + ", ".join(WORKLOADS) + "; no execution")
            else:
                columns = ("nlist", "nprobe", "router_breadth", "effective_router_breadth")
                print("\t".join(columns))
                for point in points:
                    print(*(point[k] for k in columns), sep="\t")
                action = ("3 prepared-base reuses, 1 index preparation" if args.mode == "sift100m-refine"
                          else f"{len(nlists)} index preparations")
                print(f"{args.mode}: {len(points)} query passes, {action}; no execution")
            return 0
        require(args.output is not None and args.output.is_absolute(), "supply a fresh absolute --output or EVAL_TEMPFILES")
        root = args.output
        require_absent(root, f"preserve the existing root and choose a new one: {root}")
        env = full_scale_environment(os.environ, args.dataset, args.mode)
        env["EVAL_TEMPFILES"] = str(root)
        env["INCREMENTAL_GT_BIN"] = str(root / "incremental_groundtruth")
        validate_inputs(env, args.dataset)
        previous, previous_work = None, {}
        if args.mode == "sift100m-refine":
            require(args.previous_root is not None, "sift100m-refine requires --previous-root or IVF_GRID_PREVIOUS_ROOT")
            previous = args.previous_root.resolve(strict=True)
            previous_work = validate_previous(previous, env, REFINE_NLISTS[:-1], PROFILES[args.dataset][1])
        defaults = ((21600, 7200, 900) if args.mode in ("coarse", "expanded") else (604800, 86400, 21600))
        limits = {k: getattr(args, k + "_timeout_s") or default
                  for k, default in zip(("total", "nlist", "query"), defaults)}
        root.mkdir(parents=True, exist_ok=False)
        phases, rows = Phases(root, limits), []
        plan = dict(mode=args.mode, dataset=args.dataset, source_rows=PROFILES[args.dataset][0],
                    base_rows=PROFILES[args.dataset][1], execution_mode="ivf_experiment_cli",
                    runner_identity=digest(__file__), queries_per_point=10000, points=points,
                    omissions=omissions, timeouts_s=limits,
                    settings={k: v for k, v in env.items() if k.startswith(("IVF_", "OMP_", "MKL_")) or k in (
                        "DATASET", "DATA_BIN", "QUERY_BIN", "ANN_DATA_ROOT", "GT_ARTIFACT_DIR", "EVAL_CACHED",
                        "BENCH_DATASETS_DIR", "INCREMENTAL_GT_BIN", "QUERY_THREADS", "RECALL_AT", "NTHREADS",
                        "UPDATE_PERCENT", "UPDATE_BATCH")}, selected_configuration=None,
                    previous_root=str(previous) if previous else None,
                    cache_policy="separate query drivers; no cache drop; hashing/open/copies may warm caches",
                    private_query_heap_policy=("delete_successful_pass_private_heap_and_pq_after_result_recording"
                                               if args.mode in ("sift100m-wide", "sift100m-refine") else "retain"),
                    nlist_routing_work_limits={str(n): 5000000000 if n == 400000 else 3000000000 for n in nlists},
                    gt_policy="conservative four-workload coverage; no generation in refinement")
        write_json(root / "plan.json", plan)
        try:
            preflight(phases, env, args.dataset, not args.no_generate_gt and previous is None)
            if args.mode == "workloads":
                run_workloads(phases, env)
            else:
                run_grid(phases, env, args.mode, points, previous, previous_work, rows)
            write_json(root / "summary.json", dict(completed=True, production_qualified=False, records=rows,
                       phases=phases.records, selected_configuration=None, wall_s=time.monotonic() - phases.started))
        except BaseException as error:
            write_json(root / "failure.json", dict(completed=False, error=repr(error), completed_points=len(rows),
                       wall_s=time.monotonic() - phases.started, partial_evidence_retained=True))
            raise
        print(f"Completed: {root}", flush=True)
        return 0
    except (OSError, ValueError) as error:
        p.exit(1, f"ERROR: {error}\n")


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
