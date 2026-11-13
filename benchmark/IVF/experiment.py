#!/usr/bin/env python3
"""Fixed SIFT100M query sweep; workload scripts own preparation and execution."""
import csv
import os
from pathlib import Path
import shlex
import struct
import sys

from common import execute, header, read_json, require
from settings import DEFAULT_ENV, ROUTER_BACKEND_ID, build_settings

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
FULL_COUNT = 100_000_000
BASE_COUNT = 90_000_000
DIM = 128
TIMEOUT_S = 172800
NLISTS = (1000, 10000, 100000, 200000, 300000, 400000, 500000)
PROBES = (16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 384, 512)
PRESET = {
    **DEFAULT_ENV,
    "DATASET": "sift100m", "RECALL_AT": "10",
}


def environment():
    """Keep location choices, not inherited experiment tuning."""
    def location(name, default):
        return str(Path(os.environ.get(name) or default).expanduser().absolute())

    ann_data = location("ANN_DATA_ROOT", Path.home() / "project/ann-data")
    dataset = location("SIFT100M_ROOT", os.environ.get("DATA_ROOT") or Path(ann_data) / "sift100m")
    data = location("DATA_BIN", Path(dataset) / "base.bin")
    build = location("IVF_BUILD_DIR", REPO / "build/ivf-benchmark")
    gt = location("GT_ARTIFACT_DIR", Path(dataset) / "groundtruth")
    ignored = {"INSERT_POINTS", "DELETE_POINTS", "QUERY_POINTS", "UPDATE_QUERY_BATCH", "QUERY_RATIO",
               "INSERT_CAP", "DELETE_CAP", "EVAL_CACHE_NO_STORE"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("IVF_") and k not in ignored}
    env.update(PRESET)
    env.update(
        DATA_BIN=data, QUERY_BIN=location("QUERY_BIN", Path(dataset) / "query.10K.bin"),
        IVF_BUILD_DIR=build, IVF_PQ_PREFIX=location("IVF_PQ_PREFIX", Path(data).parent / "pq/pq64"),
        EVAL_TEMPFILES=location("EVAL_TEMPFILES", Path.home() / "project/ann-eval/sift100m-query-grid"),
        EVAL_CACHED=location("EVAL_CACHED", Path.home() / "project/ann-eval/cache"),
        GT_ARTIFACT_DIR=gt,
        GT_BIN_KCACHED=location("GT_BIN_KCACHED", Path(gt) / "sift100m_100000000_gt100.bin"),
        INCREMENTAL_GT_BIN=str(Path(build) / "tests/incremental_groundtruth"),
        BENCH_DATASETS_DIR=str(REPO / "benchmark/datasets"),
    )
    return env


def validate_pq64(prefix):
    """Check PQ64 shape/framing; detailed value validation stays in native tools."""
    header(prefix + "_pq_compressed.bin", "uint8", FULL_COUNT, 64)
    path = Path(prefix + "_pq_pivots.bin")
    with path.open("rb") as source:
        rows, cols, *offsets = struct.unpack("<II5Q", source.read(48))
        require((rows, cols) == (5, 1) and offsets[0] >= 48
                and all(a < b for a, b in zip(offsets, offsets[1:]))
                and offsets[-1] == path.stat().st_size, "PQ64 requires five valid pivot offsets")
        for start, end, shape in zip(offsets, offsets[1:], ((256, DIM), (DIM, 1), (DIM, 1), (65, 1))):
            source.seek(start)
            require(struct.unpack("<II", source.read(8)) == shape and end - start == 8 + shape[0] * shape[1] * 4,
                    "PQ64 requires matching 256-centroid pivots with 64 chunks")


def next_run(root):
    sweeps = root / "sweeps"
    sweeps.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        run = sweeps / f"run-{number:04d}"
        try:
            run.mkdir()
            return run
        except FileExistsError:
            number += 1


def check_preparation(work, env):
    """Never relabel a differently configured immutable base as this preset."""
    prepared = read_json(work / "prepared.json")
    require(prepared.get("router_backend") == ROUTER_BACKEND_ID and prepared.get("settings") == build_settings(env),
            "Existing preparation has different settings; preserve it and choose another EVAL_TEMPFILES root")
    require((work / "index/sift100m.pq.bin").read_bytes() == Path(env["IVF_PQ_PREFIX"] + "_pq_pivots.bin").read_bytes(),
            "Existing preparation uses different PQ pivots; choose another EVAL_TEMPFILES root")


def run():
    env = environment()
    validate_pq64(env["IVF_PQ_PREFIX"])
    root = Path(env["EVAL_TEMPFILES"])
    evidence = next_run(root)
    execute([
        sys.executable, str(REPO / "benchmark/scripts/ensure_gt.py"),
        "--base-file", env["DATA_BIN"], "--query-file", env["QUERY_BIN"],
        "--full-count", str(FULL_COUNT), "--base-count", str(BASE_COUNT), "--query-only",
        "--k-out", "10", "--expected-queries", "10000", "--num-threads", "16",
        "--cache-dir", env["GT_ARTIFACT_DIR"], "--cached-gt", env["GT_BIN_KCACHED"], "--generate",
    ], evidence / "groundtruth.log", TIMEOUT_S, None, env=env, stdout_path=evidence / "selected_gt.txt")
    env["GT_BIN_KCACHED"] = (evidence / "selected_gt.txt").read_text().strip()
    require(Path(env["GT_BIN_KCACHED"]).is_file(), "GT helper did not return a usable path")
    names = set(PRESET) | {"DATA_BIN", "QUERY_BIN", "IVF_BUILD_DIR", "IVF_PQ_PREFIX", "EVAL_TEMPFILES",
                           "EVAL_CACHED", "GT_ARTIFACT_DIR", "GT_BIN_KCACHED", "INCREMENTAL_GT_BIN"}
    with (evidence / "settings.env").open("x") as out:
        for name in sorted(names):
            out.write(f"{name}={shlex.quote(env[name])}\n")
        out.write("TRAINING_ASSIGNMENT_SEARCH_LIMIT=3000000000\n")
    with (evidence / "grid.tsv").open("x", newline="") as out:
        grid = csv.writer(out, delimiter="\t")
        grid.writerow(("nlist", "nprobe", "router_breadth"))
        grid.writerows((nl, np, 128) for nl in NLISTS for np in PROBES)
    with (evidence / "results.csv").open("x", newline="") as out:
        results = csv.writer(out)
        results.writerow(("nlist", "nprobe", "router_breadth", "recall_at_10_percent", "query_qps", "query_time_s", "output_dir"))
        out.flush()
        for nlist in NLISTS:
            env.update(IVF_NLIST=str(nlist), EVAL_TEMPFILES=str(root / f"nlist-{nlist}"))
            work = Path(env["EVAL_TEMPFILES"]) / "sift100m/IVF/query"
            if (work / "prepared.json").is_file():
                print(f"Reusing preparation nlist={nlist}: {work}", flush=True)
                check_preparation(work, env)
            else:
                print(f"Preparing nlist={nlist}; log: {evidence / f'nl{nlist}-prepare.log'}", flush=True)
                execute(["bash", str(SCRIPT_DIR / "sift/query_prepare.sh")],
                        evidence / f"nl{nlist}-prepare.log", TIMEOUT_S, None, env=env)
            for probe in PROBES:
                env.update(IVF_NPROBE=str(probe), IVF_RUN_TAG=f"{evidence.name}_np{probe}_rb128")
                print(f"Query nlist={nlist} nprobe={probe} breadth=128; logs: {evidence}", flush=True)
                execute(["bash", str(SCRIPT_DIR / "sift/query.sh")],
                        evidence / f"nl{nlist}-np{probe}.log", TIMEOUT_S, None, env=env)
                folder = work / "runs" / env["IVF_RUN_TAG"]
                result = read_json(folder / "result.json")
                results.writerow([result[key] for key in ("nlist", "nprobe", "router_breadth", "recall", "query_qps", "query_time_s")]
                                 + [str(folder)])
                out.flush()
                (folder / "query.heap").unlink()
                (folder / "query.heap.pq.bin").unlink()
                with (evidence / "heap-cleanup.log").open("a") as log:
                    log.write(str(folder) + "\n")
    print(f"Completed {len(NLISTS) * len(PROBES)} query passes: {evidence / 'results.csv'}")


def main():
    if len(sys.argv) != 1:
        print("Usage: python3 benchmark/IVF/experiment.py (no arguments)", file=sys.stderr)
        return 2
    try:
        run()
    except (OSError, ValueError, struct.error) as error:
        print(f"experiment: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
