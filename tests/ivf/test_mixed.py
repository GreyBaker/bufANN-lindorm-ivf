#!/usr/bin/env python3
"""All-dtype update-driven/weighted/fixed CLI + two-round prepare/run/GT parity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from test_cli import write_pivots


def run(script, env, success=True):
    p = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, timeout=180)
    assert (p.returncode == 0) == success, (p.stdout, p.stderr)
    if not success:
        assert "ERROR:" in p.stdout + p.stderr, (p.stdout, p.stderr)
    return p


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="mixed_", dir=args.scratch) as tmp:
        root = Path(tmp)
        profiles = root / "profiles"
        profiles.mkdir()
        helper = root / "incremental_groundtruth"
        c = subprocess.run(
            [
                "g++",
                str(repo / "baselines/DiskANN/tests/utils/incremental_groundtruth.cpp"),
                "-O2",
                "-std=c++17",
                "-o",
                str(helper),
            ],
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert c.returncode == 0, c.stderr
        for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
            dataset = "mixed_" + dtype
            rows = [(r * 3 + (32 if dtype == "uint8" else -100), 2, 3) for r in range(64)]
            queries = [rows[60], rows[40]]
            data, query, deep = (root / (dataset + suffix) for suffix in (".bin", ".query", ".gt"))
            data.write_bytes(struct.pack("<II192" + fmt, 64, 3, *(v for row in rows for v in row)))
            query.write_bytes(struct.pack("<II6" + fmt, 2, 3, *(v for row in queries for v in row)))
            gold = [
                sorted(
                    [(i, sum((a - b) ** 2 for a, b in zip(q, v))) for i, v in enumerate(rows)],
                    key=lambda p: (p[1], p[0]),
                )
                for q in queries
            ]
            deep.write_bytes(
                struct.pack("<II", 2, 64)
                + b"".join(struct.pack("<I", i) for row in gold for i, _ in row)
                + b"".join(struct.pack("<f", d) for row in gold for _, d in row)
            )
            pq = root / (dataset + "_pq")
            write_pivots(Path(str(pq) + "_pq_pivots.bin"))
            Path(str(pq) + "_pq_compressed.bin").write_bytes(struct.pack("<II", 64, 2) + bytes(128))
            profile = profiles / (dataset + ".sh")
            profile.write_text(
                "\n".join(
                    [
                        f"DATASET={dataset}",
                        f"DATA_TYPE={dtype}",
                        "METRIC=l2",
                        "DIM=3",
                        "NPTS_FULL=64",
                        "NPTS_BASE=48",
                        "UPDATE_POINTS=16",
                        "DISKANN_R=64",
                        "DISKANN_INDEX_0P9B=/missing/not_used",
                        "RECALL_AT=4",
                        f"DATA_BIN='{data}'",
                        f"QUERY_BIN='{query}'",
                        f"GT_BIN_KCACHED='{deep}'",
                    ]
                )
                + "\n"
            )
            env = os.environ.copy()
            for key in list(env):
                if key.startswith(("IVF_", "EVAL_", "UPDATE_")) or key in (
                    "INSERT_CAP",
                    "DELETE_CAP",
                    "INSERT_POINTS",
                    "DELETE_POINTS",
                    "QUERY_POINTS",
                    "QUERY_RATIO",
                    "NTHREADS",
                ):
                    del env[key]
            env.update(
                DATASET=dataset,
                BENCH_DATASETS_DIR=str(profiles),
                IVF_BUILD_DIR=str(args.build_dir),
                IVF_PQ_PREFIX=str(pq),
                EVAL_TEMPFILES=str(root / "work"),
                EVAL_CACHED=str(root / "cache"),
                IVF_NLIST="1",
                IVF_MAX_REPS="1",
                IVF_BUILD_THREADS="1",
                IVF_BUFFER_FRAMES="2",
                UPDATE_PERCENT="25",
                UPDATE_BATCH="8",
                QUERY_POINTS="128",
                IVF_NPROBE="1",
                IVF_RERANK_FACTOR="32",
                QUERY_THREADS="2",
                NTHREADS="16",
                IVF_SCHEDULER="weighted",
                INCREMENTAL_GT_BIN=str(helper),
            )
            prepare, execute = (
                repo / "benchmark/IVF/sift/update_prepare.sh",
                repo / "benchmark/IVF/sift/update.sh",
            )
            original = {p: digest(p) for p in (data, query, deep)}
            run(prepare, env)
            work = Path(env["EVAL_TEMPFILES"]) / f"{dataset}/IVF/update"
            assert json.loads((work / "mixed_prepared.json").read_text())["rounds"] == 2
            initial = {p: digest(p) for p in (work / "index").iterdir()}
            for scheduler in ("weighted", "fixed_8_8"):
                warmups = 3 if scheduler == "weighted" else 0
                env.update(IVF_SCHEDULER=scheduler, IVF_RUN_TAG=scheduler, IVF_WARMUP_QUERIES=str(warmups))
                run(execute, env)
                output = work / "runs" / scheduler
                summary = json.loads((output / "summary.json").read_text())
                assert summary["version"] == 2 and "round_records" not in summary
                assert summary["round_results"] == [f"rounds/round_{r}/result.json" for r in range(2)]
                assert not (output / "result.json").exists()
                assert (
                    not summary["production_qualified"]
                    and summary["concurrent_query_count"] == 128
                    and summary["final_recall"] == 100
                )
                for r in range(2):
                    dest = output / f"rounds/round_{r}"
                    record = json.loads((dest / "result.json").read_text())
                    assert not (dest / "native_result.json").exists()
                    assert not (dest / "extract.log").exists()
                    assert record["explicit_warmup_query_count"] == warmups
                    assert record["query_count"] == len(queries)
                    assert "no_reopen" in record["query_cache_process_policy"]
                    assert (
                        record["insert_count"] == record["delete_count"] == 8
                        and record["live_count"] == 48
                    )
                    assert (
                        record["recall"] == 100
                        and record["concurrent_query_count"] == 64
                        and record["foreground_workers"] == 16
                    )
                    assert (
                        record["concurrent_recall_policy"] == "not_scored_weakly_consistent_queries"
                    )
                    if scheduler == "fixed_8_8":
                        assert record["mutation_workers"] == record["concurrent_query_workers"] == 8
                        assert record["overlapping_query_count"] > 0, record
                    deleted = range(32, 32 + (r + 1) * 8)
                    live = [i for i in range(48 + (r + 1) * 8) if i not in deleted]
                    expected = [
                        sorted(
                            [(i, sum((a - b) ** 2 for a, b in zip(q, rows[i]))) for i in live],
                            key=lambda p: (p[1], p[0]),
                        )[:4]
                        for q in queries
                    ]
                    actual = list(
                        struct.iter_unpack("<If", (dest / "neighbors.bin").read_bytes()[8:])
                    )
                    assert actual == [p for row in expected for p in row], (actual, expected)
                    scored = subprocess.run(
                        [
                            "python3",
                            str(repo / "benchmark/SPFresh/utils.py"),
                            "tie-aware-recall",
                            "--truth",
                            str(work / f"groundtruth/rounds/round_{r}_gt4.bin"),
                            "--result",
                            str(dest / "neighbors.bin"),
                            "--recall-at",
                            "4",
                        ],
                        text=True,
                        capture_output=True,
                    )
                    assert scored.returncode == 0 and "1.000000" in scored.stdout, scored.stderr
                    assert (dest / "checkpoint" / (dataset + ".heap.state")).is_file()
                assert all(digest(p) == value for p, value in initial.items())
                run(execute, env, False)  # Cannot overwrite evidence.
            # Default shell path is update-driven, with no hidden ratio/count.
            driven_env = env.copy()
            del driven_env["IVF_SCHEDULER"]
            del driven_env["QUERY_POINTS"]
            driven_env.update(IVF_RUN_TAG="update_driven", NTHREADS="4")
            # Preparation must also admit the default mode without a quota.
            default_prepare_env = driven_env.copy()
            default_prepare_env.pop("IVF_RUN_TAG")
            default_prepare_env["EVAL_TEMPFILES"] = str(root / "default_prepare")
            run(prepare, default_prepare_env)
            default_work = Path(default_prepare_env["EVAL_TEMPFILES"]) / f"{dataset}/IVF/update"
            assert json.loads((default_work / "mixed_prepared.json").read_text())["rounds"] == 2
            assert (default_work / "prepared.json").is_file()
            run(execute, driven_env)
            output = work / "runs/update_driven"
            summary = json.loads((output / "summary.json").read_text())
            assert summary["final_recall"] == 100 and not summary["production_qualified"]
            for r in range(2):
                dest = output / f"rounds/round_{r}"
                record = json.loads((dest / "result.json").read_text())
                assert record["scheduler"] == "update_driven"
                assert record["insert_count"] == record["delete_count"] == 8
                assert record["recall"] == 100 and record["live_count"] == 48
                assert record["mutation_workers"] == record["concurrent_query_workers"] == 2
                assert "query_ratio" not in record and "scheduler_seed" not in record
                assert (
                    record["concurrent_query_count_policy"]
                    == "until_all_mutations_complete_join_admitted_queries"
                )
                assert record["concurrent_query_latency_sample_count"] == min(
                    record["concurrent_query_count"],
                    record["concurrent_query_latency_sample_limit"],
                )
                log = (dest / "run.log").read_text()
                assert "--query_ratio" not in log and "--query_count" not in log
                assert "--scheduler_seed" not in log
            assert summary["concurrent_query_count"] == sum(
                json.loads((output / path).read_text())["concurrent_query_count"]
                for path in summary["round_results"]
            )
            assert all(digest(p) == value for p, value in initial.items())
            for key, value in (
                ("QUERY_RATIO", "80"),
                ("QUERY_POINTS", "128"),
                ("UPDATE_QUERY_BATCH", "64"),
                ("IVF_SCHEDULER_SEED", "1"),
                ("NTHREADS", "1"),
            ):
                refused_env = driven_env.copy()
                refused_env.update({key: value, "IVF_RUN_TAG": "driven_bad_" + key})
                run(execute, refused_env, False)
                assert not (work / "runs" / refused_env["IVF_RUN_TAG"]).exists()
                if key != "NTHREADS":
                    refused_prepare = default_prepare_env.copy()
                    refused_prepare.update(
                        {key: value, "EVAL_TEMPFILES": str(root / ("refused_prepare_" + key))}
                    )
                    run(prepare, refused_prepare, False)
                    assert not Path(refused_prepare["EVAL_TEMPFILES"]).exists()
            env["IVF_RUN_TAG"] = "invalid_split"
            env["NTHREADS"] = "8"
            run(execute, env, False)
            assert not (work / "runs/invalid_split").exists()
            env["NTHREADS"] = "16"
            env["IVF_RUN_TAG"] = "changed_ids"
            ids = work / "ids/round_1_insert.bin"
            saved = ids.read_bytes()
            ids.write_bytes(saved[:-4] + struct.pack("<I", 0))
            run(execute, env, False)
            assert not (work / "runs/changed_ids").exists()
            ids.write_bytes(saved)
            env["IVF_RUN_TAG"] = "bad_budget"
            env["IVF_MEMORY_BUDGET_GB"] = "0.00001"
            run(execute, env, False)
            assert not (work / "runs/bad_budget/result.json").exists()
            del env["IVF_MEMORY_BUDGET_GB"]
            assert all(digest(p) == value for p, value in original.items())
            # Native mixed-specific invalid counts/allocations/flags, with no
            # checkpoint/success record. Keep the already prepared family intact.
            base = [
                str(args.build_dir / "tests/ivf_driver"),
                "--validate_inputs",
                "0",
                "--data_type",
                dtype,
                "--index_prefix",
                str(work / "index" / dataset),
                "--workload",
                "mixed",
                "--full_data_file",
                str(data),
                "--query_file",
                str(query),
                "--insert_ids_file",
                str(work / "ids/round_0_insert.bin"),
                "--delete_ids_file",
                str(work / "ids/round_0_delete.bin"),
                "--nprobe",
                "1",
                "--recall_at",
                "4",
                "--rerank_factor",
                "32",
            ]
            for n, extra in enumerate(
                (
                    ["--query_ratio", "100"],
                    ["--pool_threads", "0"],
                    ["--query_count", "0"],
                    ["--scheduler", "bogus"],
                    ["--scheduler", "fixed_8_8", "--pool_threads", "8"],
                    ["--query_count", str(1 << 63)],
                    ["--scheduler", "update_driven", "--query_ratio", "80"],
                    ["--scheduler", "update_driven", "--query_count", "1"],
                    ["--scheduler", "update_driven", "--scheduler_seed", "1"],
                    ["--scheduler", "update_driven", "--pool_threads", "1"],
                )
            ):
                private, checkpoint = (
                    root / f"{dtype}_refused_{n}.heap",
                    root / f"{dtype}_refused_{n}",
                )
                p = subprocess.run(
                    base
                    + ["--private_heap_path", str(private), "--checkpoint_prefix", str(checkpoint)]
                    + extra,
                    text=True,
                    capture_output=True,
                    timeout=90,
                )
                assert (
                    p.returncode != 0
                    and "ERROR:" in p.stderr
                    and not Path(str(checkpoint) + ".manifest").exists()
                ), (p.stdout, p.stderr)
                assert not any(line.startswith('{"baseline":') for line in p.stdout.splitlines())
    print(
        "All-dtype update-driven/weighted/fixed rounds, checkpoint/GT/reopen/script/oracle fixtures passed"
    )


if __name__ == "__main__":
    main()
