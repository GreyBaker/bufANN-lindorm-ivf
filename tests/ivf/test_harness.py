#!/usr/bin/env python3
"""Real query prepare/run/cache smoke and non-destructive failure fixtures."""
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
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, timeout=120)
    if success:
        assert result.returncode == 0, (result.stdout, result.stderr)
    else:
        assert result.returncode != 0 and "ERROR:" in result.stdout + result.stderr, (result.stdout, result.stderr)
    return result


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    prepare, query = (repo / "benchmark/IVF/sift" / name for name in ("query_prepare.sh", "query_run.sh"))
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="harness_", dir=args.scratch) as tmp:
        root = Path(tmp)
        profiles = root / "profiles"; profiles.mkdir()
        data, queries, truth = (root / name for name in ("full vectors.bin", "queries.bin", "base gt.bin"))
        deep_truth = root / "full deep gt.bin"
        data.write_bytes(struct.pack("<II12f", 4, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3))
        queries.write_bytes(struct.pack("<II6f", 2, 3, 1, 2, 3, 7, 8, 9))
        truth.write_bytes(struct.pack("<II6I6f", 2, 3, 0, 1, 2, 2, 1, 0, 0, 27, 108, 0, 27, 108))
        deep_truth.write_bytes(struct.pack("<II8I8f", 2, 4, 0, 3, 1, 2, 2, 1, 0, 3, 0, 0, 27, 108, 0, 27, 108, 108))
        pq_prefix = root / "standalone pq"
        write_pivots(Path(str(pq_prefix) + "_pq_pivots.bin"))
        Path(str(pq_prefix) + "_pq_compressed.bin").write_bytes(struct.pack("<II", 4, 2) + bytes(8))
        profile = profiles / "tiny.sh"
        profile.write_text('\n'.join(["DATASET=tiny", "DATA_TYPE=float", "METRIC=l2", "DIM=3", "NPTS_FULL=4",
            "NPTS_BASE=3", "UPDATE_POINTS=1", "DISKANN_R=64", f"DATA_BIN='{data}'", f"QUERY_BIN='{queries}'",
            f"GT_BIN_KCACHED='{deep_truth}'", "DISKANN_INDEX_0P9B=/deliberately/missing/graph", "RECALL_AT=1"]) + '\n')
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("IVF_") or key.startswith("EVAL_"):
                del env[key]
        env.update(DATASET="tiny", BENCH_DATASETS_DIR=str(profiles), IVF_BUILD_DIR=str(args.build_dir),
                   EVAL_TEMPFILES=str(root / "work first"), EVAL_CACHED=str(root / "cache"), IVF_NLIST="2",
                   IVF_PQ_PREFIX=str(pq_prefix), IVF_GT_FILE=str(truth), IVF_MAX_REPS="2", IVF_BUILD_THREADS="2",
                   IVF_NPROBE="2", IVF_RERANK_FACTOR="3", QUERY_THREADS="2")
        originals = {p: checksum(p) for p in (data, queries, truth, deep_truth, profile)}
        run(prepare, env)
        work = Path(env["EVAL_TEMPFILES"]) / "tiny/IVF/query"
        first = json.loads((work / "prepared.json").read_text())
        assert first["cache_hit"] is False
        cache = Path(env["EVAL_CACHED"]) / "tiny/IVF" / first["cache_key"]
        assert (cache / "cache.json").is_file()
        run(query, env)
        # Repeated native sweep preserves all trials; warmups are separate
        # processes and excluded from across-run (not pooled-query) aggregates.
        sweep_script = repo / "benchmark/IVF/sift/query_sweep.sh"
        sweep_args = ["--tag", "fixture", "--nprobes", "2", "--rerank-factors", "3", "--threads", "1,2",
                      "--repeats", "2", "--warmups", "1", "--memory-budget-gb", "1", "--buffer-frames", "64",
                      "--timeout", "30", "--min-recall", "100", "--cpus", str(min(os.sched_getaffinity(0)))]
        def sweep(extra=None, success=True):
            result = subprocess.run(["bash", str(sweep_script)] + (extra or sweep_args), env=env,
                                    text=True, capture_output=True, timeout=120)
            assert (result.returncode == 0) == success, (result.stdout, result.stderr)
            if not success:
                assert "ERROR:" in result.stdout + result.stderr
            return result
        sweep()
        sweep_root = work / "sweeps/fixture"
        summary = json.loads((sweep_root / "summary.json").read_text())
        assert summary["gates_passed"] and not summary["production_qualified"]
        assert len(summary["groups"]) == 2 and len(summary["trial_records"]) == 6
        assert sum(t["warmup"] for t in summary["trial_records"]) == 2
        assert all(g["measured_repeats"] == 2 and g["stats"]["recall"]["min"] == 100 for g in summary["groups"])
        assert all(Path(t["output_dir"]).joinpath("neighbors.bin").is_file() for t in summary["trial_records"])
        assert json.loads((sweep_root / "plan.json").read_text())["affinity"]["explicitly_pinned"]
        before_summary = checksum(sweep_root / "summary.json")
        sweep(success=False)
        assert checksum(sweep_root / "summary.json") == before_summary
        invalid = sweep_args.copy(); invalid[invalid.index("--tag") + 1] = "invalid"
        invalid[invalid.index("--nprobes") + 1] = "3"
        assert "nprobe exceeds" in sweep(invalid, False).stderr
        assert not (work / "sweeps/invalid").exists()
        duplicate = invalid.copy(); duplicate[duplicate.index("--nprobes") + 1] = "1,1"
        assert "duplicate sweep" in sweep(duplicate, False).stderr
        oversized = sweep_args.copy(); oversized[oversized.index("--tag") + 1] = "oversized"
        oversized[oversized.index("--repeats") + 1] = "4096"
        assert "4096-trial" in sweep(oversized, False).stderr
        assert not (work / "sweeps/oversized").exists()
        failed = sweep_args.copy(); failed[failed.index("--tag") + 1] = "failed"
        failed[failed.index("--memory-budget-gb") + 1] = "0.00001"
        sweep(failed, False)
        assert (work / "sweeps/failed/plan.json").exists() and not (work / "sweeps/failed/summary.json").exists()
        assert not (work / "runs/failed_c0_w0/result.json").exists()
        gated = sweep_args.copy(); gated[gated.index("--tag") + 1] = "gated"
        gated[gated.index("--threads") + 1] = "1"
        gated[gated.index("--repeats") + 1] = "1"; gated[gated.index("--warmups") + 1] = "0"
        gated += ["--max-peak-rss-mb", "0.001"]
        sweep(gated, False)
        assert json.loads((work / "sweeps/gated/summary.json").read_text())["gates_passed"] is False
        # A matching input or run-tag collision fails before launching any child.
        altered = sweep_args.copy(); altered[altered.index("--tag") + 1] = "changed_input"
        saved_truth = truth.read_bytes()
        truth.write_bytes(saved_truth[:-4] + struct.pack("<f", 109))
        assert "prepared input identity changed" in sweep(altered, False).stderr
        assert not (work / "sweeps/changed_input").exists()
        truth.write_bytes(saved_truth)
        collision = sweep_args.copy(); collision[collision.index("--tag") + 1] = "collision"
        (work / "runs/collision_c0_w0").mkdir()
        assert "run destination exists" in sweep(collision, False).stderr
        assert not (work / "sweeps/collision").exists()
        record = json.loads((work / "result.json").read_text())
        assert record == json.loads((work / "native_result.json").read_text())
        assert record["baseline"] == "IVF" and record["recall"] == 100 and record["query_count"] == 2
        assert list(struct.iter_unpack("<If", (work / "neighbors.bin").read_bytes()[8:])) == [(0, 0.0), (2, 0.0)]
        before = checksum(work / "result.json")
        run(query, env, False)
        run(prepare, env, False)
        assert checksum(work / "result.json") == before
        env["IVF_RUN_TAG"] = "repeat_2"
        run(query, env)
        assert json.loads((work / "runs/repeat_2/result.json").read_text())["recall"] == 100
        env["IVF_RUN_TAG"] = "failed_probe"; env["IVF_NPROBE"] = "0"
        run(query, env, False)
        assert not (work / "runs/failed_probe/result.json").exists()
        env["IVF_NPROBE"] = "2"; del env["IVF_RUN_TAG"]
        # Restore into a fresh work tree; every bundle file is a real copy.
        env["EVAL_TEMPFILES"] = str(root / "work second")
        run(prepare, env)
        restored = Path(env["EVAL_TEMPFILES"]) / "tiny/IVF/query"
        second = json.loads((restored / "prepared.json").read_text())
        assert second["cache_hit"] is True and second["cache_key"] == first["cache_key"]
        for path in (restored / "index").iterdir():
            assert not path.is_symlink() and path.read_bytes() == (cache / "index" / path.name).read_bytes()
        run(query, env)
        # Replacing the parent index directory with a symlink is also refused,
        # even when each component appears to be a regular file through it.
        (restored / "index").rename(restored / "owned_index")
        (restored / "index").symlink_to(cache / "index", target_is_directory=True)
        env["IVF_RUN_TAG"] = "symlink_index"
        assert "working index directory must not symlink" in run(query, env, False).stderr
        (restored / "index").unlink()
        (restored / "owned_index").rename(restored / "index")
        del env["IVF_RUN_TAG"]
        # Reserved-tail byte changes still change source/cache identity, without
        # becoming indexed rows. Original inputs are restored by this fixture.
        original_data = data.read_bytes()
        data.write_bytes(original_data[:-4] + struct.pack("<f", 4))
        env["EVAL_TEMPFILES"] = str(root / "changed source"); env["EVAL_CACHE_NO_STORE"] = "1"
        run(prepare, env)
        changed_source = json.loads((Path(env["EVAL_TEMPFILES"]) / "tiny/IVF/query/prepared.json").read_text())
        assert changed_source["cache_hit"] is False and changed_source["cache_key"] != first["cache_key"]
        assert not (Path(env["EVAL_CACHED"]) / "tiny/IVF" / changed_source["cache_key"]).exists()
        data.write_bytes(original_data)
        env["EVAL_TEMPFILES"] = str(root / "restore no store")
        run(prepare, env)
        assert json.loads((Path(env["EVAL_TEMPFILES"]) / "tiny/IVF/query/prepared.json").read_text())["cache_hit"] is True
        del env["EVAL_CACHE_NO_STORE"]
        # Default GT path uses the unchanged canonical incremental helper and
        # the exact same profile base prefix; no graph family is required.
        env["EVAL_TEMPFILES"] = str(root / "derived gt")
        del env["IVF_GT_FILE"]
        env["INCREMENTAL_GT_BIN"] = str(root / "incremental_groundtruth")
        compiled = subprocess.run(["g++", str(repo / "baselines/DiskANN/tests/utils/incremental_groundtruth.cpp"),
                                   "-O3", "-std=c++17", "-o", env["INCREMENTAL_GT_BIN"]], text=True, capture_output=True, timeout=120)
        assert compiled.returncode == 0, compiled.stderr
        run(prepare, env)
        run(query, env)
        assert json.loads((Path(env["EVAL_TEMPFILES"]) / "tiny/IVF/query/result.json").read_text())["recall"] == 100
        env["IVF_GT_FILE"] = str(truth)
        # A changed training setting must not silently reuse the old bundle.
        env["EVAL_TEMPFILES"] = str(root / "changed config"); env["IVF_ROUTER_SEED"] = "101"
        run(prepare, env)
        changed = json.loads((Path(env["EVAL_TEMPFILES"]) / "tiny/IVF/query/prepared.json").read_text())
        assert changed["cache_hit"] is False and changed["cache_key"] != first["cache_key"]
        del env["IVF_ROUTER_SEED"]
        # Cache corruption is an explicit refusal, not a destructive rebuild.
        heap = cache / "index/tiny.heap"
        with heap.open("r+b") as file:
            byte = file.read(1); file.seek(0); file.write(bytes([byte[0] ^ 1]))
        env["EVAL_TEMPFILES"] = str(root / "corrupt cache")
        failed = run(prepare, env, False)
        assert "cache integrity mismatch" in failed.stderr
        assert not (Path(env["EVAL_TEMPFILES"]) / "tiny/IVF/query/prepared.json").exists()
        # Input identity mutation is refused by an existing prepared run.
        env["EVAL_TEMPFILES"] = str(root / "work first"); env["IVF_RUN_TAG"] = "changed_gt"
        truth.write_bytes(truth.read_bytes()[:-4] + struct.pack("<f", 109))
        assert "GT changed since prepare" in run(query, env, False).stderr
        truth.write_bytes(struct.pack("<II6I6f", 2, 3, 0, 1, 2, 2, 1, 0, 0, 27, 108, 0, 27, 108))
        assert all(checksum(p) == value for p, value in originals.items())
    print("IVF prepare/run/cache and non-destructive failure fixtures passed")


if __name__ == "__main__":
    main()
