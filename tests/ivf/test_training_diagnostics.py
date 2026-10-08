#!/usr/bin/env python3
"""Actual bounded trainer comparison, artifacts, budgets and failure evidence."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def run(command, success=True):
    result = subprocess.run([str(v) for v in command], text=True, capture_output=True, timeout=120)
    assert (result.returncode == 0) == success, (result.stdout, result.stderr)
    if not success:
        assert "ERROR:" in result.stderr, result.stderr
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.scratch, prefix="training_") as folder:
        root = Path(folder)
        for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
            data = root / (dtype + ".bin")
            rows = [[(r % 3) * 30 + r // 3 + d for d in range(3)] for r in range(24)]
            tail = [[float('nan')] * 3] * 6 if dtype == "float" else [[120] * 3] * 6
            flat = [v for row in rows + tail for v in row]
            data.write_bytes(struct.pack("<II", 30, 3) + struct.pack("<" + fmt * len(flat), *flat))
            output = root / (dtype + " results")
            command = [sys.executable, repo / "benchmark/IVF/qualify.py", "train", "--data", data,
                "--trainer", args.trainer, "--output", output, "--dtype", dtype, "--dim", 3,
                "--full-count", 30, "--base-count", 24, "--nlist", 3, "--sample-rate", .5,
                "--iterations", 2, "--threads", 2, "--seed", 5, "--memory-budget-gb", 1,
                "--evaluation-rows", 100, "--exact-work-limit", 1000, "--assignment-work-limit", 1000,
                "--evaluation-work-limit", 1000, "--timeout", 30,
                "--cpus", min(os.sched_getaffinity(0)), "--max-sse-ratio", 1.000001]
            run(command)
            summary = json.loads((output / "summary.json").read_text())
            plan = json.loads((output / "plan.json").read_text())
            assert summary["completed"] and summary["gates_passed"] and not summary["production_qualified"]
            assert plan["affinity"]["explicitly_pinned"]
            assert summary["hnsw_over_exact_evaluation_sse_ratio"] <= 1.000001
            assert summary["artifacts"]["exact"][".sample_ids.bin"] == summary["artifacts"]["hnsw"][".sample_ids.bin"]
            assert struct.unpack_from("<II", (output / "exact.centroids.bin").read_bytes()) == (3, 3)
            sample = (output / "exact.sample_ids.bin").read_bytes()
            assert struct.unpack_from("<II", sample) == (12, 1)
            assert max(struct.unpack_from("<12I", sample, 8)) < 24
            exact, hnsw = summary["records"]["exact"], summary["records"]["hnsw"]
            assert exact["training_distance_evaluations"] == 108
            assert not hnsw["training_distance_evaluations_known"] and "training_distance_evaluations" not in hnsw
            assert hnsw["assignment_index_builds"] == 3 and exact["assignment_index_builds"] == 0
            assert exact["evaluation_distance_evaluations"] == 36 and exact["evaluation_rows"] == 12
            assert abs(exact["exact_evaluation_sse"] - exact["final_assignment_sample_sse"]) < 1e-9
            assert len(exact["objective_history"]) == 2
            assert exact["estimated_training_and_eval_bytes"] <= exact["memory_budget_bytes"]
            run(command, False)  # completed destination cannot be overwritten
            if dtype != "float":
                continue
            for flag, value, text in (("--exact-work-limit", 1, "exact training exceeds"),
                                      ("--evaluation-work-limit", 1, "evaluation exceeds"),
                                      ("--assignment-work-limit", 1, "assignment search limit")):
                bad = command.copy()
                bad[bad.index(flag) + 1] = value
                rejected = root / (flag[2:] + " rejected")
                bad[bad.index("--output") + 1] = rejected
                assert text in run(bad, False).stderr and not rejected.exists()
            # Native estimate failure leaves the explicit plan/log, no summary.
            small = command.copy()
            partial = root / "small budget"
            small[small.index("--output") + 1] = partial
            small[small.index("--memory-budget-gb") + 1] = .0001
            run(small, False)
            assert (partial / "plan.json").exists() and (partial / "exact.log").exists()
            assert not (partial / "summary.json").exists()
            # RSS is a post-run gate: complete measurements, explicit failure.
            rss = command.copy()
            gated = root / "rss gate"
            rss[rss.index("--output") + 1] = gated
            rss += ["--max-peak-rss-mb", .001]
            run(rss, False)
            assert json.loads((gated / "summary.json").read_text())["gates_passed"] is False
            # Bypass Python preflight to exercise the native work guard itself.
            native = [args.trainer, "--data_file", data, "--data_type", dtype, "--dim", 3,
                "--base_count", 24, "--nlist", 3, "--mode", "exact", "--output_prefix", root / "native_refusal",
                "--evaluation_rows", 12, "--evaluation_work_limit", 1, "--exact_work_limit", 1000,
                "--assignment_work_limit", 1000]
            assert "evaluation exceeds" in run(native, False).stderr
            assert not (root / "native_refusal.centroids.bin").exists()
            # Timeout kills the complete child process group; no success marker.
            timed = command.copy()
            timed[timed.index("--timeout") + 1] = .000001
            timed[timed.index("--output") + 1] = root / "timeout"
            assert "timed out" in run(timed, False).stderr
            assert not (root / "timeout/summary.json").exists()
    print("Bounded exact/HNSW training diagnostics passed for all dtypes (fixtures only)")


if __name__ == "__main__":
    main()
