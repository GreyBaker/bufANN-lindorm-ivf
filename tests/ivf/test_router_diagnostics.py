#!/usr/bin/env python3
"""Synthetic persisted-router diagnostics, all dtypes/modes; no production inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from ivf_test_support import write_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", required=True)
    p.add_argument("--diagnostic", required=True)
    p.add_argument("--scratch", type=Path, required=True)
    args = p.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="router_diagnostics_", dir=args.scratch) as tmp:
        root = Path(tmp)
        fixtures = root / "fixtures"
        result = subprocess.run(
            [args.fixtures, str(fixtures)], text=True, capture_output=True, timeout=180
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        # The graph family is shared across the ablation: no centroid rebuild,
        # PCA refit or list/assignment change is allowed by this diagnostic.
        prefix = fixtures / "router_0"
        original = {f: f.read_bytes() for f in fixtures.glob("router_0.*")}
        for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
            query = root / (dtype + ".bin")
            offset = 200 if dtype == "uint8" else -20
            values = [offset + (i % 11) for i in range(52)]
            write_matrix(query, [values[r * 13 : (r + 1) * 13] for r in range(4)], fmt)
            command = [
                args.diagnostic,
                "--index_prefix",
                str(prefix),
                "--query_file",
                str(query),
                "--data_type",
                dtype,
                "--nprobe",
                "10",
                "--query_threads",
                "2",
                "--evaluation_work_limit",
                "3072",
                "--router_breadth",
                "128",
            ]
            for mode in ("exact", "hnswlib", "patched-full", "refined", "pca", "pca-refined"):
                path = root / (dtype + "_" + mode + ".json")
                cmd = command + ["--router_mode", mode, "--result_file", str(path)]
                completed = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
                assert completed.returncode == 0, (completed.stdout, completed.stderr)
                record = json.loads(path.read_text())
                assert record == json.loads(completed.stdout) and record["router_mode"] == mode
                assert record["query_rows"] == 4 and 0 <= record["centroid_recall_fraction"] <= 1
                assert not record["production_qualified"] and record["router_qps"] > 0
                assert 0 <= record["assignment_agreement_fraction"] <= 1
                assert record["distance_counted_queries"] == (0 if mode == "hnswlib" else 4)
                if mode == "exact":
                    assert (
                        record["centroid_recall_fraction"] == 1
                        and record["distance_evaluations"] == 3072
                    )
                repeated = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
                assert repeated.returncode != 0 and "exists" in repeated.stderr
            for extra in (
                ["--memory_budget_gb", "0.001"],
                ["--query_rows", "0"],
                ["--router_mode", "unknown"],
                ["--router_rerank_factor", "0.5"],
                ["--router_rerank_factor", "nan"],
            ):
                refused = subprocess.run(
                    command + extra, text=True, capture_output=True, timeout=90
                )
                assert refused.returncode != 0 and "ERROR:" in refused.stderr, (
                    refused.stdout,
                    refused.stderr,
                )
            bad = command.copy()
            bad[bad.index("--evaluation_work_limit") + 1] = "3071"
            assert subprocess.run(bad, capture_output=True, timeout=90).returncode != 0
        sweep = Path(__file__).resolve().parents[2] / "benchmark/IVF/router_sweep.py"
        output = root / "sweep"
        command = [
            sys.executable,
            "-O",
            str(sweep),
            "--diagnostic",
            args.diagnostic,
            "--index-prefix",
            str(prefix),
            "--query-file",
            str(query),
            "--data-type",
            "uint8",
            "--output",
            str(output),
            "--scope",
            "synthetic_fixture",
            "--nprobes",
            "1",
            "--breadths",
            "32",
            "--query-rows",
            "4",
            "--repeats",
            "2",
            "--evaluation-work-limit",
            "3072",
            "--timeout-s",
            "30",
        ]
        result = subprocess.run(command, text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, (result.stdout, result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["completed"] and len(summary["records"]) == 12
        assert len(list(output.glob("*.log"))) == 12
        frozen = (output / "summary.json").read_bytes()
        assert subprocess.run(command, capture_output=True, timeout=90).returncode != 0
        assert (output / "summary.json").read_bytes() == frozen
        for key, value in (
            ("--nprobes", "1,1"),
            ("--breadths", "0"),
            ("--timeout-s", "nan"),
            ("--query-rows", "0"),
            ("--recall-targets", "1.1"),
        ):
            bad = command.copy()
            bad[bad.index("--output") + 1] = str(root / (key[2:] + "_invalid"))
            if key in bad:
                bad[bad.index(key) + 1] = value
            else:
                bad += [key, value]
            assert subprocess.run(bad, capture_output=True, timeout=30).returncode != 0
            assert not Path(bad[bad.index("--output") + 1]).exists()
        failed = command.copy()
        failed[failed.index("--output") + 1] = str(root / "failed")
        failed[failed.index("--evaluation-work-limit") + 1] = "1"
        assert subprocess.run(failed, capture_output=True, timeout=30).returncode != 0
        assert (root / "failed/plan.json").exists() and (root / "failed/failure.txt").exists()
        assert list((root / "failed").glob("*.log")) and not (root / "failed/summary.json").exists()
        timeout = command.copy()
        timeout[timeout.index("--output") + 1] = str(root / "timeout")
        timeout[timeout.index("--timeout-s") + 1] = "0.000001"
        assert subprocess.run(timeout, capture_output=True, timeout=30).returncode != 0
        assert "timed out" in (root / "timeout/failure.txt").read_text()
        assert not (root / "timeout/summary.json").exists()
        assert all(path.read_bytes() == raw for path, raw in original.items())
    print(
        "Persisted native-router diagnostic modes/typed inputs/identity/work/output fixtures passed"
    )


if __name__ == "__main__":
    main()
