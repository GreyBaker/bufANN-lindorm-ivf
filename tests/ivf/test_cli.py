#!/usr/bin/env python3
"""Small validation-only CLI fixtures; no datasets or third-party Python modules."""

import argparse
import json
from pathlib import Path
import struct
import subprocess


from ivf_test_support import write_ids, write_matrix, write_pivots, write_truth


def run(command, succeeds=True):
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if succeeds:
        assert result.returncode == 0, result.stderr
        records = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith('{"baseline":')
        ]
        assert len(records) == 1, result.stdout
        record = records[0]
        assert record["baseline"] == "IVF" and record["phase"] == "input_validation"
        assert not any(key in record for key in ("recall", "query_qps", "insert_ops_per_s"))
        return record
    else:
        assert result.returncode != 0, command
        assert "ERROR:" in result.stderr, result.stderr
        assert '{"baseline":' not in result.stdout, result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--contracts", required=True)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    logger = subprocess.run(
        [args.contracts, "--logger"], text=True, capture_output=True, timeout=30
    )
    assert logger.returncode == 0, logger.stderr
    assert logger.stdout == "x" * 4097 + "::end::", "logger overflow lost or duplicated characters"
    args.scratch.mkdir(parents=True, exist_ok=True)
    prefix = args.scratch / "fixture"
    write_pivots(Path(str(prefix) + "_pq_pivots.bin"))
    write_matrix(Path(str(prefix) + "_pq_compressed.bin"), [bytes(2)] * 2, "B")
    truth = args.scratch / "truth.bin"
    write_truth(truth, [[(0, 0.0), (1, 1.0)]])
    ids = args.scratch / "ids.bin"
    write_ids(ids, [1])

    for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
        data = args.scratch / f"{dtype}.bin"
        write_matrix(data, [(1, 2, 3), (4, 5, 6)], fmt)
        query = args.scratch / f"{dtype}_query.bin"
        write_matrix(query, [(1, 2, 3)], fmt)
        build = [
            args.build,
            "--validate_inputs",
            "1",
            "--data_type",
            dtype,
            "--dim",
            "3",
            "--data_file",
            str(data),
            "--index_prefix",
            str(prefix),
            "--pq_prefix",
            str(prefix),
            "--base_count",
            "2",
            "--nlist",
            "1",
        ]
        driver = [
            args.driver,
            "--validate_inputs",
            "1",
            "--data_type",
            dtype,
            "--dim",
            "3",
            "--index_prefix",
            str(prefix),
            "--nlist",
            "1",
            "--query_file",
            str(query),
            "--gt_file",
            str(truth),
            "--recall_at",
            "2",
        ]
        run(build)
        # Existing PQ artifacts describe the full source, not just its base
        # prefix. Validation and the native list builder must agree on framing.
        prefix_build = build.copy()
        prefix_build[prefix_build.index("--base_count") + 1] = "1"
        assert run(prefix_build)["rows"] == 1
        Path(str(prefix) + "_pq_compressed.bin").write_bytes(struct.pack("<II", 1, 2) + bytes(2))
        run(prefix_build, False)
        Path(str(prefix) + "_pq_compressed.bin").write_bytes(struct.pack("<II", 2, 2) + bytes(4))
        run(driver + ["--workload", "search_only"])
        for flag, value in (
            ("--index_format", "mutable"),
            ("--private_heap_path", "new.heap"),
            ("--checkpoint_prefix", "new"),
        ):
            run(driver + ["--workload", "search_only", flag, value], False)
        for workload in ("insert_only", "delete_only", "search_after_update", "mixed"):
            run(
                driver
                + [
                    "--workload",
                    workload,
                    "--full_data_file",
                    str(data),
                    "--insert_ids_file",
                    str(ids),
                    "--delete_ids_file",
                    str(ids),
                ]
            )
        run(build + ["--nlist", "2"], False)  # duplicate, not last-value-wins
        run(build + ["--sample_rate", "0.1", "--max_reps", "0"], False)
        run(build + ["--sample_rate", "nan"], False)
        run(build + ["--memory_budget_gb", "inf"], False)
        run(build + ["--build_threads", "-1"], False)
        run(build + ["--seed", "18446744073709551616"], False)
        run(build + ["--unknown", "1"], False)
        run(build + ["--metric", "cosine"], False)
        run(driver + ["--workload", "search_only", "--nprobe", "2"], False)
        run(driver + ["--workload", "search_only", "--rerank_factor", "0.5"], False)
        run(driver + ["--workload", "search_only", "--rerank_factor", "1e100"], False)
        run(driver + ["--workload", "search_after_update"], False)
        run(driver + ["--workload", "insert_only", "--insert_threads", "2"], False)
        run(driver + ["--workload", "insert_only", "--insert_input_policy", "staged"], False)
        run(driver + ["--workload", "search_only", "--warmup_queries", "1"], False)
        for flag, value in (
            ("--scheduler", "fixed_8_8"),
            ("--query_count", "8"),
            ("--pool_threads", "16"),
            ("--query_ratio", "80"),
            ("--scheduler_seed", "1"),
        ):
            run(driver + ["--workload", "mixed", flag, value], False)
        run(driver + ["--workload", "bogus"], False)
        run([args.build, "--data_file", str(data)], False)
        run([args.driver, "--index_prefix", str(prefix)], False)
    for tool in (args.build, args.driver):
        result = subprocess.run([tool, "--help"], text=True, capture_output=True, timeout=30)
        assert result.returncode == 0 and "validation only" in result.stdout
    assert not Path(str(prefix) + ".heap").exists(), "validation unexpectedly built an index"
    print("IVF CLI contracts passed for all dtypes and workload names")


if __name__ == "__main__":
    main()
