#!/usr/bin/env python3
"""Native serial/parallel list equivalence using synthetic vectors, not queries."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

from test_cli import write_pivots


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="list_parallel_", dir=args.scratch) as tmp:
        root = Path(tmp)
        base, rows, dim, chunks = 8209, 8237, 3, 2
        cases = []
        for dtype, code in (("float", "f"), ("int8", "b"), ("uint8", "B")):
            data = root / (dtype + ".bin")
            values = [((r % 8) * 12 + (r // 8 + d) % 7) for r in range(rows) for d in range(dim)]
            data.write_bytes(struct.pack("<II", rows, dim) + struct.pack("<" + str(len(values)) + code, *values))
            pq = root / (dtype + "-pq")
            write_pivots(Path(str(pq) + "_pq_pivots.bin"))
            Path(str(pq) + "_pq_compressed.bin").write_bytes(
                struct.pack("<II", rows, chunks) + bytes((r * 7 + d * 13) % 256 for r in range(rows) for d in range(chunks)))
            source_identity = digest(data)
            pq_identity = {suffix: digest(Path(str(pq) + suffix))
                           for suffix in ("_pq_pivots.bin", "_pq_compressed.bin")}
            for mode in ("hnswlib", "pca-refined"):
                references = None
                for threads in (1, 4):
                    prefix = root / f"{dtype}-{mode}-{threads}"
                    command = [args.build, "--validate_inputs", "0", "--data_type", dtype,
                               "--metric", "l2", "--dim", str(dim), "--base_count", str(base),
                               "--data_file", str(data), "--pq_prefix", str(pq), "--index_prefix", str(prefix),
                               "--nlist", "8", "--trainer_mode", "exact", "--init", "random_partition",
                               "--max_reps", "1", "--sample_rate", "1", "--seed", "31",
                               "--build_threads", str(threads), "--memory_budget_gb", "1",
                               "--optimized_modes", "1" if mode != "hnswlib" else "0", "--router_mode", mode]
                    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
                    assert result.returncode == 0, (command, result.stdout, result.stderr)
                    record, = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"baseline":')]
                    expected_workers = min(threads, (base + 4095) // 4096)
                    assert record["build_threads"] == threads
                    assert record["list_routing_workers"] == expected_workers
                    assert record["list_routing_block_rows"] == 4096
                    assert record["list_routing_buffer_rows"] == min(base, expected_workers * 4096)
                    assert f"[IVF lists] phase=routing_start workers={expected_workers}" in result.stderr
                    assert record["base_count"] == base and record["source_rows"] == rows
                    assert record["pq_chunks"] == chunks
                    components = {suffix: digest(Path(str(prefix) + suffix))
                                   for suffix in (".centroids.bin", ".hnsw", ".router", ".lists", ".heap", ".pq.bin")}
                    if mode != "hnswlib":
                        components.update({suffix: digest(Path(str(prefix) + suffix))
                                           for suffix in (".patched.hnsw", ".refined.hnsw", ".pca")})
                    if references is None:
                        references = components
                    else:
                        assert components == references, (dtype, mode, components, references)
                    assert digest(data) == source_identity
                    assert {suffix: digest(Path(str(pq) + suffix)) for suffix in pq_identity} == pq_identity
                cases.append({"dtype": dtype, "router": mode, "parallel_workers": 3,
                              "base_count": base, "components_byte_identical": True})
        print(json.dumps({"synthetic_only": True, "cases": cases, "no_query_evaluation": True}))


if __name__ == "__main__":
    main()
