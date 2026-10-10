#!/usr/bin/env python3
"""Explicit, small trained standalone-helper parity and IVF script integration.

Not a production dataset/training/performance qualification. No Python packages.
"""
import argparse
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile


def command(args, env, log):
    result = subprocess.run(
        [str(arg) for arg in args], env=env, text=True, capture_output=True, timeout=180
    )
    log.write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (args, log, result.stdout[-3000:], result.stderr[-3000:])
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pivot-helper", type=Path, required=True)
    parser.add_argument("--codes-helper", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("IVF_") or key.startswith("EVAL_"):
            del env[key]
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    # No seed/provenance assertion about the legacy random training sampler:
    # compare against the actual codebook produced in this invocation.
    with tempfile.TemporaryDirectory(prefix="trained_", dir=args.scratch) as tmp:
        root = Path(tmp)
        profiles = root / "profiles"
        profiles.mkdir()
        for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
            rng = random.Random(1847)
            rows, dim, base_count, chunks, k = 4096, 13, 768, 3, 5
            vectors = [
                [
                    (
                        rng.randrange(-400, 601) / 8
                        if dtype == "float"
                        else rng.randrange(-120, 121) if dtype == "int8" else rng.randrange(5, 256)
                    )
                    for _ in range(dim)
                ]
                for _ in range(rows)
            ]
            data, query, truth = (root / f"{dtype}_{name}.bin" for name in ("data", "query", "gt"))
            data.write_bytes(
                struct.pack("<II", rows, dim)
                + struct.pack(f"<{rows * dim}{fmt}", *(value for row in vectors for value in row))
            )
            queries = [vectors[i] for i in (0, 127, 513, 1001, 3501)]
            query.write_bytes(
                struct.pack("<II", len(queries), dim)
                + struct.pack(
                    f"<{len(queries) * dim}{fmt}", *(value for row in queries for value in row)
                )
            )
            oracle = [
                sorted(
                    [
                        (i, sum((a - b) ** 2 for a, b in zip(q, v)))
                        for i, v in enumerate(vectors[:base_count])
                    ],
                    key=lambda hit: (hit[1], hit[0]),
                )
                for q in queries
            ]
            truth.write_bytes(
                struct.pack("<II", len(queries), base_count)
                + b"".join(struct.pack("<I", i) for row in oracle for i, _ in row)
                + b"".join(struct.pack("<f", d) for row in oracle for _, d in row)
            )
            prefix = root / f"{dtype} standalone"
            command(
                [args.pivot_helper, dtype, data, prefix, chunks], env, root / f"{dtype}_train.log"
            )
            command(
                [args.codes_helper, dtype, data, prefix, chunks, 0],
                env,
                root / f"{dtype}_encode.log",
            )
            stored_codes = Path(str(prefix) + "_pq_compressed.bin").read_bytes()
            assert struct.unpack_from("<II", stored_codes) == (rows, chunks)
            assert all(
                len(set(stored_codes[8 + c :: chunks])) >= 32 for c in range(chunks)
            ), "degenerate trained codes"
            parity = command([args.parity, dtype, data, prefix], env, root / f"{dtype}_parity.log")
            print(
                next(
                    line
                    for line in parity.splitlines()
                    if line.startswith("Trained helper parity:")
                )
            )
            dataset = "trained_" + dtype
            (profiles / f"{dataset}.sh").write_text(
                "\n".join(
                    [
                        f"DATASET={dataset}",
                        f"DATA_TYPE={dtype}",
                        "METRIC=l2",
                        f"DIM={dim}",
                        f"NPTS_FULL={rows}",
                        f"NPTS_BASE={base_count}",
                        f"UPDATE_POINTS={rows - base_count}",
                        "DISKANN_R=64",
                        f"DATA_BIN='{data}'",
                        f"QUERY_BIN='{query}'",
                        f"GT_BIN_KCACHED='{truth}'",
                        "DISKANN_INDEX_0P9B=/no/graph/required",
                        f"RECALL_AT={k}",
                    ]
                )
                + "\n"
            )
            run_env = env | {
                "DATASET": dataset,
                "BENCH_DATASETS_DIR": str(profiles),
                "IVF_BUILD_DIR": str(args.build_dir),
                "EVAL_TEMPFILES": str(root / "eval_tempfiles"),
                "EVAL_CACHED": str(root / "eval_cached"),
                "EVAL_CACHE_NO_STORE": "1",
                "IVF_PQ_PREFIX": str(prefix),
                "IVF_GT_FILE": str(truth),
                "IVF_NLIST": "8",
                "IVF_MAX_REPS": "2",
                "IVF_BUILD_THREADS": "2",
                "IVF_NPROBE": "8",
                "IVF_RERANK_FACTOR": str(base_count / k),
                "QUERY_THREADS": "2",
                "IVF_BUFFER_FRAMES": "256",
            }
            command(
                ["bash", repo / "benchmark/IVF/sift/query_prepare.sh"],
                run_env,
                root / f"{dtype}_prepare.log",
            )
            command(
                ["bash", repo / "benchmark/IVF/sift/query.sh"], run_env, root / f"{dtype}_query.log"
            )
            work = root / "eval_tempfiles" / dataset / "IVF/query"
            record = json.loads((work / "result.json").read_text())
            assert (
                record["recall"] == 100 and record["reranked_vectors"] == len(queries) * base_count
            )
            entries = list(struct.iter_unpack("<If", (work / "neighbors.bin").read_bytes()[8:]))
            expected = [hit for row in oracle for hit in row[:k]]
            assert entries == expected, (dtype, entries, expected)
            assert not (
                root / "eval_cached"
            ).exists(), "no-store prepare unexpectedly populated cache"
    print(
        "Trained standalone helpers/native PQ/scripts exact-oracle qualification passed for all dtypes (fixtures only)"
    )


if __name__ == "__main__":
    main()
