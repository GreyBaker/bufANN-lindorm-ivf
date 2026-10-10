#!/usr/bin/env python3
"""Typed trainer selection, immutable/mutable origin identity and cache plumbing."""
import argparse
import hashlib
import os
from pathlib import Path
import struct
import sys
import tempfile
from test_cli import write_pivots
from test_cli_execution import run


def reject_historical_producers(search, path, original):
    """Reject retired labels in every framing; never relabel test-owned inputs."""
    assert original.split()[3] == "hnswlib-0.8.0-stock1"
    try:
        for backend in ("hnswlib-0.8.0-prefetch1", "hnswlib-0.8.0-nomv", "unknown"):
            rejected = original.replace("hnswlib-0.8.0-stock1", backend)
            path.write_text(rejected)
            result = run(search)
            assert "Unsupported IVF index producer" in result.stderr
            assert path.read_text() == rejected
    finally:
        path.write_text(original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark/IVF"))
    import harness

    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s09_", dir=args.scratch) as tmp:
        root = Path(tmp)
        for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
            pq = root / (dtype + "_pq")
            write_pivots(Path(str(pq) + "_pq_pivots.bin"))
            Path(str(pq) + "_pq_compressed.bin").write_bytes(struct.pack("<II", 33, 2) + bytes(66))
            values = [
                ((r % 4) * 20 + (r // 4) % 5, d, d) for r, d in ((r, r % 3) for r in range(33))
            ]
            data = root / (dtype + ".bin")
            data.write_bytes(
                struct.pack("<II", 33, 3)
                + struct.pack("<" + fmt * 99, *(v for row in values for v in row))
            )
            query = root / (dtype + "_query.bin")
            query.write_bytes(
                struct.pack("<II", 2, 3) + struct.pack("<" + fmt * 6, *values[0], *values[10])
            )
            base = [
                args.build,
                "--validate_inputs",
                "0",
                "--data_type",
                dtype,
                "--dim",
                "3",
                "--base_count",
                "32",
                "--nlist",
                "4",
                "--max_reps",
                "2",
                "--data_file",
                str(data),
                "--pq_prefix",
                str(pq),
                "--buffer_frames",
                "4",
                "--build_threads",
                "2",
                "--insert_breadth",
                "7",
            ]
            for mode in (
                "exact",
                "ann-rebuild",
                "ann-reuse",
                "ann-reuse-seeded",
                "ann-reuse-count",
                "ann-reuse-seeded-count",
                "lv",
                "lv-plus",
            ):
                prefix = root / (dtype + "_" + mode)
                native = mode in ("ann-reuse-seeded", "lv-plus")
                seeded = mode in ("ann-reuse-seeded", "ann-reuse-seeded-count", "lv-plus")
                build = base + [
                    "--index_prefix",
                    str(prefix),
                    "--trainer_mode",
                    mode,
                    "--init",
                    "random_partition",
                    "--training_breadth",
                    "16",
                ]
                if mode == "lv-plus":
                    build += ["--training_rerank_factor", "2.25", "--training_refine_breadth", "16"]
                if native:
                    build += ["--optimized_modes", "1", "--router_mode", "pca-refined"]
                record = run(build, "build")
                assert (
                    record["trainer_mode"] == mode and record["trainer_init"] == "random_partition"
                )
                assert record["training_breadth"] == 16 and record["insert_breadth"] == 7
                reuse = mode != "exact" and mode != "ann-rebuild"
                assert record["training_graph_builds"] == (
                    0 if mode == "exact" else 1 if reuse else 3
                )
                assert record["training_graph_refreshes"] == (2 if reuse else 0)
                assert record["training_seeded_searches"] == (64 if seeded else 0)
                assert record["training_seed_fallbacks"] == (32 if seeded else 0)
                assert (
                    len(record["training_passes"]) == 3
                    and record["training_final_assigned_sse"] >= 0
                )
                manifest = Path(str(prefix) + ".manifest").read_text()
                version = (
                    4
                    if mode in ("ann-reuse-count", "ann-reuse-seeded-count", "lv", "lv-plus")
                    else 3
                )
                assert (
                    manifest.startswith(f"IVFINDEX{version} ")
                    and f"trainer {mode} random_partition 16\n" in manifest
                )
                if version == 4:
                    assert (
                        "training_search 2.25 16 "
                        if mode == "lv-plus"
                        else "training_search 3 128 "
                    ) in manifest
                    assert record["training_returned_candidates"] == 96 * (
                        3 if mode in ("lv", "lv-plus") else 1
                    )
                family = harness.bundle_suffixes(prefix)
                assert (".pca" in family) == native
                hashes = {
                    suffix: hashlib.sha256(Path(str(prefix) + suffix).read_bytes()).hexdigest()
                    for suffix in family
                }
                search = [
                    args.driver,
                    "--validate_inputs",
                    "0",
                    "--data_type",
                    dtype,
                    "--index_prefix",
                    str(prefix),
                    "--workload",
                    "search_only",
                    "--query_file",
                    str(query),
                    "--nprobe",
                    "4",
                    "--recall_at",
                    "1",
                    "--rerank_factor",
                    "64",
                    "--buffer_frames",
                    "4",
                ]
                observed = run(search, "query")
                reject_historical_producers(search, Path(str(prefix) + ".manifest"), manifest)
                for key in (
                    "trainer_mode",
                    "trainer_init",
                    "training_breadth",
                    "trainer_graph_reuse",
                    "trainer_previous_assignment_seed",
                    "training_rerank_factor",
                    "training_refine_breadth",
                    "training_search_policy",
                ):
                    assert observed[key] == record[key]
                if native:
                    ids = root / (dtype + ".ids")
                    ids.write_bytes(struct.pack("<II", 1, 32))
                    saved = root / (dtype + "_" + mode + "_checkpoint")
                    update = run(
                        [
                            args.driver,
                            "--validate_inputs",
                            "0",
                            "--data_type",
                            dtype,
                            "--index_prefix",
                            str(prefix),
                            "--workload",
                            "insert_only",
                            "--full_data_file",
                            str(data),
                            "--insert_ids_file",
                            str(ids),
                            "--private_heap_path",
                            str(root / (dtype + "_" + mode + "_private.heap")),
                            "--checkpoint_prefix",
                            str(saved),
                            "--buffer_frames",
                            "4",
                        ],
                        "insert",
                    )
                    assert update["trainer_mode"] == mode
                    assert Path(str(saved) + ".origin").read_text() == manifest
                    reopened = search.copy()
                    reopened[reopened.index("--index_prefix") + 1] = str(saved)
                    metrics = run(
                        reopened
                        + [
                            "--index_format",
                            "mutable",
                            "--private_heap_path",
                            str(root / (dtype + "_" + mode + "_reopened.heap")),
                        ],
                        "query",
                    )
                    assert metrics["trainer_mode"] == mode and metrics["training_breadth"] == 16
                assert hashes == {
                    s: hashlib.sha256(Path(str(prefix) + s).read_bytes()).hexdigest()
                    for s in family
                }
                # Malformed trainer metadata must be rejected, not silently defaulted.
                bad = manifest.replace(
                    f"trainer {mode} random_partition 16", "trainer unknown random_partition 16"
                )
                Path(str(prefix) + ".manifest").write_text(bad)
                run(search)
                Path(str(prefix) + ".manifest").write_text(manifest)
                if version == 4:
                    malformed = manifest.replace("IVFINDEX4 ", "IVFINDEX3 ")
                    Path(str(prefix) + ".manifest").write_text(malformed)
                    run(search)
                    Path(str(prefix) + ".manifest").write_text(manifest)
            for index, extra in enumerate(
                (
                    ["--trainer_mode", "unknown"],
                    ["--init", "unknown"],
                    ["--training_breadth", "0"],
                    ["--training_rerank_factor", "4"],
                    ["--trainer_mode", "lv", "--init", "sample"],
                    ["--trainer_mode", "lv", "--training_refine_breadth", "0"],
                )
            ):
                refused = root / (dtype + "_refused_" + str(index))
                run(base + ["--index_prefix", str(refused)] + extra)
                assert not Path(str(refused) + ".manifest").exists()
            # Older framing remains supported with rebuilt/sample semantics and
            # the sole stock1 producer. Checkpoints preserve the framing version.
            for version in (1, 2):
                prefix = root / (dtype + "_legacy" + str(version))
                legacy = base + [
                    "--index_prefix",
                    str(prefix),
                    "--trainer_mode",
                    "ann-rebuild",
                    "--init",
                    "sample",
                    "--training_breadth",
                    "7",
                    "--optimized_modes",
                    str(version - 1),
                ]
                run(legacy, "build")
                manifest = Path(str(prefix) + ".manifest")
                text = manifest.read_text().replace("IVFINDEX3 ", f"IVFINDEX{version} ")
                text = (
                    "\n".join(
                        line
                        for line in text.splitlines()
                        if not line.startswith(("trainer ", "optimized_modes "))
                    )
                    + "\n"
                )
                manifest.write_text(text)
                assert (".pca" in harness.bundle_suffixes(prefix)) == (version == 2)
                search = [
                    args.driver,
                    "--validate_inputs",
                    "0",
                    "--data_type",
                    dtype,
                    "--index_prefix",
                    str(prefix),
                    "--workload",
                    "search_only",
                    "--query_file",
                    str(query),
                    "--nprobe",
                    "4",
                    "--recall_at",
                    "1",
                    "--rerank_factor",
                    "64",
                    "--buffer_frames",
                    "4",
                ]
                metrics = run(search, "query")
                reject_historical_producers(search, manifest, text)
                assert (
                    metrics["trainer_mode"] == "ann-rebuild"
                    and metrics["trainer_init"] == "sample"
                    and metrics["training_breadth"] == 7
                )
                ids = root / (dtype + "_legacy" + str(version) + ".ids")
                ids.write_bytes(struct.pack("<II", 1, 32))
                saved = root / (dtype + "_legacy_saved" + str(version))
                run(
                    [
                        args.driver,
                        "--validate_inputs",
                        "0",
                        "--data_type",
                        dtype,
                        "--index_prefix",
                        str(prefix),
                        "--workload",
                        "insert_only",
                        "--full_data_file",
                        str(data),
                        "--insert_ids_file",
                        str(ids),
                        "--private_heap_path",
                        str(root / (dtype + "_legacy_private" + str(version) + ".heap")),
                        "--checkpoint_prefix",
                        str(saved),
                        "--buffer_frames",
                        "4",
                    ],
                    "insert",
                )
                assert Path(str(saved) + ".origin").read_text() == text
                search[search.index("--index_prefix") + 1] = str(saved)
                reopened = run(
                    search
                    + [
                        "--index_format",
                        "mutable",
                        "--private_heap_path",
                        str(root / (dtype + "_legacy_reopen" + str(version) + ".heap")),
                    ],
                    "query",
                )
                assert (
                    reopened["trainer_mode"] == "ann-rebuild" and reopened["training_breadth"] == 7
                )
        os.environ["IVF_NLIST"] = "4"
        settings = harness.build_settings()
        assert (
            settings["trainer_mode"] == "ann-reuse-seeded"
            and settings["init"] == "random_partition"
        )
        for key, value in (
            ("IVF_TRAINER_MODE", "ann-rebuild"),
            ("IVF_INIT", "sample"),
            ("IVF_TRAINING_BREADTH", "32"),
        ):
            os.environ[key] = value
            assert harness.build_settings() != settings
            del os.environ[key]
        os.environ["IVF_ROUTER_MODE"] = "exact"
        assert harness.build_settings() == settings
    print("S09 typed native trainer/manifest/reopen/mutable-origin/cache settings passed")


if __name__ == "__main__":
    main()
