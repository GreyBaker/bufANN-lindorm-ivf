#!/usr/bin/env python3
"""Bounded typed calibration/GT/radius/report/provenance fixtures, not model accuracy."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from test_cli import write_pivots


def matrix(path, fmt):
    raw = path.read_bytes()
    rows, dim = struct.unpack_from("<II", raw)
    values = struct.unpack_from("<" + fmt * (rows * dim), raw, 8)
    assert len(raw) == 8 + struct.calcsize(fmt) * rows * dim
    return [values[r * dim : (r + 1) * dim] for r in range(rows)]


def run(command, success=True, env=None):
    done = subprocess.run(
        [str(x) for x in command], env=env, text=True, capture_output=True, timeout=120
    )
    assert (done.returncode == 0) == success, (command, done.stdout, done.stderr)
    if not success:
        assert "ERROR:" in done.stderr
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "benchmark/IVF"))
    import harness

    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cost_model_", dir=args.scratch) as tmp:
        root = Path(tmp)
        for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
            vectors = [(float((r % 8) * 12 + (r // 8) % 4), float(r % 3), 2.0) for r in range(168)]
            if dtype != "float":
                vectors = [tuple(int(v) for v in row) for row in vectors]
            else:
                # The reserved insertion tail may contain nonfinite values;
                # calibration must never read it as a training/query row.
                vectors[160:] = [(math.nan, math.nan, math.nan)] * 8
            data, pq = root / (dtype + ".bin"), root / (dtype + "_pq")
            data.write_bytes(
                struct.pack("<II", 168, 3)
                + struct.pack("<" + fmt * 504, *(v for row in vectors for v in row))
            )
            write_pivots(Path(str(pq) + "_pq_pivots.bin"))
            Path(str(pq) + "_pq_compressed.bin").write_bytes(
                struct.pack("<II", 168, 2) + bytes(336)
            )
            output = root / (dtype + "_model")
            command = [
                sys.executable,
                repo / "benchmark/IVF/cost_model.py",
                "--tool",
                args.tool,
                "--output",
                output,
                "--data",
                data,
                "--pq-prefix",
                pq,
                "--data-type",
                dtype,
                "--dim",
                "3",
                "--base-count",
                "160",
                "--sample-count",
                "128",
                "--synthetic-queries",
                "4",
                "--nlist-grid",
                "1,2,4,5",
                "--topk",
                "3",
                "--target-recall",
                ".90",
                "--max-reps",
                "2",
                "--seed",
                "17",
                "--router-degree",
                "4",
                "--construction-breadth",
                "16",
                "--query-breadth",
                "16",
                "--training-breadth",
                "16",
                "--profile-codes",
                "64",
                "--profile-repeats",
                "2",
                "--routing-repeats",
                "2",
                "--work-limit",
                "100000",
                "--cpus",
                str(min(os.sched_getaffinity(0))),
            ]
            if dtype == "uint8":
                command += [
                    "--optimized-modes",
                    "1",
                    "--router-mode",
                    "pca-refined",
                    "--refine-ef",
                    "16",
                ]
            before = {
                p: harness.digest(p)
                for p in (
                    data,
                    Path(str(pq) + "_pq_pivots.bin"),
                    Path(str(pq) + "_pq_compressed.bin"),
                )
            }
            run(command)
            report = harness.read_cost_model_report(output / "report.json")
            for key, name in (("python_common", "common.py"), ("settings", "settings.py")):
                assert report["identities"][key] == harness.digest(repo / "benchmark/IVF" / name)
            assert report["profiled_codes"] == 64 * 2 * 4 and report["per_code_us"] > 0
            assert not report["production_qualified"] and not report["validation_completed"]
            assert report["affinity"]["explicitly_pinned"]
            sample_ids = [r[0] for r in matrix(output / "native/sample_ids.bin", "I")]
            query_ids = [r[0] for r in matrix(output / "native/query_ids.bin", "I")]
            assert len(set(sample_ids + query_ids)) == 132 and max(sample_ids + query_ids) < 160
            assert sample_ids == sorted(sample_ids) and query_ids == sorted(query_ids)
            sample, queries = matrix(output / "native/sample.bin", "f"), matrix(
                output / "native/queries.bin", "f"
            )
            assert sample == [vectors[id] for id in sample_ids] and queries == [
                vectors[id] for id in query_ids
            ]
            gt = matrix(output / "native/gt_slots.bin", "I")
            for q, row in zip(queries, gt):
                oracle = sorted(
                    range(128), key=lambda r: (sum((a - b) ** 2 for a, b in zip(q, sample[r])), r)
                )[:3]
                assert list(row) == oracle
            for candidate in report["candidates"]:
                n = candidate["nlist"]
                if not candidate["eligible"]:
                    assert (
                        n == 5 and candidate["drop_reason"] == "sample_count_per_cluster_below_32"
                    )
                    continue
                prefix = output / "native" / ("n" + str(n))
                labels = [r[0] for r in matrix(Path(str(prefix) + ".assignments.bin"), "I")]
                occupancy = [r[0] for r in matrix(Path(str(prefix) + ".occupancy.bin"), "I")]
                centroids = matrix(Path(str(prefix) + ".centroids.bin"), "f")
                assert occupancy == [labels.count(c) for c in range(n)]
                assert (
                    candidate["training_graph_builds"] == 1
                    and candidate["training_graph_refreshes"] == 2
                )
                for q, neighbours, qc in zip(queries, gt, candidate["queries"]):
                    d = [sum((a - b) ** 2 for a, b in zip(q, centroid)) for centroid in centroids]
                    threshold = sorted(d[labels[r]] for r in neighbours)[2]
                    chosen = [c for c in range(n) if d[c] <= threshold]
                    count = sum(occupancy[c] for c in chosen)
                    assert qc["threshold_squared"] == threshold and qc["nprobe"] == len(chosen)
                    assert qc["sample_codes"] == count and qc["scaled_codes"] == count * 160 / 128
                    assert qc["t2_us"] == qc["scaled_codes"] * report["per_code_us"]
            assert before == {p: harness.digest(p) for p in before}
            assert report["version"] == 2 and report["min_sample_per_centroid"] == 32
            assert (
                report["candidate_eligibility_policy"]
                == "declared_project_minimum_not_paper_requirement"
            )
            # Historical v1 keeps its implicit cutoff; new reports declare policy.
            legacy = copy.deepcopy(report)
            legacy["version"] = 1
            legacy.pop("min_sample_per_centroid")
            legacy.pop("candidate_eligibility_policy")
            legacy_path = root / (dtype + "_legacy_report.json")
            legacy_path.write_text(json.dumps(legacy))
            assert (
                harness.read_cost_model_report(legacy_path)["selected_nlist"]
                == report["selected_nlist"]
            )
            sparse = command.copy()
            sparse_root = root / (dtype + "_sparse")
            sparse[sparse.index("--output") + 1] = sparse_root
            sparse[sparse.index("--nlist-grid") + 1] = "5,6,7"
            sparse += [
                "--min-sample-per-centroid",
                "20",
                "--trainer-mode",
                "lv-plus",
                "--training-refine-breadth",
                "16",
            ]
            run(sparse)
            sparse_report = harness.read_cost_model_report(sparse_root / "report.json")
            assert sparse_report["min_sample_per_centroid"] == 20
            assert [c["eligible"] for c in sparse_report["candidates"]] == [True, True, False]
            assert all(c["below_legacy_32_adequacy"] for c in sparse_report["candidates"])
            assert (
                sparse_report["candidates"][-1]["drop_reason"]
                == "sample_count_per_cluster_below_declared_minimum"
            )
            assert (
                sparse_report["trainer_mode"] == "lv-plus"
                and sparse_report["training_refine_breadth"] == 16
            )
            if dtype == "float":
                structural = command.copy()
                structural_root = root / "structural_minimum"
                structural[structural.index("--output") + 1] = structural_root
                structural[structural.index("--sample-count") + 1] = "8"
                structural[structural.index("--nlist-grid") + 1] = "8,9"
                run(structural + ["--min-sample-per-centroid", "1"])
                boundary = harness.read_cost_model_report(structural_root / "report.json")
                assert boundary["selected_nlist"] == 8 and [
                    c["eligible"] for c in boundary["candidates"]
                ] == [True, False]
            for minimum in (0, -1):
                bad_policy = command.copy()
                refused = root / (dtype + "_policy_" + str(minimum))
                bad_policy[bad_policy.index("--output") + 1] = refused
                run(bad_policy + ["--min-sample-per-centroid", str(minimum)], False)
                assert not (refused / "native").exists()
            for mutation in ("minimum", "eligibility", "missing_policy"):
                invalid_policy = copy.deepcopy(sparse_report)
                if mutation == "minimum":
                    invalid_policy["min_sample_per_centroid"] = 32
                elif mutation == "eligibility":
                    invalid_policy["candidates"][-1]["eligible"] = True
                else:
                    invalid_policy.pop("candidate_eligibility_policy")
                invalid_path = root / (dtype + "_invalid_" + mutation + ".json")
                invalid_path.write_text(json.dumps(invalid_policy))
                try:
                    harness.read_cost_model_report(invalid_path)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Accepted inconsistent sample adequacy policy")
            frozen = harness.digest(output / "report.json")
            run(command, False)  # Existing destinations must never be overwritten.
            assert harness.digest(output / "report.json") == frozen
            bad = command.copy()
            bad[bad.index("--output") + 1] = root / (dtype + "_bad")
            bad[bad.index("--nlist-grid") + 1] = "1,1"
            run(bad, False)
            excluded = command.copy()
            excluded_root = root / (dtype + "_excluded")
            excluded[excluded.index("--output") + 1] = excluded_root
            excluded[excluded.index("--nlist-grid") + 1] = "5,6"
            run(excluded, False)
            drops = json.loads((excluded_root / "native/report.json").read_text())
            assert drops["selected_nlist"] is None and not drops["completed"]
            assert [c["nlist"] for c in drops["candidates"]] == [5, 6]
            assert all(
                not c["eligible"] and c["drop_reason"] == "sample_count_per_cluster_below_32"
                for c in drops["candidates"]
            )
            run([args.tool, "--query_file", "official_queries.bin"], False)
            run([args.tool, "--help"])
            env = {k: v for k, v in os.environ.items() if not k.startswith("IVF_")}
            env["IVF_COST_MODEL_REPORT"] = str(output / "report.json")
            aliases = {
                "init": "trainer_init",
                "sample_rate": "target_sample_rate",
                "nlist": "selected_nlist",
            }
            keys = (
                "max_reps",
                "seed",
                "build_threads",
                "trainer_mode",
                "init",
                "training_breadth",
                "memory_budget_gb",
                "optimized_modes",
                "router_degree",
                "construction_breadth",
                "query_breadth",
                "insert_breadth",
                "router_seed",
                "refine_ef",
                "pca_variance",
                "router_rerank_factor",
                "routing_work_limit",
            )
            for key in keys:
                env["IVF_" + key.upper()] = str(report[aliases.get(key, key)])
            env["IVF_ASSIGNMENT_ROUTER_MODE"] = report["router_mode"]
            with patch.dict(os.environ, env, clear=True):
                settings = harness.build_settings()
                assert settings["nlist"] == str(report["selected_nlist"])
                sources = [before[p] for p in before]
                profile = SimpleNamespace(dtype=dtype, dim=3, full_count=168, base_count=160)
                selection = harness.nlist_selection(profile, settings, sources)
                assert (
                    selection["source"] == "cost_model" and selection["report_identity"] == frozen
                )
                changed_sources = list(sources)
                changed_sources[0] = dict(sources[0], sha256="0" * 64)
                try:
                    harness.nlist_selection(profile, settings, changed_sources)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Accepted mismatched model source")
                mismatch = dict(settings, training_breadth="17")
                try:
                    harness.nlist_selection(profile, mismatch, sources)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Accepted mismatched training breadth")
                mismatch = dict(settings, training_rerank_factor="5")
                try:
                    harness.nlist_selection(profile, mismatch, sources)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Accepted mismatched training candidate policy")
                os.environ["IVF_NLIST"] = "999"
                try:
                    harness.build_settings()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Accepted conflicting manual nlist")
            if dtype == "float":
                # Real build, cache restore, and query sidecar; no native metrics
                # are edited to add provenance. Tail NaNs remain outside the base.
                env.update(
                    IVF_BUILD_DIR=str(args.build_dir),
                    IVF_BUILD_BIN=str(args.build_dir / "tests/ivf_build"),
                    IVF_DRIVER_BIN=str(args.build_dir / "tests/ivf_driver"),
                    IVF_NPROBE="1",
                    IVF_RERANK_FACTOR="4",
                )
                query, truth = root / "query.bin", root / "truth.bin"
                query.write_bytes(struct.pack("<II3f", 1, 3, *vectors[0]))
                nearest = sorted(
                    range(160),
                    key=lambda r: (sum((a - b) ** 2 for a, b in zip(vectors[0], vectors[r])), r),
                )[:3]
                distances = [
                    sum((a - b) ** 2 for a, b in zip(vectors[0], vectors[r])) for r in nearest
                ]
                truth.write_bytes(struct.pack("<II3I3f", 1, 3, *nearest, *distances))
                profiles = root / "shell_profiles"
                profiles.mkdir()
                (profiles / "fixture.sh").write_text(
                    "\n".join(
                        [
                            "DATASET=fixture",
                            "DATA_TYPE=float",
                            "METRIC=l2",
                            "DIM=3",
                            "NPTS_FULL=168",
                            "NPTS_BASE=160",
                            "UPDATE_POINTS=8",
                            "DISKANN_R=64",
                            "RECALL_AT=3",
                            f"DATA_BIN='{data}'",
                            f"QUERY_BIN='{query}'",
                            f"GT_BIN_KCACHED='{truth}'",  # Supplied base GT bypasses this cache.
                            "DISKANN_INDEX_0P9B=/unused",
                        ]
                    )
                    + "\n"
                )
                shell_env = dict(
                    env,
                    DATASET="fixture",
                    BENCH_DATASETS_DIR=str(profiles),
                    EVAL_TEMPFILES=str(root / "shell_work"),
                    EVAL_CACHED=str(root / "shell_cache"),
                    IVF_PQ_PREFIX=str(pq),
                    IVF_GT_FILE=str(truth),
                )
                shell_env.pop("IVF_NLIST", None)
                run(["bash", repo / "benchmark/IVF/sift/query_prepare.sh"], env=shell_env)
                shell_prepared = harness.read_json(
                    root / "shell_work/fixture/IVF/query/prepared.json"
                )
                assert shell_prepared["nlist_selection"] == selection
                mismatch_env = dict(
                    shell_env, IVF_NLIST="999", EVAL_TEMPFILES=str(root / "shell_conflict")
                )
                refused = run(
                    ["bash", repo / "benchmark/IVF/sift/query_prepare.sh"], False, env=mismatch_env
                )
                assert "IVF_NLIST disagrees with cost-model selection" in refused.stderr
                for repeat in range(2):
                    work = root / ("prepared" + str(repeat))
                    (work / "data").mkdir(parents=True)
                    shutil.copyfile(data, work / "data/full.bin")
                    common = [
                        "--dataset",
                        "fixture",
                        "--dtype",
                        dtype,
                        "--metric",
                        "l2",
                        "--dim",
                        "3",
                        "--full-count",
                        "168",
                        "--base-count",
                        "160",
                        "--work-dir",
                        work,
                        "--recall-at",
                        "3",
                    ]
                    run(
                        [sys.executable, repo / "benchmark/IVF/harness.py", "prepare"]
                        + common
                        + [
                            "--pq-prefix",
                            pq,
                            "--query",
                            query,
                            "--truth",
                            truth,
                            "--cache-root",
                            root / "cache",
                        ],
                        env=env,
                    )
                    prepared = harness.read_json(work / "prepared.json")
                    assert (
                        prepared["cache_hit"] == bool(repeat)
                        and prepared["nlist_selection"] == selection
                    )
                    cached = harness.read_json(
                        root / "cache/fixture/IVF" / prepared["cache_key"] / "cache.json"
                    )
                    assert cached["identity"]["nlist_selection"] == selection
                    destination = work / "runs/model"
                    run(
                        [sys.executable, repo / "benchmark/IVF/harness.py", "run"]
                        + common
                        + ["--output-dir", destination],
                        env=env,
                    )
                    assert harness.read_json(destination / "nlist_selection.json") == selection
                    native = harness.read_json(destination / "native_result.json")
                    assert (
                        native["phase"] == "query" and native["nlist"] == report["selected_nlist"]
                    )
                    assert "selection_source" not in native and "nlist_selection" not in native
                    # A copied, changed report cannot satisfy the prepared hash.
                    changed = root / ("changed" + str(repeat) + ".json")
                    changed.write_text((output / "report.json").read_text() + "\n")
                    prepared["nlist_selection"]["report_path"] = str(changed)
                    (work / "prepared.json").write_text(json.dumps(prepared))
                    run(
                        [sys.executable, repo / "benchmark/IVF/harness.py", "run"]
                        + common
                        + ["--output-dir", work / "runs/changed"],
                        False,
                        env=env,
                    )
                    assert not (work / "runs/changed").exists()
                assert harness.digest(output / "report.json") == frozen
            invalid = copy.deepcopy(report)
            invalid["selected_nlist"] = 5
            bad_report = output / "invalid.json"
            bad_report.write_text(json.dumps(invalid))
            try:
                harness.read_cost_model_report(bad_report)
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted a dropped/incorrect model selection")
            duplicate = output / "duplicate.json"
            duplicate.write_text('{"version":1,"version":1}')
            try:
                harness.read_cost_model_report(duplicate)
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted duplicate report fields")
    print(
        "Typed calibration, independent GT/radius/count oracle, frozen report and setting provenance fixtures passed"
    )


if __name__ == "__main__":
    main()
