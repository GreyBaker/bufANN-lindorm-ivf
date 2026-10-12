#!/usr/bin/env python3
"""Recommend a sufficient shared GT depth for declared states, not ANN tuning.

Uses the existing exact cache to certify covered rows. Only underflow/boundary-
tie queries need two chunked exact source scans. Supports uint8 squared L2 in
1..128 dimensions; requires NumPy only when scans are needed. Does not generate
GT or change any input, index, wrapper, search parameter, or timed QPS path.
"""
from bisect import bisect_left
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import check_gt_coverage as coverage


class Snapshot:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        try:
            self.resolved = self.path.resolve(strict=True)
            self.handle_signature = coverage.signature(self.file)
            self.path_signature = coverage.stat_signature(self.path.stat())
            coverage.require(self.handle_signature[:3] == self.path_signature[:3], "input changed during open")
        except BaseException:
            self.file.close()
            raise

    def check(self):
        coverage.require(coverage.signature(self.file) == self.handle_signature
                         and coverage.stat_signature(self.path.stat()) == self.path_signature
                         and self.path.resolve(strict=True) == self.resolved,
                         f"input changed during recommendation: {self.path}")

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def vector_shape(snapshot):
    snapshot.file.seek(0)
    header = snapshot.file.read(8)
    coverage.require(len(header) == 8, "truncated vector header")
    rows, dim = struct.unpack("<II", header)
    coverage.require(rows > 0 and 0 < dim <= 128, "require uint8 vectors in 1..128 dimensions")
    coverage.require(snapshot.handle_signature[2] == 8 + rows * dim,
                     "uint8 vector framing/extent mismatch")
    return rows, dim


def local_best(np, distances, first, k):
    take = min(k, len(distances))
    if not take:
        return []
    cutoff = np.partition(distances, take - 1)[take - 1]
    below = np.flatnonzero(distances < cutoff)
    tied = np.flatnonzero(distances == cutoff)[:take - len(below)]
    indices = np.concatenate((below, tied))
    return [(int(distances[i]), first + int(i)) for i in indices]


def update_best(np, best, distances, first, k):
    return sorted(best + local_best(np, distances, first, k))[:k]


def live_segments(state, first, rows):
    end = max(0, min(rows, state.prefix - first))
    left = max(0, min(end, state.delete_begin - first))
    right = max(0, min(end, state.delete_end - first))
    return ((0, left), (right, end))


def squared_distances(np, vectors, query):
    # uint8 differences, products, and positive partial sums are integral and
    # fit float32 exactly for <=128D. No GEMM cancellation, PQ or ANN shortcuts.
    diff = vectors.astype(np.float32) - query
    return np.einsum("ij,ij->i", diff, diff, dtype=np.float32)


def scan_source(np, source, rows, dim, chunk_rows, callback, phase, progress_rows):
    source.file.seek(0)
    digest = hashlib.sha256()
    header = source.file.read(8)
    coverage.require(header == struct.pack("<II", rows, dim), "source header changed")
    digest.update(header)
    start = time.monotonic()
    next_progress = progress_rows
    for first in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - first)
        raw = source.file.read(count * dim)
        coverage.require(len(raw) == count * dim, "source truncated during scan")
        digest.update(raw)
        vectors = np.frombuffer(raw, dtype=np.uint8).reshape(count, dim)
        callback(vectors, first)
        done = first + count
        if done >= next_progress or done == rows:
            print(f"[GT depth] phase={phase} rows={done}/{rows} elapsed_s={time.monotonic() - start:.3f}",
                  file=sys.stderr, flush=True)
            next_progress = (done // progress_rows + 1) * progress_rows
    coverage.require(not source.file.read(1), "source grew during scan")
    source.check()
    return digest.hexdigest()


def recommend(args):
    states = coverage.planned_states(args)
    with args.cached_gt.open("rb") as file:
        header = file.read(8)
    coverage.require(len(header) == 8, "truncated GT header")
    query_count, cached_width = struct.unpack("<II", header)
    planning_workspace = (16 * 1024 * 1024 + cached_width * 128
                          + query_count * len(states) * 128)
    coverage.require(planning_workspace <= args.memory_budget_mb * 1024 * 1024,
                     "preflight/task scratch estimate exceeds --memory-budget-mb")
    cached = coverage.check_cache(args.cached_gt, args.full_count, states, args.k_out,
                                  args.expected_queries, args.worst_queries, collect_scan_queries=True)
    # Keep the existing width as a floor: shrinking is a different optimization.
    coverage.require(cached["cached_k"] <= args.full_count, "cached depth exceeds source population")
    tasks = {}
    for state, result in zip(states, cached["states"]):
        for reason, key in (("underflow", "underflow_query_ids"), ("boundary_tie", "boundary_tie_query_ids")):
            for query in result[key]:
                tasks.setdefault(query, {})[state.name] = (state, reason)
    print(f"[GT depth] cached_K={cached['cached_k']} scan_queries={len(tasks)} "
          f"query_state_cases={sum(len(v) for v in tasks.values())}", file=sys.stderr, flush=True)
    coverage.require(len(tasks) <= args.max_scan_queries,
                     f"{len(tasks)} queries need scans, above --max-scan-queries={args.max_scan_queries}; "
                     "raise the explicit budget or use direct per-state GT; no sampled recommendation is issued")
    report = {"version": 1, "cached_coverage": cached, "scan_queries": sorted(tasks),
              "scanned_query_state_cases": sum(len(v) for v in tasks.values()),
              "distance_id_sufficient_k": cached["cached_k"], "tie_safe_sufficient_k": cached["cached_k"],
              "cases": [], "source_sha256": None, "source_distance_passes": 0,
              "assumption": "unscanned rows rely on exact cached GT for the intended source/query order",
              "native_timing_changed": False, "gt_generated": False}
    with Snapshot(args.base_file) as source, Snapshot(args.query_file) as queries, Snapshot(args.cached_gt) as gt:
        rows, dim = vector_shape(source)
        query_rows, query_dim = vector_shape(queries)
        coverage.require(rows == args.full_count, "full source row count differs from plan")
        coverage.require(query_rows == cached["queries"] and query_dim == dim,
                         "query rows/dimension differ from cache/source")
        coverage.require(coverage.sha256(gt.file) == cached["cached_gt_sha256"], "cache changed after preflight")
        query_sha_before = coverage.sha256(queries.file)
        report["source"] = {"path": str(source.resolved), "rows": rows, "dim": dim,
                            "bytes": source.handle_signature[2], "data_type": "uint8", "metric": "squared_l2"}
        report["query"] = {"path": str(queries.resolved), "rows": query_rows, "sha256": query_sha_before}
        if tasks:
            try:
                import numpy as np
            except ImportError as error:
                raise ValueError("NumPy is required for exact scans (use an environment with numpy installed)") from error
            workspace = (planning_workspace + args.chunk_rows * (5 * dim + 64)
                         + len(tasks) * cached["cached_k"] * 128
                         + sum(len(v) for v in tasks.values()) * args.k_out * 128
                         + 32 * 1024 * 1024)
            coverage.require(workspace <= args.memory_budget_mb * 1024 * 1024,
                             "scan scratch estimate exceeds --memory-budget-mb; reduce --chunk-rows")
            report["estimated_scan_workspace_bytes"] = workspace
            report["numpy_version"] = np.__version__
            qvectors, cached_rows, observed, full_best, live_best = {}, {}, {}, {}, {}
            for query, cases in tasks.items():
                queries.file.seek(8 + query * dim)
                raw = queries.file.read(dim)
                coverage.require(len(raw) == dim, "query truncated")
                qvectors[query] = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                ids = coverage.plane(gt.file, 8 + query * cached["cached_k"] * 4, cached["cached_k"], "I")
                distances = coverage.plane(gt.file, 8 + cached["queries"] * cached["cached_k"] * 4
                                            + query * cached["cached_k"] * 4, cached["cached_k"], "f")
                cached_rows[query] = (sorted(ids), dict(zip(ids, distances)), list(distances))
                observed[query], full_best[query] = 0, []
                for name in cases:
                    live_best[query, name] = []

            def neighbors(vectors, first):
                for query, cases in tasks.items():
                    distances = squared_distances(np, vectors, qvectors[query])
                    ids, expected, unused = cached_rows[query]
                    begin, end = bisect_left(ids, first), bisect_left(ids, first + len(vectors))
                    for source_id in ids[begin:end]:
                        coverage.require(float(distances[source_id - first]) == expected[source_id],
                                         f"cached distance disagrees with source/query at query {query}, ID {source_id}")
                        observed[query] += 1
                    full_best[query] = update_best(np, full_best[query], distances, first, cached["cached_k"])
                    for name, (state, reason) in cases.items():
                        for left, right in live_segments(state, first, len(vectors)):
                            if left < right:
                                live_best[query, name] = update_best(np, live_best[query, name],
                                                                    distances[left:right], first + left, args.k_out)

            hash_first = scan_source(np, source, rows, dim, args.chunk_rows, neighbors,
                                     "live_neighbors", args.progress_rows)
            thresholds, counts = {}, {}
            for query, cases in tasks.items():
                coverage.require(observed[query] == cached["cached_k"], "not all cached IDs were checked")
                coverage.require([d for d, i in full_best[query]] == cached_rows[query][2],
                                 f"cache is not exact full-source GT for scanned query {query}")
                thresholds[query] = {}
                for name in cases:
                    best = live_best[query, name]
                    coverage.require(len(best) == args.k_out, "declared live population lacks K neighbors")
                    thresholds[query][name] = best[-1]
                    counts[query, name] = [0, 0, 0]  # strictly nearer, equal, equal through boundary ID

            def ranks(vectors, first):
                for query, cases in tasks.items():
                    distances = squared_distances(np, vectors, qvectors[query])
                    # Share comparisons when several states have one threshold.
                    common = {}
                    for name, (threshold, boundary_id) in thresholds[query].items():
                        if threshold not in common:
                            common[threshold] = (int(np.count_nonzero(distances < threshold)),
                                                 int(np.count_nonzero(distances == threshold)))
                        nearer, equal = common[threshold]
                        length = max(0, min(len(vectors), boundary_id - first + 1))
                        equal_through = int(np.count_nonzero(distances[:length] == threshold))
                        total = counts[query, name]
                        total[0] += nearer
                        total[1] += equal
                        total[2] += equal_through

            hash_second = scan_source(np, source, rows, dim, args.chunk_rows, ranks,
                                      "full_ranks", args.progress_rows)
            coverage.require(hash_first == hash_second, "source changed between exact passes")
            report["source_sha256"] = hash_second
            report["source_distance_passes"] = 2
            for query in sorted(tasks):
                for name, (state, reason) in tasks[query].items():
                    nearer, equal, equal_through = counts[query, name]
                    rank, tie_safe = nearer + equal_through, nearer + equal
                    report["distance_id_sufficient_k"] = max(report["distance_id_sufficient_k"], rank)
                    report["tie_safe_sufficient_k"] = max(report["tie_safe_sufficient_k"], tie_safe)
                    report["cases"].append({"query": query, "state": name, "scan_reason": reason,
                                            "kth_live_squared_distance": thresholds[query][name][0],
                                            "kth_live_source_id": thresholds[query][name][1],
                                            "strictly_nearer_full_vectors": nearer, "equal_distance_full_vectors": equal,
                                            "distance_id_depth": rank, "tie_safe_depth": tie_safe,
                                            "live_top_k": [{"source_id": i, "squared_distance": d}
                                                           for d, i in live_best[query, name]]})
        queries.check()
        gt.check()
        source.check()
        coverage.require(coverage.sha256(queries.file) == query_sha_before, "queries changed during recommendation")
        coverage.require(coverage.sha256(gt.file) == cached["cached_gt_sha256"], "cache changed during recommendation")
    needed = report["tie_safe_sufficient_k"]
    report["recommended_k"] = min(args.full_count, ((needed + args.round_up - 1) // args.round_up) * args.round_up)
    report["recommendation_policy"] = "all_Kth_boundary_ties_included_existing_cache_width_floor_then_round_up"
    report["round_up"] = args.round_up
    report["tagged_cache_bytes"] = 8 + cached["queries"] * report["recommended_k"] * 12
    report["two_plane_cache_bytes"] = 8 + cached["queries"] * report["recommended_k"] * 8
    report["fits_coverage_row_scratch_guard"] = report["recommended_k"] * 128 <= 64 * 1024 * 1024
    report["new_cache_preflight_required"] = True
    return report


def parser(cached_required=True, report_option=True):
    p = coverage.parser(cached_required, report_option)
    p.description = __doc__
    p.add_argument("--base-file", required=True, type=Path, help="unchanged full uint8 source, not an 80M GT slice")
    p.add_argument("--query-file", required=True, type=Path, help="unchanged query order")
    p.add_argument("--chunk-rows", type=coverage.positive, default=100000)
    p.add_argument("--max-scan-queries", type=coverage.positive, default=32)
    p.add_argument("--memory-budget-mb", type=coverage.positive, default=256)
    p.add_argument("--round-up", type=coverage.positive, default=1)
    p.add_argument("--progress-rows", type=coverage.positive, default=1000000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.json_output:
            coverage.require(not os.path.lexists(args.json_output), "JSON output exists; choose a fresh path")
        report = recommend(args)
        print(f"Scanned queries: {report['scan_queries']}; source distance passes: {report['source_distance_passes']}")
        print(f"Distance/ID-order sufficient K: {report['distance_id_sufficient_k']}")
        print(f"Tie-safe sufficient K: {report['tie_safe_sufficient_k']}")
        print(f"Recommended K: {report['recommended_k']}; tagged cache bytes: {report['tagged_cache_bytes']}")
        print("Generate into a NEW path and run check_gt_coverage.py on the new cache before preparation.")
        if not report["fits_coverage_row_scratch_guard"]:
            print("WARNING: recommendation exceeds the coverage checker's row scratch guard; consider direct per-state GT.",
                  file=sys.stderr)
        if args.json_output:
            with args.json_output.open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        return 0
    except (OSError, ValueError, OverflowError, MemoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
