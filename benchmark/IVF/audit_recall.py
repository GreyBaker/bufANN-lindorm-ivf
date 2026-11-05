#!/usr/bin/env python3
"""Audit query-only prefix results against existing deeper exact GT (no ANN run).

GT: uint32 Q,K; row-major uint32 ID plane, float32 squared-L2 plane, optional
identity-tag plane. Results: uint32 Q,K; row-major interleaved uint32,float32.
Source/query/GT provenance must be verified by the caller. Prefix query subsets
are supported via --query-offset; no source vectors or new ground truths needed.
"""

import argparse
import json
import math
from pathlib import Path
import struct


class Truth:
    def __init__(self, path):
        self.file = path.open("rb")
        header = self.file.read(8)
        if len(header) != 8:
            raise ValueError("truncated GT header")
        self.rows, self.cols = struct.unpack("<II", header)
        size = path.stat().st_size
        if not self.rows or not self.cols or size not in (
            8 + self.rows * self.cols * 8, 8 + self.rows * self.cols * 12
        ):
            raise ValueError("invalid GT framing")
        self.tagged = size == 8 + self.rows * self.cols * 12

    def row(self, q, base_count, k):
        offset = q * self.cols * 4
        plane = self.rows * self.cols * 4
        self.file.seek(8 + offset)
        raw_ids = self.file.read(self.cols * 4)
        ids = struct.unpack(f"<{self.cols}I", raw_ids)
        self.file.seek(8 + plane + offset)
        distances = struct.unpack(f"<{self.cols}f", self.file.read(self.cols * 4))
        if self.tagged:
            self.file.seek(8 + 2 * plane + offset)
            if self.file.read(self.cols * 4) != raw_ids:
                raise ValueError("GT identity tags differ from source-row IDs")
        if len(set(ids)) != len(ids) or any(
            not math.isfinite(d) or d < 0 or (i and d < distances[i - 1])
            for i, d in enumerate(distances)
        ):
            raise ValueError("GT IDs/distances are invalid or unsorted")
        live = [(i, d) for i, d in zip(ids, distances) if i < base_count]
        if len(live) < k:
            raise ValueError(f"query {q}: insufficient live GT neighbors")
        cutoff = live[k - 1][1]
        valid = {i for i, d in live if d <= cutoff}
        # Seeing a strictly larger distance proves the stored prefix includes
        # every boundary tie, assuming this is exact sorted full-source GT.
        covered = distances[-1] > cutoff
        return {i for i, _ in live[:k]}, valid, cutoff, covered


def audit(gt_path, results_path, base_count, k=10, query_offset=0):
    truth = Truth(gt_path)
    try:
        with results_path.open("rb") as result:
            header = result.read(8)
            if len(header) != 8:
                raise ValueError("truncated result header")
            rows, cols = struct.unpack("<II", header)
            if (not rows or cols < k or query_offset < 0 or
                    query_offset + rows > truth.rows or
                    results_path.stat().st_size != 8 + rows * cols * 8):
                raise ValueError("result framing/query range differs from GT")
            strict_hits = tie_hits = incomplete = outside = unexplained = boundary_queries = 0
            for q in range(rows):
                strict, valid, cutoff, covered = truth.row(query_offset + q, base_count, k)
                incomplete += not covered
                boundary_queries += len(valid) > k
                hits = list(struct.iter_unpack("<If", result.read(cols * 8)))[:k]
                present = [i for i, _ in hits if i != 0xFFFFFFFF]
                if len(set(present)) != len(present) or any(i >= base_count for i in present):
                    raise ValueError("duplicate or non-live result ID")
                for i, d in hits:
                    strict_hits += i in strict
                    tie_hits += i in valid
                    if i == 0xFFFFFFFF:
                        continue
                    if not math.isfinite(d) or d < 0:
                        raise ValueError("invalid result distance")
                    outside += d > cutoff
                    unexplained += i not in valid and d <= cutoff
            return {
                "queries": rows, "recall_at": k, "base_count": base_count,
                "query_offset": query_offset, "deep_gt_columns": truth.cols,
                "strict_first_k_id_recall_percent": 100 * strict_hits / (rows * k),
                "tie_expanded_recall_percent": 100 * tie_hits / (rows * k) if not incomplete else None,
                "additional_hits_from_tie_expansion": tie_hits - strict_hits,
                "queries_with_boundary_ties_in_stored_gt": boundary_queries,
                "queries_without_proven_complete_boundary_ties": incomplete,
                "returned_slots_outside_exact_kth_distance": outside,
                "returned_ids_absent_from_gt_but_distance_within_cutoff": unexplained,
                "provenance_policy": "caller_verified_exact_GT_same_source_rows_queries_squared_L2",
            }
    finally:
        truth.file.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deep-gt", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--base-count", type=int, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--query-offset", type=int, default=0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.base_count <= 0 or args.k <= 0:
        p.error("base-count and k must be positive")
    report = audit(args.deep_gt, args.results, args.base_count, args.k, args.query_offset)
    report.update(deep_gt=str(args.deep_gt.resolve()), results=str(args.results.resolve()))
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
