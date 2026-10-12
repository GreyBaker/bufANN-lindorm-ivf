#!/usr/bin/env python3
"""Read-only cached-GT coverage preflight for prefix/tail-update workloads.

Exit 0: every declared state has K survivors for every query.
Exit 3: valid cache, but some declared states underflow.
Exit 1/2: invalid input/cache/plan. No GT generation or wrapper execution.
Coverage assumes this is exact GT for the intended original source/query order.
"""
import argparse
from array import array
from bisect import bisect_left
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import struct
import sys


@dataclass(frozen=True)
class State:
    name: str
    prefix: int
    delete_begin: int = 0
    delete_end: int = 0

    def contains(self, source_id):
        return source_id < self.prefix and not self.delete_begin <= source_id < self.delete_end

    def count(self, sorted_ids):
        return (bisect_left(sorted_ids, self.prefix)
                - bisect_left(sorted_ids, min(self.delete_end, self.prefix))
                + bisect_left(sorted_ids, min(self.delete_begin, self.prefix)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def round_cap(total, rounds, supplied, name):
    if supplied is not None:
        return supplied
    require(total % rounds == 0, f"{name}: count must divide rounds, or supply an explicit cap")
    return total // rounds


def make_states(full, base, inserts, deletes, rounds, mixed_insert_cap=None,
                mixed_delete_cap=None, insert_round_cap=None, delete_round_cap=None):
    require(0 < base < full <= 0xFFFFFFFF, "require 0 < base < full <= UINT32_MAX")
    require(0 < inserts <= full - base, "insert count exceeds the source tail")
    require(0 < deletes < base, "delete count must be positive and below base")
    require(0 < rounds <= 4096, "mixed rounds must be in [1,4096]")
    ic = round_cap(inserts, rounds, mixed_insert_cap, "mixed inserts")
    dc = round_cap(deletes, rounds, mixed_delete_cap, "mixed deletes")
    require(ic > 0 and base + rounds * ic <= full, "mixed insert rounds exceed source extent")
    require(dc > 0 and rounds * dc < base, "mixed delete rounds exhaust the base")
    mixed_delete_begin = base - rounds * dc
    states = [State("query", base), State("insert", base + inserts),
              State("delete", base - deletes)]
    for name, count, cap in (("insert", inserts, insert_round_cap),
                             ("delete", deletes, delete_round_cap)):
        if cap is None:
            continue
        require(cap > 0 and count % cap == 0, f"{name} count must divide its round cap")
        require(count // cap <= 4096, f"too many {name} rounds")
        for r in range(1, count // cap + 1):
            states.append(State(f"{name}_round_{r}", base + r * cap) if name == "insert"
                          else State(f"{name}_round_{r}", base, base - count, base - count + r * cap))
    for r in range(1, rounds + 1):
        states.append(State(f"mixed_round_{r}", base + r * ic,
                            mixed_delete_begin, mixed_delete_begin + r * dc))
    return states


def stat_signature(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def signature(file):
    return stat_signature(os.fstat(file.fileno()))


def sha256(file):
    file.seek(0)
    digest = hashlib.sha256()
    while block := file.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def plane(file, offset, cells, code):
    file.seek(offset)
    raw = file.read(cells * 4)
    require(len(raw) == cells * 4, "GT changed or truncated during preflight")
    values = array(code)
    require(values.itemsize == 4, "platform lacks required 32-bit array types")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def check_cache(path, full_count, states, k_out, expected_queries=None, worst_queries=5,
                collect_scan_queries=False):
    require(k_out > 0 and worst_queries > 0, "K and worst-query count must be positive")
    stats = [{**asdict(state), "min_survivors": None, "underflow_queries": 0,
              "max_kth_survivor_rank_among_covered": 0, "max_rank_query": None,
              "_worst": []} for state in states]
    if collect_scan_queries:
        for result in stats:
            result["underflow_query_ids"] = []
            result["boundary_tie_query_ids"] = []
    with Path(path).open("rb") as file:
        resolved = Path(path).resolve(strict=True)
        before = signature(file)
        path_before = stat_signature(Path(path).stat())
        # Windows can expose different timestamp snapshots through stat/fstat.
        # Compare identity across APIs and stability within each API, plus hashes.
        require(path_before[:3] == before[:3], "GT path changed during open")
        header = file.read(8)
        require(len(header) == 8, "truncated GT header")
        queries, cached_k = struct.unpack("<II", header)
        require(0 < queries <= 0x7FFFFFFF and 0 < cached_k <= 0x7FFFFFFF,
                "GT query count/depth must be positive int32 values")
        require(expected_queries is None or queries == expected_queries,
                f"GT query count mismatch: expected {expected_queries}, got {queries}")
        cells = queries * cached_k
        size = before[2]
        require(size in (8 + cells * 8, 8 + cells * 12),
                "GT framing mismatch: require header + IDs/distances, optionally identity tags")
        with_tags = size == 8 + cells * 12
        # Sorting/validation needs one whole cached row; refuse extravagant rows
        # before allocation. This is a scratch estimate, not an enforced RSS cap.
        require(cached_k * 128 <= 64 * 1024 * 1024, "cached row exceeds 64MiB scratch estimate")
        rows_per_block = max(1, min(256, (8 * 1024 * 1024) // (cached_k * 12)))
        digest_before = sha256(file)
        for first in range(0, queries, rows_per_block):
            count = min(rows_per_block, queries - first)
            n = count * cached_k
            ids = plane(file, 8 + first * cached_k * 4, n, "I")
            distances = plane(file, 8 + cells * 4 + first * cached_k * 4, n, "f")
            if with_tags:
                tags = plane(file, 8 + cells * 8 + first * cached_k * 4, n, "I")
                require(tags == ids, f"GT tags differ from source IDs near query {first}")
            for local in range(count):
                query = first + local
                begin = local * cached_k
                row = ids[begin:begin + cached_k]
                ordered = sorted(row)
                require(ordered[-1] < full_count, f"GT ID outside full source at query {query}")
                require(all(a != b for a, b in zip(ordered, ordered[1:])),
                        f"duplicate GT ID at query {query}")
                previous = -1.0
                for distance in distances[begin:begin + cached_k]:
                    require(math.isfinite(distance) and distance >= 0 and distance >= previous,
                            f"invalid/unsorted GT distance at query {query}")
                    previous = distance
                for state, result in zip(states, stats):
                    survivors = state.count(ordered)
                    result["min_survivors"] = survivors if result["min_survivors"] is None else min(
                        survivors, result["min_survivors"])
                    result["underflow_queries"] += survivors < k_out
                    if collect_scan_queries and survivors < k_out:
                        result["underflow_query_ids"].append(query)
                    item = (-survivors, -query)
                    worst = result["_worst"]
                    if len(worst) < worst_queries:
                        heapq.heappush(worst, item)
                    elif item > worst[0]:
                        heapq.heapreplace(worst, item)
                    if survivors >= k_out:
                        kept = 0
                        for rank, source_id in enumerate(row, 1):
                            kept += state.contains(source_id)
                            if kept == k_out:
                                # A new cache can choose a different subset of a
                                # tie cut off at its boundary. The recommender
                                # scans these cases too, rather than guessing.
                                if (collect_scan_queries and cached_k < full_count
                                        and distances[begin + rank - 1] == distances[begin + cached_k - 1]):
                                    result["boundary_tie_query_ids"].append(query)
                                if rank > result["max_kth_survivor_rank_among_covered"]:
                                    result["max_kth_survivor_rank_among_covered"] = rank
                                    result["max_rank_query"] = query
                                break
        digest_after = sha256(file)
        require(before == signature(file) and path_before == stat_signature(Path(path).stat())
                and Path(path).resolve(strict=True) == resolved and digest_before == digest_after,
                "GT changed during coverage preflight")
    for result in stats:
        result["worst_queries"] = [{"query": -query, "survivors": -count}
                                   for count, query in sorted(result.pop("_worst"), reverse=True)]
        result["required_cache_depth"] = (None if result["underflow_queries"] else
                                           result["max_kth_survivor_rank_among_covered"])
    return {"version": 1, "coverage_passed": all(not s["underflow_queries"] for s in stats),
            "cached_gt": str(Path(path).resolve()), "cached_gt_sha256": digest_after,
            "cache_bytes": size, "queries": queries, "cached_k": cached_k, "k_out": k_out,
            "with_identity_tags": with_tags, "full_count": full_count,
            "query_id_policy": "zero_based_query_rows", "source_id_policy": "original_source_row_ids",
            "scope": "coverage_of_declared_states_not_recomputation_of_exact_GT_or_provenance",
            "states": stats}


def planned_states(args):
    inserts = args.insert_count if args.insert_count is not None else args.full_count - args.base_count
    deletes = args.delete_count if args.delete_count is not None else args.full_count - args.base_count
    states = make_states(args.full_count, args.base_count, inserts, deletes, args.mixed_rounds,
                         args.mixed_insert_cap, args.mixed_delete_cap,
                         args.insert_round_cap, args.delete_round_cap)
    require(args.k_out <= min(s.prefix - (s.delete_end - s.delete_begin) for s in states),
            "K exceeds a declared live population")
    return states


def parser(cached_required=True, report_option=True):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cached-gt", required=cached_required, type=Path)
    p.add_argument("--full-count", required=True, type=positive)
    p.add_argument("--base-count", required=True, type=positive)
    p.add_argument("--insert-count", type=positive, help="default: full minus base")
    p.add_argument("--delete-count", type=positive, help="default: full minus base")
    p.add_argument("--mixed-rounds", type=positive, default=10)
    p.add_argument("--mixed-insert-cap", type=positive, help="default: insert count / mixed rounds")
    p.add_argument("--mixed-delete-cap", type=positive, help="default: delete count / mixed rounds")
    p.add_argument("--insert-round-cap", type=positive, help="also check intermediate insert-only states")
    p.add_argument("--delete-round-cap", type=positive, help="also check intermediate delete-only states")
    p.add_argument("--k-out", type=positive, default=10)
    p.add_argument("--expected-queries", type=positive)
    p.add_argument("--worst-queries", type=positive, default=5)
    if report_option:
        p.add_argument("--json-output", type=Path, help="optional new evidence file; never overwritten")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.json_output:
            require(not os.path.lexists(args.json_output), "JSON output already exists; choose a fresh path")
        inserts = args.insert_count if args.insert_count is not None else args.full_count - args.base_count
        deletes = args.delete_count if args.delete_count is not None else args.full_count - args.base_count
        states = planned_states(args)
        report = check_cache(args.cached_gt, args.full_count, states, args.k_out,
                             args.expected_queries, args.worst_queries)
        report["plan"] = {"base_count": args.base_count, "insert_count": inserts,
                          "delete_count": deletes, "mixed_rounds": args.mixed_rounds,
                          "mixed_insert_cap": round_cap(inserts, args.mixed_rounds, args.mixed_insert_cap, "mixed inserts"),
                          "mixed_delete_cap": round_cap(deletes, args.mixed_rounds, args.mixed_delete_cap, "mixed deletes"),
                          "insert_round_cap": args.insert_round_cap, "delete_round_cap": args.delete_round_cap}
        print(f"Cached GT: Q={report['queries']} K={report['cached_k']} K_out={args.k_out} "
              f"bytes={report['cache_bytes']} SHA256={report['cached_gt_sha256']}")
        for state in report["states"]:
            worst = ", ".join(f"{q['query']}:{q['survivors']}" for q in state["worst_queries"])
            depth = state["required_cache_depth"]
            print(f"{state['name']}: prefix={state['prefix']} "
                  f"delete=[{state['delete_begin']},{state['delete_end']}) "
                  f"min_survivors={state['min_survivors']} underflow_queries={state['underflow_queries']} "
                  f"required_depth={depth if depth is not None else 'unknown (>cached K for some queries)'} "
                  f"worst(query:survivors)=[{worst}]")
        if args.json_output:
            with args.json_output.open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        print("PASS: declared-state coverage complete" if report["coverage_passed"] else
              "FAIL: cache too shallow; generate a new deeper cache and rerun preflight")
        return 0 if report["coverage_passed"] else 3
    except (OSError, ValueError, OverflowError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
