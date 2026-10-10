#!/usr/bin/env python3
"""Measure fixed-router ablations at declared centroid-recall targets; no builds."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from common import (
    affinity,
    digest,
    execute,
    finite,
    positive,
    read_json,
    require,
    runtime_environment,
    shared_module_paths,
    values,
    write_json,
)

MODES = ("exact", "hnswlib", "patched-full", "refined", "pca", "pca-refined")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", type=Path, required=True)
    p.add_argument("--index-prefix", required=True)
    p.add_argument("--query-file", required=True)
    p.add_argument("--data-type", choices=("float", "int8", "uint8"), default="float")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--scope", required=True, help="e.g. synthetic_fixture or sift1m_tuning; never inferred"
    )
    p.add_argument("--nprobes", default="1,4,10,32")
    p.add_argument("--breadths", default="32,64,128")
    p.add_argument("--recall-targets", default="0.90,0.95,0.99")
    p.add_argument("--query-rows", type=positive, default=100)
    p.add_argument("--query-threads", type=positive, default=1)
    p.add_argument("--router-rerank-factor", type=finite, default=3)
    p.add_argument("--memory-budget-gb", type=finite, default=1)
    p.add_argument("--evaluation-work-limit", type=positive, required=True)
    p.add_argument("--repeats", type=positive, default=1)
    p.add_argument("--timeout-s", type=finite, required=True)
    p.add_argument("--cpus", help="optional comma-separated CPU IDs within current affinity")
    args = p.parse_args()
    require(sys.platform == "linux", "router sweeps target Linux/WSL")
    probes = values(args.nprobes, int, 1)
    breadths = values(args.breadths, int, 1)
    targets = values(args.recall_targets, float, 0)
    require(all(0 < target <= 1 for target in targets), "recall targets must be in (0,1]")
    require(args.router_rerank_factor >= 1, "router rerank factor must be >=1")
    require(
        len(probes) * len(breadths) * len(MODES) * args.repeats <= 4096,
        "router sweep exceeds 4096-trial safety limit",
    )
    cpu = affinity(args.cpus)
    require(
        args.diagnostic.is_file() and os.access(args.diagnostic, os.X_OK),
        "missing router diagnostic tool",
    )
    inputs = {
        "diagnostic": args.diagnostic.resolve(),
        "query": Path(args.query_file).resolve(),
        "orchestrator": Path(__file__).resolve(),
        **shared_module_paths(),
    }
    for suffix in (".centroids.bin", ".hnsw", ".router", ".patched.hnsw", ".refined.hnsw", ".pca"):
        inputs[suffix] = Path(args.index_prefix + suffix).resolve()
    identities = {name: digest(path) for name, path in inputs.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    write_json(args.output / "settings.json", settings)
    write_json(
        args.output / "plan.json",
        {
            "kind": "ivf_router_sweep",
            "settings": settings,
            "inputs": {name: str(path) for name, path in inputs.items()},
            "identities": identities,
            "affinity": cpu,
            "runtime_environment": runtime_environment(),
            "production_qualified": False,
            "cache_policy": "uncontrolled_after_input_hashes_no_cache_drop",
            "selection_policy": "mean_recall_and_mean_qps_across_repeats_fastest_observed_per_mode",
        },
    )
    records = []
    for probe in probes:
        for breadth in breadths:
            for mode in MODES:
                for repeat in range(args.repeats):
                    path = args.output / f"p{probe}_ef{breadth}_{mode}_r{repeat}.json"
                    command = [
                        str(args.diagnostic),
                        "--index_prefix",
                        args.index_prefix,
                        "--query_file",
                        args.query_file,
                        "--data_type",
                        args.data_type,
                        "--nprobe",
                        str(probe),
                        "--router_breadth",
                        str(breadth),
                        "--router_mode",
                        mode,
                        "--router_rerank_factor",
                        str(args.router_rerank_factor),
                        "--query_rows",
                        str(args.query_rows),
                        "--query_threads",
                        str(args.query_threads),
                        "--memory_budget_gb",
                        str(args.memory_budget_gb),
                        "--evaluation_work_limit",
                        str(args.evaluation_work_limit),
                        "--result_file",
                        str(path),
                    ]
                    log = path.with_suffix(".log")
                    try:
                        wall = execute(command, log, args.timeout_s, args.cpus)
                        record = read_json(path)
                        require(
                            record["nprobe"] == probe
                            and record["router_breadth"] == breadth
                            and record["router_mode"] == mode,
                            "router diagnostic configuration mismatch",
                        )
                        recall, qps = record["centroid_recall_fraction"], record["router_qps"]
                        require(
                            math.isfinite(recall)
                            and 0 <= recall <= 1
                            and math.isfinite(qps)
                            and qps > 0,
                            "invalid router measurement",
                        )
                        require(
                            {name: digest(item) for name, item in inputs.items()} == identities,
                            "router sweep input/tool identity changed",
                        )
                    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
                        with (args.output / "failure.txt").open("x") as failure:
                            failure.write(
                                str(error) + "\n" + (log.read_text() if log.is_file() else "")
                            )
                        raise
                    records.append(
                        {**record, "repeat": repeat, "record": path.name, "process_wall_s": wall}
                    )
    # Average repeats before selection; compare identical probes and declared
    # minimum recall targets, not arbitrary equal breadths or fitted speedups.
    matched = []
    for probe in probes:
        for target in targets:
            choices = {}
            for mode in MODES[1:]:
                groups = []
                for breadth in breadths:
                    rows = [
                        r
                        for r in records
                        if r["nprobe"] == probe
                        and r["router_mode"] == mode
                        and r["router_breadth"] == breadth
                    ]
                    recall = sum(r["centroid_recall_fraction"] for r in rows) / len(rows)
                    qps = sum(r["router_qps"] for r in rows) / len(rows)
                    if recall >= target:
                        groups.append({"breadth": breadth, "mean_recall": recall, "mean_qps": qps})
                choices[mode] = max(groups, key=lambda x: x["mean_qps"]) if groups else None
            matched.append(
                {
                    "nprobe": probe,
                    "minimum_centroid_recall": target,
                    "fastest_observed_per_mode": choices,
                }
            )
    write_json(
        args.output / "summary.json",
        {
            "scope": args.scope,
            "production_qualified": False,
            "completed": True,
            "plan_identity": digest(args.output / "plan.json"),
            "comparison": "fixed_centroid_router_modes_no_spfresh",
            "matched_minimum_recall": matched,
            "records": records,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
