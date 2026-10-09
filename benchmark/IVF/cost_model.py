#!/usr/bin/env python3
"""Run/freeze paper-style calibration. No official query or GT input accepted."""
import argparse
import math
import os
from pathlib import Path
import sys

from harness import digest, flags, require, write_json, read_cost_model_report
from qualify import affinity, execute, runtime_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pq-prefix", type=Path, required=True)
    parser.add_argument("--cpus")
    parser.add_argument("--timeout", type=float, default=1800)
    # Match native flag spelling via a single explicit map; no pass-through flags
    # can inject official queries, result paths, or a supplied timing coefficient.
    defaults = {"data_type": "float", "dim": None, "base_count": None, "sample_count": None,
        "nlist_grid": None, "synthetic_queries": "100", "topk": "10", "target_recall": ".90", "seed": "0",
        "max_reps": "15", "build_threads": "1", "memory_budget_gb": "1", "trainer_mode": "ann-reuse-seeded",
        "init": "random_partition", "training_breadth": "64", "router_mode": "hnswlib", "optimized_modes": "0",
        "router_degree": "16", "construction_breadth": "200", "query_breadth": "128", "insert_breadth": "64",
        "router_seed": "100", "refine_ef": "128", "pca_variance": ".95", "router_rerank_factor": "3",
        "routing_work_limit": "1000000000", "profile_codes": "100000", "profile_repeats": "3",
        "routing_repeats": "3", "rerank_factor": "4", "work_limit": "1000000000"}
    for key, value in defaults.items():
        parser.add_argument("--" + key.replace("_", "-"), default=value, required=value is None)
    args = parser.parse_args()
    require(math.isfinite(args.timeout) and args.timeout > 0, "invalid calibration timeout")
    root, tool, data, pq = (p.resolve() for p in (args.output, args.tool, args.data, args.pq_prefix))
    require(not root.exists(), f"calibration output exists: {root}")
    require(tool.is_file() and os.access(tool, os.X_OK), "missing native cost-model tool")
    paths = {"data": data, "pq_pivots": Path(str(pq) + "_pq_pivots.bin"),
        "pq_codes": Path(str(pq) + "_pq_compressed.bin"), "tool": tool, "orchestrator": Path(__file__).resolve(),
        "harness": Path(__file__).with_name("harness.py").resolve(),
        "qualifier": Path(__file__).with_name("qualify.py").resolve()}
    for name in ("libBufANN.a", "libBufANN.so"):
        library = tool.parent.parent / "src" / name
        if library.is_file():
            paths[name] = library
    identities = {key: digest(path) for key, path in paths.items()}
    settings = {key: getattr(args, key) for key in defaults}
    cpu = affinity(args.cpus)
    root.mkdir(parents=True)
    command = [str(tool)] + flags(settings | {"data_file": str(data), "pq_prefix": str(pq), "output_dir": str(root / "native")})
    write_json(root / "plan.json", {"kind": "ivf_cost_model_calibration", "settings": settings,
        "paths": {key: str(path) for key, path in paths.items()}, "identities": identities,
        "affinity": cpu, "runtime_environment": runtime_environment(), "command": command,
        "timeout_s": args.timeout, "official_queries_used": False, "production_qualified": False})
    wall = execute(command, root / "calibration.log", args.timeout, args.cpus)
    native_path = root / "native/report.json"
    report = read_cost_model_report(native_path)
    require(report["kind"] == "ivf_cost_model" and report["completed"] and report["selected_nlist"] is not None,
            "native calibration did not choose an eligible candidate")
    require(identities == {key: digest(path) for key, path in paths.items()}, "calibration input/tool changed")
    # Freeze actual sample, queries, GT, routers and memberships, not just seeds.
    artifacts = {str(path.relative_to(root / "native")): digest(path)
                 for path in sorted((root / "native").iterdir()) if path.is_file()}
    report.update(identities=identities, artifact_identities=artifacts,
        native_report_identity=digest(native_path), process_wall_s=wall,
        affinity=cpu, runtime_environment=runtime_environment(), validation_completed=False,
        calibration_only=True, target_sample_rate=1)
    write_json(root / "report.json", report)
    print(f"Frozen cost-model report: {root / 'report.json'}; selected nlist={report['selected_nlist']}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
