#!/usr/bin/env python3
"""Measure fixed-router ablations at declared centroid-recall targets; no builds."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", type=Path, required=True)
    p.add_argument("--index-prefix", required=True)
    p.add_argument("--query-file", required=True)
    p.add_argument("--data-type", choices=("float", "int8", "uint8"), default="float")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scope", required=True, help="e.g. synthetic_fixture or sift1m_tuning; never inferred")
    p.add_argument("--nprobes", default="1,4,10,32")
    p.add_argument("--breadths", default="32,64,128")
    p.add_argument("--recall-targets", default="0.90,0.95,0.99")
    p.add_argument("--query-rows", type=int, default=100)
    p.add_argument("--query-threads", type=int, default=1)
    p.add_argument("--router-rerank-factor", type=float, default=3)
    p.add_argument("--memory-budget-gb", type=float, default=1)
    p.add_argument("--evaluation-work-limit", type=int, required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout-s", type=int, required=True)
    args = p.parse_args()
    probes = [int(x) for x in args.nprobes.split(",")]
    breadths = [int(x) for x in args.breadths.split(",")]
    targets = [float(x) for x in args.recall_targets.split(",")]
    assert min(probes+ breadths + [args.repeats,args.timeout_s]) > 0 and all(0 < x <= 1 for x in targets)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"settings.json").write_text(json.dumps(vars(args),default=str,indent=2)+"\n")
    records = []
    for probe in probes:
        for breadth in breadths:
            for mode in ("exact","hnswlib","patched-full","refined","pca","pca-refined"):
                for repeat in range(args.repeats):
                    path = args.output / f"p{probe}_ef{breadth}_{mode}_r{repeat}.json"
                    command = [str(args.diagnostic),"--index_prefix",args.index_prefix,"--query_file",args.query_file,
                        "--data_type",args.data_type,"--nprobe",str(probe),"--router_breadth",str(breadth),
                        "--router_mode",mode,"--router_rerank_factor",str(args.router_rerank_factor),
                        "--query_rows",str(args.query_rows),"--query_threads",str(args.query_threads),
                        "--memory_budget_gb",str(args.memory_budget_gb),"--evaluation_work_limit",str(args.evaluation_work_limit),
                        "--result_file",str(path)]
                    done = subprocess.run(command,text=True,capture_output=True,timeout=args.timeout_s)
                    if done.returncode:
                        (args.output/"failure.txt").write_text(done.stdout+done.stderr)
                        raise RuntimeError("Diagnostic failed; partial evidence retained")
                    records.append({**json.loads(path.read_text()),"repeat":repeat,"record":path.name})
    # Average repeats before selection; compare identical probes and declared
    # minimum recall targets, not arbitrary equal breadths or fitted speedups.
    matched = []
    for probe in probes:
        for target in targets:
            choices = {}
            for mode in ("hnswlib","patched-full","refined","pca","pca-refined"):
                groups = []
                for breadth in breadths:
                    rows = [r for r in records if r["nprobe"] == probe and r["router_mode"] == mode and r["router_breadth"] == breadth]
                    recall = sum(r["centroid_recall_fraction"] for r in rows)/len(rows)
                    qps = sum(r["router_qps"] for r in rows)/len(rows)
                    if recall >= target: groups.append({"breadth":breadth,"mean_recall":recall,"mean_qps":qps})
                choices[mode] = max(groups,key=lambda x:x["mean_qps"]) if groups else None
            matched.append({"nprobe":probe,"minimum_centroid_recall":target,"fastest_observed_per_mode":choices})
    (args.output/"summary.json").write_text(json.dumps({"scope":args.scope,"production_qualified":False,
        "comparison":"fixed_centroid_router_modes_no_spfresh","matched_minimum_recall":matched,"records":records},indent=2)+"\n")


if __name__ == "__main__":
    main()
