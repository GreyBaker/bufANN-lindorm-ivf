#!/usr/bin/env python3
"""Focused S08 typed CLI/mode/cache-family smoke, not a full workload rerun."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from test_cli import write_pivots


def run(command, success=True):
    done = subprocess.run(command,text=True,capture_output=True,timeout=90)
    assert (done.returncode == 0) == success, (command,done.stdout,done.stderr)
    if not success:
        assert "ERROR:" in done.stderr
        return
    records = [json.loads(line) for line in done.stdout.splitlines() if line.startswith('{"baseline":')]
    assert len(records) == 1
    return records[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build",required=True)
    p.add_argument("--driver",required=True)
    p.add_argument("--scratch",type=Path,required=True)
    args = p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"benchmark/IVF"))
    import harness
    args.scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s08_cli_",dir=args.scratch) as tmp:
        root = Path(tmp)
        for dtype,fmt in (("float","f"),("int8","b"),("uint8","B")):
            pq = root/(dtype+"_pq"); write_pivots(Path(str(pq)+"_pq_pivots.bin"))
            Path(str(pq)+"_pq_compressed.bin").write_bytes(struct.pack("<II",4,2)+bytes(8))
            offset = -5 if dtype == "int8" else 5
            vectors = [(offset+1,2,3),(offset-1,2,3),(offset+4,2,3),(offset,2,3)]
            queries = [vectors[2],vectors[0]]
            data,query,prefix = (root/(dtype+suffix) for suffix in (".bin","_queries.bin","_index"))
            data.write_bytes(struct.pack("<II12"+fmt,4,3,*(x for row in vectors for x in row)))
            query.write_bytes(struct.pack("<II6"+fmt,2,3,*(x for row in queries for x in row)))
            record = run([args.build,"--validate_inputs","0","--data_type",dtype,"--dim","3",
                "--data_file",str(data),"--pq_prefix",str(pq),"--index_prefix",str(prefix),"--base_count","3",
                "--nlist","2","--max_reps","2","--optimized_modes","1","--router_mode","pca-refined","--buffer_frames","4"])
            assert record["optimized_modes_built"] == 1 and record["pca_retained_dim"] == 1
            family = harness.bundle_suffixes(prefix)
            assert {".patched.hnsw",".refined.hnsw",".pca"} <= set(family)
            original = {Path(str(prefix)+s): hashlib.sha256(Path(str(prefix)+s).read_bytes()).digest() for s in family}
            command = [args.driver,"--validate_inputs","0","--data_type",dtype,"--index_prefix",str(prefix),
                "--workload","search_only","--query_file",str(query),"--nprobe","1","--recall_at","1",
                "--rerank_factor","8","--query_threads","2","--buffer_frames","4"]
            for mode in ("exact","hnswlib","patched-full","refined","pca","pca-refined"):
                metrics = run(command+["--router_mode",mode,"--router_breadth","128","--router_rerank_factor","3"])
                assert metrics["router_mode"] == mode and metrics["assignment_router_mode"] == "pca-refined"
                assert metrics["router_distance_counted_queries"] == (0 if mode == "hnswlib" else 2)
                assert metrics["logical_fetch_stages"] == 2
            for extra in (["--router_mode","unknown"],["--router_rerank_factor","0.5"],["--router_breadth","0"]):
                run(command+extra,False)
            ids = root/(dtype+".ids"); ids.write_bytes(struct.pack("<II",1,3))
            saved = root/(dtype+"_checkpoint")
            inserted = run([args.driver,"--validate_inputs","0","--data_type",dtype,"--index_prefix",str(prefix),
                "--workload","insert_only","--full_data_file",str(data),"--insert_ids_file",str(ids),
                "--private_heap_path",str(root/(dtype+"_private.heap")),"--checkpoint_prefix",str(saved),"--buffer_frames","4"])
            assert inserted["insert_count"] == 1
            assert set(harness.bundle_suffixes(saved,mutable=True)) >= {".patched.hnsw",".refined.hnsw",".pca"}
            for suffix in (".patched.hnsw",".refined.hnsw",".pca"):
                assert Path(str(prefix)+suffix).read_bytes() == Path(str(saved)+suffix).read_bytes()
            assert all(hashlib.sha256(path.read_bytes()).digest() == value for path,value in original.items())
        # Query-only ablations must not affect build/cache settings.
        os.environ["IVF_NLIST"] = "2"
        settings = harness.build_settings()
        os.environ["IVF_ROUTER_MODE"] = "pca"
        os.environ["IVF_ROUTER_RERANK_FACTOR"] = "5"
        assert harness.build_settings() == settings
        assert harness.query_router_flags() == ["--router_mode","pca","--router_rerank_factor","5"]
    print("S08 CLI modes, typed inputs, frozen assignment, component families and cache settings passed")


if __name__ == "__main__":
    main()
