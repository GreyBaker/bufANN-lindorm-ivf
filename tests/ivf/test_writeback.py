#!/usr/bin/env python3
"""Final writeback accounting and deterministic short/failed-write publication guards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from ivf_test_support import build_tiny_population, reserve_attempt, run_native_cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    root = reserve_attempt(args.scratch).resolve()
    population = build_tiny_population(args.build, root, "float", "f",
                                       reverse_insert_ids=False, include_empty_ids=False,
                                       run_build=run_native_cli)
    source = root / "fail_writeback.c"
    source.write_text(r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset) {
    char link[64], path[4096];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    ssize_t size = readlink(link, path, sizeof(path) - 1);
    if (size >= 0) {
        path[size] = 0;
        const char *target = getenv("IVF_TEST_WRITEBACK_HEAP");
        if (target && strcmp(path, target) == 0) {
            const char message[] = "[fixture] injected final page write failure\n";
            write(2, message, sizeof(message) - 1);
            if (strcmp(getenv("IVF_TEST_WRITEBACK_FAULT"), "short") == 0)
                return count / 2; /* Simulate a short direct-write completion. */
            errno = EIO;
            return -1;
        }
    }
    ssize_t (*real_write)(int, const void *, size_t, off_t) = dlsym(RTLD_NEXT, "pwrite");
    return real_write(fd, buf, count, offset);
}
''')
    library = root / "fail_writeback.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-ldl", "-o", str(library)],
                   check=True, capture_output=True, text=True, timeout=60)
    for workload in ("insert_only", "insert_then_search", "mixed"):
        for fault in (None, "error", "short"):
            out = root / (workload + "-" + (fault or "success"))
            out.mkdir()
            heap, result, neighbors = out / "private.heap", out / "result.json", out / "neighbors.bin"
            command = [args.driver, "--validate_inputs", "0", "--data_type", "float",
                       "--index_prefix", str(population.prefix), "--workload", workload,
                       "--full_data_file", str(population.data),
                       "--insert_ids_file", str(population.insert_ids),
                       "--private_heap_path", str(heap), "--buffer_frames", "64",
                       "--result_file", str(result)]
            if workload != "insert_only":
                command += ["--query_file", str(population.query), "--nprobe", "2",
                            "--recall_at", "4", "--rerank_factor", "16",
                            "--search_result_file", str(neighbors)]
            if workload == "mixed":
                command += ["--delete_ids_file", str(population.delete_ids),
                            "--scheduler", "weighted", "--pool_threads", "2", "--query_count", "8",
                            "--round_results_dir", str(out / "rounds")]
            env = dict(os.environ)
            if fault:
                env.update(LD_PRELOAD=str(library), IVF_TEST_WRITEBACK_HEAP=str(heap),
                           IVF_TEST_WRITEBACK_FAULT=fault)
            run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=90)
            (out / "run.log").write_text(run.stdout + run.stderr)
            foreground = "mixed_foreground" if workload == "mixed" else "insert"
            assert run.stderr.index(f"phase={foreground} event=complete") < run.stderr.index("phase=final_heap_writeback event=begin")
            if fault:
                assert run.returncode != 0 and "BufferPool page write failed" in run.stderr, run.stderr
                assert "phase=final_heap_writeback event=failed" in run.stderr, run.stderr
                assert run.stderr.count("[fixture] injected") == 1, run.stderr
                assert ('actual=2048' if fault == "short" else 'actual=-1') in run.stderr
                assert not any(line.startswith('{"baseline":') for line in run.stdout.splitlines())
                assert not result.exists() and not neighbors.exists()
                assert "phase=post_update_query event=begin" not in run.stderr
                assert heap.is_file() and Path(str(heap) + ".pq.bin").is_file()
                if workload == "mixed":
                    part = json.loads((out / "rounds/round_0.json").read_text())
                    assert part["insert_count"] == 56 and part["recall_available"] is False
            else:
                assert run.returncode == 0, run.stderr
                record = json.loads(result.read_text())
                assert record["update_protocol_version"] == 4
                assert record["writeback_dirty_pages_before"] > 0 and record["writeback_dirty_pages_after"] == 0
                assert record["writeback_heap_page_writes"] == record["writeback_dirty_pages_before"]
                assert record["writeback_heap_flush_bytes"] == record["writeback_heap_page_writes"] * 4096
                assert record["foreground_heap_page_writes"] == 0
                foreground_s = record["mixed_wall_s"] if workload == "mixed" else record["insert_time_s"]
                assert record["driver_workflow_time_s"] >= foreground_s + record["writeback_time_s"]
                assert record["writeback_time_s"] > 0
                if workload != "insert_only":
                    assert run.stderr.index("phase=final_heap_writeback event=complete") < run.stderr.index("phase=post_update_query event=begin")
            assert not Path(str(heap) + ".meta").exists() and not Path(str(heap) + ".manifest").exists()
    assert hashlib.sha256(population.data.read_bytes()).hexdigest() == population.source_hash
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p, value in population.originals.items())
    print("Final writeback accounting and short/failed-write publication guards passed")


if __name__ == "__main__":
    main()
