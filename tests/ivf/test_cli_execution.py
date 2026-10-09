#!/usr/bin/env python3
"""Native CLI fixtures; exact IDs/distances and unchanged SPFresh GT scoring."""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from test_cli import write_pivots


def run(command, phase=None):
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    def unique_fields(pairs):
        assert len(dict(pairs)) == len(pairs), ("duplicate JSON metric field", pairs)
        return dict(pairs)
    records = [json.loads(line, object_pairs_hook=unique_fields) for line in result.stdout.splitlines() if line.startswith('{"baseline":')]
    if phase is None:
        assert result.returncode != 0 and "ERROR:" in result.stderr, (command, result.stdout, result.stderr)
        assert not records, result.stdout
        return result
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    assert len(records) == 1 and records[0]["phase"] == phase, result.stdout
    assert records[0]["baseline"] == "IVF"
    return records[0]


def write_truth(path, rows):
    width = len(rows[0])
    path.write_bytes(struct.pack("<II", len(rows), width) +
                     b"".join(struct.pack("<I", vid) for row in rows for vid, _ in row) +
                     b"".join(struct.pack("<f", distance) for row in rows for _, distance in row))


def mutation_fixtures(args, root, dtype, prefix, data, query, vectors, queries):
    insert_ids, delete_ids = root / f"{dtype}_insert.ids", root / f"{dtype}_delete.ids"
    insert_ids.write_bytes(struct.pack("<II", 1, 3))
    delete_ids.write_bytes(struct.pack("<II", 1, 2))
    original = {p: p.read_bytes() for p in root.glob(prefix.name + ".*")}
    base = [args.driver, "--validate_inputs", "0", "--data_type", dtype, "--index_prefix", str(prefix),
            "--full_data_file", str(data), "--buffer_frames", "4"]
    counter = 0
    def command(workload, extra=()):
        nonlocal counter
        counter += 1
        private = root / f"{dtype}_private_{counter}.heap"
        saved = root / f"{dtype}_saved_{counter}"
        return base + ["--workload", workload, "--private_heap_path", str(private),
                       "--checkpoint_prefix", str(saved)] + list(extra), saved, private
    def search(saved, live):
        nonlocal counter
        counter += 1
        truth = root / f"{dtype}_updated_{counter}.gt"
        oracle = [sorted([(i, sum((a - b) ** 2 for a, b in zip(q, vectors[i]))) for i in live],
                         key=lambda hit: (hit[1], hit[0])) for q in queries]
        write_truth(truth, oracle)
        result = root / f"{dtype}_updated_{counter}.bin"
        record = run([args.driver, "--validate_inputs", "0", "--data_type", dtype, "--index_prefix", str(saved),
                      "--index_format", "mutable", "--private_heap_path", str(root / f"{dtype}_query_{counter}.heap"),
                      "--workload", "search_only", "--query_file", str(query), "--gt_file", str(truth),
                      "--nprobe", "2", "--recall_at", "1", "--rerank_factor", "8", "--query_threads", "2",
                      "--buffer_frames", "4", "--search_result_file", str(result)], "query")
        assert record["recall"] == 100 and record["live_count"] == len(live)
        assert read_results(result) == [[row[0]] for row in oracle]
        scorer = Path(__file__).resolve().parents[2] / "benchmark/SPFresh/utils.py"
        scored = subprocess.run([sys.executable, str(scorer), "tie-aware-recall", "--truth", str(truth),
                                 "--result", str(result), "--recall-at", "1"], text=True, capture_output=True)
        assert scored.returncode == 0 and "1.000000" in scored.stdout, scored.stderr
        return truth, oracle
    cmd, inserted, _ = command("insert_only", ["--insert_ids_file", str(insert_ids)])
    record = run(cmd, "insert")
    assert record["insert_count"] == 1 and record["delete_count"] == 0 and record["live_count"] == 4
    assert record["mutation_workers"] == 1 and record["insert_qps"] > 0 and "recall" not in record
    search(inserted, range(4))
    cmd, deleted, _ = command("delete_only", ["--delete_ids_file", str(delete_ids)])
    record = run(cmd, "delete")
    assert record["delete_count"] == 1 and record["retained_heap_records"] == 3 and record["live_count"] == 2
    search(deleted, range(2))
    # Repeat a committed deletion after reopening: counted idempotent no-op.
    cmd, repeated, _ = command("delete_only", ["--delete_ids_file", str(delete_ids), "--index_format", "mutable"])
    cmd[cmd.index("--index_prefix") + 1] = str(deleted)
    record = run(cmd, "delete")
    assert record["delete_count"] == 0 and record["missing_delete_count"] == 1
    search(repeated, range(2))
    # Compose sequential updates with post-update query; one canonical record.
    oracle = [sorted([(i, sum((a - b) ** 2 for a, b in zip(q, vectors[i]))) for i in (0, 1, 3)],
                     key=lambda hit: (hit[1], hit[0])) for q in queries]
    truth = root / f"{dtype}_combined.gt"; write_truth(truth, oracle)
    result = root / f"{dtype}_combined.bin"
    cmd, combined, _ = command("search_after_update", ["--insert_ids_file", str(insert_ids), "--delete_ids_file", str(delete_ids),
        "--query_file", str(query), "--gt_file", str(truth), "--nprobe", "2", "--recall_at", "1",
        "--rerank_factor", "8", "--search_result_file", str(result)])
    record = run(cmd, "query")
    assert record["workload"] == "search_after_update" and record["insert_count"] == record["delete_count"] == 1
    assert record["live_count"] == 3 and record["recall"] == 100 and read_results(result) == [[row[0]] for row in oracle]
    search(combined, [0, 1, 3])
    # Invalid/framing/duplicate/live/no-reuse/source/output refusals do not publish success.
    bad_ids = root / f"{dtype}_bad.ids"
    for payload in (struct.pack("<I", 2) + struct.pack("<I", 3), struct.pack("<III", 2, 3, 3), struct.pack("<II", 1, 4),
                    struct.pack("<II", 1, 0)):
        bad_ids.write_bytes(payload)
        cmd, saved, _ = command("insert_only", ["--insert_ids_file", str(bad_ids)])
        run(cmd); assert not Path(str(saved) + ".manifest").exists()
    bad_ids.write_bytes(struct.pack("<II", 1, 2))
    cmd, saved, _ = command("insert_only", ["--insert_ids_file", str(bad_ids), "--index_format", "mutable"])
    cmd[cmd.index("--index_prefix") + 1] = str(deleted)
    run(cmd); assert not Path(str(saved) + ".manifest").exists()
    changed = root / f"{dtype}_changed_source.bin"
    changed.write_bytes(data.read_bytes()[:-1] + bytes([data.read_bytes()[-1] ^ 1]))
    cmd, saved, _ = command("insert_only", ["--insert_ids_file", str(insert_ids)])
    cmd[cmd.index("--full_data_file") + 1] = str(changed)
    assert "identity mismatch" in run(cmd).stderr and not Path(str(saved) + ".manifest").exists()
    for extra in (["--pq_prefix", "unused"], ["--memory_budget_gb", "0.001"], ["--query_threads", "0"]):
        cmd, saved, _ = command("insert_only", ["--insert_ids_file", str(insert_ids)] + extra)
        run(cmd); assert not Path(str(saved) + ".manifest").exists()
    cmd, saved, private = command("insert_only", ["--insert_ids_file", str(insert_ids)])
    cmd += ["--result_file", str(private)]
    run(cmd); assert not private.exists() and not Path(str(saved) + ".manifest").exists()
    cmd, saved, private = command("insert_only", ["--insert_ids_file", str(insert_ids)])
    Path(str(saved) + ".heap").symlink_to(Path(str(prefix) + ".heap"))
    run(cmd); assert not private.exists() and not Path(str(saved) + ".manifest").exists()
    cmd, saved, private = command("insert_only", ["--insert_ids_file", str(insert_ids)])
    Path(str(private) + ".pq.bin").write_bytes(b"existing")
    run(cmd); assert not private.exists()
    assert all(p.read_bytes() == raw for p, raw in original.items())


def read_results(path):
    data = path.read_bytes()
    rows, k = struct.unpack_from("<II", data)
    assert len(data) == 8 + rows * k * 8
    entries = list(struct.iter_unpack("<If", data[8:]))
    return [entries[q * k:(q + 1) * k] for q in range(rows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    scorer = Path(__file__).resolve().parents[2] / "benchmark" / "SPFresh" / "utils.py"
    # Fresh owned directory each run, so collision protections remain enabled.
    with tempfile.TemporaryDirectory(prefix="cli_native_", dir=args.scratch) as tmp:
        root = Path(tmp)
        for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
            pq_prefix = root / f"{dtype}_pq"
            write_pivots(Path(str(pq_prefix) + "_pq_pivots.bin"))
            codes = Path(str(pq_prefix) + "_pq_compressed.bin")
            codes.write_bytes(struct.pack("<II", 4, 2) + bytes(8))
            # Last row is reserved tail, must not be returned even though it is
            # closest to the first query. Nontrivial negative int8 coordinates.
            offset = -5 if dtype == "int8" else 5
            base = [(offset + 1, 2, 3), (offset - 1, 2, 3), (offset + 4, 2, 3)]
            vectors = base + [(offset, 2, 3)]
            queries = [(offset, 2, 3), base[2], base[0]]
            data = root / f"{dtype}_data.bin"
            data.write_bytes(struct.pack("<II12" + fmt, 4, 3, *(v for row in vectors for v in row)))
            query = root / f"{dtype}_queries.bin"
            query.write_bytes(struct.pack("<II9" + fmt, 3, 3, *(v for row in queries for v in row)))
            truth = root / f"{dtype}_gt.bin"
            oracle = [sorted([(i, sum((a - b) ** 2 for a, b in zip(q, v))) for i, v in enumerate(base)],
                             key=lambda hit: (hit[1], hit[0])) for q in queries]
            # First row deliberately puts the other tied ID first. At K=1 the
            # canonical tie-aware scorer must accept the ID-order winner too.
            gold = [row.copy() for row in oracle]
            gold[0][0], gold[0][1] = gold[0][1], gold[0][0]
            write_truth(truth, gold)
            prefix = root / f"{dtype} native bundle"
            build = [args.build, "--validate_inputs", "0", "--data_type", dtype, "--dim", "3",
                     "--data_file", str(data), "--pq_prefix", str(pq_prefix), "--index_prefix", str(prefix),
                     "--base_count", "3", "--nlist", "2", "--max_reps", "2", "--build_threads", "2",
                      "--buffer_frames", "256", "--query_breadth", "2", "--insert_breadth", "2"]
            for number, extra in enumerate((["--id_capacity", "2"], ["--page_size", "123"],
                                            ["--router_degree", "0"], ["--query_breadth", "0"],
                                            ["--buffer_frames", "1"], ["--memory_budget_gb", "0.001"])):
                bad = build.copy()
                bad_prefix = root / f"{dtype}_refused_{number}"
                bad[bad.index("--index_prefix") + 1] = str(bad_prefix)
                if extra[0] in bad:
                    bad[bad.index(extra[0]) + 1] = extra[1]
                else:
                    bad += extra
                run(bad)
                assert not Path(str(bad_prefix) + ".manifest").exists()
            build_metrics = root / f"{dtype}_build.json"
            record = run(build + ["--result_file", str(build_metrics)], "build")
            assert json.loads(build_metrics.read_text()) == record
            assert record["base_count"] == 3 and record["source_rows"] == 4 and record["nlist"] == 2
            assert record["build_time_s"] > 0 and "recall" not in record and "query_qps" not in record
            manifest = Path(str(prefix) + ".manifest").read_bytes()
            run(build)
            assert Path(str(prefix) + ".manifest").read_bytes() == manifest
            mutation_fixtures(args, root, dtype, prefix, data, query, vectors, queries)
            # Search bundle must need neither original vectors nor source PQ.
            data.unlink()
            codes.unlink()
            Path(str(pq_prefix) + "_pq_pivots.bin").unlink()
            driver = [args.driver, "--validate_inputs", "0", "--data_type", dtype, "--index_prefix", str(prefix),
                      "--workload", "search_only", "--query_file", str(query), "--nprobe", "2",
                      "--recall_at", "1", "--rerank_factor", "3", "--gt_file", str(truth), "--query_threads", "3",
                      "--buffer_frames", "4"]
            result_path = root / f"{dtype}_results.bin"
            metrics_path = root / f"{dtype}_query.json"
            record = run(driver + ["--search_result_file", str(result_path), "--result_file", str(metrics_path)], "query")
            assert json.loads(metrics_path.read_text()) == record and record["workload"] == "query"
            assert read_results(result_path) == [[row[0]] for row in oracle]
            assert record["recall"] == 100 and record["query_count"] == 3
            assert record["scanned_codes"] == 9 and record["reranked_vectors"] == 9
            assert record["logical_fetch_stages"] == 3
            assert record["backend_batch_calls"] == 9  # four 1-frame shards => 1-row batches
            assert record["active_query_threads"] == 3 and record["queries_with_short_results"] == 0
            assert record["query_qps"] > 0 and record["query_time_s"] > 0 and record["open_time_s"] > 0
            assert 0 <= record["query_lat_p50_us"] <= record["query_lat_p95_us"] <= record["query_lat_p99_us"]
            assert record["query_lat_avg_us"] > 0 and record["query_lat_p999_us"] >= record["query_lat_p99_us"]
            for key in ("routing_us_total", "adc_us_total", "scan_select_us_total", "fetch_us_total", "rerank_us_total"):
                assert math.isfinite(record[key]) and record[key] >= 0
            scored = subprocess.run([sys.executable, str(scorer), "tie-aware-recall", "--truth", str(truth),
                                     "--result", str(result_path), "--recall-at", "1"], text=True, capture_output=True)
            assert scored.returncode == 0 and "TieAwareRecall1@1: 1.000000" in scored.stdout, scored.stderr
            # Single-worker parity; more frames changes actual batch counts, not IDs.
            single = driver.copy()
            single[single.index("--query_threads") + 1] = "1"
            single[single.index("--buffer_frames") + 1] = "256"
            single_path = root / f"{dtype}_single.bin"
            single_record = run(single + ["--search_result_file", str(single_path)], "query")
            assert single_path.read_bytes() == result_path.read_bytes()
            assert single_record["backend_batch_calls"] == 3
            run(driver + ["--search_result_file", str(result_path)])  # must not overwrite
            run(driver + ["--result_file", str(metrics_path)])
            assert json.loads(metrics_path.read_text()) == record
            same_output = root / f"{dtype}_same_output"
            run(driver + ["--result_file", str(same_output), "--search_result_file", str(same_output)])
            assert not same_output.exists()
            assert result_path.read_bytes() == single_path.read_bytes()
            allowance = str((record["estimated_workspace_bytes"] - 4096) / (1 << 30))
            refused = run(driver + ["--memory_budget_gb", allowance])
            assert "CLI composed search exceeds workspace allowance" in refused.stderr
            # No GT means no invented recall field. Short rows get -1/inf only
            # in exported fixed-width slots and are counted explicitly.
            no_gt = driver.copy()
            del no_gt[no_gt.index("--gt_file"):no_gt.index("--gt_file") + 2]
            no_gt[no_gt.index("--recall_at") + 1] = "5"
            short_path = root / f"{dtype}_short.bin"
            short = run(no_gt + ["--search_result_file", str(short_path)], "query")
            assert "recall" not in short and short["queries_with_short_results"] == 3
            for row, expected in zip(read_results(short_path), oracle):
                assert row[:3] == expected and row[3:] == [(0xFFFFFFFF, math.inf)] * 2
            for extra in (["--dim", "4"], ["--nlist", "3"], ["--nprobe", "0"],
                          ["--query_threads", "0"], ["--buffer_frames", "1"],
                          ["--memory_budget_gb", "0.001"], ["--pq_prefix", "unused"]):
                # Avoid duplicate flags: replace supplied setting when present.
                bad = driver.copy()
                if extra[0] in bad:
                    bad[bad.index(extra[0]) + 1] = extra[1]
                else:
                    bad += extra
                run(bad)
            for workload in ("insert_only", "delete_only", "search_after_update", "bogus"):
                bad = driver.copy(); bad[bad.index("--workload") + 1] = workload
                run(bad)
            bad_truth = root / f"{dtype}_bad_gt.bin"
            malformed = [row.copy() for row in gold]
            malformed[0][1] = (malformed[0][0][0], math.nan)
            write_truth(bad_truth, malformed)
            bad = driver.copy(); bad[bad.index("--gt_file") + 1] = str(bad_truth)
            run(bad)
            malformed = [row.copy() for row in gold]
            malformed[0][1] = malformed[0][0]
            write_truth(bad_truth, malformed)
            assert "Duplicate GT ID" in run(bad).stderr
            malformed = [row.copy() for row in gold]
            malformed[0].reverse()
            write_truth(bad_truth, malformed)
            assert "GT distances" in run(bad).stderr
            # Canonical helper GT has a third, identity tags plane. Consume it
            # unchanged; do not silently score unsupported tag-space remaps.
            tagged_truth = root / f"{dtype}_tagged_gt.bin"
            tagged_truth.write_bytes(truth.read_bytes() +
                                     b"".join(struct.pack("<I", vid) for row in gold for vid, _ in row))
            tagged = driver.copy(); tagged[tagged.index("--gt_file") + 1] = str(tagged_truth)
            assert run(tagged, "query")["recall"] == 100
            tagged_truth.write_bytes(tagged_truth.read_bytes()[:-4] + struct.pack("<I", 99))
            assert "GT tags must equal source row IDs" in run(tagged).stderr
            if dtype == "float":
                query.write_bytes(struct.pack("<II9f", 3, 3, math.nan, *([0.0] * 8)))
                run(driver)
    print("Native IVF CLI execution/oracle/SPFresh recall fixtures passed for all dtypes")


if __name__ == "__main__":
    main()
