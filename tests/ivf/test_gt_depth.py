"""Small integer-oracle tests for untimed, targeted shared-GT depth advice."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark/scripts"))
import recommend_gt_depth as depth
import check_gt_coverage as coverage


def write_vectors(path, rows):
    rows = np.asarray(rows, dtype=np.uint8)
    path.write_bytes(struct.pack("<II", *rows.shape) + rows.tobytes())


def exact_distances(base, query):
    diff = base.astype(np.int64) - query.astype(np.int64)
    return np.sum(diff * diff, axis=1, dtype=np.int64)


def write_cache(path, base, queries, k, custom_ids=None, tags=True):
    all_ids, all_distances = [], []
    for q, query in enumerate(queries):
        distances = exact_distances(base, query)
        ids = np.lexsort((np.arange(len(base)), distances))[:k] if custom_ids is None else custom_ids[q]
        all_ids.extend(int(i) for i in ids)
        all_distances.extend(float(distances[i]) for i in ids)
    path.write_bytes(struct.pack("<II", len(queries), k)
                     + np.asarray(all_ids, dtype="<u4").tobytes()
                     + np.asarray(all_distances, dtype="<f4").tobytes()
                     + (np.asarray(all_ids, dtype="<u4").tobytes() if tags else b""))


@unittest.skipUnless(np is not None, "exact GT depth tests require NumPy")
class DepthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gt_depth_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.queries, self.cache = [self.root / name for name in ("base.bin", "query.bin", "gt.bin")]
        self.base = np.zeros((100, 2), dtype=np.uint8)
        self.base[:80, 0] = np.arange(100, 180, dtype=np.uint8)
        self.base[1, 0], self.base[2, 0] = 20, 21
        self.base[80:, 0] = np.arange(20, dtype=np.uint8)
        self.q = np.zeros((1, 2), dtype=np.uint8)
        self.prepare(k=10)

    def prepare(self, k, custom_ids=None, tags=True):
        write_vectors(self.source, self.base)
        write_vectors(self.queries, self.q)
        write_cache(self.cache, self.base, self.q, k, custom_ids, tags)

    def arguments(self, extra=()):
        return depth.parser().parse_args([
            "--cached-gt", str(self.cache), "--base-file", str(self.source), "--query-file", str(self.queries),
            "--full-count", "100", "--base-count", "90", "--k-out", "2",
            "--chunk-rows", "13", "--expected-queries", str(len(self.q)), "--progress-rows", "1000", *extra])

    def test_exact_rank_and_two_shared_scans(self):
        calls = []
        original = depth.scan_source

        def spy(*args):
            calls.append(args[-2])
            return original(*args)

        with patch.object(depth, "scan_source", spy):
            result = depth.recommend(self.arguments())
        self.assertEqual(calls, ["live_neighbors", "full_ranks"])
        self.assertEqual(result["scan_queries"], [0])
        self.assertEqual(result["recommended_k"], 22)
        self.assertEqual(result["distance_id_sufficient_k"], 22)
        self.assertEqual(result["tie_safe_sufficient_k"], 22)
        deletion = next(c for c in result["cases"] if c["state"] == "delete")
        self.assertEqual(deletion["live_top_k"], [{"source_id": 1, "squared_distance": 400},
                                                {"source_id": 2, "squared_distance": 441}])
        self.assertEqual(deletion["strictly_nearer_full_vectors"], 21)
        self.assertTrue(result["new_cache_preflight_required"])

    def test_advice_and_regenerated_coverage(self):
        result = depth.recommend(self.arguments())
        new_cache = self.root / "new-gt.bin"
        write_cache(new_cache, self.base, self.q, result["recommended_k"])
        states = coverage.make_states(100, 90, 10, 10, 10)
        self.assertTrue(coverage.check_cache(new_cache, 100, states, 2)["coverage_passed"])
        write_cache(new_cache, self.base, self.q, 21)
        self.assertFalse(coverage.check_cache(new_cache, 100, states, 2)["coverage_passed"])
        rounded = depth.recommend(self.arguments(["--round-up", "10"]))
        self.assertEqual(rounded["recommended_k"], 30)
        self.assertEqual(rounded["tagged_cache_bytes"], 8 + 30 * 12)

    def test_already_covered_avoids_vector_scans(self):
        self.q = np.array([[150, 0]], dtype=np.uint8)
        self.prepare(k=10, tags=False)
        with patch.object(depth, "scan_source", side_effect=AssertionError("unnecessary scan")):
            result = depth.recommend(self.arguments())
        self.assertEqual(result["source_distance_passes"], 0)
        self.assertEqual(result["scan_queries"], [])
        self.assertIsNone(result["source_sha256"])
        self.assertEqual(result["recommended_k"], 10)

    def test_ties_are_safe_not_claimed_minimum(self):
        self.base[:] = 0
        self.prepare(k=10)
        result = depth.recommend(self.arguments())
        self.assertEqual(result["cached_coverage"]["coverage_passed"], True)
        self.assertEqual(result["scan_queries"], [0])
        self.assertEqual(result["distance_id_sufficient_k"], 10)
        self.assertEqual(result["tie_safe_sufficient_k"], 100)
        self.assertEqual(result["recommended_k"], 100)
        self.assertTrue(all(c["scan_reason"] == "boundary_tie" for c in result["cases"]))
        self.prepare(k=10, custom_ids=[list(range(90, 100))])
        arbitrary_ties = depth.recommend(self.arguments())
        self.assertEqual(arbitrary_ties["recommended_k"], 100)

    def test_integer_oracle_multiple_queries_128d(self):
        rng = np.random.default_rng(412)
        self.base = rng.integers(0, 256, (100, 128), dtype=np.uint8)
        self.q = rng.integers(0, 256, (7, 128), dtype=np.uint8)
        self.prepare(k=1)
        with patch.object(depth, "squared_distances", wraps=depth.squared_distances) as measured:
            result = depth.recommend(self.arguments(["--chunk-rows", "7"]))
        self.assertEqual(measured.call_count, 2 * 15 * 7)
        states = {s.name: s for s in coverage.make_states(100, 90, 10, 10, 10)}
        maximum = 1
        for case in result["cases"]:
            d = exact_distances(self.base, self.q[case["query"]])
            ids = [i for i in range(100) if states[case["state"]].contains(i)]
            expected = sorted((int(d[i]), i) for i in ids)[:2]
            self.assertEqual(case["live_top_k"], [{"source_id": i, "squared_distance": v} for v, i in expected])
            v, i = expected[-1]
            self.assertEqual(case["distance_id_depth"], sum((int(x), j) <= (v, i) for j, x in enumerate(d)))
            self.assertEqual(case["tie_safe_depth"], int(np.count_nonzero(d <= v)))
            maximum = max(maximum, case["tie_safe_depth"])
        self.assertEqual(result["recommended_k"], maximum)

    def test_cached_distances_and_rank_mismatch_rejected(self):
        raw = bytearray(self.cache.read_bytes())
        raw[8 + 10 * 4:8 + 10 * 4 + 4] = struct.pack("<f", 0.5)
        self.cache.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            depth.recommend(self.arguments())
        # Distances and IDs agree with actual coordinates, but are not nearest.
        self.prepare(k=10, custom_ids=[list(range(90, 100))])
        with self.assertRaisesRegex(ValueError, "not exact full-source"):
            depth.recommend(self.arguments())

    def test_query_and_source_shape_rejected(self):
        self.source.write_bytes(self.source.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "framing"):
            depth.recommend(self.arguments())
        self.prepare(k=10)
        write_vectors(self.queries, np.zeros((2, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "query rows"):
            depth.recommend(self.arguments())
        self.prepare(k=10)
        write_vectors(self.queries, np.zeros((1, 129), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "1..128"):
            depth.recommend(self.arguments())

    def test_scan_admission(self):
        with self.assertRaisesRegex(ValueError, "scratch"):
            depth.recommend(self.arguments(["--memory-budget-mb", "1"]))
        self.q = np.zeros((2, 2), dtype=np.uint8)
        self.prepare(k=10)
        with patch.object(depth, "scan_source", side_effect=AssertionError("must reject first")), \
                self.assertRaisesRegex(ValueError, "above --max-scan"):
            depth.recommend(self.arguments(["--max-scan-queries", "1"]))

    def test_input_changes_detected(self):
        for name in ("source", "queries", "cache"):
            self.prepare(k=10)
            original = depth.scan_source
            calls = 0

            def mutate(*args):
                nonlocal calls
                calls += 1
                result = original(*args)
                if calls == 1:
                    path = getattr(self, name)
                    raw = bytearray(path.read_bytes())
                    raw[-1] ^= 1
                    path.write_bytes(raw)
                return result

            with self.subTest(name=name), patch.object(depth, "scan_source", mutate), \
                    self.assertRaisesRegex(ValueError, "changed"):
                depth.recommend(self.arguments())

    def test_cli_never_overwrites_or_generates_gt(self):
        args = self.arguments(["--json-output", str(self.root / "advice.json")])
        command = [sys.executable, str(REPO / "benchmark/scripts/recommend_gt_depth.py"),
                   "--cached-gt", str(args.cached_gt), "--base-file", str(args.base_file),
                   "--query-file", str(args.query_file), "--full-count", "100", "--base-count", "90",
                   "--k-out", "2", "--chunk-rows", "13", "--progress-rows", "1000",
                   "--json-output", str(args.json_output)]
        before = {p: p.read_bytes() for p in (self.source, self.queries, self.cache)}
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(args.json_output.read_text())
        self.assertEqual(report["recommended_k"], 22)
        self.assertFalse(report["gt_generated"])
        saved = args.json_output.read_bytes()
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(args.json_output.read_bytes(), saved)
        self.assertEqual(before, {p: p.read_bytes() for p in before})


if __name__ == "__main__":
    if np is None:
        print("Exact GT depth tests require NumPy; no exact-scan qualification performed.")
        sys.exit(77)
    unittest.main()
