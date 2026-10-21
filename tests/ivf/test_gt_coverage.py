"""Synthetic, offline tests for standalone cached-GT coverage preflight."""
from array import array
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "benchmark/scripts/check_gt_coverage.py"
spec = importlib.util.spec_from_file_location("check_gt_coverage", SCRIPT)
coverage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = coverage
spec.loader.exec_module(coverage)


def plane(values, code):
    result = array(code, values)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def write_gt(path, rows, tags=True, distances=None):
    width = len(rows[0])
    assert rows and all(len(r) == width for r in rows)
    ids = [i for row in rows for i in row]
    ds = distances if distances is not None else [float(k) for _ in rows for k in range(width)]
    path.write_bytes(struct.pack("<II", len(rows), width) + plane(ids, "I") + plane(ds, "f")
                     + (plane(ids, "I") if tags else b""))


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gt_coverage_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gt = self.root / "cache.bin"
        self.states = coverage.make_states(100, 90, 10, 10, 10)

    def check(self, k=2, **kwargs):
        return coverage.check_cache(self.gt, 100, self.states, k, **kwargs)

    def cli(self, extra=()):
        return subprocess.run([sys.executable, str(SCRIPT), "--cached-gt", str(self.gt),
                               "--full-count", "100", "--base-count", "90", "--k-out", "2",
                               *extra], capture_output=True, text=True, timeout=30)

    def test_defaults_and_boundaries(self):
        states = coverage.make_states(100_000_000, 90_000_000, 10_000_000, 10_000_000, 10)
        self.assertEqual(len(states), 13)
        self.assertEqual(states[0].prefix, 90_000_000)
        self.assertEqual(states[1].prefix, 100_000_000)
        self.assertEqual(states[2].prefix, 80_000_000)
        self.assertEqual(states[3], coverage.State("mixed_round_1", 91_000_000, 80_000_000, 81_000_000))
        self.assertEqual(states[-1], coverage.State("mixed_round_10", 100_000_000, 80_000_000, 90_000_000))
        self.assertTrue(states[3].contains(79_999_999))
        self.assertFalse(states[3].contains(80_000_000))
        self.assertTrue(states[3].contains(81_000_000))
        self.assertTrue(states[3].contains(90_000_000))
        self.assertFalse(states[3].contains(91_000_000))

    def plan_args(self, extra=()):
        return coverage.parser().parse_args([
            "--cached-gt", str(self.gt), "--full-count", "100", "--base-count", "90", "--k-out", "2", *extra])

    def test_resolved_plan_defaults_and_cli_report(self):
        args = self.plan_args()
        states, plan = coverage.resolved_plan(args)
        self.assertEqual(states, self.states)
        self.assertEqual(states, coverage.planned_states(args))
        self.assertEqual([s.name for s in states], ["query", "insert", "delete"]
                         + [f"mixed_round_{r}" for r in range(1, 11)])
        self.assertEqual(plan, {"base_count": 90, "insert_count": 10, "delete_count": 10,
                               "mixed_rounds": 10, "mixed_insert_cap": 1, "mixed_delete_cap": 1,
                               "insert_round_cap": None, "delete_round_cap": None})
        self.assertIsNone(args.insert_count)
        self.assertIsNone(args.delete_count)
        write_gt(self.gt, [[1, 2, 3]])
        output = self.root / "plan.json"
        result = self.cli(["--json-output", str(output)])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["plan"], plan)
        self.assertEqual([s["name"] for s in report["states"]], [s.name for s in states])

    def test_resolved_plan_independent_caps_and_state_order(self):
        extra = ["--base-count", "80", "--insert-count", "6", "--delete-count", "8",
                 "--mixed-rounds", "3", "--mixed-insert-cap", "5", "--mixed-delete-cap", "7",
                 "--insert-round-cap", "3", "--delete-round-cap", "2"]
        args = self.plan_args(extra)
        states, plan = coverage.resolved_plan(args)
        expected = [coverage.State("query", 80), coverage.State("insert", 86), coverage.State("delete", 72),
                    coverage.State("insert_round_1", 83), coverage.State("insert_round_2", 86)]
        expected += [coverage.State(f"delete_round_{r}", 80, 72, 72 + r * 2) for r in range(1, 5)]
        expected += [coverage.State(f"mixed_round_{r}", 80 + r * 5, 59, 59 + r * 7) for r in range(1, 4)]
        self.assertEqual(states, expected)
        self.assertEqual(coverage.planned_states(args), expected)
        self.assertEqual(plan, {"base_count": 80, "insert_count": 6, "delete_count": 8,
                               "mixed_rounds": 3, "mixed_insert_cap": 5, "mixed_delete_cap": 7,
                               "insert_round_cap": 3, "delete_round_cap": 2})
        write_gt(self.gt, [[1, 2, 3]])
        output = self.root / "custom-plan.json"
        result = self.cli([*extra, "--json-output", str(output)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["plan"], plan)

    def test_resolved_plan_retains_validation_order(self):
        for extra, error in ((["--base-count", "100", "--k-out", "1000"], "require 0 < base"),
                             (["--insert-count", "5"], "mixed inserts: count must divide rounds"),
                             (["--k-out", "81"], "K exceeds a declared live population")):
            for resolve in (coverage.resolved_plan, coverage.planned_states):
                with self.subTest(extra=extra, resolve=resolve.__name__), self.assertRaisesRegex(ValueError, error):
                    resolve(self.plan_args(extra))

    def test_layout_and_row_scratch_boundary(self):
        layout = coverage.GTLayout(3, 5)
        self.assertEqual(layout.cells, 15)
        for index in range(3):
            self.assertEqual(layout.plane_offset(index), 8 + index * 15 * 4)
            self.assertEqual(layout.plane_offset(index, 2), 8 + (index * 15 + 10) * 4)
        for tags, size in ((False, 128), (True, 188)):
            self.assertEqual(layout.byte_size(tags), size)
            self.assertEqual(layout.identity_tags(size), tags)
        with self.assertRaisesRegex(ValueError, "GT framing mismatch"):
            layout.identity_tags(129)
        limit = 524288
        self.assertEqual(coverage.row_scratch_bytes(limit), 64 * 1024 * 1024)
        self.assertTrue(coverage.fits_row_scratch(limit))
        self.assertFalse(coverage.fits_row_scratch(limit + 1))

    def test_framing_and_scratch_checks_precede_hashing(self):
        width = 524289
        self.gt.write_bytes(struct.pack("<II", 1, width))
        with patch.object(coverage, "sha256", side_effect=AssertionError("must reject before hashing")):
            with self.assertRaisesRegex(ValueError, "query count mismatch"):
                self.check(expected_queries=2)
            with self.assertRaisesRegex(ValueError, "GT framing mismatch"):
                self.check()
            with self.gt.open("r+b") as file:
                file.truncate(8 + width * 8)
            with self.assertRaisesRegex(ValueError, "cached row exceeds 64MiB scratch estimate"):
                self.check()

    def test_importlib_loading_without_scripts_on_sys_path(self):
        code = '''import importlib.util, pathlib, sys
path = pathlib.Path(sys.argv[1])
assert str(path.parent) not in sys.path
spec = importlib.util.spec_from_file_location("standalone_coverage", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.GTLayout(2, 3).byte_size(True) == 80
assert module.fits_row_scratch(524288)
'''
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(SCRIPT)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_underflow_and_deeper_cache(self):
        shallow = [[80, 81, 82, 83, 84, 85, 86, 87, 88, 1],
                   [0, 10, 20, 30, 40, 50, 60, 70, 75, 79]]
        write_gt(self.gt, shallow)
        original = self.gt.read_bytes()
        report = self.check(worst_queries=1)
        self.assertFalse(report["coverage_passed"])
        query, insert, delete = report["states"][:3]
        self.assertEqual(query["min_survivors"], 10)
        self.assertEqual(insert["min_survivors"], 10)
        self.assertEqual(delete["min_survivors"], 1)
        self.assertEqual(delete["underflow_queries"], 1)
        self.assertEqual(delete["worst_queries"], [{"query": 0, "survivors": 1}])
        self.assertIsNone(delete["required_cache_depth"])
        self.assertEqual(self.gt.read_bytes(), original)
        deep = self.root / "deeper.bin"
        write_gt(deep, [shallow[0] + list(range(2, 12)), shallow[1] + list(range(80, 90))])
        fixed = coverage.check_cache(deep, 100, self.states, 2)
        self.assertTrue(fixed["coverage_passed"])
        self.assertEqual(fixed["states"][2]["required_cache_depth"], 11)
        self.assertEqual(fixed["states"][2]["max_rank_query"], 0)

    def test_count_oracle_and_worst_ties(self):
        rng = random.Random(771)
        rows = [rng.sample(range(100), 25) for _ in range(37)]
        write_gt(self.gt, rows)
        report = self.check(worst_queries=5)
        for state, result in zip(self.states, report["states"]):
            counts = [sum(state.contains(i) for i in row) for row in rows]
            self.assertEqual(result["min_survivors"], min(counts))
            self.assertEqual(result["underflow_queries"], sum(c < 2 for c in counts))
            worst = sorted((count, query) for query, count in enumerate(counts))[:5]
            self.assertEqual(result["worst_queries"], [{"query": q, "survivors": c} for c, q in worst])
        write_gt(self.gt, [list(range(10))] * 9)
        result = self.check(worst_queries=3)
        self.assertEqual([q["query"] for q in result["states"][0]["worst_queries"]], [0, 1, 2])

    def test_equivalent_deletion_and_extra_rounds(self):
        states = coverage.make_states(100, 90, 10, 10, 10,
                                      insert_round_cap=2, delete_round_cap=2)
        final_prefix = coverage.State("prefix", 80)
        final_mask = coverage.State("mask", 90, 80, 90)
        for i in range(100):
            self.assertEqual(final_prefix.contains(i), final_mask.contains(i))
        self.assertEqual(next(s for s in states if s.name == "insert_round_1").prefix, 92)
        self.assertEqual(next(s for s in states if s.name == "delete_round_1"),
                         coverage.State("delete_round_1", 90, 80, 82))
        self.assertEqual(next(s for s in states if s.name == "delete_round_5").count(list(range(100))), 80)

    def test_custom_mixed_caps(self):
        states = coverage.make_states(100, 90, 5, 8, 10, mixed_insert_cap=1, mixed_delete_cap=2)
        self.assertEqual(states[-1], coverage.State("mixed_round_10", 100, 70, 90))
        self.assertEqual(states[1].prefix, 95)
        self.assertEqual(states[2].prefix, 82)

    def test_both_formats_and_small_depth(self):
        for tags in (False, True):
            write_gt(self.gt, [[1, 2, 3]], tags=tags)
            result = self.check(expected_queries=1)
            self.assertEqual(result["with_identity_tags"], tags)
            self.assertTrue(result["coverage_passed"])
        result = self.check(k=4)
        self.assertFalse(result["coverage_passed"])
        self.assertTrue(all(s["underflow_queries"] == 1 for s in result["states"]))

    def test_cli_evidence_and_exit_codes(self):
        write_gt(self.gt, [[80, 81, 82, 1]])
        output = self.root / "underflow.json"
        result = self.cli(["--json-output", str(output), "--expected-queries", "1"])
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("delete:", result.stdout)
        self.assertFalse(json.loads(output.read_text())["coverage_passed"])
        saved = output.read_bytes()
        result = self.cli(["--json-output", str(output)])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(output.read_bytes(), saved)
        result = self.cli(["--expected-queries", "2"])
        self.assertEqual(result.returncode, 1)
        write_gt(self.gt, [[1, 2, 3, 4]])
        self.assertEqual(self.cli().returncode, 0)
        self.assertEqual(self.cli(["--mixed-rounds", "0"]).returncode, 2)

    def test_malformed_cache(self):
        for name, rows, distances, error in (
            ("duplicate", [[1, 1, 2]], None, "duplicate GT ID at query 0"),
            ("out_of_range", [[1, 2, 100]], None, "GT ID outside full source at query 0"),
            ("nan", [[1, 2, 3]], [0.0, float("nan"), 2.0], "invalid/unsorted GT distance at query 0"),
            ("infinite", [[1, 2, 3]], [0.0, 1.0, float("inf")], "invalid/unsorted GT distance at query 0"),
            ("negative", [[1, 2, 3]], [-1.0, 1.0, 2.0], "invalid/unsorted GT distance at query 0"),
            ("unsorted", [[1, 2, 3]], [1.0, 0.0, 2.0], "invalid/unsorted GT distance at query 0"),
        ):
            with self.subTest(name=name):
                write_gt(self.gt, rows, distances=distances)
                with self.assertRaisesRegex(ValueError, error):
                    self.check()
        write_gt(self.gt, [[1, 2, 3]])
        bad = bytearray(self.gt.read_bytes())
        bad[-4:] = struct.pack("<I", 99)
        self.gt.write_bytes(bad)
        with self.assertRaisesRegex(ValueError, "tags"):
            self.check()
        for raw in (b"", struct.pack("<II", 0, 10), struct.pack("<II", 1, 0),
                    struct.pack("<II", 1, 3) + b"x" * 100):
            self.gt.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.check()

    def test_invalid_plans(self):
        for args, kw in (
            ((100, 100, 10, 10, 10), {}), ((100, 90, 11, 10, 10), {}),
            ((100, 90, 10, 90, 10), {}), ((100, 90, 5, 10, 10), {}),
            ((100, 90, 10, 10, 10), {"mixed_insert_cap": 2}),
            ((100, 90, 10, 10, 10), {"mixed_delete_cap": 9}),
            ((100, 90, 10, 10, 10), {"delete_round_cap": 3}),
            ((100, 90, 10, 10, 4097), {}),
        ):
            with self.subTest(args=args, kw=kw), self.assertRaises(ValueError):
                coverage.make_states(*args, **kw)

    def test_cache_change_detection(self):
        write_gt(self.gt, [[1, 2, 3]])
        original = coverage.sha256
        calls = 0

        def mutate(file):
            nonlocal calls
            calls += 1
            if calls == 2:
                raw = bytearray(self.gt.read_bytes())
                raw[8:12] = struct.pack("<I", 4)
                self.gt.write_bytes(raw)
            return original(file)

        with patch.object(coverage, "sha256", mutate), self.assertRaisesRegex(ValueError, "changed"):
            self.check()

    def test_blocked_sift_like_underflow_at_1662(self):
        # Synthetic IDs reproduce the observed block counts, not real SIFT GT.
        rows = [list(range(100)) for _ in range(2000)]
        rows[1662] = [34_931_164, 59_286_694, 51_896_615] + list(range(80_000_000, 80_000_092)) + list(range(90_000_000, 90_000_005))
        write_gt(self.gt, rows)
        states = coverage.make_states(100_000_000, 90_000_000, 10_000_000, 10_000_000, 10)
        report = coverage.check_cache(self.gt, 100_000_000, states, 10, expected_queries=2000)
        self.assertEqual(report["states"][0]["min_survivors"], 95)
        self.assertEqual(report["states"][1]["min_survivors"], 100)
        self.assertEqual(report["states"][2]["min_survivors"], 3)
        self.assertEqual(report["states"][2]["worst_queries"][0], {"query": 1662, "survivors": 3})
        self.assertEqual(report["states"][2]["underflow_queries"], 1)
        self.assertTrue(all(s["underflow_queries"] == 1 for s in report["states"][3:]))
        deep = self.root / "gt1000.bin"
        deeper = [list(range(1000)) for _ in rows]
        deeper[1662] = rows[1662] + list(range(100, 1000))
        write_gt(deep, deeper)
        fixed = coverage.check_cache(deep, 100_000_000, states, 10)
        self.assertTrue(fixed["coverage_passed"])
        self.assertEqual(fixed["states"][2]["required_cache_depth"], 107)

    def test_path_replacement_detection(self):
        write_gt(self.gt, [[1, 2, 3]])
        replacement = self.root / "replacement.bin"
        replacement.write_bytes(self.gt.read_bytes())
        original = coverage.sha256
        calls = 0

        def replace(file):
            nonlocal calls
            calls += 1
            if calls == 2:
                replacement.replace(self.gt)
            return original(file)

        # Windows may reject replacement of an open file; this is also safe.
        with patch.object(coverage, "sha256", replace), self.assertRaises((ValueError, OSError)):
            self.check()


if __name__ == "__main__":
    unittest.main()
