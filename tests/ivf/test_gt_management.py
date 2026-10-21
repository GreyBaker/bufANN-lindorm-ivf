"""Synthetic one-command GT lifecycle tests; never run real generator/wrappers."""
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark/scripts"))
import ensure_gt as manager

try:
    import numpy
except ImportError:
    numpy = None


def vectors(path, values):
    path.write_bytes(struct.pack("<II", len(values), 1) + bytes(values))


def cache(path, base, k):
    best = sorted((x * x, i) for i, x in enumerate(base))[:k]
    ids = struct.pack(f"<{k}I", *(i for d, i in best))
    distances = struct.pack(f"<{k}f", *(d for d, i in best))
    path.write_bytes(struct.pack("<II", 1, k) + ids + distances + ids)


class ManagementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gt manager ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.query = self.root / "base.bin", self.root / "query.bin"
        self.base = list(range(100, 180)) + list(range(20))
        self.base[1:3] = [20, 21]
        vectors(self.source, self.base)
        vectors(self.query, [0])
        self.gt = self.root / "full-query.gt10.bin"
        self.generator = self.root / "fake-generator.py"
        # Deliberately stdlib-only integer-oracle producer, independent of NumPy.
        self.generator.write_text('''import json, os, struct, sys
from pathlib import Path
a=sys.argv[1:]
def arg(key): return a[a.index(key)+1]
p=Path(arg("--gt_file"))
assert not p.exists()
base=list(Path(arg("--base_file")).read_bytes()[8:])
k=int(arg("--K"))
mode=os.environ.get("FAKE_MODE", "good")
if mode == "partial":
    p.write_bytes(b"partial")
    print("deliberate generation failure")
    sys.exit(7)
if mode == "malformed":
    p.write_bytes(b"malformed")
    sys.exit(0)
order=sorted((v*v,i) for i,v in enumerate(base))[:k]
if mode == "underflow": order=[(i*i,80+i) for i in range(k)]
ids=struct.pack(f"<{k}I", *(i for d,i in order))
dist=struct.pack(f"<{k}f", *(d for d,i in order))
p.write_bytes(struct.pack("<II",1,k)+ids+dist+ids)
if mode == "change-source":
    path=Path(arg("--base_file"))
    raw=bytearray(path.read_bytes()); raw[-1]^=1; path.write_bytes(raw)
print("generator stdout must not contaminate the returned path")
''')
        self.env = patch.dict(os.environ, {"GT_BIN_KCACHED": "", "IVF_BUILD_DIR": "", "FAKE_MODE": "good"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def args(self, extra=()):
        return ["--base-file", str(self.source), "--query-file", str(self.query),
                "--cache-dir", str(self.root), "--full-count", "100", "--base-count", "90",
                "--k-out", "2", "--expected-queries", "1", "--chunk-rows", "13",
                "--gt-generator", str(self.generator), *extra]

    def run_manager(self, extra=()):
        stdout, stderr = io.StringIO(), io.StringIO()
        original = subprocess.run
        calls = []

        def fake_native(command, **kwargs):
            calls.append(command)
            return original([sys.executable, *command], **kwargs)

        with patch.object(manager.subprocess, "run", fake_native), redirect_stdout(stdout), redirect_stderr(stderr):
            code = manager.main(self.args(extra))
        return code, stdout.getvalue(), stderr.getvalue(), calls

    def latest(self):
        return sorted((self.root / ".gt").glob("run-*"))[-1]

    def test_covered_cache_is_rechecked_and_reused_without_scans(self):
        cache(self.gt, self.base, 100)
        before = self.gt.read_bytes()
        with patch.object(manager.depth, "recommend", side_effect=AssertionError("no recommendation needed")):
            for unused in range(2):
                code, stdout, stderr, calls = self.run_manager(["--generate"])
                self.assertEqual(code, 0, stderr)
                self.assertEqual(stdout, str(self.gt.resolve()) + "\n")
                self.assertEqual(calls, [])
        self.assertEqual(self.gt.read_bytes(), before)
        self.assertEqual(len(list((self.root / ".gt").glob("run-*/result.json"))), 2)

    def test_missing_export_discovers_legacy_cache(self):
        cache(self.gt, self.base, 100)
        with patch.dict(os.environ, GT_BIN_KCACHED=str(self.root / "full-query.gt-old-stale.bin")):
            code, stdout, stderr, calls = self.run_manager()
        self.assertEqual(code, 0, stderr)
        self.assertIn("Ignoring missing cache", stderr)
        self.assertEqual(stdout.strip(), str(self.gt.resolve()))
        self.assertEqual(calls, [])

    def test_existing_explicit_custom_cache_has_priority(self):
        cache(self.gt, self.base, 100)
        custom = self.root / "custom cache.bin"
        cache(custom, self.base, 100)
        code, stdout, stderr, calls = self.run_manager(["--cached-gt", str(custom)])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stdout.strip(), str(custom.resolve()))

    def test_no_cache_without_consent_returns_no_path(self):
        code, stdout, stderr, calls = self.run_manager()
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("--generate", stderr)
        self.assertEqual(calls, [])
        self.assertTrue((self.latest() / "error.json").is_file())

    def test_no_cache_generates_then_reuses_success(self):
        code, stdout, stderr, calls = self.run_manager(["--generate", "--initial-k", "30"])
        self.assertEqual(code, 0, stderr)
        selected = Path(stdout.strip())
        self.assertEqual(selected.name, "full-query.gt30.bin")
        self.assertEqual(len(calls), 1)
        self.assertTrue((self.latest() / "result.json").is_file())
        self.assertNotIn("generator stdout", stdout)
        second = self.run_manager(["--generate"])
        self.assertEqual(second[0], 0, second[2])
        self.assertEqual(second[1], stdout)
        self.assertEqual(second[3], [])

    @unittest.skipUnless(numpy is not None, "recommendation scans require NumPy")
    def test_underflow_recommends_repairs_and_preserves_original(self):
        cache(self.gt, self.base, 10)
        before = self.gt.read_bytes()
        code, stdout, stderr, calls = self.run_manager(["--generate"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(Path(stdout.strip()).name, "full-query.gt22.bin")
        self.assertEqual(self.gt.read_bytes(), before)
        recommendation = json.loads((self.latest() / "recommendation.json").read_text())
        self.assertEqual(recommendation["recommended_k"], 22)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index("--K") + 1], "22")

    @unittest.skipUnless(numpy is not None, "recommendation scans require NumPy")
    def test_check_only_reports_advice_without_generation(self):
        cache(self.gt, self.base, 10)
        code, stdout, stderr, calls = self.run_manager()
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout, "")
        self.assertEqual(calls, [])
        self.assertEqual(json.loads((self.latest() / "recommendation.json").read_text())["recommended_k"], 22)

    @unittest.skipUnless(numpy is not None, "recommendation scans require NumPy")
    def test_bootstrap_underflow_is_repaired_once(self):
        code, stdout, stderr, calls = self.run_manager(["--generate", "--initial-k", "10"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(calls), 2)
        self.assertEqual(Path(stdout.strip()).name, "full-query.gt22.bin")

    @unittest.skipUnless(numpy is not None, "recommendation scans require NumPy")
    def test_completed_bootstrap_is_not_regenerated_after_admission_failure(self):
        first = self.run_manager(["--generate", "--initial-k", "10", "--memory-budget-mb", "1"])
        self.assertEqual(first[0], 1, first[2])
        self.assertEqual(first[1], "")
        self.assertEqual(len(first[3]), 1)
        self.assertTrue((self.latest() / "generated-k10.json").is_file())
        second = self.run_manager(["--generate", "--initial-k", "10"])
        self.assertEqual(second[0], 0, second[2])
        self.assertEqual(len(second[3]), 1)  # Only K=22 repair, not K=10 bootstrap again.
        self.assertEqual(Path(second[1].strip()).name, "full-query.gt22.bin")

    @unittest.skipUnless(numpy is not None, "recommendation scans require NumPy")
    def test_final_new_cache_underflow_returns_no_path(self):
        cache(self.gt, self.base, 10)
        original = manager.generate

        def shallow(*args):
            path = original(*args)
            cache(path, self.base, 10)  # Producer unexpectedly outputs too little GT.
            return path

        with patch.object(manager, "generate", shallow):
            code, stdout, stderr, calls = self.run_manager(["--generate"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("new GT still underflows", stderr)
        self.assertFalse((self.latest() / "result.json").exists())

    def test_generation_failures_preserve_partial_evidence(self):
        for mode in ("partial", "malformed", "underflow", "change-source"):
            with self.subTest(mode=mode), patch.dict(os.environ, FAKE_MODE=mode):
                code, stdout, stderr, calls = self.run_manager(["--generate", "--initial-k", "30"])
                self.assertEqual(code, 1, stderr)
                self.assertEqual(stdout, "")
                self.assertTrue((self.latest() / "error.json").is_file())
                self.assertFalse((self.latest() / "result.json").exists())
                self.assertTrue((self.latest() / "generate-k30.log").exists())
        # None of the failed run outputs is selected on retry.
        vectors(self.source, self.base)
        result = self.run_manager(["--generate", "--initial-k", "30"])
        self.assertEqual(result[0], 0, result[2])
        self.assertEqual(len(result[3]), 1)

    def test_malformed_explicit_cache_cannot_silently_fall_back(self):
        cache(self.gt, self.base, 100)
        invalid = self.root / "broken.bin"
        invalid.write_bytes(b"broken")
        code, stdout, stderr, calls = self.run_manager(["--generate", "--cached-gt", str(invalid)])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(calls, [])
        self.assertEqual(invalid.read_bytes(), b"broken")

    def test_invalid_plan_shape_and_generator_fail_before_work(self):
        for extra in (["--base-count", "100"], ["--expected-queries", "2"],
                      ["--generate", "--gt-generator", str(self.root / "missing")]):
            with self.subTest(extra=extra):
                code, stdout, stderr, calls = self.run_manager(extra)
                self.assertEqual(code, 1)
                self.assertEqual(stdout, "")
                self.assertEqual(calls, [])

    def test_invalid_plan_is_rejected_before_evidence_directory(self):
        for extra, error in ((["--base-count", "100"], "require 0 < base"),
                             (["--insert-count", "5"], "mixed inserts: count must divide rounds"),
                             (["--k-out", "81"], "K exceeds a declared live population")):
            with self.subTest(extra=extra):
                code, stdout, stderr, calls = self.run_manager(extra)
                self.assertEqual(code, 1)
                self.assertEqual(stdout, "")
                self.assertEqual(calls, [])
                self.assertIn(error, stderr)
                self.assertFalse((self.root / ".gt").exists())

    def test_generation_scratch_admission_boundary_precedes_generator_lookup(self):
        args = manager.parser().parse_args(self.args())
        args.gt_generator = None
        for full_count, k, error in ((100, 101, "GT depth exceeds preflight admission"),
                                     (600000, 524289, "GT depth exceeds preflight admission"),
                                     (600000, 524288, "GT generator not found")):
            args.full_count = full_count
            with self.subTest(full_count=full_count, k=k), \
                    patch.object(manager.subprocess, "run", side_effect=AssertionError("must reject before generation")), \
                    self.assertRaisesRegex(ValueError, error):
                manager.generate(args, self.root / "uncreated", k)
        self.assertFalse((self.root / "uncreated").exists())

    def test_input_changes_prevent_managed_cache_reuse(self):
        first = self.run_manager(["--generate", "--initial-k", "30"])
        self.assertEqual(first[0], 0, first[2])
        # Different query content despite identical framing: previous run is not eligible.
        vectors(self.query, [1])
        code, stdout, stderr, calls = self.run_manager()
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")

    def test_current_plan_is_rechecked_not_taken_from_previous_result(self):
        first = self.run_manager(["--generate", "--initial-k", "30"])
        self.assertEqual(first[0], 0, first[2])
        code, stdout, stderr, calls = self.run_manager(["--k-out", "20", "--max-scan-queries", "1"])
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        reports = list(self.latest().glob("coverage-*.json"))
        self.assertEqual(len(reports), 1)
        self.assertFalse(json.loads(reports[0].read_text())["coverage_passed"])

    def test_numbered_attempt_reservations_do_not_collide(self):
        directory = self.root / ".gt"
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda unused: manager.new_attempt(directory), range(8)))
        self.assertEqual(len(set(paths)), 8)
        self.assertTrue(all(p.is_dir() for p in paths))

    def test_interrupted_record_write_does_not_poison_retry(self):
        attempt = manager.new_attempt(self.root / ".gt")
        (attempt / "plan.json.partial").write_text('{"inputs":')
        code, stdout, stderr, calls = self.run_manager(["--generate", "--initial-k", "30"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual((attempt / "plan.json.partial").read_text(), '{"inputs":')

    def test_record_publication_never_overwrites_evidence(self):
        record = self.root / "evidence.json"
        record.write_text("retained")
        with self.assertRaises(FileExistsError):
            manager.save(record, {"new": "attempt"})
        self.assertEqual(record.read_text(), "retained")

    def test_cli_stdout_is_exactly_a_path_or_empty_on_failure(self):
        cache(self.gt, self.base, 100)
        command = [sys.executable, "-B", str(REPO / "benchmark/scripts/ensure_gt.py"), *self.args()]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(self.gt.resolve()) + "\n")
        self.gt.write_bytes(b"bad")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
