#!/usr/bin/env python3
"""Dependency-free contracts for shared IVF settings and evidence/process helpers."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark/IVF"))
import common
import harness
import mixed
import settings

SCRATCH = None


class SharedOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="shared_helpers_", dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_legacy_build_settings_and_query_override_isolation(self):
        expected = {
            "nlist": "8",
            "sample_rate": "1",
            "max_reps": "15",
            "seed": "0",
            "build_threads": "1",
            "memory_budget_gb": "1",
            "page_size": "4096",
            "buffer_frames": "64",
            "router_degree": "16",
            "construction_breadth": "200",
            "query_breadth": "128",
            "insert_breadth": "64",
            "router_seed": "100",
            "optimized_modes": "0",
            "refine_ef": "128",
            "pca_variance": "0.95",
            "routing_work_limit": "1000000000",
            "trainer_mode": "ann-reuse-seeded",
            "init": "random_partition",
            "training_breadth": "64",
            "router_mode": "hnswlib",
            "router_rerank_factor": "3",
        }
        with patch.dict(os.environ, {"IVF_NLIST": "8"}, clear=True):
            self.assertEqual(settings.build_settings(), expected)
            identity = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
            os.environ.update(
                IVF_ROUTER_MODE="pca", IVF_ROUTER_RERANK_FACTOR="5", IVF_ROUTER_BREADTH="128"
            )
            actual = settings.build_settings()
            self.assertEqual(actual, expected)
            self.assertEqual(
                hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(), identity
            )
            self.assertEqual(
                settings.query_router_flags(),
                ["--router_mode", "pca", "--router_rerank_factor", "5", "--router_breadth", "128"],
            )
            os.environ["IVF_TRAINER_MODE"] = "lv-plus"
            actual = settings.build_settings()
            self.assertEqual(actual["training_rerank_factor"], "3")
            self.assertEqual(actual["training_refine_breadth"], "128")
            os.environ.update(IVF_TRAINER_MODE="ann-reuse", IVF_TRAINING_RERANK_FACTOR="3")
            actual = settings.build_settings()
            self.assertEqual(actual["training_rerank_factor"], "3")
            self.assertNotIn("training_refine_breadth", actual)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            ValueError, "set IVF_NLIST"
        ):
            settings.build_settings()

    def test_historical_and_declared_report_policy(self):
        report = {
            "version": 1,
            "kind": "ivf_cost_model",
            "completed": True,
            "selected_nlist": 1,
            "sample_count": 32,
            "candidates": [
                {"nlist": 1, "eligible": True, "t1_us": 1, "t2_us": 2, "total_us": 3},
                {"nlist": 2, "eligible": False},
            ],
        }
        path = self.root / "report.json"
        path.write_text(json.dumps(report))
        self.assertEqual(settings.read_cost_model_report(path), report)
        report.update(
            version=2,
            min_sample_per_centroid=20,
            candidate_eligibility_policy="declared_project_minimum_not_paper_requirement",
            trainer_mode="lv-plus",
            training_search_policy="pca95_refit_each_pass_initial_refinement",
        )
        path.write_text(json.dumps(report))
        self.assertEqual(settings.read_cost_model_report(path), report)
        with patch.dict(
            os.environ,
            {"IVF_COST_MODEL_REPORT": str(path), "IVF_TRAINER_MODE": "lv-plus"},
            clear=True,
        ):
            self.assertEqual(settings.build_settings()["nlist"], "1")
            os.environ["IVF_NLIST"] = "2"
            with self.assertRaisesRegex(ValueError, "disagrees"):
                settings.build_settings()
        report["min_sample_per_centroid"] = 1
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "eligibility"):
            settings.read_cost_model_report(path)
        path.write_text('{"version":1,"version":1}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            settings.read_cost_model_report(path)

    def test_independent_insertion_worker_setting(self):
        with patch.dict(os.environ, {"NTHREADS": "20"}, clear=True):
            self.assertEqual(settings.insertion_threads(), "1")
            for value in ("1", "2", "8", "256"):
                os.environ["IVF_INSERT_THREADS"] = value
                self.assertEqual(settings.insertion_threads(), value)
            for value in ("0", "257", "-1", "1.5", "bad"):
                os.environ["IVF_INSERT_THREADS"] = value
                with self.assertRaises(ValueError):
                    settings.insertion_threads()
        common.write_json(self.root / "prepared.json", {"settings": {"page_size": "4096"}})
        args = argparse.Namespace(work_dir=self.root, workload="insert", output_dir=self.root / "runs")
        with patch.dict(os.environ, {"IVF_INSERT_THREADS": "0"}, clear=True), \
             patch.object(common.subprocess, "Popen") as launch, \
             self.assertRaisesRegex(ValueError, "IVF_INSERT_THREADS"):
            harness.run(args)
        launch.assert_not_called()
        self.assertFalse(args.output_dir.exists())

    def test_resource_lower_bound_and_unchanged_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.runtime_resources(4096),
                             {"buffer_frames": "64", "page_size": "4096", "memory_budget_gb": "1"})
            os.environ.update(IVF_BUFFER_FRAMES="262144", IVF_MEMORY_BUDGET_GB="96", IVF_NLIST="8")
            self.assertEqual(settings.build_settings()["buffer_frames"], "262144")
            self.assertEqual(settings.runtime_resources(4096)["memory_budget_gb"], "96")
            os.environ["IVF_QUERY_MEMORY_BUDGET_GB"] = "1"
            # Insufficient post-update query budget is rejected before mutation.
            with self.assertRaisesRegex(ValueError, "frame allowance alone"):
                settings.runtime_resources(4096, query=True)
            os.environ["IVF_MEMORY_BUDGET_GB"] = "1"
            with self.assertRaisesRegex(ValueError, "frame allowance alone"):
                settings.build_settings()
        for frames, page, budget in (("1", "4096", "1"), ("4294967296", "4096", "1"),
                                      ("64", "512", "1"), ("64", "6144", "1"),
                                      ("64", "4096", "nan"), ("64", "4096", "inf"),
                                      ("64", "4096", "0")):
            with self.subTest(frames=frames, page=page, budget=budget), self.assertRaises(ValueError):
                settings.validate_resources({"buffer_frames": frames, "page_size": page,
                                             "memory_budget_gb": budget})

    def test_runtime_resource_rejection_precedes_outputs_and_processes(self):
        common.write_json(self.root / "prepared.json", {"settings": {"page_size": "4096"}})
        args = argparse.Namespace(work_dir=self.root, workload="insert", output_dir=self.root / "runs")
        for run in (harness.run, mixed.run):
            with patch.dict(os.environ, {"IVF_BUFFER_FRAMES": "262144", "IVF_MEMORY_BUDGET_GB": "1"}, clear=True), \
                 patch.object(common.subprocess, "Popen") as launch, \
                 self.assertRaisesRegex(ValueError, "frame allowance alone"):
                run(args)
            launch.assert_not_called()
            self.assertFalse(args.output_dir.exists())
        with patch.dict(os.environ, {"IVF_BUFFER_FRAMES": "262144", "IVF_MEMORY_BUDGET_GB": "96",
                                     "IVF_QUERY_MEMORY_BUDGET_GB": "1"}, clear=True), \
             self.assertRaisesRegex(ValueError, "frame allowance alone"):
            harness.run(args)
        # Build preflight must not need data/PQ/tools to reject this pairing.
        with patch.dict(os.environ, {"IVF_NLIST": "8", "IVF_BUFFER_FRAMES": "262144",
                                     "IVF_MEMORY_BUDGET_GB": "1"}, clear=True), \
             self.assertRaisesRegex(ValueError, "frame allowance alone"):
            harness.prepare(args)

    def test_insertion_policy_and_staging_lower_bound(self):
        resources = {"buffer_frames": "64", "page_size": "4096", "memory_budget_gb": "1"}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.insertion_input_policy(), "streaming")
            os.environ["IVF_INSERT_INPUT_POLICY"] = "staged"
            self.assertEqual(settings.insertion_input_policy(), "staged")
            os.environ["IVF_INSERT_INPUT_POLICY"] = "unknown"
            with self.assertRaisesRegex(ValueError, "IVF_INSERT_INPUT_POLICY"):
                settings.insertion_input_policy()
        for dtype in ("float", "int8", "uint8"):
            settings.validate_insertion_staging(resources, 8192, 128, dtype)
        with self.assertRaisesRegex(ValueError, "staging and frame"):
            settings.validate_insertion_staging(resources, 10000000, 128, "float")
        common.write_json(self.root / "prepared.json", {"settings": {"page_size": "4096"}})
        args = argparse.Namespace(work_dir=self.root, workload="insert", output_dir=self.root / "runs",
                                  update_count=10000000, dim=128, dtype="float")
        for policy, expected in (("unknown", "IVF_INSERT_INPUT_POLICY"), ("staged", "staging and frame")):
            with patch.dict(os.environ, {"IVF_INSERT_INPUT_POLICY": policy}, clear=True), \
                 patch.object(common.subprocess, "Popen") as launch, self.assertRaisesRegex(ValueError, expected):
                harness.run(args)
            launch.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_independent_warmup_setting_and_preflight(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.warmup_queries(), "0")
            for value in ("0", "1", "100000"):
                os.environ["IVF_WARMUP_QUERIES"] = value
                self.assertEqual(settings.warmup_queries(), value)
            for value in ("-1", "100001", "1.5", "invalid"):
                os.environ["IVF_WARMUP_QUERIES"] = value
                with self.assertRaises(ValueError):
                    settings.warmup_queries()
        common.write_json(self.root / "prepared.json", {"settings": {"page_size": "4096"}})
        args = argparse.Namespace(work_dir=self.root, workload="query", output_dir=self.root / "runs")
        for execute in (harness.run, mixed.run):
            with patch.dict(os.environ, {"IVF_WARMUP_QUERIES": "100001"}, clear=True), \
                 patch.object(common.subprocess, "Popen") as launch, self.assertRaisesRegex(ValueError, "IVF_WARMUP_QUERIES"):
                execute(args)
            launch.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    @unittest.skipUnless(sys.platform == "linux", "quickstart gate executes in Bash")
    def test_quickstart_gt_gate_stops_preparation(self):
        text = (Path(__file__).resolve().parents[2] / "benchmark/IVF/README.md").read_text()
        marker = "# Each workload gets an independent prepared base; stop on any failure."
        block = text.split(marker, 1)[1].split("```", 1)[0]
        self.assertLess(block.index("GT_BIN_KCACHED=$(python3"), block.index("for workload in"))
        self.assertIn("export GT_BIN_KCACHED", block)
        self.assertNotIn("export GT_BIN_KCACHED=", block)
        self.assertIn("--memory-budget-gb 96 --buffer-frames 262144", text)
        env = dict(os.environ, DATA_BIN="/source.bin", QUERY_BIN="/queries.bin", NPTS_FULL="100000000",
                   NPTS_BASE="90000000", UPDATE_POINTS="10000000", UPDATE_BATCH="1000000", RECALL_AT="10")
        for status in (0, 3):
            script = ("python3() { printf '/covered-fixture.gt\\n'; return " + str(status) + "; }\n"
                      "bash() { printf 'workload %s gt=%s\\n' \"$1\" \"$GT_BIN_KCACHED\"; }\n" + block)
            result = subprocess.run(["bash", "-c", script], env=env, text=True,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertEqual(result.stdout.count("workload "), 8 if status == 0 else 0)
            if status == 0:
                self.assertEqual(result.stdout.count("gt=/covered-fixture.gt"), 8)

    def test_identity_framing_and_exclusive_evidence(self):
        vectors = self.root / "vectors.bin"
        vectors.write_bytes(struct.pack("<II2f", 1, 2, 1, 2))
        common.header(vectors, "float", 1, 2)
        truth = self.root / "truth.bin"
        truth.write_bytes(struct.pack("<IIIf", 1, 1, 0, 0))
        common.query_truth_shapes(vectors, truth, "float", 2, 1)
        self.assertEqual(
            common.digest(vectors),
            {
                "bytes": len(vectors.read_bytes()),
                "sha256": hashlib.sha256(vectors.read_bytes()).hexdigest(),
            },
        )
        with self.assertRaises(ValueError):
            common.header(vectors, "float", 2, 2)
        out = self.root / "record.json"
        common.write_json(out, {"complete": True})
        original = out.read_bytes()
        with self.assertRaises(FileExistsError):
            common.write_json(out, {})
        self.assertEqual(out.read_bytes(), original)
        self.assertEqual(common.read_json(out), {"complete": True})
        out.write_text(" " * 65537)
        with self.assertRaisesRegex(ValueError, "oversized"):
            common.read_json(out)

    @unittest.skipUnless(
        sys.platform == "linux", "experiment process groups and affinity target Linux/WSL"
    )
    def test_process_logs_failure_timeout_and_affinity(self):
        log = self.root / "success.log"
        self.assertGreater(
            common.execute([sys.executable, "-c", "print('success')"], log, 10, None), 0
        )
        original = log.read_bytes()
        with self.assertRaises(FileExistsError):
            common.execute([sys.executable, "-c", "print('overwrite')"], log, 10, None)
        self.assertEqual(log.read_bytes(), original)
        failed = self.root / "failure.log"
        with self.assertRaisesRegex(ValueError, "command failed"):
            common.execute(
                [sys.executable, "-c", "print('failure'); raise SystemExit(7)"], failed, 10, None
            )
        self.assertIn("failure", failed.read_text())
        timed = self.root / "timeout.log"
        with self.assertRaisesRegex(ValueError, "timed out"):
            common.execute([sys.executable, "-c", "import time; time.sleep(60)"], timed, 0.1, None)
        self.assertTrue(timed.is_file())
        with patch.object(common.os, "sched_getaffinity", return_value={2, 4}):
            self.assertEqual(common.affinity("4")["selected"], [4])
            with self.assertRaisesRegex(ValueError, "outside"):
                common.affinity("1")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            common.values("2,2", int, 0)

    def test_compatibility_exports_and_shared_provenance(self):
        for name in (
            "digest",
            "flags",
            "header",
            "require",
            "read_json",
            "write_json",
            "command_log",
        ):
            self.assertIs(getattr(harness, name), getattr(common, name))
        for name in (
            "build_settings",
            "read_cost_model_report",
            "nlist_selection",
            "prepared_selection",
            "query_router_flags",
        ):
            self.assertIs(getattr(harness, name), getattr(settings, name))
        self.assertEqual(set(common.shared_module_paths()), {"python_common", "settings"})
        for path in common.shared_module_paths().values():
            self.assertTrue(path.is_file())
        self.assertEqual(settings.CALIBRATION_DEFAULTS["pca_variance"], ".95")
        self.assertEqual(settings.CALIBRATION_DEFAULTS["rerank_factor"], "4")

    def test_command_log_flushes_progress_and_preserves_failure(self):
        class Process:
            returncode = 0
            def __init__(self):
                self.stdout = iter(["[IVF driver] phase=insert event=begin\n",
                                    "[IVF driver] phase=insert event=running completed=1024\n"])
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                return False
        class Flushed(io.StringIO):
            count = 0
            def flush(self):
                self.count += 1
        output = Flushed()
        log = self.root / "progress.log"
        process = Process()
        def lines():
            yield "[IVF driver] phase=insert event=begin\n"
            self.assertIn("event=begin", log.read_text())
            yield "[IVF driver] phase=insert event=running completed=1024\n"
            self.assertIn("completed=1024", log.read_text())
        process.stdout = lines()
        with patch.object(common.subprocess, "Popen", return_value=process), \
             patch.object(common.sys, "stdout", output):
            common.command_log(["fake-native-driver"], log)
        self.assertEqual(output.count, 2)
        self.assertIn("event=running completed=1024", log.read_text())
        before = log.read_bytes()
        with self.assertRaises(FileExistsError):
            common.command_log(["fake-native-driver"], log)
        self.assertEqual(log.read_bytes(), before)
        process = Process()
        process.returncode = 7
        with patch.object(common.subprocess, "Popen", return_value=process), \
             patch.object(common.sys, "stdout", Flushed()), \
             self.assertRaisesRegex(ValueError, "command failed"):
            common.command_log(["fake-native-driver"], self.root / "failed-progress.log")
        self.assertIn("event=running", (self.root / "failed-progress.log").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    SCRATCH = args.scratch
    unittest.main(argv=[sys.argv[0]])
