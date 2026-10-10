#!/usr/bin/env python3
"""Dependency-free contracts for shared IVF settings and evidence/process helpers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark/IVF"))
import common
import harness
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    SCRATCH = args.scratch
    unittest.main(argv=[sys.argv[0]])
