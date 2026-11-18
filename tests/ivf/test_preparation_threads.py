"""Preparation CPU defaults and child environments; no ANN/GT workloads."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark/IVF"))
import common
import settings

SCRATCH = None


class PreparationThreadTests(unittest.TestCase):
    def test_available_logical_processors_respect_affinity(self):
        with patch.object(os, "sched_getaffinity", return_value={2, 4, 6, 8}, create=True), \
             patch.object(os, "cpu_count", return_value=128):
            self.assertEqual(settings.logical_processors(), 4)

    def test_cpu_count_fallback_and_unknown_cpu_count(self):
        for error in (AttributeError, OSError):
            with patch.object(os, "sched_getaffinity", side_effect=error, create=True), \
                 patch.object(os, "cpu_count", return_value=12):
                self.assertEqual(settings.logical_processors(), 12)
            with patch.object(os, "sched_getaffinity", side_effect=error, create=True), \
                 patch.object(os, "cpu_count", return_value=None):
                self.assertEqual(settings.logical_processors(), 1)

    def test_build_settings_resolve_default_at_call_time(self):
        with patch.object(settings, "logical_processors", return_value=24):
            self.assertEqual(settings.build_settings({})["build_threads"], "24")
            self.assertEqual(settings.build_settings({"IVF_BUILD_THREADS": ""})["build_threads"], "24")
            self.assertEqual(settings.build_settings({"IVF_BUILD_THREADS": "6"})["build_threads"], "6")

    def test_invalid_worker_override_fails(self):
        for value in ("0", "-1", "1.5", "many"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "IVF_BUILD_THREADS"):
                settings.build_settings({"IVF_BUILD_THREADS": value})

    def test_preparation_environment_does_not_mutate_parent(self):
        with patch.dict(os.environ, {"MKL_NUM_THREADS": "1", "QUERY_THREADS": "8", "NTHREADS": "16"}, clear=True), \
             patch.object(settings, "logical_processors", return_value=24):
            before = dict(os.environ)
            env = settings.preparation_environment()
            self.assertEqual(env["IVF_BUILD_THREADS"], "24")
            self.assertEqual(env["OMP_NUM_THREADS"], "24")
            self.assertEqual(env["OMP_THREAD_LIMIT"], "24")
            self.assertEqual(env["MKL_NUM_THREADS"], "1")
            self.assertEqual(env["QUERY_THREADS"], "8")
            self.assertEqual(env["NTHREADS"], "16")
            self.assertEqual(dict(os.environ), before)

    def test_explicit_build_and_openmp_overrides_are_preserved(self):
        original = dict(IVF_BUILD_THREADS="6", OMP_NUM_THREADS="3", OMP_THREAD_LIMIT="4")
        self.assertEqual(settings.preparation_environment(original), original)
        env = settings.preparation_environment({"IVF_BUILD_THREADS": "6"})
        self.assertEqual(env["OMP_NUM_THREADS"], "6")
        self.assertEqual(env["OMP_THREAD_LIMIT"], "6")

    def test_command_log_passes_preparation_environment_to_child(self):
        env = settings.preparation_environment(dict(os.environ, IVF_BUILD_THREADS="6",
                                                   OMP_NUM_THREADS="6", OMP_THREAD_LIMIT="6"))
        command = [sys.executable, "-c",
                   "import json, os; print(json.dumps({k: os.environ[k] for k in "
                   "('IVF_BUILD_THREADS', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT')}))"]
        with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="prepare-threads-") as folder:
            output = io.StringIO()
            with redirect_stdout(output):
                common.command_log(command, Path(folder) / "child.log", env=env)
            self.assertEqual(json.loads(output.getvalue()),
                             dict(IVF_BUILD_THREADS="6", OMP_NUM_THREADS="6", OMP_THREAD_LIMIT="6"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    SCRATCH = args.scratch
    unittest.main(argv=[sys.argv[0]])
