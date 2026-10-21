#!/usr/bin/env python3
"""Pure stdlib contracts for test support; no native tools or benchmark imports."""

import builtins
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import ivf_test_support as support


def legacy_pivots():
    """Independent byte reference for the original synthetic PQ artifact."""
    return (struct.pack("<II5Q", 5, 1, 4096, 7176, 7196, 7216, 7236)
            + bytes(4096 - 48)
            + struct.pack("<II", 256, 3) + bytes(3072)
            + struct.pack("<II3f", 3, 1, 1, 2, 3)
            + struct.pack("<II3I", 3, 1, 0, 1, 2)
            + struct.pack("<II3I", 3, 1, 0, 1, 3))


class FileContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ivf_test_support_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "fixture.bin"

    def test_typed_matrix_framing(self):
        for fmt, rows in (("f", [(1.5, -2, 3), (4, 5, 6)]),
                          ("b", [(-128, -3, 127), (4, 5, 6)]),
                          ("B", [(128, 200, 255), (4, 5, 6)]),
                          ("I", [(0, 0xFFFFFFFF, 7), (4, 5, 6)])):
            with self.subTest(fmt=fmt):
                support.write_matrix(self.path, rows, fmt)
                expected = struct.pack("<II6" + fmt, 2, 3, *rows[0], *rows[1])
                self.assertEqual(self.path.read_bytes(), expected)
                self.assertEqual(support.read_matrix(self.path, fmt), rows)
        support.write_matrix(self.path, [bytes(2)] * 4, "B")
        self.assertEqual(self.path.read_bytes(), struct.pack("<II", 4, 2) + bytes(8))
        support.write_matrix(self.path, [], "f", width=3)
        self.assertEqual(self.path.read_bytes(), struct.pack("<II", 0, 3))

    def test_counted_ids_are_not_matrices(self):
        support.write_ids(self.path, iter([7, 0xFFFFFFFF, 7]))
        self.assertEqual(self.path.read_bytes(), struct.pack("<4I", 3, 7, 0xFFFFFFFF, 7))
        support.write_ids(self.path, [])
        self.assertEqual(self.path.read_bytes(), struct.pack("<I", 0))

    def test_truth_planes_and_optional_tags(self):
        rows = [[(7, 1.5), (3, 4.0)], [(2, 0.0), (9, 8.0)]]
        expected = struct.pack("<II4I4f", 2, 2, 7, 3, 2, 9, 1.5, 4, 0, 8)
        support.write_truth(self.path, rows)
        self.assertEqual(self.path.read_bytes(), expected)
        support.write_truth(self.path, rows, tags=[[7, 3], [2, 9]])
        self.assertEqual(self.path.read_bytes(), expected + struct.pack("<4I", 7, 3, 2, 9))

    def test_writers_do_not_validate_away_bad_fixtures(self):
        support.write_truth(self.path, [[(5, math.nan), (5, -1)]], tags=[[99]])
        self.assertEqual(struct.unpack_from("<II2I", self.path.read_bytes()), (1, 2, 5, 5))
        self.assertTrue(math.isnan(struct.unpack_from("<f", self.path.read_bytes(), 16)[0]))
        self.assertEqual(self.path.read_bytes()[-8:], struct.pack("<fI", -1, 99))
        support.write_matrix(self.path, [[math.nan, 2], [3]], "f")
        self.assertEqual(struct.unpack_from("<II", self.path.read_bytes()), (2, 2))
        self.assertEqual(len(self.path.read_bytes()), 20)  # Deliberately short/ragged.

    def test_results_are_interleaved_and_exact_length(self):
        raw = struct.pack("<IIIfIfIfIf", 2, 2, 7, 1.5, 3, 4, 9, 0, 0xFFFFFFFF, math.inf)
        self.path.write_bytes(raw)
        self.assertEqual(support.read_results(self.path),
                         [[(7, 1.5), (3, 4)], [(9, 0), (0xFFFFFFFF, math.inf)]])
        for malformed in (raw[:-1], raw + b"\0", struct.pack("<II", 3, 2) + raw[8:]):
            with self.subTest(raw=malformed):
                self.path.write_bytes(malformed)
                with self.assertRaises(AssertionError):
                    support.read_results(self.path)
        self.path.write_bytes(b"\0" * 7)
        with self.assertRaises(struct.error):
            support.read_results(self.path)
        self.path.write_bytes(struct.pack("<II", 2, 0))
        self.assertEqual(support.read_results(self.path), [[], []])

    def test_matrix_reader_length_contract(self):
        raw = struct.pack("<II2I", 2, 1, 17, 23)
        for malformed, error in ((raw[:-1], struct.error), (raw + b"\0", AssertionError),
                                 (raw[:7], struct.error)):
            self.path.write_bytes(malformed)
            with self.assertRaises(error):
                support.read_matrix(self.path, "I")

    def test_pivot_bytes_and_4096_offsets(self):
        support.write_pivots(self.path)
        self.assertEqual(self.path.read_bytes(), legacy_pivots())
        self.assertEqual(len(self.path.read_bytes()), 7236)

    def test_reservation_preserves_existing_evidence(self):
        first = support.reserve_attempt(self.root)
        evidence = first / "evidence"
        evidence.write_bytes(b"owned")
        (self.root / "run-0002").write_bytes(b"also owned")
        self.assertEqual(support.reserve_attempt(self.root), self.root / "run-0003")
        self.assertEqual(evidence.read_bytes(), b"owned")
        self.assertEqual((self.root / "run-0002").read_bytes(), b"also owned")

    def test_reservation_exhaustion_contract(self):
        with patch.object(Path, "mkdir", side_effect=[None] + [FileExistsError()] * 10000):
            with self.assertRaisesRegex(RuntimeError, "parallel insertion fixture attempts exhausted"):
                support.reserve_attempt(self.root)

    def test_population_fixture_matches_original_all_dtypes_and_modes(self):
        for name, reverse, empty in (("parallel", True, True), ("policy", True, True),
                                     ("warmup", False, False)):
            for dtype, fmt in (("float", "f"), ("int8", "b"), ("uint8", "B")):
                with self.subTest(test=name, dtype=dtype):
                    root = self.root / (name + dtype)
                    root.mkdir()
                    work = root / dtype
                    data, query, prefix = work / "source.bin", work / "queries.bin", work / "base"
                    pq = work / "pq"
                    vectors = [tuple((i * 7 + d * 3) % 31 - (16 if dtype == "int8" else 0)
                                     for d in range(3)) for i in range(64)]
                    queries = vectors[::8]
                    data_bytes = struct.pack("<II192" + fmt, 64, 3, *(v for r in vectors for v in r))
                    query_bytes = struct.pack("<II24" + fmt, 8, 3, *(v for r in queries for v in r))
                    expected_build = ["build", "--validate_inputs", "0", "--data_type", dtype,
                        "--data_file", str(data), "--pq_prefix", str(pq), "--index_prefix", str(prefix),
                        "--dim", "3", "--base_count", "8", "--nlist", "2", "--max_reps", "2",
                        "--build_threads", "2", "--query_breadth", "2", "--insert_breadth", "2"]

                    def build(command, phase):
                        self.assertEqual(command, expected_build)
                        self.assertEqual(phase, "build")
                        self.assertEqual(data.read_bytes(), data_bytes)
                        self.assertEqual(query.read_bytes(), query_bytes)
                        self.assertEqual(Path(str(pq) + "_pq_pivots.bin").read_bytes(), legacy_pivots())
                        self.assertEqual(Path(str(pq) + "_pq_compressed.bin").read_bytes(),
                                         struct.pack("<II", 64, 2) + bytes(128))
                        self.assertFalse((work / "insert.ids").exists())
                        Path(str(prefix) + ".manifest").write_bytes(b"native manifest stand-in")
                        Path(str(prefix) + ".heap").write_bytes(b"native heap stand-in")

                    runner = Mock(side_effect=build)
                    fixture = support.build_tiny_population("build", root, dtype, fmt,
                        reverse_insert_ids=reverse, include_empty_ids=empty, run_build=runner)
                    runner.assert_called_once_with(expected_build, "build")
                    self.assertEqual(fixture.build, expected_build)
                    self.assertEqual(fixture.vectors, vectors)
                    self.assertEqual(fixture.queries, queries)
                    self.assertEqual((fixture.work, fixture.data, fixture.query, fixture.prefix),
                                     (work, data, query, prefix))
                    tail = list(range(8, 64))
                    self.assertEqual(fixture.insert_ids.read_bytes(),
                                     struct.pack("<57I", 56, *(tail[::-1] if reverse else tail)))
                    self.assertEqual(fixture.delete_ids.read_bytes(), struct.pack("<II", 1, 0))
                    self.assertEqual(fixture.source_hash, hashlib.sha256(data_bytes).hexdigest())
                    self.assertEqual(fixture.originals,
                        {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in work.glob("base.*")})
                    self.assertEqual(len(fixture.originals), 2)
                    if empty:
                        self.assertEqual(fixture.empty_ids, work / "empty.ids")
                        self.assertEqual(fixture.empty_ids.read_bytes(), struct.pack("<I", 0))
                    else:
                        self.assertIsNone(fixture.empty_ids)
                        self.assertFalse((work / "empty.ids").exists())

    def test_population_build_failure_does_not_write_id_files(self):
        with self.assertRaisesRegex(RuntimeError, "build failed"):
            support.build_tiny_population("build", self.root, "float", "f",
                reverse_insert_ids=True, include_empty_ids=True,
                run_build=Mock(side_effect=RuntimeError("build failed")))
        self.assertEqual(list((self.root / "float").glob("*.ids")), [])
        with self.assertRaises(FileExistsError):
            support.build_tiny_population("build", self.root, "float", "f",
                reverse_insert_ids=True, include_empty_ids=True, run_build=Mock())


class ImportContracts(unittest.TestCase):
    def test_import_is_stdlib_only_without_io_or_process_configuration(self):
        source = Path(support.__file__).read_text(encoding="utf-8")
        code = compile(source, support.__file__, "exec")
        module = ModuleType("_ivf_support_import_contract")
        before_path, before_env, before_cwd = list(sys.path), dict(os.environ), os.getcwd()
        real_import = builtins.__import__
        imported = []

        def stdlib_import(name, *args, **kwargs):
            imported.append(name)
            self.assertFalse(name.startswith(("test_", "benchmark")), name)
            return real_import(name, *args, **kwargs)

        def forbidden(*args, **kwargs):
            self.fail("support import performed IO or process configuration")

        with patch.dict(sys.modules, {module.__name__: module}), \
             patch("builtins.__import__", side_effect=stdlib_import), \
             patch("builtins.open", side_effect=forbidden), \
             patch("io.open", side_effect=forbidden), \
             patch.object(Path, "mkdir", side_effect=forbidden), \
             patch("subprocess.run", side_effect=forbidden), \
             patch("os.chdir", side_effect=forbidden), \
             patch("os.getenv", side_effect=forbidden), \
             patch("os.putenv", side_effect=forbidden):
            exec(code, module.__dict__)
        self.assertTrue(set(imported) <= {"dataclasses", "hashlib", "json", "pathlib", "struct",
                                         "subprocess", "typing"}, imported)
        self.assertEqual(sys.path, before_path)
        self.assertEqual(dict(os.environ), before_env)
        self.assertEqual(os.getcwd(), before_cwd)


class NativeRunnerContracts(unittest.TestCase):
    build_progress = "\n".join("[IVF build] phase=" + phase for phase in (
        "input_identity", "training", "final_router", "posting_lists", "coordinate_heap",
        "heap_checkpoint", "artifact_verification", "complete"))
    open_progress = "[IVF driver] phase=open_private_copy_verify event=begin"

    def invoke(self, record, phase, stderr, returncode=0):
        stdout = json.dumps(record) if isinstance(record, dict) else record
        done = subprocess.CompletedProcess(["native"], returncode, stdout, stderr)
        with patch.object(support.subprocess, "run", return_value=done) as run:
            observed = support.run_native_cli(["native"], phase)
        run.assert_called_once_with(["native"], text=True, capture_output=True, timeout=90)
        return observed

    def test_build_requires_single_metric_and_complete_ordered_progress(self):
        record = {"baseline": "IVF", "phase": "build"}
        self.assertEqual(self.invoke(record, "build", self.build_progress), record)
        for stdout, stderr in ((record, ""), (record, self.build_progress.replace("training", "oops")),
                               (json.dumps(record) + "\n" + json.dumps(record), self.build_progress),
                               ({"baseline": "IVF", "phase": "query"}, self.build_progress)):
            with self.subTest(stdout=stdout, stderr=stderr), self.assertRaises(AssertionError):
                self.invoke(stdout, "build", stderr)

    def test_duplicate_json_keys_rejected_even_in_nested_metrics_and_failures(self):
        for raw in ('{"baseline":"IVF","phase":"build","phase":"build"}',
                    '{"baseline":"IVF","phase":"build","nested":{"x":1,"x":2}}'):
            for phase, rc, stderr in (("build", 0, self.build_progress), (None, 1, "ERROR: fail")):
                with self.subTest(raw=raw, phase=phase), self.assertRaisesRegex(AssertionError, "duplicate JSON"):
                    self.invoke(raw, phase, stderr, rc)

    def test_failure_requires_error_and_no_success_record(self):
        observed = self.invoke("", None, "ERROR: refused", 1)
        self.assertEqual(observed.returncode, 1)
        for raw, rc, stderr in (("", 0, "ERROR: refused"), ("", 1, "refused"),
                                ('{"baseline":"IVF","phase":"build"}', 1, "ERROR: refused")):
            with self.subTest(raw=raw, rc=rc), self.assertRaises(AssertionError):
                self.invoke(raw, None, stderr, rc)

    def test_query_preserves_progress_and_fetch_timing_checks(self):
        record = {"baseline": "IVF", "phase": "query", "fetch_adapter_mutex_wait_us_total": 1,
                  "fetch_backend_pin_us_total": 2, "fetch_pin_admission_wait_us_total": 3,
                  "fetch_us_total": 10, "fetch_pin_admission_policy": "bounds_reads_and_writes",
                  "fetch_substage_timing_scope": "not_additional_stages"}
        self.assertEqual(self.invoke(record, "query", self.open_progress), record)
        for key, value in (("fetch_adapter_mutex_wait_us_total", -1),
                           ("fetch_backend_pin_us_total", -1), ("fetch_pin_admission_wait_us_total", -1),
                           ("fetch_us_total", 1), ("fetch_pin_admission_policy", "wrong"),
                           ("fetch_substage_timing_scope", "wrong")):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.invoke(dict(record, **{key: value}), "query", self.open_progress)
        for stderr in ("", self.open_progress.replace("open_private_copy_verify", "wrong"),
                       *(self.open_progress + "\n[IVF driver] phase=query event=" + event
                         for event in ("failed", "abandoned", "incomplete"))):
            with self.subTest(stderr=stderr), self.assertRaises(AssertionError):
                self.invoke(record, "query", stderr)

    def test_mutation_counts_and_heap_flush_accounting(self):
        progress = (self.open_progress + "\n[IVF driver] phase=insert event=complete completed=3"
                    + "\n[IVF driver] phase=delete event=complete completed=2")
        for phase in ("insert", "delete"):
            record = {"baseline": "IVF", "phase": phase, "insert_count": 3,
                      "delete_count": 1, "missing_delete_count": 1,
                      "foreground_heap_flush_bytes": 8192, "foreground_heap_page_writes": 2,
                      "checkpoint_heap_flush_bytes": 4096, "checkpoint_heap_page_writes": 1,
                      "heap_io_counter_scope": "not_device_io"}
            self.assertEqual(self.invoke(record, phase, progress), record)
            for key, value in (("insert_count", 4), ("missing_delete_count", 0),
                               ("foreground_heap_flush_bytes", 1), ("checkpoint_heap_flush_bytes", 1),
                               ("heap_io_counter_scope", "wrong")):
                with self.subTest(phase=phase, key=key), self.assertRaises(AssertionError):
                    self.invoke(dict(record, **{key: value}), phase, progress)
            with self.assertRaises(AssertionError):
                self.invoke(record, phase, progress + "\n[IVF driver] phase=insert event=complete completed=3")


if __name__ == "__main__":
    unittest.main()
