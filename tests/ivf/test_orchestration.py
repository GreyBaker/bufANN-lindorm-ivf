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
import recall_diagnostic
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

    def test_recall_audit_boundary_ties_and_genuine_misses(self):
        from audit_recall import audit

        gt = self.root / "deep.gt"
        # ID 5 is outside the live prefix. IDs 0/1 tie at the live cutoff.
        gt.write_bytes(struct.pack("<II4I4f4I", 1, 4, 5, 0, 1, 2,
                                   0, 1, 1, 2, 5, 0, 1, 2))
        result = self.root / "neighbors.bin"
        result.write_bytes(struct.pack("<IIIf", 1, 1, 1, 1))
        report = audit(gt, result, base_count=3, k=1)
        self.assertEqual(report["strict_first_k_id_recall_percent"], 0)
        self.assertEqual(report["tie_expanded_recall_percent"], 100)
        self.assertEqual(report["additional_hits_from_tie_expansion"], 1)
        self.assertEqual(report["returned_slots_outside_exact_kth_distance"], 0)
        result.write_bytes(struct.pack("<IIIf", 1, 1, 2, 2))
        report = audit(gt, result, base_count=3, k=1)
        self.assertEqual(report["tie_expanded_recall_percent"], 0)
        self.assertEqual(report["returned_slots_outside_exact_kth_distance"], 1)
        gt.write_bytes(struct.pack("<II2I2f", 1, 2, 0, 1, 1, 1))
        report = audit(gt, result, base_count=3, k=1)
        self.assertIsNone(report["tie_expanded_recall_percent"])
        self.assertEqual(report["queries_without_proven_complete_boundary_ties"], 1)
        with self.assertRaisesRegex(ValueError, "result framing/query range"):
            audit(gt, result, base_count=3, k=1, query_offset=1)

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
        args = argparse.Namespace(work_dir=self.root, workload="insert", output_dir=self.root / "runs")
        for execute in (harness.run, mixed.run):
            with patch.dict(os.environ, {"IVF_WARMUP_QUERIES": "100001"}, clear=True), \
                 patch.object(harness, "mutation_identity") as mutation_identity, \
                 patch.object(mixed, "identity") as mixed_identity, \
                 patch.object(common.subprocess, "Popen") as launch, self.assertRaisesRegex(ValueError, "IVF_WARMUP_QUERIES"):
                execute(args)
            launch.assert_not_called()
            mutation_identity.assert_not_called()
            mixed_identity.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_counted_source_ranges_and_compatibility(self):
        path = self.root / "insert_ids.bin"
        first, count = 17, 4101
        raw = struct.pack("<" + "I" * (count + 1), count, *range(first, first + count))
        path.write_bytes(raw)
        expected = common.digest(path)
        self.assertEqual(common.counted_range_identity(path, first, count), expected)
        self.assertEqual(mixed.counted_identity(path, first, count), expected)
        (self.root / "data").mkdir()
        source = self.root / "data/full.bin"
        source.write_bytes(b"entire source identity, including rows outside the update range")
        args = argparse.Namespace(workload="insert", update_count=count, base_count=first,
                                  full_count=first + count)
        self.assertEqual(harness.mutation_identity(args, self.root), {
            "workload": "insert", "count": count, "first_id": first,
            "ids_identity": expected, "source_identity": common.digest(source),
        })
        for malformed, message, mixed_message in (
            (raw[:-1], "framing", "length mismatch"),
            (raw + b"extra", "framing", "length mismatch"),
            (struct.pack("<I", count - 1) + raw[4:], "count mismatch", "count mismatch"),
            # Swap two IDs strictly beyond the first 4096-row validation block.
            (raw[:4 + 4097 * 4] + raw[4 + 4098 * 4:4 + 4099 * 4]
             + raw[4 + 4097 * 4:4 + 4098 * 4] + raw[4 + 4099 * 4:],
             "source-row range", "source-row ranges"),
        ):
            path.write_bytes(malformed)
            with self.subTest(message=message), patch.object(common, "digest") as hash_file:
                with self.assertRaisesRegex(ValueError, message):
                    common.counted_range_identity(path, first, count)
                with self.assertRaisesRegex(ValueError, mixed_message):
                    mixed.counted_identity(path, first, count)
                hash_file.assert_not_called()

    def test_absent_destinations_include_dangling_symlinks(self):
        missing = self.root / "new-output"
        common.require_absent(missing, "occupied")
        for path in (self.root / "file", self.root / "directory"):
            if path.name == "file":
                path.write_bytes(b"preserved")
            else:
                path.mkdir()
            with self.assertRaisesRegex(ValueError, "occupied"):
                common.require_absent(path, "occupied")
        self.assertEqual((self.root / "file").read_bytes(), b"preserved")
        dangling = self.root / "dangling"
        try:
            dangling.symlink_to(missing)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        self.assertFalse(dangling.exists())
        with self.assertRaisesRegex(ValueError, "occupied"):
            common.require_absent(dangling, "occupied")
        live = self.root / "live-link"
        live.symlink_to(self.root / "file")
        with self.assertRaisesRegex(ValueError, "occupied"):
            common.require_absent(live, "occupied")

    def test_named_hash_parity_and_order(self):
        one, two = self.root / "one", self.root / "two"
        one.write_bytes(b"first")
        two.write_bytes(b"second")
        named = {"second": two, "first": one, "alias": two}
        expected = {name: common.digest(path) for name, path in named.items()}
        with patch.object(common, "digest", wraps=common.digest) as hash_file:
            actual = common.digests(named)
        self.assertEqual(actual, expected)
        self.assertEqual(list(actual), list(named))
        self.assertEqual([call.args[0] for call in hash_file.call_args_list], [two, one, two])
        self.assertEqual(common.digests({}), {})
        two.write_bytes(b"changed")
        self.assertNotEqual(common.digests(named), expected)

    def test_query_flags_preserve_defaults_order_and_validated_warmups(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.query_search_flags(), [
                "--nprobe", "1", "--rerank_factor", "2", "--query_threads", "1",
            ])
            self.assertEqual(settings.query_policy_flags(settings.warmup_queries()),
                             ["--warmup_queries", "0"])
            os.environ.update(IVF_NPROBE="4", IVF_RERANK_FACTOR="3.0", QUERY_THREADS="2",
                              IVF_ROUTER_MODE="pca", IVF_ROUTER_RERANK_FACTOR="5",
                              IVF_ROUTER_BREADTH="64", IVF_WARMUP_QUERIES="7")
            validated = settings.warmup_queries()
            # Flag formatting must not reread/revalidate the preflight warmup policy.
            os.environ["IVF_WARMUP_QUERIES"] = "invalid"
            self.assertEqual(settings.query_search_flags(), [
                "--nprobe", "4", "--rerank_factor", "3.0", "--query_threads", "2",
            ])
            with patch.object(settings, "warmup_queries", side_effect=AssertionError):
                self.assertEqual(settings.query_policy_flags(validated), [
                    "--router_mode", "pca", "--router_rerank_factor", "5",
                    "--router_breadth", "64", "--warmup_queries", "7",
                ])
        args = argparse.Namespace(dtype="uint8", metric="l2", dim=128)
        with patch.dict(os.environ, {"IVF_DRIVER_BIN": "driver"}, clear=True):
            self.assertEqual(harness.driver_open_command(args, "8", self.root), [
                "driver", "--validate_inputs", "0", "--data_type", "uint8",
                "--metric", "l2", "--dim", "128", "--nlist", "8",
                "--index_prefix", str(self.root),
            ])

    def test_query_record_recognition_leaves_workload_checks_local(self):
        record = {"baseline": "IVF", "phase": "query", "workload": "mixed"}
        self.assertTrue(common.is_query_record(record, "mixed"))
        self.assertNotIn("recall", record)  # Recognition does not certify scoring/counts.
        self.assertFalse(common.is_query_record(record, "query"))
        for key, value in (("baseline", "other"), ("phase", "insert")):
            self.assertFalse(common.is_query_record({**record, key: value}, "mixed"))
        with self.assertRaises(KeyError):
            common.is_query_record({}, "query")

    def test_json_read_size_limits_are_explicit_and_inclusive(self):
        path = self.root / "metadata.json"
        raw = b'{"ok": true}'
        path.write_bytes(raw)
        self.assertEqual(common.read_json(path, max_bytes=len(raw)), {"ok": True})
        with self.assertRaisesRegex(ValueError, "oversized"):
            common.read_json(path, max_bytes=len(raw) - 1)
        path.write_bytes(raw + b" " * (65537 - len(raw)))
        with self.assertRaisesRegex(ValueError, "oversized"):
            common.read_json(path)
        self.assertEqual(common.read_json(path, max_bytes=8 << 20), {"ok": True})

    def test_runner_argv_and_native_records_remain_owned_by_driver(self):
        work = self.root.resolve()
        for directory in ("data", "index", "ids", "groundtruth/rounds"):
            (work / directory).mkdir(parents=True)
        data, query = work / "data/full.bin", work / "data/query.bin"
        truth = work / "groundtruth/rounds/round_0_gt1.bin"
        data.write_bytes(struct.pack("<II8f", 4, 2, *range(8)))
        query.write_bytes(struct.pack("<II2f", 1, 2, 0, 1))
        truth.write_bytes(struct.pack("<IIIf", 1, 1, 0, 0))
        prefix = work / "index/tiny"
        for suffix in harness.SUFFIXES:
            Path(str(prefix) + suffix).write_text("IVFINDEX1\n" if suffix == ".manifest" else suffix)
        for name, row in (("insert", 3), ("delete", 2)):
            (work / f"ids/round_0_{name}.bin").write_bytes(struct.pack("<II", 1, row))
        args = argparse.Namespace(
            dataset="tiny", dtype="float", metric="l2", dim=2, full_count=4, base_count=3,
            recall_at=1, work_dir=work, output_dir=work / "query-run", workload="query",
            update_count=0, rounds=1, insert_cap=1, delete_cap=1, insert_count=1,
            delete_count=1, query_count=5,
        )
        common.write_json(work / "prepared.json", {
            "version": 1, "profile": settings.profile(args),
            "settings": {"page_size": "4096", "nlist": "2"},
            "query": str(query), "query_identity": common.digest(query),
            "truth": str(truth), "truth_identity": common.digest(truth),
            "manifest_identity": common.digest(Path(str(prefix) + ".manifest")),
        })
        common.write_json(work / "mixed_prepared.json", mixed.identity(args))
        commands, native_bytes = [], {}

        def native_driver(command, log):
            commands.append(command)
            Path(log).write_text("fake native execution\n")
            result = Path(command[command.index("--result_file") + 1])
            workload = "mixed" if command[command.index("--workload") + 1] == "mixed" else "query"
            record = {
                "baseline": "IVF", "phase": "query", "workload": workload, "recall": 100,
                "explicit_warmup_query_count": 7, "warmup_policy": "fixture",
                "insert_count": 1, "delete_count": 1, "live_count": 3,
                "mixed_wall_s": 0.25, "concurrent_query_count": 5, "overlapping_query_count": 3,
                "native_only_field": "must survive without a Python rewrite",
                "mixed_protocol_version": 3, "mixed_round_count": 1,
            }
            if workload == "mixed":
                rounds = Path(command[command.index("--round_results_dir") + 1])
                rounds.mkdir()
                common.write_json(rounds / "round_0.json", dict(
                    version=3, round=0, insert_count=1, delete_count=1,
                    concurrent_query_count=5, overlapping_query_count=3, recall_available=False))
            raw = ("  " + json.dumps(record, separators=(",", ":")) + "\n\n").encode()
            with result.open("xb") as out:
                out.write(raw)
            native_bytes[result] = raw

        env = {
            "IVF_DRIVER_BIN": "fake-driver", "IVF_MEMORY_BUDGET_GB": "3",
            "IVF_QUERY_MEMORY_BUDGET_GB": "2", "IVF_WARMUP_QUERIES": "7",
            "IVF_ROUTER_MODE": "pca", "IVF_SCHEDULER": "weighted", "NTHREADS": "8",
        }
        with patch.dict(os.environ, env, clear=True), \
             patch.object(harness, "command_log", side_effect=native_driver), \
             patch.object(mixed, "command_log", side_effect=native_driver):
            # Private-session collisions fail before launching or writing evidence.
            collision = work / "private-collision"
            collision.mkdir()
            (collision / "query.heap").write_bytes(b"owned heap")
            args.output_dir = collision
            with self.assertRaisesRegex(ValueError, "run output exists"):
                harness.run(args)
            self.assertEqual((collision / "query.heap").read_bytes(), b"owned heap")
            self.assertFalse((collision / "nlist_selection.json").exists())
            self.assertEqual(commands, [])
            args.output_dir = work / "query-run"
            harness.run(args)
            with self.assertRaisesRegex(ValueError, "run output exists"):
                harness.run(args)
            args.output_dir = work / "mixed-run"
            mixed.run(args)
            with self.assertRaisesRegex(ValueError, "mixed output exists"):
                mixed.run(args)
            # A native failure after completed round accounting must not cause
            # the harness to erase that evidence or publish a success summary.
            failed_out = work / "mixed-failed"
            args.output_dir = failed_out

            def fail_later(command, log):
                Path(log).write_text("later native round failed\n")
                rounds = failed_out / "rounds"
                rounds.mkdir()
                common.write_json(rounds / "round_0.json", {"completed": True})
                (failed_out / "private.heap").write_bytes(b"partial owned session")
                raise ValueError("later round failed")

            with patch.object(mixed, "command_log", side_effect=fail_later):
                with self.assertRaisesRegex(ValueError, "later round failed"):
                    mixed.run(args)
            self.assertTrue((failed_out / "rounds/round_0.json").is_file())
            self.assertEqual((failed_out / "private.heap").read_bytes(), b"partial owned session")
            self.assertFalse((failed_out / "result.json").exists())
            self.assertFalse((failed_out / "summary.json").exists())
        self.assertEqual(len(commands), 2)
        opening = [
            "fake-driver", "--validate_inputs", "0", "--data_type", "float", "--metric", "l2",
            "--dim", "2", "--nlist", "2", "--index_prefix", str(prefix),
        ]
        search = ["--nprobe", "1", "--rerank_factor", "2", "--query_threads", "1"]
        policy = ["--router_mode", "pca", "--warmup_queries", "7"]
        query_out = work / "query-run"
        self.assertEqual(commands[0], opening + [
            "--workload", "search_only", "--query_file", str(query), "--gt_file", str(truth),
            "--recall_at", "1",
        ] + search + [
            "--buffer_frames", "64", "--memory_budget_gb", "2",
            "--result_file", str(query_out / "result.json"),
            "--search_result_file", str(query_out / "neighbors.bin"),
        ] + ["--private_heap_path", str(query_out / "query.heap")] + policy)
        round_out = work / "mixed-run"
        self.assertEqual(commands[1], opening + [
            "--index_format", "immutable", "--workload", "mixed", "--scheduler", "weighted",
            "--pool_threads", "8", "--full_data_file", str(data), "--query_file", str(query),
            "--round_schedule", str(round_out / "round_schedule.txt"),
            "--round_results_dir", str(round_out / "rounds"),
            "--gt_file", str(truth), "--recall_at", "1",
        ] + search + [
            "--private_heap_path", str(round_out / "private.heap"),
            "--buffer_frames", "64", "--memory_budget_gb", "3",
            "--result_file", str(round_out / "result.json"),
            "--search_result_file", str(round_out / "neighbors.bin"),
            "--query_ratio", "80", "--scheduler_seed", str(0xA5A5A5A5), "--query_count", "5",
        ] + policy)
        for path, raw in native_bytes.items():
            self.assertEqual(path.read_bytes(), raw)
            self.assertFalse(path.with_name("native_result.json").exists())
        protocol = common.read_json(query_out / "execution_protocol.json")
        self.assertFalse(protocol["canonical_native_query_metrics_rewritten"])
        self.assertEqual(protocol["query_session_policy"],
                         "immutable_base_forked_to_mutable_session_without_updates")
        self.assertEqual(protocol["warmup_query_count_requested"], 7)

    def test_readme_uses_checked_in_runners_not_executable_markdown(self):
        text = (Path(__file__).resolve().parents[2] / "benchmark/IVF/README.md").read_text()
        for mode in ("workloads", "sift100m-wide", "sift100m-refine"):
            self.assertIn(f"bash benchmark/IVF/experiment.sh {mode}", text)
        self.assertNotIn("exec python3 - <<", text)
        self.assertNotIn("grid-search-guide.md", text)

    def test_diagnostic_query_prefix_is_bounded_and_exclusive(self):
        source, output = self.root / "query.bin", self.root / "prefix.bin"
        source.write_bytes(struct.pack("<II", 10000, 128) + bytes(range(128)) * 10000)
        recall_diagnostic.query_prefix(source, output, 256)
        self.assertEqual(output.read_bytes(), struct.pack("<II", 256, 128) + bytes(range(128)) * 256)
        with self.assertRaises(FileExistsError):
            recall_diagnostic.query_prefix(source, output, 256)
        for count in (0, 10001):
            with self.assertRaisesRegex(ValueError, "query count"):
                recall_diagnostic.query_prefix(source, self.root / "absent", count)
        self.assertFalse((self.root / "absent").exists())

    def test_all_lists_diagnostic_retains_native_evidence_and_failed_heaps(self):
        previous = self.root / "previous"
        work = previous / "nlist-100/sift100m/IVF/query"  # Historical layout is supported.
        work.mkdir(parents=True)
        query, truth = self.root / "query.bin", self.root / "deep.gt"
        query.write_bytes(struct.pack("<II", 10000, 128) + bytes(10000 * 128))
        truth.write_bytes(b"checked full-source truth fixture")
        (previous / "selected_gt.txt").write_text(str(truth) + "\n")
        common.write_json(work / "prepared.json", dict(
            profile={"base_count": 90000000}, settings={"nlist": "100"}, query_identity=common.digest(query)))
        for valid in (True, False):
            output = self.root / ("diagnostic-good" if valid else "diagnostic-failed")
            args = argparse.Namespace(previous_root=previous, output=output, query=query,
                                      driver=Path(sys.executable), query_count=1, timeout_s=10)

            def execute(command, log, timeout, cpus):
                self.assertEqual(command[command.index("--nprobe") + 1], "100")
                private = Path(command[command.index("--private_heap_path") + 1])
                private.write_bytes(b"private heap")
                Path(str(private) + ".pq.bin").write_bytes(b"private PQ")
                result = Path(command[command.index("--result_file") + 1])
                common.write_json(result, dict(query_count=1, nprobe=100, live_count=90000000,
                    rerank_factor=int(command[command.index("--rerank_factor") + 1]),
                    scanned_codes=90000000 if valid else 0))
                Path(command[command.index("--search_result_file") + 1]).write_bytes(b"native neighbors")
                log.write_text("native log")

            with patch.object(recall_diagnostic, "execute", execute), \
                    patch.object(recall_diagnostic, "audit", return_value={"tie_expanded_recall_percent": 100}), \
                    patch("sys.stdout", io.StringIO()):
                if valid:
                    recall_diagnostic.run(args)
                else:
                    with self.assertRaisesRegex(ValueError, "complete live base"):
                        recall_diagnostic.run(args)
            self.assertTrue((output / "plan.json").is_file())
            self.assertTrue((output / "rf3/result.json").is_file())
            self.assertTrue((output / "rf3/neighbors.bin").is_file())
            self.assertEqual((output / "rf3/query.heap").is_file(), not valid)
            self.assertEqual((output / "rf3/audit.json").is_file(), valid)
            if valid:
                self.assertTrue((output / "rf30/audit.json").is_file())

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
        selected, gt_log = self.root / "selected_gt.txt", self.root / "groundtruth.log"
        command = [sys.executable, "-c", "import os, sys; print(os.getcwd()); print('GT progress', file=sys.stderr)"]
        common.execute(command, gt_log, 10, None, cwd=self.root, stdout_path=selected)
        self.assertEqual(selected.read_text().strip(), str(self.root))
        self.assertIn("GT progress", gt_log.read_text())
        self.assertNotIn("GT progress", selected.read_text())
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
            "counted_range_identity",
            "digest",
            "flags",
            "header",
            "require",
            "require_absent",
            "is_query_record",
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
            "query_search_flags",
            "query_policy_flags",
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
