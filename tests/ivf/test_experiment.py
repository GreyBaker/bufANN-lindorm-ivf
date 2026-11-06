"""Exercise the Bash recipes with tiny fake GT/prepare/query commands, not ANN."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "benchmark/IVF/experiment.sh"
SCRATCH = None


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bash recipe ", dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("base.bin", "query.bin", "truth.bin", "pq_pq_pivots.bin", "pq_pq_compressed.bin"):
            (self.root / name).write_bytes(b"fixture")
        native = self.root / "fake_native.py"
        native.write_text('''import json, os, sys
from pathlib import Path
script=Path(sys.argv[1]).stem
env=os.environ
root=Path(env['FIXTURE_ROOT'])
record=dict(script=script, nlist=env.get('IVF_NLIST'), max_reps=env['IVF_MAX_REPS'],
            routing_limit=env['IVF_ROUTING_WORK_LIMIT'], insert_override=env.get('INSERT_POINTS'))
with (root/'calls.jsonl').open('a') as out: out.write(json.dumps(record)+'\\n')
workload=script.removesuffix('_prepare')
work=Path(env['EVAL_TEMPFILES'])/env['DATASET']/'IVF'/workload
if script.endswith('_prepare'):
    (work/'index').mkdir(parents=True)
    (work/'index/base.heap').write_bytes(b'prepared base')
    (work/'prepared.json').write_text('{}')
else:
    out=work/'runs'/env['IVF_RUN_TAG'] if env.get('IVF_RUN_TAG') else work
    out.mkdir(parents=True,exist_ok=True)
    for name in ('query.heap','query.heap.pq.bin','neighbors.bin'):
        (out/name).write_bytes(b'native artifact')
    if env.get('FIXTURE_FAIL_QUERY') == '1': sys.exit(7)
    result=dict(nlist=int(env['IVF_NLIST']),nprobe=int(env.get('IVF_NPROBE',1)),
                router_breadth=int(env['IVF_ROUTER_BREADTH']),recall=99.5,query_qps=100,query_time_s=100)
    (out/'result.json').write_text(json.dumps(result))
''')
        bootstrap = self.root / "commands.sh"
        bootstrap.write_text('''timeout() { shift 3; "$@"; }
python3() {
    if [[ "$1" == */ensure_gt.py ]]; then
        printf '%s\\n' "$*" > "$FIXTURE_ROOT/gt.args"
        printf '%s\\n' "$FIXTURE_ROOT/truth.bin"
        return "${FIXTURE_GT_STATUS:-0}"
    fi
    command python3 "$@"
}
bash() { command python3 "$FIXTURE_NATIVE" "$1"; }
''')
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(
            ("IVF_", "EVAL_", "GT_", "DATA_", "QUERY_", "UPDATE_", "SIFT"))}
        self.env.update(DATASET="sift100m", DATA_BIN=str(self.root / "base.bin"),
                        QUERY_BIN=str(self.root / "query.bin"), IVF_PQ_PREFIX=str(self.root / "pq"),
                        GT_ARTIFACT_DIR=str(self.root / "groundtruth"),
                        EVAL_TEMPFILES=str(self.root / "output"), EVAL_CACHED=str(self.root / "cache"),
                        BASH_ENV=str(bootstrap), FIXTURE_ROOT=str(self.root), FIXTURE_NATIVE=str(native))

    def invoke(self, mode, *args, **env):
        return subprocess.run(["bash", str(SCRIPT), mode, *args], env=dict(self.env, **env),
                              cwd=self.root, text=True, capture_output=True, timeout=180)

    def calls(self):
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_previews_keep_the_two_documented_grids_and_do_no_work(self):
        for mode, nlists, count in (("sift100m-wide", [100, 1000, 10000, 100000, 200000, 300000], 73),
                                    ("sift100m-refine", [100000, 200000, 300000, 400000], 80)):
            result = self.invoke(mode, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            points = [tuple(map(int, line.split())) for line in result.stdout.splitlines()[1:-1]]
            self.assertEqual(len(points), count)
            self.assertEqual(list(dict.fromkeys(p[0] for p in points)), nlists)
            self.assertEqual(len({(p[0], p[1], p[3]) for p in points}), count)
            for nlist in nlists:
                probes = [p[1] for p in points if p[0] == nlist and p[2] == 128]
                expected = ([2 ** i for i in range(14) if 2 ** i <= nlist] if mode.endswith('wide') else
                            [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 384, 512])
                self.assertEqual(probes, expected)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "output").exists())
        self.assertNotEqual(self.invoke("workloads", "--dry-run", DATASET="sift1m").returncode, 0)

    def test_wide_records_results_and_removes_only_successful_private_heaps(self):
        result = self.invoke("sift100m-wide", INSERT_POINTS="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.root / "output"
        self.assertEqual([int(c['nlist']) for c in self.calls() if c['script']=='query_prepare'],
                         [100, 1000, 10000, 100000, 200000, 300000])
        self.assertTrue(all(c['max_reps']=='15' and c['insert_override'] is None for c in self.calls()))
        with (output / "results.csv").open() as results:
            self.assertEqual(len(list(csv.DictReader(results))), 73)
        self.assertEqual(len(list(output.rglob("base.heap"))), 6)
        self.assertEqual(len(list(output.rglob("neighbors.bin"))), 73)
        self.assertEqual(list(output.rglob("query.heap*")), [])
        self.assertTrue((output / "settings.env").is_file())
        self.assertEqual(len((output / "heap-cleanup.log").read_text().splitlines()), 73)

    def test_refinement_reuses_three_bases_and_prepares_only_400k(self):
        previous = self.root / "previous"
        previous.mkdir()
        (previous / "selected_gt.txt").write_text(str(self.root / "truth.bin") + "\n")
        for nlist in (100000, 200000, 300000):
            work = previous / f"nlist-{nlist}/sift100m/IVF/query"
            work.mkdir(parents=True)
            (work / "prepared.json").write_text(json.dumps(dict(settings=dict(max_reps="15", nlist=str(nlist)))))
        before = {p: p.read_bytes() for p in previous.rglob("prepared.json")}
        result = self.invoke("sift100m-refine", IVF_GRID_PREVIOUS_ROOT=str(previous))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([c['nlist'] for c in self.calls() if c['script']=='query_prepare'], ['400000'])
        self.assertTrue(all(c['routing_limit']=='5000000000' for c in self.calls() if c['nlist']=='400000'))
        self.assertNotIn('--generate', (self.root / 'gt.args').read_text())
        with (self.root / "output/results.csv").open() as results:
            self.assertEqual(len(list(csv.DictReader(results))), 80)
        for path in previous.glob('nlist-*/sift100m/IVF/query/prepared.json'):
            self.assertEqual(path.read_bytes(), before[path])

    def test_gt_failure_stops_work_and_existing_roots_are_not_overwritten(self):
        result = self.invoke("workloads", FIXTURE_GT_STATUS="3")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])
        self.assertTrue((self.root / "output/groundtruth.log").is_file())
        marker = self.root / "output/keep"
        marker.write_text("retain failed evidence")
        self.assertNotEqual(self.invoke("workloads").returncode, 0)
        self.assertEqual(marker.read_text(), "retain failed evidence")

    def test_workloads_are_independent_pairs_and_use_15_passes(self):
        result = self.invoke("workloads", IVF_MAX_REPS="15", IVF_GENERATE_GT="0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([c['script'] for c in self.calls()],
                         [w+s for w in ('query','insert','delete','update') for s in ('_prepare','')])
        self.assertTrue(all(c['max_reps']=='15' for c in self.calls()))
        self.assertNotIn('--generate', (self.root / 'gt.args').read_text())

    def test_conflicting_passes_and_old_preparations_are_rejected(self):
        for mode in ("workloads", "sift100m-wide", "sift100m-refine"):
            self.assertNotEqual(self.invoke(mode, "--dry-run", IVF_MAX_REPS="10").returncode, 0)
        previous = self.root / "previous"
        previous.mkdir()
        (previous / "selected_gt.txt").write_text(str(self.root / "truth.bin"))
        work = previous / "nlist-100000/sift100m/IVF/query"
        work.mkdir(parents=True)
        (work / "prepared.json").write_text(json.dumps(dict(settings=dict(max_reps="10", nlist="100000"))))
        result = self.invoke("sift100m-refine", IVF_GRID_PREVIOUS_ROOT=str(previous))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "output").exists())
        self.assertEqual(self.calls(), [])

    def test_query_failure_preserves_private_heaps(self):
        result = self.invoke("sift100m-wide", FIXTURE_FAIL_QUERY="1")
        self.assertNotEqual(result.returncode, 0)
        folder = self.root / "output/nlist-100/sift100m/IVF/query/runs/np1_rb128"
        self.assertTrue((folder / "query.heap").is_file())
        self.assertTrue((folder / "query.heap.pq.bin").is_file())
        self.assertEqual(len((self.root / "output/results.csv").read_text().splitlines()), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    SCRATCH = args.scratch
    unittest.main(argv=[sys.argv[0]])
