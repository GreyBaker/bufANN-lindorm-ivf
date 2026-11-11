"""Exercise the fixed controller with tiny fake GT/prepare/query commands, not ANN."""
import argparse
from contextlib import redirect_stdout
import csv
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
sys.path.insert(0, str(REPO / 'benchmark/IVF'))
import experiment
from settings import ROUTER_BACKEND_ID, build_settings

SCRATCH = None
NLISTS = [1000, 10000, 100000, 200000, 300000, 400000, 500000]
PROBES = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 384, 512]


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='query recipe ', dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('base.bin', 'query.bin', 'truth.bin'):
            (self.root / name).write_bytes(b'fixture')
        self.write_pq()
        self.env = dict(DATA_BIN=str(self.root / 'base.bin'), QUERY_BIN=str(self.root / 'query.bin'),
                        IVF_PQ_PREFIX=str(self.root / 'pq'), GT_ARTIFACT_DIR=str(self.root / 'groundtruth'),
                        EVAL_TEMPFILES=str(self.root / 'output'), EVAL_CACHED=str(self.root / 'cache'))
        self.calls = []
        self.commands = []
        self.gt_status = 0
        self.fail_query = False
        self.fail_prepare = False
        self.bad_result = False

    def write_pq(self, chunks=64, rows=4, dim=128):
        blocks = [struct.pack('<II', 256, dim) + bytes(256 * dim * 4),
                  struct.pack('<II', dim, 1) + bytes(dim * 4),
                  struct.pack('<II', dim, 1) + struct.pack(f'<{dim}I', *range(dim)),
                  struct.pack('<II', chunks + 1, 1)
                  + struct.pack(f'<{chunks + 1}I', *(i * dim // chunks for i in range(chunks + 1)))]
        offsets = [48]
        for block in blocks:
            offsets.append(offsets[-1] + len(block))
        (self.root / 'pq_pq_pivots.bin').write_bytes(struct.pack('<II5Q', 5, 1, *offsets) + b''.join(blocks))
        (self.root / 'pq_pq_compressed.bin').write_bytes(struct.pack('<II', rows, chunks) + bytes(rows * chunks))

    def fake_execute(self, command, log, timeout, cpus, env, stdout_path=None):
        self.assertEqual(timeout, 86400)
        self.assertIsNone(cpus)
        self.commands.append(list(command))
        with log.open('x') as file:
            file.write('fake command evidence\n')
        if Path(command[1]).name == 'ensure_gt.py':
            self.assertEqual(command[0], sys.executable)
            with stdout_path.open('x') as file:
                file.write(str(self.root / 'truth.bin') + '\n')
            if self.gt_status:
                raise ValueError(f'command failed ({self.gt_status})')
            return
        self.assertEqual(command[0], 'bash')
        script = Path(command[1]).stem
        self.calls.append(dict(script=script, settings=build_settings(env), env=dict(env)))
        work = Path(env['EVAL_TEMPFILES']) / 'sift100m/IVF/query'
        if script == 'query_prepare':
            work.mkdir(parents=True)
            (work / 'index').mkdir()
            (work / 'index/base.heap').write_bytes(b'prepared base')
            (work / 'index/sift100m.pq.bin').write_bytes(Path(env['IVF_PQ_PREFIX'] + '_pq_pivots.bin').read_bytes())
            if self.fail_prepare:
                raise ValueError('command failed (6)')
            (work / 'prepared.json').write_text(json.dumps(dict(settings=build_settings(env), router_backend=ROUTER_BACKEND_ID)))
        elif script == 'query':
            out = work / 'runs' / env['IVF_RUN_TAG']
            out.mkdir(parents=True)
            for name in ('query.heap', 'query.heap.pq.bin', 'neighbors.bin'):
                (out / name).write_bytes(b'native artifact')
            if self.fail_query:
                raise ValueError('command failed (7)')
            result = dict(nlist=int(env['IVF_NLIST']), nprobe=int(env['IVF_NPROBE']),
                          router_breadth=int(env['IVF_ROUTER_BREADTH']), recall=99.5, query_qps=100, query_time_s=100)
            (out / 'result.json').write_text('broken' if self.bad_result else json.dumps(result))
        else:
            self.fail(script)

    def invoke(self, **env):
        # Only the source count shrinks. Real PQ64 validation and recipe settings run unchanged.
        with patch.dict(os.environ, dict(self.env, **env), clear=True), \
                patch.object(experiment, 'FULL_COUNT', 4), \
                patch.object(experiment, 'execute', self.fake_execute), redirect_stdout(io.StringIO()):
            experiment.run()

    def evidence(self, number=1):
        return self.root / f'output/sweeps/run-{number:04d}'

    def work(self, nlist=1000):
        return self.root / f'output/nlist-{nlist}/sift100m/IVF/query'

    def test_fixed_grid_results_and_repeated_preparation_reuse(self):
        self.invoke(INSERT_POINTS='1', IVF_MAX_REPS='2', IVF_ROUTING_WORK_LIMIT='1',
                    IVF_COST_MODEL_REPORT='missing', IVF_NLIST='1', QUERY_THREADS='99',
                    IVF_BUILD_THREADS='99', IVF_WARMUP_QUERIES='0')
        calls = list(self.calls)
        self.assertEqual([int(c['settings']['nlist']) for c in calls if c['script'] == 'query_prepare'], NLISTS)
        self.assertEqual([c['script'] for c in calls], (['query_prepare'] + ['query'] * 15) * 7)
        fixed = dict(max_reps='15', trainer_mode='lv-plus', build_threads='16', memory_budget_gb='96',
                     page_size='4096', buffer_frames='262144', optimized_modes='1', pca_variance='0.95',
                     routing_work_limit='10000000000', router_mode='pca-refined', router_rerank_factor='3',
                     training_rerank_factor='3', training_refine_breadth='128')
        execution = dict(IVF_ROUTER_MODE='pca-refined', IVF_ROUTER_RERANK_FACTOR='3', IVF_ROUTER_BREADTH='128',
                         IVF_RERANK_FACTOR='3', IVF_WARMUP_QUERIES='64', IVF_QUERY_MEMORY_BUDGET_GB='96',
                         QUERY_THREADS='8', NTHREADS='16', OMP_NUM_THREADS='16', OMP_THREAD_LIMIT='16',
                         MKL_NUM_THREADS='1', INSERT_POINTS=None, IVF_COST_MODEL_REPORT=None)
        for call in calls:
            self.assertEqual({k: call['settings'][k] for k in fixed}, fixed)
            self.assertEqual({k: call['env'].get(k) for k in execution}, execution)
        with (self.evidence() / 'results.csv').open() as file:
            reader = csv.DictReader(file)
            self.assertNotIn('effective_breadth', reader.fieldnames)
            rows = list(reader)
        expected = [(nl, np, 128) for nl in NLISTS for np in PROBES]
        self.assertEqual([(int(r['nlist']), int(r['nprobe']), int(r['router_breadth'])) for r in rows], expected)
        with (self.evidence() / 'grid.tsv').open() as file:
            self.assertEqual([tuple(map(int, r)) for r in list(csv.reader(file, delimiter='\t'))[1:]], expected)
        gt = self.commands[0]
        self.assertIn('--query-only', gt)
        self.assertEqual(gt[gt.index('--num-threads') + 1], '16')
        self.assertIn('--generate', gt)
        self.assertEqual(len(list((self.root / 'output').rglob('base.heap'))), 7)
        self.assertEqual(len(list((self.root / 'output').rglob('neighbors.bin'))), 105)
        self.assertEqual(list((self.root / 'output').rglob('query.heap*')), [])
        self.assertEqual(len((self.evidence() / 'heap-cleanup.log').read_text().splitlines()), 105)
        before = {p: p.read_bytes() for p in (self.root / 'output').rglob('*') if p.is_file()}
        self.invoke()
        self.assertEqual([c['script'] for c in self.calls[len(calls):]], ['query'] * 105)
        self.assertTrue((self.evidence(2) / 'results.csv').is_file())
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents, str(path))

    def test_no_mode_or_dry_run_arguments(self):
        for arg in ('workloads', 'sift100m-wide', 'sift100m-refine', '--dry-run'):
            result = subprocess.run([sys.executable, str(experiment.__file__), arg],
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((REPO / 'benchmark/IVF/experiment.sh').exists())
        self.assertEqual(self.calls, [])

    def test_pq64_validation_precedes_any_work(self):
        for kwargs in (dict(chunks=32), dict(rows=3), dict(dim=64)):
            with self.subTest(kwargs=kwargs):
                self.write_pq(**kwargs)
                with self.assertRaises(ValueError):
                    self.invoke()
                self.assertEqual(self.commands, [])
                self.assertFalse((self.root / 'output').exists())
        self.write_pq()
        with (self.root / 'pq_pq_compressed.bin').open('ab') as file:
            file.write(b'extra')
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            self.invoke()
        self.write_pq()
        with (self.root / 'pq_pq_pivots.bin').open('ab') as file:
            file.write(b'extra')
        with self.assertRaisesRegex(ValueError, 'valid pivot offsets'):
            self.invoke()

    def test_gt_failure_stops_work_but_retains_evidence_on_retry(self):
        self.gt_status = 3
        for number in (1, 2):
            with self.assertRaisesRegex(ValueError, r'command failed \(3\)'):
                self.invoke()
            self.assertEqual(self.calls, [])
            self.assertTrue((self.evidence(number) / 'groundtruth.log').is_file())
        self.assertFalse((self.evidence() / 'results.csv').exists())

    def test_query_failure_preserves_private_heaps_and_reuses_base(self):
        self.fail_query = True
        for number in (1, 2):
            with self.assertRaisesRegex(ValueError, r'command failed \(7\)'):
                self.invoke()
            folder = self.work() / f'runs/run-{number:04d}_np16_rb128'
            self.assertTrue((folder / 'query.heap').is_file())
            self.assertTrue((folder / 'query.heap.pq.bin').is_file())
            self.assertEqual(len((self.evidence(number) / 'results.csv').read_text().splitlines()), 1)
        self.assertEqual([c['script'] for c in self.calls], ['query_prepare', 'query', 'query'])

    def test_bad_result_preserves_private_heaps_and_empty_csv(self):
        self.bad_result = True
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertTrue((self.work() / 'runs/run-0001_np16_rb128/query.heap').is_file())
        self.assertEqual(len((self.evidence() / 'results.csv').read_text().splitlines()), 1)

    def test_partial_preparation_is_never_overwritten(self):
        self.fail_prepare = True
        with self.assertRaisesRegex(ValueError, r'command failed \(6\)'):
            self.invoke()
        marker = self.work() / 'index/base.heap'
        before = marker.read_bytes()
        self.fail_prepare = False
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(marker.read_bytes(), before)
        self.assertFalse((self.work() / 'prepared.json').exists())

    def test_incompatible_preparations_are_preserved(self):
        self.fail_query = True
        with self.assertRaises(ValueError):
            self.invoke()
        path = self.work() / 'prepared.json'
        original = json.loads(path.read_text())
        for field, value in (('build_threads', '20'), ('routing_work_limit', '3000000000')):
            record = json.loads(json.dumps(original))
            record['settings'][field] = value
            path.write_text(json.dumps(record))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'different settings'):
                self.invoke()
            self.assertEqual(path.read_bytes(), before)
        path.write_text(json.dumps(dict(original, router_backend='old-backend')))
        with self.assertRaisesRegex(ValueError, 'different settings'):
            self.invoke()
        path.write_text(json.dumps(original))
        pivots = self.work() / 'index/sift100m.pq.bin'
        pivots.write_bytes(b'other pivots')
        with self.assertRaisesRegex(ValueError, 'different PQ pivots'):
            self.invoke()
        self.assertEqual(pivots.read_bytes(), b'other pivots')
        self.assertEqual(len(self.calls), 2)

    def test_locations_and_explicit_build_environment_do_not_mutate_parent(self):
        with patch.dict(os.environ, {}, clear=True):
            env = experiment.environment()
            self.assertEqual(env['DATA_BIN'], str(Path.home() / 'project/ann-data/sift100m/base.bin'))
            self.assertEqual(env['IVF_BUILD_DIR'], str(REPO / 'build/ivf-benchmark'))
            self.assertEqual(env['IVF_PQ_PREFIX'], str(Path(env['DATA_BIN']).parent / 'pq/pq64'))
        with patch.dict(os.environ, dict(self.env, IVF_MAX_REPS='2', IVF_BUILD_DIR=str(self.root / 'build')), clear=True):
            before = dict(os.environ)
            env = experiment.environment()
            env['IVF_NLIST'] = '1000'
            self.assertEqual(build_settings(env)['max_reps'], '15')
            self.assertEqual(env['INCREMENTAL_GT_BIN'], str(self.root / 'build/tests/incremental_groundtruth'))
            self.assertEqual(dict(os.environ), before)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    SCRATCH = args.scratch
    unittest.main(argv=[sys.argv[0]])
