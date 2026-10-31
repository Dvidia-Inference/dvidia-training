"""Pure evidence/protocol tests; never start a native comparison."""
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import contextlib
import io
import subprocess

import numpy as np


_PATH = Path(__file__).resolve().parents[1]/'benchmarks/piper-control-profile-v01/compare.py'
_SPEC = importlib.util.spec_from_file_location('piper_control_compare', _PATH)
compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare)


class ComparisonProtocolTests(unittest.TestCase):
    def test_fixed_balanced_schedule_and_no_single_pair(self):
        self.assertEqual(compare.schedule(2), ['baseline', 'candidate', 'candidate', 'baseline'])
        self.assertEqual(compare.schedule(4), compare.schedule(2)*2)
        for invalid in (True, 0, 1, 3, 10):
            with self.assertRaises(ValueError):
                compare.schedule(invalid)

    def test_source_change_and_undeclared_difference_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'src/dvidia_training'
            folder.mkdir(parents=True)
            for name in compare.SOURCE_FILES:
                (folder/name).write_text(name)
            frozen = compare.inventory(root)
            compare.check_inventory(root, frozen)
            (folder/'piper_task.py').write_text('candidate')
            candidate = compare.inventory(root)
            self.assertEqual(compare.source_differences(frozen, candidate), ['piper_task.py'])
            with self.assertRaises(ValueError):
                compare.check_inventory(root, frozen)
            (folder/'reference_task.py').write_text('changed scorer')
            with self.assertRaises(ValueError):
                compare.source_differences(frozen, compare.inventory(root))

    def test_document_comparison_catches_contact_and_scorer_mismatch(self):
        left = {'contacts': {'left': 1}, 'scorer': {'retention_lost': False}}
        self.assertTrue(compare.exact_document(left, left)['exact_match'])
        right = {'contacts': {'left': 0}, 'scorer': {'retention_lost': True}}
        self.assertEqual(compare.exact_document(left, right)['different_fields'], ['contacts', 'scorer'])

    def test_bitwise_array_comparison_catches_signed_zero_and_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary)/'a.npz', Path(temporary)/'b.npz'
            np.savez(left, qpos=np.array([[0., 1.], [2., 3.]]))
            np.savez(right, qpos=np.array([[0., 1.], [2., 3.]]))
            self.assertTrue(compare.compare_arrays(left, right)['exact_match'])
            np.savez(right, qpos=np.array([[-0., 1.], [2., 3.]]))
            result = compare.compare_arrays(left, right)
            self.assertFalse(result['exact_match'])
            self.assertEqual(result['differences'][0]['first_row'], 0)
            np.savez(right, qpos=np.array([[0., 1.]]))
            self.assertEqual(compare.compare_arrays(left, right)['differences'][0]['reason'], 'shape_or_dtype')

    def test_reject_executable_npz_and_duplicate_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary)/'a.npz', Path(temporary)/'b.npz'
            np.savez(left, qpos=np.array([{'unsafe': True}], dtype=object))
            np.savez(right, qpos=np.array([{'unsafe': True}], dtype=object))
            with self.assertRaises(ValueError):
                compare.compare_arrays(left, right)
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                compare.decode(text)

    def test_gzip_bound_and_write_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'rows.json.gz'
            compare.write_gzip(target, {'value': [1, 2, 3]})
            self.assertEqual(compare.read_gzip(target), {'value': [1, 2, 3]})
            with self.assertRaises(FileExistsError):
                compare.write_gzip(target, {})
            target.write_bytes(gzip.compress(json.dumps({'large': 'a'*1000}).encode()))
            with self.assertRaises(ValueError):
                compare.read_gzip(target, 100)

    def test_foreign_tape_rejected_before_native_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'episode.json.gz'
            target.write_bytes(gzip.compress(b'{}'))
            with self.assertRaisesRegex(ValueError, 'exact preserved'):
                compare.load_tape(target)

    def test_dense_case_contract_and_step_statistics(self):
        self.assertEqual(len(compare.CASES), 5)
        self.assertIn('feedback-no-pulse', compare.CASES)
        self.assertIn('feedback-pulse', compare.CASES)
        self.assertIn('qfrc_applied', compare.NATIVE_ARRAYS)
        self.assertIn('retention_lost', compare.SCORER_FIELDS)
        result = compare.summary([.01, .02, .03])
        self.assertEqual(result['p50_s'], .02)
        self.assertEqual(result['over_nominal_20ms'], 1)

    def test_predeclared_adoption_requires_both_clocks_and_both_pairs(self):
        def row(variant, wall, cpu):
            return {'variant': variant, 'status': 'complete',
                    'step_wall': {'mean_s': wall, 'sum_s': wall*1000, 'count': 1000},
                    'step_cpu': {'mean_s': cpu, 'sum_s': cpu*1000, 'count': 1000}}
        runs = [row('baseline', .01, .01), row('candidate', .009, .009),
                row('candidate', .009, .009), row('baseline', .01, .01)]
        self.assertTrue(compare.adoption_result(runs, True)['eligible_under_predeclared_rule'])
        self.assertFalse(compare.adoption_result(runs, False)['eligible_under_predeclared_rule'])
        runs[2] = row('candidate', .0101, .008)
        self.assertFalse(compare.adoption_result(runs, True)['eligible_under_predeclared_rule'])
        runs[2] = row('candidate', .009, .011)
        self.assertFalse(compare.adoption_result(runs, True)['eligible_under_predeclared_rule'])
        runs[2] = row('candidate', .0099, .0099)
        runs[1] = row('candidate', .0099, .0099)
        self.assertFalse(compare.adoption_result(runs, True)['eligible_under_predeclared_rule'])

    def test_every_timeout_retained_without_retries_or_native_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {}
            for variant in ('baseline', 'candidate'):
                roots[variant] = root/variant
                folder = roots[variant]/'src/dvidia_training'
                folder.mkdir(parents=True)
                for name in compare.SOURCE_FILES:
                    (folder/name).write_text(variant if name == 'piper_task.py' else name)
            episode = root/'old-episode.gz'
            episode.write_bytes(b'immutable-tape')
            python = root/'venv-python'
            python.write_bytes(b'immutable-python')
            args = SimpleNamespace(output=root/'evidence', pairs=2,
                                   baseline_root=roots['baseline'], candidate_root=roots['candidate'],
                                   episode=episode, python=python, assets=root/'assets', worker_timeout=1.)
            tape = ({'seed': 0, 'config': {}}, [], compare.identity(episode.read_bytes()))
            with patch.object(compare, 'load_tape', return_value=tape), \
                    patch.object(compare.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 1)) as runner, \
                    contextlib.redirect_stdout(io.StringIO()):
                report = compare.run(args)
            self.assertEqual(runner.call_count, 14)
            self.assertEqual(len(report['attempts']), 14)
            self.assertTrue(all('TimeoutExpired' in row['error'] for row in report['attempts']))
            self.assertFalse(report['all_passed'])
            self.assertFalse(report['adoption']['eligible_under_predeclared_rule'])
            self.assertEqual(len(list(args.output.glob('*/started.json'))), 14)
            self.assertEqual(len(list(args.output.glob('*/process.json'))), 14)
            self.assertTrue((args.output/'comparison.json').is_file())


if __name__ == '__main__':
    unittest.main()
