"""Pure numeric contract tests; no native simulation or model fitting.

Set DVIDIA_PREFLIGHT_PIPER_XML to the existing hash-pinned piper.xml to run
source/XML binding tests too. Those tests do not load meshes or MuJoCo.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('camera_action_v02_numerical_preflight', HERE/'preflight.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
REPO = HERE.parents[1]


class NumericalTests(unittest.TestCase):
    def setUp(self):
        import ast
        self.limiter = p.Limiter(compile(ast.parse(p.EXPECTED_UPDATES), '<unit-numeric-update>', 'exec'))
        self.middle = np.asarray(p.LIMITS).mean(axis=1)
        self.reference = p.reference(self.middle)
        self.target = p.action(self.middle)

    def test_stationary_reference_is_stationary(self):
        actual = self.limiter.trajectory(self.reference, self.target)
        np.testing.assert_array_equal(actual, np.repeat(self.middle[None], 20, axis=0))

    def test_twenty_monotone_reference_steps_keep_original_rates(self):
        target = p.action(np.asarray(p.LIMITS)[:, 1])
        actual = self.limiter.trajectory(self.reference, target)
        expected = self.middle[None] + np.arange(1, 21)[:, None]*np.asarray([.0015]*6+[.00005])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.)
        np.testing.assert_allclose(actual[-1]-self.middle, [.03]*6+[.001], atol=1e-12, rtol=0.)

    def test_inside_one_microstep_target_reaches_and_holds(self):
        target = p.action(self.middle+np.asarray([.0005]*6+[.00001]))
        actual = self.limiter.trajectory(self.reference, target)
        np.testing.assert_allclose(actual, np.repeat(p._action(target)[None], 20, axis=0), atol=1e-12, rtol=0.)

    def test_both_bound_directions_and_mixed_axes_equivalent(self):
        for row in p.cases()[:146]:
            with self.subTest(case=row['case_id']):
                self.assertTrue(p.compare_case(self.limiter, row)['passed'])

    def test_generated_vectors_are_deterministic_and_separate_from_episodes(self):
        first, second = p.cases(), p.cases()
        self.assertEqual(p.digest(first), p.digest(second))
        self.assertEqual(len(first), 4242)
        self.assertEqual(len({r['case_id'] for r in first}), len(first))
        self.assertTrue(all(r['usage'] == 'numerical_training_development_only' for r in first))
        for row in first:
            p._reference(row['previous_reference']); p._action(row['teacher_target'])

    def test_random_cases_include_target_reached_and_rate_limited(self):
        for row in p.cases()[146:166] + p.cases()[-20:]:
            self.assertTrue(p.compare_case(self.limiter, row)['passed'])

    def test_all_negative_contract_and_authority_cases_rejected(self):
        negatives = p.negative_cases(self.limiter)
        self.assertGreaterEqual(len(negatives), 24)
        self.assertTrue(all(r['original_and_projection_rejected'] for r in negatives))

    def test_authority_is_previous_applied_reference_not_measured_position(self):
        target = p.action(np.asarray(p.LIMITS)[:, 1])
        altered = self.middle.copy(); altered[0] += .3
        a = self.limiter.trajectory(self.reference, target)
        b = self.limiter.trajectory(p.reference(altered), target)
        self.assertGreater(float(np.max(np.abs(a-b))), .29)
        measured = deepcopy(self.reference); measured['kind'] = 'measured_joint_position'
        with self.assertRaises(ValueError):
            self.limiter.trajectory(measured, target)

    def test_future_or_teacher_reference_cannot_be_substituted(self):
        for authority in ('future_applied_reference', 'teacher_query_target'):
            ref = deepcopy(self.reference); ref['kind'] = authority
            with self.assertRaises(ValueError):
                self.limiter.endpoint(ref, self.target)

    def test_no_silent_joint_reordering(self):
        ref = deepcopy(self.reference); ref['joint_order'] = ref['joint_order'][::-1]
        with self.assertRaises(ValueError):
            self.limiter.trajectory(ref, self.target)

    def test_no_extra_truth_or_reference_inputs(self):
        for field in ('reference.action', 'native_object_position', 'teacher_phase', 'future_observation'):
            ref = deepcopy(self.reference); ref[field] = 0
            with self.assertRaises(ValueError):
                self.limiter.endpoint(ref, self.target)

    def test_no_silent_clipping_of_invalid_targets(self):
        target = deepcopy(self.target); target['gripper_position_m'] = .04
        with self.assertRaises(ValueError):
            self.limiter.endpoint(self.reference, target)

    def test_delta_decode_uses_previous_reference_and_physical_allowance(self):
        target = p.action(np.asarray(p.LIMITS)[:, 0])
        endpoint = self.limiter.endpoint(self.reference, target)
        delta = (endpoint-self.middle)/p.interval_allowance()
        np.testing.assert_allclose(delta, [-1.]*7, atol=1e-12, rtol=0.)
        decoded = self.middle+delta*p.interval_allowance()
        np.testing.assert_allclose(decoded, endpoint, atol=1e-12, rtol=0.)

    def test_microstep_history_not_only_endpoint_compared(self):
        class ReorderedLimiter(p.Limiter):
            calls = 0
            def trajectory(self, *args, **kwargs):
                result = super().trajectory(*args, **kwargs)
                self.calls += 1
                if self.calls == 2:
                    result[[0, 1]] = result[[1, 0]]
                return result
        mutant = ReorderedLimiter(self.limiter.compiled)
        row = p.cases()[3]
        result = p.compare_case(mutant, row)
        self.assertFalse(result['passed'])
        self.assertGreater(max(result['max_abs_error_by_axis_native']), 1e-12)

    def test_inputs_are_not_mutated(self):
        original = deepcopy((self.reference, self.target, p.CONTRACT))
        self.limiter.trajectory(self.reference, self.target)
        self.limiter.endpoint(self.reference, self.target)
        self.assertEqual((self.reference, self.target, p.CONTRACT), original)

    def test_no_native_import(self):
        self.assertFalse(any(n == 'mujoco' or n.startswith('dvidia_training') for n in sys.modules))


@unittest.skipUnless(os.environ.get('DVIDIA_PREFLIGHT_PIPER_XML'), 'Set hash-pinned DVIDIA_PREFLIGHT_PIPER_XML for source/XML tests.')
class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.xml = Path(os.environ['DVIDIA_PREFLIGHT_PIPER_XML'])

    def test_original_sources_and_xml_bind(self):
        compiled, bindings = p.load_bindings(REPO, self.xml)
        self.assertEqual(bindings['base_source_commit'], p.BASE_COMMIT)
        self.assertEqual(bindings['source_sha256'], p.SOURCE_HASHES)
        self.assertEqual(bindings['limiter_source_lines'], [307, 312])
        self.assertTrue(p.compare_case(p.Limiter(compiled), p.cases()[3])['passed'])

    def test_changed_xml_rejected_before_any_step(self):
        with tempfile.TemporaryDirectory() as temp:
            xml = Path(temp)/'piper.xml'
            xml.write_bytes(self.xml.read_bytes().replace(b'joint1', b'joint9', 1))
            with self.assertRaisesRegex(ValueError, 'XML differs'):
                p.load_bindings(REPO, xml)

    def test_changed_native_limiter_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in p.SOURCE_HASHES:
                target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO/name, target)
            target = root/'src/dvidia_training/piper_task.py'
            target.write_text(target.read_text().replace('joint_target_rate_rad_s: float = 1.5', 'joint_target_rate_rad_s: float = 1.6'))
            with self.assertRaisesRegex(ValueError, 'Frozen source hash differs'):
                p.load_bindings(root, self.xml)

    def test_existing_output_refused_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'evidence'; output.mkdir()
            sentinel = output/'report.json'; sentinel.write_text('preserved')
            with self.assertRaisesRegex(ValueError, 'output must be new'):
                p.run(REPO, self.xml, output)
            self.assertEqual(sentinel.read_text(), 'preserved')


if __name__ == '__main__':
    unittest.main()
