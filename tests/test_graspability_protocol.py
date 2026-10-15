"""Independent object-agnostic retention and initialization evidence tests."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dvidia_training import graspability_protocol as grasp


def fixture(ticks=600, origin=.02):
    columns = list(grasp._REQUIRED) + ['object_lowest_z_m']
    rows = []
    for i in range(1, ticks + 1):
        settled = i <= 200
        z = origin if settled else origin + .06
        rows.append([round(i * .002, 6), 0., 0. if settled else 1.,
                     0. if settled else .6, 0. if settled else .4,
                     .4, -.1, z, 2.9 if settled else 0., z - origin])
    return {'valid': True, 'success': True,
            'case': {'id': 'synthetic-audit', 'object_shape': 'easy-block'},
            'metrics': {'simulation_seconds': ticks * .002, 'native_steps': ticks},
            'events': [{'type': 'settled_reference', 't': .4}],
            'native_trace_columns': columns, 'native_trace': rows}


def index(result, key):
    return result['native_trace_columns'].index(key)


class GraspabilityScoreTests(unittest.TestCase):
    def test_seal_expands_complete_matched_factorials_without_simulating(self):
        self.assertEqual(sha256(grasp.FIXTURE.read_bytes()).hexdigest(), grasp.FIXTURE_SHA256)
        development = grasp.protocol('development')
        fresh = grasp.protocol('heldout')
        self.assertEqual(len(development['attempts']), 12)
        self.assertEqual(len(fresh['attempts']), 24)
        self.assertFalse({row['case']['id'] for row in development['attempts']}
                         & {row['case']['id'] for row in fresh['attempts']})
        for plan in (development, fresh):
            ids = [row['id'] for row in plan['attempts']]
            self.assertEqual(len(ids), len(set(ids)))
            for identifier in {row['case']['id'] for row in plan['attempts']}:
                cases = [row['case'] for row in plan['attempts'] if row['case']['id'] == identifier]
                self.assertEqual(len(cases), 6)
                self.assertEqual({(case['object_shape'], case['tool_condition']) for case in cases},
                                 {(shape, tool) for shape in grasp.SHAPES for tool in grasp.TOOL_CONDITIONS})
                for name in ('mass', 'friction', 'object_x', 'object_y', 'replicate_id', 'load_band'):
                    self.assertEqual(len({case[name] for case in cases}), 1)
            for replicate in {row['case']['replicate_id'] for row in plan['attempts']}:
                cases = [row['case'] for row in plan['attempts'] if row['case']['replicate_id'] == replicate]
                self.assertEqual({case['mass'] for case in cases}, {.03, .30})
                for name in ('friction', 'object_x', 'object_y'):
                    self.assertEqual(len({case[name] for case in cases}), 1)
        fresh['criteria']['relative_origin_lift_min_m'] = 0.
        self.assertEqual(grasp.protocol('heldout')['criteria']['relative_origin_lift_min_m'], .035)
        with self.assertRaises(ValueError):
            grasp.protocol('test')

    def test_changed_fixture_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_bytes(grasp.FIXTURE.read_bytes() + b' ')
            with patch.object(grasp, 'FIXTURE', path), self.assertRaises(ValueError):
                grasp.protocol()

    def test_full_retained_lift_uses_native_reference_and_final_500ms(self):
        result = fixture()
        result['events'][0]['object_z_m'] = -100.
        score = grasp.audit_result(result)
        self.assertTrue(score['task_success'])
        self.assertTrue(score['controller_claim_agrees'])
        self.assertEqual(score['reference']['object_z_m'], .02)
        self.assertEqual(score['reference']['window_ticks'], 50)
        self.assertEqual(score['final_window_ticks'], 250)
        self.assertEqual(score['strict_passing_ticks'], 250)
        self.assertAlmostEqual(score['final_window_start_exclusive_s'], .7)
        self.assertAlmostEqual(score['minimum_final_relative_lift_m'], .06)
        self.assertAlmostEqual(score['minimum_final_collision_surface_z_m'], .06)

    def test_body_origin_offset_is_not_object_lift(self):
        low, high = fixture(origin=.02), fixture(origin=.22)
        self.assertTrue(grasp.score_result(low)['task_success'])
        self.assertTrue(grasp.score_result(high)['task_success'])
        for row in high['native_trace'][200:]:
            row[index(high, 'object_z_m')] = .22
        score = grasp.score_result(high)
        self.assertFalse(score['task_success'])
        self.assertEqual(score['relative_lift_failed_ticks'], 250)
        self.assertEqual(score['terminal_relative_lift_m'], 0.)

    def test_one_bad_final_native_tick_fails_even_with_success_claim(self):
        for key, value in [('left_finger_force_n', .01), ('right_finger_force_n', .01),
                           ('world_object_force_n', .01), ('tool_object_force_n', .01),
                           ('object_robot_force_n', 1.01), ('object_z_m', .054)]:
            result = fixture()
            row = result['native_trace'][-127]
            row[index(result, key)] = value
            if key == 'tool_object_force_n':
                row[index(result, 'object_robot_force_n')] += value
            score = grasp.audit_result(result)
            with self.subTest(key=key):
                self.assertFalse(score['task_success'])
                self.assertFalse(score['controller_claim_agrees'])
                self.assertTrue(score['valid'])

    def test_35mm_is_an_inclusive_relative_threshold(self):
        result = fixture(origin=0.)
        for row in result['native_trace'][200:]:
            row[index(result, 'object_z_m')] = .035
        self.assertTrue(grasp.score_result(result)['task_success'])

    def test_reference_requires_unassisted_stable_world_support(self):
        for change in ('world', 'robot', 'drift', 'too_early', 'later', 'missing'):
            result = fixture()
            if change == 'world':
                result['native_trace'][177][index(result, 'world_object_force_n')] = 0.
            elif change == 'robot':
                result['native_trace'][177][index(result, 'object_robot_force_n')] = .01
            elif change == 'drift':
                result['native_trace'][177][index(result, 'object_x_m')] += .0011
            elif change == 'too_early':
                result['events'][0]['t'] = .02
            elif change == 'later':
                result['events'][0]['t'] = .38
            else:
                result['events'] = []
            score = grasp.score_result(result)
            with self.subTest(change=change):
                self.assertTrue(score['valid'])
                self.assertFalse(score['task_success'])

    def test_reference_cannot_be_reset_or_forged_off_native_clock(self):
        for change in ('duplicate', 'off_clock', 'outside'):
            result = fixture()
            if change == 'duplicate':
                result['events'].append({'type': 'settled_reference', 't': .8})
            elif change == 'off_clock':
                result['events'][0]['t'] = .401
            else:
                result['events'][0]['t'] = 4.
            with self.subTest(change=change), self.assertRaises(ValueError):
                grasp.score_result(result)

    def test_short_or_invalid_physics_attempt_never_passes(self):
        short = fixture(ticks=300)
        short['success'] = False
        score = grasp.score_result(short)
        self.assertTrue(score['valid'])
        self.assertFalse(score['final_window_complete'])
        self.assertFalse(score['task_success'])
        self.assertIn('retention_window_incomplete', score['failed_criteria'])
        aborted = fixture(ticks=20)
        aborted['events'] = []
        self.assertFalse(grasp.score_result(aborted)['task_success'])
        invalid = fixture()
        invalid['valid'] = False
        self.assertFalse(grasp.score_result(invalid)['task_success'])

    def test_dense_trace_must_be_complete_finite_and_force_consistent(self):
        for change in ('drop', 'duplicate', 'truncate', 'nan', 'force', 'negative', 'steps', 'column'):
            result = fixture()
            if change == 'drop': result['native_trace'].pop(150)
            elif change == 'duplicate': result['native_trace'][150][0] = result['native_trace'][149][0]
            elif change == 'truncate': result['native_trace'].pop()
            elif change == 'nan': result['native_trace'][150][2] = float('nan')
            elif change == 'force': result['native_trace'][-1][2] = 0.
            elif change == 'negative': result['native_trace'][150][1] = -.1
            elif change == 'steps': result['metrics']['native_steps'] -= 1
            else: result['native_trace_columns'][-1] = result['native_trace_columns'][0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                grasp.score_result(result)

    def test_controller_flag_does_not_change_independent_score(self):
        result = fixture()
        result['success'] = False
        score = grasp.audit_result(result)
        self.assertTrue(score['task_success'])
        self.assertFalse(score['controller_claim_agrees'])
        changed = deepcopy(result)
        changed['case']['object_shape'] = 'current-shoe'
        self.assertEqual(grasp.score_result(result), grasp.score_result(changed))


if __name__ == '__main__':
    unittest.main()
