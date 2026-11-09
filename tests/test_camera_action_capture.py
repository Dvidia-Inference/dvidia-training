"""Data-only rollout tests. No renderer, native physics or hardware commands."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import action_demonstration as tape
from dvidia_training import camera_action_capture as capture
from dvidia_training import piper_rgbd_tracker as tracker
from tests.test_action_demonstration import fixture

CASE = dict(id='test-case', group_id='test-start', seed=4,
            object_position_m=[.30, -.03, .082], goal_position_m=[.35, .08, .08])
ACTION = dict(joint_targets_rad=[.01, .5, -.5, 0., 0., 0.], gripper_position_m=.024)
STUDENT = dict(joint_targets_rad=[.03, .5, -.5, 0., 0., 0.], gripper_position_m=.021)


class FakeSession:
    instances = []
    fail_at = None

    def __init__(self, assets, config, case):
        self.config, self.case = config, case
        self.position = 0; self.teacher_calls = 0; self.closed = False
        self.demo, self.assets = fixture(round(config.duration_s / .02))
        self.header = {key: deepcopy(self.demo[key]) for key in tape.HEADER}
        self.initialized = False; self.last_camera = None
        self.latest_rgbd = (np.zeros((240, 320, 3), dtype=np.uint8), np.ones((240, 320), dtype=np.float32))
        self.instances.append(self)

    def open(self):
        if self.fail_at == 'open':
            raise RuntimeError('Authored setup failure')
        self.initialized = True
        return self.header, deepcopy(self.demo['initial_applied_targets']), {'authored_test_double': True}

    def _snapshot(self):
        snapshot = deepcopy(self.demo['initial'] if self.position == 0 else self.demo['steps'][self.position-1]['after'])
        for name in tracker.SIGNALS:
            snapshot['signals'][name] = capture.base._missing()
        return snapshot

    def sample(self, index):
        if self.fail_at == 'sample' and self.position == 4:
            raise RuntimeError('Authored camera failure')
        snapshot = self._snapshot()
        camera = snapshot['camera']; old_index = camera['value']['index']
        frames = {}
        for channel in ('rgb', 'depth'):
            old_name = camera['value'][channel]['path']
            name = old_name.replace(f'{old_index:04}', f'{index:04}')
            body = self.assets[old_name]
            camera['value'][channel] = dict(path=name, bytes=len(body), sha256=sha256(body).hexdigest())
            frames[name] = body
        camera['value']['index'] = snapshot['vision_frame_index'] = index
        self.last_camera = deepcopy(camera)
        return snapshot, frames, {'policy_observation': {'test': True}}, {'render_encode_wall_seconds': 0.}

    def measured(self, previous_camera):
        snapshot = self._snapshot()
        snapshot['camera'] = capture._sample_at(previous_camera, self.position*.02)
        snapshot['vision_frame_index'] = previous_camera['value']['index']
        return snapshot, {}, {'policy_observation': {'test': True}}, {'render_encode_wall_seconds': 0.}

    def decide(self):
        self.teacher_calls += 1
        return deepcopy(ACTION), {'phase': 'fake_teacher'}

    def advance(self, action):
        before = self.position*.02
        self.position += 1
        if self.fail_at == 'advance' and self.position == 3:
            raise RuntimeError('Authored partial interval failure')
        ramped = deepcopy(action); ramped['joint_targets_rad'][0] += .001
        return dict(interval_start_seconds=before, interval_end_seconds=self.position*.02,
                    ramped_targets=ramped, native_position_controls=[*ramped['joint_targets_rad'],ramped['gripper_position_m']],
                    rate_limited=True, scope='final_native_tick_only')

    def status(self):
        return dict(physics_attempted=False, native_initialized=False,
            actual_native_elapsed_seconds=self.position*.02, native_info={'unexpected_contact_native_ticks': 0},
            authored_test_double=True)

    def native_evidence(self):
        return {'test_only': np.zeros((self.position*20,))}, {'test_only': True}

    def anchor_error(self, estimate):
        return .001 if estimate['status'] == 'available' else None

    def close(self):
        self.closed = True


def fake_estimate(session, snapshot, calibration, negative=None):
    seconds = snapshot['timestamp_seconds']
    status = 'missing' if negative in ('initial_missing', 'initial_stale', 'altered_calibration') else 'available'
    estimate = tracker._empty(snapshot['vision_frame_index'], seconds, seconds, seconds)
    estimate['status'] = status; estimate['age_seconds'] = 0.
    if status == 'available':
        values = dict(object_position_estimate_m=[.30, -.03, .08],
            object_quaternion_estimate_wxyz=[1., 0., 0., 0.], object_position_uncertainty_m=[.001],
            object_orientation_uncertainty_rad=[.01])
        for name, value in values.items():
            estimate[name] = value[0] if 'uncertainty' in name else value
            estimate['signals'][name] = capture.base._sample(value, seconds)
    return estimate, dict(estimator_wall_seconds=0., input_calibration_sha256=tape.digest(calibration), negative_control=negative)


class CameraCaptureTests(unittest.TestCase):
    def setUp(self):
        FakeSession.instances = []; FakeSession.fail_at = None
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        for target, value in (('_Session', FakeSession), ('_fresh_estimate', fake_estimate),
                              ('_native_episode', lambda *args: {'test_only': True}),
                              ('_score', lambda *args: dict(valid=False, success=False, reason='authored_fixture_not_physics'))):
            self.enterContext(patch.object(capture, target, value))

    def run_case(self, name='run', **kwargs):
        root = Path(self.temp.name) / name
        report = capture.run_rollout('unused', root, case=CASE, mode=kwargs.pop('mode', 'teacher'),
                                     duration_s=kwargs.pop('duration_s', .2), **kwargs)
        return root, report

    def test_25hz_camera_50hz_measurements_and_unique_frame_budget(self):
        self.assertEqual(capture.RolloutConfig().duration_s, 20.)
        self.assertEqual(20*25+1, 501)
        root, report = self.run_case()
        self.assertEqual(report['status'], 'complete'); self.assertEqual(report['camera_pairs'], 6)
        recording = capture.load_rollout(root)
        self.assertEqual(len(recording['steps']), 10)
        for i, step in enumerate(recording['steps']):
            self.assertAlmostEqual(step['before']['timestamp_seconds'], i*.02)
            camera = step['before']['camera']
            self.assertEqual(camera['value']['index'], i//2)
            self.assertAlmostEqual(camera['captured_at_seconds'], (i//2)*.04)
            self.assertAlmostEqual(camera['age_seconds'], (i % 2)*.02)
            for name in tracker.SIGNALS:
                self.assertEqual(step['before']['signals'][name]['status'], 'missing')
        self.assertEqual(len(list((root / 'frames').iterdir())), 12)
        self.assertFalse(report['strict_success'])

    def test_queries_once_per_boundary_actual_mixed_authority_separate(self):
        observed = []
        def student(context):
            self.assertEqual(set(context), {'snapshot', 'estimate', 'previous_applied', 'goal_position_m'})
            observed.append(deepcopy(context)); return deepcopy(STUDENT)
        root, report = self.run_case(mode='mixed', policy=student, duration_s=.4)
        self.assertEqual(report['teacher_queries'], 20); self.assertEqual(report['student_calls'], 20)
        self.assertEqual(FakeSession.instances[-1].teacher_calls, 20)
        recording = capture.load_rollout(root)
        authorities = {step['command']['authority'] for step in recording['steps']}
        self.assertEqual(authorities, {'student', 'teacher'})
        for step in recording['steps']:
            query = json.loads((root / step['teacher_query']['path']).read_text())
            self.assertEqual(query['action'], ACTION)
            expected = ACTION if step['command']['authority'] == 'teacher' else STUDENT
            self.assertEqual(step['command']['issued'], expected)
            self.assertNotIn('reference', step)
        self.assertEqual(observed[1]['previous_applied'], recording['steps'][0]['applied']['ramped_targets'])
        other, _ = self.run_case('other', mode='mixed', policy=student, duration_s=.4)
        self.assertEqual([s['command']['authority'] for s in recording['steps']],
                         [s['command']['authority'] for s in capture.load_rollout(other)['steps']])

    def test_student_and_fixed_have_no_teacher_queries(self):
        for mode in ('student', 'fixed'):
            root, report = self.run_case(mode, mode=mode, policy=lambda context: deepcopy(STUDENT))
            self.assertEqual(report['teacher_queries'], 0)
            self.assertFalse((root / 'teacher').exists())
            self.assertEqual(FakeSession.instances[-1].teacher_calls, 0)
            self.assertTrue(all(s['teacher_query'] is None for s in capture.load_rollout(root)['steps']))

    def test_initial_missing_refuses_without_any_native_action_or_teacher_query(self):
        for control in ('initial_missing', 'initial_stale', 'altered_calibration'):
            root, report = self.run_case(control, mode='mixed', policy=lambda _: deepcopy(STUDENT), negative_control=control)
            self.assertEqual(report['status'], 'refused'); self.assertEqual(report['started_intervals'], 0)
            self.assertEqual(report['teacher_queries'], 0); self.assertEqual(report['student_calls'], 0)
            self.assertEqual(FakeSession.instances[-1].position, 0)
            self.assertFalse(report['initial_anchor_available'])
            self.assertTrue((root / 'perception/0000.json').exists())
            capture.inspect_rollout(root)

    def test_teacher_diagnostic_retains_missing_and_continues(self):
        root, report = self.run_case(negative_control='initial_missing')
        self.assertEqual(report['status'], 'complete'); self.assertFalse(report['initial_anchor_available'])
        first = json.loads((root / 'perception/0000.json').read_text())
        second = json.loads((root / 'perception/0001.json').read_text())
        third = json.loads((root / 'perception/0002.json').read_text())
        self.assertEqual(first['estimate']['status'], 'missing')
        self.assertEqual(second['negative_control'], 'initial_missing')
        self.assertEqual(third['estimate']['status'], 'available')

    def test_open_gripper_records_requested_and_issued_truthfully(self):
        root, report = self.run_case(mode='student', policy=lambda _: deepcopy(STUDENT), negative_control='open_gripper')
        for step in capture.load_rollout(root)['steps']:
            self.assertEqual(step['command']['requested']['gripper_position_m'], STUDENT['gripper_position_m'])
            self.assertEqual(step['command']['issued']['gripper_position_m'], .035)
            self.assertEqual(step['applied']['native_position_controls'][-1], .035)
            self.assertEqual(step['command']['issue_reason'], 'open_gripper_negative_control')

    def test_interrupt_failure_and_uncommitted_interval_are_retained(self):
        root, report = self.run_case(interrupt_after=3)
        self.assertEqual(report['status'], 'interrupted'); self.assertEqual(report['recorded_complete_intervals'], 3)
        self.assertEqual(report['uncommitted_native_intervals'], 0)
        capture.inspect_rollout(root)
        FakeSession.fail_at = 'advance'
        root, report = self.run_case('failed')
        self.assertEqual(report['status'], 'interrupted'); self.assertEqual(report['recorded_complete_intervals'], 2)
        self.assertEqual(report['started_intervals'], 3); self.assertEqual(report['uncommitted_native_intervals'], 1)
        self.assertTrue((root / 'decisions/0002.json').exists())
        self.assertFalse((root / 'transitions/0002.json').exists())
        capture.inspect_rollout(root)

    def test_camera_failure_retains_prior_prefix_and_unique_written_frames(self):
        FakeSession.fail_at = 'sample'
        root, report = self.run_case()
        self.assertEqual(report['recorded_complete_intervals'], 3)
        self.assertEqual(report['uncommitted_native_intervals'], 1)
        self.assertEqual(report['camera_pairs'], 2)
        self.assertTrue(FakeSession.instances[-1].closed)
        capture.inspect_rollout(root)

    def test_setup_failure_is_sealed_and_output_cannot_be_reused(self):
        FakeSession.fail_at = 'open'
        root, report = self.run_case()
        self.assertEqual(report['status'], 'refused')
        capture.inspect_rollout(root)
        with self.assertRaises(FileExistsError):
            self.run_case()

    def test_tampered_sidecar_and_rebound_snapshot_are_rejected(self):
        root, report = self.run_case()
        path = root / 'perception/0000.json'
        document = json.loads(path.read_text()); document['snapshot_sha256'] = '0'*64
        body = tape.canonical(document)+b'\n'; path.write_bytes(body)
        with self.assertRaisesRegex(ValueError, 'byte identity'):
            capture.inspect_rollout(root)
        manifest = json.loads((root / 'integrity.json').read_text())
        identity = dict(bytes=len(body), sha256=sha256(body).hexdigest())
        manifest['files']['perception/0000.json'] = identity
        (root / 'integrity.json').write_bytes(tape.canonical(manifest)+b'\n')
        with self.assertRaisesRegex(ValueError, 'Sidecar binding'):
            capture.inspect_rollout(root)

    def test_reused_estimate_has_historical_time_and_no_missing_forward_fill(self):
        snap = dict(timestamp_seconds=0.,vision_frame_index=0)
        estimate, _ = fake_estimate(None, snap, {})
        aged = capture._estimate_at(estimate, .02)
        self.assertEqual(aged['captured_at_seconds'], 0.)
        self.assertEqual(aged['age_seconds'], .02)
        self.assertEqual(aged['signals']['object_position_estimate_m']['age_seconds'], .02)
        stale = capture._estimate_at(estimate, .12)
        self.assertEqual(stale['status'], 'stale')
        self.assertIsNone(stale['object_position_estimate_m'])
        self.assertIsNone(stale['signals']['object_position_estimate_m']['value'])
        self.assertEqual(estimate['age_seconds'], 0.)

    def test_registered_configuration_rejects_silent_cadence_and_scope_changes(self):
        for changes in ({'width': 160}, {'camera_every_controls': 1}, {'duration_s': .11},
                        {'duration_s': True}, {'duration_s': float('nan')}, {'duration_s': 21.},
                        {'seed': True}, {'interrupt_after': -1}, {'interrupt_after': 1000}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                capture.RolloutConfig(**changes)
        with self.assertRaises(ValueError):
            capture.run_rollout('unused',Path(self.temp.name)/'bad',case={**CASE,'object_position_m':[.3,-.03,.3]},mode='teacher')

    def test_sealed_capture_exports_actual_commands_and_separate_mixed_queries(self):
        from dvidia_training.camera_action_export import export_episode
        root, _ = self.run_case()
        exported = export_episode(root)
        self.assertEqual(exported['features'].shape, (10, 269))
        np.testing.assert_array_equal(exported['targets'], np.tile(
            [*ACTION['joint_targets_rad'], ACTION['gripper_position_m']], (10, 1)))
        root, _ = self.run_case('mixed-export', mode='mixed', policy=lambda _: deepcopy(STUDENT))
        exported = export_episode(root, label_source='teacher_query')
        np.testing.assert_array_equal(exported['targets'], np.tile(
            [*ACTION['joint_targets_rad'], ACTION['gripper_position_m']], (10, 1)))
        with self.assertRaisesRegex(ValueError, 'requires teacher control'):
            export_episode(root, label_source='issued')


if __name__ == '__main__':
    unittest.main()
