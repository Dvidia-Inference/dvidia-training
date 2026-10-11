"""Causal grasp checks, measured-target stopping and bounded recovery."""
from dataclasses import replace
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dvidia_training import piper_grip_guard as guard
from dvidia_training import piper_task as task


def observation(t, *, tcp_rise=0., object_rise=0., supported=True, both=True, joints=None, opened=False):
    return {'t_seconds': t, 'tcp_position_m': [.3, 0., .082+tcp_rise],
            'tcp_quaternion_wxyz': [1., 0., 0., 0.], 'object_position_m': [.3, 0., .08+object_rise],
            'target_position_m': [.35, .08, .08], 'joint_position_rad': joints or [.02]*6,
            'gripper_joint_position_m': [.035, -.035] if opened else [.02, -.02],
            'grasp_contact_counts': {'left': int(both), 'right': int(both)},
            'pad_normal_forces_n': {'left': .4 if both else 0., 'right': .4 if both else 0.},
            'table_object_contact_count': int(supported), 'unexpected_contact_count': 0}


class Teacher:
    def __init__(self, env):
        self.env, self.phase, self.ticks, self.issued_phase = env, 'close', 35, 'close'
        self.grasp_position, self.held_offset = [.3, 0., .082], None

    def __call__(self, obs):
        self.issued_phase = self.phase
        if self.phase == 'close':
            self.phase, self.ticks, self.held_offset = 'lift', 0, [0., 0., -.002]
        elif self.phase == 'descend':
            self.phase, self.ticks = 'close', 35
        return {'joint_targets_rad': [.1]*6, 'gripper_position_m': 0.}


class Environment:
    def __init__(self):
        # Deliberately no mass, friction, future observations or scorer flags.
        self.config = SimpleNamespace(control_dt=.02, tool_orientation_wxyz=[1., 0., 0., 0.],
                                      goal_position_tolerance_m=.025)
        self.pose_requests = []

    def pose_action(self, goal, quat, jaw):
        self.pose_requests.append((list(goal), list(quat), jaw))
        return {'joint_targets_rad': [.1]*6, 'gripper_position_m': jaw}


def policy():
    env = Environment()
    with patch.object(task, 'PiperTaskPolicy', Teacher):
        value = guard.GuardedPiperTaskPolicy(env)
    value(observation(0.))  # The original closure action schedules a probe.
    value(observation(.02))
    return value, env


class GripGuardTests(unittest.TestCase):
    def test_parameters_are_finite_and_mode_validation_precedes_assets(self):
        for changes in ({'probe_height_m': True}, {'probe_height_m': float('nan')},
                        {'probe_height_m': 10**1000}, {'consecutive_alarm_samples': 1},
                        {'max_retries': 2}, {'probe_min_object_rise_m': .02}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(guard.GuardConfig(), **changes)
        with patch.object(task.robot_profiles, 'verify_assets') as verify, self.assertRaises(ValueError):
            task.run_episode('unused', policy_mode='guess')
        verify.assert_not_called()
        self.assertFalse(guard.inactive_summary('baseline')['active'])
        self.assertEqual(guard.guard_contract()['parameters']['max_retries'], 1)

    def test_probe_is_slow_and_cannot_dispatch_the_full_lift_before_qualification(self):
        value, env = policy()
        self.assertEqual(value.state, 'probe')
        self.assertAlmostEqual(env.pose_requests[-1][0][2]-.082, .0004)
        value(observation(.04))
        self.assertAlmostEqual(env.pose_requests[-1][0][2]-.082, .0008)
        self.assertFalse(value.qualified)
        self.assertEqual([event['type'] for event in value.events], ['probe_started'])

    def test_contact_alone_is_not_acquisition_and_probe_timeout_is_bounded(self):
        value, _ = policy()
        value(observation(1.44))
        self.assertFalse(value.qualified)
        self.assertEqual(value.retry_count, 1)
        self.assertEqual(value.events[-2]['reason'], 'probe_timeout')
        self.assertEqual(value.state, 'retry_open')

    def test_qualification_requires_follow_unsupported_bilateral_dwell(self):
        value, _ = policy()
        for time in (.12, .14, .16, .18, .20):
            value(observation(time, tcp_rise=.008, object_rise=.008, supported=False))
            self.assertFalse(value.qualified)
        value(observation(.22, tcp_rise=.008, object_rise=.008, supported=False))
        self.assertTrue(value.qualified)
        self.assertEqual(value.qualification_count, 1)
        self.assertEqual(value.events[-1]['type'], 'acquisition_qualified')
        self.assertGreaterEqual(value.diagnostic['qualification_dwell_seconds']+1e-9, .1)
        self.assertEqual(value.state, 'normal')

    def test_slip_requires_two_snapshots_then_holds_measured_targets_before_opening(self):
        value, env = policy()
        value(observation(.04, tcp_rise=.004, joints=[.4]*6))
        self.assertEqual(value.state, 'probe')
        requests = len(env.pose_requests)
        action = value(observation(.06, tcp_rise=.008, joints=[.5]*6))
        self.assertEqual(action, {'joint_targets_rad': [.5]*6, 'gripper_position_m': 0.})
        self.assertEqual(len(env.pose_requests), requests)
        self.assertEqual(value.retry_count, 1)
        self.assertEqual(value.events[-2]['type'], 'alarm')
        self.assertEqual(value.events[-2]['reason'], 'object_not_following')
        self.assertEqual(value.events[-2]['t_seconds'], .06)
        self.assertEqual(value(observation(.08))['gripper_position_m'], .035)

    def test_support_loss_during_retry_aborts_closed_and_never_opens_again(self):
        value, _ = policy()
        value(observation(.04, tcp_rise=.004))
        value(observation(.06, tcp_rise=.008))
        action = value(observation(.08, supported=False, joints=[.3]*6))
        self.assertEqual(action, {'joint_targets_rad': [.3]*6, 'gripper_position_m': 0.})
        self.assertTrue(value.aborted)
        self.assertEqual(value.abort_reason, 'support_lost_during_retry')
        later = value(observation(5., opened=True, joints=[.9]*6))
        self.assertEqual(later, action)
        self.assertEqual(value.retry_count, 1)

    def test_second_failed_probe_exhausts_retry_without_another_recenter(self):
        value, _ = policy()
        value(observation(.04, tcp_rise=.004))
        value(observation(.06, tcp_rise=.008))
        value(observation(.08, opened=True, both=False))
        value(observation(.10, tcp_rise=.010, opened=True, both=False))
        self.assertEqual(value.state, 'normal')
        self.assertEqual(value.events[-1]['type'], 'retry_completed')
        value(observation(.12))
        value(observation(.14))
        value(observation(.16))
        value(observation(.18, tcp_rise=.004))
        value(observation(.20, tcp_rise=.008))
        self.assertTrue(value.aborted)
        self.assertEqual(value.retry_count, 1)
        self.assertIn('retry_budget_exhausted', value.abort_reason)
        self.assertEqual(sum(event['type'] == 'retry_started' for event in value.events), 1)

    def test_tcp_local_offset_handles_orientation_instead_of_world_displacement(self):
        value = observation(0.)
        value['tcp_position_m'] = [0., 0., 0.]
        value['tcp_quaternion_wxyz'] = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
        value['object_position_m'] = [0., .01, -.002]
        expected = [.01, 0., -.002]
        self.assertLess(math.dist(guard._local_offset(value), expected), 1e-10)


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set DVIDIA_ROBOT_PROFILE_ASSETS to pinned assets')
class NativeGripGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.nominal = task.run_episode(cls.assets, policy_mode='guarded', seed=881039)
        cls.low = task.run_episode(cls.assets, replace(task.PiperTaskConfig(), friction=.15), policy_mode='guarded', seed=881039)

    def test_default_fixture_qualifies_before_full_lift_and_completes_without_alarm(self):
        result = self.nominal
        self.assertTrue(result['valid'], result['error'])
        self.assertTrue(result['success'], result['metrics'])
        events = result['grip_guard']['events']
        self.assertEqual([event['type'] for event in events], ['probe_started', 'acquisition_qualified'])
        self.assertEqual(result['grip_guard']['retry_count'], 0)
        qualification = events[1]['t_seconds']
        first_full_lift = next(row['t_seconds'] for row in result['trajectory']
                               if row['object_position_m'][2]-result['initial_object_position_m'][2] >= .06)
        self.assertLess(qualification, first_full_lift)
        self.assertTrue(result['metrics']['carry_complete'])
        self.assertFalse(result['metrics']['retention_lost'])
        self.assertEqual(result['metrics']['unexpected_contact_native_ticks'], 0)
        self.assertEqual(result['warnings'], [])

    def test_low_friction_retries_only_supported_then_aborts_closed_for_full_duration(self):
        result = self.low
        self.assertTrue(result['valid'], result['error'])
        self.assertFalse(result['success'])
        self.assertFalse(result['metrics']['lift_seen'])
        self.assertFalse(result['grip_guard']['acquisition_qualified'])
        self.assertTrue(result['grip_guard']['aborted'])
        self.assertEqual(result['grip_guard']['retry_count'], 1)
        self.assertEqual(sum(event['type'] == 'retry_started' for event in result['grip_guard']['events']), 1)
        self.assertIn('retry_budget_exhausted', result['grip_guard']['abort_reason'])
        self.assertEqual(result['metrics']['unexpected_contact_native_ticks'], 0)
        for row in result['trajectory'][:-1]:
            if row['command']['phase'] in ('guard_retry_open', 'guard_retry_recenter'):
                self.assertGreater(row['table_object_contact_count'], 0)
            if row['command']['phase'] == 'guard_abort':
                self.assertEqual(row['command']['gripper_position_m'], 0.)
        abort = next(row for row in result['trajectory'][:-1] if row['command']['phase'] == 'guard_abort')
        self.assertEqual(abort['command']['joint_targets_rad'], abort['joint_position_rad'])
        frozen = abort['command']['joint_targets_rad']
        self.assertTrue(all(row['joint_targets_rad'] == frozen for row in result['actions']
                            if row['phase'] == 'guard_abort'))

    def test_native_models_and_before_command_trace_remain_unchanged(self):
        for result, config in ((self.nominal, task.PiperTaskConfig()), (self.low, replace(task.PiperTaskConfig(), friction=.15))):
            self.assertEqual(result['compiled_model_sha256'], task.sim.model_digest(task.build_task_model(self.assets, config)))
            self.assertAlmostEqual(result['simulated_seconds'], 20., places=8)
            self.assertEqual(len(result['actions']), 1000)
            self.assertEqual(len(result['trajectory']), 1001)
            self.assertEqual(result['grip_guard']['diagnostic_count'], 1000)
            for index, row in enumerate(result['trajectory']):
                if index < 1000:
                    self.assertEqual(row['guard_diagnostic']['t_seconds'], row['t_seconds'])
                    self.assertEqual(row['command'], result['actions'][index])
                    self.assertEqual(row['issued_joint_targets_rad'], result['arm_tape'][index])
                else:
                    self.assertIsNone(row['guard_diagnostic'])
                    self.assertIsNone(row['command'])
            self.assertEqual(result['native_servo_gains'], [80., 80., 80., 40., 10., 10., 40.])
            self.assertEqual(result['native_force_limits'][-1], [-10., 10.])


if __name__ == '__main__':
    unittest.main()
