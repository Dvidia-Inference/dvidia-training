"""Synthetic decision-boundary tests; no native simulation or hardware motion."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dvidia_training import piper_probe as probe
from dvidia_training import piper_task


class Environment:
    def __init__(self, jaw=.012):
        self.config = SimpleNamespace(control_dt=.02, timeout_s=20.,
                                      tool_orientation_wxyz=[1., 0., 0., 0.],
                                      goal_position_tolerance_m=.025)
        self.requests, self.jaw = [], jaw

    def pose_action(self, goal, quaternion, jaw):
        self.requests.append((list(goal), list(quaternion), jaw))
        # A deliberately nonzero last-issued jaw tests freezing rather than
        # silently replacing it with the base guard's default zero target.
        return {'joint_targets_rad': [.1]*6, 'gripper_position_m': self.jaw}


class Teacher:
    def __init__(self, env):
        self.env, self.phase, self.issued_phase, self.ticks = env, 'close', 'close', 35
        self.held_offset, self.grasp_position = None, [.3, 0., .082]
        self.calls = []

    def __call__(self, observation):
        self.calls.append(self.phase)
        if self.phase not in ('approach', 'descend', 'close'):
            raise AssertionError('Full-lift teacher must never be dispatched')
        self.issued_phase = self.phase
        if self.phase == 'close': self.phase, self.ticks = 'lift', 0
        return {'joint_targets_rad': [.1]*6, 'gripper_position_m': self.env.jaw}


def observation(t, *, rise=None, supported=None, both=True, force=.4, slip=0., unexpected=0, joints=.02):
    rise = max(0., min(.010, .020*(t-.02))) if rise is None else rise
    return {'t_seconds': t, 'tcp_position_m': [.3, 0., .082+rise],
            'tcp_quaternion_wxyz': [1., 0., 0., 0.], 'object_position_m': [.3+slip, 0., .08+rise],
            'target_position_m': [.35, .08, .08], 'joint_position_rad': [joints]*6,
            'gripper_joint_position_m': [.02, -.02], 'grasp_contact_counts': {'left': int(both), 'right': int(both)},
            'pad_normal_forces_n': {'left': force if both else 0., 'right': force if both else 0.},
            'table_object_contact_count': int(rise < .006 if supported is None else supported),
            'unexpected_contact_count': unexpected, 'native_actuator_controls': [0.]*6+[.027]}


def policy(variant='evidence', jaw=.012):
    env = Environment(jaw)
    with patch.object(piper_task, 'PiperTaskPolicy', Teacher):
        value = probe.ProbeStopPolicy(env, variant)
    return value, env


def until_admitted(value):
    start = 0 if value.last_observed_s is None else round(value.last_observed_s/.02)+1
    for index in range(start, 40):
        action = value(observation(index*.02))
        if value.admitted: return action
    raise AssertionError('Fixture did not admit within bounded time')


class ProbeStopTests(unittest.TestCase):
    def test_contract_labels_native_scope_and_fair_timer_limit(self):
        contract = probe.probe_contract()
        self.assertEqual(contract['timer_admission_s'], .6)
        self.assertEqual(contract['parameters']['probe_height_m'], .010)
        self.assertEqual(contract['parameters']['probe_speed_m_s'], .020)
        self.assertFalse(contract['physical_robot_ready'])
        self.assertIn('not a fully blind', contract['prefix'])
        for variant in ('blind', None, True):
            with self.assertRaises(ValueError):policy(variant)

    def test_probe_context_is_exact_and_every_goal_stays_within_ten_mm(self):
        value, env = policy()
        value(observation(0.));value(observation(.02));value(observation(.04))
        self.assertEqual(value.probes[0]['start_time_s'], .02)
        self.assertEqual(value.probes[0]['object_origin_m'], observation(.02)['object_position_m'])
        self.assertEqual(value.probes[0]['tcp_origin_m'], observation(.02)['tcp_position_m'])
        self.assertAlmostEqual(env.requests[0][0][2]-.082, .0004)
        self.assertAlmostEqual(env.requests[1][0][2]-.082, .0008)
        self.assertTrue(all(request[0][2] <= .092+1e-12 for request in env.requests))

    def test_evidence_dwell_stops_at_measured_joints_and_freezes_nonzero_jaw(self):
        value, env = policy()
        action = until_admitted(value)
        self.assertEqual(value.status, 'qualified_hold')
        self.assertEqual(action, {'joint_targets_rad': [.02]*6, 'gripper_position_m': .012})
        self.assertGreaterEqual(value.admission_time_s-value.qualification_since+1e-9, .10)
        self.assertEqual(value.teacher.calls, ['close'])
        before_requests = len(env.requests)
        now = value.last_observed_s+.02
        held = value(observation(now, joints=.04))
        self.assertEqual(held, action)
        self.assertEqual(len(env.requests), before_requests)
        self.assertEqual(value.events[-1]['type'], 'evidence_qualified')
        self.assertEqual(value.qualification_checks[-1], {'t_seconds': now, 'qualified': True, 'probe_index': 0})

    def test_timer_admits_bad_probe_at_fixed_clock_then_hold_invalidates(self):
        value, _ = policy('timer')
        value(observation(0.))
        for index in range(1, 32):
            value(observation(index*.02, rise=0., supported=True, both=False))
        self.assertEqual(value.status, 'timer_hold')
        self.assertAlmostEqual(value.admission_time_s-value.probes[0]['start_time_s'], .6)
        self.assertEqual(value.events[-1]['type'], 'timer_admitted')
        self.assertTrue(value.qualification_checks[-1]['qualified'])
        action = value(observation(.64, rise=0., supported=True, both=False))
        self.assertEqual(value.status, 'invalidated_hold')
        self.assertFalse(value.admitted)
        self.assertEqual(action['gripper_position_m'], .012)
        self.assertEqual(value.events[-1]['reason'], 'bilateral_contact_or_force_loss')

    def test_qualification_window_resets_on_weak_force(self):
        value, _ = policy()
        for index in range(20):
            value(observation(index*.02, force=.01 if index == 19 else .4))
        self.assertFalse(value.admitted)
        self.assertIsNone(value.qualification_since)
        until_admitted(value)
        self.assertGreaterEqual(value.qualification_since, .4)
        self.assertGreaterEqual(value.admission_time_s-value.qualification_since+1e-9, .1)

    def test_hold_invalidates_on_first_contact_force_support_slip_or_collision_failure(self):
        for changes, reason in (({'both': False}, 'bilateral_contact_or_force_loss'),
                                ({'force': .01}, 'bilateral_contact_or_force_loss'),
                                ({'supported': True}, 'support_returned'),
                                ({'slip': .0031}, 'relative_slip'),
                                ({'unexpected': 1}, 'unexpected_contact'),
                                ({'rise': .005, 'supported': False}, 'probe_height_not_retained')):
            with self.subTest(changes=changes):
                value, env = policy();until_admitted(value)
                action = value(observation(value.last_observed_s+.02, joints=.05, **changes))
                self.assertEqual(value.abort_reason, reason)
                self.assertEqual(action, {'joint_targets_rad': [.05]*6, 'gripper_position_m': .012})
                self.assertFalse(value.admitted)
                held = value(observation(value.last_observed_s+.02))
                self.assertEqual(held, action)
                self.assertEqual(value.teacher.calls, ['close'])

    def test_initial_unexpected_contact_uses_actual_control_not_zero(self):
        value, _ = policy()
        action = value(observation(0., unexpected=1))
        self.assertEqual(action['joint_targets_rad'], [.02]*6)
        self.assertEqual(action['gripper_position_m'], .027)
        self.assertEqual(value.teacher.calls, [])
        self.assertEqual(value.status, 'invalidated_hold')

    def test_original_supported_retry_is_bounded_and_preserves_probe_context(self):
        value, _ = policy()
        value(observation(0.));value(observation(.02))
        value.diagnostic = value._diagnose(observation(.02))
        first = value._alarm(observation(.02), 'probe_timeout')
        self.assertEqual(value.retry_count, 1)
        self.assertEqual(value.state, 'retry_open')
        self.assertEqual(first['gripper_position_m'], .012)
        self.assertEqual(value.probes[0]['disposition'], 'interrupted')
        second = value._alarm(observation(.02), 'probe_timeout')
        self.assertEqual(value.retry_count, 1)
        self.assertEqual(value.status, 'invalidated_hold')
        self.assertEqual(second['gripper_position_m'], .012)
        self.assertIn('retry_budget_exhausted', value.abort_reason)

    def test_final_observation_checks_validity_without_an_extra_control(self):
        value, _ = policy();until_admitted(value)
        count = len(value.requested_controls)
        result = value.observe_final(observation(value.last_observed_s+.02, both=False))
        self.assertEqual(len(value.requested_controls), count)
        self.assertEqual(result['status'], 'invalidated_hold')
        self.assertTrue(result['final_observed'])
        with self.assertRaises(ValueError):value(observation(value.last_observed_s+.02))
        with self.assertRaises(ValueError):value.observe_final(observation(value.last_observed_s+.02))

    def test_fixed_budget_holds_only_and_accepts_native_terminal_roundoff(self):
        value, env = policy();until_admitted(value)
        first_held = deepcopy(value.last_requested)
        pose_request_count = len(env.requests)
        for index in range(round(value.last_observed_s/.02)+1, 1000):
            self.assertEqual(value(observation(index*.02)), first_held)
        self.assertEqual(len(value.requested_controls), 1000)
        self.assertEqual(len(env.requests), pose_request_count)
        self.assertEqual(value.teacher.calls, ['close'])
        with self.assertRaises(ValueError):value(observation(20.))
        result = value.observe_final(observation(20.00000000000146))
        self.assertTrue(result['admitted'])
        self.assertTrue(result['final_observed'])
        self.assertEqual(len(result['requested_controls']), 1000)
        # The numeric tolerance is much smaller than even one extra 1 ms tick.
        other, _ = policy()
        other.last_observed_s = 19.98
        with self.assertRaises(ValueError):other.observe_final(observation(20.00000002))

    def test_no_admission_times_out_without_a_full_lift_or_more_squeeze(self):
        value, _ = policy()
        value.timeout_s, value.command_budget = 1., 50
        for index in range(50):
            action = value(observation(index*.02, rise=0., both=False, supported=True))
        self.assertEqual(value.status, 'timed_out_hold')
        self.assertEqual(value.abort_reason, 'episode_acquisition_timeout')
        self.assertFalse(value.ever_admitted)
        self.assertEqual(action['joint_targets_rad'], [.02]*6)
        self.assertEqual(action['gripper_position_m'], .012)
        self.assertEqual(value.teacher.calls, ['close'])
        final = value.observe_final(observation(1., rise=0., both=False, supported=True))
        self.assertEqual(final['status'], 'timed_out_hold')
        self.assertEqual(len(final['requested_controls']), 50)

    def test_observation_clock_and_nonfinite_or_malformed_inputs_fail_closed(self):
        for changes in ({'t_seconds': True}, {'t_seconds': float('nan')},
                        {'joint_position_rad': [0.]*5}, {'pad_normal_forces_n': {'left': .4}},
                        {'grasp_contact_counts': {'left': True, 'right': 1}},
                        {'tcp_quaternion_wxyz': [0.]*4}):
            value, _ = policy();bad=observation(0.);bad.update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):value(bad)
            self.assertEqual(value.requested_controls, [])
        value, _ = policy();value(observation(0.))
        for timestamp in (0., .01, .04):
            with self.subTest(timestamp=timestamp),self.assertRaises(ValueError):value(observation(timestamp))
        self.assertEqual(len(value.requested_controls), 1)

    def test_summary_and_returned_controls_do_not_alias_internal_records(self):
        value, _ = policy();action=value(observation(0.));action['joint_targets_rad'][0]=99.
        self.assertEqual(value.requested_controls[0]['joint_targets_rad'][0], .1)
        result=value.summary();result['events'].append({'fake':True});result['requested_controls'][0]['gripper_position_m']=0.
        self.assertEqual(value.events, [])
        self.assertEqual(value.requested_controls[0]['gripper_position_m'], .012)


if __name__ == '__main__':unittest.main()
