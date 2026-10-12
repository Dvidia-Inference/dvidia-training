"""Causal history, probe ablation and unchanged native model regressions."""
from dataclasses import replace
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dvidia_training import piper_contact_memory as memory
from dvidia_training import piper_grip_guard as guard
from dvidia_training import piper_task as task
from tests.test_piper_grip_guard import Environment, Teacher, observation


def policy(mode='history', *, qualified=False):
    env = Environment()
    with patch.object(task, 'PiperTaskPolicy', Teacher):
        value = memory.ContactMemoryPiperTaskPolicy(env, mode)
    if qualified:
        value.qualified, value.qualification_count = True, 1
        value.teacher.phase = 'lift'
        value.anchor = [0., 0., -.002]
        value.probe_origin = {'tcp': [.3, 0., .082], 'object': [.3, 0., .08]}
    return value


def carry(t, *, offset=0., both=True, supported=False, joints=None):
    value = observation(t, tcp_rise=.008, object_rise=.008, supported=supported, both=both, joints=joints)
    value['object_position_m'][0] += offset
    return value


class ContactMemoryTests(unittest.TestCase):
    def test_parameter_bounds_and_mode_validation(self):
        for changes in ({'history_window_s': True}, {'trend_rate_m_s': 10**1000},
                        {'settled_rate_m_s': float('nan')}, {'maximum_reassessments': 3},
                        {'minimum_history_s': .2}, {'minimum_straightness': 1.1},
                        {'resumed_joint_delta_scale': 1.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(memory.ContactMemoryConfig(), **changes)
        with self.assertRaises(ValueError):
            memory.ContactMemoryPiperTaskPolicy(Environment(), 'guess')
        self.assertEqual(memory.contact_contract(), memory.contact_memory_contract())
        inactive = memory.inactive_summary('history')
        self.assertEqual(inactive['schema'], memory.SCHEMA)
        self.assertFalse(inactive['active'])
        self.assertEqual([inactive[k] for k in ('slowdown_count', 'reassessment_count', 'resume_count')], [0]*3)

    def test_both_modes_reuse_exact_acquisition_actions_and_events(self):
        env = Environment()
        with patch.object(task, 'PiperTaskPolicy', Teacher):
            old = guard.GuardedPiperTaskPolicy(env)
        values = [old, policy('probe_only'), policy('history')]
        series = [observation(0.), observation(.02)]
        series += [observation(t, tcp_rise=.008, object_rise=.008, supported=False) for t in (.12, .14, .16, .18, .20, .22)]
        for row in series:
            actions = [value(row) for value in values]
            self.assertEqual(actions[0], actions[1])
            self.assertEqual(actions[1], actions[2])
        self.assertTrue(all(value.qualified for value in values))
        self.assertEqual([[e['type'] for e in value.events] for value in values], [['probe_started', 'acquisition_qualified']]*3)

    def test_probe_only_removes_carry_monitoring_without_changing_teacher(self):
        value = policy('probe_only', qualified=True)
        for index in range(12):
            action = value(carry(index*.02, offset=index*.004, both=False))
            self.assertEqual(action['joint_targets_rad'], [.1]*6)
        self.assertFalse(value.aborted)
        self.assertEqual(value.events, [])
        self.assertEqual(value.summary()['reassessment_count'], 0)

    def test_short_history_is_unknown_and_stable_rigid_offset_is_not_slip(self):
        value = policy(qualified=True)
        value(carry(0.))
        self.assertFalse(value.diagnostic['history_known'])
        self.assertIsNone(value.diagnostic['history_rate_m_s'])
        self.assertEqual(value.diagnostic['decision'], 'history_unknown_continue')
        for index in range(1, 10):
            value(carry(index*.02))
        self.assertTrue(value.diagnostic['history_known'])
        self.assertEqual(value.diagnostic['history_rate_m_s'], 0.)
        self.assertEqual(value.events, [])

    def test_persistent_trend_slowdown_then_measured_closed_hold_and_stable_resume(self):
        value = policy(qualified=True)
        for index in range(13):
            action = value(carry(index*.02, offset=min(index*.02, .24)*.04, joints=[.3]*6))
        self.assertEqual([e['type'] for e in value.events], ['slowdown_started', 'reassessment_started'])
        slowdown = value.events[0]
        self.assertGreaterEqual(slowdown['diagnostic']['history_rate_m_s'], .025)
        self.assertEqual(action, {'joint_targets_rad': [.3]*6, 'gripper_position_m': 0.})
        for index in range(13, 27):
            action = value(carry(index*.02, offset=.0096, joints=[.8]*6))
        self.assertEqual(value.resume_count, 1)
        self.assertEqual(value.events[-1]['type'], 'carry_resumed')
        self.assertAlmostEqual(value.teacher.held_offset[0], .0096)
        self.assertLess(action['joint_targets_rad'][0], .8)
        self.assertAlmostEqual(action['joint_targets_rad'][0], .8+.45*(.1-.8))
        self.assertFalse(value.aborted)

    def test_oscillation_cannot_settle_by_endpoint_cancellation(self):
        value = policy(qualified=True)
        value(carry(0., both=False))
        value(carry(.02, both=False))
        self.assertEqual(value.reassessment_count, 1)
        for index in range(2, 43):
            # Endpoint can cancel after six samples while the object reverses.
            value(carry(index*.02, offset=.004 if index % 2 else -.004))
        self.assertTrue(value.aborted)
        self.assertEqual(value.resume_count, 0)
        self.assertGreater(value.diagnostic['history_excursion_m'], .0015)
        self.assertEqual(value.abort_reason, 'reassessment_timeout_unsettled_or_unknown')

    def test_contact_loss_cannot_open_unsupported_and_hold_budget_is_bounded(self):
        value = policy(qualified=True)
        first = value(carry(0., both=False))
        self.assertEqual(first['joint_targets_rad'], [.1]*6)
        held = value(carry(.02, both=False, joints=[.4]*6))
        self.assertEqual(held, {'joint_targets_rad': [.4]*6, 'gripper_position_m': 0.})
        for index in range(2, 43):
            self.assertEqual(value(carry(index*.02, both=False))['gripper_position_m'], 0.)
        self.assertTrue(value.aborted)
        self.assertEqual(value.retry_count, 0)
        self.assertEqual(value.resume_count, 0)
        self.assertEqual(value.reassessment_count, 1)

    def test_repeated_losses_exhaust_two_reassessments_without_extra_retry(self):
        value = policy(qualified=True)
        for start in (0., .42):
            value(carry(start, both=False))
            value(carry(start+.02, both=False))
            for index in range(2, 21):
                value(carry(start+index*.02))
        self.assertEqual(value.reassessment_count, 2)
        self.assertEqual(value.resume_count, 2)
        value(carry(.84, both=False))
        action = value(carry(.86, both=False, joints=[.7]*6))
        self.assertEqual(action, {'joint_targets_rad': [.7]*6, 'gripper_position_m': 0.})
        self.assertTrue(value.aborted)
        self.assertIn('reassessment_budget_exhausted', value.abort_reason)
        self.assertEqual(value.reassessment_count, 2)
        self.assertEqual(value.retry_count, 0)

    def test_supported_or_unknown_hold_window_cannot_claim_settling(self):
        for support in (True, False):
            with self.subTest(supported=support):
                value = policy(qualified=True)
                value(carry(0., both=False))
                value(carry(.02, both=False))
                # Sparse time jump produces an unknown window; supported
                # snapshots also cannot stand in for load-bearing settling.
                value(carry(.7, supported=support))
                self.assertFalse(value.diagnostic['history_known'])
                value(carry(.84, supported=support))
                self.assertEqual(value.resume_count, 0)
                self.assertTrue(value.aborted)

    def test_history_uses_tcp_local_offsets_and_past_samples_only(self):
        value = policy(qualified=True)
        for index in range(8):
            row = carry(index*.02)
            angle = index*.1
            row['tcp_quaternion_wxyz'] = [math.cos(angle/2), 0., 0., math.sin(angle/2)]
            row['object_position_m'] = [row['tcp_position_m'][0]+.01*math.cos(angle),
                                        row['tcp_position_m'][1]+.01*math.sin(angle), row['tcp_position_m'][2]-.002]
            value(row)
        self.assertLess(value.diagnostic['history_excursion_m'], 1e-12)
        self.assertLess(value.diagnostic['history_rate_m_s'], 1e-12)
        self.assertTrue(all(item['t'] <= row['t_seconds'] for item in value.history))


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set DVIDIA_ROBOT_PROFILE_ASSETS to pinned assets')
class NativeContactMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.moderate = task.run_episode(cls.assets, replace(task.PiperTaskConfig(), friction=.55), seed=921707, policy_mode='history')
        cls.low = task.run_episode(cls.assets, replace(task.PiperTaskConfig(), friction=.15), seed=921707, policy_mode='history')

    def test_development_moderate_fixture_can_hold_resume_and_complete(self):
        result = self.moderate
        self.assertTrue(result['valid'], result['error'])
        self.assertTrue(result['success'], result['metrics'])
        self.assertEqual([result['grip_guard'][k] for k in ('slowdown_count', 'reassessment_count', 'resume_count')], [1, 1, 1])
        events = result['grip_guard']['events']
        self.assertEqual([e['type'] for e in events], ['probe_started', 'acquisition_qualified', 'slowdown_started', 'reassessment_started', 'carry_resumed'])
        resume = events[-1]
        self.assertGreaterEqual(resume['diagnostic']['settled_dwell_seconds']+1e-9, .1)
        self.assertLessEqual(resume['diagnostic']['history_excursion_m'], .0015)
        self.assertTrue(result['metrics']['carry_complete'])
        self.assertFalse(result['metrics']['retention_lost'])

    def test_low_friction_retains_acquisition_failure_with_supported_retry(self):
        result = self.low
        self.assertTrue(result['valid'], result['error'])
        self.assertFalse(result['success'])
        self.assertEqual(result['grip_guard']['retry_count'], 1)
        self.assertFalse(result['grip_guard']['acquisition_qualified'])
        self.assertTrue(result['grip_guard']['aborted'])
        self.assertEqual(result['grip_guard']['reassessment_count'], 0)
        self.assertTrue(all(row['gripper_position_m'] == 0. for row in result['actions'] if row['phase'] == 'guard_abort'))

    def test_full_duration_and_model_dynamics_unchanged(self):
        for result, friction in ((self.moderate, .55), (self.low, .15)):
            self.assertEqual(result['compiled_model_sha256'], task.sim.model_digest(task.build_task_model(self.assets, replace(task.PiperTaskConfig(), friction=friction))))
            self.assertEqual(len(result['actions']), 1000)
            self.assertEqual(len(result['trajectory']), 1001)
            self.assertIsNone(result['trajectory'][-1]['guard_diagnostic'])
            self.assertEqual(result['native_servo_gains'], [80., 80., 80., 40., 10., 10., 40.])
            self.assertEqual(result['warnings'], [])
            self.assertEqual(result['metrics']['unexpected_contact_native_ticks'], 0)
            self.assertEqual(set(result['policy_implementation_sha256']), {'piper_grip_guard.py', 'piper_contact_memory.py'})


if __name__ == '__main__':
    unittest.main()
