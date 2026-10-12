"""Bounded re-probe requires physical travel and original contact/height gates."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from dvidia_training import piper_task as task
from dvidia_training import piper_reacquisition as reacquisition
from dvidia_training import piper_contact_recovery as recovery
from tests.test_piper_grip_guard import Environment, Teacher, observation


def policy(*, qualified=True):
    with patch.object(task, 'PiperTaskPolicy', Teacher):
        result = reacquisition.ReacquisitionPiperTaskPolicy(Environment())
    if qualified:
        result.qualified, result.qualification_count = True, 1
        result.teacher.phase = 'lift'
        result.anchor = [0., 0., -.002]
        result.probe_origin = {'tcp': [.3, 0., .082], 'object': [.3, 0., .08]}
    return result


def low(t, *, rise=.004, tcp=None, both=True, supported=False, joints=None):
    return observation(t, tcp_rise=rise if tcp is None else tcp, object_rise=rise,
                       both=both, supported=supported, joints=joints)


def start_probe(value):
    value(low(0., both=False))
    value(low(.02, both=False))
    for index in range(2, 15):
        value(low(index*.02))
    assert value.memory_stage == 'second_probe'
    return value.second_probe_started


class ReacquisitionTests(unittest.TestCase):
    def test_invalid_parameters_and_mode(self):
        for change in ({'probe_height_m': True}, {'probe_speed_m_s': math.nan},
                       {'probe_timeout_s': 10**1000}, {'maximum_probes': 2},
                       {'low_hold_minimum_rise_m': .006}, {'probe_minimum_tcp_rise_m': .010},
                       {'probe_timeout_s': .2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(reacquisition.ReacquisitionConfig(), **change)
        with self.assertRaises(ValueError):
            reacquisition.ReacquisitionPiperTaskPolicy(Environment(), 'other')
        self.assertEqual(reacquisition.reacquisition_contract()['prior_recovery_contract'], recovery.recovery_contract())
        self.assertFalse(reacquisition.inactive_summary('reacquire')['active'])

    def test_v04_acquisition_refusal_preserved(self):
        value = policy(qualified=False)
        with patch.object(task, 'PiperTaskPolicy', Teacher):
            old = recovery.ContactRecoveryPiperTaskPolicy(Environment())
        for index in range(10):
            row = observation(index*.02, tcp_rise=.004 if index >= 8 else 0.)
            self.assertEqual(value(row), old(row))
        self.assertTrue(value.unchanged_retry_refused)
        self.assertEqual(value.abort_reason, old.abort_reason)
        self.assertEqual(value.second_probe_count, 0)

    def test_second_probe_requires_stable_history_minimum_hold_and_dwell(self):
        value = policy()
        started = start_probe(value)
        self.assertGreaterEqual(started-.02, .22-1e-9)
        self.assertEqual(value.second_probe_count, 1)
        event = next(e for e in value.events if e['type'] == 'second_probe_started')
        self.assertTrue(event['diagnostic']['history_known'])
        self.assertTrue(event['diagnostic']['history_loaded_bilateral_all'])
        self.assertTrue(event['diagnostic']['history_unsupported_all'])
        self.assertGreaterEqual(event['diagnostic']['low_settled_dwell_s'], .1-1e-9)
        self.assertEqual(value.env.pose_requests[-1][2], 0.)

    def test_supported_or_unknown_low_hold_never_gets_second_lift(self):
        for kind in ('support', 'unloaded', 'jitter', 'zero_clearance'):
            with self.subTest(kind=kind):
                value = policy()
                value(low(0., both=False))
                value(low(.02, both=False))
                for index in range(2, 45):
                    row = low(index*.02, supported=kind=='support', rise=0. if kind=='zero_clearance' else .004)
                    if kind == 'unloaded':
                        row['pad_normal_forces_n']['left'] = 0.
                    if kind == 'jitter':
                        row['object_position_m'][0] += .003 if index % 2 else 0.
                    value(row)
                self.assertTrue(value.aborted)
                self.assertEqual(value.second_probe_count, 0)

    def test_full_gate_not_height_alone_allows_resume(self):
        value = policy()
        started = start_probe(value)
        # The absolute original six-mm floor can be met before measured
        # second-probe travel. It cannot dispatch the old full lift yet.
        for index in range(15, 25):
            value(low(index*.02, rise=.007))
        self.assertEqual(value.resume_count, 0)
        self.assertEqual(value.memory_stage, 'second_probe')
        # Finish a bounded 6.5 mm travel and allow the unchanged settling dwell.
        for index in range(25, 42):
            value(low(index*.02, rise=.0105))
        self.assertEqual(value.resume_count, 1)
        self.assertEqual(value.second_probe_resume_count, 1)
        self.assertFalse(value.aborted)
        qualified = next(e for e in value.events if e['type']=='second_probe_qualified')
        self.assertGreaterEqual(qualified['diagnostic']['second_probe_qualification_dwell_s'], .1-1e-9)
        self.assertAlmostEqual(value.teacher.held_offset[2], -.002)

    def test_second_probe_fails_closed_on_loss_support_slip_unknown_or_contact(self):
        for kind in ('loss', 'unloaded', 'support', 'slip', 'unknown', 'collision'):
            with self.subTest(kind=kind):
                value = policy()
                started = start_probe(value)
                row = low(started+.02, joints=[.8]*6)
                if kind == 'loss':
                    row['grasp_contact_counts']['left'] = 0
                elif kind == 'unloaded':
                    row['pad_normal_forces_n']['left'] = 0.
                elif kind == 'support':
                    row['table_object_contact_count'] = 1
                elif kind == 'slip':
                    row['object_position_m'][0] += .004
                elif kind == 'unknown':
                    value.history = []
                else:
                    row['unexpected_contact_count'] = 1
                action = value(row)
                self.assertTrue(value.aborted)
                self.assertEqual(action, {'joint_targets_rad': [.8]*6, 'gripper_position_m': 0.})
                self.assertEqual(value.resume_count, 0)

    def test_support_at_target_cannot_bypass_second_probe_guard(self):
        value = policy()
        started = start_probe(value)
        row = low(started+.02, supported=True)
        row['target_position_m'] = list(row['object_position_m'])
        value(row)
        self.assertTrue(value.aborted)
        self.assertFalse(value.carry_supported)
        self.assertEqual(value.abort_reason, 'second_probe_table_support_returned')

    def test_probe_request_bound_timeout_and_no_second_retry(self):
        value = policy()
        started = start_probe(value)
        for index in range(15, 78):
            value(low(index*.02))
        self.assertTrue(value.aborted)
        self.assertEqual(value.abort_reason, 'second_probe_timeout')
        self.assertEqual(value.second_probe_count, 1)
        goals = [goal for goal, _, _ in value.env.pose_requests]
        self.assertLessEqual(max(goal[2] for goal in goals)-.086, .008+1e-9)
        self.assertEqual(value.resume_count, 0)
        for index in range(78, 95):
            value(low(index*.02, rise=.012))
        self.assertEqual(value.second_probe_count, 1)
        self.assertEqual(value.resume_count, 0)

    def test_qualification_cannot_bypass_second_probe_deadline(self):
        for eligible_step, should_resume in ((55, True), (56, False)):
            with self.subTest(eligible_step=eligible_step):
                value = policy()
                started = start_probe(value)
                for step in range(1, 62):
                    # Loaded relative contact remains stable throughout. The
                    # object/TCP travel gates become eligible at 1.10 or 1.12 s,
                    # so the 100 ms dwell ends at the deadline or one tick late.
                    rise = .0105 if step >= eligible_step else .004
                    action = value(low(started+step*.02, rise=rise, joints=[.7]*6))
                    if value.resume_count or value.aborted:
                        break
                self.assertEqual(bool(value.resume_count), should_resume)
                self.assertEqual(value.aborted, not should_resume)
                if should_resume:
                    event = next(e for e in value.events if e['type']=='second_probe_qualified')
                    self.assertAlmostEqual(event['t_seconds']-started, 1.2)
                else:
                    self.assertEqual(value.abort_reason, 'second_probe_timeout')
                    self.assertEqual(action, {'joint_targets_rad': [.7]*6, 'gripper_position_m': 0.})
                    self.assertNotIn('second_probe_qualified', [e['type'] for e in value.events])
                    self.assertAlmostEqual(value.events[-1]['t_seconds']-started, 1.22)

    def test_high_stable_hold_uses_original_resume_without_second_probe(self):
        value = policy()
        value(low(0., rise=.008, both=False))
        value(low(.02, rise=.008, both=False))
        for index in range(2, 30):
            value(low(index*.02, rise=.008))
        self.assertEqual(value.second_probe_count, 0)
        self.assertEqual(value.resume_count, 1)
        self.assertFalse(value.aborted)


if __name__ == '__main__':
    unittest.main()
