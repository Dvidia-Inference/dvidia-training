"""Causal refusal and bounded speed restoration without changing native physics."""
from dataclasses import replace
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dvidia_training import piper_contact_recovery as recovery
from dvidia_training import piper_contact_memory as memory
from dvidia_training import piper_task as task
from tests.test_piper_grip_guard import Environment, Teacher, observation
from tests.test_piper_contact_memory import carry


def policy(mode='recovery', *, qualified=False):
    with patch.object(task, 'PiperTaskPolicy', Teacher):
        result = recovery.ContactRecoveryPiperTaskPolicy(Environment(), mode)
    if qualified:
        result.qualified, result.qualification_count = True, 1
        result.teacher.phase = 'lift'
        result.anchor = [0., 0., -.002]
        result.probe_origin = {'tcp': [.3, 0., .082], 'object': [.3, 0., .08]}
    return result


class ContactRecoveryTests(unittest.TestCase):
    def test_config_and_modes_reject_unknown_or_nonfinite_values(self):
        for change in ({'restoration_ramp_s': True}, {'restoration_dwell_s': math.nan},
                       {'acquisition_window_s': 10**1000}, {'acquisition_minimum_span_s': .2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(recovery.RecoveryConfig(), **change)
        with self.assertRaises(ValueError):
            recovery.ContactRecoveryPiperTaskPolicy(Environment(), 'unknown')
        with self.assertRaises(ValueError):
            task.run_episode('unused', record_step_timings=1)
        self.assertEqual(recovery.recovery_contract()['prior_contact_contract'], memory.contact_memory_contract())

    def test_stable_supported_closed_probe_refuses_unchanged_retry(self):
        value = policy()
        for index in range(8):
            value(observation(index*.02))
        value(observation(.16, tcp_rise=.0035))
        action = value(observation(.18, tcp_rise=.004, joints=[.7]*6))
        self.assertEqual(action, {'joint_targets_rad': [.7]*6, 'gripper_position_m': 0.})
        self.assertTrue(value.unchanged_retry_refused)
        self.assertTrue(value.aborted)
        self.assertFalse(value.qualified)
        self.assertEqual(value.abort_reason, 'unchanged_grasp_not_following')
        self.assertEqual(value.retry_count, 0)
        self.assertEqual([e['type'] for e in value.events], ['probe_started', 'alarm', 'unchanged_retry_refused', 'abort'])

    def test_short_history_or_varying_jaw_force_preserves_original_retry(self):
        for kind in ('short', 'jaw', 'force', 'unloaded'):
            with self.subTest(kind=kind):
                value = policy()
                count = 2 if kind == 'short' else 8
                for index in range(count):
                    row = observation(index*.02)
                    if index % 2:
                        if kind == 'jaw':
                            row['gripper_joint_position_m'] = [.022, -.022]
                        elif kind == 'force':
                            row['pad_normal_forces_n']['left'] = .5
                        elif kind == 'unloaded':
                            row['pad_normal_forces_n']['left'] = 0.
                    value(row)
                value(observation(count*.02, tcp_rise=.0035))
                value(observation((count+1)*.02, tcp_rise=.004))
                self.assertFalse(value.unchanged_retry_refused)
                self.assertEqual(value.retry_count, 1)
                self.assertEqual(value.state, 'retry_open')

    def test_exhausted_retry_is_not_counted_as_a_new_refusal(self):
        value = policy()
        for index in range(8):
            value(observation(index*.02))
        value.retry_count = value.config.max_retries
        value(observation(.16, tcp_rise=.0035))
        value(observation(.18, tcp_rise=.004))
        self.assertTrue(value.aborted)
        self.assertEqual(value.abort_reason, 'retry_budget_exhausted:object_not_following')
        self.assertFalse(value.unchanged_retry_refused)
        self.assertNotIn('unchanged_retry_refused', [event['type'] for event in value.events])

    def test_carry_ablation_preserves_original_acquisition_actions_events(self):
        value = policy('recovery_carry')
        with patch.object(task, 'PiperTaskPolicy', Teacher):
            prior = memory.ContactMemoryPiperTaskPolicy(Environment())
        for index in range(10):
            row = observation(index*.02, tcp_rise=.004 if index >= 8 else 0.)
            self.assertEqual(value(row), prior(row))
        self.assertEqual(value.retry_count, prior.retry_count)
        self.assertEqual([(e['type'], e['reason']) for e in value.events], [(e['type'], e['reason']) for e in prior.events])
        self.assertFalse(value.unchanged_retry_refused)

    def test_loaded_stability_restores_scale_with_bounded_ramp_after_hold(self):
        value = policy(qualified=True)
        value(carry(0., both=False))
        value(carry(.02, both=False))
        for index in range(2, 19):
            value(carry(index*.02))
        self.assertEqual(value.resume_count, 1)
        scales = []
        for index in range(19, 48):
            action = value(carry(index*.02, joints=[.2]*6))
            scales.append(value.diagnostic['joint_delta_scale'])
            self.assertAlmostEqual(action['joint_targets_rad'][0], .2+scales[-1]*(.1-.2))
        self.assertAlmostEqual(scales[0], .45)
        self.assertAlmostEqual(scales[-1], 1.)
        self.assertTrue(all(b-a <= .55*.02/.3+1e-9 for a, b in zip(scales, scales[1:])))
        self.assertEqual(value.full_speed_restored_count, 1)
        self.assertEqual(value.restoration_started_count, 1)
        self.assertEqual(value.reassessment_count, 1)
        events = [event for event in value.events if event['type'] in ('speed_restoration_started', 'full_speed_restored')]
        self.assertTrue(events[0]['diagnostic']['restoration_started'])
        self.assertFalse(events[0]['diagnostic']['restoration_complete'])
        self.assertEqual(events[0]['diagnostic']['joint_delta_scale'], .45)
        self.assertTrue(events[1]['diagnostic']['restoration_complete'])
        self.assertAlmostEqual(events[1]['diagnostic']['joint_delta_scale'], 1.)
        self.assertEqual(events[1]['diagnostic']['unscaled_joint_targets_rad'], [.1]*6)

    def test_unstable_window_pauses_ramp_and_new_loss_resets_progress(self):
        value = policy(qualified=True)
        value.resumed = True
        for index in range(16):
            value(carry(index*.02))
        self.assertTrue(value.restoration_started)
        self.assertGreater(value.restoration_progress, 0.)
        prior = value.restoration_progress
        # A 2mm non-monotone excursion fails settling without meeting the
        # original 2.5mm directed-trend alarm. Restoration must pause.
        value(carry(.32, offset=.002))
        self.assertEqual(value.restoration_progress, prior)
        value(carry(.34, both=False))
        value(carry(.36, both=False))
        self.assertEqual(value.memory_stage, 'hold')
        self.assertEqual(value.restoration_progress, 0.)
        self.assertFalse(value.restoration_started)
        self.assertEqual(value.events[-1]['diagnostic']['restoration_progress_seconds'], 0.)
        self.assertFalse(value.events[-1]['diagnostic']['restoration_started'])

    def test_loss_while_restoring_still_holds_closed_and_cannot_resume_supported(self):
        value = policy(qualified=True)
        value.resumed = True
        for index in range(40):
            value(carry(index*.02))
        self.assertTrue(value.restoration_complete)
        value(carry(.80, both=False))
        action = value(carry(.82, both=False, joints=[.4]*6))
        self.assertEqual(action, {'joint_targets_rad': [.4]*6, 'gripper_position_m': 0.})
        for index in range(42, 84):
            value(carry(index*.02, supported=True))
        self.assertTrue(value.aborted)
        self.assertEqual(value.resume_count, 0)


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set DVIDIA_ROBOT_PROFILE_ASSETS to pinned assets')
class NativeContactRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.low = task.run_episode(cls.assets, replace(task.PiperTaskConfig(), friction=.15), seed=921707, policy_mode='recovery', record_step_timings=True)
        cls.moderate = task.run_episode(cls.assets, replace(task.PiperTaskConfig(), friction=.55), seed=921707, policy_mode='recovery')

    def test_refusal_preserves_failed_outcome_and_full_episode(self):
        raw = self.low
        self.assertTrue(raw['valid'], raw['error'])
        self.assertFalse(raw['success'])
        self.assertEqual(raw['grip_guard']['abort_reason'], 'unchanged_grasp_not_following')
        self.assertEqual(raw['grip_guard']['retry_count'], 0)
        self.assertEqual(len(raw['actions']), 1000)
        self.assertEqual(len(raw['trajectory']), 1001)
        timings = raw['timing']['control_step_samples_seconds']
        self.assertEqual(len(timings), 1000)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in timings))
        self.assertEqual(raw['timing']['control_step_seconds'], task.sim._summary(timings))
        self.assertEqual(raw['timing']['missed_control_boundaries'], sum(value > .02 for value in timings))

    def test_restoration_preserves_native_mechanics_and_contact_success(self):
        raw = self.moderate
        self.assertTrue(raw['valid'], raw['error'])
        self.assertTrue(raw['success'], raw['metrics'])
        self.assertGreater(raw['grip_guard']['full_speed_restored_count'], 0)
        self.assertFalse(raw['metrics']['retention_lost'])
        self.assertEqual(raw['native_servo_gains'], [80., 80., 80., 40., 10., 10., 40.])
        self.assertEqual(raw['warnings'], [])
        self.assertEqual(raw['metrics']['unexpected_contact_native_ticks'], 0)
        self.assertNotIn('control_step_samples_seconds', raw['timing'])
        self.assertEqual(raw['compiled_model_sha256'], task.sim.model_digest(task.build_task_model(self.assets, replace(task.PiperTaskConfig(), friction=.55))))


if __name__ == '__main__':
    unittest.main()
