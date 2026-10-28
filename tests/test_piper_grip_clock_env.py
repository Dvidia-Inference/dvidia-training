"""Native-independent fixtures for post-grip eligibility and one-shot faults."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training.piper_feedback_env import FeedbackPiperTaskEnv, NATIVE_FIELDS
from dvidia_training.piper_grip_clock_env import (
    PostGripParameters, PostGripPiperTaskEnv, analyze_post_grip,
)
from dvidia_training.piper_task import PiperTaskConfig


def fixture(**kwargs):
    options = dict(interruption=True, eligibility_dwell_s=.003,
                   delay_s=.002, pulse_duration_s=.004)
    options.update(kwargs)
    env = PostGripPiperTaskEnv('unused', **options)
    env.ready = True
    env.command_targets = np.arange(6, dtype=float)/10
    env.command_gripper = env._normal_jaw = env._applied_jaw = 0.
    env.data = SimpleNamespace(ctrl=np.zeros(7), xpos=np.asarray([[.3, 0., .15]]))
    env.block = 0
    env.initial_object_position = np.asarray([.3, 0., .08])
    env._post_grip_initial_height = .08
    env._last_contacts = ({'left': 1, 'right': 1}, {'left': .1, 'right': .1}, 0, 0, [])
    return env


def append_parent_row(env, **kwargs):
    """Stand in for the unchanged parent's post-physics numeric row emission."""
    if not env.ready:
        return
    env._native_tick += 1
    counts, forces, supported, critical, _ = env._last_contacts
    values = {name: [0.]*width for name, width in NATIVE_FIELDS}
    values.update(t_seconds=[env._native_tick*.001], native_tick=[env._native_tick],
                  object_position_m=env.data.xpos[0].tolist(),
                  grasp_contact_counts=[counts['left'], counts['right']],
                  pad_normal_forces_n=[forces['left'], forces['right']],
                  table_object_contact_count=[supported], unexpected_contact_count=[critical],
                  normal_rate_limited_jaw_target_m=[env._normal_jaw],
                  applied_jaw_target_m=[float(env.data.ctrl[6])],
                  pulse_active=[int(env._pulse_active)],
                  pulse_minimum_opening_m=[env._pulse_floor],
                  pulse_closing_tail=[int(env._closing_tail)])
    env._native_rows.append(tuple(x for name, _ in NATIVE_FIELDS for x in values[name]))


def advance(env, ticks=1):
    with patch.object(FeedbackPiperTaskEnv, '_outcome_tick', append_parent_row):
        for _ in range(ticks):
            env._controls()
            env._outcome_tick(sentinel='unchanged scoring arguments')


class PostGripEnvironmentTests(unittest.TestCase):
    def test_parameters_are_finite_bounded_tick_aligned_and_immutable(self):
        invalid = [dict(interruption=1), dict(force_threshold_n=False),
                   dict(force_threshold_n=float('nan')), dict(force_threshold_n=1001),
                   dict(force_threshold_n=10**1000), dict(lift_threshold_m=-.001),
                   dict(lift_threshold_m=.201), dict(eligibility_dwell_s=0),
                   dict(eligibility_dwell_s=.0015), dict(delay_s=-.001),
                   dict(delay_s=2.001), dict(pulse_duration_s=.501),
                   dict(minimum_opening_m=.03501)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                PostGripPiperTaskEnv('unused', **options)
        env = fixture()
        with self.assertRaises(FrozenInstanceError):
            env.post_grip_parameters.delay_s = .1
        with self.assertRaises(AttributeError):
            env.post_grip_parameters = PostGripParameters()
        with self.assertRaises(ValueError):
            PostGripPiperTaskEnv('unused', PiperTaskConfig(timestep=.002))

    def test_no_eligibility_without_each_required_native_predicate(self):
        cases = [(dict(left=0, right=1), dict(left=.1, right=.1), 0, .15),
                 (dict(left=1, right=0), dict(left=.1, right=.1), 0, .15),
                 (dict(left=1, right=1), dict(left=.099, right=.1), 0, .15),
                 (dict(left=1, right=1), dict(left=.1, right=.099), 0, .15),
                 (dict(left=1, right=1), dict(left=.1, right=.1), 1, .15),
                 (dict(left=1, right=1), dict(left=.1, right=.1), 0, .14)]
        for counts, forces, support, height in cases:
            with self.subTest(counts=counts, forces=forces, support=support, height=height):
                env = fixture()
                env._last_contacts = counts, forces, support, 0, []
                env.data.xpos[0, 2] = height
                advance(env, 12)
                self.assertEqual(env.pulses, ())
                result = analyze_post_grip(env.native_arrays(), env.native_metadata())
                self.assertTrue(result['valid'], result)
                self.assertFalse(result['eligible'])
                self.assertEqual(result['nonactivation_reason'], 'eligibility_not_reached')

    def test_support_breaks_consecutive_dwell_and_delay_has_exact_boundaries(self):
        env = fixture()
        advance(env, 2)
        env._last_contacts = (*env._last_contacts[:2], 1, 0, [])
        advance(env)
        env._last_contacts = (*env._last_contacts[:2], 0, 0, [])
        advance(env, 3)
        self.assertEqual(env._native_tick, 6)
        self.assertEqual(env._pulse_ticks, ((8, 12, .02),))
        self.assertEqual(env.pulses[0].start_s, .008)
        self.assertEqual(env.pulses[0].end_s, .012)
        advance(env, 18)
        arrays, metadata = env.native_arrays(), env.native_metadata()
        self.assertEqual(arrays['native_tick'][arrays['pulse_active'] > 0].tolist(), [9, 10, 11, 12])
        np.testing.assert_array_equal(arrays['pulse_minimum_opening_m'][8:12], [.02]*4)
        self.assertEqual(metadata['post_grip_intervention']['active_native_ticks'], 4)
        self.assertEqual(len(metadata['post_grip_intervention']['events']), 1)
        result = analyze_post_grip(arrays, metadata)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['eligibility_native_tick'], 6)
        self.assertEqual(result['pulse_start_tick'], 8)
        self.assertEqual(result['pulse_end_tick'], 12)
        self.assertEqual(result['first_effect_native_tick'], 9)
        self.assertTrue(result['activated'])
        self.assertTrue(result['effective'])
        self.assertGreater(result['effective_native_ticks'], result['active_native_ticks'])
        self.assertLessEqual(np.max(np.abs(np.diff(arrays['applied_jaw_target_m']))), .00005+1e-15)
        np.testing.assert_array_equal(env.data.ctrl[:6], env.command_targets)
        self.assertEqual(set(arrays), {name for name, _ in NATIVE_FIELDS})

    def test_nominal_latches_eligibility_but_never_schedules_a_fault(self):
        env = fixture(interruption=False)
        advance(env, 12)
        result = analyze_post_grip(env.native_arrays(), env.native_metadata())
        self.assertTrue(result['valid'], result)
        self.assertTrue(result['eligible'])
        self.assertFalse(result['interruption_condition'])
        self.assertFalse(result['scheduled'])
        self.assertFalse(result['activated'])
        self.assertFalse(result['effective'])
        self.assertEqual(result['nonactivation_reason'], 'nominal_condition')
        self.assertEqual(env.pulses, ())

    def test_zero_delay_starts_next_native_row_and_zero_floor_can_be_ineffective(self):
        env = fixture(delay_s=0., minimum_opening_m=0.)
        advance(env, 10)
        result = analyze_post_grip(env.native_arrays(), env.native_metadata())
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['pulse_start_tick'], 3)
        self.assertEqual(env.native_arrays()['native_tick'][env.native_arrays()['pulse_active'] > 0].tolist(), [4, 5, 6, 7])
        self.assertTrue(result['activated'])
        self.assertFalse(result['effective'])

    def test_short_attempt_and_late_eligibility_are_explicitly_reported(self):
        env = fixture()
        advance(env, 3)
        result = analyze_post_grip(env.native_arrays(), env.native_metadata())
        self.assertTrue(result['valid'], result)
        self.assertTrue(result['scheduled'])
        self.assertFalse(result['activated'])
        self.assertEqual(result['nonactivation_reason'], 'episode_ended_before_pulse')
        env = fixture(config=PiperTaskConfig(timeout_s=.1))
        env._last_contacts = (*env._last_contacts[:2], 1, 0, [])
        advance(env, 95)
        env._last_contacts = (*env._last_contacts[:2], 0, 0, [])
        advance(env, 5)
        result = analyze_post_grip(env.native_arrays(), env.native_metadata())
        self.assertTrue(result['valid'], result)
        self.assertTrue(result['eligible'])
        self.assertFalse(result['scheduled'])
        self.assertEqual(result['nonactivation_reason'], 'insufficient_episode_budget')

    def test_reset_clears_one_shot_state_before_settling_and_preserves_return(self):
        env = fixture()
        advance(env, 15)
        expected = {'unchanged': 'observation'}, {'unchanged': 'info'}
        def reset_parent(instance, seed):
            self.assertEqual(seed, 7)
            self.assertEqual(instance.pulses, ())
            self.assertEqual(instance._pulse_ticks, ())
            self.assertIsNone(instance._post_grip_event)
            instance.ready = False
            instance._clear_sidecar()
            instance._controls()
            instance._outcome_tick()
            self.assertEqual(instance._native_tick, 0)
            self.assertEqual(instance._post_grip_dwell_ticks, 0)
            instance.initial_object_position = np.asarray([.3, 0., .081])
            instance.ready = True
            return expected
        with patch.object(FeedbackPiperTaskEnv, 'reset', reset_parent), patch.object(
                FeedbackPiperTaskEnv, '_outcome_tick', append_parent_row):
            self.assertIs(env.reset(7), expected)
        self.assertEqual(env._post_grip_initial_height, .081)
        self.assertEqual(env._post_grip_active_ticks, 0)
        self.assertEqual(env._post_grip_effective_ticks, 0)
        self.assertTrue(analyze_post_grip(env.native_arrays(), env.native_metadata())['valid'])
        with patch.object(FeedbackPiperTaskEnv, 'reset', side_effect=RuntimeError('reset failed')):
            with self.assertRaises(RuntimeError):
                env.reset()
        self.assertIsNone(env._post_grip_initial_height)
        self.assertIsNone(env._post_grip_event)
        self.assertEqual(env._pulse_ticks, ())

    def test_analysis_rejects_tampered_event_schedule_active_floor_and_effect(self):
        env = fixture()
        advance(env, 14)
        arrays, metadata = env.native_arrays(), env.native_metadata()
        for name, mutate, expected in [
            ('event', lambda a, m: m['post_grip_intervention']['event'].update(eligibility_native_tick=4), 'metadata:event'),
            ('count', lambda a, m: m['post_grip_intervention'].update(active_native_ticks=3), 'metadata:active_native_ticks'),
            ('schedule', lambda a, m: m['pulse_tick_schedule'][0].update(start_tick=6), 'pulse_tick_schedule'),
            ('active', lambda a, m: a['pulse_active'].__setitem__(4, 1.), 'native:pulse_active'),
            ('floor', lambda a, m: a['pulse_minimum_opening_m'].__setitem__(5, .01), 'native:pulse_minimum_opening_m'),
            ('early-effect', lambda a, m: a['applied_jaw_target_m'].__setitem__(0, .001), 'native:effect_before_scheduled_pulse'),
        ]:
            with self.subTest(name=name):
                changed, declared = deepcopy(arrays), deepcopy(metadata)
                mutate(changed, declared)
                result = analyze_post_grip(changed, declared)
                self.assertFalse(result['valid'])
                self.assertIn(expected, result['mismatches'])

    def test_analysis_rejects_invalid_cadence_arrays_and_missing_metadata(self):
        env = fixture()
        advance(env, 4)
        arrays, metadata = env.native_arrays(), env.native_metadata()
        for field, bad in [('native_tick', np.asarray([1., 2., 4., 5.])),
                           ('pad_normal_forces_n', np.zeros((4, 3))),
                           ('pulse_active', np.asarray([0., 0., 0., .5])),
                           ('object_position_m', np.full((4, 3), float('nan')))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze_post_grip({**arrays, field: bad}, metadata)
        with self.assertRaises(ValueError):
            analyze_post_grip(arrays, {})


if __name__ == '__main__':
    unittest.main()
