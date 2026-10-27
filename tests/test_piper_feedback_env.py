"""Independent control-path fixtures for native jaw faults and dense sidecars."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training.piper_task import PiperTaskConfig, PiperTaskEnv
from dvidia_training.piper_feedback_env import FeedbackPiperTaskEnv, JawPulse, NATIVE_FIELDS


def controls_fixture(pulses=()):
    env = FeedbackPiperTaskEnv('unused', pulses=pulses)
    env.ready = True
    env.command_targets = np.asarray([.1, .2, .3, .4, .5, .6])
    env.command_gripper = 0.
    env._applied_jaw = env._normal_jaw = 0.
    env.data = SimpleNamespace(ctrl=np.zeros(7))
    return env


class JawFaultTests(unittest.TestCase):
    def test_pulse_is_finite_bounded_and_immutable(self):
        for args in ((False, .1, .01), (0., float('nan'), .01), (0., float('inf'), .01),
                     (-.01, .1, .01), (.2, .1, .01), (0., .1, -.001), (0., .1, .036),
                     (0., 10**1000, .01)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                JawPulse(*args)
        pulse = JawPulse(0., .1, .01)
        with self.assertRaises(FrozenInstanceError):
            pulse.start_s = .2
        with self.assertRaises(ValueError):
            FeedbackPiperTaskEnv('unused', pulses=(JawPulse(.0005, .1, .01),))
        with self.assertRaises(ValueError):
            FeedbackPiperTaskEnv('unused', pulses=(JawPulse(19.9, 20.001, .01),))
        with self.assertRaises(ValueError):
            FeedbackPiperTaskEnv('unused', pulses=({'start_s': 0., 'end_s': .1, 'minimum_opening_m': .01},))
        source = [pulse]
        env = FeedbackPiperTaskEnv('unused', pulses=source)
        source.clear()
        self.assertEqual(env.pulses, (pulse,))
        with self.assertRaises(AttributeError):
            env.pulses = ()

    def test_no_pulse_controls_are_bitwise_identical_to_original(self):
        env = controls_fixture()
        baseline = PiperTaskEnv('unused')
        baseline.data = SimpleNamespace(ctrl=np.zeros(7))
        baseline.command_targets = env.command_targets.copy()
        for value in (.035, .03495, .0349, .0123456789012345, 0., .00005, .035):
            env.command_gripper = baseline.command_gripper = value
            baseline._controls()
            env._controls()
            np.testing.assert_array_equal(env.data.ctrl, baseline.data.ctrl)
            env._native_tick += 1

    def test_half_open_pulse_and_closing_tail_are_both_slew_limited(self):
        env = controls_fixture((JawPulse(.002, .006, .01),))
        applied, active, tail = [], [], []
        for tick in range(12):
            env._native_tick = tick
            env._controls()
            applied.append(float(env.data.ctrl[6]))
            active.append(env._pulse_active)
            tail.append(env._closing_tail)
            np.testing.assert_array_equal(env.data.ctrl[:6], env.command_targets)
        np.testing.assert_allclose(applied, [0., 0., .00005, .0001, .00015, .0002,
                                             .00015, .0001, .00005, 0., 0., 0.], atol=1e-18)
        self.assertEqual(active, [False, False, True, True, True, True, False, False, False, False, False, False])
        self.assertEqual(tail, [False, False, False, False, False, False, True, True, True, False, False, False])
        self.assertLessEqual(max(abs(a-b) for a, b in zip(applied, applied[1:])), .00005+1e-18)
        self.assertEqual(env.command_gripper, 0.)

    def test_fault_does_not_affect_settling_and_overlaps_choose_maximum(self):
        env = controls_fixture((JawPulse(0., .1, .01), JawPulse(0., .1, .02)))
        env.ready = False
        env._controls()
        np.testing.assert_array_equal(env.data.ctrl, [*env.command_targets, 0.])
        self.assertEqual(env._native_tick, 0)
        env.ready = True
        env._controls()
        self.assertEqual(env._pulse_floor, .02)
        self.assertEqual(env.data.ctrl[6], .00005)

    def test_dense_rows_are_post_tick_numeric_copies_with_original_scoring(self):
        env = controls_fixture((JawPulse(0., .003, .01),))
        env.tcp, env.block = 0, 0
        env.elapsed = .001
        env._requested_jaw = .004
        env._last_contacts = ({'left': 2, 'right': 1}, {'left': .2, 'right': .3}, 0, 0, [])
        env.data.qpos = np.asarray([.1, .2, .3, .4, .5, .6, .01, -.01])
        env.data.xpos = np.asarray([[.31, .02, .12]])
        env.data.xquat = np.asarray([[1., 0., 0., 0.]])
        env.data.site_xpos = np.asarray([[.30, .02, .12]])
        env.data.site_xmat = np.asarray([[1., 0., 0., 0., 1., 0., 0., 0., 1.]])
        env.model = object()
        fake_mj = SimpleNamespace(
            mju_mat2Quat=lambda target, mat: target.__setitem__(slice(None), [1., 0., 0., 0.]),
            mj_objectVelocity=lambda model, data, typ, body, target, local: target.__setitem__(slice(None), [0., .1, 0., .01, 0., 0.]),
            mjtObj=SimpleNamespace(mjOBJ_BODY=1))
        outcome = {'both': True, 'supported': False, 'lift': .04, 'distance': .1,
                   'released': False, 'settled': True, 'clear': False, 'critical': 0, 'pad_force': .3}
        with patch.object(PiperTaskEnv, '_outcome_tick') as parent, patch(
                'dvidia_training.piper_feedback_env.sim._native_imports', return_value=(fake_mj, np)):
            env._controls()
            env._outcome_tick(**outcome)
        parent.assert_called_once_with(**outcome)
        arrays = env.native_arrays()
        self.assertEqual(set(arrays), {name for name, _ in NATIVE_FIELDS})
        self.assertTrue(all(array.dtype.kind in 'fiu' for array in arrays.values()))
        self.assertTrue(all(len(array) == 1 for array in arrays.values()))
        self.assertEqual(arrays['native_tick'].tolist(), [1.])
        self.assertEqual(arrays['t_seconds'].tolist(), [.001])
        self.assertEqual(arrays['requested_jaw_target_m'].tolist(), [.004])
        self.assertEqual(arrays['normal_rate_limited_jaw_target_m'].tolist(), [0.])
        self.assertEqual(arrays['applied_jaw_target_m'].tolist(), [.00005])
        np.testing.assert_allclose(arrays['object_minus_tcp_local_m'], [[.01, 0., 0.]])
        np.testing.assert_array_equal(arrays['object_linear_velocity_m_s'], [[.01, 0., 0.]])
        np.testing.assert_array_equal(arrays['object_angular_velocity_rad_s'], [[0., .1, 0.]])
        arrays['t_seconds'][0] = 9.
        self.assertEqual(env.native_arrays()['t_seconds'].tolist(), [.001])
        self.assertEqual(env.native_metadata()['rows'], 1)

    def test_empty_arrays_and_reset_sidecar_are_well_formed(self):
        env = controls_fixture()
        for name, size in NATIVE_FIELDS:
            self.assertEqual(env.native_arrays()[name].shape, (0,) if size == 1 else (0, size))
        env._native_rows.append(tuple(0. for _ in range(sum(size for _, size in NATIVE_FIELDS))))
        env._native_tick = 7
        sentinel_observation = {'measured': 'only'}
        def parent_reset(instance, seed):
            self.assertEqual(instance._native_rows, [])
            self.assertEqual(instance._native_tick, 0)
            instance.ready = True
            instance.command_gripper = .035
            return sentinel_observation, {'success': False}
        with patch.object(PiperTaskEnv, 'reset', parent_reset), patch.object(env, '_refresh_contact_names'):
            observation, info = env.reset(1)
        self.assertIs(observation, sentinel_observation)
        self.assertEqual(info, {'success': False})
        self.assertEqual(env.native_metadata()['rows'], 0)
        self.assertEqual(env._applied_jaw, .035)


if __name__ == '__main__':
    unittest.main()
