"""Independent synthetic checks of progress timing and frozen physical-time jaw.

These tests run no simulator, inverse kinematics, teacher rollout or asset load.
The changing reference fixture makes a held path observably different from the
physical-time path without changing the model format or native command caps.
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_grip_adapter as grip
from dvidia_training import piper_grip_clock as runtime
from dvidia_training import piper_grip_clock_model as clock_model
from dvidia_training import piper_imitation as core
from dvidia_training import piper_motor_primitive as primitive
from tests.test_piper_feedback_model import base_model, observation


def changing_path_model():
    model = base_model()
    weights = np.asarray(model['weights'])
    # Original model weights describe a smooth, changing six-joint reference.
    # Jaw reference opens, closes, then opens, so it has a real grasp window.
    for axis in range(6):
        weights[:, 0, :, axis] += .004*np.sin(primitive.CENTERS*(axis+1))
    weights[:, 0, :, 6] = np.where(
        (primitive.CENTERS >= .4) & (primitive.CENTERS < 1.2), 0., 1.)
    model['weights'] = weights.tolist()
    return primitive.validate_model(model)


class GripClockRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = changing_path_model()
        x = np.zeros((16, grip.FEATURE_COUNT))
        x[:, grip.CONTEXT_NAMES.index('elapsed_seconds/20')] = np.arange(16)*.02/20.
        jaw_targets = np.where(np.arange(16) < 6, .035, 0.)
        cls.jaw = grip.train(
            dict(features=x, targets=jaw_targets, groups=['jaw-a']*8+['jaw-b']*8),
            base_sha256=core.digest(cls.base), max_depth=4, min_leaf=2)
        cls.clocks = {}
        for blind in (False, True):
            for advance in (False, True):
                cls.clocks[blind, advance] = clock_model.train(
                    dict(features=np.zeros((8, clock_model.FEATURE_COUNT)),
                         targets=np.full(8, .02 if advance else 0.),
                         groups=['clock-a']*4+['clock-b']*4),
                    base_sha256=core.digest(cls.base), jaw_sha256=core.digest(cls.jaw),
                    blind=blind, max_depth=4, min_leaf=2)

    def teacher_at_gate(self, *, acquisition_only=False):
        policy = runtime.TeacherClock(
            self.base, self.jaw, config=runtime.TeacherConfig(acquisition_only=acquisition_only))
        # Isolate the readiness predicate from the reference's chosen gate.
        policy.gate, policy.release = 0, 1000
        return policy

    def test_fixed_unit_clock_has_bitwise_action_parity_with_frozen_v02_jaw_policy(self):
        actual = runtime.FixedClock(self.base, self.jaw)
        original = grip.GripPolicy(self.base, self.jaw)
        for tick in range(80):
            obs = observation(tick*.02, contacts=tick%3)
            obs['joint_position_rad'][0] += .2 if tick%2 else -.2
            self.assertEqual(actual(obs), original(obs))
            decision = actual.last_decision
            self.assertEqual(decision['progress_before_s'], tick*.02)
            self.assertEqual(decision['progress_after_s'], (tick+1)*.02)
            self.assertEqual(decision['increment_s'], .02)

    def test_coordinated_fixed_clock_has_bitwise_all_seven_action_parity_with_original_primitive(self):
        manifest = runtime.reference_jaw_manifest(self.base)
        actual = runtime.FixedClock(self.base, manifest)
        original = primitive.MotorPrimitivePolicy(self.base)
        for tick in range(80):
            obs = observation(tick*.02, contacts=tick%3)
            obs['joint_position_rad'][0] += .2 if tick%2 else -.2
            self.assertEqual(actual(obs), original(obs))
        self.assertEqual(actual.diagnostics()['jaw_mode'], 'coordinated_reference')
        self.assertEqual(actual.diagnostics()['frozen_jaw_clock'], 'shared_virtual_progress')
        self.assertIsNone(actual.jaw)

    def test_coordinated_delayed_reference_retimes_jaw_explicitly_without_changing_physical_context(self):
        manifest = runtime.reference_jaw_manifest(self.base)
        delayed = runtime.TeacherClock(self.base, manifest)
        delayed.gate, delayed.release = 0, 1000
        physical = primitive.MotorPrimitivePolicy(self.base)
        observed_difference = False
        for tick in range(50):
            obs = observation(tick*.02, contacts=0)
            action, nominal = delayed(obs), physical(obs)
            expected_jaw = float(.035*np.clip(delayed.path.raw(0.)[6], 0., 1.))
            self.assertEqual(action['gripper_position_m'], expected_jaw)
            self.assertEqual(delayed.last_decision['jaw_mode'], 'coordinated_reference')
            self.assertEqual(delayed.last_decision['progress_before_s'], 0.)
            self.assertEqual(delayed.last_decision['nominal_physical_jaw_m'], nominal['gripper_position_m'])
            self.assertEqual(delayed.last_decision['features'][clock_model.ELAPSED_INDEX], tick*.02/20.)
            observed_difference |= action['gripper_position_m'] != nominal['gripper_position_m']
        self.assertTrue(observed_difference)
        self.assertIsNone(delayed.jaw)

    def test_coordinated_learned_clock_uses_v2_manifest_and_never_constructs_v02_jaw_adapter(self):
        manifest = runtime.reference_jaw_manifest(self.base)
        document = clock_model.train(
            dict(features=np.zeros((8, clock_model.FEATURE_COUNT)), targets=np.zeros(8),
                 groups=['coordinated-a']*4+['coordinated-b']*4),
            base_sha256=core.digest(self.base), jaw_sha256=core.digest(manifest),
            jaw_mode='coordinated_reference', max_depth=4, min_leaf=2)
        self.assertEqual(document['schema'], 'dvidia.piper-grip-progress-clock/v2')
        with patch.object(grip, 'GripPolicy', side_effect=AssertionError('v0.2 jaw must be absent')):
            learned = runtime.LearnedClock(self.base, manifest, document)
            for tick in range(40):
                action = learned(observation(tick*.02, contacts=tick%2))
                expected = float(.035*np.clip(learned.path.raw(0.)[6], 0., 1.))
                self.assertEqual(action['gripper_position_m'], expected)
                self.assertEqual(learned.last_decision['progress_before_s'], 0.)
        self.assertEqual(learned.last_decision['decision_origin'], 'learned_grip_clock')
        with self.assertRaises(ValueError):
            runtime.LearnedClock(self.base, self.jaw, document)

    def test_delayed_arm_keeps_exact_physical_time_jaw_actions_and_feature_context(self):
        held = runtime.LearnedClock(self.base, self.jaw, self.clocks[False, False])
        original = grip.GripPolicy(self.base, self.jaw)
        nominal = primitive.MotorPrimitivePolicy(self.base)
        jaw_values, different_arm = [], False
        for tick in range(40):
            obs = observation(tick*.02, contacts=tick%2)
            action, expected, physical = held(obs), original(obs), nominal(obs)
            self.assertEqual(action['gripper_position_m'], expected['gripper_position_m'])
            self.assertEqual(held.last_decision['nominal_physical_jaw_m'], physical['gripper_position_m'])
            self.assertEqual(held.last_decision['progress_before_s'], 0.)
            self.assertEqual(held.last_decision['progress_after_s'], 0.)
            self.assertEqual(held.last_decision['features'][clock_model.ELAPSED_INDEX], tick*.02/20.)
            self.assertEqual(held.last_decision['features'][-2], 0.)
            self.assertEqual(held.last_decision['features'][-1], tick*.02/20.)
            np.testing.assert_array_equal(
                action['joint_targets_rad'], held.path.targets(obs, 0.))
            different_arm |= action['joint_targets_rad'] != expected['joint_targets_rad']
            jaw_values.append(action['gripper_position_m'])
        self.assertTrue(different_arm)
        self.assertEqual(set(jaw_values), {0., .035})
        self.assertEqual(held.jaw.encoder.last_time, .78)
        self.assertEqual(held.physical_nominal.calls, 40)
        self.assertEqual(held.jaw.nominal.calls, 40)

    def test_retiming_preserves_raw_reference_function_and_unchanged_native_caps(self):
        frozen = runtime.FrozenPath(self.base)
        obs = observation()
        frozen.latch(obs)
        initial = primitive._start(obs['object_position_m'])
        weights = np.asarray(self.base['weights'])
        for seconds in (0., .02, .38, .40, .86, 1.20, 7.32, 19.98, 20.):
            expected = np.einsum(
                's,t,stj->j', primitive._spatial(initial), primitive._time_basis(seconds),
                weights[0], optimize=True)
            np.testing.assert_array_equal(frozen.raw(seconds), expected)
            q = np.asarray(obs['joint_position_rad'])
            limits = np.asarray(core.JOINT_LIMITS)
            capped = np.clip(q+np.clip(expected[:6]*3.-q, -.06, .06),
                             limits[:, 0]+.02, limits[:, 1]-.02)
            np.testing.assert_array_equal(frozen.targets(obs, seconds), capped)
        displaced = deepcopy(obs)
        displaced['joint_position_rad'] = [1., 1., -1., 1., 1., 1.]
        target = np.asarray(frozen.targets(displaced, .40))
        self.assertTrue(np.all(abs(target-np.asarray(displaced['joint_position_rad'])) <= .06+1e-12))
        for seconds in (-.02, 20.02, float('nan')):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                frozen.raw(seconds)

    def test_progress_label_is_decided_before_issue_and_only_affects_next_reference(self):
        policy = runtime.LearnedClock(self.base, self.jaw, self.clocks[False, True])
        for tick in range(6):
            obs = observation(tick*.02, contacts=1)
            action = policy(obs)
            logged = policy.last_decision
            self.assertEqual(logged['progress_before_s'], tick*.02)
            self.assertEqual(logged['progress_after_s'], (tick+1)*.02)
            self.assertEqual(logged['features'][-2], tick*.02/20.)
            self.assertEqual(logged['increment_s'], .02)
            np.testing.assert_array_equal(action['joint_targets_rad'], policy.path.targets(obs, tick*.02))

    def test_supported_bilateral_grip_requires_full_100ms_observed_boundary_span(self):
        policy = self.teacher_at_gate()
        for tick in range(6):
            obs = observation(tick*.02, contacts=1)
            self.assertEqual(obs['table_object_contact_count'], 1)
            policy(obs)
            decision = policy.last_decision
            self.assertEqual(decision['secure_grip'], tick == 5)
            self.assertEqual(decision['increment_s'], .02 if tick == 5 else 0.)
            self.assertAlmostEqual(decision['grip_dwell_s'], tick*.02)
        self.assertEqual(policy.progress_ticks, 1)

    def test_force_and_bilateral_contact_failures_reset_readiness_dwell(self):
        for mode in ('low-force', 'one-sided'):
            with self.subTest(mode=mode):
                policy = self.teacher_at_gate()
                for tick in range(12):
                    obs = observation(tick*.02, contacts=1)
                    if tick == 5:
                        if mode == 'low-force':
                            obs['pad_normal_forces_n']['right'] = .199
                        else:
                            obs['grasp_contact_counts']['right'] = 0
                    policy(obs)
                    self.assertEqual(policy.last_decision['secure_grip'], tick == 11)
                self.assertEqual(policy.progress_ticks, 1)

    def test_pad_force_threshold_is_inclusive_and_a_single_high_force_sample_is_insufficient(self):
        policy = self.teacher_at_gate()
        for tick in range(6):
            obs = observation(tick*.02, contacts=1)
            obs['pad_normal_forces_n'] = {'left': .20, 'right': .20}
            policy(obs)
        self.assertTrue(policy.last_decision['secure_grip'])
        other = self.teacher_at_gate()
        for tick in range(6):
            obs = observation(tick*.02, contacts=1)
            obs['pad_normal_forces_n'] = {'left': .01, 'right': .01} if tick < 5 else {'left': 2., 'right': 2.}
            other(obs)
        self.assertFalse(other.last_decision['secure_grip'])
        self.assertEqual(other.progress_ticks, 0)

    def test_relative_motion_beyond_2mm_blocks_secure_until_it_leaves_causal_window(self):
        policy = self.teacher_at_gate()
        for tick in range(12):
            obs = observation(tick*.02, contacts=1)
            if tick == 5:
                obs['tcp_position_m'][0] += .003
            policy(obs)
            self.assertEqual(policy.last_decision['secure_grip'], tick == 11)
        self.assertTrue(all(t >= .22-.10-1e-8 for t, _, _ in policy.history))

    def test_rigid_object_rotating_with_tool_has_zero_grip_relative_drift(self):
        policy = self.teacher_at_gate()
        for tick in range(6):
            obs = observation(tick*.02, contacts=1)
            angle = np.deg2rad(tick)
            quaternion = [float(np.cos(angle/2)), 0., float(np.sin(angle/2)), 0.]
            # The object is rigidly attached at tool-local (0,0,-80mm).
            # Rotating both together moves its world offset by almost 7mm,
            # while its actual relative position in the gripper is unchanged.
            rotated_offset = np.asarray([-.08*np.sin(angle), 0., -.08*np.cos(angle)])
            obs['object_position_m'] = (np.asarray(obs['tcp_position_m'])+rotated_offset).tolist()
            obs['tcp_quaternion_wxyz'] = quaternion
            obs['object_quaternion_wxyz'] = quaternion.copy()
            policy(obs)
        self.assertTrue(policy.last_decision['secure_grip'],
                        'Rigid tool rotation must not be interpreted as object/gripper slip.')
        self.assertAlmostEqual(policy.last_decision['relative_drift_m'], 0., places=12)

    def test_teacher_gate_changes_timing_without_requiring_airborne_grip(self):
        policy = runtime.TeacherClock(self.base, self.jaw)
        first = observation()
        policy.path.latch(first)
        gate, release = policy.path.grasp_window(.16)
        self.assertLess(gate, release)
        self.assertLess(float(policy.path.raw((gate-8)*.02)[6])*.035, .01)
        for tick in range(gate+3):
            policy(observation(tick*.02, contacts=0))
        self.assertEqual(policy.progress_ticks, gate)
        self.assertTrue(policy.last_decision['in_grasp_window'])
        self.assertFalse(policy.last_decision['secure_grip'])

    def test_acquisition_latch_advances_after_one_supported_secure_grip_without_claiming_recovery(self):
        policy = self.teacher_at_gate(acquisition_only=True)
        for tick in range(6):
            policy(observation(tick*.02, contacts=1))
        self.assertTrue(policy.last_decision['acquisition_released'])
        self.assertTrue(policy.last_decision['secure_grip'])
        for tick in range(6, 12):
            policy(observation(tick*.02, contacts=0))
            self.assertTrue(policy.last_decision['acquisition_released'])
            self.assertFalse(policy.last_decision['secure_grip'])
            self.assertEqual(policy.last_decision['increment_s'], .02)
        self.assertEqual(policy.progress_ticks, 7)
        continuous = self.teacher_at_gate(acquisition_only=False)
        for tick in range(6):
            continuous(observation(tick*.02, contacts=1))
        continuous(observation(.12, contacts=0))
        self.assertTrue(continuous.last_decision['acquisition_released'])
        self.assertEqual(continuous.last_decision['increment_s'], 0.)

    def test_learned_runtime_has_no_teacher_grasp_gate_or_future_state_handle(self):
        clean = runtime.LearnedClock(self.base, self.jaw, self.clocks[False, True])
        poison = runtime.LearnedClock(self.base, self.jaw, self.clocks[False, True])
        with patch.object(runtime.TeacherClock, '_advance', side_effect=AssertionError('teacher unavailable')), \
                patch.object(runtime.FrozenPath, 'grasp_window', side_effect=AssertionError('gate unavailable')):
            for tick in range(8):
                a, b = observation(tick*.02, contacts=tick%2), observation(tick*.02, contacts=tick%2)
                b.update(expert_phase='release', success=True, future_state=object(),
                         intervention_id='not a feature', expert_action=object(), teacher=object())
                self.assertEqual(clean(a), poison(b))
                self.assertEqual(clean.last_decision['features'], poison.last_decision['features'])
        self.assertFalse(any('teacher' in key or key in ('env', 'environment') for key in vars(clean)))
        self.assertEqual(clean.last_decision['decision_origin'], 'learned_grip_clock')

    def test_blind_clock_masks_sensors_while_shared_jaw_remains_the_exact_live_model(self):
        policy = runtime.LearnedClock(self.base, self.jaw, self.clocks[True, True], blind=True)
        self.assertFalse(policy.jaw.model['blind'])
        self.assertEqual(core.digest(policy.jaw.model), core.digest(self.jaw))
        for tick in range(8):
            policy(observation(tick*.02, contacts=tick))
            feature = policy.last_decision['features']
            self.assertTrue(np.all(np.asarray(feature)[clock_model.LIVE_START:clock_model.LIVE_STOP] == 0.))
        self.assertEqual(policy.last_decision['decision_origin'], 'learned_blind_clock')
        with self.assertRaises(ValueError):
            runtime.LearnedClock(self.base, self.jaw, self.clocks[True, True], blind=False)
        changed = deepcopy(self.jaw)
        changed['nodes'][0]['open_probability'] = .25
        with self.assertRaises(ValueError):
            runtime.LearnedClock(self.base, changed, self.clocks[False, True])

    def test_teacher_grid_and_bounds_are_explicit(self):
        for config in (dict(gate_offset_s=.01), dict(secure_dwell_s=.05),
                       dict(minimum_pad_force_n=0.), dict(maximum_relative_drift_m=float('nan')),
                       dict(acquisition_only='yes')):
            with self.subTest(config=config), self.assertRaises(ValueError):
                runtime.TeacherConfig(**config)


if __name__ == '__main__':
    unittest.main()
