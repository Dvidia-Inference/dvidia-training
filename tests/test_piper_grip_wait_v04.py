"""State/command checks independent of native physics and held-out outcomes."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_grip_clock as frozen
from dvidia_training import piper_grip_clock_model as clock_model
from dvidia_training import piper_grip_wait_v04 as runtime
from dvidia_training import piper_imitation as core
from tests.test_piper_grip_clock_runtime import changing_path_model
from tests.test_piper_feedback_model import observation


class BoundedGripWaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = changing_path_model()
        cls.jaw = frozen.reference_jaw_manifest(cls.base)
        cls.models = {}
        for advance in (False, True):
            cls.models[advance] = clock_model.train(
                dict(features=np.zeros((8, clock_model.FEATURE_COUNT)),
                     targets=np.full(8, .02 if advance else 0.),
                     groups=['a']*4+['b']*4),
                base_sha256=core.digest(cls.base), jaw_sha256=core.digest(cls.jaw),
                jaw_mode='coordinated_reference', max_depth=4, min_leaf=2)

    def policy(self, *, learner_advance=False, maximum_wait_s=.20, inside=True):
        policy = runtime.GuardedLearnedClock(self.base, self.jaw, self.models[learner_advance],
                    config=runtime.GripWaitConfig(maximum_wait_s=maximum_wait_s))
        # Isolate supervision from path timing; frozen path/caps tested below.
        policy.gate, policy.release = (0, 1000) if inside else (900, 1000)
        if inside:
            actual_raw = policy.path.raw
            def closed_reference(t):
                value = actual_raw(t).copy(); value[6] = 0.; return value
            policy.path.raw = closed_reference
        return policy

    def eligible_observation(self, tick, force=.10):
        obs = observation(tick*.02, contacts=1)
        obs['pad_normal_forces_n'] = dict(left=force, right=force)
        return obs

    def test_outside_window_advances_even_when_learned_clock_requests_hold(self):
        policy = self.policy(inside=False)
        for tick in range(10):
            policy(observation(tick*.02))
            self.assertEqual(policy.last_decision['increment_s'], .02)
            self.assertEqual(policy.last_decision['reason'], 'outside_acquisition_window')
            self.assertTrue(policy.last_decision['supervisor_override'])
        self.assertEqual(policy.wait_ticks, 0)

    def test_one_sided_contact_can_continue_centering_instead_of_freezing(self):
        policy = self.policy()
        for tick in range(10):
            obs = self.eligible_observation(tick)
            obs['grasp_contact_counts']['right'] = 0
            policy(obs)
            self.assertEqual(policy.last_decision['increment_s'], .02)
            self.assertEqual(policy.last_decision['reason'], 'centering_without_bilateral_support')
        self.assertFalse(policy.acquisition_started)

    def test_far_object_and_open_reference_cannot_start_an_acquisition_wait(self):
        for mode in ('far-object', 'open-jaw'):
            with self.subTest(mode=mode):
                policy = self.policy()
                obs = self.eligible_observation(0)
                if mode == 'far-object':
                    obs['tcp_position_m'][0] += .2
                else:
                    actual = policy.path.raw
                    def raw(t):
                        value = actual(t).copy(); value[6] = 1.; return value
                    policy.path.raw = raw
                policy(obs)
                self.assertEqual(policy.last_decision['increment_s'], .02)
                self.assertFalse(policy.last_decision['eligible_for_wait'])

    def test_supported_bilateral_acquisition_forces_wait_even_if_learner_says_advance(self):
        policy = self.policy(learner_advance=True)
        policy(self.eligible_observation(0))
        logged = policy.last_decision
        self.assertEqual(logged['increment_s'], 0.)
        self.assertTrue(logged['learner_advance'])
        self.assertTrue(logged['learned_prediction_advisory'])
        self.assertTrue(logged['supervisor_override'])
        self.assertEqual(logged['reason'], 'supervised_acquisition_wait')

    def test_full_100ms_dwell_resumes_and_latches_against_later_learned_holds(self):
        policy = self.policy()
        for tick in range(6):
            policy(self.eligible_observation(tick, .20))
            self.assertEqual(policy.last_decision['increment_s'], .02 if tick == 5 else 0.)
            self.assertEqual(policy.last_decision['secure_grip'], tick == 5)
        self.assertEqual(policy.wait_ticks, 5)
        self.assertTrue(policy.acquisition_released)
        # This is acquisition-only supervision, not continuous grip monitoring.
        for tick in range(6, 12):
            obs = observation(tick*.02, contacts=0)
            obs['table_object_contact_count'] = 0
            policy(obs)
            self.assertEqual(policy.last_decision['increment_s'], .02)
            self.assertEqual(policy.last_decision['reason'], 'secure_grip_latched')
            self.assertFalse(policy.aborted)

    def test_wait_budget_is_finite_and_exhaustion_never_forces_insecure_advance(self):
        for budget, ticks in ((.20, 10), (.36, 18), (.50, 25)):
            with self.subTest(budget=budget):
                policy = self.policy(maximum_wait_s=budget)
                for tick in range(ticks):
                    policy(self.eligible_observation(tick))
                    self.assertFalse(policy.aborted)
                    self.assertEqual(policy.last_decision['increment_s'], 0.)
                policy(self.eligible_observation(ticks))
                self.assertTrue(policy.aborted)
                self.assertEqual(policy.abort_reason, 'wait_budget_exhausted')
                self.assertEqual(policy.wait_ticks, ticks)
                for tick in range(ticks+1, ticks+5):
                    policy(self.eligible_observation(tick, .20))
                    self.assertEqual(policy.last_decision['increment_s'], 0.)
                    self.assertEqual(policy.last_decision['reason'], 'aborted_pose_hold')
                self.assertFalse(policy.acquisition_released)
                self.assertEqual(policy.progress_ticks, 0)

    def test_contact_flicker_cannot_reset_or_evade_the_total_wait_budget(self):
        policy = self.policy()
        for tick in range(11):
            obs = self.eligible_observation(tick)
            if tick%2:
                obs['grasp_contact_counts']['right'] = 0
            policy(obs)
        self.assertTrue(policy.aborted)
        self.assertEqual(policy.wait_ticks, 10)
        self.assertEqual(policy.abort_reason, 'wait_budget_exhausted')

    def test_support_loss_before_secure_aborts_even_if_bilateral_eligibility_never_seen(self):
        for prior_eligible in (False, True):
            with self.subTest(prior_eligible=prior_eligible):
                policy = self.policy()
                policy(self.eligible_observation(0) if prior_eligible else observation())
                obs = observation(.02)
                obs['table_object_contact_count'] = 0
                policy(obs)
                self.assertTrue(policy.aborted)
                self.assertEqual(policy.abort_reason, 'support_lost_before_secure')
                self.assertEqual(policy.last_decision['increment_s'], 0.)

    def test_force_drop_and_relative_motion_reset_secure_dwell(self):
        for mode in ('force', 'drift'):
            with self.subTest(mode=mode):
                policy = self.policy(maximum_wait_s=.50)
                for tick in range(12):
                    obs = self.eligible_observation(tick, .20)
                    if tick == 5:
                        if mode == 'force':
                            obs['pad_normal_forces_n']['right'] = .199
                        else:
                            obs['tcp_position_m'][0] += .003
                    policy(obs)
                    self.assertEqual(policy.acquisition_released, tick == 11)
                self.assertFalse(policy.aborted)

    def test_rigid_tool_rotation_is_not_mistaken_for_relative_grip_slip(self):
        policy = self.policy()
        for tick in range(6):
            obs = self.eligible_observation(tick, .20)
            angle = np.deg2rad(tick)
            obs['tcp_quaternion_wxyz'] = [float(np.cos(angle/2)), 0., float(np.sin(angle/2)), 0.]
            obs['object_position_m'] = (np.asarray(obs['tcp_position_m'])
                                      +[-.08*np.sin(angle), 0., -.08*np.cos(angle)]).tolist()
            policy(obs)
        self.assertTrue(policy.acquisition_released)
        self.assertAlmostEqual(policy.last_decision['relative_drift_m'], 0., places=12)

    def test_abort_holds_measured_pose_on_first_abort_action_and_respects_native_caps(self):
        policy = self.policy()
        for tick in range(11):
            obs = self.eligible_observation(tick)
            if tick == 10:
                obs['joint_position_rad'][0] = .3
            action = policy(obs)
        np.testing.assert_allclose(action['joint_targets_rad'], obs['joint_position_rad'])
        original_hold = np.asarray(action['joint_targets_rad'])
        later = self.eligible_observation(11)
        later['joint_position_rad'][0] = -.3
        action = policy(later)
        q = np.asarray(later['joint_position_rad'])
        limits = np.asarray(core.JOINT_LIMITS)
        expected = np.clip(q+np.clip(original_hold-q, -.06, .06), limits[:, 0]+.02, limits[:, 1]-.02)
        np.testing.assert_array_equal(action['joint_targets_rad'], expected)
        self.assertEqual(action['gripper_position_m'], 0.)
        self.assertFalse(policy.diagnostics()['original_arm_path_preserved'])
        self.assertTrue(policy.diagnostics()['native_command_caps_preserved'])

    def test_never_waiting_policy_has_exact_fixed_actions_and_unchanged_frozen_geometry(self):
        policy = self.policy(inside=False)
        fixed = frozen.FixedClock(self.base, self.jaw)
        for tick in range(80):
            obs = observation(tick*.02)
            self.assertEqual(policy(obs), fixed(obs))
        self.assertTrue(policy.diagnostics()['original_arm_path_preserved'])
        self.assertEqual(policy.diagnostics()['frozen_jaw_model_sha256'], core.digest(self.jaw))
        self.assertFalse(policy.diagnostics()['newly_trained_model'])
        self.assertTrue(policy.diagnostics()['learned_prediction_advisory'])

    def test_quaternion_and_config_validation_reject_malformed_supervisor_inputs(self):
        for kwargs in ({'maximum_wait_s': .35}, {'maximum_wait_s': True},
                       {'secure_dwell_s': .12, 'maximum_wait_s': .10},
                       {'secure_minimum_pad_force_n': .05},
                       {'maximum_relative_drift_m': float('nan')},
                       {'maximum_object_tcp_distance_m': 0.}, {'gate_offset_s': -.02}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                runtime.GripWaitConfig(**kwargs)
        with self.assertRaises(ValueError):
            runtime.GuardedLearnedClock(self.base, self.jaw, self.models[False], config={})
        policy = self.policy()
        obs = self.eligible_observation(0)
        obs['tcp_quaternion_wxyz'] = [1., 0., 0., .2]
        with self.assertRaises(ValueError):
            policy(obs)

    def test_no_teacher_or_native_state_write_and_no_full_vs_blind_supervisor_claim(self):
        # The supervisor uses a declared authored predicate but never constructs
        # the frozen collection teacher or reads its hidden phase/outcome labels.
        with patch.object(frozen, 'TeacherClock', side_effect=AssertionError('No collection teacher')):
            left, right = self.policy(), self.policy(learner_advance=True)
            for tick in range(6):
                clean, poison = self.eligible_observation(tick, .20), self.eligible_observation(tick, .20)
                poison.update(success=True, future={'ready': True}, phase='expert-secret')
                before = deepcopy(poison)
                self.assertEqual(left(clean), right(poison))
                self.assertEqual(before, poison)
                self.assertEqual(left.last_decision['increment_s'], right.last_decision['increment_s'])
        self.assertTrue(left.diagnostics()['authored_acquisition_supervisor'])
        self.assertFalse(left.diagnostics()['physical_robot_ready'])


if __name__ == '__main__':
    unittest.main()
