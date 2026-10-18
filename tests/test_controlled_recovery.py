"""Experimental recovery must establish support before opening and stay bounded."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from dvidia_training.arm_env import ArmConfig
from dvidia_training.arm_runner import rollout
from dvidia_training.controlled_recovery_policy import ControlledRecoveryPolicy, LOWER_TIMEOUT_SECONDS
from dvidia_training.controlled_recovery_experiment import measurement, rollout_recovery, summarize
from dvidia_training.grip_diagnostic import GoalProxy, _same_trajectory


class RecoveryTests(unittest.TestCase):
    def make_policy(self):
        env = SimpleNamespace(config=ArmConfig(scene_jitter=0), _elapsed=1.,
                              joint_targets_for_pose=lambda goal: [float(goal[0]), 0., 0., 0., 0., 0.])
        policy = ControlledRecoveryPolicy(env)
        policy._lower_active = True
        policy._lower_xy = [.42, -.04]
        policy._lower_started = 1.
        observation = {'object_position': [.42, -.04, .31], 'end_effector_position': [.42, -.04, .33],
                       'object_velocity': [0., 0., 0.], 'object_angular_velocity': [0., 0., 0.],
                       'object_quaternion': [1., 0., 0., 0.], 'object_size': [.04, .04, .04],
                       'joint_position': [0., 0., 0., 0., 0., 0.], 'table_contacts': 2,
                       'gripper_width': .04, 'grasp_contacts': {'left': 4, 'right': 4},
                       'pad_normal_forces': {'left': .3, 'right': .3},
                       'effective_sliding_friction': {'jaw_object': 1.2}}
        return policy, observation

    def test_needs_five_consecutive_supported_and_settled_observations(self):
        policy, obs = self.make_policy()
        for _ in range(4):
            self.assertLess(policy(obs)['gripper_width'], .065)
        self.assertEqual(policy(obs)['gripper_width'], .08)
        self.assertEqual(policy.stage, 'recover_up')
        self.assertEqual(policy.diagnostics()['supported_open_observation']['consecutive_control_samples'], 5)

    def test_height_contact_speed_and_angular_speed_all_gate_opening(self):
        invalid = [dict(table_contacts=0), dict(object_position=[.42, -.04, .34]),
                   dict(object_velocity=[0., 0., .04]), dict(object_angular_velocity=[0., 0., .4])]
        for change in invalid:
            with self.subTest(change=change):
                policy, obs = self.make_policy()
                for _ in range(12):
                    self.assertLess(policy({**obs, **change})['gripper_width'], .065)
                self.assertIsNone(policy.diagnostics()['supported_open_observation'])

    def test_a_transient_contact_resets_support_confirmation(self):
        policy, obs = self.make_policy()
        for _ in range(4):
            policy(obs)
        policy({**obs, 'table_contacts': 0})
        self.assertEqual(policy.diagnostics()['support_samples'], 0)
        for _ in range(4):
            self.assertLess(policy(obs)['gripper_width'], .065)
        self.assertEqual(policy(obs)['gripper_width'], .08)

    def test_airborne_timeout_holds_without_opening_or_resetting_clock(self):
        policy, obs = self.make_policy()
        obs.update(table_contacts=0, object_position=[.42, -.04, .42], end_effector_position=[.42, -.04, .44])
        policy._lower_ticks = round(LOWER_TIMEOUT_SECONDS/policy.env.config.control_dt)
        action = policy(obs)
        self.assertLess(action['gripper_width'], .065)
        self.assertEqual(action['joint_targets'], obs['joint_position'])
        self.assertEqual(policy.failure_reason, 'controlled_recovery_support_timeout')
        self.assertEqual(policy.env._elapsed, 1.)
        for _ in range(4):
            self.assertLess(policy(obs)['gripper_width'], .065)
        self.assertTrue(policy.diagnostics()['controlled_recovery_abort'])

    def test_unsupported_orientation_aborts_closed_even_after_table_settles(self):
        policy, obs = self.make_policy()
        obs['object_quaternion'] = [.92387953, 0., 0., .38268343]
        for _ in range(5):
            action = policy(obs)
        self.assertLess(action['gripper_width'], .065)
        self.assertEqual(policy.failure_reason, 'controlled_recovery_unsupported_orientation')

    def test_native_baseline_parity_and_candidate_noop_before_recovery(self):
        config = replace(ArmConfig(scene_jitter=0), horizon=.4)
        old = rollout(config, 317)
        baseline = rollout_recovery(config, 317)
        candidate = rollout_recovery(config, 317, variant='controlled_lowering')
        self.assertTrue(_same_trajectory(baseline, old))
        self.assertTrue(_same_trajectory(candidate, old))
        self.assertIsNone(measurement(candidate)['recovery']['first_open_command'])

    def test_lowering_uses_the_student_mapping_without_native_ik_fallback(self):
        policy, obs = self.make_policy()
        obs.update(table_contacts=0, object_position=[.42, -.04, .42], end_effector_position=[.42, -.04, .44])
        called = []
        policy.env.joint_targets_for_pose = lambda goal: self.fail('Student must not invoke native IK.')
        head = SimpleNamespace(predict=lambda observation, goal: called.append((observation, list(goal))) or [0.] * 6)
        proxy = GoalProxy(policy.env, head)
        proxy.observation, proxy.policy = obs, policy
        policy.env = proxy
        action = policy(obs)
        self.assertEqual(action['joint_targets'], [0.] * 6)
        self.assertEqual(len(called), 1)
        self.assertIs(called[0][0], obs)
        self.assertAlmostEqual(called[0][1][2], .437)

    def test_unsupported_open_is_a_pre_action_command_not_post_action_release(self):
        config = replace(ArmConfig(scene_jitter=0), horizon=.4)
        run = rollout_recovery(config, 317)
        frame = deepcopy(run['trace'][-1])
        frame['stage'] = 'recover_wait'
        frame['table_contacts'] = 3
        frame['pre_action'].update(table_contacts=0, requested_gripper_width=.08,
                                   object_velocity=[0., 0., -.2])
        run['trace'] = [frame]
        result = measurement(run)['recovery']
        self.assertEqual(result['unsupported_open_command_samples'], 1)
        self.assertFalse(result['first_open_command']['supported_by_contact_sample'])
        self.assertEqual(result['sampled_peak_descent_speed_before_support_m_s'], .2)

    def test_completion_gate_rejects_hold_only_controls_errors_and_regressions(self):
        def row(variant, success):
            return {'case_id': 'scene-00', 'policy': 'student', 'variant': variant,
                    'group': 'nominal', 'status': 'completed', 'control_status': 'completed',
                    'measurement': {'success': success}, 'control': {'success': False}}
        pair = [row('baseline', False), row('controlled_lowering', False)]
        self.assertFalse(summarize(pair)['fresh_confirmation_gate_met'])
        pair[1]['measurement']['success'] = True
        self.assertTrue(summarize(pair)['fresh_confirmation_gate_met'])
        pair[1]['control']['success'] = True
        self.assertFalse(summarize(pair)['fresh_confirmation_gate_met'])
        pair[1]['control']['success'] = False
        pair[1]['control_status'] = 'error'
        self.assertFalse(summarize(pair)['fresh_confirmation_gate_met'])
        self.assertFalse(summarize([row('baseline', True), row('controlled_lowering', False)])['fresh_confirmation_gate_met'])


if __name__ == '__main__':
    unittest.main()
