"""A supported open command does not permit withdrawal before actual clearance."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from dvidia_training.arm_env import ArmConfig
from dvidia_training.arm_runner import rollout
from dvidia_training.controlled_recovery_release_policy import SupportedReleaseRecoveryPolicy
from dvidia_training.controlled_recovery_release_experiment import rollout_recovery, measurement
from dvidia_training.grip_diagnostic import _same_trajectory


class ReleaseTests(unittest.TestCase):
    def make_policy(self):
        env = SimpleNamespace(config=ArmConfig(scene_jitter=0), _elapsed=1.,
                              joint_targets_for_pose=lambda goal: [0.] * 6)
        policy = SupportedReleaseRecoveryPolicy(env)
        policy._lower_active = True
        policy._lower_xy = [.42, -.04]
        policy._lower_started = 1.
        obs = {'object_position': [.42, -.04, .31], 'end_effector_position': [.42, -.04, .33],
               'object_velocity': [0., 0., 0.], 'object_angular_velocity': [0., 0., 0.],
               'object_quaternion': [1., 0., 0., 0.], 'object_size': [.04, .04, .04],
               'joint_position': [.1, .2, .3, .4, .5, .6], 'table_contacts': 2,
               'gripper_width': .04, 'grasp_contacts': {'left': 4, 'right': 4},
               'pad_normal_forces': {'left': .3, 'right': .3},
               'effective_sliding_friction': {'jaw_object': 1.2}}
        for _ in range(5):
            policy(obs)
        self.assertEqual(policy.stage, 'recover_wait_clearance')
        return policy, obs

    def test_open_command_holds_pose_until_measured_clearance(self):
        policy, obs = self.make_policy()
        for _ in range(8):
            action = policy(obs)
            self.assertEqual(action['joint_targets'], obs['joint_position'])
            self.assertEqual(action['gripper_width'], .08)
            self.assertEqual(policy.stage, 'recover_wait_clearance')
        self.assertIsNone(policy.diagnostics()['actual_release_observation'])

    def test_gap_contact_free_and_settled_support_are_all_required(self):
        invalid = [dict(gripper_width=.06, grasp_contacts={'left': 0, 'right': 0}),
                   dict(gripper_width=.07, grasp_contacts={'left': 1, 'right': 0}),
                   dict(gripper_width=.07, grasp_contacts={'left': 0, 'right': 0}, table_contacts=0),
                   dict(gripper_width=.07, grasp_contacts={'left': 0, 'right': 0}, object_velocity=[0., 0., .04])]
        for changes in invalid:
            with self.subTest(changes=changes):
                policy, obs = self.make_policy()
                for _ in range(6):
                    policy({**obs, **changes})
                self.assertEqual(policy.stage, 'recover_wait_clearance')
                self.assertIsNone(policy.diagnostics()['actual_release_observation'])

    def test_five_clear_samples_confirm_then_allow_withdrawal_next_tick(self):
        policy, obs = self.make_policy()
        obs.update(gripper_width=.07, grasp_contacts={'left': 0, 'right': 0})
        policy.env._elapsed = 1.3
        for _ in range(4):
            policy(obs)
            self.assertEqual(policy.stage, 'recover_wait_clearance')
        action = policy(obs)
        self.assertEqual(action['joint_targets'], obs['joint_position'])
        self.assertEqual(policy.stage, 'recover_up')
        record = policy.diagnostics()['actual_release_observation']
        self.assertEqual(record['consecutive_control_samples'], 5)
        self.assertEqual(record['contacts'], {'left': 0, 'right': 0})
        self.assertEqual(record['gripper_width'], .07)
        self.assertAlmostEqual(record['elapsed_recovery_seconds'], .3)

    def test_contact_reappearance_resets_clearance_streak(self):
        policy, obs = self.make_policy()
        clear = {**obs, 'gripper_width': .07, 'grasp_contacts': {'left': 0, 'right': 0}}
        for _ in range(4):
            policy(clear)
        policy(obs)
        self.assertEqual(policy.diagnostics()['clearance_stable_samples'], 0)
        for _ in range(4):
            policy(clear)
        self.assertEqual(policy.stage, 'recover_wait_clearance')

    def test_clearance_shares_original_recovery_deadline_and_never_withdraws_on_timeout(self):
        policy, obs = self.make_policy()
        policy.env._elapsed = 5.001
        action = policy(obs)
        self.assertEqual(policy.failure_reason, 'controlled_recovery_clearance_timeout')
        self.assertEqual(action['joint_targets'], obs['joint_position'])
        self.assertLessEqual(action['gripper_width'], obs['gripper_width'])
        self.assertIsNone(policy.diagnostics()['actual_release_observation'])
        self.assertEqual(policy.env._elapsed, 5.001)

    def test_v2_native_noop_baseline_and_candidate_before_recovery(self):
        config = replace(ArmConfig(scene_jitter=0), horizon=.4)
        old = rollout(config, 317)
        for variant in ('baseline', 'controlled_lowering'):
            result = rollout_recovery(config, 317, variant=variant)
            self.assertTrue(_same_trajectory(result, old))
            self.assertIsNone(measurement(result)['recovery']['actual_release_confirmation'])


if __name__ == '__main__':
    unittest.main()
