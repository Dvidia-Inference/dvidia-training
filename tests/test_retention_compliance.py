"""Compliant path changes never weaken the native guard or outcome predicates."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from dvidia_training.arm_env import ArmConfig
from dvidia_training.arm_runner import rollout
from dvidia_training.grip_diagnostic import _same_trajectory
from dvidia_training.retention_compliance_policy import CompliantTransferGoalProxy, CompliantTransferPolicy
from dvidia_training.retention_compliance_experiment import rollout_retention


class ComplianceTests(unittest.TestCase):
    def setup_policy(self, high_margin=False):
        config=replace(ArmConfig(scene_jitter=0),object_mass=.06,object_friction=1.2,
                       jaw_force_limit=3. if high_margin else .3)
        calls=[]
        env=SimpleNamespace(config=config,_elapsed=11.,initial_object_position=[.3,.1,.31],
                            joint_targets_for_pose=lambda goal:self.fail('No student native IK fallback.'))
        proxy=CompliantTransferGoalProxy(env,SimpleNamespace(predict=lambda obs,goal:calls.append(np.array(goal))or[0.]*6))
        policy=CompliantTransferPolicy(proxy);proxy.policy=policy
        policy.stage=policy._action_phase='transfer'
        policy._held_offset=np.array([0.,0.,-.006])
        proxy.observation={'effective_sliding_friction':{'jaw_object':1.2},
            'end_effector_position':[.3,.1,.44],'object_position':[.3,.1,.42],
            'target_position':[.5,-.1,.31],'grasp_contacts':{'left':4,'right':4},
            'pad_normal_forces':{'left':.3,'right':.3},'joint_actuator_torque':[0.]*6,
            'object_size':[.03,.04,.04],'object_velocity':[0.]*3,'object_angular_velocity':[0.]*3,
            'joint_position':[0.]*6,'gripper_width':.03,'table_contacts':0}
        return policy,proxy,calls

    def test_vertical_response_has_both_caps_and_unchanged_reference(self):
        policy,p,calls=self.setup_policy();held=policy._held_offset.copy();config=deepcopy(p.config)
        p.joint_targets_for_pose([.305,.1,.441])
        delta=calls[-1]-p.observation['end_effector_position']
        self.assertLess(delta[2],0)
        self.assertLessEqual(abs(delta[2]),.002)
        self.assertLessEqual(np.linalg.norm(delta),.005+1e-12)
        np.testing.assert_array_equal(policy._held_offset,held)
        self.assertEqual(config,p.config);self.assertEqual(p._elapsed,11.)

    def test_downward_request_never_crosses_excursion_floor(self):
        policy,p,calls=self.setup_policy()
        p.joint_targets_for_pose([.305,.1,.441])
        p.observation['end_effector_position'][2]=.416
        p.observation['object_position'][2]=.4
        p.joint_targets_for_pose([.305,.1,.441])
        self.assertGreaterEqual(calls[-1][2],.415)
        # Actual servo overshoot cannot be undone instantaneously; bounded
        # correction must point upward, never demand more downward motion.
        p.observation['end_effector_position'][2]=.410
        p.joint_targets_for_pose([.305,.1,.441])
        self.assertGreater(calls[-1][2],.410)
        self.assertLessEqual(calls[-1][2]-.410,.002+1e-12)

    def test_clearance_or_contact_loss_falls_back_without_changing_observation(self):
        for change in ({'object_position':[.3,.1,.38]}, {'grasp_contacts':{'left':0,'right':4}}):
            _,p,calls=self.setup_policy();p.observation.update(change);before=deepcopy(p.observation)
            p.joint_targets_for_pose([.305,.1,.441])
            np.testing.assert_array_equal(calls[-1],[.305,.1,.441])
            self.assertFalse(p.last_intervention['active']);self.assertEqual(p.observation,before)

    def test_high_headroom_and_nontransfer_phases_are_exact_noops(self):
        for phase,high in [('transfer',True),('lift',False),('place',False),('recover_wait',False)]:
            policy,p,calls=self.setup_policy(high);policy._action_phase=phase
            p.joint_targets_for_pose([.305,.1,.441])
            np.testing.assert_array_equal(calls[-1],[.305,.1,.441])
            self.assertFalse(p.last_intervention['active'])
            self.assertIsNone(p.first_intervention_time)

    def test_original_offset_and_lost_contact_guards_still_trigger(self):
        for missing in (False,True):
            policy,p,_=self.setup_policy();policy._contact_loss_ticks=4
            if missing:p.observation['grasp_contacts']['left']=0
            else:p.observation['object_position'][2]=.42 # 14mm original-relative error
            action=policy(p.observation)
            self.assertEqual(policy.stage,'recover_wait')
            self.assertEqual(policy._recovery_count,1)
            self.assertEqual(action['gripper_width'],.08)
            np.testing.assert_array_equal(policy._held_offset,[0.,0.,-.006])

    def test_compliant_phase_completion_requires_arrival_contact_clearance(self):
        policy,p,_=self.setup_policy();p.last_intervention={'active':True}
        obs=deepcopy(p.observation)
        self.assertFalse(policy._compliant_transfer_complete(obs))
        obs['end_effector_position'][:2]=obs['target_position'][:2]
        obs['object_position']=[*obs['target_position'][:2],.43]
        self.assertTrue(policy._compliant_transfer_complete(obs))
        obs['grasp_contacts']['left']=0
        self.assertFalse(policy._compliant_transfer_complete(obs))
        obs['grasp_contacts']['left']=4;obs['object_position'][2]=.38
        self.assertFalse(policy._compliant_transfer_complete(obs))

    def test_new_arrival_path_cannot_bypass_pending_original_offset_guard(self):
        policy,p,_=self.setup_policy()
        p.observation['end_effector_position']=[.5,-.1,.42]
        p.observation['object_position']=[.5,-.1,.401]
        policy._contact_loss_ticks=0
        policy(p.observation)
        self.assertEqual(policy.stage,'transfer')
        self.assertEqual(policy._contact_loss_ticks,1)
        np.testing.assert_array_equal(policy._held_offset,[0.,0.,-.006])

    def test_native_baseline_and_before_transfer_actions_identical(self):
        config=replace(ArmConfig(scene_jitter=0),horizon=.4)
        original=rollout(config,317)
        self.assertTrue(_same_trajectory(rollout_retention(config,317),original))
        self.assertTrue(_same_trajectory(rollout_retention(config,317,variant='vertical_compliance'),original))


if __name__=='__main__':unittest.main()
