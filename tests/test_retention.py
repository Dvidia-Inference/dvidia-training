"""Frozen retention intervention must preserve physics, timing and mapping."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from dvidia_training.arm_env import ArmConfig
from dvidia_training.arm_runner import rollout
from dvidia_training.grip_diagnostic import _same_trajectory
from dvidia_training.retention_policy import SmoothedLoadedGoalProxy, SMOOTHING_ALPHA
from dvidia_training.retention_experiment import rollout_retention, measurement, summarize


class RetentionTests(unittest.TestCase):
    def proxy(self, margin_low=True, phase='lift'):
        config = replace(ArmConfig(), object_mass=.06, object_friction=1.2,
                         jaw_force_limit=.3 if margin_low else 3.)
        calls = []
        env = SimpleNamespace(config=config, _elapsed=9.,
                              joint_targets_for_pose=lambda goal: self.fail('Student must not use native IK.'))
        head = SimpleNamespace(predict=lambda obs, goal: calls.append(np.array(goal)) or [0.] * 6)
        proxy = SmoothedLoadedGoalProxy(env, head)
        proxy.policy = SimpleNamespace(_action_phase=phase)
        proxy.observation = {'effective_sliding_friction': {'jaw_object': 1.2},
                             'end_effector_position': [.3, .1, .4]}
        return proxy, calls

    def test_ramp_preserves_cap_and_head_and_clock(self):
        p, calls = self.proxy()
        p.joint_targets_for_pose([.3, .1, .405])
        np.testing.assert_allclose(calls[-1], [.3, .1, .401])
        for _ in range(100):
            p.joint_targets_for_pose([.3, .1, .405])
        self.assertLessEqual(np.linalg.norm(calls[-1]-[.3,.1,.4]), .005+1e-12)
        self.assertAlmostEqual(p.first_intervention_time, 9.)
        self.assertEqual(p._elapsed, 9.)
        self.assertEqual(p.config.jaw_force_limit, .3)

    def test_direction_change_preserves_filtered_increment_across_loaded_phases(self):
        p, calls = self.proxy()
        for _ in range(100):
            p.joint_targets_for_pose([.3,.1,.405])
        p.policy._action_phase = 'transfer'
        p.joint_targets_for_pose([.305,.1,.4])
        np.testing.assert_allclose(calls[-1], [.301,.1,.404], atol=1e-12)
        self.assertLessEqual(np.linalg.norm(calls[-1]-[.3,.1,.4]),.005)

    def test_recovery_and_unloaded_phases_are_not_filtered_and_reset(self):
        for phase in ('close','approach','place','recover_wait','recover_lower_supported'):
            p,calls=self.proxy()
            p.joint_targets_for_pose([.3,.1,.405])
            p.policy._action_phase=phase
            p.joint_targets_for_pose([.32,.12,.45])
            np.testing.assert_allclose(calls[-1],[.32,.12,.45])
            self.assertFalse(p.last_intervention['active'])
            p.policy._action_phase='lift'
            p.joint_targets_for_pose([.3,.1,.405])
            np.testing.assert_allclose(calls[-1],[.3,.1,.401])

    def test_high_headroom_is_exact_noop(self):
        p,calls=self.proxy(False)
        p.joint_targets_for_pose([.32,.12,.45])
        np.testing.assert_array_equal(calls[-1],[.32,.12,.45])
        self.assertEqual(p.intervention_calls,0)
        self.assertIsNone(p.first_intervention_time)

    def test_native_original_reproduction_and_preloaded_noop(self):
        config=replace(ArmConfig(scene_jitter=0),horizon=.4)
        original=rollout(config,317)
        baseline=rollout_retention(config,317)
        candidate=rollout_retention(config,317,variant='loaded_goal_smoothing')
        self.assertTrue(_same_trajectory(baseline,original))
        self.assertTrue(_same_trajectory(candidate,original))
        self.assertIsNone(measurement(candidate)['retention']['first_intervention_time'])

    def test_report_gate_requires_task_gain_and_clean_controls(self):
        def row(v,success):
            return {'case_id':'scene-00','policy':'student','variant':v,'group':'nominal',
                    'status':'completed','control_status':'completed',
                    'measurement':{'success':success},'control':{'success':False}}
        pair=[row('baseline',False),row('loaded_goal_smoothing',False)]
        self.assertFalse(summarize(pair)['fresh_confirmation_gate_met'])
        pair[1]['measurement']['success']=True
        self.assertTrue(summarize(pair)['fresh_confirmation_gate_met'])
        pair[1]['control']['success']=True
        self.assertFalse(summarize(pair)['fresh_confirmation_gate_met'])


if __name__=='__main__':
    unittest.main()
