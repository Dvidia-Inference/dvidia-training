"""Alternate target planning preserves the native world and action limits."""
from dataclasses import asdict, replace
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from dvidia_training import piper_grasp_strategy as grasp
from dvidia_training import piper_task as task


class GraspStrategyTests(unittest.TestCase):
    def test_validation_rejects_unbounded_or_nonfinite_candidates(self):
        for change in ({'yaw_degrees': 46}, {'yaw_degrees': float('nan')}, {'yaw_degrees': True},
                       {'orientation_error_limit_rad': .21}, {'orientation_error_limit_rad': .049},
                       {'orientation_error_limit_rad': 10**1000}, {'retain_carry_yaw': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(grasp.GraspStrategyConfig(), **change)
        with self.assertRaises(ValueError):
            grasp.world_yaw((1, 1, 0, 0), 0)
        for bad in (False, {}, 0, 'unknown'):
            with self.subTest(config=bad), self.assertRaises(TypeError):
                grasp.grasp_strategy_contract(bad)
            with self.subTest(config=bad), self.assertRaises(TypeError):
                grasp.GraspPlanningEnvironment(SimpleNamespace(config=task.PiperTaskConfig()), bad)

    def test_world_yaw_composition_preserves_downward_tool_axis(self):
        q = grasp.world_yaw((0., 0., 1., 0.), -45.)
        self.assertAlmostEqual(sum(v*v for v in q), 1.)
        self.assertEqual((q[0], q[3]), (0., 0.))
        self.assertAlmostEqual(q[1], math.sin(math.pi/8))
        self.assertAlmostEqual(q[2], math.cos(math.pi/8))
        self.assertEqual(grasp.world_yaw((0., 0., 1., 0.), 0), (0., 0., 1., 0.))

    def test_policy_view_changes_only_tool_orientation_and_forwards_native_state(self):
        original = task.PiperTaskConfig()
        marker = object()
        env = SimpleNamespace(config=original, model=marker, elapsed=.14)
        view = grasp.GraspPlanningEnvironment(env)
        self.assertIs(env.config, original)
        self.assertIs(view.model, marker)
        self.assertEqual(view.elapsed, .14)
        before, after = asdict(original), asdict(view.config)
        self.assertEqual([key for key in before if before[key] != after[key]], ['tool_orientation_wxyz'])
        env.elapsed = .16
        self.assertEqual(view.elapsed, .16)

    def test_retained_yaw_keeps_original_approach_and_changes_only_postapproach_targets(self):
        original = task.PiperTaskConfig()
        env = SimpleNamespace(config=original)
        view = grasp.GraspPlanningEnvironment(env, grasp.GraspStrategyConfig(retain_carry_yaw=True))
        self.assertEqual(view.config.carry_orientation_wxyz, original.carry_orientation_wxyz)
        view.motion_phase = 'lift'
        self.assertEqual(view.config.carry_orientation_wxyz, grasp.world_yaw(original.carry_orientation_wxyz, -45))
        self.assertIs(env.config, original)
        self.assertEqual(view.config.tool_orientation_wxyz, grasp.world_yaw(original.tool_orientation_wxyz, -45))

    def test_combined_contract_preserves_mode_and_explicit_scope(self):
        env = SimpleNamespace(config=task.PiperTaskConfig())
        policy = grasp.GraspReacquisitionPiperTaskPolicy(env)
        self.assertEqual(policy.mode, 'recovery')
        self.assertEqual(policy.summary()['mode'], 'reacquire_grasp')
        self.assertEqual(policy.summary()['schema'], grasp.SCHEMA)
        self.assertEqual(policy.summary()['contract'], grasp.combined_contract())
        self.assertEqual(grasp.inactive_summary('reacquire_grasp')['contract'], grasp.combined_contract())
        self.assertFalse(grasp.inactive_summary('reacquire_grasp')['active'])
        with self.assertRaises(ValueError):
            grasp.GraspReacquisitionPiperTaskPolicy(env, 'unknown')


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set DVIDIA_ROBOT_PROFILE_ASSETS to pinned assets')
class NativeGraspPlanningTests(unittest.TestCase):
    def test_zero_rotation_original_clamp_matches_original_ik_without_state_writes(self):
        env = task.PiperTaskEnv(Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS']))
        env.reset()
        _, np = task.sim._native_imports()
        view = grasp.GraspPlanningEnvironment(env, grasp.GraspStrategyConfig(0., .05))
        original_model = task.sim.model_digest(env.model)
        original_config, qpos, qvel, controls = env.config, env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()
        for pose, quaternion in (([.3, -.03, .18], env.config.carry_orientation_wxyz),
                                 ([.3, -.03, .082], env.config.tool_orientation_wxyz)):
            expected = env.pose_action(pose, quaternion, 0.)
            actual = view.pose_action(pose, quaternion, 0.)
            self.assertEqual(actual, expected)
        self.assertIs(env.config, original_config)
        self.assertEqual(task.sim.model_digest(env.model), original_model)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.qvel, qvel)
        np.testing.assert_array_equal(env.data.ctrl, controls)

    def test_alternate_grasp_preserves_original_approach_ik_and_action_bounds(self):
        env = task.PiperTaskEnv(Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS']))
        env.reset()
        _, np = task.sim._native_imports()
        view = grasp.GraspPlanningEnvironment(env)
        pose = [.3, -.03, .18]
        self.assertEqual(view.pose_action(pose, env.config.carry_orientation_wxyz, .035),
                         env.pose_action(pose, env.config.carry_orientation_wxyz, .035))
        action = view.pose_action([.3, -.03, .082], view.config.tool_orientation_wxyz, 0.)
        q = np.asarray(action['joint_targets_rad'])
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(np.all(q >= env.model.actuator_ctrlrange[:6, 0]+.02))
        self.assertTrue(np.all(q <= env.model.actuator_ctrlrange[:6, 1]-.02))
        self.assertEqual(action['gripper_position_m'], 0.)


if __name__ == '__main__':
    unittest.main()
