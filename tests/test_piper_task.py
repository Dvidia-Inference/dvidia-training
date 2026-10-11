"""PiPER contact acquisition, native units, causal traces and independent outcomes."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dvidia_training import piper_task as task
from dvidia_training import robot_profiles


class PiperTaskContractTests(unittest.TestCase):
    def test_config_bounds_units_support_and_roundtrip(self):
        config = task.PiperTaskConfig()
        self.assertEqual(task.PiperTaskConfig(**json.loads(json.dumps(asdict(config)))), config)
        for changes in ({'timestep': True}, {'timestep': .003}, {'timeout_s': .11}, {'timeout_s': 31},
                        {'mass_kg': 10**1000}, {'friction': float('nan')}, {'control_dt': .01},
                        {'tool_orientation_wxyz': [1, 1, 0, 0]}, {'target_position_m': [.35, .08, .1]},
                        {'object_position_m': [.35, .04, .082]}, {'jaw_target_rate_m_s': 5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)
        contract = task.task_contract()
        self.assertEqual(contract['arm_joint_order'], ['joint'+str(i) for i in range(1, 7)])
        self.assertIn('metres', contract['actions']['gripper_position_m'])
        self.assertIn('unchanged', contract['servo_gains'])
        self.assertFalse(contract['physical_robot_ready'])

    def test_setup_failure_and_incomplete_control_stays_in_attempt_denominator(self):
        with patch.object(robot_profiles, 'verify_assets', side_effect=ValueError('fixture bad asset')):
            attempts = task.run_matched('unused')
        self.assertEqual([row['control'] for row in attempts], ['placement', 'replay_open_jaw', 'hold'])
        self.assertTrue(all(not row['valid'] and not row['success'] for row in attempts))
        self.assertEqual(attempts[1]['error']['phase'], 'validation')
        for result in attempts:
            self.assertEqual(result['trajectory'], [])
            self.assertEqual(result['actions'], [])
            self.assertGreater(result['timing']['episode_wall_seconds'], 0)
            for name, digest in result['implementation_sha256'].items():
                self.assertEqual(digest, sha256(Path(task.__file__).with_name(name).read_bytes()).hexdigest())
        for config in (False, {}, 'unknown'):
            with self.assertRaises(TypeError):
                task.run_episode('unused', config)

    def test_invalid_complete_tape_cannot_claim_a_matched_control(self):
        placement = {'valid': False, 'arm_tape': [[0]*6]*1000}
        with patch.object(task, 'run_episode', side_effect=[placement, {}, {}]) as run:
            task.run_matched('unused')
        self.assertIsNone(run.call_args_list[1].kwargs['arm_tape'])


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set DVIDIA_ROBOT_PROFILE_ASSETS to pinned assets')
class NativePiperTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.mj, cls.np = task.sim._native_imports()
        cls.attempts = task.run_matched(cls.assets)

    def test_stock_servos_forces_coupling_and_gravity_are_preserved(self):
        model = task.build_task_model(self.assets)
        original = robot_profiles.build_model('piper', self.assets)
        self.assertEqual((model.nq, model.nv, model.nu, model.neq, model.nmocap), (15, 14, 7, 1, 0))
        self.np.testing.assert_array_equal(model.actuator_gainprm, original.actuator_gainprm)
        self.np.testing.assert_array_equal(model.actuator_biasprm, original.actuator_biasprm)
        self.np.testing.assert_array_equal(model.actuator_forcerange, original.actuator_forcerange)
        self.np.testing.assert_array_equal(model.actuator_ctrlrange, original.actuator_ctrlrange)
        self.np.testing.assert_array_equal(model.body_gravcomp[:10], original.body_gravcomp)
        joint = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_JOINT, 'task_block_free')
        block = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_BODY, 'task_block')
        self.assertEqual(model.jnt_type[joint], self.mj.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.body_gravcomp[block], 0)
        self.assertEqual(model.eq_type[0], self.mj.mjtEq.mjEQ_JOINT)
        self.assertNotIn(joint, model.eq_obj1id)
        self.assertNotIn(joint, model.eq_obj2id)
        marker = self.mj.mj_name2id(model, self.mj.mjtObj.mjOBJ_GEOM, 'task_target')
        self.assertEqual((model.geom_contype[marker], model.geom_conaffinity[marker]), (0, 0))

    def test_native_acquisition_carry_supported_release_and_retreat(self):
        attempt = self.attempts[0]
        self.assertTrue(attempt['valid'], attempt['error'])
        self.assertTrue(attempt['success'], attempt['metrics'])
        self.assertEqual(attempt['warnings'], [])
        metrics = attempt['metrics']
        self.assertTrue(metrics['grasp_seen'])
        self.assertTrue(metrics['lift_seen'])
        self.assertTrue(metrics['carry_complete'])
        self.assertFalse(metrics['retention_lost'])
        self.assertEqual(metrics['unexpected_contact_native_ticks'], 0)
        self.assertGreaterEqual(metrics['max_object_lift_m'], .06)
        self.assertLess(metrics['target_distance_m'], .025)
        final = attempt['trajectory'][-1]
        self.assertEqual(final['grasp_contact_counts'], {'left': 0, 'right': 0})
        self.assertGreater(final['table_object_contact_count'], 0)
        self.assertGreater(final['gripper_joint_position_m'][0], .028)
        self.assertLess(final['gripper_joint_position_m'][1], -.028)
        self.assertGreaterEqual(final['tcp_position_m'][2]-final['object_position_m'][2], .10)
        self.assertLess(self.np.linalg.norm(final['object_linear_velocity_m_s']), .035)
        self.assertLess(self.np.linalg.norm(final['object_angular_velocity_rad_s']), .3)

    def test_matched_open_and_hold_fail_without_lift_or_transport(self):
        placement, opened, held = self.attempts
        self.assertEqual(opened['arm_tape'], placement['arm_tape'])
        for attempt in (opened, held):
            self.assertTrue(attempt['valid'], attempt['error'])
            self.assertFalse(attempt['success'])
            self.assertFalse(attempt['metrics']['lift_seen'])
            self.assertEqual(len(attempt['actions']), len(placement['actions']))
            self.assertEqual(attempt['compiled_model_sha256'], placement['compiled_model_sha256'])
            self.assertTrue(all(row['gripper_position_m'] == .035 for row in attempt['actions']))
            self.np.testing.assert_allclose(attempt['trajectory'][-1]['object_position_m'], attempt['initial_object_position_m'], atol=1e-6, rtol=0)

    def test_causal_native_trace_and_commands_have_no_legacy_jaw_units(self):
        for attempt in self.attempts:
            self.assertEqual(len(attempt['actions']), 1000)
            self.assertEqual(len(attempt['trajectory']), 1001)
            for index, row in enumerate(attempt['trajectory']):
                self.assertAlmostEqual(row['t_seconds'], index*.02, places=8)
                self.assertEqual(len(row['qpos']), 15)
                self.assertEqual(len(row['qvel']), 14)
                self.assertEqual(row['joint_position_rad'], row['qpos'][:6])
                self.assertEqual(row['gripper_joint_position_m'], row['qpos'][6:8])
                self.assertNotIn('issued_gripper_0_255', row)
                if index < len(attempt['actions']):
                    action = attempt['actions'][index]
                    self.assertEqual(row['command'], action)
                    self.assertEqual(row['issued_gripper_position_m'], action['gripper_position_m'])
                    self.assertEqual(row['issued_joint_targets_rad'], action['joint_targets_rad'])
                else:
                    self.assertIsNone(row['command'])
                    self.assertIsNone(row['issued_gripper_position_m'])
            json.dumps(attempt, allow_nan=False)

    def test_action_bounds_native_rate_limits_and_external_force_rejection(self):
        env = task.PiperTaskEnv(self.assets)
        obs, _ = env.reset()
        before, jaw = env.command_targets.copy(), env.command_gripper
        targets = list(obs['joint_position_rad'])
        targets[0] += .2
        env.step({'joint_targets_rad': targets, 'gripper_position_m': 0})
        self.assertLessEqual(max(abs(env.command_targets-before)), .03+1e-8)
        self.assertLessEqual(abs(env.command_gripper-jaw), .001+1e-8)
        for action in ({'joint_targets_rad': [True]*6, 'gripper_position_m': 0},
                       {'joint_targets_rad': [0]*5, 'gripper_position_m': 0},
                       {'joint_targets_rad': targets, 'gripper_position_m': .036},
                       {'joint_targets_rad': targets, 'gripper_position_m': True},
                       {'joint_targets_rad': targets, 'gripper_0_255': 0}):
            with self.assertRaises(ValueError):
                env.step(action)
        env.data.xfrc_applied[env.block, 2] = .4
        env.lift_seen = env.grasp_seen = env.carry_complete = True
        env.dwell = env.config.goal_dwell_s
        self.assertFalse(env.info()['valid'])
        self.assertFalse(env.info()['success'])
        with patch.object(self.mj, 'mj_step', side_effect=AssertionError('External-force state cannot advance')):
            observation, info = env.step({'joint_targets_rad': targets, 'gripper_position_m': .035})
        self.assertIsNone(observation)
        self.assertFalse(info['valid'])
        self.assertFalse(info['success'])

    def test_late_solver_warning_cannot_preserve_a_previous_success_claim(self):
        env = task.PiperTaskEnv(self.assets)
        env.reset()
        env.lift_seen = env.grasp_seen = env.carry_complete = True
        env.dwell = env.config.goal_dwell_s
        self.assertTrue(env.info()['success'])
        env.warnings.append('fixture late native warning')
        env.reason = 'solver_warning'
        self.assertFalse(env.info()['valid'])
        self.assertFalse(env.info()['success'])

    def test_pose_ik_does_not_rewrite_arm_or_free_object_state(self):
        env = task.PiperTaskEnv(self.assets)
        observation, _ = env.reset()
        before = env.data.qpos.copy()
        env.pose_action(observation['tcp_position_m'], env.config.tool_orientation_wxyz, .035)
        self.np.testing.assert_array_equal(env.data.qpos, before)

    def test_reset_rejects_displaced_or_unsupported_initial_block(self):
        env = task.PiperTaskEnv(self.assets)
        with patch.object(env, '_contacts', return_value=({'left': 0, 'right': 0}, {'left': 0., 'right': 0.}, 0, 0, [])):
            with self.assertRaisesRegex(RuntimeError, 'Reset must settle'):
                env.reset()


if __name__ == '__main__':
    unittest.main()
