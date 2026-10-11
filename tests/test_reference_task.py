"""Native free-block task, bounded six-axis actuation and causal evidence."""
from dataclasses import asdict, replace
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dvidia_training import reference_sim as sim
from dvidia_training.reference_task import (GRIPPER_JOINTS, ReferenceTaskConfig, ReferenceTaskEnv,
                                           build_task_model, run_episode, run_matched, task_contract)


class ReferenceTaskContractTests(unittest.TestCase):
    def test_config_validates_units_bounds_timing_and_support(self):
        config = ReferenceTaskConfig()
        self.assertEqual(ReferenceTaskConfig(**json.loads(json.dumps(asdict(config)))), config)
        for changes in ({'timestep': True}, {'timestep': .003}, {'control_dt': .03}, {'timeout_s': .11},
                        {'timeout_s': 31}, {'mass_kg': float('nan')}, {'mass_kg': 10**1000},
                        {'friction': 0}, {'object_position_m': [.04, .49, .302]},
                        {'target_position_m': [.08, .49, .31]}, {'object_half_size_m': [.01, .02, .02]},
                        {'tool_orientation_wxyz': [1, 1, 0, 0]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)

    def test_profile_keeps_native_jaw_command_and_privileged_sensing_explicit(self):
        contract = task_contract()
        self.assertFalse(contract['physical_robot_ready'])
        self.assertEqual(contract['native_sensor_definitions'], 0)
        self.assertEqual(contract['arm_joint_order'], list(sim.ARM_JOINTS))
        self.assertEqual(contract['gripper_linkage_order'], list(GRIPPER_JOINTS))
        self.assertIn('not measured gap or force', contract['actions']['gripper_0_255'])
        self.assertIn('privileged', contract['observation_origin'])
        self.assertIn('free rigid box', contract['object'])

    def test_setup_exceptions_are_retained_as_attempts_without_success(self):
        with patch.object(sim, 'verify_assets', side_effect=ValueError('fixture asset mismatch')):
            attempt = run_episode('unused')
        self.assertFalse(attempt['success'])
        self.assertFalse(attempt['valid'])
        self.assertEqual(attempt['reason'], 'exception')
        self.assertEqual(attempt['error']['phase'], 'setup')
        self.assertEqual(attempt['error']['type'], 'ValueError')
        self.assertEqual(attempt['trajectory'], [])
        self.assertEqual(attempt['actions'], [])
        self.assertGreater(attempt['timing']['episode_wall_seconds'], 0)
        for name, digest in attempt['implementation_sha256'].items():
            path = Path(__import__('dvidia_training.reference_task', fromlist=['x']).__file__).parent / name
            self.assertEqual(digest, sha256(path.read_bytes()).hexdigest())
        for config in (False, {}, 'unknown'):
            with self.assertRaises(TypeError):
                run_episode('unused', config)

    def test_incomplete_open_jaw_tapes_are_retained_invalid_without_native_setup(self):
        with patch.object(sim, 'verify_assets', side_effect=AssertionError('Invalid tapes must not compile')):
            for tape in (None, [], [[0]*6], [[0]*6]*999, [[0]*6]*1001, [[True]*6]*1000):
                with self.subTest(length=len(tape) if tape is not None else None):
                    attempt = run_episode('unused', control='replay_open_jaw', arm_tape=tape)
                    self.assertFalse(attempt['valid'])
                    self.assertFalse(attempt['success'])
                    self.assertEqual(attempt['error']['phase'], 'validation')
                    self.assertEqual(attempt['actions'], [])
                    self.assertEqual(attempt['trajectory'], [])
        with patch.object(sim, 'verify_assets', side_effect=ValueError('fixture unavailable assets')):
            attempts = run_matched('unused')
        self.assertEqual([attempt['control'] for attempt in attempts], ['placement', 'replay_open_jaw', 'hold'])
        self.assertTrue(all(not attempt['valid'] and not attempt['success'] for attempt in attempts))


@unittest.skipUnless(os.environ.get('DVIDIA_REFERENCE_ASSETS'),
                     'Set DVIDIA_REFERENCE_ASSETS to pinned native assets')
class NativeReferenceTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_REFERENCE_ASSETS'])
        cls.mj, cls.np = sim._native_imports()
        cls.matched = run_matched(cls.assets)

    def test_native_success_requires_free_contact_grasp_lift_release_and_retreat(self):
        attempt = self.matched[0]
        self.assertTrue(attempt['valid'], attempt['error'])
        self.assertTrue(attempt['success'], attempt['metrics'])
        self.assertEqual(attempt['reason'], 'success')
        self.assertEqual(attempt['warnings'], [])
        metrics = attempt['metrics']
        self.assertTrue(metrics['grasp_seen'])
        self.assertTrue(metrics['lift_seen'])
        self.assertTrue(metrics['carry_complete'])
        self.assertIs(type(metrics['carry_complete']), bool)
        self.assertFalse(metrics['retention_lost'])
        self.assertEqual(metrics['carry_contact_loss_events'], 0)
        self.assertGreaterEqual(metrics['max_object_lift_m'], .06)
        self.assertGreaterEqual(metrics['dwell_seconds'], .25)
        self.assertEqual(metrics['unexpected_contact_native_ticks'], 0)
        self.assertLess(metrics['target_distance_m'], .025)
        final = attempt['trajectory'][-1]
        self.assertEqual(final['grasp_contact_counts'], {'left': 0, 'right': 0})
        self.assertGreater(final['table_object_contact_count'], 0)
        self.assertGreaterEqual(final['tcp_position_m'][2]-final['object_position_m'][2], .10)
        self.assertLess(self.np.linalg.norm(final['object_linear_velocity_m_s']), .035)
        self.assertLess(self.np.linalg.norm(final['object_angular_velocity_rad_s']), .3)
        self.assertLess(max(final['linkage_position_rad'][0], final['linkage_position_rad'][4]), .1)

    def test_matched_open_jaw_keeps_every_arm_target_and_never_acquires_the_block(self):
        placement, opened, held = self.matched
        for control in (opened, held):
            self.assertTrue(control['valid'], control['error'])
            self.assertFalse(control['success'])
            self.assertFalse(control['metrics']['grasp_seen'])
            self.assertFalse(control['metrics']['lift_seen'])
            self.assertEqual(len(control['actions']), len(placement['actions']))
            self.assertAlmostEqual(control['simulated_seconds'], placement['simulated_seconds'])
            self.assertEqual(control['compiled_model_sha256'], placement['compiled_model_sha256'])
            for row in control['trajectory']:
                self.np.testing.assert_allclose(row['object_position_m'], control['initial_object_position_m'], atol=1e-6, rtol=0)
        self.assertEqual(opened['arm_tape'], placement['arm_tape'])
        self.assertTrue(all(row['gripper_0_255'] == 0 for row in opened['actions']))

    def test_trace_preserves_before_decision_commands_and_final_unissued_state(self):
        for attempt in self.matched:
            self.assertEqual(len(attempt['trajectory']), len(attempt['actions'])+1)
            self.assertEqual(len(attempt['actions']), 1000)
            for index, row in enumerate(attempt['trajectory']):
                self.assertAlmostEqual(row['t_seconds'], index*.02, places=8)
                self.assertEqual(len(row['qpos']), 21)
                self.assertEqual(len(row['qvel']), 20)
                self.assertEqual(len(row['joint_position_rad']), 6)
                self.assertEqual(len(row['joint_velocity_rad_s']), 6)
                self.assertEqual(len(row['linkage_position_rad']), 8)
                if index < len(attempt['actions']):
                    action = attempt['actions'][index]
                    self.assertEqual(row['command'], action)
                    self.assertEqual(row['issued_joint_targets_rad'], action['joint_targets_rad'])
                    self.assertEqual(row['issued_gripper_0_255'], action['gripper_0_255'])
                    self.assertEqual(action['t_seconds'], row['t_seconds'])
                else:
                    self.assertIsNone(row['command'])
                    self.assertIsNone(row['issued_joint_targets_rad'])
                    self.assertIsNone(row['issued_gripper_0_255'])
            json.dumps(attempt, allow_nan=False)
            self.assertEqual(attempt['timing']['control_boundaries'], len(attempt['actions']))
            self.assertGreaterEqual(attempt['timing']['missed_control_boundaries'], 0)
            self.assertIn('not real-time sensor-to-actuator', attempt['timing']['scope'])

    def test_carry_gate_rejects_drops_and_tosses_even_if_the_destination_settles(self):
        environment = ReferenceTaskEnv(self.assets)
        environment.reset()

        def tick(**changes):
            evidence = dict(both=True, supported=False, lift=.08, distance=.15,
                            released=False, settled=False, clear=False, critical=0, pad_force=2.)
            evidence.update(changes)
            environment.elapsed += environment.config.timestep
            environment._outcome_tick(**evidence)

        for _ in range(100):
            tick()
        self.assertTrue(environment.lift_seen)
        for _ in range(39):
            tick(both=False)
        self.assertFalse(environment.retention_lost)
        tick(both=False)
        self.assertTrue(environment.retention_lost)
        self.assertEqual(environment.carry_contact_loss_events, 1)
        self.assertAlmostEqual(environment.max_carry_contact_loss, .04)
        for _ in range(300):
            tick(both=False, supported=True, lift=0., distance=0., released=True, settled=True, clear=True)
        self.assertFalse(environment.carry_complete)
        self.assertFalse(environment.info()['success'])
        # Regrasping at the destination cannot erase the failed carry.
        tick(supported=True, lift=0., distance=0.)
        self.assertTrue(environment.carry_complete)
        for _ in range(300):
            tick(both=False, supported=True, lift=0., distance=0., released=True, settled=True, clear=True)
        self.assertFalse(environment.info()['success'])

    def test_short_contact_gap_recovers_but_release_requires_supported_held_delivery(self):
        environment = ReferenceTaskEnv(self.assets)
        environment.reset()
        evidence = dict(both=True, supported=False, lift=.08, distance=.15,
                        released=False, settled=False, clear=False, critical=0, pad_force=2.)
        for _ in range(100):
            environment._outcome_tick(**evidence)
        for _ in range(39):
            environment._outcome_tick(**{**evidence, 'both': False})
        environment._outcome_tick(**evidence)
        self.assertEqual(environment.carry_contact_loss, 0.)
        self.assertFalse(environment.retention_lost)
        environment._outcome_tick(**{**evidence, 'distance': 0.})
        self.assertFalse(environment.carry_complete, 'Passing over a destination is insufficient')
        environment._outcome_tick(**{**evidence, 'distance': 0., 'supported': True, 'lift': 0.})
        self.assertTrue(environment.carry_complete)
        for _ in range(300):
            environment._outcome_tick(**{**evidence, 'both': False, 'distance': 0., 'supported': True,
                                         'lift': 0., 'released': True, 'settled': True, 'clear': True})
        self.assertTrue(environment.info()['success'])

    def test_late_native_warning_revokes_an_already_successful_dwell(self):
        environment = ReferenceTaskEnv(self.assets)
        observation, _ = environment.reset()
        environment.lift_seen = environment.grasp_seen = environment.carry_complete = True
        environment.dwell = environment.config.goal_dwell_s
        environment.first_success_time = 11.49
        self.assertTrue(environment.info()['success'])

        @contextmanager
        def warning(messages):
            messages.append('fixture late native solver warning')
            yield

        with patch.object(sim, 'native_warnings', warning):
            _, _, terminated, _, info = environment.step({'joint_targets_rad': observation['joint_position_rad'],
                                                        'gripper_0_255': 0.})
        self.assertTrue(terminated)
        self.assertFalse(info['valid'])
        self.assertFalse(info['success'])
        self.assertEqual(info['reason'], 'solver_warning')
        self.assertEqual(info['first_success_time_seconds'], 11.49)
        self.assertEqual(info['warnings'], ['fixture late native solver warning'])

    def test_task_adds_only_free_block_and_preserves_coupled_reference_constraints(self):
        model = build_task_model(self.assets)
        mj = self.mj
        self.assertEqual((model.nq, model.nv, model.nu, model.neq), (21, 20, 7, 3))
        block = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'task_block')
        joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'task_block_free')
        self.assertEqual(model.jnt_type[joint], mj.mjtJoint.mjJNT_FREE)
        self.assertNotIn(block, model.eq_obj1id)
        self.assertNotIn(block, model.eq_obj2id)
        self.assertEqual(model.nsensor, 0)
        block_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'task_block_geom')
        self.assertEqual(model.geom_priority[block_geom], 1)
        changed = build_task_model(self.assets, replace(ReferenceTaskConfig(), friction=.2))
        self.assertAlmostEqual(changed.geom_friction[block_geom, 0], .2)
        environment = ReferenceTaskEnv(self.assets, replace(ReferenceTaskConfig(), friction=.2))
        environment.reset()
        contacts = [contact for contact in environment.data.contact[:environment.data.ncon]
                    if block_geom in (contact.geom1, contact.geom2)]
        self.assertTrue(contacts)
        self.assertTrue(all(abs(float(contact.friction[0])-.2) < 1e-9 for contact in contacts))

    def test_targets_ramp_inside_original_native_force_limits_and_invalid_state_cannot_advance(self):
        environment = ReferenceTaskEnv(self.assets)
        observation, _ = environment.reset()
        before = environment.command_targets.copy()
        targets = list(observation['joint_position_rad'])
        targets[0] += .2
        environment.step({'joint_targets_rad': targets, 'gripper_0_255': 255.})
        self.assertLessEqual(max(abs(environment.command_targets-before)), .03+1e-8)
        self.assertLessEqual(environment.command_gripper, 10+1e-8)
        self.assertTrue(self.np.all(environment.data.actuator_force >= environment.model.actuator_forcerange[:, 0]-1e-8))
        self.assertTrue(self.np.all(environment.data.actuator_force <= environment.model.actuator_forcerange[:, 1]+1e-8))
        for action in ({'joint_targets_rad': [0]*5, 'gripper_0_255': 0},
                       {'joint_targets_rad': [True]*6, 'gripper_0_255': 0},
                       {'joint_targets_rad': [0]*6, 'gripper_0_255': True},
                       {'joint_targets_rad': [0]*6, 'gripper_0_255': 256},
                       {'joint_targets_rad': [0]*6, 'gripper_0_255': 0, 'object_position': [0, 0, 0]}):
            with self.assertRaises(ValueError):
                environment.step(action)
        environment.data.qpos[0] = float('nan')
        with patch.object(self.mj, 'mj_step', side_effect=AssertionError('Corrupted state must not step')):
            observation, _, terminated, _, info = environment.step({'joint_targets_rad': targets, 'gripper_0_255': 0})
        self.assertIsNone(observation)
        self.assertTrue(terminated)
        self.assertFalse(info['valid'])
        self.assertEqual(info['reason'], 'invalid_state')

    def test_pose_ik_reduces_orientation_error_without_position_teleport(self):
        environment = ReferenceTaskEnv(self.assets)
        observation, _ = environment.reset()
        mj, np = self.mj, self.np
        goal = np.asarray(observation['tcp_quaternion_wxyz'])
        increment = np.asarray([__import__('math').cos(.05), 0., 0., __import__('math').sin(.05)])
        target = np.empty(4)
        mj.mju_mulQuat(target, increment, goal)
        initial_qpos = environment.data.qpos.copy()
        action = environment.pose_action(observation['tcp_position_m'], target.tolist(), 0)
        np.testing.assert_array_equal(environment.data.qpos, initial_qpos)
        error = np.empty(3)
        mj.mju_subQuat(error, target, goal)
        initial_error = np.linalg.norm(error)
        for _ in range(100):
            observation, *_ = environment.step(action)
            action = environment.pose_action(observation['tcp_position_m'], target.tolist(), 0)
        mj.mju_subQuat(error, target, np.asarray(observation['tcp_quaternion_wxyz']))
        self.assertLess(np.linalg.norm(error), initial_error*.25)


if __name__ == '__main__':
    unittest.main()
