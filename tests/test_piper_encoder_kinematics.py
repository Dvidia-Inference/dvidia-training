"""Static geometry checks only: no settling, integration, rendering or policy fit.

Set DVIDIA_ROBOT_PROFILE_ASSETS to the existing pinned asset directory. Tests
compile robot/full-task snapshots but forbid Python dynamics entry points.
"""
from contextlib import ExitStack
from hashlib import sha256
import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from dvidia_training import piper_encoder_kinematics as kin


def quaternion_matrix(q):
    # Independent closed-form rotation, without MuJoCo quaternion utilities.
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def manual_xml_fk(xml, q):
    """Independent homogeneous-transform chain read from the locked XML."""
    body = ET.fromstring(xml).find('worldbody/body')
    rotation, position = np.eye(3), np.zeros(3)
    for index in range(7):
        local_position = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
        local_rotation = quaternion_matrix(np.fromstring(body.get('quat', '1 0 0 0'), sep=' '))
        position += rotation @ local_position
        rotation = rotation @ local_rotation
        if index:
            joint = body.find('joint')
            axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
            axis /= np.linalg.norm(axis)
            anchor = np.fromstring(joint.get('pos', '0 0 0'), sep=' ')
            x, y, z = axis
            cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            angle = q[index-1]
            turn = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross @ cross)
            position += rotation @ (anchor - turn @ anchor)
            rotation = rotation @ turn
        if index < 6:
            body = body.find('body')
    return position, rotation


class EncoderInterfaceTests(unittest.TestCase):
    def test_no_live_model_or_data_constructor_and_no_dynamics_calls(self):
        self.assertEqual(list(inspect.signature(kin.PiperEncoderKinematics).parameters), ['assets'])
        source = inspect.getsource(kin)
        for forbidden in ('.mj_step(', '.mj_forward(', '.mj_collision(', '.ncon', '.contact', '.ctrl'):
            self.assertNotIn(forbidden, source)
        self.assertIn('measured_joint_position_rad', inspect.signature(kin.PiperEncoderKinematics.pose_action).parameters)

    def test_wrong_lock_refused_before_assets_or_native_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'changed-lock.json'
            path.write_text('{}')
            with patch.object(kin.robot_profiles, 'LOCK_PATH', path), \
                    patch.object(kin.robot_profiles, '_payload') as payload, \
                    patch.object(kin.sim, '_native_imports') as native:
                with self.assertRaisesRegex(ValueError, 'asset lock'):
                    kin.PiperEncoderKinematics(directory)
                payload.assert_not_called()
                native.assert_not_called()


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set DVIDIA_ROBOT_PROFILE_ASSETS to pinned assets')
class StaticEncoderGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Path(os.environ['DVIDIA_ROBOT_PROFILE_ASSETS'])
        cls.mj, _ = kin.sim._native_imports()
        cls.guards = ExitStack()
        cls.addClassCleanup(cls.guards.close)
        for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse', 'mj_collision', 'mj_fwdPosition'):
            cls.guards.enter_context(patch.object(cls.mj, name, side_effect=AssertionError('Dynamics forbidden in static geometry test')))
        cls.robot = kin.PiperEncoderKinematics(cls.assets)
        cls.xml = (cls.assets/'agilex_piper/piper.xml').read_text()
        cls.lower, cls.upper = np.array(kin.ARM_LIMITS_RAD).T
        cls.reset = np.array([0., 1.57, -1.3485, 0., 0., 0.])
        cls.configurations = [cls.reset, np.array([.4, .9, -1.9, .3, -.4, .8]),
                              np.array([-.7, 2.1, -.8, -.6, .5, -1.1]), cls.lower, cls.upper]
        cls.configurations += list(np.random.default_rng(20261109).uniform(cls.lower+.1, cls.upper-.1, (12, 6)))
        from dvidia_training import piper_task
        cls.task = piper_task
        # Original, unchanged method is the parity oracle; no copied DLS test oracle.
        if sha256(Path(piper_task.__file__).read_bytes()).hexdigest() != 'e66a44a8df4e7a8ee42e4f3866f889f8797cdfb8e2688a146e09b75d2c76664f':
            raise AssertionError('Original pose_action oracle source changed')
        cls.task_model = piper_task.build_task_model(cls.assets)

    def test_fk_matches_independent_xml_chain_across_joint_space(self):
        for q in self.configurations:
            with self.subTest(q=q.tolist()):
                link_position, rotation = manual_xml_fk(self.xml, q)
                pose = self.robot.fk(q)
                np.testing.assert_allclose(pose['link6_position_base_m'], link_position, atol=2e-12, rtol=0)
                np.testing.assert_allclose(quaternion_matrix(pose['link6_quaternion_base_wxyz']), rotation, atol=2e-12, rtol=0)
                expected = link_position + rotation @ np.array([0., 0., .12003])
                np.testing.assert_allclose(pose['tcp_position_world_m'], expected, atol=2e-12, rtol=0)
                np.testing.assert_allclose(quaternion_matrix(pose['tcp_quaternion_world_wxyz']), rotation, atol=2e-12, rtol=0)
                self.assertAlmostEqual(np.linalg.norm(expected-link_position), .12003, places=12)

    def test_world_spatial_jacobian_matches_central_finite_differences(self):
        epsilon = 1e-6
        for q in self.configurations[:3] + self.configurations[5:]:
            analytic = self.robot.jacobian(q)
            expected = np.empty((6, 6))
            for joint in range(6):
                step = np.eye(6)[joint]*epsilon
                plus, minus = self.robot.fk(q+step), self.robot.fk(q-step)
                expected[:3, joint] = (np.array(plus['tcp_position_world_m'])-minus['tcp_position_world_m'])/(2*epsilon)
                rp = quaternion_matrix(plus['tcp_quaternion_world_wxyz'])
                rm = quaternion_matrix(minus['tcp_quaternion_world_wxyz'])
                world_relative = rp @ rm.T
                expected[3:, joint] = np.array([world_relative[2, 1]-world_relative[1, 2],
                    world_relative[0, 2]-world_relative[2, 0], world_relative[1, 0]-world_relative[0, 1]])/(4*epsilon)
            np.testing.assert_allclose(analytic, expected, atol=7e-10, rtol=0)

    def test_original_full_task_static_pose_action_parity(self):
        for index, q in enumerate(self.configurations):
            env = self.task.PiperTaskEnv.__new__(self.task.PiperTaskEnv)
            env.ready, env.done = True, False
            env.model, env.data = self.task_model, self.mj.MjData(self.task_model)
            env.tcp = self.mj.mj_name2id(env.model, self.mj.mjtObj.mjOBJ_SITE, 'piper_task_tcp')
            env.data.qpos[:6] = q
            # Different fingers and object pose cannot enter the private encoder model.
            env.data.qpos[6:8] = [.005*index % .035, -(.005*index % .035)]
            env.data.qpos[8:11] = [.2+index*.01, -.2, .3]
            self.mj.mj_kinematics(env.model, env.data)
            self.mj.mj_comPos(env.model, env.data)
            for desired, quat in (([.31, -.07, .12], [0., 0., 1., 0.]),
                                  ([-.1, .2, .3], [.5, -.5, .5, -.5])):
                expected = env.pose_action(desired, quat, .017)
                actual = self.robot.pose_action(q, desired, quat, .017)
                np.testing.assert_allclose(actual['joint_targets_rad'], expected['joint_targets_rad'], atol=2e-13, rtol=0)
                self.assertEqual(actual['gripper_position_m'], expected['gripper_position_m'])
            self.assertEqual(env.data.time, 0.)

    def test_quaternion_sign_equivalence_and_zero_error(self):
        for q in self.configurations[:3]:
            positive = self.robot.pose_action(q, [.3, .1, .2], [.5, -.5, .5, -.5], 0.)
            negative = self.robot.pose_action(q, [.3, .1, .2], [-.5, .5, -.5, .5], 0.)
            np.testing.assert_allclose(positive['joint_targets_rad'], negative['joint_targets_rad'], atol=2e-15, rtol=0)
            pose = self.robot.fk(q)
            action = self.robot.pose_action(q, pose['tcp_position_world_m'], pose['tcp_quaternion_world_wxyz'], .035)
            np.testing.assert_allclose(action['joint_targets_rad'], q, atol=2e-15, rtol=0)

    def test_clipped_targets_honor_delta_and_margin_including_boundaries(self):
        for q in self.configurations:
            action = self.robot.pose_action(q, [9., -9., 9.], [0., 1., 0., 0.], .035)
            target = np.array(action['joint_targets_rad'])
            self.assertTrue(np.all(np.abs(target-q) <= .06+1e-12))
            self.assertTrue(np.all(target >= self.lower+.02))
            self.assertTrue(np.all(target <= self.upper-.02))

    def test_stateless_copied_results_and_inputs(self):
        a, b = self.configurations[:2]
        original = a.copy()
        first = self.robot.fk(a)
        jacobian = self.robot.jacobian(a)
        action = self.robot.pose_action(a, [.3, 0., .2], [0., 0., 1., 0.], .02)
        self.robot.fk(b)
        self.robot.pose_action(b, [.4, .1, .15], [1., 0., 0., 0.], 0.)
        self.assertEqual(self.robot.fk(a), first)
        np.testing.assert_array_equal(self.robot.jacobian(a), jacobian)
        self.assertEqual(self.robot.pose_action(a, [.3, 0., .2], [0., 0., 1., 0.], .02), action)
        first['tcp_position_world_m'][0] = 100.
        jacobian[:] = 100.
        action['joint_targets_rad'][0] = 100.
        self.assertNotEqual(self.robot.fk(a)['tcp_position_world_m'][0], 100.)
        self.assertFalse(np.any(self.robot.jacobian(a) == 100.))
        np.testing.assert_array_equal(a, original)
        self.assertEqual(self.robot._data.time, 0.)
        np.testing.assert_array_equal(self.robot._data.qvel, np.zeros(8))

    def test_bad_measured_values_and_overshoot_are_refused(self):
        bad = [None, '000000', [0.]*5, [[0.]*6], np.zeros((6, 1)), np.zeros(6, dtype=bool),
               [False, 1., -1., 0., 0., 0.], ['0', 1., -1., 0., 0., 0.],
               [np.nan, 1., -1., 0., 0., 0.], [np.inf, 1., -1., 0., 0., 0.],
               self.lower-np.array([1e-12, 0., 0., 0., 0., 0.]), self.upper+1e-12]
        for values in bad:
            for method in (self.robot.fk, self.robot.jacobian):
                with self.subTest(value=repr(values), method=method.__name__), self.assertRaises(ValueError):
                    method(values)
            with self.assertRaises(ValueError):
                self.robot.pose_action(values, [.3, 0., .2], [1., 0., 0., 0.], .02)

    def test_bad_targets_and_jaw_units_are_refused(self):
        for position in ([0., 0.], [[.3], [0.], [.2]], [True, 0., .2], [np.nan, 0., .2], [11., 0., .2]):
            with self.subTest(position=position), self.assertRaises(ValueError):
                self.robot.pose_action(self.reset, position, [1., 0., 0., 0.], .02)
        for quaternion in ([0., 0., 0., 0.], [1., 1., 0., 0.], [1.000000001, 0., 0., 0.],
                           [1., 0., 0.], [True, 0., 0., 0.], [np.inf, 0., 0., 0.]):
            with self.subTest(quaternion=quaternion), self.assertRaises(ValueError):
                self.robot.pose_action(self.reset, [.3, 0., .2], quaternion, .02)
        for jaw in (-1e-12, .035000000001, .07, True, '0.02', np.nan, np.inf, [.02]):
            with self.subTest(jaw=jaw), self.assertRaises(ValueError):
                self.robot.pose_action(self.reset, [.3, 0., .2], [1., 0., 0., 0.], jaw)


if __name__ == '__main__':
    unittest.main()
