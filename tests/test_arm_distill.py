"""Learning, data-only validation and absence of hidden teacher fallback."""
from copy import deepcopy
import json
import unittest

import numpy as np

from dvidia_training.arm_distill import (DistilledPlacementPolicy, MovementHead, canonical,
                               features, make_protocol, train, validate_model)
from dvidia_training.arm_env import ArmConfig


def observation():
    return {'joint_position': [0.] * 6, 'end_effector_position': [.4, 0., .4],
            'end_effector_quaternion': [1., 0., 0., 0.],
            'object_position': [.42, -.04, .31], 'target_position': [.54, .1, .31],
            'object_size': [.04] * 3, 'pad_normal_forces': {'left': 0., 'right': 0.},
            'joint_actuator_torque': [0.] * 6, 'effective_sliding_friction': {'jaw_object': 1.},
            'grasp_contacts': {'left': 0, 'right': 0}, 'table_contacts': 1,
            'end_effector_velocity': [0.] * 3, 'object_velocity': [0.] * 3,
            'object_angular_velocity': [0.] * 3, 'object_quaternion': [1., 0., 0., 0.],
            'gripper_width': .08}


class MovementHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(14)
        # An independently defined supervision function; the fit must learn it.
        cls.mapping = np.array([[.008, .002, 0., 0., 0., 0.],
                                [0., .009, -.002, 0., 0., 0.],
                                [0., 0., .011, .002, 0., 0.],
                                [.001, 0., 0., .012, 0., 0.],
                                [0., 0., 0., 0., .008, .001],
                                [0., 0., 0., 0., 0., .009]])
        rows = []
        for _ in range(1024):
            context = rng.normal(0., .2, 12)
            error = rng.uniform(-1, 1, 6)
            rows.append({'context': context.tolist(), 'error': error.tolist(),
                         'delta': (error @ cls.mapping).tolist(), 'case_id': 'synthetic-train'})
        cls.dataset = {'scene_ids': ['synthetic-train'], 'samples': rows}
        cls.model = train(cls.dataset, centers=8, regularization=1e-8)

    def test_real_supervised_fit_generalizes_an_unseen_pose_error(self):
        obs = observation()
        goal = [.403, -.001, .402]
        context, error = features(obs, goal)
        prediction = MovementHead(self.model).predict(obs, goal)
        np.testing.assert_allclose(prediction, error @ self.mapping, atol=3e-6)
        zero = deepcopy(self.dataset)
        for row in zero['samples']:
            row['delta'] = [0.] * 6
        zero_model = train(zero, centers=8, regularization=1e-8)
        self.assertNotEqual(zero_model['dataset_sha256'], self.model['dataset_sha256'])
        np.testing.assert_allclose(MovementHead(zero_model).predict(obs, goal), [0.] * 6, atol=1e-10)

    def test_data_only_roundtrip_and_invalid_models(self):
        self.assertEqual(validate_model(canonical(self.model)), self.model)
        self.assertLess(len(canonical(self.model)), 128*1024)
        mutations = [('scope', 'physical'), ('physical_robot_ready', True),
                     ('kernel_width', 0.), ('centers', [[0.] * 12]),
                     ('context_scale', [0.] * 12), ('weights', []),
                     ('dataset_sha256', 'invalid'), ('train_cases', ['same', 'same'])]
        for key, value in mutations:
            bad = {**self.model, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_model(bad)
        for bad in ({**self.model, 'code': 'import os'},
                    {**self.model, 'context_mean': [True] * 12},
                    {**self.model, 'context_mean': [float('nan')] * 12}):
            with self.assertRaises(ValueError):
                validate_model(bad)
        raw = canonical(self.model).decode().replace('"schema_version":1', '"schema_version":1,"schema_version":1')
        with self.assertRaises(ValueError):
            validate_model(raw)

    def test_zero_pose_error_holds_current_joints(self):
        obs = observation()
        obs['joint_position'] = [.1, -.5, .2, .1, .1, -.1]
        np.testing.assert_allclose(MovementHead(self.model).predict(obs, obs['end_effector_position']), obs['joint_position'], atol=1e-8)

    def test_student_supervisor_does_not_call_teacher_ik(self):
        class Environment:
            config = ArmConfig(scene_jitter=0.)
            initial_object_position = np.array([.42, -.04, .31])
            _elapsed = 0.
            def joint_targets_for_pose(self, goal):
                raise AssertionError('Hidden teacher IK fallback')
        policy = DistilledPlacementPolicy(Environment(), self.model)
        action = policy(observation())
        self.assertEqual(len(action['joint_targets']), 6)
        self.assertEqual(action['gripper_width'], .08)
        self.assertEqual(policy.diagnostics()['teacher_fallback_calls'], 0)
        self.assertEqual(policy.diagnostics()['student_inference_calls'], 1)
        obs = observation()
        obs['end_effector_position'] = [.42, -.04, .46]
        policy(obs)
        self.assertEqual(policy.stage, 'descend')

    def test_new_protocol_declares_disjoint_scenes_with_valid_units(self):
        protocol = make_protocol()
        scenes = [case for split in ('development', 'selection', 'evaluation') for case in protocol[split]]
        self.assertEqual(len({c['scene_sha256'] for c in scenes}), len(scenes))
        for case in scenes:
            ArmConfig(**case['config'])
        self.assertEqual(sum(c['group']=='actuator_stress' for c in protocol['evaluation']), 2)


if __name__ == '__main__':
    unittest.main()
