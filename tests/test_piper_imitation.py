"""Learned commands, causal measured inputs, and safe JSON model interchange."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_imitation as imitation


def observation(a=0., b=0.):
    return {'joint_position_rad': [0., 1.4, -1.3, 0., 0., 0.],
            'joint_velocity_rad_s': [0.] * 6,
            'gripper_joint_position_m': [.02, -.02],
            'tcp_position_m': [.3, 0., .17],
            'tcp_quaternion_wxyz': [0., 0., 1., 0.],
            'object_position_m': [.3+.03*a, .03*b, .08],
            'object_quaternion_wxyz': [1., 0., 0., 0.],
            'object_linear_velocity_m_s': [0., 0., 0.],
            'object_angular_velocity_rad_s': [0., 0., 0.],
            'target_position_m': [.35, .08, .08],
            'pad_normal_forces_n': {'left': 0., 'right': 0.},
            'grasp_contact_counts': {'left': 0, 'right': 0},
            'table_object_contact_count': 1, 't_seconds': 2.}


def supervised_action(a, b):
    # Independent nonlinear fixture, deliberately expressed in native units.
    delta = [.025*a, -.021*b, .01*a*b, 0., .009*a*a, -.012*b*b]
    q = observation()['joint_position_rad']
    return {'joint_targets_rad': [current+change for current, change in zip(q, delta)],
            'gripper_position_m': .0175+.012*a}


class LearnedPiperPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(82)
        cls.dataset = {'scene_ids': ['fixture-train'], 'samples': [
            imitation.training_sample(observation(a, b), supervised_action(a, b), 'fixture-train')
            for a, b in rng.uniform(-1., 1., (256, 2))]}
        cls.model = imitation.train(cls.dataset, hidden_units=24, epochs=260,
                                    batch_size=64, learning_rate=.004, seed=5)

    def test_supervised_actions_generalize_and_respond_to_measured_object_state(self):
        policy = imitation.LearnedPiperPolicy(self.model)
        rng = np.random.default_rng(991)
        expected, actual = [], []
        for a, b in rng.uniform(-.95, .95, (32, 2)):
            expected.append(supervised_action(a, b))
            actual.append(policy(observation(a, b)))
        np.testing.assert_allclose([r['joint_targets_rad'] for r in actual],
                                   [r['joint_targets_rad'] for r in expected], atol=.002)
        np.testing.assert_allclose([r['gripper_position_m'] for r in actual],
                                   [r['gripper_position_m'] for r in expected], atol=.001)
        left, right = policy(observation(-.8, 0.)), policy(observation(.8, 0.))
        self.assertGreater(right['joint_targets_rad'][0]-left['joint_targets_rad'][0], .03)
        self.assertGreater(right['gripper_position_m']-left['gripper_position_m'], .015)
        self.assertLess(self.model['training']['final_loss'], self.model['training']['initial_loss']/100)

    def test_changed_supervision_changes_learned_commands_without_teacher_access(self):
        changed = deepcopy(self.dataset)
        for row in changed['samples']:
            row['targets'] = [0.] * 6 + [1.]
        zero_model = imitation.train(changed, hidden_units=24, epochs=120,
                                    batch_size=64, learning_rate=.004, seed=5)
        with patch('dvidia_training.piper_task.PiperTaskPolicy', side_effect=AssertionError('Teacher unavailable')):
            learned = imitation.LearnedPiperPolicy(self.model)
            altered = imitation.LearnedPiperPolicy(zero_model)
            obs = observation(.8, -.7)
            actual, zero = learned(obs), altered(obs)
        self.assertNotEqual(self.model['dataset_sha256'], zero_model['dataset_sha256'])
        self.assertGreater(abs(actual['joint_targets_rad'][0]-zero['joint_targets_rad'][0]), .015)
        np.testing.assert_allclose(zero['joint_targets_rad'], obs['joint_position_rad'], atol=.001)
        self.assertAlmostEqual(zero['gripper_position_m'], .035, delta=.0005)
        self.assertEqual(altered.diagnostics()['teacher_fallback_calls'], 0)
        self.assertFalse(altered.diagnostics()['authored_task_phase_machine'])

    def test_outcome_and_teacher_fields_cannot_influence_features_or_actions(self):
        clean = observation(.4, -.2)
        poisoned = {**clean, 'phase': 'release', 'issued_phase': 'close',
                    'success': True, 'lift_seen': True, 'native_lift_seen': True,
                    'carry_complete': True, 'teacher_action': [1.] * 7,
                    'command': {'joint_targets_rad': [1.] * 6, 'gripper_position_m': 0.}}
        np.testing.assert_array_equal(imitation.features(clean), imitation.features(poisoned))
        policy = imitation.LearnedPiperPolicy(self.model)
        self.assertEqual(policy(clean), policy(poisoned))
        first = policy(clean)
        policy(observation(-.9, .9))
        self.assertEqual(first, policy(clean))
        batch = policy.normalized_actions([clean, observation(-.9, .9)])
        self.assertEqual(batch.shape, (2, 7))
        self.assertTrue(np.isfinite(batch).all())
        self.assertEqual(len(self.model['feature_names']), imitation.FEATURE_COUNT)
        self.assertEqual(self.model['feature_names'][-1], 'episode_elapsed_seconds/20')

    def test_json_roundtrip_preserves_exact_predictions_and_is_deterministic(self):
        self.assertEqual(imitation.validate_model(imitation.canonical(self.model)), self.model)
        obs = observation(.3, -.6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'candidate.json'
            digest = imitation.save_model(self.model, path)
            restored = imitation.load_model(path)
            self.assertEqual(digest, imitation.digest(restored))
            self.assertEqual(imitation.LearnedPiperPolicy(self.model)(obs),
                             imitation.LearnedPiperPolicy(restored)(obs))
            self.assertEqual(json.loads(path.read_text())['schema'], imitation.SCHEMA)
        dataset = {'scene_ids': ['fixture-train'], 'samples': deepcopy(self.dataset['samples'][:16])}
        one = imitation.train(dataset, hidden_units=8, epochs=4, batch_size=8, seed=41)
        for row in dataset['samples']:
            row['features'] = np.asarray(row['features'])
            row['targets'] = np.asarray(row['targets'])
        two = imitation.train(dataset, hidden_units=8, epochs=4, batch_size=8, seed=41)
        self.assertEqual(one, two)

    def test_native_action_limits_apply_even_for_saturated_network_predictions(self):
        saturated = deepcopy(self.model)
        saturated['weights'][-1] = np.zeros((24, 7)).tolist()
        saturated['biases'][-1] = [64., -64., 64., -64., 64., -64., 64.]
        policy = imitation.LearnedPiperPolicy(saturated)
        obs = observation()
        limits = np.asarray(imitation.JOINT_LIMITS)
        obs['joint_position_rad'] = (limits[:, 1]-.001).tolist()
        action = policy(obs)
        targets = np.asarray(action['joint_targets_rad'])
        self.assertTrue(np.all(targets >= limits[:, 0]+.02))
        self.assertTrue(np.all(targets <= limits[:, 1]-.02))
        self.assertTrue(np.isfinite(targets).all())
        self.assertEqual(action['gripper_position_m'], .035)
        saturated['biases'][-1][-1] = -64.
        self.assertEqual(imitation.LearnedPiperPolicy(saturated)(obs)['gripper_position_m'], 0.)

    def test_invalid_or_executable_model_payloads_are_rejected(self):
        mutations = [('scope', 'physical'), ('physical_robot_ready', True),
                     ('hidden_units', True), ('feature_names', list(reversed(self.model['feature_names']))),
                     ('joint_limits_rad', [[-10., 10.]] * 6), ('joint_delta_limit_rad', .6),
                     ('dataset_sha256', 'bad'), ('train_cases', ['same', 'same']),
                     ('feature_mean', [True] * imitation.FEATURE_COUNT),
                     ('feature_scale', [0.] * imitation.FEATURE_COUNT),
                     ('feature_mean', [float('nan')] * imitation.FEATURE_COUNT),
                     ('weights', []), ('code', 'import os')]
        for key, value in mutations:
            bad = {**self.model, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                imitation.validate_model(bad)
        for value in (float('inf'), 65., 10**1000, True):
            bad = deepcopy(self.model)
            bad['weights'][0][0][0] = value
            with self.subTest(weight=str(value)[:30]), self.assertRaises(ValueError):
                imitation.validate_model(bad)
        raw = imitation.canonical(self.model).decode()
        duplicate = raw.replace('"hidden_units":24', '"hidden_units":24,"hidden_units":24')
        for invalid in (duplicate, b'\x80\x04pickle', raw[:-3], ' '*(imitation.MAX_MODEL_BYTES+1), '\ud800'):
            with self.assertRaises(ValueError):
                imitation.validate_model(invalid)

    def test_bad_measurements_and_noncausal_dataset_fields_are_rejected(self):
        for key, value in (('joint_position_rad', [float('nan')]*6),
                           ('joint_velocity_rad_s', [True]*6), ('t_seconds', float('inf')),
                           ('tcp_quaternion_wxyz', [0., 0., 0., 0.])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                imitation.features({**observation(), key: value})
        with self.assertRaises(ValueError):
            imitation.training_sample(observation(), {'joint_targets_rad': [10.]*6,
                                                      'gripper_position_m': 0.}, 'case')
        bad = deepcopy(self.dataset)
        bad['samples'][0]['phase'] = 'close'
        with self.assertRaises(ValueError):
            imitation.train(bad, epochs=1)

    def test_v2_absolute_supervision_generalizes_native_targets_and_roundtrips(self):
        model = imitation.train(self.dataset, hidden_units=24, epochs=260, batch_size=64,
                                learning_rate=.004, seed=5, action_representation='absolute')
        self.assertEqual(model['schema'], imitation.ABSOLUTE_SCHEMA)
        self.assertEqual(model['action_representation'], 'absolute')
        self.assertEqual(model['training']['source_dataset_sha256'], self.model['dataset_sha256'])
        self.assertNotEqual(model['dataset_sha256'], self.model['dataset_sha256'])
        self.assertEqual(imitation.validate_model(imitation.canonical(model)), model)
        policy = imitation.LearnedPiperPolicy(model)
        for a, b in ((-.8, .3), (.6, -.7), (.15, .2)):
            actual, expected = policy(observation(a, b)), supervised_action(a, b)
            np.testing.assert_allclose(actual['joint_targets_rad'], expected['joint_targets_rad'], atol=.002)
            self.assertAlmostEqual(actual['gripper_position_m'], expected['gripper_position_m'], delta=.001)
        self.assertEqual(policy.diagnostics()['action_representation'], 'absolute')
        # Loading V2 must not change the old decoder or serialized V1 contract.
        self.assertEqual(self.model['schema'], imitation.SCHEMA)
        self.assertNotIn('action_mean', self.model)
        self.assertEqual(imitation.LearnedPiperPolicy(self.model).diagnostics()['action_representation'], 'delta')

    def test_v2_standardization_and_complete_absolute_label_shuffle_are_bound(self):
        ordinary = imitation.train(self.dataset, hidden_units=8, epochs=2, batch_size=128, seed=11,
                                   action_representation='absolute')
        shuffled = imitation.train(self.dataset, hidden_units=8, epochs=2, batch_size=128, seed=11,
                                   action_representation='absolute', label_shuffle_seed=19217)
        repeated = imitation.train(self.dataset, hidden_units=8, epochs=2, batch_size=128, seed=11,
                                   action_representation='absolute', label_shuffle_seed=19217)
        self.assertEqual(shuffled, repeated)
        self.assertNotEqual(ordinary['dataset_sha256'], shuffled['dataset_sha256'])
        self.assertEqual(ordinary['training']['source_dataset_sha256'], shuffled['training']['source_dataset_sha256'])
        self.assertEqual(shuffled['training']['label_shuffle_seed'], 19217)
        np.testing.assert_allclose(ordinary['action_mean'], shuffled['action_mean'], atol=1e-7)
        for field, value in (('action_mean', [float('nan')]*7), ('action_scale', [0.]*7),
                             ('action_scale', [3.]*7), ('action_representation', 'teacher')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                imitation.validate_model({**ordinary, field: value})
        with self.assertRaises(ValueError):
            imitation.train(self.dataset, epochs=1, label_shuffle_seed=19217)

    def test_v2_absolute_decoder_retains_per_step_and_native_safety_caps(self):
        model = imitation.train(self.dataset, hidden_units=8, epochs=1, batch_size=128,
                                action_representation='absolute')
        model['weights'][-1] = np.zeros((8, 7)).tolist()
        model['action_mean'] = [0.] * 6 + [1.]
        model['action_scale'] = [.01] * 7
        model['biases'][-1] = [64., -64., 64., -64., 64., -64., 0.]
        obs = observation()
        action = imitation.LearnedPiperPolicy(model)(obs)
        delta = np.asarray(action['joint_targets_rad'])-obs['joint_position_rad']
        self.assertTrue(np.all(np.abs(delta) <= .06+1e-8))
        limits = np.asarray(imitation.JOINT_LIMITS)
        self.assertTrue(np.all(np.asarray(action['joint_targets_rad']) >= limits[:, 0]+.02))
        self.assertTrue(np.all(np.asarray(action['joint_targets_rad']) <= limits[:, 1]-.02))
        self.assertEqual(action['gripper_position_m'], .035)
        bad = deepcopy(self.dataset)
        bad['scene_ids'].append('held-out-without-training-samples')
        with self.assertRaises(ValueError):
            imitation.train(bad, epochs=1)


if __name__ == '__main__':
    unittest.main()
