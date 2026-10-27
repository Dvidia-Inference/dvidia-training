"""Conditional learned references, strict data loading and causal initialization."""
from copy import deepcopy
from hashlib import sha256
import json
import unittest

import numpy as np

from dvidia_training import piper_imitation as core
from dvidia_training import piper_motor_primitive as primitive


BASE_Q = np.asarray([0., 1.4, -1.3, 0., 0., 0.])


def native_target(x, y, goal, seconds):
    # Independent fixture: smooth nonlinear time motion and spatial interactions.
    x, y = (x-.299)/.015, (y+.03)/.015
    delta = np.asarray([.012*x+.004*np.sin(seconds/2), .010*y,
                        .006*x*y, .008*np.cos(seconds/3), .006*x*x,
                        .004*y*y+.007*goal])
    return np.concatenate((BASE_Q+delta, [.0175+.003*x-.002*y+.004*goal+.001*np.sin(seconds/3)]))


def observation(x=.299, y=-.03, goal=0, seconds=0.):
    return {'t_seconds': seconds, 'joint_position_rad': BASE_Q.tolist(),
            'object_position_m': [x, y, .08],
            'target_position_m': [*primitive.GOALS[goal], .08]}


def dataset():
    samples, cases = [], []
    tcp = np.asarray([.3, 0., .17])
    for goal in range(2):
        for i, x in enumerate((.284, .294, .304, .314)):
            for j, y in enumerate((-.045, -.035, -.025, -.015)):
                case = f'fixture-g{goal}-x{i}-y{j}'; cases.append(case)
                for tick in range(1000):
                    seconds = tick*.02
                    feature = np.zeros(43)
                    feature[:6] = BASE_Q/3.
                    feature[14:17] = tcp/.5
                    feature[21:24] = (np.asarray([x, y, .08])-tcp)/.2
                    feature[24:27] = (np.asarray([*primitive.GOALS[goal], .08])-tcp)/.2
                    feature[-1] = tick/1000.
                    target = native_target(x, y, goal, seconds)
                    labels = np.concatenate(((target[:6]-BASE_Q)/.06, [target[6]/.035]))
                    samples.append({'features': feature.tolist(), 'targets': labels.tolist(), 'case_id': case})
    return {'scene_ids': cases, 'samples': samples}


class MotorPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = dataset()
        cls.model = primitive.train(cls.dataset)
        cls.shuffled = primitive.train(cls.dataset, shuffled=True)

    def test_jointly_fitted_weights_generalize_an_unseen_start_and_known_goals(self):
        for goal in range(2):
            policy = primitive.MotorPrimitivePolicy(self.model)
            for tick in range(51):
                obs = observation(.291, -.022, goal, tick*.02)
                action = policy(obs)
                expected = native_target(.291, -.022, goal, tick*.02)
                np.testing.assert_allclose(action['joint_targets_rad'], expected[:6], atol=.0001)
                self.assertAlmostEqual(action['gripper_position_m'], expected[6], delta=.00002)
        zero, one = primitive.MotorPrimitivePolicy(self.model), primitive.MotorPrimitivePolicy(self.model)
        a, b = zero(observation()), one(observation(goal=1))
        self.assertGreater(b['joint_targets_rad'][5]-a['joint_targets_rad'][5], .006)
        self.assertGreater(b['gripper_position_m']-a['gripper_position_m'], .0035)

    def test_only_initial_object_goal_clock_and_generic_joint_cap_affect_commands(self):
        clean, poisoned = primitive.MotorPrimitivePolicy(self.model), primitive.MotorPrimitivePolicy(self.model)
        self.assertEqual(clean(observation()), poisoned(observation()))
        for tick in range(1, 6):
            a = observation(seconds=tick*.02)
            b = {**a, 'object_position_m': [999., -999., 999.],
                 'grasp_contact_counts': {'left': 999, 'right': 999},
                 'pad_normal_forces_n': {'left': float('nan')},
                 'phase': 'release', 'success': True, 'lift_seen': True,
                 'command': {'joint_targets_rad': [999.]*6}}
            self.assertEqual(clean(a), poisoned(b))
        self.assertFalse(clean.diagnostics()['reactive_contact_feedback'])
        self.assertEqual(clean.diagnostics()['teacher_fallback_calls'], 0)

    def test_initial_time_cadence_goal_and_workspace_cannot_change_silently(self):
        for obs in (observation(seconds=.02), observation(x=.28), observation(y=-.01),
                    {**observation(), 'target_position_m': [.36, .08, .08]}):
            with self.subTest(obs=obs), self.assertRaises(ValueError):
                primitive.MotorPrimitivePolicy(self.model)(obs)
        policy = primitive.MotorPrimitivePolicy(self.model); policy(observation())
        for obs in (observation(seconds=0.), observation(seconds=.04), observation(seconds=.02, goal=1)):
            with self.assertRaises(ValueError): policy(obs)
        policy(observation(seconds=.02))

    def test_full_label_shuffle_retains_source_budget_and_changes_supervised_fit(self):
        self.assertEqual(self.model['source_dataset_sha256'], self.shuffled['source_dataset_sha256'])
        self.assertNotEqual(self.model['dataset_sha256'], self.shuffled['dataset_sha256'])
        self.assertNotEqual(self.model['supervision_sha256'], self.shuffled['supervision_sha256'])
        self.assertEqual(self.shuffled['training']['label_shuffle_seed'], 19217)
        self.assertEqual(self.model['training']['sample_count'], 32000)
        self.assertEqual(self.shuffled['training']['sample_count'], 32000)
        clean, shuffled = primitive.MotorPrimitivePolicy(self.model), primitive.MotorPrimitivePolicy(self.shuffled)
        a, b = clean(observation(.284, -.045)), shuffled(observation(.284, -.045))
        self.assertGreater(abs(a['joint_targets_rad'][0]-b['joint_targets_rad'][0]), .004)
        self.assertLess(self.model['training']['normalized_mse'], self.shuffled['training']['normalized_mse']/100)

    def test_core_dispatch_roundtrip_keeps_data_only_and_preserves_safety_limits(self):
        raw = core.canonical(self.model)
        self.assertLess(len(raw), core.MAX_MODEL_BYTES)
        self.assertEqual(core.validate_model(raw), self.model)
        self.assertEqual(core.LearnedPiperPolicy(raw)(observation()), primitive.MotorPrimitivePolicy(raw)(observation()))
        with self.assertRaises(ValueError): core.LearnedPiperPolicy(raw).normalized_actions([observation()])
        saturated = deepcopy(self.model)
        saturated['weights'] = np.full((2, 9, 201, 7), 64.).tolist()
        action = primitive.MotorPrimitivePolicy(saturated)(observation())
        q = np.asarray(action['joint_targets_rad'])
        self.assertTrue(np.all(abs(q-BASE_Q) <= .06+1e-8))
        self.assertTrue(np.isfinite(q).all())
        self.assertEqual(action['gripper_position_m'], .035)
        self.assertEqual(sha256(raw).hexdigest(), core.digest(self.model))

    def test_mutated_bases_nonfinite_weights_executable_fields_and_bad_groups_fail(self):
        for key, value in (('reactive_contact_feedback', True), ('time_width_s', .1),
                           ('time_centers_s', [0.]*201), ('spatial_terms', ['phase']),
                           ('weights', []), ('code', 'import os')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                primitive.validate_model({**self.model, key: value})
        for weight in (float('nan'), float('inf'), 65., True):
            bad = deepcopy(self.model); bad['weights'][0][0][0][0] = weight
            with self.assertRaises(ValueError): primitive.validate_model(bad)
        bad = {'scene_ids': self.dataset['scene_ids'], 'samples': list(self.dataset['samples'])}
        bad['samples'][1] = bad['samples'][0]
        with self.assertRaises(ValueError): primitive.train(bad)
        raw = core.canonical(self.model).decode().replace('"schema":', '"schema":"duplicate","schema":', 1)
        with self.assertRaises(ValueError): primitive.validate_model(raw)

    def test_one_high_resolution_profile_fits_cubic_fast_motion_and_dispatches(self):
        changed = {'scene_ids': self.dataset['scene_ids'], 'samples': []}
        for row in self.dataset['samples']:
            feature = row['features']
            obj_x = feature[14]*.5+feature[21]*.2
            obj_y = feature[15]*.5+feature[22]*.2
            x, y, t = (obj_x-.299)/.015, (obj_y+.03)/.015, feature[-1]*20.
            target = list(row['targets'])
            target[0] += .008*x**3*y**3*np.sin(8*np.pi*t)/.06
            changed['samples'].append({**row, 'targets': target})
        coarse = primitive.train(changed)
        fine = primitive.train(changed, high_resolution=True)
        self.assertEqual(fine['schema'], primitive.HIGH_RESOLUTION_SCHEMA)
        self.assertEqual(len(fine['spatial_terms']), 16)
        self.assertEqual(len(fine['time_centers_s']), 401)
        self.assertEqual(fine['time_width_s'], .025)
        self.assertEqual(fine['training']['spatial_ridge'], 1e-7)
        self.assertLess(fine['training']['normalized_mse'], coarse['training']['normalized_mse']/10)
        raw = core.canonical(fine)
        self.assertLess(len(raw), core.MAX_MODEL_BYTES)
        self.assertEqual(core.validate_model(raw), fine)
        self.assertEqual(core.LearnedPiperPolicy(raw)(observation()), primitive.MotorPrimitivePolicy(fine)(observation()))
        bad = {**fine, 'time_width_s': .075}
        with self.assertRaises(ValueError): primitive.validate_model(bad)
        with self.assertRaises(ValueError): primitive.train(self.dataset, high_resolution='search')


if __name__ == '__main__':
    unittest.main()
