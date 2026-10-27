"""Learned jaw prediction preserves every frozen six-axis arm command."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dvidia_training import piper_grip_adapter as grip
from dvidia_training import piper_imitation as core
from dvidia_training import piper_motor_primitive as primitive
from tests.test_piper_feedback_model import base_model, observation


class GripAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = base_model()
        cls.x = np.zeros((256, grip.FEATURE_COUNT))
        cls.feature = grip.LIVE_START+11
        cls.x[:, cls.feature] = np.linspace(-1., 1., 256)
        cls.y = np.where(cls.x[:, cls.feature] > 0., .035, 0.)
        cls.data = dict(features=cls.x, targets=cls.y, groups=['teacher-a']*128+['teacher-b']*128)
        cls.model = grip.train(cls.data, base_sha256=core.digest(cls.base), max_depth=4, min_leaf=2)

    def test_small_deterministic_tree_learns_grip_signal_and_binary_native_targets(self):
        again = grip.train(self.data, base_sha256=core.digest(self.base), max_depth=4, min_leaf=2)
        self.assertEqual(core.canonical(again), core.canonical(self.model))
        self.assertEqual(len(self.model['nodes']), 3)
        self.assertEqual(self.model['nodes'][0]['feature'], self.feature)
        self.assertEqual(self.model['training']['weighted_classification_error'], 0.)
        self.assertEqual(self.model['training']['jaw_mae_m'], 0.)
        self.assertEqual(grip._probability(self.model['nodes'], self.x[0]), 0.)
        self.assertEqual(grip._probability(self.model['nodes'], self.x[-1]), 1.)

    def test_encoder_excludes_joint_control_phase_future_and_unrelated_sensors(self):
        nominal = {'gripper_position_m': .035, 'joint_targets_rad': 'must never be read'}
        a, b = grip.GripFeatures(), grip.GripFeatures()
        for tick in range(8):
            clean, poison = observation(tick*.02, contacts=tick), observation(tick*.02, contacts=tick)
            poison.update(joint_position_rad='ignored', joint_velocity_rad_s='ignored',
                          native_actuator_controls='ignored', actuator_force_native='ignored',
                          tcp_quaternion_wxyz='ignored', object_quaternion_wxyz='ignored',
                          phase='release', success=True, intervention_id='fault', future_state='ignored')
            np.testing.assert_array_equal(a(clean, nominal), b(poison, nominal))
        self.assertEqual(grip.FEATURE_COUNT, 56)

    def test_history_is_causal_and_bound_to_120ms_without_future_frames(self):
        encoder = grip.GripFeatures()
        nominal = {'gripper_position_m': .035}
        rows = [encoder(observation(tick*.02, contacts=tick), nominal) for tick in range(8)]
        contact = grip.SNAPSHOT_NAMES.index('left_finger_contacts/2')
        self.assertEqual(rows[-1][grip.LIVE_START+contact], 3.5)
        self.assertEqual(rows[-1][grip.LIVE_START+16+contact], 2.5)
        self.assertEqual(rows[-1][grip.LIVE_START+32+contact], .5)
        self.assertTrue(all(t >= .14-.12-1e-8 for t, _ in encoder.history))
        self.assertEqual(rows[0][grip.LIVE_START+contact], 0.)
        with self.assertRaises(ValueError): encoder(observation(.14), nominal)
        with self.assertRaises(ValueError): encoder(observation(.12), nominal)

    def test_original_arm_targets_are_exactly_preserved_while_only_jaw_is_replaced(self):
        adapter = grip.GripPolicy(self.base, core.canonical(self.model))
        original = primitive.MotorPrimitivePolicy(self.base)
        for tick in range(20):
            obs = observation(tick*.02, contacts=tick%2)
            obs['joint_position_rad'][0] = .4 if tick%2 else 0.
            action, expected = adapter(obs), original(obs)
            self.assertEqual(action['joint_targets_rad'], expected['joint_targets_rad'])
            self.assertIn(action['gripper_position_m'], (0., .035))
        info = adapter.diagnostics()
        self.assertTrue(info['original_arm_geometry_preserved'])
        self.assertEqual(info['nominal_clock'], 'physical_elapsed_time')
        self.assertFalse(info['runtime_teacher_available']); self.assertFalse(info['authored_task_phase_machine'])
        self.assertEqual(info['teacher_fallback_calls'], 0)

    def test_blind_same_feature_family_ignores_every_live_measurement_after_reset(self):
        x = self.x.copy(); x[:, grip.LIVE_START:] = 0.
        blind = grip.train({**self.data, 'features': x}, base_sha256=core.digest(self.base),
                           blind=True, max_depth=4, min_leaf=2)
        self.assertEqual(blind['training']['train_groups'], self.model['training']['train_groups'])
        self.assertEqual(blind['training']['sample_count'], self.model['training']['sample_count'])
        self.assertEqual(blind['feature_names'], self.model['feature_names'])
        self.assertEqual(len(blind['nodes']), 1)
        a, b = grip.GripFeatures(blind=True), grip.GripFeatures(blind=True)
        nominal = {'gripper_position_m': .035}
        np.testing.assert_array_equal(a(observation(), nominal), b(observation(), nominal))
        poison = observation(.02)
        for key in list(poison):
            if key not in ('t_seconds', 'target_position_m'):
                poison[key] = 'must not be read'
        np.testing.assert_array_equal(a(observation(.02), nominal), b(poison, nominal))
        with self.assertRaises(ValueError): grip.GripPolicy(self.base, blind, blind=False)
        with self.assertRaises(ValueError): grip.train(self.data, base_sha256=core.digest(self.base), blind=True)

    def test_json_rejects_executable_fields_cycles_nonfinite_capacity_and_wrong_base(self):
        self.assertEqual(grip.validate_model(core.canonical(self.model)), self.model)
        for key, value in (('code', 'import os'), ('physical_robot_ready', True), ('nodes', []), ('base_sha256', 'bad')):
            with self.subTest(key=key), self.assertRaises(ValueError): grip.validate_model({**self.model, key: value})
        for key, value in (('left', 0), ('right', 99999), ('feature', True), ('threshold', float('nan')),
                           ('open_probability', 1.1), ('samples', 1)):
            changed = deepcopy(self.model); changed['nodes'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): grip.validate_model(changed)
        changed = deepcopy(self.model); changed['nodes'].append(deepcopy(changed['nodes'][-1]))
        with self.assertRaises(ValueError): grip.validate_model(changed)
        raw = core.canonical(self.model).decode().replace('"schema":', '"schema":"duplicate","schema":', 1)
        with self.assertRaises(ValueError): grip.validate_model(raw)
        changed = deepcopy(self.base); changed['weights'][0][0][0][0] += .01
        with self.assertRaises(ValueError): grip.GripPolicy(changed, self.model)

    def test_training_rejects_nonbinary_labels_bad_groups_weights_and_outcome_features(self):
        for changes in ({'max_depth': 11}, {'min_leaf': 1}, {'max_bins': 2}, {'blind': 'yes'}, {'sample_weights': np.zeros(256)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                grip.train(self.data, base_sha256=core.digest(self.base), **changes)
        data = {**self.data, 'targets': self.y.copy()}; data['targets'][0] = .0175
        with self.assertRaises(ValueError): grip.train(data, base_sha256=core.digest(self.base))
        with self.assertRaises(ValueError): grip.train({**self.data, 'success': True}, base_sha256=core.digest(self.base))

    def test_parent_reader_uses_exact64_teacher_prefixes_and_excludes_other_folders(self):
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            (parent/'base-model.json').write_bytes(core.canonical(self.base))
            cases = []
            for index in range(64):
                goal = index%2
                case = dict(id=f'parent-train-{index:02}', seed=index, pulses=[], goal_xy=list(primitive.GOALS[goal]))
                cases.append(case)
                states = [observation(tick*.02, contacts=index%2) for tick in range(3)]
                for state in states: state['target_position_m'] = [*primitive.GOALS[goal], .08]
                actions = [dict(expert_action={'joint_targets_rad': [0.]*6, 'gripper_position_m': .035 if index%2 else 0.},
                                expert_phase='close') for _ in range(2)]
                episode = dict(case_id=case['id'], seed=index, pulses=[], policy_kind='teacher',
                               base_model_sha256=core.digest(self.base), trajectory=states, actions=actions)
                path = parent/'demonstrations'/case['id']/'episode.json.gz'; path.parent.mkdir(parents=True)
                path.write_bytes(gzip.compress(core.canonical(episode), mtime=0))
            (parent/'protocol.json').write_bytes(core.canonical({'cases': {'train': cases}}))
            # Invalid content in excluded directories must never be opened.
            (parent/'dagger').mkdir(); (parent/'dagger'/'invalid').write_bytes(b'not training input')
            (parent/'confirmation').mkdir(); (parent/'confirmation'/'invalid').write_bytes(b'not training input')
            model = grip.train_parent_data(parent, max_depth=3, min_leaf=2)
            self.assertEqual(model['training']['sample_count'], 128)
            self.assertEqual(model['training']['scene_count'], 64)
            self.assertEqual(len(model['training']['parent_artifacts']['episodes']), 64)
            self.assertEqual(model['training']['train_groups'], [case['id'] for case in cases])
            first = parent/'demonstrations'/cases[0]['id']/'episode.json.gz'
            episode = json.loads(gzip.decompress(first.read_bytes())); episode['policy_kind'] = 'mixture'
            first.write_bytes(gzip.compress(core.canonical(episode), mtime=0))
            with self.assertRaises(ValueError): grip.train_parent_data(parent, max_depth=3, min_leaf=2)


if __name__ == '__main__':
    unittest.main()
