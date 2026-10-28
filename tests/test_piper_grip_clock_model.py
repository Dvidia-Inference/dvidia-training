"""Learned progress decisions preserve causal sensing and strict data contracts."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dvidia_training import piper_grip_adapter as grip
from dvidia_training import piper_grip_clock_model as clock
from dvidia_training import piper_imitation as core
from tests.test_piper_feedback_model import observation


class GripClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_hash, cls.jaw_hash = 'a'*64, 'b'*64
        cls.signal = grip.LIVE_START+grip.SNAPSHOT_NAMES.index('left_finger_contacts/2')
        cls.x = np.zeros((256, clock.FEATURE_COUNT))
        cls.x[:, cls.signal] = np.tile(np.arange(128) % 2, 2)
        elapsed = np.tile(np.arange(128)*clock.INTERVAL_S/clock.HORIZON_S, 2)
        cls.x[:, clock.ELAPSED_INDEX] = elapsed
        cls.x[:, -2] = elapsed
        cls.y = np.where(cls.x[:, cls.signal] > 0., clock.INTERVAL_S, 0.)
        cls.groups = ['teacher-a']*128+['teacher-b']*128
        cls.data = dict(features=cls.x, targets=cls.y, groups=cls.groups)
        cls.model = clock.train(cls.data, base_sha256=cls.base_hash, jaw_sha256=cls.jaw_hash,
                                max_depth=4, min_leaf=2,
                                parent_artifacts={'protocol_sha256': 'c'*64})

    def test_small_tree_learns_signal_with_deterministic_reloadable_native_decisions(self):
        again = clock.train(self.data, base_sha256=self.base_hash, jaw_sha256=self.jaw_hash,
                            max_depth=4, min_leaf=2, parent_artifacts={'protocol_sha256': 'c'*64})
        self.assertEqual(core.canonical(again), core.canonical(self.model))
        self.assertEqual(len(self.model['nodes']), 3)
        self.assertEqual(self.model['nodes'][0]['feature'], self.signal)
        self.assertEqual(self.model['training']['classification_error'], 0.)
        self.assertEqual(self.model['training']['progress_mae_s'], 0.)
        self.assertEqual(clock.predict(self.model, self.x[0]), {'advance': False, 'advance_probability': 0.})
        self.assertEqual(clock.predict(self.model, self.x[1]), {'advance': True, 'advance_probability': 1.})
        with TemporaryDirectory() as directory:
            path = Path(directory)/'clock.json'
            path.write_bytes(core.canonical(self.model))
            loaded = clock.validate_model(path.read_bytes())
            self.assertEqual(clock.predict(loaded, self.x[1]), clock.predict(self.model, self.x[1]))
            self.assertEqual(json.loads(path.read_bytes())['physical_robot_ready'], False)

    def test_coordinated_reference_mode_versions_all_seven_clock_contract_without_changing_features(self):
        model = clock.train(self.data, base_sha256=self.base_hash, jaw_sha256=self.jaw_hash,
                            jaw_mode='coordinated_reference', min_leaf=2)
        self.assertEqual(clock.SCHEMA, 'dvidia.piper-grip-progress-clock/v1')
        self.assertEqual(model['schema'], clock.COORDINATED_SCHEMA)
        self.assertEqual(clock.jaw_mode(model), 'coordinated_reference')
        self.assertEqual(clock.jaw_mode(self.model), 'physical_adapter')
        self.assertEqual(model['scope'], clock.COORDINATED_SCOPE)
        self.assertEqual(model['jaw_command_contract'], clock.COORDINATED_JAW_CONTRACT)
        self.assertEqual(model['feature_names'], self.model['feature_names'])
        self.assertEqual(model['nodes'], self.model['nodes'])
        self.assertEqual(model['training']['dataset_sha256'], self.model['training']['dataset_sha256'])
        loaded = clock.validate_model(core.canonical(model), base_sha256=self.base_hash, jaw_sha256=self.jaw_hash)
        self.assertEqual(clock.predict(loaded, self.x[1]), clock.predict(self.model, self.x[1]))
        # Version changes alone must not silently reinterpret a different jaw
        # architecture. The exact scope and contract change together.
        for changes in ({'schema': clock.SCHEMA}, {'scope': clock.SCOPE},
                        {'jaw_command_contract': clock.JAW_CONTRACT}, {'jaw_mode': 'physical_adapter'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                clock.validate_model({**model, **changes})
        for kwargs in ({'base_sha256': 'd'*64}, {'jaw_sha256': 'd'*64}, {'jaw_sha256': 'bad'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                clock.validate_model(model, **kwargs)

    def test_runtime_rejects_clock_version_or_dependency_confusion_for_reference_jaw_manifest(self):
        # Constructor validation only: no stepping, inference or physics.
        from dvidia_training import piper_grip_clock as runtime
        from tests.test_piper_feedback_model import base_model
        base = base_model()
        jaw = runtime.reference_jaw_manifest(base)
        model = clock.train(self.data, base_sha256=core.digest(base), jaw_sha256=core.digest(jaw),
                            jaw_mode='coordinated_reference', min_leaf=2)
        policy = runtime.LearnedClock(base, jaw, model)
        self.assertEqual(policy.jaw_mode, 'coordinated_reference')
        for name in ('base_sha256', 'jaw_sha256'):
            changed = {**model, name: 'd'*64}
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.LearnedClock(base, jaw, changed)
        a_model = clock.train(self.data, base_sha256=core.digest(base), jaw_sha256=core.digest(jaw),
                              min_leaf=2)
        with self.assertRaises(ValueError):
            runtime.LearnedClock(base, jaw, a_model)
        bad_manifest = {**jaw, 'base_sha256': 'd'*64}
        with self.assertRaises(ValueError):
            runtime.LearnedClock(base, bad_manifest, model)

    def test_clock_encoder_is_exact_frozen_feature_prefix_plus_causal_progress_and_debt(self):
        actual, original = clock.ClockFeatures(), grip.GripFeatures()
        nominal = {'gripper_position_m': .035, 'joint_targets_rad': 'must not read'}
        progress = 0.
        for tick in range(8):
            if tick and tick % 2:
                progress += .02
            obs = observation(tick*.02, contacts=tick)
            encoded = actual(obs, nominal, progress)
            np.testing.assert_array_equal(encoded[:grip.FEATURE_COUNT], original(obs, nominal))
            np.testing.assert_allclose(encoded[-2:], [progress/20., (tick*.02-progress)/20.])
        self.assertEqual(clock.FEATURE_COUNT, 58)
        self.assertEqual(clock.FEATURE_NAMES[:56], grip.FEATURE_NAMES)
        self.assertAlmostEqual(encoded[clock.ELAPSED_INDEX], encoded[-2]+encoded[-1])
        self.assertTrue(all(t >= .14-.12-1e-8 for t, _ in actual.grip.history))

    def test_history_uses_past_observations_not_future_phase_labels_or_actuator_targets(self):
        a, b = clock.ClockFeatures(), clock.ClockFeatures()
        nominal = {'gripper_position_m': .035, 'joint_targets_rad': 'ignored'}
        rows = []
        for tick in range(8):
            clean, poison = observation(tick*.02, contacts=tick), observation(tick*.02, contacts=tick)
            poison.update(phase='release', success=True, fault_id='secret', intervention_id='pulse',
                          future_state=observation(20.), expert_progress_s=19., expert_action='ignored',
                          joint_position_rad='ignored', joint_velocity_rad_s='ignored',
                          native_actuator_controls='ignored', actuator_force_native='ignored')
            left, right = a(clean, nominal, tick*.02), b(poison, nominal, tick*.02)
            np.testing.assert_array_equal(left, right)
            rows.append(left)
        self.assertEqual(rows[-1][self.signal], 3.5)
        self.assertEqual(rows[-1][self.signal+16], 2.5)
        self.assertEqual(rows[-1][self.signal+32], .5)
        self.assertEqual(rows[0][self.signal], 0.)
        with self.assertRaises(ValueError):
            a(observation(.12), nominal, .12)

    def test_nonmonotonic_or_future_progress_and_skipped_ticks_do_not_consume_history(self):
        nominal = {'gripper_position_m': .035}
        for first_t, first_p in ((.02, 0.), (0., .02), (0., True), (0., float('nan'))):
            with self.subTest(first_t=first_t, first_p=first_p), self.assertRaises(ValueError):
                clock.ClockFeatures()(observation(first_t), nominal, first_p)
        encoded = clock.ClockFeatures()
        encoded(observation(), nominal, 0.)
        encoded(observation(.02), nominal, .02)
        before = (encoded.last_time, encoded.last_progress_s, len(encoded.grip.history))
        for t, p in ((.04, 0.), (.04, .03), (.04, .06), (.02, .02), (.06, .04), (.04, -1.)):
            with self.subTest(t=t, p=p), self.assertRaises(ValueError):
                encoded(observation(t), nominal, p)
            self.assertEqual((encoded.last_time, encoded.last_progress_s, len(encoded.grip.history)), before)
        encoded(observation(.04), nominal, .02)
        # Rejecting a missing measurement also leaves both encoders retryable.
        bad = observation(.06); del bad['pad_normal_forces_n']
        with self.assertRaises(ValueError):
            encoded(bad, nominal, .04)
        encoded(observation(.06), nominal, .04)

    def test_blind_masks_live_history_but_keeps_context_progress_and_clock_debt(self):
        a, b = clock.ClockFeatures(blind=True), clock.ClockFeatures(blind=True)
        nominal = {'gripper_position_m': .035}
        np.testing.assert_array_equal(a(observation(), nominal, 0.), b(observation(), nominal, 0.))
        for tick in range(1, 8):
            clean, poison = observation(tick*.02), observation(tick*.02)
            for key in poison:
                if key not in ('t_seconds', 'target_position_m'):
                    poison[key] = 'must not be read'
            left, right = a(clean, nominal, 0.), b(poison, nominal, 0.)
            np.testing.assert_array_equal(left, right)
            self.assertTrue(np.all(left[clock.LIVE_START:clock.LIVE_STOP] == 0.))
            self.assertGreater(left[-1], 0.)
            self.assertEqual(left[-2], 0.)
        x = self.x.copy(); x[:, clock.LIVE_START:clock.LIVE_STOP] = 0.
        blind = clock.train({**self.data, 'features': x}, base_sha256=self.base_hash,
                            jaw_sha256=self.jaw_hash, blind=True, min_leaf=2)
        self.assertEqual(blind['feature_names'], self.model['feature_names'])
        self.assertEqual(blind['training']['train_groups'], self.model['training']['train_groups'])
        with self.assertRaises(ValueError):
            clock.train(self.data, base_sha256=self.base_hash, jaw_sha256=self.jaw_hash, blind=True)
        with self.assertRaises(ValueError):
            clock.predict(blind, self.x[1])
        changed = deepcopy(self.model); changed['blind'] = True
        with self.assertRaises(ValueError):
            clock.validate_model(changed)
        # The final two clock features remain usable by an otherwise blind tree.
        y = np.where(x[:, -2] > .064, .02, 0.)
        clock_only = clock.train({**self.data, 'features': x, 'targets': y}, base_sha256=self.base_hash,
                                 jaw_sha256=self.jaw_hash, blind=True, min_leaf=2)
        changed = deepcopy(clock_only)
        changed['nodes'][0]['feature'] = clock.FEATURE_COUNT-2
        clock.validate_model(changed)

    def test_group_dataset_and_sample_receipts_bind_training_inputs_and_weights(self):
        receipt = self.model['training']
        self.assertEqual(receipt['groups_sha256'], core.digest({'groups': self.groups}))
        self.assertEqual(receipt['dataset_sha256'], clock._dataset_hash(self.x, self.y, self.groups, np.ones(256)))
        self.assertEqual(receipt['group_samples'], [{'case_id': 'teacher-a', 'samples': 128},
                                                  {'case_id': 'teacher-b', 'samples': 128}])
        weighted = clock.train(self.data, base_sha256=self.base_hash, jaw_sha256=self.jaw_hash,
                               min_leaf=2, sample_weights=np.full(256, 2.))
        self.assertNotEqual(weighted['training']['dataset_sha256'], receipt['dataset_sha256'])
        self.assertEqual(weighted['training']['groups_sha256'], receipt['groups_sha256'])
        self.assertEqual(weighted['training']['weight_sum'], 512.)
        for changes in ({'dataset_sha256': 'bad'}, {'groups_sha256': 'bad'}, {'scene_count': True},
                        {'group_samples': [{'case_id': 'teacher-a', 'samples': 127},
                                           {'case_id': 'teacher-b', 'samples': 128}]},
                        {'progress_mae_s': .01}, {'parent_artifacts': {'local_path': '/tmp/teacher'}}):
            changed = deepcopy(self.model); changed['training'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                clock.validate_model(changed)

    def test_weighted_probability_ties_hold_without_runtime_teacher(self):
        x = np.zeros((2, clock.FEATURE_COUNT))
        data = dict(features=x, targets=[0., .02], groups=['a', 'b'])
        tied = clock.train(data, base_sha256=self.base_hash, jaw_sha256=self.jaw_hash)
        self.assertEqual(clock.predict(tied, x[0]), {'advance': False, 'advance_probability': .5})
        weighted = clock.train(data, base_sha256=self.base_hash, jaw_sha256=self.jaw_hash,
                               sample_weights=[1., 3.])
        self.assertEqual(clock.predict(weighted, x[0]), {'advance': True, 'advance_probability': .75})
        positive = clock.train({**self.data, 'targets': np.full(256, .02)},
                               base_sha256=self.base_hash, jaw_sha256=self.jaw_hash,
                               sample_weights=np.linspace(.001, .999, 256))
        self.assertTrue(clock.predict(positive, self.x[0])['advance'])

    def test_model_rejects_scope_drift_executable_fields_nonfinite_cycles_and_shared_branches(self):
        self.assertEqual(clock.validate_model(core.canonical(self.model)), self.model)
        for key, value in (('code', 'import os'), ('schema', 'future/v2'), ('scope', 'hardware ready'),
                           ('physical_robot_ready', True), ('base_sha256', 'bad'), ('jaw_sha256', 'bad'),
                           ('control_interval_s', .01), ('feature_names', self.model['feature_names'][::-1]),
                           ('nodes', []), ('progress_increment_classes_s', [False, .02])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                clock.validate_model({**self.model, key: value})
        for key, value in (('left', 0), ('right', 1), ('right', 999), ('feature', True),
                           ('threshold', float('nan')), ('advance_probability', 1.1),
                           ('samples', 1), ('weight_sum', 1.), ('advance_probability', .1)):
            changed = deepcopy(self.model); changed['nodes'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                clock.validate_model(changed)
        changed = deepcopy(self.model); changed['nodes'].append(deepcopy(changed['nodes'][-1]))
        with self.assertRaises(ValueError):
            clock.validate_model(changed)
        changed = deepcopy(self.model); changed['nodes'] *= 11
        with self.assertRaisesRegex(ValueError, 'capacity'):
            clock.validate_model(changed)
        raw = core.canonical(self.model).decode().replace('"schema":', '"schema":"duplicate","schema":', 1)
        with self.assertRaises(ValueError):
            clock.validate_model(raw)
        changed = deepcopy(self.model); changed['nodes'][1]['threshold'] = 1.
        with self.assertRaises(ValueError):
            clock.validate_model(changed)
        changed = deepcopy(self.model); changed['training']['max_depth'] = 1
        with self.assertRaises(ValueError):
            clock.validate_model(changed)

    def test_training_rejects_invalid_labels_causal_clocks_extra_outcomes_and_bad_hyperparameters(self):
        args = dict(base_sha256=self.base_hash, jaw_sha256=self.jaw_hash)
        for changes in ({'max_depth': 3}, {'max_depth': True}, {'min_leaf': 1}, {'max_bins': 128},
                        {'blind': 'yes'}, {'sample_weights': np.zeros(256)}, {'sample_weights': np.full(256, np.nan)},
                        {'parent_artifacts': {'protocol_sha256': 'bad'}}, {'jaw_sha256': 'bad'},
                        {'jaw_mode': 'new_jaw_policy'}, {'jaw_mode': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                clock.train(self.data, **(args | changes))
        for replacement in (.01, -.02, float('inf'), True):
            y = self.y.tolist(); y[0] = replacement
            with self.subTest(label=replacement), self.assertRaises(ValueError):
                clock.train({**self.data, 'targets': y}, **args)
        with self.assertRaises(ValueError):
            clock.train({**self.data, 'success': True}, **args)
        for feature, replacement in ((-2, 1.), (-1, -1.), (clock.ELAPSED_INDEX, 1.1), (0, np.nan)):
            x = self.x.copy(); x[0, feature] = replacement
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                clock.train({**self.data, 'features': x}, **args)
        x = np.zeros(clock.FEATURE_COUNT); x[-2] = .01
        with self.assertRaises(ValueError):
            clock.predict(self.model, x)


if __name__ == '__main__':
    unittest.main()
