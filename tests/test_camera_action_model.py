from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import camera_action_model as m


def sample(value, now, captured=None):
    captured = now if captured is None else captured
    return dict(status='available', captured_at_seconds=captured,
        available_at_seconds=captured, age_seconds=now-captured, value=value)


def context(now=0., x=.3, y=0.):
    signals = {}
    values = ([.2, 1.4, -1.3, 0., 0., 0.], [0.]*6, [.02, -.02], [0., 0.],
              [.3, 0., .2], [1., 0., 0., 0.])
    for (name, _, _), value in zip(m.SENSOR_SPECS, values):
        signals[name] = sample(value, now)
    captured = round(int(round(now/.02))//2*.04, 8)
    position = [x, y, .08]
    estimate = dict(schema=m.tracker.SCHEMA, status='available', decision_at_seconds=now,
        frame_index=int(round(captured/.04)),
        captured_at_seconds=captured, available_at_seconds=captured, age_seconds=now-captured,
        object_position_estimate_m=position, object_position_uncertainty_m=.002,
        signals={'object_position_estimate_m': sample(position, now, captured),
                 'object_position_uncertainty_m': sample([.002], now, captured)})
    return dict(snapshot={'timestamp_seconds': now, 'signals': signals,
        'camera': sample({'index': int(round(captured/.04))}, now, captured),
        'vision_frame_index': int(round(captured/.04))}, estimate=estimate,
        previous_applied={'joint_targets_rad': values[0], 'gripper_position_m': .02},
        goal_position_m=[.35, .08, .08])


def missing_visual(value):
    value = deepcopy(value)
    value['estimate']['status'] = 'missing'
    for key in ('object_position_estimate_m', 'object_position_uncertainty_m'):
        value['estimate'][key] = None
        value['estimate']['signals'][key] = dict(status='missing', captured_at_seconds=None,
            available_at_seconds=None, age_seconds=None, value=None)
    return value


def dataset(blind=False, prefix='train'):
    rows, targets, groups = [], [], []
    for group in range(4):
        encoder = m.CameraActionFeatures(blind=blind)
        for index in range(16):
            c = context(index*.02, x=.28+group*.015, y=.01*group)
            c['snapshot']['signals']['joint_position_rad']['value'][0] = .1+index*.01
            rows.append(encoder(c))
            targets.append([.1+group*.03+index*.001, 1.42, -1.28, 0., 0., 0., .015+group*.001])
            groups.append(prefix+'-'+str(group))
    return dict(schema=m.DATASET_SCHEMA, features=np.asarray(rows), targets=np.asarray(targets),
        groups=groups, provenance={'test_fixture': True, 'origin': prefix})


class FeatureTests(unittest.TestCase):
    def test_dimensions_and_source_whitelist(self):
        clean = context()
        poisoned = deepcopy(clean)
        poisoned.update(reference={'action': {'jaw': 999}}, teacher_phase='release',
            current_issued=[999]*7, after={'object_position': [9]*3}, native_truth={'contact': 999})
        poisoned['snapshot'].update(reference={'action': [99]*7}, future={'time': .02})
        poisoned['snapshot']['signals']['object_position_estimate_m'] = sample([8, 8, 8], 0.)
        poisoned['estimate'].update(object_quaternion_estimate_wxyz=[0, 0, 0, 0],
                                    diagnostics={'native_truth': [99]*3})
        a = m.CameraActionFeatures()(clean)
        b = m.CameraActionFeatures()(poisoned)
        self.assertEqual(m.FEATURE_COUNT, 269)
        self.assertEqual(a.shape, (269,))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(any('reference' in name or 'object_quaternion' in name for name in m.FEATURE_NAMES))

    def test_missing_live_estimate_stays_missing_and_anchor_remains_historical(self):
        encoder = m.CameraActionFeatures()
        first = encoder(context())
        live = encoder(missing_visual(context(.02, x=.8)))
        np.testing.assert_array_equal(first[:5], live[:5])
        index = m.FEATURE_NAMES.index('lag0:visual_available')
        self.assertEqual(first[index], 1.)
        self.assertEqual(live[index], 0.)
        self.assertEqual(live[m.FEATURE_NAMES.index('lag0:visual_x_m/.5')], 0.)
        self.assertAlmostEqual(float(live[m.FEATURE_NAMES.index('lag0:visual_age_s/.1')]), .2)
        self.assertEqual(live[m.FEATURE_NAMES.index('lag40:history_present')], 0.)

    def test_camera_age_is_original_and_no_future_history(self):
        encoder = m.CameraActionFeatures()
        encoder(context())
        result = encoder(context(.02))
        self.assertAlmostEqual(float(result[m.FEATURE_NAMES.index('lag0:visual_age_s/.1')]), .2)
        encoder(context(.04)); encoder(context(.06))
        result = encoder(context(.08))
        # Duplicate names refer to command and sensor lag presence; both are causal.
        i = m.LIVE_FEATURE_START+len(m.SENSOR_NAMES)*2
        self.assertEqual(result[i+len(m.SENSOR_NAMES)-2], 1.)
        self.assertAlmostEqual(float(result[i+len(m.SENSOR_NAMES)-1]), .5)

    def test_initial_missing_or_future_or_stale_refused(self):
        with self.assertRaisesRegex(ValueError, 'initial visual anchor'):
            m.CameraActionFeatures()(missing_visual(context()))
        bad = context()
        for key in ('object_position_estimate_m', 'object_position_uncertainty_m'):
            bad['estimate']['signals'][key]['captured_at_seconds'] = .04
            bad['estimate']['signals'][key]['available_at_seconds'] = .04
        with self.assertRaisesRegex(ValueError, 'future'):
            m.CameraActionFeatures()(bad)
        bad = context(); bad['estimate']['decision_at_seconds'] = .02
        with self.assertRaisesRegex(ValueError, 'decision time'):
            m.CameraActionFeatures()(bad)

    def test_blind_live_values_masks_and_ages_cannot_change_features(self):
        a, b = m.CameraActionFeatures(blind=True), m.CameraActionFeatures(blind=True)
        np.testing.assert_array_equal(a(context()), b(context()))
        altered = missing_visual(context(.02))
        altered['snapshot']['signals'] = {'untrusted_native_truth': [100]*9}
        left, right = a(context(.02)), b(altered)
        np.testing.assert_array_equal(left, right)
        self.assertTrue(np.all(left[m.LIVE_FEATURE_START:] == 0.))
        altered = context(.04)
        altered['previous_applied']['joint_targets_rad'][0] = .3
        self.assertFalse(np.array_equal(a(context(.04)), b(altered)))

    def test_bounds_goal_and_timestamp_failures_do_not_advance_history(self):
        encoder = m.CameraActionFeatures(); encoder(context())
        bad = context(.02); bad['previous_applied']['joint_targets_rad'][1] = -1.
        with self.assertRaisesRegex(ValueError, 'native joint bounds'):
            encoder(bad)
        bad = context(.02); bad['goal_position_m'][0] = .4
        with self.assertRaisesRegex(ValueError, 'destination'):
            encoder(bad)
        with self.assertRaisesRegex(ValueError, '20 ms'):
            encoder(context(.04))
        encoder(context(.02))
        self.assertAlmostEqual(encoder.last_time, .02)

    def test_unavailable_nonnull_and_inconsistent_receipts_refused(self):
        bad = context(); bad['estimate']['signals']['object_position_estimate_m']['status'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'forward-fill'):
            m.CameraActionFeatures()(bad)
        bad = context(); bad['estimate']['object_position_estimate_m'] = [.8, 0., .08]
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            m.CameraActionFeatures()(bad)

    def test_camera_frame_binding_and_uncertainty_capture_cannot_be_rebound(self):
        bad = context(); bad['estimate']['frame_index'] = 99
        with self.assertRaisesRegex(ValueError, 'frame index'):
            m.CameraActionFeatures()(bad)
        encoder = m.CameraActionFeatures(); encoder(context())
        bad = context(.02)
        bad['estimate']['signals']['object_position_uncertainty_m']['available_at_seconds'] = .01
        with self.assertRaisesRegex(ValueError, 'signal times'):
            encoder(bad)
        bad = context(.02); bad['estimate']['captured_at_seconds'] = .01
        with self.assertRaisesRegex(ValueError, 'source timestamps'):
            encoder(bad)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = dataset()
        cls.model = m.train(cls.data, validation_dataset=dataset(prefix='development'))

    def test_learning_absolute_targets_and_recipe(self):
        model = self.model
        self.assertEqual(model['hidden_units'], [64, 64])
        self.assertEqual(model['action_representation'], 'absolute_native_targets')
        self.assertLess(model['training']['final_loss'], model['training']['initial_loss']/10.)
        self.assertLess(model['training']['validation']['normalized_mse'], .1)
        validation = model['training']['validation']
        expected = float(np.mean((np.asarray(validation['native_rmse'])/m.ACTION_SCALE_NATIVE)**2))
        self.assertAlmostEqual(validation['action_scale_mse'], expected, places=12)
        with self.assertRaisesRegex(ValueError, 'registered recipe'):
            m.train(self.data, epochs=51)

    def test_training_only_normalization_and_split_isolation(self):
        altered = dataset(prefix='different-development')
        altered['features'] = np.clip(altered['features']+2., -10., 10.)
        alternative = m.train(self.data, validation_dataset=altered)
        for key in ('weights', 'biases', 'feature_mean', 'feature_scale', 'action_mean', 'action_scale'):
            self.assertEqual(self.model[key], alternative[key])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            m.train(self.data, validation_dataset=self.data)

    def test_policy_has_no_teacher_and_does_not_read_truth(self):
        clean, poisoned = context(), context()
        poisoned['reference'] = {'action': [100]*7}
        poisoned['current_teacher_query'] = {'phase': 'release'}
        with patch('dvidia_training.piper_task.PiperTaskPolicy', side_effect=AssertionError('No teacher')):
            a, b = m.ModelPolicy(self.model), m.ModelPolicy(self.model)
            self.assertEqual(a(clean), b(poisoned))
        self.assertEqual(a.diagnostics()['teacher_fallback_calls'], 0)
        self.assertEqual(a.diagnostics()['student_inference_calls'], 1)

    def test_strict_json_roundtrip_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.json'
            saved = m.save_model(self.model, path)
            self.assertEqual(saved, m.digest(m.load_model(path)))
            self.assertEqual(m.ModelPolicy(self.model)(context()), m.ModelPolicy(m.load_model(path))(context()))
            with self.assertRaises(FileExistsError):
                m.save_model(self.model, path)
        for key, value in (('schema', 'arbitrary'), ('physical_robot_ready', True), ('jaw_limit_m', .2)):
            bad = deepcopy(self.model); bad[key] = value
            with self.assertRaises(ValueError):
                m.validate_model(bad)
        bad = deepcopy(self.model); bad['training']['provenance']['test_fixture'] = False
        with self.assertRaisesRegex(ValueError, 'provenance hash'):
            m.validate_model(bad)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            m.validate_model('{"schema":"x","schema":"x"}')

    def test_native_action_bounds_and_blind_mask_training(self):
        extreme = deepcopy(self.model)
        extreme['weights'] = [np.zeros_like(w).tolist() for w in extreme['weights']]
        extreme['biases'][-1] = [64., -64., 64., -64., 64., -64., 64.]
        action = m.ModelPolicy(extreme)(context())
        m.action_vector(action)
        blind = m.train(dataset(blind=True), blind=True)
        self.assertTrue(blind['blind'])
        with self.assertRaisesRegex(ValueError, 'zero all'):
            m.train(self.data, blind=True)

    def test_dataset_hash_and_fixed_training_are_reproducible(self):
        again = m.train(self.data, validation_dataset=dataset(prefix='development'))
        self.assertEqual(self.model, again)
        changed = deepcopy(self.data); changed['targets'][0, 0] += .01
        self.assertNotEqual(self.model['training']['dataset_sha256'], m.train(changed)['training']['dataset_sha256'])


if __name__ == '__main__':
    unittest.main()
