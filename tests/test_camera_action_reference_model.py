"""Synthetic model fits only; no original recordings, trained models or physics."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import camera_action_model as base
from dvidia_training import camera_action_reference_model as m
from tests.test_camera_action_model import context, missing_visual
from tests import test_camera_action_reference_export as fixtures

action, native_trace, write = fixtures.action, fixtures.native_trace, fixtures.write


def reset_context(now=0., previous=None):
    c = context(now)
    c['previous_applied'] = action(m.R0 if previous is None else previous)
    return c


def set_reset(recording):
    previous = np.asarray(m.R0, dtype=np.float64)
    recording['initial_applied_targets'] = action(previous)
    for step in recording['steps']:
        previous = native_trace(previous, base.action_vector(step['command']['issued']))[-1]
        step['applied']['ramped_targets'] = action(previous)


class ReferenceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These are authored six-row fixtures, never copies of study recordings.
        cls.fixtures = fixtures.ReferenceExportTests(); cls.fixtures.setUp()
        cls.demo = cls.fixtures.parent(mutation=set_reset)
        cls.mixed = cls.fixtures.parent(mixed=True, mutation=set_reset)
        # Both stages must bind one protocol. Rebuild only synthetic parent caches.
        protocol = json.loads((cls.demo['root']/'protocol.json').read_bytes())
        protocol['rosters'].update(json.loads((cls.mixed['root']/'protocol.json').read_bytes())['rosters'])
        cls.sources = []
        for f, stage in ((cls.demo, 'demonstrations'), (cls.mixed, 'corrections')):
            ph = write(f['root']/'protocol.json', protocol)
            write(f['root']/'registration.json', dict(schema=m.ref.cache.study.SCHEMA+'/registration',
                protocol_sha256=ph, source_sha256=protocol['source_sha256']))
            stage_file = f['root']/f'stages/{stage}.json'
            doc = json.loads(stage_file.read_bytes()); doc['protocol_sha256'] = ph
            sh = write(stage_file, doc)
            write(f['root']/f'stages/{stage}.integrity.json', dict(protocol_sha256=ph, sha256=sh))
            f['parent'] = f['parent'].with_name(f['parent'].name+'-paired')
            with patch('dvidia_training.camera_action_capture.load_rollout', side_effect=lambda p, records=f['recordings']: records[str(p)]):
                pins = m.ref.cache.build_cache(f['root'], f['parent'], expected_protocol_sha256=ph, stage=stage)
            f['args'].update(expected_parent_manifest_sha256=pins['manifest_sha256'], expected_protocol_sha256=ph)
            output = cls.fixtures.root/(stage+'-references')
            directory, seal = cls.fixtures.build(f, output=output)
            cls.sources.append(dict(directory=directory, parent_cache=f['parent'],
                pins=dict(f['args'], expected_manifest_sha256=seal['manifest_sha256'])))
        cls.models = {variant: m.train(cls.sources, variant=variant) for variant in m.VARIANTS}

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.tearDown(); cls.fixtures.doCleanups()

    def test_fixed_synthetic_fits_share_data_initialization_and_normalization(self):
        a, d = (self.models[v] for v in m.VARIANTS)
        for field in ('reference_dataset_sha256', 'parent_dataset_sha256', 'parent_provenance_sha256', 'initial_parameters_sha256'):
            self.assertEqual(a['training'][field], d['training'][field])
        self.assertEqual(a['feature_mean'], d['feature_mean']); self.assertEqual(a['feature_scale'], d['feature_scale'])
        for model in (a, d):
            self.assertEqual(model['training']['samples'], 12)
            self.assertEqual(model['training']['optimizer_updates'], 50)
            self.assertEqual(model['recipe'], m.RECIPE)
            self.assertLess(model['training']['final_loss'], model['training']['initial_loss'])
            self.assertFalse(model['physical_robot_ready']); self.assertFalse(model['task_qualified'])
            self.assertIsNone(model['scientific_registration'])
        # Mixed row's executed joint0 reference decreases, its query label increases.
        x, r, y, *_ = m._load_training(self.sources)
        self.assertLess(r[7, 0], r[6, 0]); self.assertGreater(y[6, 0], r[6, 0])

    def test_full_network_gradients_and_one_adam_update_match_at_reset(self):
        rng = np.random.default_rng(4401)
        x = rng.normal(0., .3, (5, base.FEATURE_COUNT)).astype(np.float32)
        r = np.tile(np.asarray(m.R0, dtype='<f8'), (len(x), 1))
        y = m.project(r+rng.uniform(-1., 1., r.shape)*m.D, r)
        results = []
        for variant in m.VARIANTS:
            w, b, _ = m._parameters()
            loss, gradients = m._loss_gradients(x, r, y, w, b, variant)
            p = w+b; first = [np.zeros_like(v) for v in p]; second = [v.copy() for v in first]
            m._adam_update(p, gradients, first, second, 1)
            results.append((loss, gradients, p, first, second))
        self.assertEqual(results[0][0], results[1][0])
        for left, right in zip(results[0][1:], results[1][1:]):
            for a, b in zip(left, right): np.testing.assert_array_equal(a, b)

    def test_backprop_directional_finite_differences_cover_all_layers(self):
        rng = np.random.default_rng(4402)
        x = rng.normal(0., .2, (4, base.FEATURE_COUNT))
        r = np.tile(np.asarray(m.R0, dtype='<f8'), (len(x), 1)); r[:, 0] = .15
        y = m.project(r+.4*m.D, r)
        w, b, _ = m._parameters()
        # A smooth float64 oracle checks the analytic chain rule; production
        # float32 parameters/updates are separately compared bit-for-bit above.
        parameters = [p.astype(np.float64) for p in w+b]
        for variant in m.VARIANTS:
            _, gradients = m._loss_gradients(x, r, y, parameters[:3], parameters[3:], variant)
            for layer, p in enumerate(parameters):
                direction = rng.normal(size=p.shape); direction /= np.linalg.norm(direction)
                step = 1e-5
                plus, minus = [a.copy() for a in parameters], [a.copy() for a in parameters]
                plus[layer] += step*direction; minus[layer] -= step*direction
                finite = (m._loss(x, r, y, plus[:3], plus[3:], variant)-m._loss(x, r, y, minus[:3], minus[3:], variant))/(2*step)
                analytic = float(np.sum(gradients[layer]*direction))
                self.assertAlmostEqual(finite, analytic, delta=1e-9)

    def test_later_reference_difference_is_intended_and_preprojection(self):
        rng = np.random.default_rng(4403)
        x = rng.normal(size=(3, base.FEATURE_COUNT)).astype(np.float32)
        r = np.tile(np.asarray(m.R0, dtype='<f8'), (3, 1)); r[:, 0] = .3
        w, b, _ = m._parameters(); head = base._forward(x, w, b)[2]
        a, d = [m.decode(head, r, variant) for variant in m.VARIANTS]
        np.testing.assert_allclose(d-a, r-np.asarray(m.R0), atol=1e-15, rtol=0.)
        target = m.project(r-.4*m.D, r)
        la, ga = m._loss_gradients(x, r, target, w, b, 'absolute')
        ld, gd = m._loss_gradients(x, r, target, w, b, 'delta')
        self.assertNotEqual(la, ld)
        self.assertTrue(any(not np.array_equal(a, d) for a, d in zip(ga, gd)))

    def test_normalization_matches_training_and_policy_exactly(self):
        x, *_ = m._load_training(self.sources)
        for model in self.models.values():
            np.testing.assert_array_equal(np.asarray(model['feature_mean']), x.mean(axis=0))
            np.testing.assert_array_equal(np.asarray(model['feature_scale']), np.maximum(x.std(axis=0), 1e-4))
            c = reset_context(); raw = base.CameraActionFeatures()(c)
            expected = np.clip((raw.astype(np.float64)-x.mean(axis=0))/np.maximum(x.std(axis=0), 1e-4), -10., 10.).astype(np.float32)[None, :]
            forward = base._forward
            seen = []
            def spy(values, w, b):
                seen.append(values.copy()); return forward(values, w, b)
            with patch.object(base, '_forward', side_effect=spy): m.ModelPolicy(model)(c)
            self.assertEqual(len(seen), 1); np.testing.assert_array_equal(seen[0], expected)

    def test_save_load_policy_parity_preserves_existing_file(self):
        for variant, model in self.models.items():
            dest = self.fixtures.root/(variant+'-model.json')
            digest = m.save_model(model, dest); restored = m.load_model(dest)
            self.assertEqual(base.digest(restored), digest)
            with self.assertRaises(FileExistsError): m.save_model(model, dest)
            left, right = m.ModelPolicy(model), m.ModelPolicy(restored)
            previous = np.asarray(m.R0)
            for i in range(5):
                c = reset_context(i*.02, previous)
                if i == 1: c = missing_visual(c)
                a, b = left(c), right(c); self.assertEqual(a, b)
                previous = base.action_vector(a)
            self.assertEqual(left.diagnostics()['student_inference_calls'], 5)
            self.assertEqual(left.diagnostics()['teacher_fallback_calls'], 0)

    def test_exact_double_reference_does_not_roundtrip_through_features(self):
        model = deepcopy(self.models['delta'])
        model['weights'] = [np.zeros_like(w).tolist() for w in model['weights']]
        model['biases'] = [np.zeros_like(b).tolist() for b in model['biases']]
        previous = np.asarray(m.R0); previous[0] = .10000000000000002
        rounded = previous.copy(); rounded[0] = float(np.float32(previous[0]))
        raw = []
        for value in (previous, rounded):
            encoder = base.CameraActionFeatures(); encoder(reset_context())
            raw.append(encoder(reset_context(.02, value)))
        np.testing.assert_array_equal(raw[0], raw[1])
        a, b = m.ModelPolicy(model), m.ModelPolicy(model)
        a(reset_context()); b(reset_context())
        outa = a(reset_context(.02, previous)); outb = b(reset_context(.02, rounded))
        self.assertEqual(outa['joint_targets_rad'][0], previous[0])
        self.assertEqual(outb['joint_targets_rad'][0], rounded[0])
        self.assertNotEqual(outa['joint_targets_rad'][0], outb['joint_targets_rad'][0])
        # A caller can lie using Python floats; the bridge does not claim to
        # authenticate it. Explicit float32 arrays are rejected before decoding.
        c = reset_context(); c['previous_applied']['joint_targets_rad'] = np.asarray(m.R0[:6], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'reduced-precision'): m.ModelPolicy(model)(c)

    def test_context_causality_reset_and_no_label_inputs(self):
        for model in self.models.values():
            clean = reset_context(); poisoned = deepcopy(clean)
            poisoned.update(teacher_query={'action': [99]*7}, native_truth={'contacts': 999}, reference={'action': [99]*7}, after={'timestamp_seconds': 20})
            poisoned['snapshot']['signals']['object_position_estimate_m'] = {'value': [99]*3}
            self.assertEqual(m.ModelPolicy(model)(clean), m.ModelPolicy(model)(poisoned))
            for bad in (context(), reset_context(.02), missing_visual(reset_context())):
                with self.assertRaises(ValueError): m.ModelPolicy(model)(bad)
            policy = m.ModelPolicy(model); policy(clean)
            with self.assertRaises(ValueError): policy(clean)
            policy = m.ModelPolicy(model); policy(clean)
            with self.assertRaises(ValueError): policy(reset_context(.04))

    def test_changed_source_schema_units_reset_and_recipe_are_refused(self):
        document = self.models['delta']
        mutations = [lambda v: v.update(task_qualified=True), lambda v: v.update(physical_robot_ready=True),
            lambda v: v.update(scientific_registration={'claim': 'qualified'}),
            lambda v: v.update(schema=base.SCHEMA), lambda v: v.update(variant='residual'),
            lambda v: v['contract'].update(reset_reference=[0.]*7), lambda v: v['contract'].update(reference_kind='measured_q'),
            lambda v: v['contract'].update(action_units=['metres']*7), lambda v: v['contract'].update(joint_order=list(reversed(v['contract']['joint_order']))),
            lambda v: v['recipe'].update(epochs=51), lambda v: v['training'].update(optimizer_updates=49),
            lambda v: v['training'].update(parent_provenance_sha256='0'*64), lambda v: v.update(future_teacher_command=[0.]*7)]
        for mutate in mutations:
            bad = deepcopy(document); mutate(bad)
            with self.assertRaises(ValueError): m.validate_model(bad)
        with patch.object(m.ref, 'source_identity', side_effect=ValueError('source drift')):
            with self.assertRaises(ValueError): m.ModelPolicy(document)

    def test_train_rejects_holdout_wrong_source_stage_order_and_array_substitution(self):
        for field, bad_value in (('expected_split', 'confirmation'), ('expected_split', 'development'),
                ('expected_source_sha256', '0'*64), ('expected_parent_manifest_sha256', '0'*64),
                ('expected_manifest_sha256', '0'*64), ('expected_protocol_sha256', '0'*64)):
            inputs = deepcopy(self.sources); inputs[0]['pins'][field] = bad_value
            with self.assertRaises(ValueError): m.train(inputs, variant='delta')
        with self.assertRaises(ValueError): m.train(self.sources[::-1], variant='delta')
        with self.assertRaises(ValueError): m.train([{'features': np.zeros((2, 269))}], variant='delta')
        with self.assertRaises(ValueError): m.train(self.sources, variant='unknown')
        bad = deepcopy(self.sources); bad[1]['pins']['expected_protocol_sha256'] = '0'*64
        with self.assertRaises(ValueError): m.train(bad, variant='delta')

    def test_training_consumes_only_derived_inputs_and_not_raw_or_models(self):
        with (patch('dvidia_training.camera_action_capture.load_rollout', side_effect=AssertionError('no native reader')),
              patch.object(m.export, 'export_episode', side_effect=AssertionError('no raw export')),
              patch.object(base, 'load_model', side_effect=AssertionError('no old model'))):
            x, r, y, *_ = m._load_training(self.sources)
        self.assertEqual(x.shape, (12, 269)); self.assertEqual(r.dtype, np.dtype('float64')); self.assertEqual(y.shape, r.shape)

    def test_non_source_reset_is_refused_before_fitting(self):
        f = self.fixtures.parent()  # Source reference fixture deliberately has a different reset.
        directory, seal = self.fixtures.build(f)
        inputs = [dict(directory=directory, parent_cache=f['parent'],
            pins=dict(f['args'], expected_manifest_sha256=seal['manifest_sha256']))]
        with patch.object(m, '_parameters', side_effect=AssertionError('must not fit')):
            with self.assertRaisesRegex(ValueError, 'source-bound reset'):
                m.train(inputs, variant='delta')

    def test_aggregate_row_budget_refuses_before_combination_or_fitting(self):
        # Each six-row sealed fixture fits this limit, but their sum does not.
        with (patch.object(m.ref, 'MAX_SAMPLES', 11),
              patch.object(m.export, 'combine_datasets', side_effect=AssertionError('must not concatenate')),
              patch.object(m, '_parameters', side_effect=AssertionError('must not fit'))):
            with self.assertRaisesRegex(ValueError, 'Combined training rows'):
                m.train(self.sources, variant='delta')

    def test_projection_uses_same_original_twenty_updates_and_native_bounds(self):
        r = np.tile(np.asarray(m.R0), (2, 1)); r[1, 0] = .31
        head = np.asarray([[1e4]*7, [-1e4]*7])
        limits = np.asarray(m.ref.CONTRACT['limits_native'])
        for variant in m.VARIANTS:
            decoded = m.decode(head, r, variant)
            result = m.project(decoded, r)
            targets = np.clip(decoded, limits[:, 0], limits[:, 1])
            expected = np.asarray([native_trace(a, b)[-1] for a, b in zip(r, targets)])
            np.testing.assert_array_equal(result, expected)
        with self.assertRaises(ValueError): m.decode(head, r.astype(np.float32), 'delta')
        with self.assertRaises(ValueError): m.decode(head*np.nan, r, 'delta')


if __name__ == '__main__':
    unittest.main()
