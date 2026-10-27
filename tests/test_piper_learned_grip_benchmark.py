"""Jaw-only driver isolation, immutable attempts and imported-lineage guards.

All evidence here is explicitly synthetic test data; native physics and expert
action generation are mocked. No fixtures represent a scientific benchmark.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_learned_grip_benchmark as bench
from dvidia_training import piper_grip_adapter as grip
from dvidia_training import piper_imitation as core
from tests.test_piper_feedback_model import base_model, observation


RUNTIME = {'fixture-new-runtime.py': 'a'*64}
SCORE = {'valid': True, 'success': False, 'reason': 'fixture'}


def replace(path, value):
    Path(path).write_bytes(core.canonical(value))


class LearnedGripDriverTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.base = self.home/'base.json'
        self.base.write_bytes(core.canonical(base_model()))
        for owner in (bench, bench.prior):
            scope = patch.object(owner, 'BASE_SHA', bench.sha(self.base))
            scope.start(); self.addCleanup(scope.stop)
        runtime = patch.object(bench, 'runtime_hashes', return_value=deepcopy(RUNTIME))
        runtime.start(); self.addCleanup(runtime.stop)
        self.root = self.home/'study'
        self.protocol = bench.prior.prepare(self.root, self.base)
        self.parent = self.root/'prior-studies'/'study01'
        self.parent.mkdir(parents=True)
        bench.write(self.parent/'protocol.json', self.protocol)
        (self.parent/'base-model.json').write_bytes(self.base.read_bytes())
        for case in self.protocol['cases']['train']:
            # Freeze verifies original episode bytes, not these test labels.
            bench.save_episode(self.parent/'demonstrations'/case['id']/'episode.json.gz', {'fixture': case['id']})
        manifest = dict(schema='dvidia.piper-feedback-training-parent/v1', path='prior-studies/study01',
                        protocol_sha256=bench.sha(self.parent/'protocol.json'), files=bench._files(self.parent))
        bench.write(self.root/'training-parent.json', manifest)
        self.protocol['training_parent_manifest_sha256'] = bench.sha(self.root/'training-parent.json')
        replace(self.root/'protocol.json', self.protocol)

    def model(self, blind=False):
        value = grip.train(dict(features=np.zeros((2, grip.FEATURE_COUNT)), targets=np.zeros(2), groups=['fixture']*2),
                           base_sha256=core.digest(bench.read(self.base)), blind=blind, max_depth=4, min_leaf=2)
        groups = [case['id'] for case in self.protocol['cases']['train']]
        value['training'].update(sample_count=64000, scene_count=64, train_groups=groups,
            weight_sum=64000., dataset_sha256='d'*64, parent_artifacts=dict(
                protocol_sha256=bench.sha(self.parent/'protocol.json'), base_model_raw_sha256=bench.BASE_SHA,
                episodes=[dict(case_id=name, episode_sha256=bench.sha(self.parent/'demonstrations'/name/'episode.json.gz'), samples=1000)
                          for name in groups]))
        value['nodes'][0].update(samples=64000, weight_sum=64000.)
        return grip.validate_model(value)

    def candidates(self):
        for name, blind in [('jaw4', False), ('blind4', True)]:
            with patch.object(grip, 'train_parent_data', return_value=self.model(blind)):
                bench.fit(self.root, name, blind=blind, max_depth=4, min_leaf=2)
            directory = self.root/'candidates'/name
            rows = [dict(case_id=case['id'], condition=case['condition'], score=deepcopy(SCORE))
                    for case in self.protocol['cases']['development']]
            bench.write(directory/'development.json', dict(model_sha256=bench.sha(directory/'model.json'),
                protocol_sha256=bench.sha(self.root/'protocol.json'), runtime_sha256=deepcopy(RUNTIME),
                rows=rows, successes=0, scheduled=len(rows)))

    def freeze(self):
        self.candidates()
        with patch.object(bench, '_completed_attempt', return_value=deepcopy(SCORE)):
            return bench.freeze(self.root, 'jaw4', 'blind4')

    def fake_rollout(self, assets, case, base_document, *, variant, feedback_document=None, **kwargs):
        episode = dict(schema='dvidia.piper-feedback-episode/v0.2', config=bench._canonical(asdict(bench.config(case))),
            case_id=case['id'], seed=case['seed'], condition=case['condition'], pulses=case['pulses'], policy_kind=variant,
            base_model_sha256=core.digest(base_document), model_sha256=core.digest(feedback_document) if feedback_document else None,
            implementation_sha256=deepcopy(RUNTIME), valid=True, success=False, trajectory=[], actions=[])
        return episode, {'fixture': np.asarray([1.])}, {'pulses': case['pulses']}, {}

    def test_literal_eval_only_binding_rejects_training_labels_and_early_confirmation(self):
        development = self.protocol['cases']['development'][0]
        binding = bench._request_binding(self.root, development, 'feedforward', None)
        self.assertFalse(binding['collect_labels'])
        self.assertEqual(binding['expert_probability'], 1.)
        self.assertEqual(binding['runtime_sha256'], RUNTIME)
        for case, labels, probability in [(self.protocol['cases']['train'][0], False, 1.),
                                           (self.protocol['cases']['confirmation'][0], False, 1.),
                                           (development, True, 1.), (development, False, .5)]:
            with self.subTest(case=case['id'], labels=labels), self.assertRaises(ValueError):
                bench._request_binding(self.root, case, 'teacher', None, labels, probability)

    def test_imported_parent_or_manifest_tampering_is_rejected_before_fit(self):
        self.assertEqual(bench._parent(self.root), self.parent)
        file = self.parent/'base-model.json'
        file.write_bytes(file.read_bytes()+b' ')
        with patch.object(grip, 'train_parent_data', side_effect=AssertionError('must not train')):
            with self.assertRaisesRegex(ValueError, 'parent evidence'):
                bench.fit(self.root, 'bad')
        file.write_bytes(self.base.read_bytes())
        manifest = bench.read(self.root/'training-parent.json')
        manifest['new-unbound-field'] = True
        replace(self.root/'training-parent.json', manifest)
        with self.assertRaisesRegex(ValueError, 'lineage changed'):
            bench._parent(self.root)

    def test_freeze_binds_exact_models_receipts_runtime_and_parent(self):
        frozen = self.freeze()
        self.assertEqual(frozen['training_parent_sha256'], bench.sha(self.root/'training-parent.json'))
        self.assertEqual(frozen['runtime_sha256'], RUNTIME)
        self.assertEqual(frozen['selected_sha256'], bench.sha(self.root/'candidates/jaw4/model.json'))
        self.assertEqual(frozen['blind_development_sha256'], bench.sha(self.root/'candidates/blind4/development.json'))
        with self.assertRaises(ValueError):
            bench.freeze(self.root, 'jaw4', 'blind4')
        with self.assertRaises(ValueError):
            bench.fit(self.root, 'afterfreeze')
        with self.assertRaises(ValueError):
            bench.development(self.root, 'assets', 'jaw4')

    def test_freeze_rejects_parent_episode_rebinding_and_unmatched_training_recipe(self):
        self.candidates()
        directory = self.root/'candidates/jaw4'
        model = bench.read(directory/'model.json')
        model['training']['parent_artifacts']['episodes'][0]['episode_sha256'] = 'e'*64
        replace(directory/'model.json', model)
        receipt = bench.read(directory/'receipt.json')
        receipt.update(model_sha256=bench.sha(directory/'model.json'), training=model['training'])
        replace(directory/'receipt.json', receipt)
        evaluation = bench.read(directory/'development.json')
        evaluation['model_sha256'] = bench.sha(directory/'model.json')
        replace(directory/'development.json', evaluation)
        with patch.object(bench, '_completed_attempt', return_value=deepcopy(SCORE)), self.assertRaisesRegex(ValueError, '64 original'):
            bench.freeze(self.root, 'jaw4', 'blind4')
        self.assertFalse((self.root/'frozen.json').exists())

    def test_freeze_rejects_a_rehashed_but_unmatched_blind_tree_recipe(self):
        self.candidates()
        directory = self.root/'candidates/blind4'
        model = bench.read(directory/'model.json')
        model['training']['max_depth'] += 1
        grip.validate_model(model)
        replace(directory/'model.json', model)
        receipt = bench.read(directory/'receipt.json')
        receipt.update(model_sha256=bench.sha(directory/'model.json'), training=model['training'])
        replace(directory/'receipt.json', receipt)
        evaluation = bench.read(directory/'development.json')
        evaluation['model_sha256'] = bench.sha(directory/'model.json')
        replace(directory/'development.json', evaluation)
        with patch.object(bench, '_completed_attempt', return_value=deepcopy(SCORE)), self.assertRaisesRegex(ValueError, 'share the complete training budget'):
            bench.freeze(self.root, 'jaw4', 'blind4')

    def test_confirmation_checks_frozen_artifacts_and_imported_parent_before_any_attempt(self):
        frozen = self.freeze()
        receipt_path = self.root/'candidates/jaw4/receipt.json'
        original = receipt_path.read_bytes()
        receipt = bench.read(receipt_path); receipt['training_seconds'] += 1.
        replace(receipt_path, receipt)
        with patch.object(bench, 'execute', side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(ValueError, 'candidate artifact'):
                bench.confirm(self.root, 'assets')
        receipt_path.write_bytes(original)
        parent_file = self.parent/'base-model.json'
        parent_file.write_bytes(parent_file.read_bytes()+b' ')
        with patch.object(bench, 'execute', side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(ValueError, 'parent evidence'):
                bench.confirm(self.root, 'assets')
        parent_file.write_bytes(self.base.read_bytes())
        changed = deepcopy(frozen); changed['training_parent_sha256'] = 'f'*64
        replace(self.root/'frozen.json', changed)
        with patch.object(bench, 'execute', side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(ValueError, 'Frozen runtime'):
                bench.confirm(self.root, 'assets')

    def test_attempts_are_fresh_reused_only_when_complete_and_never_overwritten(self):
        case = self.protocol['cases']['development'][0]
        folder = self.root/'attempt'
        with patch.object(bench, 'rollout', side_effect=self.fake_rollout), patch(
                'dvidia_training.piper_feedback_score.score_feedback_episode', return_value=deepcopy(SCORE)):
            self.assertEqual(bench.execute(self.root, 'assets', case, 'feedforward', folder), SCORE)
        first_bytes = (folder/'episode.json.gz').read_bytes()
        with patch.object(bench, 'rollout', side_effect=AssertionError('must not replay physics')), patch(
                'dvidia_training.piper_feedback_score.score_feedback_episode', return_value=deepcopy(SCORE)):
            self.assertEqual(bench.execute(self.root, 'assets', case, 'feedforward', folder), SCORE)
            with self.assertRaises(ValueError):
                bench.execute(self.root, 'assets', case, 'teacher', folder)
            interrupted = self.root/'interrupted'
            bench.write(interrupted/'started.json', {'retained': True})
            with self.assertRaisesRegex(ValueError, 'automatic rerun refused'):
                bench.execute(self.root, 'assets', case, 'feedforward', interrupted)
        self.assertEqual(first_bytes, (folder/'episode.json.gz').read_bytes())

    def test_all_student_variants_never_construct_or_query_teacher(self):
        class FakeEnv:
            def __init__(self, *args, **kwargs):
                self.done = False
                self.initial_object_position = np.asarray([.299, -.03, .08])
                self.model = SimpleNamespace(actuator_gainprm=np.zeros((7, 1)), actuator_forcerange=np.zeros((7, 2)))
                self.contact_pair_native_ticks = {}
            def reset(self, seed):
                return observation(), {}
            def step(self, action):
                self.done = True
                return observation(.02), dict(valid=True, success=False, reason='fixture', simulation_time=.02)
            def native_arrays(self):
                return {}
            def native_metadata(self):
                return {}
        class FakeGrip:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, obs):
                return dict(joint_targets_rad=list(obs['joint_position_rad']), gripper_position_m=0.)
        case = self.protocol['cases']['development'][0]
        with (patch('dvidia_training.piper_feedback_env.FeedbackPiperTaskEnv', FakeEnv),
              patch('dvidia_training.piper_grip_adapter.GripPolicy', FakeGrip),
              patch('dvidia_training.piper_task.PiperTaskPolicy', side_effect=AssertionError('teacher unavailable')),
              patch('dvidia_training.robot_profiles.verify_assets', return_value={'fixture': True}),
              patch('dvidia_training.reference_sim._native_imports', return_value=(SimpleNamespace(__version__='fixture'), np)),
              patch('dvidia_training.reference_sim.model_digest', return_value='fixture')):
            for variant in ('feedback', 'blind', 'feedforward', 'open_jaw'):
                with self.subTest(variant=variant):
                    episode, _, _, samples = bench.rollout('assets', case, bench.read(self.base), variant=variant)
                    self.assertIsNone(episode['error'])
                    self.assertEqual(len(episode['actions']), 1)
                    self.assertEqual(samples, {})
                    if variant == 'open_jaw':
                        self.assertEqual(episode['actions'][0]['gripper_position_m'], .035)


if __name__ == '__main__':
    unittest.main()
