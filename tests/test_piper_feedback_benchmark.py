"""Protocol novelty, immutable attempts and independently bound model freeze."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_feedback_benchmark as bench
from dvidia_training import piper_feedback_model as feedback
from dvidia_training import piper_imitation as core
from tests.test_piper_feedback_model import base_model, observation


RUNTIME = {'test-runtime.py': 'a'*64}
SCORE = {'valid': True, 'success': False, 'reason': 'fixture'}


def replace_json(path, value):
    Path(path).write_bytes(core.canonical(value))


class FeedbackBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.base = self.home/'base.json'
        self.base.write_bytes(core.canonical(base_model()))
        self.base_patch = patch.object(bench, 'BASE_SHA', bench.sha(self.base))
        self.base_patch.start(); self.addCleanup(self.base_patch.stop)
        self.runtime_patch = patch.object(bench, 'runtime_hashes', return_value=deepcopy(RUNTIME))
        self.runtime_patch.start(); self.addCleanup(self.runtime_patch.stop)
        self.root = self.home/'study'
        self.protocol = bench.prepare(self.root, self.base)

    def fake_rollout(self, assets, case, base_document, *, variant, feedback_document=None,
                     collect_labels=False, expert_probability=1.):
        result = dict(schema='dvidia.piper-feedback-episode/v0.2',
                      config=bench._canonical(asdict(bench.config(case))),
                      case_id=case['id'], seed=case['seed'], condition=case['condition'],
                      pulses=case['pulses'], policy_kind=variant,
                      base_model_sha256=core.digest(base_document),
                      model_sha256=core.digest(feedback_document) if feedback_document else None,
                      implementation_sha256=deepcopy(RUNTIME), valid=True,
                      success=False, trajectory=[], actions=[])
        return result, {'fixture': np.asarray([1.])}, {'pulses': case['pulses']}, {}

    def complete_teacher(self):
        case = self.protocol['cases']['train'][0]
        folder = self.root/'demonstrations'/case['id']
        with patch.object(bench, 'rollout', side_effect=self.fake_rollout), patch(
                'dvidia_training.piper_feedback_score.score_feedback_episode', return_value=deepcopy(SCORE)):
            bench.execute(self.root, 'assets', case, 'teacher', folder)
        return case, folder

    def training_fixture(self, *, blind_basis=128):
        rng = np.random.default_rng(81)
        for case in self.protocol['cases']['train']:
            x = rng.uniform(-.5, .5, (2, feedback.FEATURE_COUNT))
            blind = x.copy(); blind[:, feedback.LIVE_FEATURE_START:] = 0.
            path = self.root/'demonstrations'/case['id']/'samples.npz'
            bench._save_npz(path, dict(features=x, blind_features=blind,
                targets=np.zeros((2, 7)), groups=np.asarray([case['id']]*2), weights=np.ones(2)))
        bench.fit(self.root, 'full', basis=128)
        bench.fit(self.root, 'blind', blind=True, basis=blind_basis)
        for name in ('full', 'blind'):
            directory = self.root/'candidates'/name
            rows = [dict(case_id=case['id'], condition=case['condition'], score=deepcopy(SCORE))
                    for case in self.protocol['cases']['development']]
            bench.write(directory/'development.json', dict(model_sha256=bench.sha(directory/'model.json'),
                protocol_sha256=bench.sha(self.root/'protocol.json'), runtime_sha256=deepcopy(RUNTIME),
                rows=rows, successes=0, scheduled=len(rows)))

    def test_protocol_starts_are_grouped_disjoint_and_new_relative_to_all_v01_splits(self):
        starts = {}
        ids = set()
        for split, cases in self.protocol['cases'].items():
            starts[split] = {tuple(case['start_xy']) for case in cases}
            self.assertFalse(starts[split]&bench.V01_STARTS)
            for point in starts[split]:
                rows = [case for case in cases if tuple(case['start_xy']) == point]
                self.assertEqual(len(rows), 4)
                self.assertEqual({row['condition'] for row in rows}, {'nominal', 'jaw_interruption'})
                self.assertEqual({tuple(row['goal_xy']) for row in rows}, {(.35, .08), (.40, -.08)})
            for case in cases:
                self.assertNotIn(case['id'], ids); ids.add(case['id'])
        self.assertFalse(starts['train']&starts['development'])
        self.assertFalse(starts['train']&starts['confirmation'])
        self.assertFalse(starts['development']&starts['confirmation'])
        self.assertEqual([len(self.protocol['cases'][s]) for s in ('train', 'development', 'confirmation')], [64, 16, 24])

    def test_config_canonicalization_survives_json_tuple_to_list_roundtrip(self):
        case = self.protocol['cases']['train'][0]
        raw = asdict(bench.config(case))
        self.assertIsInstance(raw['object_position_m'], tuple)
        canonical = bench._canonical(raw)
        self.assertIsInstance(canonical['object_position_m'], list)
        self.assertEqual(canonical, json.loads(json.dumps(raw)))
        self.assertEqual(canonical, bench._canonical(asdict(bench.config(json.loads(json.dumps(case))))))

    def test_completed_attempt_reuse_checks_case_variant_runtime_model_and_rescores(self):
        case, folder = self.complete_teacher()
        with patch.object(bench, 'rollout', side_effect=AssertionError('must not rerun native physics')), patch(
                'dvidia_training.piper_feedback_score.score_feedback_episode', return_value=deepcopy(SCORE)):
            self.assertEqual(bench.execute(self.root, 'assets', case, 'teacher', folder), SCORE)
            with self.assertRaises(ValueError):
                bench.execute(self.root, 'assets', self.protocol['cases']['train'][1], 'teacher', folder)
            with self.assertRaises(ValueError):
                bench.execute(self.root, 'assets', case, 'feedforward', folder)
            with patch.object(bench, 'runtime_hashes', return_value={'changed.py': 'b'*64}), self.assertRaises(ValueError):
                bench.execute(self.root, 'assets', case, 'teacher', folder)
        with patch('dvidia_training.piper_feedback_score.score_feedback_episode', return_value={**SCORE, 'success': True}), self.assertRaises(ValueError):
            bench.execute(self.root, 'assets', case, 'teacher', folder)

    def test_completed_tape_cannot_change_identity_even_if_its_receipt_hash_is_rewritten(self):
        case, folder = self.complete_teacher()
        tape = bench.load_episode(folder/'episode.json.gz'); tape['seed'] += 1
        # Test an identity change with a fully updated outer artifact hash.
        (folder/'episode.json.gz').unlink(); bench.save_episode(folder/'episode.json.gz', tape)
        receipt = bench.read(folder/'receipt.json')
        receipt['files']['episode.json.gz'] = bench.sha(folder/'episode.json.gz')
        replace_json(folder/'receipt.json', receipt)
        with self.assertRaises(ValueError): bench.execute(self.root, 'assets', case, 'teacher', folder)

    def test_completed_feedback_attempt_rejects_a_different_valid_model(self):
        self.training_fixture()
        case = self.protocol['cases']['development'][0]
        folder = self.root/'candidate-attempt'
        model_path = self.root/'candidates'/'full'/'model.json'
        with patch.object(bench, 'rollout', side_effect=self.fake_rollout), patch(
                'dvidia_training.piper_feedback_score.score_feedback_episode', return_value=deepcopy(SCORE)):
            bench.execute(self.root, 'assets', case, 'feedback', folder, model_path=model_path)
        changed = bench.read(model_path); changed['weights'][0][0] += .01
        alternative = self.root/'different-valid-model.json'; bench.write(alternative, changed)
        feedback.validate_model(changed)
        with patch.object(bench, 'rollout', side_effect=AssertionError('must not rerun')), self.assertRaisesRegex(ValueError, 'identity, model'):
            bench.execute(self.root, 'assets', case, 'feedback', folder, model_path=alternative)

    def test_unfinished_attempt_or_undeclared_case_cannot_be_rerun(self):
        case = self.protocol['cases']['train'][0]
        folder = self.root/'unfinished'
        bench.write(folder/'started.json', {'old': 'attempt'})
        with patch.object(bench, 'rollout', side_effect=AssertionError('must not rerun')):
            with self.assertRaises(ValueError): bench.execute(self.root, 'assets', case, 'teacher', folder)
            unknown = {**case, 'start_xy': [.299, -.03]}
            with self.assertRaises(ValueError): bench.execute(self.root, 'assets', unknown, 'teacher', self.root/'unknown')
            with self.assertRaises(ValueError):
                bench.execute(self.root, 'assets', self.protocol['cases']['development'][0], 'teacher',
                              self.root/'labels-on-development', collect_labels=True)
            with self.assertRaises(ValueError):
                bench.execute(self.root, 'assets', self.protocol['cases']['confirmation'][0], 'teacher', self.root/'early-confirmation')

    def test_fit_uses_native_residual_training_api_and_freeze_binds_exact_pair(self):
        self.training_fixture()
        with patch.object(bench, '_completed_attempt', return_value=deepcopy(SCORE)):
            frozen = bench.freeze(self.root, 'full', 'blind')
        self.assertEqual(frozen['runtime_sha256'], RUNTIME)
        self.assertEqual(frozen['protocol_sha256'], bench.sha(self.root/'protocol.json'))
        self.assertEqual(frozen['base_model_sha256'], bench.BASE_SHA)
        for name in ('selected', 'blind'):
            directory = self.root/'candidates'/frozen[name]
            self.assertEqual(frozen[f'{name}_receipt_sha256'], bench.sha(directory/'receipt.json'))
            self.assertEqual(frozen[f'{name}_development_sha256'], bench.sha(directory/'development.json'))
        with self.assertRaises(ValueError): bench.freeze(self.root, 'full', 'blind')
        with self.assertRaises(ValueError): bench.fit(self.root, 'postfreeze', basis=128)

    def test_freeze_rejects_unmatched_full_blind_architecture(self):
        self.training_fixture(blind_basis=256)
        with patch.object(bench, '_completed_attempt', return_value=deepcopy(SCORE)), self.assertRaisesRegex(ValueError, 'share architecture'):
            bench.freeze(self.root, 'full', 'blind')
        self.assertFalse((self.root/'frozen.json').exists())

    def test_freeze_rejects_protocol_runtime_training_group_and_development_changes(self):
        self.training_fixture()
        with patch.object(bench, '_completed_attempt', return_value=deepcopy(SCORE)):
            with patch.object(bench, 'runtime_hashes', return_value={'changed.py': 'b'*64}), self.assertRaises(ValueError):
                bench.freeze(self.root, 'full', 'blind')
            protocol = deepcopy(self.protocol); protocol['selection'] = 'changed after fitting'
            replace_json(self.root/'protocol.json', protocol)
            with self.assertRaises(ValueError): bench.freeze(self.root, 'full', 'blind')
            replace_json(self.root/'protocol.json', self.protocol)
            # Restore the original byte formatting/hash, not just its JSON values.
            (self.root/'protocol.json').unlink(); bench.write(self.root/'protocol.json', self.protocol)
            directory = self.root/'candidates'/'blind'
            model = bench.read(directory/'model.json')
            model['training']['train_groups'].reverse()
            replace_json(directory/'model.json', model)
            receipt = bench.read(directory/'receipt.json')
            receipt['training'] = model['training']; receipt['model_sha256'] = bench.sha(directory/'model.json')
            replace_json(directory/'receipt.json', receipt)
            with self.assertRaisesRegex(ValueError, 'complete collected training'):
                bench.freeze(self.root, 'full', 'blind')

    def test_confirmation_checks_frozen_receipts_before_any_physics(self):
        self.training_fixture()
        with patch.object(bench, '_completed_attempt', return_value=deepcopy(SCORE)):
            bench.freeze(self.root, 'full', 'blind')
        directory = self.root/'candidates'/'full'
        changed = bench.read(directory/'receipt.json'); changed['training_seconds'] += 1.
        replace_json(directory/'receipt.json', changed)
        with patch.object(bench, 'execute', side_effect=AssertionError('must not start physics')), self.assertRaisesRegex(ValueError, 'receipt changed'):
            bench.confirm(self.root, 'assets')

    def test_student_rollout_never_constructs_or_queries_expert(self):
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

        class FakeStudent:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, obs):
                return {'joint_targets_rad': list(obs['joint_position_rad']), 'gripper_position_m': .0175}

        case = self.protocol['cases']['train'][0]
        with patch('dvidia_training.piper_feedback_env.FeedbackPiperTaskEnv', FakeEnv), patch(
                'dvidia_training.piper_task.PiperTaskPolicy', side_effect=AssertionError('expert must not be constructed')), patch(
                'dvidia_training.robot_profiles.verify_assets', return_value={'fixture': True}), patch(
                'dvidia_training.reference_sim._native_imports', return_value=(SimpleNamespace(__version__='fixture'), np)), patch(
                'dvidia_training.reference_sim.model_digest', return_value='fixture'), patch(
                'dvidia_training.piper_feedback_model.FeedbackPolicy', FakeStudent):
            for variant in ('feedforward', 'feedback', 'blind', 'open_jaw'):
                with self.subTest(variant=variant):
                    result, _, _, samples = bench.rollout('assets', case, bench.read(self.base), variant=variant)
                    self.assertIsNone(result['error'])
                    self.assertEqual(len(result['actions']), 1)
                    self.assertEqual(samples['targets'], [])


if __name__ == '__main__':
    unittest.main()
