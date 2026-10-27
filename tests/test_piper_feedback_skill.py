"""Downloaded feedback installation boundaries and immutable local evidence."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_feedback_benchmark as bench
from dvidia_training import piper_feedback_model as feedback
from dvidia_training import piper_feedback_skill as skill
from dvidia_training import piper_grip_adapter as grip
from dvidia_training import piper_learned_grip_benchmark as grip_bench
from dvidia_training import piper_imitation as core
from tests.test_piper_feedback_model import base_model


PROVENANCE = {'profile': 'piper', 'fixture_verified': True}


class FeedbackSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_document = base_model()
        cls.base_raw = core.canonical(cls.base_document)+b'\n'
        cls.feedback_document = feedback.train(
            {'features': np.zeros((2, feedback.FEATURE_COUNT)), 'targets': np.zeros((2, 7)),
             'groups': ['fixture-episode']*2}, base_sha256=core.digest(cls.base_document), basis=128)
        cls.feedback_raw = core.canonical(cls.feedback_document)+b'\n'

    def setUp(self):
        self.temporary = TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base, self.model = self.root/'base.json', self.root/'feedback.json'
        self.base.write_bytes(self.base_raw); self.model.write_bytes(self.feedback_raw)
        self.assets = self.root/'assets'; self.assets.mkdir()
        self.output = self.root/'attempt'
        base_patch = patch.object(bench, 'BASE_SHA', sha256(self.base_raw).hexdigest())
        base_patch.start(); self.addCleanup(base_patch.stop)
        assets_patch = patch('dvidia_training.robot_profiles.verify_assets', return_value=deepcopy(PROVENANCE))
        self.asset_check = assets_patch.start(); self.addCleanup(assets_patch.stop)

    def fake_rollout(self, assets, case, base_document, *, variant, feedback_document, collect_labels):
        self.assertFalse(collect_labels)
        self.assertEqual(case['condition'], 'nominal'); self.assertEqual(case['pulses'], [])
        self.assertEqual(variant, 'blind' if feedback_document['blind'] else 'feedback')
        self.assertEqual(base_document, self.base_document)
        self.assertEqual(assets, self.assets)
        self.assertEqual((self.output/'base-model.json').read_bytes(), self.base_raw)
        self.assertEqual((self.output/'feedback-model.json').read_bytes(), core.canonical(feedback_document)+b'\n')
        result = dict(config=bench._canonical(bench.asdict(bench.config(case))),
                      case_id=case['id'], seed=case['seed'], policy_kind=variant,
                      condition='nominal', pulses=[], valid=True, success=False,
                      base_model_sha256=core.digest(base_document), model_sha256=core.digest(feedback_document),
                      implementation_sha256=bench.runtime_hashes(), asset_provenance=deepcopy(PROVENANCE),
                      trajectory=[], actions=[], metrics={}, error=None, compiled_model_sha256='fixture')
        return result, {'fixture': np.asarray([1., 2.])}, {'rows': 2, 'pulses': []}, {'targets': []}

    def run_with_score(self, score):
        with patch.object(bench, 'rollout', side_effect=self.fake_rollout), patch(
                'dvidia_training.piper_feedback_score.score_feedback_episode', return_value=score):
            return skill.run_feedback_skill(self.base, self.model, self.assets, self.output, goal=1)

    def assert_receipt(self, manifest):
        receipt = json.loads((self.output/'receipt.json').read_text())
        self.assertEqual(set(receipt['files']), {p.name for p in self.output.iterdir() if p.name != 'receipt.json'})
        for name, expected in receipt['files'].items():
            self.assertEqual(sha256((self.output/name).read_bytes()).hexdigest(), expected)
        self.assertEqual(json.loads((self.output/'install.json').read_text()), manifest)

    def test_install_exact_bytes_preserves_valid_failure_and_hashes_every_artifact(self):
        score = {'valid': True, 'success': False, 'reason': 'retention_lost'}
        with patch('dvidia_training.piper_task.PiperTaskPolicy', side_effect=AssertionError('Expert unavailable')):
            manifest = self.run_with_score(score)
        self.assertTrue(manifest['valid']); self.assertFalse(manifest['success'])
        self.assertEqual(manifest['reason'], 'retention_lost')
        self.assertFalse(manifest['physical_robot_ready']); self.assertFalse(manifest['teacher_available'])
        self.assertFalse(manifest['user_fault_injection'])
        self.assertEqual(manifest['case']['goal_xy'], [.40, -.08])
        self.assertEqual(manifest['base_model_raw_sha256'], sha256(self.base_raw).hexdigest())
        self.assertEqual(manifest['feedback_model_raw_sha256'], sha256(self.feedback_raw).hexdigest())
        self.assertEqual(manifest['feedback_model_sha256'], core.digest(self.feedback_document))
        self.assertNotEqual(manifest['feedback_model_raw_sha256'], manifest['feedback_model_sha256'])
        with np.load(self.output/'native.npz', allow_pickle=False) as native:
            np.testing.assert_array_equal(native['fixture'], [1., 2.])
        self.assertEqual(json.loads((self.output/'score.json').read_text()), score)
        self.assertEqual(json.loads(gzip.decompress((self.output/'episode.json.gz').read_bytes()))['case_id'], manifest['case']['id'])
        self.assert_receipt(manifest)

    def test_existing_directory_file_symlink_and_asset_collision_are_never_modified(self):
        for kind in ('directory', 'file', 'symlink', 'dangling-symlink', 'inside-assets'):
            output = self.root/kind
            if kind == 'directory':
                output.mkdir(); (output/'prior.txt').write_bytes(b'keep')
            elif kind == 'file':
                output.write_bytes(b'keep')
            elif kind == 'symlink':
                output.symlink_to(self.assets, target_is_directory=True)
            elif kind == 'dangling-symlink':
                output.symlink_to(self.root/'missing', target_is_directory=True)
            else:
                output = self.assets/'new-attempt'
            with self.subTest(kind=kind), patch.object(bench, 'rollout', side_effect=AssertionError('must not run')), self.assertRaises(ValueError):
                skill.run_feedback_skill(self.base, self.model, self.assets, output)
        self.assertEqual((self.root/'directory'/'prior.txt').read_bytes(), b'keep')
        self.assertEqual((self.root/'file').read_bytes(), b'keep')
        self.assertFalse((self.assets/'new-attempt').exists())

    def test_nonfinite_out_of_scope_starts_and_unknown_goals_are_rejected_before_run(self):
        options = [{'start_x': value} for value in (float('nan'), float('inf'), .2839, .3141, True, 10**1000)]
        options += [{'start_y': value} for value in (-.0451, -.0149, float('-inf'))]
        options += [{'goal': value} for value in (-1, 2, .0, True, '0')]
        with patch.object(bench, 'rollout', side_effect=AssertionError('must not run')):
            for changes in options:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    skill.run_feedback_skill(self.base, self.model, self.assets, self.output, **changes)
                self.assertFalse(self.output.exists())

    def test_json_only_bound_models_reject_pickle_duplicates_nonfinite_and_symlinks(self):
        duplicate = self.feedback_raw.decode().replace('"schema":', '"schema":"duplicate","schema":', 1).encode()
        for index, (name, raw) in enumerate((('model.pkl', b'pickle'), ('code.json', b'{"code":"exec"}'),
                          ('oversized.json', b' '*(core.MAX_MODEL_BYTES+1)), ('duplicate.json', duplicate),
                          ('nonfinite.json', b'{"x":NaN}'))):
            source = self.root/name; source.write_bytes(raw)
            with self.subTest(name=name), self.assertRaises(ValueError):
                skill.run_feedback_skill(self.base, source, self.assets, self.output)
            self.assertFalse(self.output.exists())
        source = self.root/'linked.json'; source.symlink_to(self.model)
        with self.assertRaises(ValueError): skill.run_feedback_skill(self.base, source, self.assets, self.output)
        wrong = deepcopy(self.feedback_document); wrong['base_sha256'] = 'f'*64
        source = self.root/'wrong-base.json'; source.write_bytes(core.canonical(wrong))
        with self.assertRaises(ValueError): skill.run_feedback_skill(self.base, source, self.assets, self.output)
        self.base.write_bytes(self.base_raw+b' ')
        with self.assertRaises(ValueError): skill.run_feedback_skill(self.base, self.model, self.assets, self.output)

    def test_runtime_exception_is_preserved_as_invalid_attempt_with_no_retry(self):
        with patch.object(bench, 'rollout', side_effect=RuntimeError('native reset failed')) as execute:
            manifest = skill.run_feedback_skill(self.base, self.model, self.assets, self.output)
        execute.assert_called_once()
        self.assertFalse(manifest['valid']); self.assertFalse(manifest['success'])
        self.assertEqual(manifest['error']['message'], 'native reset failed')
        self.assert_receipt(manifest)
        with self.assertRaises(ValueError): skill.run_feedback_skill(self.base, self.model, self.assets, self.output)

    def test_runtime_model_config_or_asset_mismatch_can_never_pass(self):
        for change in ('digest', 'configuration', 'asset', 'installed-bytes'):
            self.output = self.root/change
            def run(*args, **kwargs):
                result, arrays, metadata, samples = self.fake_rollout(*args, **kwargs)
                if change == 'digest': result['model_sha256'] = '0'*64
                elif change == 'configuration': result['config']['target_position_m'][0] += .01
                elif change == 'asset': result['asset_provenance'] = {'wrong': True}
                else: (self.output/'feedback-model.json').write_bytes(self.feedback_raw+b' ')
                return result, arrays, metadata, samples
            with self.subTest(change=change), patch.object(bench, 'rollout', side_effect=run), patch(
                    'dvidia_training.piper_feedback_score.score_feedback_episode', return_value={'valid': True, 'success': True, 'reason': 'success'}):
                manifest = skill.run_feedback_skill(self.base, self.model, self.assets, self.output)
            self.assertFalse(manifest['valid']); self.assertFalse(manifest['success'])
            self.assertIn(manifest['reason'], ('runtime_model_or_source_digest_mismatch', 'runtime_case_or_configuration_mismatch',
                                              'runtime_asset_provenance_mismatch', 'installed_model_changed'))
            self.assert_receipt(manifest)

    def test_object_arrays_never_emit_a_pickle_bearing_native_archive(self):
        def run(*args, **kwargs):
            result, arrays, metadata, samples = self.fake_rollout(*args, **kwargs)
            arrays['unsafe'] = np.asarray([{'python': 'object'}], dtype=object)
            return result, arrays, metadata, samples
        with patch.object(bench, 'rollout', side_effect=run), patch(
                'dvidia_training.piper_feedback_score.score_feedback_episode', return_value={'valid': True, 'success': True, 'reason': 'success'}):
            manifest = skill.run_feedback_skill(self.base, self.model, self.assets, self.output)
        self.assertFalse(manifest['valid']); self.assertFalse(manifest['success'])
        with np.load(self.output/'native.npz', allow_pickle=False) as native:
            self.assertEqual(native.files, [])
        self.assertIn('archive_error', json.loads((self.output/'native.json').read_text()))
        self.assert_receipt(manifest)

    def test_blind_model_mode_is_inferred_from_its_validated_document(self):
        blind = feedback.train({'features': np.zeros((2, feedback.FEATURE_COUNT)), 'targets': np.zeros((2, 7)),
                                'groups': ['fixture-episode']*2}, base_sha256=core.digest(self.base_document), basis=128, blind=True)
        self.model.write_bytes(core.canonical(blind)+b'\n')
        manifest = self.run_with_score({'valid': True, 'success': False, 'reason': 'fixture'})
        self.assertEqual(manifest['policy_kind'], 'blind')

    def test_cli_status_distinguishes_success_valid_failure_and_invalid_attempt(self):
        for valid, success, expected in ((True, True, 0), (True, False, 1), (False, False, 2)):
            self.output = self.root/f'status-{expected}'
            args = ['--base-model', str(self.base), '--model', str(self.model), '--assets', str(self.assets), '--output', str(self.output)]
            with patch.object(bench, 'rollout', side_effect=self.fake_rollout), patch(
                    'dvidia_training.piper_feedback_score.score_feedback_episode', return_value={'valid': valid, 'success': success, 'reason': 'fixture'}), redirect_stdout(io.StringIO()):
                self.assertEqual(skill.main(args), expected)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(skill.main(['--base-model', str(self.base), '--model', str(self.model),
                '--assets', str(self.assets), '--output', str(self.root/'bad-input'), '--start-x', 'nan']), 2)


class JawAdapterInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_document = base_model()
        cls.base_raw = core.canonical(cls.base_document)+b'\n'
        cls.models = {blind: grip.train(dict(features=np.zeros((2, grip.FEATURE_COUNT)),
                      targets=np.asarray([0., .035]), groups=['teacher-fixture']*2),
                      base_sha256=core.digest(cls.base_document), blind=blind, min_leaf=2)
                      for blind in (False, True)}

    def setUp(self):
        self.temporary = TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base, self.model = self.root/'base.json', self.root/'jaw-model.json'
        self.base.write_bytes(self.base_raw)
        self.model.write_bytes(core.canonical(self.models[False])+b'\n')
        self.assets = self.root/'assets'; self.assets.mkdir()
        self.output = self.root/'installed'
        base_patch = patch.object(bench, 'BASE_SHA', sha256(self.base_raw).hexdigest())
        base_patch.start(); self.addCleanup(base_patch.stop)
        assets_patch = patch('dvidia_training.robot_profiles.verify_assets', return_value=deepcopy(PROVENANCE))
        assets_patch.start(); self.addCleanup(assets_patch.stop)

    def episode(self, assets, case, base_document, *, variant, feedback_document, collect_labels):
        self.assertFalse(collect_labels)
        self.assertEqual(feedback_document['schema'], grip.SCHEMA)
        self.assertEqual(variant, 'blind' if feedback_document['blind'] else 'feedback')
        self.assertEqual(case['pulses'], [])
        self.assertEqual((self.output/'feedback-model.json').read_bytes(), self.model.read_bytes())
        value = dict(case_id=case['id'], seed=case['seed'], config=bench._canonical(bench.asdict(bench.config(case))),
                     condition='nominal', pulses=[], policy_kind=variant,
                     base_model_sha256=core.digest(base_document), model_sha256=core.digest(feedback_document),
                     implementation_sha256=grip_bench.runtime_hashes(), valid=True, success=False,
                     asset_provenance=deepcopy(PROVENANCE), trajectory=[], actions=[], metrics={}, error=None)
        return value, {'fixture': np.asarray([1.])}, {'pulses': []}, {'targets': []}

    def test_full_and_blind_jaw_models_dispatch_to_their_own_runtime_without_expert_labels(self):
        for blind in (False, True):
            self.output = self.root/f'jaw-{blind}'
            self.model.write_bytes(core.canonical(self.models[blind])+b'\n')
            with self.subTest(blind=blind), patch.object(grip_bench, 'rollout', side_effect=self.episode) as dispatch, patch.object(
                    bench, 'rollout', side_effect=AssertionError('wrong architecture runtime')), patch(
                    'dvidia_training.piper_task.PiperTaskPolicy', side_effect=AssertionError('expert forbidden')), patch(
                    'dvidia_training.piper_feedback_score.score_feedback_episode', return_value={'valid': True, 'success': False, 'reason': 'fixture'}):
                manifest = skill.run_feedback_skill(self.base, self.model, self.assets, self.output)
            dispatch.assert_called_once()
            self.assertTrue(manifest['valid'])
            self.assertEqual(manifest['architecture'], 'learned_absolute_jaw_over_frozen_six_joint_path')
            self.assertEqual(manifest['runtime_sha256'], grip_bench.runtime_hashes())
            self.assertIn('piper_grip_adapter.py', manifest['runtime_sha256'])
            self.assertIn('piper_learned_grip_benchmark.py', manifest['runtime_sha256'])
            self.assertFalse(manifest['teacher_available']); self.assertEqual(manifest['teacher_fallback_calls'], 0)
            self.assertEqual(manifest['policy_kind'], 'blind' if blind else 'feedback')
            receipt = json.loads((self.output/'receipt.json').read_text())
            for name, expected in receipt['files'].items():
                self.assertEqual(sha256((self.output/name).read_bytes()).hexdigest(), expected)

    def test_jaw_json_duplicate_fields_cannot_be_lost_during_schema_dispatch(self):
        raw = core.canonical(self.models[False]).decode().replace('"schema":', '"schema":"duplicate","schema":', 1)
        self.model.write_text(raw)
        with patch.object(grip_bench, 'rollout', side_effect=AssertionError('invalid model executed')), self.assertRaises(ValueError):
            skill.run_feedback_skill(self.base, self.model, self.assets, self.output)
        self.assertFalse(self.output.exists())

    def test_jaw_attempt_with_residual_runtime_hashes_is_rejected_even_if_scored_successful(self):
        def mismatched(*args, **kwargs):
            episode, arrays, metadata, samples = self.episode(*args, **kwargs)
            episode['implementation_sha256'] = bench.runtime_hashes()
            return episode, arrays, metadata, samples
        with patch.object(grip_bench, 'rollout', side_effect=mismatched), patch(
                'dvidia_training.piper_feedback_score.score_feedback_episode', return_value={'valid': True, 'success': True, 'reason': 'success'}):
            manifest = skill.run_feedback_skill(self.base, self.model, self.assets, self.output)
        self.assertFalse(manifest['valid']); self.assertFalse(manifest['success'])
        self.assertEqual(manifest['reason'], 'runtime_model_or_source_digest_mismatch')

    def test_jaw_attempt_rejects_unexpected_label_collection(self):
        def collected(*args, **kwargs):
            episode, arrays, metadata, samples = self.episode(*args, **kwargs)
            samples['targets'] = [.035]
            return episode, arrays, metadata, samples
        with patch.object(grip_bench, 'rollout', side_effect=collected):
            manifest = skill.run_feedback_skill(self.base, self.model, self.assets, self.output)
        self.assertFalse(manifest['valid']); self.assertFalse(manifest['success'])
        self.assertIn('unexpectedly collected expert labels', manifest['error']['message'])


if __name__ == '__main__':
    unittest.main()
