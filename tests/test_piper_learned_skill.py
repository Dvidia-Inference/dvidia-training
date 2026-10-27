"""Installed data-only policies run once locally and preserve every outcome."""
from contextlib import redirect_stdout, redirect_stderr
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_imitation as imitation
from dvidia_training import piper_learned_skill as skill


class InstalledPiperSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [{'features': [0.] * imitation.FEATURE_COUNT,
                 'targets': [0.] * 6 + [.5], 'case_id': 'fixture-train'} for _ in range(16)]
        cls.raw = imitation.canonical(imitation.train({'scene_ids': ['fixture-train'], 'samples': rows},
                                                      hidden_units=8, epochs=1, batch_size=8, seed=7)) + b'\n'

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root/'candidate.json'; self.model.write_bytes(self.raw)
        self.assets = self.root/'native-assets'; self.assets.mkdir()
        self.output = self.root/'installed'
        self.model_sha = sha256(self.raw).hexdigest()

    def episode(self, **changes):
        return {'valid': True, 'success': False, 'reason': 'timeout', 'error': None,
                'trajectory': [{'qpos': [0.] * 15}], 'actions': [{'gripper_position_m': .035}],
                'metrics': {'valid': True, 'success': False}, 'timing': {'wall_seconds': .1},
                'model_sha256': self.model_sha, **changes}

    def test_install_runs_exact_copied_model_once_and_preserves_native_failure(self):
        episode = self.episode()
        score = {'success': False, 'reason': 'native_outcome_failed', 'terminal_xy_error_m': .12}
        def run(assets, case, model, *, variant):
            self.assertEqual(variant, 'learned')
            self.assertEqual(assets, self.assets)
            self.assertTrue(self.output.is_dir())
            self.assertEqual(model, self.output/'model.json')
            self.assertEqual(model.read_bytes(), self.raw)
            self.assertEqual(json.loads((self.output/'install.json').read_text())['status'], 'running')
            self.assertEqual(case['start_xy'], [.30, -.03])
            self.assertEqual(case['goal_xy'], [.40, -.08])
            return episode
        with patch('dvidia_training.piper_imitation_benchmark.rollout', side_effect=run) as rollout, \
             patch('dvidia_training.piper_imitation_score.score_episode', return_value=score) as scorer, \
             patch('dvidia_training.piper_task.PiperTaskPolicy', side_effect=AssertionError('Teacher must remain unavailable')):
            manifest = skill.run_learned_skill(self.model, self.assets, self.output, goal=1)
        rollout.assert_called_once(); scorer.assert_called_once_with(episode, [.40, -.08])
        self.assertFalse(manifest['success']); self.assertEqual(manifest['status'], 'completed')
        self.assertFalse(manifest['physical_robot_ready']); self.assertFalse(manifest['teacher_available'])
        self.assertEqual(manifest['teacher_fallback_calls'], 0)
        self.assertEqual(manifest['model_sha256'], self.model_sha)
        self.assertEqual(manifest['installed_model_sha256'], self.model_sha)
        self.assertEqual(json.loads(gzip.decompress((self.output/'episode.json.gz').read_bytes())), episode)
        self.assertEqual(json.loads((self.output/'score.json').read_text()), score)
        self.assertEqual(json.loads((self.output/'install.json').read_text()), manifest)
        self.assertEqual(manifest['trained_scope']['known_goals_xy_m'], [[.35, .08], [.40, -.08]])

    def test_rollout_exception_is_recorded_as_a_failed_attempt(self):
        with patch('dvidia_training.piper_imitation_benchmark.rollout', side_effect=RuntimeError('native reset failed')):
            manifest = skill.run_learned_skill(self.model, self.assets, self.output)
        self.assertFalse(manifest['success']); self.assertEqual(manifest['error']['message'], 'native reset failed')
        episode = json.loads(gzip.decompress((self.output/'episode.json.gz').read_bytes()))
        self.assertEqual(episode['reason'], 'exception'); self.assertEqual(episode['trajectory'], [])
        self.assertEqual(episode['policy_kind'], 'learned')
        self.assertFalse(json.loads((self.output/'score.json').read_text())['success'])

    def test_changed_installed_model_or_wrong_runtime_digest_cannot_pass(self):
        for mutate in (False, True):
            output = self.root/('mutated' if mutate else 'wrong-digest')
            def run(assets, case, model, *, variant):
                if mutate: model.write_bytes(self.raw+b' ')
                return self.episode(model_sha256='0'*64 if not mutate else self.model_sha)
            with patch('dvidia_training.piper_imitation_benchmark.rollout', side_effect=run), \
                 patch('dvidia_training.piper_imitation_score.score_episode', return_value={'success': True, 'reason': 'success'}):
                manifest = skill.run_learned_skill(self.model, self.assets, output)
            self.assertFalse(manifest['success'])
            self.assertEqual(manifest['reason'], 'installed_model_changed' if mutate else 'runtime_model_digest_mismatch')

    def test_nonfinite_out_of_scope_or_invalid_goals_are_rejected_before_execution(self):
        changes = [{'start_x': value} for value in (float('nan'), float('inf'), .28399, .31401, True, 10**1000)]
        changes += [{'start_y': value} for value in (-.04501, -.01499, float('-inf'))]
        changes += [{'goal': value} for value in (-1, 2, .0, True, '0')]
        with patch('dvidia_training.piper_imitation_benchmark.rollout', side_effect=AssertionError('Invalid request ran')):
            for values in changes:
                with self.subTest(values=values), self.assertRaises(ValueError):
                    skill.run_learned_skill(self.model, self.assets, self.output, **values)
                self.assertFalse(self.output.exists())

    def test_unsafe_oversized_executable_and_reused_install_inputs_are_rejected(self):
        with patch('dvidia_training.piper_imitation_benchmark.rollout', side_effect=AssertionError('Invalid model ran')):
            for name, raw in (('unsafe name.json', self.raw), ('candidate..json', self.raw),
                              ('model.pkl', b'pickle'), ('oversized.json', b' '*(imitation.MAX_MODEL_BYTES+1)),
                              ('executable.json', b'{"code":"import os"}')):
                model = self.root/name; model.write_bytes(raw)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    skill.run_learned_skill(model, self.assets, self.output)
                self.assertFalse(self.output.exists())
            linked = self.root/'linked.json'; linked.symlink_to(self.model)
            with self.assertRaises(ValueError): skill.run_learned_skill(linked, self.assets, self.output)
            self.output.mkdir()
            with self.assertRaises(ValueError): skill.run_learned_skill(self.model, self.assets, self.output)
            self.assertEqual(list(self.output.iterdir()), [])

    def test_cli_exit_code_tracks_independent_outcome_and_validation(self):
        for success in (False, True):
            output = self.root/str(success)
            args = ['--model', str(self.model), '--assets', str(self.assets), '--output', str(output)]
            with patch('dvidia_training.piper_imitation_benchmark.rollout', return_value=self.episode()), \
                 patch('dvidia_training.piper_imitation_score.score_episode', return_value={'success': success, 'reason': 'success' if success else 'failed'}), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(skill.main(args), 0 if success else 1)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(skill.main(['--model', str(self.model), '--assets', str(self.assets),
                                         '--output', str(self.output), '--start-x', 'nan']), 2)


if __name__ == '__main__':
    unittest.main()
