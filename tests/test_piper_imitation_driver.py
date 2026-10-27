"""Recovery adapter handles only the frozen driver's serialization guard."""
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_imitation_benchmark as study
from dvidia_training import piper_imitation_driver as driver


GUARD_ERROR = 'Confirmation tape identity, model or runtime differs from the schedule.'
RUNTIME = {'piper_imitation_benchmark.py': 'a' * 64}
MODEL_SHA = 'b' * 64


@dataclass
class Config:
    object_position_m: tuple
    target_position_m: tuple


class FrozenDriverRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case = {'id': 'heldout-0', 'start_xy': [.291, -.018],
                     'goal_xy': [.35, .08], 'seed': 1729}
        (self.root / 'protocol.json').write_text(json.dumps(
            {'cases': {'confirmation': [self.case]}}))
        (self.root / 'frozen.json').write_text(json.dumps(
            {'runtime_sha256': RUNTIME, 'model_sha256': MODEL_SHA,
             'shuffled_model_sha256': 'c' * 64}))
        self.config_patch = patch.object(study, 'config', self._config)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.variants_patch = patch.object(study, 'VARIANTS', ('learned',))
        self.variants_patch.start()
        self.addCleanup(self.variants_patch.stop)
        self.runtime_patch = patch.object(study, 'runtime_hashes', return_value=RUNTIME)
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    def _config(self, case):
        return Config((*case['start_xy'], .082), (*case['goal_xy'], .08))

    @property
    def stem(self):
        return self.case['id'] + '-learned'

    @property
    def tape_path(self):
        return self.root / 'attempts' / (self.stem + '.json.gz')

    def _write_saved_attempt(self, *, valid_binding=True):
        config = json.loads(json.dumps(study.asdict(study.config(self.case))))
        episode = {
            'config': config,
            'seed': self.case['seed'] if valid_binding else self.case['seed'] + 1,
            'implementation_sha256': RUNTIME,
            'policy_kind': 'learned',
            'model_sha256': MODEL_SHA,
            'valid': False,
            'success': False,
            'reason': 'timeout',
            'trajectory': [],
            'actions': [],
            'metrics': {},
        }
        self.tape_path.parent.mkdir(parents=True, exist_ok=True)
        self.tape_path.write_bytes(gzip.compress(json.dumps(episode).encode(), mtime=0))
        marker = self.root / 'attempt-starts' / (self.stem + '.json')
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'frozen_sha256': study.sha(self.root / 'frozen.json')}))
        return episode

    def _open_confirmation(self):
        opened = self.root / 'confirmation-opened.json'
        if not opened.exists():
            opened.write_text(json.dumps({'opened': True}))

    def test_only_exact_serialization_guard_is_caught(self):
        calls = []

        def fail_with_other_error(root, assets, resume=False):
            calls.append(resume)
            raise ValueError('Confirmation tape identity differs for another reason.')

        with patch.object(study, 'confirm', side_effect=fail_with_other_error):
            with self.assertRaisesRegex(ValueError, 'differs for another reason'):
                driver.complete(self.root, Path('unused-assets'))
        self.assertEqual(calls, [False])
        self.assertFalse((self.root / 'driver-recovery').exists())

    def test_invalid_saved_binding_is_rejected_without_resuming(self):
        calls = []

        def create_bad_tape_then_raise(root, assets, resume=False):
            calls.append(resume)
            self._open_confirmation()
            self._write_saved_attempt(valid_binding=False)
            raise ValueError(GUARD_ERROR)

        with patch.object(study, 'confirm', side_effect=create_bad_tape_then_raise):
            with self.assertRaisesRegex(ValueError, 'Persisted tape identity'):
                driver.complete(self.root, Path('unused-assets'))
        self.assertEqual(calls, [False])
        self.assertFalse((self.root / 'attempt-scores' / (self.stem + '.json')).exists())
        self.assertTrue(self.tape_path.exists())  # Preserve evidence; never replace it.

    def test_progress_scores_the_same_saved_tape_and_calls_original_confirm(self):
        calls = []
        physical_attempts = []
        original_bytes = []

        def frozen_confirm(root, assets, resume=False):
            calls.append(resume)
            self._open_confirmation()
            if not self.tape_path.exists():
                physical_attempts.append(self.stem)
                self._write_saved_attempt()
                original_bytes.append(self.tape_path.read_bytes())
                raise ValueError(GUARD_ERROR)

            # This is the unchanged driver's resume path: consume the persisted
            # tape and write its score. No environment or policy is run here.
            self.assertEqual(self.tape_path.read_bytes(), original_bytes[0])
            self.assertEqual(study.load_episode(self.tape_path)['seed'], self.case['seed'])
            score_path = root / 'attempt-scores' / (self.stem + '.json')
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.write_text(json.dumps({'id': self.case['id'], 'variant': 'learned'}))
            return {'variants': {'learned': {'scheduled': 1, 'successes': 0}}}

        with patch.object(study, 'confirm', side_effect=frozen_confirm):
            result = driver.complete(self.root, Path('unused-assets'))

        self.assertEqual(calls, [False, True])
        self.assertEqual(physical_attempts, [self.stem])
        self.assertEqual(result['variants']['learned']['scheduled'], 1)
        recovery = json.loads((self.root / 'driver-recovery.json').read_text())
        self.assertEqual(recovery['serialization_guard_resumes'], 1)
        self.assertEqual(recovery['completed_tapes_reexecuted'], 0)
        self.assertTrue((self.root / 'driver-recovery' / (self.stem + '.json')).exists())

    def test_repeated_guard_for_same_tape_stops_without_third_call(self):
        calls = []

        def repeat_guard(root, assets, resume=False):
            calls.append(resume)
            self._open_confirmation()
            if not self.tape_path.exists():
                self._write_saved_attempt()
            raise ValueError(GUARD_ERROR)

        with patch.object(study, 'confirm', side_effect=repeat_guard):
            with self.assertRaisesRegex(ValueError, 'repeated without forward progress'):
                driver.complete(self.root, Path('unused-assets'))
        self.assertEqual(calls, [False, True])
        self.assertTrue(self.tape_path.exists())
        self.assertFalse((self.root / 'attempt-scores' / (self.stem + '.json')).exists())


if __name__ == '__main__':
    unittest.main()
