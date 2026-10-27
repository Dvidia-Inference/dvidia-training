"""Resume-path integrity tests for the PiPER imitation confirmation schedule."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_imitation_benchmark as benchmark


class ConfirmationResumeTests(unittest.TestCase):
    def test_resume_rejects_score_row_for_a_different_scheduled_attempt(self):
        """A cached score must not silently change the report's case/variant roster."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {'id': 'confirmation-00-goal0', 'start_xy': [.287, -.032],
                    'goal_xy': [.35, .08], 'seed': 261000}
            protocol = {'schema': benchmark.SCHEMA,
                        'cases': {'confirmation': [case]},
                        'variants': list(benchmark.VARIANTS)}
            protocol_path = root / 'protocol.json'
            protocol_path.write_text(json.dumps(protocol))
            files = {
                'model/model.json': b'{}',
                'model/shuffled-model.json': b'{}',
                'training/dataset.json': b'{}',
                'training/samples.npz': b'fixture',
            }
            for name, raw in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            selected_receipt = root / 'development_models/mlp160/receipt.json'
            selected_receipt.parent.mkdir(parents=True)
            selected_receipt.write_text('{}')
            runtime = {'benchmark.py': 'a' * 64}
            frozen = {
                'model_sha256': benchmark.sha(root / 'model/model.json'),
                'shuffled_model_sha256': benchmark.sha(root / 'model/shuffled-model.json'),
                'dataset_sha256': benchmark.sha(root / 'training/dataset.json'),
                'samples_sha256': benchmark.sha(root / 'training/samples.npz'),
                'protocol_sha256': benchmark.sha(protocol_path),
                'runtime_sha256': runtime,
                'selected_name': 'mlp160',
            }
            frozen_path = root / 'frozen.json'
            frozen_path.write_text(json.dumps(frozen))
            schedule = [dict(id=case['id'], variant=variant,
                             episode=f"attempts/{case['id']}-{variant}.json.gz")
                        for variant in benchmark.VARIANTS]
            opened = {'frozen_sha256': benchmark.sha(frozen_path),
                      'planned_attempts': schedule}
            (root / 'confirmation-opened.json').write_text(json.dumps(opened))
            marker=root/'attempt-starts'/f"{case['id']}-learned.json"
            marker.parent.mkdir()
            marker.write_text(json.dumps({'frozen_sha256':benchmark.sha(frozen_path)}))

            # The learned trace and score are hash-consistent, but the row claims
            # to represent another case and variant. Resume must reject it.
            episode = {'success': False, 'valid': False}
            attempt = root / schedule[0]['episode']
            attempt.parent.mkdir(parents=True)
            attempt.write_bytes(gzip.compress(json.dumps(episode).encode(), mtime=0))
            score = {'success': False}
            cached = {
                **case,
                'id': 'different-case',
                'variant': 'teacher',
                'score': score,
                'episode': schedule[0]['episode'],
                'sha256': benchmark.sha(attempt),
            }
            cached_path = root / 'attempt-scores' / f"{case['id']}-learned.json"
            cached_path.parent.mkdir(parents=True)
            cached_path.write_text(json.dumps(cached))

            with (patch.object(benchmark, 'runtime_hashes', return_value=runtime),
                  patch.object(benchmark, 'config', return_value={}),
                  patch('dvidia_training.piper_imitation_score.score_episode', return_value=score),
                  patch('dvidia_training.piper_task.run_episode', return_value=episode),
                  patch.object(benchmark, 'rollout', return_value=episode)):
                with self.assertRaisesRegex(ValueError, 'confirmation identity'):
                    benchmark.confirm(root, assets=Path('unused'), resume=True)

    def test_atomic_episode_write_preserves_first_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'attempt.json.gz'
            benchmark.save_episode(path,{'identity':'first'})
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):
                benchmark.save_episode(path,{'identity':'second'})
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(path.parent.iterdir()),[path])

    def test_failed_atomic_link_exposes_no_partial_tape(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'attempt.json.gz'
            with patch.object(benchmark.os,'link',side_effect=OSError('interrupted install')):
                with self.assertRaises(OSError): benchmark.save_episode(path,{'identity':'failed'})
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()),[])


if __name__ == '__main__':
    unittest.main()
