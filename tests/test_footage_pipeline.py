"""Real encoded-media integration, export integrity and failed-job boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from dvidia_training.footage_example import create_example
from dvidia_training.footage_pipeline import inspect_run, prepare_run, run


class FootagePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve()
        cls.source = cls.root / 'source'
        create_example(cls.source, clips=5)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def fresh(self):
        return Path(tempfile.mkdtemp(dir=self.root))

    def test_actual_media_to_learned_weights_and_verified_data_only_export(self):
        intake = prepare_run(self.source, self.fresh())
        self.assertEqual(intake['counts']['episodes'], 5)
        events = []
        destination = self.fresh()
        result = run(intake['dataset_path'], destination, progress=events.append)
        self.assertEqual(result, inspect_run(destination))
        self.assertTrue(result['capability']['visual_trained'])
        self.assertFalse(result['capability']['robot_skill_acquired'])
        self.assertFalse(result['capability']['physical_robot_ready'])
        self.assertFalse(result['capability']['movement_candidate_trained'])
        self.assertGreater(result['visual']['test']['transitions'], 0)
        self.assertGreater(result['visual']['test']['persistence_rgb_mse'], 0)
        self.assertEqual([row['stage'] for row in events], ['intake', 'training', 'export', 'complete'])
        with zipfile.ZipFile(destination / 'training-result.zip') as archive:
            self.assertIn('training/visual_model.npz', archive.namelist())
            self.assertNotIn('dataset/dataset.json', archive.namelist())
            self.assertFalse(any(name.endswith('.mp4') for name in archive.namelist()))
            receipt = json.loads(archive.read('dataset-receipt.json'))
            self.assertFalse(receipt['media_included'])
            self.assertNotIn(str(self.source), archive.read('run.json').decode())
        model = destination / 'training' / 'visual_model.npz'
        model.write_bytes(model.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            inspect_run(destination)

    def test_candidate_requires_movement_and_incompatible_task_cannot_claim_success(self):
        output = self.fresh()
        with self.assertRaisesRegex(ValueError, 'validated movement supervision'):
            run(self.source, output, task_source=Path(__file__).parents[1] / 'src' / 'dvidia_training' / 'place_cup.skill.json')
        self.assertFalse((output / 'run.json').exists())
        self.assertFalse((output / 'training-result.zip').exists())

    def test_prepared_dataset_tamper_or_path_escape_stops_before_training(self):
        intake = prepare_run(self.source, self.fresh())
        dataset_path = Path(intake['dataset_path'])
        original = json.loads(dataset_path.read_text())
        for relative in ('/etc/passwd', '../outside.mp4'):
            value = deepcopy(original)
            value['episodes'][0]['media_path'] = relative
            dataset_path.write_text(json.dumps(value))
            output = self.fresh()
            with self.assertRaises(ValueError):
                run(dataset_path, output)
            self.assertFalse((output / 'run.json').exists())
        value = deepcopy(original)
        value['episodes'][0]['media_sha256'] = '0' * 64
        dataset_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            run(dataset_path, self.fresh())

    def test_archive_tamper_and_prior_output_are_rejected(self):
        output = self.fresh()
        run(self.source, output)
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            run(self.source, output)
        with zipfile.ZipFile(output / 'training-result.zip', 'a') as archive:
            archive.writestr('unexpected.py', 'print("no")')
        with self.assertRaisesRegex(ValueError, 'Unexpected files'):
            inspect_run(output)

    def test_prepared_readiness_and_groups_cannot_override_actual_checks(self):
        intake = prepare_run(self.source, self.fresh())
        dataset_path = Path(intake['dataset_path'])
        original = json.loads(dataset_path.read_text())
        value = deepcopy(original)
        value['capability']['movement_ready'] = True
        value['capability']['physical_robot_ready'] = True
        dataset_path.write_text(json.dumps(value))
        audited = prepare_run(dataset_path, self.fresh())
        self.assertFalse(audited['capability']['movement_ready'])
        self.assertFalse(audited['capability']['movement_training_ready'])
        self.assertNotIn('physical_robot_ready', audited['capability'])
        value = deepcopy(original)
        value['episodes'][0]['source_recording_id'] = value['episodes'][1]['source_recording_id']
        dataset_path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            prepare_run(dataset_path, self.fresh())
        value = deepcopy(original)
        value['episodes'][0]['duration_seconds'] += 1
        dataset_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'actual media'):
            prepare_run(dataset_path, self.fresh())

    def test_prepared_original_manifest_is_retained_and_hash_verified(self):
        intake = prepare_run(self.source, self.fresh())
        original_path = Path(intake['dataset_path']).parent / 'source_manifest.json'
        audited = prepare_run(intake['dataset_path'], self.fresh())
        copied_path = Path(audited['dataset_path']).parent / 'source_manifest.json'
        self.assertEqual(original_path.read_bytes(), copied_path.read_bytes())
        original_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'source manifest'):
            prepare_run(intake['dataset_path'], self.fresh())


if __name__ == '__main__':
    unittest.main()
