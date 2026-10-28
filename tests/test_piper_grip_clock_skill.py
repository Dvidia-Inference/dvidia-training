from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import piper_grip_clock_skill as skill
from dvidia_training.piper_imitation_benchmark import write, read


class InstallTests(unittest.TestCase):
    def test_one_nominal_attempt_preserves_models_and_failure(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            sources = [root/name for name in ('base.json', 'jaw.json', 'clock.json')]
            for path, value in zip(sources, ({'base': 1}, {'jaw': 1}, {'blind': False})):
                path.write_text(json.dumps(value))
            assets = root/'assets'; assets.mkdir()
            destination = root/'installed'
            stack.enter_context(patch('dvidia_training.piper_grip_clock_model.validate_model', return_value={'blind': False}))
            stack.enter_context(patch('dvidia_training.piper_grip_clock_benchmark._documents', return_value=({}, {}, {})))
            stack.enter_context(patch('dvidia_training.piper_grip_clock_benchmark.BASE_SHA', sha256(sources[0].read_bytes()).hexdigest()))
            stack.enter_context(patch('dvidia_training.piper_grip_clock_benchmark.jaw_contract', return_value=('coordinated_reference', {})))
            stack.enter_context(patch('dvidia_training.piper_task._asset_root', return_value=assets))
            stack.enter_context(patch('dvidia_training.robot_profiles.verify_assets', return_value={'verified': True}))
            def execute(study, actual_assets, case, variant, folder, **kwargs):
                self.assertEqual(actual_assets, assets)
                self.assertEqual(variant, 'clock')
                self.assertEqual(case['condition'], 'nominal')
                self.assertFalse(kwargs['collect_labels'])
                self.assertEqual(kwargs['model_path'].read_bytes(), sources[2].read_bytes())
                self.assertEqual(read(study/'protocol.json')['cases']['train'], [])
                write(folder/'receipt.json', {'files': {'score.json': 'a'*64}})
                return dict(valid=True, success=False, reason='native_outcome_failed')
            runner = stack.enter_context(patch('dvidia_training.piper_grip_clock_benchmark.execute', side_effect=execute))
            result = skill.run_clock_skill(*sources, assets, destination)
            self.assertTrue(result['valid']); self.assertFalse(result['success'])
            self.assertFalse(result['teacher_available']); self.assertFalse(result['physical_robot_ready'])
            self.assertEqual(result['teacher_fallback_calls'], 0)
            self.assertTrue((destination/'receipt.json').is_file())
            with self.assertRaisesRegex(ValueError, 'fresh directory'):
                skill.run_clock_skill(*sources, assets, destination)
            self.assertEqual(runner.call_count, 1)

    def test_bad_workspace_and_non_model_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'output'
            with self.assertRaises(ValueError):
                skill.run_clock_skill(root/'a', root/'b', root/'c', root, output, start_x=.4)
            self.assertFalse(output.exists())
            for name in ('a.json', 'b.json', 'c.json'):
                (root/name).write_text('[]')
            with self.assertRaises(ValueError):
                skill.run_clock_skill(root/'a.json', root/'b.json', root/'c.json', root, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
