"""Fresh-process coverage for the existing Terminal entry point."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class CameraCacheCliTests(unittest.TestCase):
    def command(self, *arguments):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'),
                   PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run(
            [sys.executable, '-c', 'from dvidia_training.footage_pipeline import main; main()', *arguments],
            env=env, capture_output=True, text=True, timeout=20)

    def test_main_help_exposes_cache_and_existing_footage_commands(self):
        result = self.command('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ('cache', 'prepare', 'run', 'inspect'):
            self.assertIn(name, result.stdout)

    def test_cache_source_can_be_pinned_from_existing_terminal_command(self):
        result = self.command('cache', 'source')
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(len(value['source_sha256']), 64)
        self.assertEqual(len(value['source_identity']['runtime']), 19)

    def test_cache_hit_requires_external_pins(self):
        result = self.command('cache', 'load', '/private/tmp/not-a-dvidia-cache')
        self.assertEqual(result.returncode, 2)
        for flag in ('--manifest-sha256', '--protocol-sha256', '--split', '--source-sha256'):
            self.assertIn(flag, result.stderr)

    def test_original_robot_command_keeps_its_stages(self):
        result = self.command('robot', '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ('prepare', 'collect', 'train', 'evaluate', 'inspect', 'package'):
            self.assertIn(name, result.stdout)


if __name__ == '__main__':
    unittest.main()
