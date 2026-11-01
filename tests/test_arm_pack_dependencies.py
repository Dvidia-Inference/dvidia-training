"""Adapter archives retain only their executable closure; no physics is run."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from dvidia_training import arm_pack
from dvidia_training.capsule_installer import EXTRA_RUNTIME_FILES
from dvidia_training.installer import RUNTIME_SOURCE_FILES


class ArmPackDependenciesTests(unittest.TestCase):
    def test_all_relative_imports_and_recorded_runtime_are_included(self):
        root = Path(arm_pack.__file__).parent
        included = set(arm_pack.BUNDLE_SOURCE_FILES)
        self.assertTrue(set(RUNTIME_SOURCE_FILES).issubset(included))
        self.assertTrue(set(EXTRA_RUNTIME_FILES).issubset(included))
        self.assertTrue({'studio.py', 'arm_qualify.py'}.issubset(included))
        for filename in included:
            for node in ast.walk(ast.parse((root/filename).read_text())):
                if isinstance(node, ast.ImportFrom) and node.level:
                    self.assertEqual(node.level, 1, filename)
                    names = [node.module] if node.module else [entry.name for entry in node.names]
                    for name in names:
                        self.assertIn(name.split('.')[0]+'.py', included, filename)

    def test_unrelated_installed_modules_do_not_expand_archive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            required = (*arm_pack.BUNDLE_SOURCE_FILES, *arm_pack.BUNDLE_RESOURCE_FILES)
            for name in required:
                (root/name).write_text(name)
            for index in range(150):
                (root/f'unrelated_{index}.py').write_text('')
            files = arm_pack._bundle_files(root)
            self.assertEqual(set(files), {'simlab/'+name for name in required})
            self.assertLess(len(files)+6, 128)
            (root/'arm_benchmark_summary.json').write_text('{}')
            self.assertEqual(arm_pack._bundle_files(root)['simlab/arm_benchmark_summary.json'], b'{}')
            (root/'studio.html').unlink()
            with self.assertRaisesRegex(ValueError, 'dependency is absent'):
                arm_pack._bundle_files(root)

    def test_symlink_dependency_is_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (*arm_pack.BUNDLE_SOURCE_FILES, *arm_pack.BUNDLE_RESOURCE_FILES):
                (root/name).write_text(name)
            (root/'studio.html').unlink()
            (root/'studio.html').symlink_to(root/'LICENSE')
            with self.assertRaisesRegex(ValueError, 'not an ordinary file'):
                arm_pack._bundle_files(root)

    def test_export_round_trip_retains_exact_runtime_with_small_archive(self):
        raw = (Path(__file__).with_name('fixtures')/'arm_place_skillspace.json').read_bytes()
        skill = arm_pack.load_skillspace(raw)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {'task': 'ArmPickPlace-v0', 'scope': 'simulation-only',
                'runtime': arm_pack.runtime_contract(), 'skill': skill.to_dict(),
                'summary': {'successes': 0, 'episodes': 0}, 'episodes': [],
                'limitations': ['Synthetic packaging unit test; no physics executed.']}
            (root/'run.json').write_text(json.dumps(payload))
            (root/'source.json').write_bytes(raw)
            manifest = arm_pack.export_arm_pack(root)
            self.assertEqual(arm_pack.verify_arm_pack(root/'skill-adapter.zip'), manifest)
            self.assertEqual(manifest['runtime'], payload['runtime'])
            with ZipFile(root/'skill-adapter.zip') as archive:
                self.assertLess(len(archive.namelist()), 40)
                self.assertIn('simlab/studio.html', archive.namelist())
                self.assertNotIn('simlab/footage_pipeline.py', archive.namelist())

    def test_existing_archive_entry_budget_is_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'oversized.zip'
            with ZipFile(path, 'w') as archive:
                for index in range(129):
                    archive.writestr(str(index), b'')
            with self.assertRaisesRegex(ValueError, 'resource budget exceeded'):
                arm_pack.verify_arm_pack(path)


if __name__ == '__main__':
    unittest.main()
