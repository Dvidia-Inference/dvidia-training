"""Study isolation and refusal behavior; no native experiment hidden in tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dvidia_training import piper_probe_benchmark as study


class ProbeStudyTests(unittest.TestCase):
    def test_existing_output_is_not_overwritten_or_resumed(self):
        with TemporaryDirectory() as folder, patch.object(study, 'make_protocol') as protocol:
            marker = Path(folder)/'original.txt'
            marker.write_text('preserve this failed attempt')
            with self.assertRaises(ValueError):
                study.prepare(folder, 'unused-assets')
            self.assertEqual(marker.read_text(), 'preserve this failed attempt')
            protocol.assert_not_called()

    def test_changed_frozen_roster_refuses_before_any_native_attempt(self):
        original = {'schema':study.SCHEMA, 'planned_attempts':24, 'cases':[{'id':'original'}]}
        changed = {**original, 'planned_attempts':23}
        with TemporaryDirectory() as folder, patch.object(study, 'make_protocol', return_value=original), \
                patch.object(study, '_attempt') as attempt:
            study.write(Path(folder)/'protocol.json', changed)
            with self.assertRaises(ValueError):
                study.run(folder, 'unused-assets')
            attempt.assert_not_called()
            self.assertFalse((Path(folder)/'started.json').exists())

    def test_started_study_refuses_second_launch(self):
        protocol = {'schema':study.SCHEMA, 'cases':[], 'planned_attempts':24}
        with TemporaryDirectory() as folder, patch.object(study, 'make_protocol', return_value=protocol), \
                patch.object(study, '_attempt') as attempt:
            study.write(Path(folder)/'protocol.json', protocol)
            study.write(Path(folder)/'started.json', {'original':'retained'})
            with self.assertRaises(FileExistsError):
                study.run(folder, 'unused-assets')
            attempt.assert_not_called()
            self.assertEqual(study.read(Path(folder)/'started.json'), {'original':'retained'})

    def test_input_file_bounds_and_symlink_refused(self):
        with TemporaryDirectory() as folder:
            data = Path(folder)/'data.json'; data.write_text('{"x":1}')
            alias = Path(folder)/'alias.json'; alias.symlink_to(data)
            for path, limit in ((alias, 100), (data, 2)):
                with self.assertRaises(ValueError): study.read(path, limit)


if __name__ == '__main__':
    unittest.main()
