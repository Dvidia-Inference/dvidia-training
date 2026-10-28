"""Bounded offline payload checks; these tests do not import or run physics."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from publishing import prepare_piper_grip_clock_v03 as prepare


class GripClockPayloadTests(unittest.TestCase):
    def test_integrity_manifest_hashes_every_existing_file(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'model'
            root.mkdir()
            (root/'README.md').write_text('card')
            (root/'selected.json').write_text('{}')
            count, total = prepare._payload_integrity(root, 'model')
            manifest = json.loads((root/'integrity.json').read_text())
            self.assertEqual(count, 3)
            self.assertEqual(manifest['schema'], prepare.SCHEMA)
            self.assertEqual(manifest['payload'], 'model')
            self.assertEqual(set(manifest['files']), {'README.md', 'selected.json'})
            self.assertEqual(manifest['files']['README.md']['sha256'], prepare._hash(root/'README.md'))
            self.assertGreater(total, 0)

    def test_preparation_rejects_a_report_without_the_frozen_confirmation_roster(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'study'
            root.mkdir()
            audit = Path(temporary)/'audit.json'
            audit.write_text('{}')
            cases = {'train': [], 'development': [], 'confirmation': [{'id': f'case-{i}'} for i in range(128)]}
            protocol = {'schema': prepare.STUDY_SCHEMA, 'cases': cases,
                        'variants': list(prepare.VARIANTS), 'source_sha256': {}, 'physical_robot_ready': False}
            frozen = {'lifecycle_source_sha256': {}, 'physical_robot_ready': False}
            for name, value in (('protocol.json', protocol), ('frozen.json', frozen), ('report.json', {})):
                (root/name).write_text(json.dumps(value))
            with patch.object(prepare.lifecycle, '_frozen', return_value=(protocol, frozen)), \
                 patch.object(prepare.lifecycle, 'make_cases', return_value=cases), \
                 self.assertRaisesRegex(ValueError, 'complete, bound 512-attempt confirmation'):
                prepare._verify_inputs(root, audit)

    def test_payload_builder_refuses_a_preexisting_destination(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)/'payload'
            output.mkdir()
            study = Path(temporary)/'study'
            study.mkdir()
            guide, host, lineage = (Path(temporary)/name for name in ('guide.md', 'host.json', 'lineage.md'))
            guide.write_text('guide')
            host.write_text('{}')
            lineage.write_text('lineage')
            with patch.object(prepare, '_verify_inputs', return_value=(Path(temporary), Path(temporary)/'audit', {}, {}, {}, {})), \
                 self.assertRaisesRegex(ValueError, 'fresh payload directory'):
                prepare.prepare(study, Path(temporary)/'missing-audit', output,
                                guide=guide, host=host, lineage=lineage)


if __name__ == '__main__':
    unittest.main()
