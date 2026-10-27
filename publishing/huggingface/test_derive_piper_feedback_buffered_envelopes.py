"""Native buffering is an isolated, immutable hosting compatibility experiment."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import derive_piper_feedback_buffered_envelopes as buffered
import derive_piper_feedback_text_transport as text
import derive_piper_feedback_transport_diagnostics as diagnostics


class BufferedProjectionTests(unittest.TestCase):
    def test_derived_source_only_preserves_exact_parent_data_and_provenance(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory)/'source', Path(directory)/'derived'
            source.mkdir()
            canonical = Path(__file__).resolve().parents[2]/'src/dvidia_training/piper_feedback_viewer.js'
            js = text.patched_source(diagnostics.patched_source(buffered.geometry.patched_source(canonical.read_bytes())))
            files = {'piper_feedback_viewer.js':js, 'piper_feedback_viewer.bundle.js':b'parent bundle',
                     'scene.json':b'{"original_geometry_sha":"unchanged"}', 'motion.json':b'{"all_attempts":120}',
                     'poses-00.bin.json':text.envelope(b'original saved pose'),
                     'report.json':b'{"feedback":6,"blind":8}'}
            for name, raw in files.items(): (source/name).write_bytes(raw)
            parent = dict(schema=buffered.hosting.study.VIEWER_SCHEMA,
                          text_transport_derivation=dict(schema='dvidia.piper-feedback-text-transport/v1',
                                                         original_scientific_viewer_integrity_sha256='1'*64),
                          files={name:buffered.hosting._receipt(raw) for name,raw in files.items()})
            seal = buffered.hosting.study._encoded(parent); (source/'viewer-integrity.json').write_bytes(seal)
            before = {p.name:p.read_bytes() for p in source.iterdir()}
            def build(arguments, check): Path(arguments[-1]).write_bytes(b'new compiled fixture')
            with patch.object(buffered.subprocess, 'run', side_effect=build):
                result = buffered.derive(source, output, expected_viewer_sha256=sha256(seal).hexdigest())
            self.assertEqual(before, {p.name:p.read_bytes() for p in source.iterdir()})
            self.assertTrue(result['original_data_and_envelopes_unchanged'])
            for name, raw in files.items():
                if name not in buffered.CHANGED: self.assertEqual((output/name).read_bytes(), raw)
            derived = json.loads((output/'viewer-integrity.json').read_bytes())
            for name, receipt in derived['files'].items():
                self.assertEqual(buffered.hosting._receipt((output/name).read_bytes()), receipt)
            note = derived['native_buffered_envelopes_derivation']
            self.assertEqual(note['parent_viewer_integrity_sha256'], sha256(seal).hexdigest())
            self.assertIn('cannot enforce a streaming allocation cap', note['bounds'])
            self.assertIn('Unproven', note['status'])
            with self.assertRaises(buffered.hosting.PublicationError):
                buffered.patched_source(js.replace(b'    const packed = await bounded(', b'    const changed = await bounded('))
            with self.assertRaises(buffered.hosting.PublicationError):
                buffered.derive(source, output, expected_viewer_sha256=sha256(seal).hexdigest())


if __name__ == '__main__': unittest.main()
