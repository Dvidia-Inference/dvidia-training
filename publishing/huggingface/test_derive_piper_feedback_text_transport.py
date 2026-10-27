"""The text-only hosting projection preserves every underlying evidence byte."""
import base64
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import derive_piper_feedback_text_transport as text
import derive_piper_feedback_transport_diagnostics as diagnostics


class TextProjectionTests(unittest.TestCase):
    def test_envelopes_are_canonical_exact_and_bounded(self):
        for raw in (b'1', b'12', b'123', bytes(range(256))*1024):
            packed = text.envelope(raw)
            document = json.loads(packed)
            self.assertLessEqual(len(packed), 1048576)
            self.assertEqual(base64.b64decode(document['data'], validate=True), raw)
            self.assertEqual(document['sha256'], sha256(raw).hexdigest())
            self.assertEqual(packed, (json.dumps(document, sort_keys=True, separators=(',', ':'))+'\n').encode())
        for raw in (b'', b'x'*(text.MAX_RAW+1), b'x'*text.MAX_RAW):
            with self.assertRaises(text.hosting.PublicationError):
                text.envelope(raw)

    def test_complete_projection_preserves_parent_and_raw_scientific_files(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory)/'source', Path(directory)/'derived'
            source.mkdir()
            files = {'report.json': b'{"scientific":"unchanged"}', 'geometry.bin': b'raw geometry fixture',
                     'piper_feedback_viewer.bundle.js': b'compiled parent fixture'}
            pieces, cases = [], []
            for i in range(26):
                name = f'geometry-part-{i:03d}.bin'; raw = bytes([i, i+1, i+2])
                files[name] = raw; files[name+'.gz'] = b'original gzip fixture '+raw
                pieces.append(dict(href=name, **text.hosting._receipt(raw)))
            for i in range(28):
                name = f'poses-{i:02d}.bin'; raw = bytes([i])*12
                files[name] = raw; files[name+'.gz'] = b'original gzip fixture '+raw
                cases.append(dict(pose_binary=dict(href=name, **text.hosting._receipt(raw))))
            scene = dict(meshes=[dict(id=0, vertex_count=3, index_count=3)],
                         geometry_binary=dict(sha256=sha256(files['geometry.bin']).hexdigest(), chunks=pieces))
            motion = dict(cases=cases, all_attempts=[dict(index=i) for i in range(120)])
            files['scene.json'] = text.hosting.study._encoded(scene)
            files['motion.json'] = text.hosting.study._encoded(motion)
            canonical = Path(__file__).resolve().parents[2]/'src/dvidia_training/piper_feedback_viewer.js'
            files['piper_feedback_viewer.js'] = diagnostics.patched_source(text.geometry.patched_source(canonical.read_bytes()))
            for name, raw in files.items(): (source/name).write_bytes(raw)
            parent = dict(schema=text.hosting.study.VIEWER_SCHEMA,
                          files={name:text.hosting._receipt(raw) for name,raw in files.items()})
            seal = text.hosting.study._encoded(parent)
            (source/'viewer-integrity.json').write_bytes(seal)
            before = {p.name:p.read_bytes() for p in source.iterdir()}
            def fake_build(arguments, check): Path(arguments[-1]).write_bytes(b'compiled derived fixture')
            with patch.object(text.subprocess, 'run', side_effect=fake_build):
                result = text.derive(source, output, expected_viewer_sha256=sha256(seal).hexdigest())
            self.assertEqual(result['envelopes'], 54)
            self.assertEqual(before, {p.name:p.read_bytes() for p in source.iterdir()})
            for name, raw in files.items():
                if name not in text.CHANGED: self.assertEqual((output/name).read_bytes(), raw)
            actual_scene, actual_motion = (json.loads((output/name).read_bytes()) for name in ('scene.json','motion.json'))
            self.assertEqual(actual_scene['meshes'], scene['meshes'])
            self.assertEqual(actual_scene['geometry_binary']['sha256'], scene['geometry_binary']['sha256'])
            for descriptor in actual_scene['geometry_binary']['chunks']+[c['pose_binary'] for c in actual_motion['cases']]:
                envelope = descriptor.pop('json_envelope')
                packed = (output/envelope['href']).read_bytes()
                self.assertEqual(text.hosting._receipt(packed), {k:envelope[k] for k in ('bytes','sha256')})
                self.assertEqual(base64.b64decode(json.loads(packed)['data'], validate=True), files[descriptor['href']])
            self.assertEqual(actual_scene, scene)
            self.assertEqual(actual_motion, motion)
            derived = json.loads((output/'viewer-integrity.json').read_bytes())
            for name, receipt in derived['files'].items():
                self.assertEqual(text.hosting._receipt((output/name).read_bytes()), receipt)
            with self.assertRaises(text.hosting.PublicationError):
                text.derive(source, output, expected_viewer_sha256=sha256(seal).hexdigest())


if __name__ == '__main__': unittest.main()
