"""Hosting-only geometry derivation preserves original scientific bytes."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import derive_piper_feedback_geometry_chunks as chunks


class GeometryDerivationTests(unittest.TestCase):
    def test_fresh_projection_preserves_original_geometry_evidence_and_mesh_inventory(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory)/'original', Path(directory)/'derived'
            source.mkdir()
            geometry = b'01234567'*70000
            scene = {'schema':'dvidia.piper-feedback-scene/v0.2','meshes':[{'id':0,'vertex_count':31,'index_count':42}],
                     'geometry_binary':dict(href='geometry.bin',bytes=len(geometry),sha256=sha256(geometry).hexdigest())}
            root=Path(__file__).resolve().parents[2]
            files={'geometry.bin':geometry,'scene.json':chunks.hosting.study._encoded(scene),
                   'piper_feedback_viewer.js':(root/'src/dvidia_training/piper_feedback_viewer.js').read_bytes(),
                   'piper_feedback_viewer.bundle.js':b'original bundle fixture',
                   'report.json':b'{"fixture":"scientific evidence unchanged"}'}
            for name,raw in files.items():(source/name).write_bytes(raw)
            seal={'schema':chunks.hosting.study.VIEWER_SCHEMA,'files':{name:chunks.hosting._receipt(raw) for name,raw in files.items()}}
            seal_raw=chunks.hosting.study._encoded(seal);(source/'viewer-integrity.json').write_bytes(seal_raw)
            before={p.name:p.read_bytes() for p in source.iterdir()}
            def fake_build(arguments,check):
                Path(arguments[-1]).write_bytes(b'derived compiled fixture; native build tested separately')
            with patch.object(chunks.subprocess,'run',side_effect=fake_build):
                result=chunks.derive(source,output,expected_viewer_sha256=sha256(seal_raw).hexdigest())
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            self.assertTrue(result['scientific_evidence_unchanged'])
            self.assertEqual((output/'geometry.bin').read_bytes(),geometry)
            self.assertEqual((output/'report.json').read_bytes(),files['report.json'])
            derived=json.loads((output/'scene.json').read_bytes())
            self.assertEqual(derived['meshes'],scene['meshes'])
            self.assertEqual(derived['geometry_binary']['sha256'],scene['geometry_binary']['sha256'])
            joined=b''.join((output/part['href']).read_bytes() for part in derived['geometry_binary']['chunks'])
            self.assertEqual(joined,geometry)
            self.assertTrue(all(part['bytes']<=262144 for part in derived['geometry_binary']['chunks']))
            final=json.loads((output/'viewer-integrity.json').read_bytes())
            for name,receipt in final['files'].items():
                self.assertEqual(chunks.hosting._receipt((output/name).read_bytes()),receipt)
            self.assertEqual(final['geometry_transport_derivation']['parent_viewer_integrity_sha256'],sha256(seal_raw).hexdigest())


if __name__=='__main__':unittest.main()
