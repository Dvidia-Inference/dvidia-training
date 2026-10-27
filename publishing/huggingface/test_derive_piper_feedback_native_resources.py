"""Fixed, SRI-bound literal data scripts preserve the complete replay study."""
import base64
import gzip
from hashlib import sha256, sha384
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import derive_piper_feedback_native_resources as resources


def fixture(source):
    canonical = Path(__file__).resolve().parents[2]/'src/dvidia_training/piper_feedback_viewer.js'
    variants = ['feedback', 'blind', 'feedforward', 'teacher', 'open_jaw']
    attempts = []
    for i in range(24):
        for variant in variants:
            case_id = f'confirmation-{i:02}-goal0-nominal'
            attempts.append(dict(id=case_id, variant=variant,
                evidence={'native.npz':resources.hosting.OLD_NATIVE_ROOT+f'confirmation/{case_id}/{variant}/native.npz',
                          'native_npz_sha256':'1'*64, 'native_npz_bytes':1}))
    selected = [r for r in attempts if r['variant']=='feedback']+[next(r for r in attempts if r['variant']==v) for v in variants[1:]]
    files = {'piper_feedback_viewer.js':canonical.read_bytes(), 'piper_feedback_viewer.bundle.js':b'parent bundle',
             'index.html':b'<html><head></head><body><script src="piper_feedback_viewer.bundle.js" defer></script></body></html>',
             'geometry.bin':b'original geometry', 'report.json':b'{"feedback":6,"blind":8}'}
    files['geometry.bin.gz'] = gzip.compress(files['geometry.bin'],mtime=0)
    cases = []
    for i,row in enumerate(selected):
        row = json.loads(json.dumps(row)); row['frame_count']=1
        for key,name,raw in [('rows_file',f'frames-{i:02}.json',b'[{"t_seconds":0}]'),('pose_binary',f'poses-{i:02}.bin',bytes([i])*28)]:
            files[name]=raw;row[key]=dict(href=name,**resources.hosting._receipt(raw))
            if key=='pose_binary':
                files[name+'.gz']=gzip.compress(raw,mtime=0)
                row[key].update(compressed_href=name+'.gz',compressed_bytes=len(files[name+'.gz']),compression='gzip')
        cases.append(row)
    files['scene.json']=resources.hosting.study._encoded(dict(geometry_binary=dict(href='geometry.bin',**resources.hosting._receipt(files['geometry.bin']))))
    files['motion.json']=resources.hosting.study._encoded(dict(schema=resources.hosting.study.VIEWER_SCHEMA,cases=cases,all_attempts=attempts,
        native_dataset=dict(repo_id=resources.hosting.study.DATASET_REPO,revision='main')))
    for name,raw in files.items():(source/name).write_bytes(raw)
    parent=dict(schema=resources.hosting.study.VIEWER_SCHEMA,files={name:resources.hosting._receipt(raw) for name,raw in files.items()})
    seal=resources.hosting.study._encoded(parent);(source/'viewer-integrity.json').write_bytes(seal)
    return files,sha256(seal).hexdigest()


class NativeResourceTests(unittest.TestCase):
    def test_literal_wrappers_and_expansion_bounds(self):
        row=resources.resource('poses-01.bin',b'original saved pose')
        payload=dict(schema=resources.PAYLOAD_SCHEMA,id='replay-01',resources=[row,row])
        wrapper=resources.script(payload)
        prefix=b'globalThis["__dvidiaPiperFeedbackResourcesV1"]("replay-01",'
        self.assertTrue(wrapper.startswith(prefix));self.assertTrue(wrapper.endswith(b');\n'))
        self.assertEqual(json.loads(wrapper[len(prefix):-3]),payload)
        for invalid in [dict(payload,id='../other'),dict(payload,extra=True),dict(payload,resources=[]),dict(payload,resources=[dict(row,data='A'*1048576),row])]:
            with self.assertRaises(resources.hosting.PublicationError):resources.script(invalid)
        for name,raw,packed in [('other.js',b'1',None),('geometry.bin',b'',None),('geometry.bin',b'1',b'bad gzip'),
                                ('geometry.bin',b'1',gzip.compress(b'1'*1000000))]:
            with self.assertRaises(resources.hosting.PublicationError):resources.resource(name,raw,packed)

    def test_derived_projection_preserves_originals_and_embeds_exact_external_motion_with_optional_pin(self):
        for revision in (None,'a'*40):
            with self.subTest(revision=revision),TemporaryDirectory() as directory:
                source,output=Path(directory)/'source',Path(directory)/'output';source.mkdir()
                files,seal=fixture(source);before={p.name:p.read_bytes() for p in source.iterdir()}
                def build(arguments,check):Path(arguments[-1]).write_bytes(b'compiled native resource fixture')
                with patch.object(resources.subprocess,'run',side_effect=build):
                    result=resources.derive(source,output,expected_viewer_sha256=seal,dataset_revision=revision)
                self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
                self.assertEqual(result['selected_case_scripts'],27)
                source_js=(output/'piper_feedback_viewer.js').read_text()
                offset=source_js.index('const BROWSER_RESOURCES = ')+len('const BROWSER_RESOURCES = ')
                manifest,_=json.JSONDecoder().raw_decode(source_js[offset:])
                embedded_motion=gzip.decompress(base64.b64decode(manifest['embedded']['motion.json']['data'],validate=True))
                self.assertEqual(embedded_motion,(output/'motion.json').read_bytes())
                for name in ('scene.json','geometry.bin'):
                    self.assertEqual(gzip.decompress(base64.b64decode(manifest['embedded'][name]['data'],validate=True)),files[name])
                if revision:
                    pinned=json.loads(embedded_motion);self.assertEqual(pinned['native_dataset']['revision'],revision)
                    self.assertTrue(all('/resolve/'+revision+'/' in r['evidence']['native.npz'] for r in pinned['cases']+pinned['all_attempts']))
                else:self.assertEqual(embedded_motion,files['motion.json'])
                for descriptor in manifest['cases']:
                    if descriptor['script'] is None:payload=manifest['first']
                    else:
                        raw=(output/descriptor['script']['href']).read_bytes()
                        self.assertLessEqual(len(raw),1048576)
                        self.assertEqual('sha384-'+base64.b64encode(sha384(raw).digest()).decode(),descriptor['script']['integrity'])
                        payload=json.loads(raw.split(b',',1)[1][:-3])
                    for item in payload['resources']:
                        self.assertEqual(gzip.decompress(base64.b64decode(item['data'],validate=True)),files[item['href']])
                allowed=resources.CHANGED|({'motion.json'} if revision else set())
                for name,raw in files.items():
                    if name not in allowed:self.assertEqual((output/name).read_bytes(),raw)
                integrity=json.loads((output/'viewer-integrity.json').read_bytes())
                for name,receipt in integrity['files'].items():self.assertEqual(resources.hosting._receipt((output/name).read_bytes()),receipt)
                html=(output/'index.html').read_text();self.assertIn("script-src 'self'",html);self.assertIn('integrity="sha384-',html)
                self.assertIn('crossorigin="anonymous"',html);self.assertIn('referrerpolicy="no-referrer"',html)
                bundle=(output/'piper_feedback_viewer.bundle.js').read_bytes()
                self.assertIn('src="piper_feedback_viewer.bundle.js?sha256='+sha256(bundle).hexdigest()+'"',html)
                with self.assertRaises(resources.hosting.PublicationError):resources.derive(source,Path(directory)/'invalid',expected_viewer_sha256=seal,dataset_revision='main')


if __name__=='__main__':unittest.main()
