"""Test additive publication without credentials, network or native simulation."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_asset_library as prepare
import publish_robot_sim as publisher
from dvidia_training import asset_benchmark, asset_library

class AssetLibraryPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        self.space=self.root/'base/space'
        self.space.mkdir(parents=True)
        self.original={'README.md':b'---\nsdk: static\napp_file: index.html\nlicense: mit\n---\nPrior card\n',
                       'index.html':b'<section aria-labelledby="replay-title">Prior replay</section>',
                       'provenance.json':b'{"prior":"kept"}', 'LICENSE':b'MIT',
                       'tasks/prior/report.json':b'{"all_failures_retained":true}',
                       'replay/geometry.bin':b'original-native-geometry'}
        for name,raw in self.original.items():
            path=self.space/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
        self.plan=self.root/'base/publication-plan.json'
        self.plan.write_bytes(prepare.encoded({'kind':publisher.KIND,'schema_version':1,'namespace':'Dvidia','private':False,
             'source_commit':'a'*40,'expected_remote_commit':'b'*40,'remove_files':[],
             'repository':{'repo_id':publisher.REPO_ID,'repo_type':'space','space_sdk':'static','directory':'space',
                'files':[{'path':n,**prepare.record(b)} for n,b in self.original.items()]},
             'total_bytes':sum(map(len,self.original.values()))}))
        self.library=self.root/'library'
        self.library.mkdir()
        (self.library/'index.html').write_bytes(b'Asset library')
        (self.library/'library-integrity.json').write_bytes(b'{}')
        guard=patch.object(publisher,'client',side_effect=AssertionError('No credentials'))
        guard.start();self.addCleanup(guard.stop)

    def stage(self, output='output'):
        return prepare.stage(self.plan,self.library,self.root/output,'c'*40,'d'*40)

    def test_addition_preserves_every_prior_result_and_exact_parent(self):
        with patch.object(prepare,'check_library',return_value={'source_commit':'c'*40}):self.stage()
        output=self.root/'output/space'
        for name,raw in self.original.items():
            if name not in {'README.md','index.html','provenance.json'}:
                self.assertEqual((output/name).read_bytes(),raw)
        self.assertEqual(json.loads((output/'provenance.json').read_bytes())['prior'],'kept')
        self.assertIn(b'asset-library/v01/index.html',(output/'index.html').read_bytes())
        result=publisher.checked_plan(self.root/'output/publication-plan.json')
        self.assertEqual(result['expected_remote_commit'],'d'*40)
        self.assertEqual(result['remove_files'],[])

    def test_rejects_tampered_prior_bytes_without_creating_output(self):
        (self.space/'tasks/prior/report.json').write_bytes(b'changed')
        with self.assertRaises(publisher.PublicationError):self.stage()
        self.assertFalse((self.root/'output').exists())

    def test_rejects_library_source_mismatch_before_output(self):
        with patch.object(prepare,'check_library',return_value={'source_commit':'f'*40}),self.assertRaises(ValueError):
            self.stage()
        self.assertFalse((self.root/'output').exists())

    def test_reader_rejects_hidden_files_private_paths_and_symlinks(self):
        for name,raw in [('.env',b'data'),('secret.txt',b'/Users/roguedev/private')]:
            path=self.library/name;path.write_bytes(raw)
            with self.assertRaises(publisher.PublicationError):prepare.read_payload(self.library)
            path.unlink()
        (self.library/'redirect.json').symlink_to(self.library/'library-integrity.json')
        with self.assertRaises(publisher.PublicationError):prepare.read_payload(self.library)

    def test_reader_and_publisher_enforce_file_and_byte_budgets(self):
        self.assertEqual(publisher.MAX_FILES,900)
        with patch.object(prepare,'MAX_LIBRARY_FILES',1),self.assertRaises(ValueError):prepare.read_payload(self.library)
        with patch.object(prepare,'MAX_LIBRARY_BYTES',1),self.assertRaises(ValueError):prepare.read_payload(self.library)
        with patch.object(publisher,'MAX_FILES',5),self.assertRaises(publisher.PublicationError):publisher.checked_plan(self.plan)

    def runtime_fixture(self):
        (self.root/'docs').mkdir()
        (self.root/'docs/asset-library.md').write_text('Synthetic scene fixture; no learned policy.\n')
        (self.root/'LICENSE').write_text('MIT fixture\n')
        source_root=patch.object(prepare,'ROOT',self.root)
        source_root.start()
        self.addCleanup(source_root.stop)
        benchmark=self.root/'benchmark'
        benchmark.mkdir()
        protocol=asset_benchmark.make_protocol(source_commit='c'*40)
        (benchmark/'protocol.json').write_bytes(prepare.encoded(protocol))
        previews={}
        for robot in prepare.ROBOTS:
            path=self.root/'previews'/robot
            path.mkdir(parents=True)
            (path/'scene.json').write_bytes(prepare.encoded(asset_library.scene_config(robot)))
            previews[robot]=path
        return benchmark,previews

    @staticmethod
    def inspect_fixture_preview(path):
        return {'robot':json.loads((Path(path)/'scene.json').read_bytes())['robot']}

    def reseal_library(self, output):
        files={path.relative_to(output).as_posix():path.read_bytes()
               for path in output.rglob('*') if path.is_file() and path.name!='library-integrity.json'}
        (output/'library-integrity.json').write_bytes(prepare.encoded({
            'schema':prepare.SCHEMA,'files':{name:prepare.record(raw) for name,raw in files.items()}}))

    def test_bundle_rejects_swapped_robot_previews_before_output(self):
        benchmark,previews=self.runtime_fixture()
        previews['piper']=previews['ur5e']
        with patch('dvidia_training.asset_benchmark.inspect_benchmark'), \
             patch('dvidia_training.asset_preview.inspect_preview',side_effect=self.inspect_fixture_preview):
            with self.assertRaisesRegex(ValueError,'different robot'):
                prepare.bundle(benchmark,previews,self.root/'wrong-robot','c'*40)
        self.assertFalse((self.root/'wrong-robot').exists())

    def test_bundle_rejects_wrong_source_or_modified_runtime_before_output(self):
        benchmark,previews=self.runtime_fixture()
        path=benchmark/'protocol.json'
        original=json.loads(path.read_bytes())
        mutations=[{**original,'source_commit':'f'*40},
                   {**original,'implementation_sha256':{**original['implementation_sha256'],'asset_library.py':'0'*64}}]
        for index,protocol in enumerate(mutations):
            path.write_bytes(prepare.encoded(protocol))
            destination=self.root/('wrong-source-'+str(index))
            with self.subTest(index=index), patch('dvidia_training.asset_benchmark.inspect_benchmark'), \
                 patch('dvidia_training.asset_preview.inspect_preview',side_effect=self.inspect_fixture_preview):
                with self.assertRaisesRegex(ValueError,'frozen benchmark'):
                    prepare.bundle(benchmark,previews,destination,'c'*40)
            self.assertFalse(destination.exists())

    def test_explicit_viewer_revision_keeps_frozen_benchmark_revision(self):
        benchmark,previews=self.runtime_fixture()
        output=self.root/'corrected-viewer'
        with patch('dvidia_training.asset_benchmark.inspect_benchmark'), \
             patch('dvidia_training.asset_preview.inspect_preview',side_effect=self.inspect_fixture_preview):
            prepare.bundle(benchmark,previews,output,'d'*40,benchmark_source_commit='c'*40)
            provenance=prepare.check_library(output)
            self.assertEqual(provenance['source_commit'],'d'*40)
            self.assertEqual(provenance['benchmark_source_commit'],'c'*40)

    def test_check_library_rejects_resealed_preview_robot_mismatch(self):
        benchmark,previews=self.runtime_fixture()
        output=self.root/'reviewed-library'
        with patch('dvidia_training.asset_benchmark.inspect_benchmark'), \
             patch('dvidia_training.asset_preview.inspect_preview',side_effect=self.inspect_fixture_preview):
            prepare.bundle(benchmark,previews,output,'c'*40)
            self.assertEqual(prepare.check_library(output)['source_commit'],'c'*40)
            (output/'previews/piper/scene.json').write_bytes(
                prepare.encoded(asset_library.scene_config('ur5e')))
            self.reseal_library(output)
            with self.assertRaisesRegex(ValueError,'different robot'):
                prepare.check_library(output)

    def test_check_library_rejects_resealed_source_or_runtime_mismatch(self):
        benchmark,previews=self.runtime_fixture()
        output=self.root/'reviewed-library'
        with patch('dvidia_training.asset_benchmark.inspect_benchmark'), \
             patch('dvidia_training.asset_preview.inspect_preview',side_effect=self.inspect_fixture_preview):
            prepare.bundle(benchmark,previews,output,'c'*40)
            path=output/'benchmark/protocol.json'
            original=json.loads(path.read_bytes())
            mutations=[{**original,'source_commit':'f'*40},
                       {**original,'implementation_sha256':{**original['implementation_sha256'],'asset_catalog.json':'0'*64}}]
            for index,protocol in enumerate(mutations):
                path.write_bytes(prepare.encoded(protocol))
                self.reseal_library(output)
                with self.subTest(index=index), self.assertRaisesRegex(ValueError,'frozen benchmark'):
                    prepare.check_library(output)

if __name__=='__main__':unittest.main()
