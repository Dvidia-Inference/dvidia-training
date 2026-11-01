"""Miniature packaging contracts; never run or repack the native study."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

MODULE_PATH=Path(__file__).resolve().parents[1]/'benchmarks/piper-rgbd-tracker-v01/package_evidence.py'
spec=importlib.util.spec_from_file_location('rgbd_package_test_module',MODULE_PATH)
pkg=importlib.util.module_from_spec(spec);spec.loader.exec_module(pkg)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.study=self.root/'study';self.study.mkdir()
        self.counts=patch.object(pkg,'COUNTS',{'development':1,'confirmation':1});self.counts.start();self.addCleanup(self.counts.stop)
        self.write(self.study/'protocol.json',{'test':'miniature immutable protocol'})
        sources={}
        for name in ('piper_rgbd_tracker.py','piper_rgbd_benchmark.py'):
            source=self.study/'frozen-source'/name;self.put(source,b'# frozen miniature source\n')
            sources[name]=pkg.identity(source.read_bytes())['sha256']
        self.write(self.study/'freeze.json',{'sources':sources,'protocol_sha256':pkg.identity((self.study/'protocol.json').read_bytes())['sha256']})
        self.write(self.study/'development-lineage.json',{'source_attempt':'test','identical_files':[]})
        self.put(self.study/'index.html',b'<p>Miniature evidence fixture</p>')
        for split,pose in (('development','pose00'),('confirmation','pose08')):
            folder=self.study/split;ident=pose+'-160-clear';files={}
            for name in pkg.FIXTURE_FILES:
                path=folder/ident/name;self.put(path,b'{}\n' if name.endswith('.json') else b'fixture bytes\n')
                files[name]=pkg.identity(path.read_bytes())
            receipt=folder/ident/'receipt.json';self.write(receipt,{'fixture':ident})
            row={'id':ident,'files':files,'receipt_sha256':pkg.identity(receipt.read_bytes())['sha256']}
            self.write(folder/'manifest.json',{'records':[row]})
            estimate=folder/'estimates'/(ident+'.json');self.write(estimate,{'estimate':ident})
            self.write(folder/'estimates'/(ident+'-started.json'),{'id':ident})
            self.write(folder/'estimates/started.json',{'started':True})
            self.write(folder/'predictions.json',{'predictions':[{'id':ident,'receipt_sha256':pkg.identity(estimate.read_bytes())['sha256']}]})
            self.write(folder/'report.json',{'scheduled':1,'rows':[{'id':ident}],
                'manifest_sha256':pkg.identity((folder/'manifest.json').read_bytes())['sha256'],
                'predictions_sha256':pkg.identity((folder/'predictions.json').read_bytes())['sha256']})
            for name in ('scheduled.json','renderer-160.json','renderer-320.json'):self.write(folder/name,{'fixture':True})
        inventory={p.relative_to(self.study).as_posix():pkg.identity(p.read_bytes()) for p in self.study.rglob('*') if p.is_file()
                   and not (p.relative_to(self.study).parts[0]=='development' and len(p.relative_to(self.study).parts)>2)}
        self.audit=self.root/'audit.json';self.proof=dict(schema='dvidia.rgbd-independent-audit/v1',passed=True,mismatch_count=0,
            mismatches=[],source_bytes_verified=True,expected_confirmation_ids=1,processed_confirmation_ids=1,evidence_inventory=inventory,
            evidence_inventory_sha256=pkg.sha256(pkg.canonical(inventory)).hexdigest())
        self.write(self.audit,self.proof)
        self.history=[]
        for label in ('baseline','earlier'):
            root=self.root/label;self.history.append(root)
            for name in ('predictions.json','report.json'):self.write(root/'development'/name,{'historical':True})
            self.put(root/'development-source-v1/piper_rgbd_tracker.py',b'# historical source\n')
        self.failed=self.root/'failed';self.write(self.failed/'development/manifest.json',{'graphics_failed':True})
        self.write(self.failed/'protocol.json',{'historical':True})
        self.card=self.root/'README.md';self.put(self.card,b'# Miniature fixture\n')
        self.output=self.root/'release'

    @staticmethod
    def put(path,raw):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)

    def write(self,path,value):self.put(path,pkg.canonical(value)+b'\n')

    def run_package(self):
        return pkg.package(self.study,self.audit,self.output,baseline=self.history[0],earlier=self.history[1],failed=self.failed,card=self.card)

    def test_package_contains_full_frames_sources_history_and_verified_payload(self):
        result=self.run_package();self.assertTrue(result['zip_readback_verified'])
        archive=self.output/result['archive'];self.assertEqual(pkg.archive_identity(archive)['sha256'],result['sha256'])
        with zipfile.ZipFile(archive) as zipped:
            self.assertIn('development/pose00-160-clear/depth.npy',zipped.namelist())
            self.assertIn('confirmation/pose08-160-clear/rgb.png',zipped.namelist())
            self.assertIn('frozen-source/piper_rgbd_tracker.py',zipped.namelist())
            self.assertIn('history/initial-development/source/piper_rgbd_tracker.py',zipped.namelist())
            self.assertIn('history/sandbox-graphics-failure/manifest.json',zipped.namelist())
            self.assertEqual(zipped.read('confirmation/report.json'),(self.output/'huggingface/report.json').read_bytes())
        self.assertEqual({p.name for p in (self.output/'huggingface').iterdir()},
            {result['archive'],'README.md','LICENSE','report.json','protocol.json','inspection.json'})

    def test_missing_or_failed_audit_refused_before_output(self):
        for proof in (dict(self.proof,passed=False),{k:v for k,v in self.proof.items() if k!='evidence_inventory'}):
            self.write(self.audit,proof)
            with self.assertRaises(ValueError):self.run_package()
            self.assertFalse(self.output.exists())

    def test_wrong_audit_inventory_digest_refused(self):
        self.write(self.audit,dict(self.proof,evidence_inventory_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'inventory digest'):self.run_package()

    def test_post_audit_frozen_source_tamper_refused(self):
        (self.study/'frozen-source/piper_rgbd_tracker.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'differs from independent audit'):self.run_package()
        self.assertFalse(self.output.exists())

    def test_post_audit_report_tamper_refused(self):
        (self.study/'confirmation/report.json').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'sealed identity'):self.run_package()

    def test_development_frame_tamper_refused_by_sealed_manifest(self):
        (self.study/'development/pose00-160-clear/depth.npy').write_bytes(b'changed depth')
        with self.assertRaisesRegex(ValueError,'differs from independent audit or sealed manifest'):self.run_package()

    def test_symlink_file_and_directory_refused(self):
        path=self.study/'frozen-source/piper_rgbd_tracker.py';raw=path.read_bytes();path.unlink()
        elsewhere=self.root/'elsewhere.py';elsewhere.write_bytes(raw);path.symlink_to(elsewhere)
        with self.assertRaisesRegex(ValueError,'Symlinks'):self.run_package()
        path.unlink();path.write_bytes(raw)
        (self.study/'extra').symlink_to(self.history[0],target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Symlinks'):self.run_package()

    def test_unlisted_secret_or_mesh_file_refused(self):
        path=self.study/'credentials.json';self.write(path,{'not_a_real_credential':'test'})
        with self.assertRaisesRegex(ValueError,'Unrecognized study file'):self.run_package()
        path.unlink();(self.study/'arm.stl').write_bytes(b'not a mesh')
        with self.assertRaisesRegex(ValueError,'Unrecognized study file'):self.run_package()

    def test_change_between_preflight_and_actual_copy_refused(self):
        original=pkg.bound_read;calls=0
        def mutate_on_copy(path,expected):
            nonlocal calls
            if Path(path)==self.study/'protocol.json':
                calls+=1
                if calls==2:Path(path).write_text('{"changed":true}\n')
            return original(path,expected)
        with patch.object(pkg,'bound_read',mutate_on_copy):
            with self.assertRaisesRegex(ValueError,'sealed identity'):self.run_package()
        self.assertFalse((self.output/'huggingface').exists())

    def test_zip_readback_detects_changed_compressed_member(self):
        original=zipfile.ZipFile.writestr
        def corrupt(zipped,info,raw,*args,**kwargs):
            if info.filename=='protocol.json':raw=b'changed archive member'
            return original(zipped,info,raw,*args,**kwargs)
        with patch.object(zipfile.ZipFile,'writestr',corrupt):
            with self.assertRaisesRegex(ValueError,'Archived bytes differ'):self.run_package()
        self.assertFalse((self.output/'huggingface').exists())


if __name__=='__main__':unittest.main()
