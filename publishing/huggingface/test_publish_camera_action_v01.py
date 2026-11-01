from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import stat
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import publish_camera_action_v01 as publisher


class CameraPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        names = (publisher.REQUIRED - {'manifest.json', 'SHA256SUMS'}) | {
            'source/fixture.whl', 'source/fixture.tar.gz'}
        for name in names:
            path = self.dataset / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'Synthetic test evidence only.')
        (self.dataset / 'report.json').write_text(json.dumps(dict(
            qualification='research_candidate', qualified=False, physical_robot_ready=False, status='blocked')))
        files = {name: publisher.identity(self.dataset / name) for name in sorted(names)}
        manifest = dict(schema=publisher.EVIDENCE_SCHEMA, repo_id=publisher.REPO, repo_type='dataset',
            status='blocked', qualification='research_candidate', qualified=False,
            original_footage_included=False, physical_robot_ready=False,
            files=files, episodes=[], episode_count=0, trained_model_files=[])
        (self.dataset / 'manifest.json').write_text(json.dumps(manifest))
        files['manifest.json'] = publisher.identity(self.dataset / 'manifest.json')
        (self.dataset / 'SHA256SUMS').write_text(''.join(f'{row["sha256"]}  {name}\n' for name, row in sorted(files.items())))
        self.plan = self.root / 'plan.json'
        self.receipt = publisher.stage(self.dataset, self.plan)
        class Operation:
            def __init__(self, path_in_repo, path_or_fileobj):
                self.path_in_repo, self.path_or_fileobj = path_in_repo, path_or_fileobj
        class NotFound(Exception):
            pass
        patches = patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(CommitOperationAdd=Operation,
            constants=SimpleNamespace(HF_HUB_DISABLE_XET=True)),
            'huggingface_hub.errors': SimpleNamespace(RepositoryNotFoundError=NotFound)})
        patches.start()
        self.addCleanup(patches.stop)

    def info(self, names=('.gitattributes',), *, private=False, sha='a' * 40):
        return SimpleNamespace(private=private, sha=sha,
            siblings=[SimpleNamespace(rfilename=name,size=(self.dataset/name).stat().st_size if (self.dataset/name).is_file() else 0) for name in names])

    def downloaded(self,name,revision,cache,*,wrong=False):
        self.assertEqual(revision,'a'*40)
        target=Path(cache)/'synthetic-download'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(b'not the published bytes' if wrong else (self.dataset/name).read_bytes())
        return target

    def test_stage_check_and_full_public_readback_never_access_credentials(self):
        value = publisher.checked(self.plan, self.receipt['plan_sha256'])
        api = Mock()
        api.repo_info.return_value = self.info([r['path'] for r in value['files']] + ['.gitattributes'])
        with patch.object(publisher, 'client', side_effect=AssertionError('Unexpected credential access')), \
                patch.object(publisher, 'public_client', return_value=api), \
                patch.object(publisher, 'download', side_effect=self.downloaded) as download:
            proof = publisher.verify(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
        self.assertEqual(proof['anonymous_files_verified'], len(value['files']))
        self.assertEqual(proof['anonymous_bytes_verified'],sum(r['bytes'] for r in value['files']))
        self.assertEqual(proof['reviewed_plan_sha256'],self.receipt['plan_sha256'])
        self.assertEqual(len({call.args[2] for call in download.call_args_list}),len(value['files']))
        self.assertTrue(all(not Path(call.args[2]).exists() for call in download.call_args_list))
        self.assertEqual(list((self.root/'cache').iterdir()),[])
        api.repo_info.assert_called_once_with(publisher.REPO, repo_type='dataset', files_metadata=True, token=False)

    def test_mutated_payload_and_changed_destination_fail_before_keychain(self):
        (self.dataset / 'README.md').write_bytes(b'changed')
        with patch.object(publisher, 'client') as client:
            with self.assertRaises(publisher.PublicationError):
                publisher.publish(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
            client.assert_not_called()
        value = json.loads(self.plan.read_bytes())
        value['repo_id'] = 'Dvidia/robot-simulation-lab'
        self.plan.write_bytes(publisher.encoded(value))
        with self.assertRaisesRegex(publisher.PublicationError, 'destination'):
            publisher.checked(self.plan, sha256(self.plan.read_bytes()).hexdigest())

    def test_existing_private_or_nonempty_dataset_cannot_be_overwritten(self):
        for info in (self.info(['README.md']), self.info(private=True), self.info(sha='main')):
            api = Mock()
            api.repo_info.return_value = info
            with self.subTest(info=info), patch.object(publisher, 'client', return_value=(api, 'fixture')):
                with self.assertRaises(publisher.PublicationError):
                    publisher.publish(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
            api.create_commit.assert_not_called()

    def test_payload_is_frozen_before_credential_boundary_and_uses_parent_cas(self):
        api = Mock()
        api.repo_info.return_value = self.info()
        uploaded = {}
        def commit(repo, **kwargs):
            self.assertEqual(repo, publisher.REPO)
            self.assertEqual(kwargs['parent_commit'], 'a' * 40)
            for operation in kwargs['operations']:
                uploaded[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
            return SimpleNamespace(oid='b' * 40)
        api.create_commit.side_effect = commit
        original = (self.dataset / 'README.md').read_bytes()
        def credential_boundary(_cache):
            (self.dataset / 'README.md').write_bytes(b'late mutation must not be uploaded')
            return api, 'fixture'
        with patch.object(publisher, 'client', side_effect=credential_boundary):
            result = publisher.publish(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
        self.assertEqual(uploaded['README.md'], original)
        self.assertEqual(result['commit'], 'b' * 40)
        self.assertEqual(list((self.root / 'cache').glob('camera-action-frozen-*')), [])

    def test_extra_file_symlink_and_false_qualification_are_refused(self):
        target = self.dataset / 'secret.env'
        target.write_text('not a credential')
        with self.assertRaises(publisher.PublicationError):
            publisher.files(self.dataset)
        target.unlink()
        readme = self.dataset / 'README.md'
        readme.unlink()
        readme.symlink_to(self.dataset / 'LICENSE')
        with self.assertRaises(publisher.PublicationError):
            publisher.files(self.dataset)
        readme.unlink()
        readme.write_bytes(b'Synthetic test evidence only.')
        report = json.loads((self.dataset / 'report.json').read_bytes())
        report['qualified'] = True
        (self.dataset / 'report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(publisher.PublicationError, 'qualification'):
            publisher.files(self.dataset)

    def test_public_readback_detects_changed_remote_bytes(self):
        value = publisher.checked(self.plan, self.receipt['plan_sha256'])
        api = Mock()
        api.repo_info.return_value = self.info([r['path'] for r in value['files']] + ['.gitattributes'])
        with patch.object(publisher, 'public_client', return_value=api), \
                patch.object(publisher, 'download', side_effect=lambda *args:self.downloaded(*args,wrong=True)):
            with self.assertRaisesRegex(publisher.PublicationError, 'Anonymous public bytes'):
                publisher.verify(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
        self.assertEqual(list((self.root/'cache').iterdir()),[])

    def test_readback_protects_reserve_and_never_deletes_shared_cache_files(self):
        value=publisher.checked(self.plan,self.receipt['plan_sha256']);api=Mock()
        api.repo_info.return_value=self.info([r['path'] for r in value['files']]+['.gitattributes'])
        cache=self.root/'cache';cache.mkdir();shared=cache/'user-shared-cache';shared.write_bytes(b'keep existing user data')
        with patch.object(publisher,'public_client',return_value=api), \
                patch.object(publisher.shutil,'disk_usage',return_value=SimpleNamespace(free=publisher.RESERVE)), \
                patch.object(publisher,'download') as download:
            with self.assertRaisesRegex(publisher.PublicationError,'protected disk reserve'):
                publisher.verify(self.plan,self.receipt['plan_sha256'],cache)
            download.assert_not_called()
        with patch.object(publisher,'public_client',return_value=api),patch.object(publisher,'download',return_value=shared):
            with self.assertRaisesRegex(publisher.PublicationError,'escaped its owned'):
                publisher.verify(self.plan,self.receipt['plan_sha256'],cache)
        self.assertEqual(shared.read_bytes(),b'keep existing user data')
        self.assertEqual(list(cache.iterdir()),[shared])

    def test_download_is_anonymous_and_revision_pinned_in_explicit_private_cache(self):
        cache=self.root/'private-readback';cache.mkdir();blob=cache/'blob';blob.write_bytes(b'fixture')
        download=Mock(return_value=str(blob))
        with patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(hf_hub_download=download,
                constants=SimpleNamespace(HF_HUB_DISABLE_XET=True))}):
            self.assertEqual(publisher.download('source/fixture.whl','a'*40,cache),blob.resolve())
        download.assert_called_once_with(publisher.REPO,'source/fixture.whl',repo_type='dataset',revision='a'*40,
            token=False,endpoint='https://huggingface.co',cache_dir=str(cache),force_download=True)

    def test_missing_or_changed_remote_size_refuses_before_any_download(self):
        value=publisher.checked(self.plan,self.receipt['plan_sha256'])
        for size in (None,True,-1,10**12):
            api=Mock();info=self.info([r['path'] for r in value['files']]+['.gitattributes'])
            info.siblings[0].size=size;api.repo_info.return_value=info
            with self.subTest(size=size),patch.object(publisher,'public_client',return_value=api), \
                    patch.object(publisher,'download') as download:
                with self.assertRaisesRegex(publisher.PublicationError,'remote size differs'):
                    publisher.verify(self.plan,self.receipt['plan_sha256'],self.root/'cache')
                download.assert_not_called()

    def test_preimported_xet_enabled_sdk_refuses_before_remote_calls(self):
        with patch.dict(sys.modules,{'huggingface_hub':SimpleNamespace(constants=SimpleNamespace(HF_HUB_DISABLE_XET=False))}), \
                patch.object(publisher,'public_client') as client:
            with self.assertRaisesRegex(publisher.PublicationError,'fresh process'):
                publisher.verify(self.plan,self.receipt['plan_sha256'],self.root/'cache')
            client.assert_not_called()

    def test_qualified_label_requires_full_study_audit_even_when_report_matches_manifest(self):
        for name in ('manifest.json','report.json'):
            path=self.dataset/name; value=json.loads(path.read_bytes())
            value.update(qualified=True,qualification='bounded_simulation_skill',status='complete')
            path.write_text(json.dumps(value))
        with self.assertRaisesRegex(publisher.PublicationError,'requires the full-study'):
            publisher.files(self.dataset)

    def test_private_hardware_fields_are_rejected_before_upload(self):
        from test_camera_action_hardware import fixture
        value=fixture();value['hardware']['serial']='private fixture'
        target=self.dataset/'hardware.json';target.write_text(json.dumps(value))
        manifest=json.loads((self.dataset/'manifest.json').read_bytes())
        manifest['hardware_context_file']='hardware.json';manifest['files']['hardware.json']=publisher.identity(target)
        (self.dataset/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(publisher.PublicationError,'private or unbounded'):
            publisher.files(self.dataset)

    def test_private_freeze_has_independent_inode_attributes_and_mutations(self):
        source=self.root/'source';source.write_bytes(b'reviewed bytes'*1000);source.chmod(0o640)
        target=self.root/'private'/'frozen';expected=publisher.identity(source)
        method=publisher.freeze_file(source,target,expected)
        self.assertIn(method,('apfs_clone','stream_copy'))
        self.assertNotEqual(source.stat().st_ino,target.stat().st_ino)
        self.assertEqual(target.stat().st_nlink,1)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode),0o400)
        self.assertEqual(stat.S_IMODE(source.stat().st_mode),0o640)
        source.write_bytes(b'original mutated after private freeze')
        self.assertEqual(publisher.identity(target),expected)
        target.chmod(0o600);target.write_bytes(b'private mutation')
        self.assertEqual(source.read_bytes(),b'original mutated after private freeze')

    def test_freeze_rejects_destination_collision_and_symlinks_without_overwrite(self):
        source=self.root/'source';source.write_bytes(b'reviewed bytes');expected=publisher.identity(source)
        target=self.root/'frozen';target.write_bytes(b'existing destination')
        with self.assertRaisesRegex(publisher.PublicationError,'already exists'):
            publisher.freeze_file(source,target,expected)
        self.assertEqual(target.read_bytes(),b'existing destination')
        target.unlink();target.symlink_to(source)
        with self.assertRaises(publisher.PublicationError):publisher.freeze_file(source,target,expected)
        target.unlink();source_link=self.root/'source-link';source_link.symlink_to(source)
        with self.assertRaises(publisher.PublicationError):publisher.freeze_file(source_link,target,expected)
        directory=self.root/'linked-parent';directory.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(publisher.PublicationError):publisher.freeze_file(source,directory/'frozen',expected)

    def test_stream_fallback_is_guarded_per_file_and_during_copy(self):
        source=self.root/'source';source.write_bytes(b'x'*(2*1024*1024));expected=publisher.identity(source)
        target=self.root/'private'/'frozen'
        def usage(free):return SimpleNamespace(free=publisher.RESERVE+free)
        with patch.object(publisher,'_clone_from_fd',return_value=False), \
                patch.object(publisher.shutil,'disk_usage',return_value=usage(1536*1024)):
            with self.assertRaisesRegex(publisher.PublicationError,'protected disk reserve'):
                publisher.freeze_file(source,target,expected)
        self.assertFalse(target.exists())
        with patch.object(publisher,'_clone_from_fd',return_value=False), \
                patch.object(publisher.shutil,'disk_usage',side_effect=[usage(5*1024*1024)]*3+[usage(-1)]):
            with self.assertRaisesRegex(publisher.PublicationError,'protected disk reserve'):
                publisher.freeze_file(source,target,expected)
        self.assertFalse(target.exists())
        with patch.object(publisher,'_clone_from_fd',return_value=False):
            self.assertEqual(publisher.freeze_file(source,target,expected),'stream_copy')
        source.write_bytes(b'late change');self.assertEqual(publisher.identity(target),expected)

    def test_growing_stream_source_refuses_before_writing_excess_and_cleans_target(self):
        source=self.root/'source';source.write_bytes(b'x'*(2*1024*1024));expected=publisher.identity(source)
        target=self.root/'private'/'frozen';read=os.read;calls=[]
        def growing(fd,count):
            chunk=read(fd,count);calls.append(len(chunk))
            if len(calls)==1:
                with source.open('ab') as stream:stream.write(b'g'*(3*1024*1024))
            return chunk
        with patch.object(publisher,'_clone_from_fd',return_value=False),patch.object(publisher.os,'read',side_effect=growing):
            with self.assertRaisesRegex(publisher.PublicationError,'grew beyond'):
                publisher.freeze_file(source,target,expected)
        self.assertEqual(calls,[1024*1024]*3)
        self.assertFalse(target.exists())

    def test_atomic_clone_collision_does_not_delete_the_existing_destination(self):
        source=self.root/'source';source.write_bytes(b'reviewed');target=self.root/'private'/'frozen'
        def collision(*_args):
            target.write_bytes(b'racing destination')
            raise FileExistsError('Destination already created.')
        with patch.object(publisher,'_clone_from_fd',side_effect=collision):
            with self.assertRaises(FileExistsError):publisher.freeze_file(source,target,publisher.identity(source))
        self.assertEqual(target.read_bytes(),b'racing destination')


if __name__ == '__main__':
    unittest.main()
