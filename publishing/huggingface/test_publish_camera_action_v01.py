from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
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
        patches = patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(CommitOperationAdd=Operation),
            'huggingface_hub.errors': SimpleNamespace(RepositoryNotFoundError=NotFound)})
        patches.start()
        self.addCleanup(patches.stop)

    def info(self, names=('.gitattributes',), *, private=False, sha='a' * 40):
        return SimpleNamespace(private=private, sha=sha,
            siblings=[SimpleNamespace(rfilename=name) for name in names])

    def test_stage_check_and_full_public_readback_never_access_credentials(self):
        value = publisher.checked(self.plan, self.receipt['plan_sha256'])
        api = Mock()
        api.repo_info.return_value = self.info([r['path'] for r in value['files']] + ['.gitattributes'])
        with patch.object(publisher, 'client', side_effect=AssertionError('Unexpected credential access')), \
                patch.object(publisher, 'public_client', return_value=api), \
                patch.object(publisher, 'download', side_effect=lambda name, revision, cache: self.dataset / name):
            proof = publisher.verify(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
        self.assertEqual(proof['anonymous_files_verified'], len(value['files']))
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
        changed = self.root / 'changed'
        changed.write_bytes(b'not the published bytes')
        with patch.object(publisher, 'public_client', return_value=api), \
                patch.object(publisher, 'download', return_value=changed):
            with self.assertRaisesRegex(publisher.PublicationError, 'Anonymous public bytes'):
                publisher.verify(self.plan, self.receipt['plan_sha256'], self.root / 'cache')

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


if __name__ == '__main__':
    unittest.main()
