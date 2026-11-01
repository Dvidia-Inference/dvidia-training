from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import publish_piper_retention_v02 as publisher


class RetentionPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root / 'dataset'; self.dataset.mkdir()
        for name in publisher.NAMES:
            (self.dataset / name).write_bytes(b'synthetic retention fixture')
        self.plan = self.root / 'plan.json'
        self.receipt = publisher.stage(self.dataset, self.plan)
        class Operation:
            def __init__(self, path_in_repo, path_or_fileobj):
                self.path_in_repo, self.path_or_fileobj = path_in_repo, path_or_fileobj
        class NotFound(Exception):
            pass
        sdk = SimpleNamespace(CommitOperationAdd=Operation)
        errors = SimpleNamespace(RepositoryNotFoundError=NotFound)
        patcher = patch.dict(sys.modules, {'huggingface_hub': sdk,
                                           'huggingface_hub.errors': errors})
        patcher.start(); self.addCleanup(patcher.stop)

    def info(self, names=('.gitattributes',), private=False, revision='a' * 40):
        return SimpleNamespace(private=private, sha=revision,
                               siblings=[SimpleNamespace(rfilename=name) for name in names])

    def test_staging_checking_and_public_readback_do_not_load_credentials(self):
        api = Mock(); api.repo_info.return_value = self.info(sorted(publisher.NAMES | {'.gitattributes'}))
        with patch.object(publisher, 'client', side_effect=AssertionError('Credential accessed')), \
             patch.object(publisher, 'public_client', return_value=api), \
             patch.object(publisher, 'download', side_effect=lambda repo, name, kind, revision, cache:
                          (self.dataset / name).read_bytes()) as download:
            value = publisher.checked(self.plan, self.receipt['plan_sha256'])
            proof = publisher.verify(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
        self.assertEqual(value['repo_id'], 'Dvidia/piper-retention-v02')
        self.assertEqual(proof['anonymous_files_verified'], 6)
        self.assertEqual(download.call_count, 6)
        api.repo_info.assert_called_once_with(publisher.REPO, repo_type='dataset',
                                              files_metadata=True, token=False)
        self.assertTrue(all(call.args[3] == 'a' * 40 for call in download.call_args_list))
        with self.assertRaises(FileExistsError):
            publisher.stage(self.dataset, self.plan)

    def test_mutated_payload_and_unreviewed_roster_fail_before_credentials(self):
        with patch.object(publisher, 'client') as client:
            (self.dataset / 'README.md').write_bytes(b'changed')
            with self.assertRaises(publisher.PublicationError):
                publisher.publish(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
            (self.dataset / 'README.md').write_bytes(b'synthetic retention fixture')
            (self.dataset / 'unreviewed.json').write_text('{}')
            with self.assertRaises(publisher.PublicationError):
                publisher.publish(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
            client.assert_not_called()

    def test_resealed_destination_duplicate_and_nonfinite_plans_are_rejected(self):
        original = json.loads(self.plan.read_bytes())
        for field, replacement in [('repo_id', 'Dvidia/robot-simulation-lab'),
                                   ('repo_type', 'space'), ('private', True)]:
            value = deepcopy(original); value[field] = replacement
            raw = publisher.encoded(value); self.plan.write_bytes(raw)
            with self.subTest(field=field), self.assertRaises(publisher.PublicationError):
                publisher.checked(self.plan, sha256(raw).hexdigest())
        for raw in (b'{"schema":1,"schema":2}', b'{"private":NaN}'):
            self.plan.write_bytes(raw)
            with self.assertRaises(publisher.PublicationError):
                publisher.checked(self.plan, sha256(raw).hexdigest())
        value = deepcopy(original); value['files'][1] = value['files'][0]
        raw = publisher.encoded(value); self.plan.write_bytes(raw)
        with self.assertRaises(publisher.PublicationError):
            publisher.checked(self.plan, sha256(raw).hexdigest())

    def test_existing_private_or_unpinned_dataset_cannot_be_overwritten(self):
        for info in (self.info(['README.md']), self.info(private=True), self.info(revision='main')):
            api = Mock(); api.repo_info.return_value = info
            with self.subTest(info=info), patch.object(publisher, 'client', return_value=(api, 'fixture')):
                with self.assertRaises(publisher.PublicationError):
                    publisher.publish(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
                api.create_commit.assert_not_called()
                api.create_repo.assert_not_called()

    def test_payload_freeze_and_upload_bytes_are_immutable_across_credential_boundary(self):
        original = publisher.read
        reads = 0
        def changed(path, *args, **kwargs):
            nonlocal reads
            if Path(path).name == 'README.md':
                reads += 1
                if reads == 2:
                    return b'changed after review'
            return original(path, *args, **kwargs)
        with patch.object(publisher, 'read', side_effect=changed), patch.object(publisher, 'client') as client:
            with self.assertRaisesRegex(publisher.PublicationError, 'freezing'):
                publisher.publish(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
            client.assert_not_called()
        api = Mock(); api.repo_info.return_value = self.info()
        api.create_commit.return_value = SimpleNamespace(oid='b' * 40)
        def credential_boundary(_cache):
            (self.dataset / 'README.md').write_bytes(b'late mutation not uploaded')
            return api, 'fixture'
        with patch.object(publisher, 'client', side_effect=credential_boundary):
            result = publisher.publish(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
        self.assertEqual(result['files_added'], 6)
        commit = api.create_commit.call_args
        self.assertEqual(commit.kwargs['parent_commit'], 'a' * 40)
        uploaded = {operation.path_in_repo: operation.path_or_fileobj
                    for operation in commit.kwargs['operations']}
        self.assertEqual(uploaded['README.md'], b'synthetic retention fixture')

    def test_oversized_file_and_changed_anonymous_bytes_are_rejected(self):
        path = self.dataset / publisher.ARCHIVE
        with path.open('wb') as stream:
            stream.truncate(publisher.LIMITS[publisher.ARCHIVE] + 1)
        with self.assertRaisesRegex(publisher.PublicationError, 'oversized'):
            publisher.read(path)
        (self.dataset / publisher.ARCHIVE).write_bytes(b'synthetic retention fixture')
        self.plan.unlink(); self.receipt = publisher.stage(self.dataset, self.plan)
        api = Mock(); api.repo_info.return_value = self.info(sorted(publisher.NAMES | {'.gitattributes'}))
        with patch.object(publisher, 'client', side_effect=AssertionError('Credential accessed')), \
             patch.object(publisher, 'public_client', return_value=api), \
             patch.object(publisher, 'download', return_value=b'changed bytes'):
            with self.assertRaisesRegex(publisher.PublicationError, 'Anonymous public bytes'):
                publisher.verify(self.plan, self.receipt['plan_sha256'], self.root / 'cache')
        target = self.root / 'unreviewed'; target.write_text('fixture')
        (self.dataset / 'README.md').unlink()
        (self.dataset / 'README.md').symlink_to(target)
        with self.assertRaises(publisher.PublicationError):
            publisher.checked(self.plan, self.receipt['plan_sha256'])


if __name__ == '__main__':
    unittest.main()
