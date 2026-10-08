"""Exercise publication boundaries with temporary fixtures and a mocked Hub.

No test accesses Keychain, reads a real token, or sends a network request.
"""
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import huggingface_hub
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

import publish as publisher
from prepare import REPOS, safe_relative


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / 'publication-plan.json'
        self.payload = {}
        rows = []
        for kind, name in REPOS.items():
            base = self.root / kind
            base.mkdir()
            raw = (kind + ' reviewed fixture\n').encode()
            (base / 'README.md').write_bytes(raw)
            self.payload[kind, 'README.md'] = raw
            row = {'repo_id': 'Dvidia/' + name, 'repo_type': kind,
                   'directory': kind, 'files': [self.item('README.md', raw)]}
            if kind == 'space':
                row['space_sdk'] = 'static'
            rows.append(row)
        self.plan = {'schema_version': 1, 'kind': 'dvidia.huggingface-publication',
                     'namespace': 'Dvidia', 'private': False,
                     'evidence_release': '2026-10-08-footage-pipeline-v1',
                     'repositories': rows}
        self.save_plan()
        guard = mock.patch.object(publisher, 'keychain_token',
                                  side_effect=AssertionError('Credential access forbidden in tests.'))
        guard.start()
        self.addCleanup(guard.stop)

    @staticmethod
    def item(name, raw):
        return {'path': name, 'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}

    def save_plan(self):
        self.path.write_text(json.dumps(self.plan))

    def replace_fixture(self, kind, raw):
        (self.root / kind / 'README.md').write_bytes(raw)
        self.payload[kind, 'README.md'] = raw
        row = next(row for row in self.plan['repositories'] if row['repo_type'] == kind)
        row['files'] = [self.item('README.md', raw)]
        self.save_plan()

    def api(self):
        api = mock.create_autospec(HfApi, instance=True)
        api.whoami.return_value = {'name': 'test-publisher', 'orgs': [{'name': 'Dvidia'}]}
        api.repo_info.return_value = SimpleNamespace(
            private=False, sdk='static', sha='a' * 40, siblings=[], gated=False)
        api.create_commit.return_value = SimpleNamespace(oid='b' * 40)
        return api

    def download(self, repo_id, filename, *, repo_type, revision, token, endpoint):
        self.assertEqual(endpoint, 'https://huggingface.co')
        self.assertEqual(repo_id, 'Dvidia/' + REPOS[repo_type])
        self.assertIn(revision, ('a' * 40, 'b' * 40))
        target = self.root / 'mock-downloads' / repo_type / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payload[repo_type, filename])
        return str(target)

    def assert_no_remote_mutation(self, api):
        api.create_repo.assert_not_called()
        api.create_commit.assert_not_called()

    def test_complete_matching_inventory_is_accepted(self):
        self.assertEqual(publisher.checked_plan(self.path), self.plan)

    def test_added_file_missing_file_and_changed_bytes_are_rejected(self):
        base = self.root / 'dataset'
        (base / 'unreviewed.json').write_text('{}')
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(self.path)
        (base / 'unreviewed.json').unlink()
        original = (base / 'README.md').read_bytes()
        (base / 'README.md').unlink()
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(self.path)
        (base / 'README.md').write_bytes(bytes(byte ^ 1 for byte in original))
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(self.path)

    def test_duplicate_inventory_and_hidden_files_are_rejected(self):
        row = self.plan['repositories'][0]
        row['files'].append(dict(row['files'][0]))
        self.save_plan()
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(self.path)
        row['files'].pop()
        raw = b'{}'
        (self.root / row['directory'] / '.private.json').write_bytes(raw)
        row['files'].append(self.item('.private.json', raw))
        self.save_plan()
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(self.path)

    def test_paths_cannot_escape_or_change_the_planned_repository(self):
        for name in ('../outside.json', '/absolute.json', 'a//b.json',
                     'a/./b.json', 'a\\b.json', 'a:b.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_relative(name)
        self.plan['repositories'][0]['directory'] = '../dataset'
        self.save_plan()
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(self.path)

    def test_private_plan_and_nonstatic_space_are_rejected(self):
        self.plan['private'] = True
        self.save_plan()
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(self.path)
        self.plan['private'] = False
        next(row for row in self.plan['repositories'] if row['repo_type'] == 'space')['space_sdk'] = 'docker'
        self.save_plan()
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(self.path)

    def test_plan_and_artifact_symlinks_are_rejected(self):
        linked_plan = self.root / 'linked-plan.json'
        linked_plan.symlink_to(self.path)
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(linked_plan)
        artifact = self.root / 'dataset' / 'README.md'
        target = self.root / 'protected.md'
        target.write_bytes(artifact.read_bytes())
        artifact.unlink()
        artifact.symlink_to(target)
        with self.assertRaises(publisher.PublicationError):
            publisher.checked_plan(self.path)

    def test_cache_and_receipt_symlinks_are_rejected_before_credentials(self):
        protected = self.root / 'protected'
        protected.mkdir()
        cache = self.root / 'publisher-cache'
        cache.symlink_to(protected, target_is_directory=True)
        with self.assertRaises(publisher.PublicationError):
            publisher.client(cache)
        target = protected / 'receipt.json'
        target.write_text('must remain unchanged')
        (self.root / 'publication-receipt.json').symlink_to(target)
        with mock.patch.object(publisher, 'client') as client:
            with self.assertRaises(publisher.PublicationError):
                publisher.publish(self.path)
            client.assert_not_called()
        self.assertEqual(target.read_text(), 'must remain unchanged')

    def test_existing_private_or_nonstatic_repo_prevents_all_mutation(self):
        for restricted in ('private', 'sdk'):
            with self.subTest(restricted=restricted):
                api = self.api()
                def repo_info(repo_id, *, repo_type):
                    return SimpleNamespace(private=restricted == 'private',
                                           sdk='docker' if restricted == 'sdk' and repo_type == 'space' else 'static',
                                           sha='a' * 40, siblings=[])
                api.repo_info.side_effect = repo_info
                with mock.patch.object(publisher, 'client', return_value=(api, 'fixture-token')):
                    with self.assertRaises(publisher.PublicationError):
                        publisher.publish(self.path)
                self.assert_no_remote_mutation(api)

    def test_unrelated_existing_files_or_different_remote_bytes_are_not_overwritten(self):
        for remote_name in ('unrelated.json', 'README.md'):
            with self.subTest(remote_name=remote_name):
                api = self.api()
                api.repo_info.return_value = SimpleNamespace(private=False, sdk='static', sha='a' * 40,
                    siblings=[SimpleNamespace(rfilename=remote_name)])
                remote = self.root / 'different-remote.md'
                remote.write_text('remote content belongs to someone else')
                with mock.patch.object(publisher, 'client', return_value=(api, 'fixture-token')):
                    with mock.patch.object(huggingface_hub, 'hf_hub_download', return_value=str(remote)):
                        with self.assertRaises(publisher.PublicationError):
                            publisher.publish(self.path)
                self.assert_no_remote_mutation(api)

    def test_incomplete_remote_visibility_inventory_or_commit_fails_closed(self):
        for field in ('private', 'siblings', 'sha'):
            with self.subTest(field=field):
                api = self.api()
                setattr(api.repo_info.return_value, field, None)
                with mock.patch.object(publisher, 'client', return_value=(api, 'fixture-token')):
                    with self.assertRaises(publisher.PublicationError):
                        publisher.publish(self.path)
                self.assert_no_remote_mutation(api)

    def test_secret_detection_happens_before_remote_calls_or_mutations(self):
        self.replace_fixture('dataset', b'hf_' + b'X' * 32)
        with mock.patch.object(publisher, 'client') as client:
            with self.assertRaises(publisher.PublicationError):
                publisher.publish(self.path)
            client.assert_not_called()
        token = 'fixture-only-memory-secret'
        self.replace_fixture('dataset', token.encode())
        api = self.api()
        with mock.patch.object(publisher, 'client', return_value=(api, token)):
            with self.assertRaises(publisher.PublicationError):
                publisher.publish(self.path)
        api.whoami.assert_not_called()
        api.repo_info.assert_not_called()
        self.assert_no_remote_mutation(api)

    def test_frozen_bytes_survive_local_mutation_during_preflight(self):
        api = self.api()
        def identity():
            (self.root / 'dataset' / 'README.md').write_bytes(b'changed during mocked API preflight')
            return {'name': 'test-publisher', 'orgs': [{'name': 'Dvidia'}]}
        api.whoami.side_effect = identity
        uploaded = {}
        def commit(repo_id, *, repo_type, operations, parent_commit, commit_message):
            self.assertEqual(parent_commit, 'a' * 40)
            for operation in operations:
                self.assertIsInstance(operation.path_or_fileobj, bytes)
                uploaded[repo_type, operation.path_in_repo] = operation.path_or_fileobj
            return SimpleNamespace(oid='b' * 40)
        api.create_commit.side_effect = commit
        with mock.patch.object(publisher, 'client', return_value=(api, 'fixture-token')):
            with mock.patch.object(huggingface_hub, 'hf_hub_download', side_effect=self.download) as download:
                with redirect_stdout(io.StringIO()):
                    result = publisher.publish(self.path)
        self.assertEqual(uploaded, self.payload)
        self.assertEqual(len(result), 3)
        self.assertEqual(api.create_commit.call_count, 3)
        api.create_repo.assert_not_called()
        self.assertTrue(all(call.kwargs['token'] is False for call in download.call_args_list))
        receipt = json.loads((self.root / 'publication-receipt.json').read_text())
        self.assertEqual(receipt['repositories'], result)
        self.assertEqual(list(self.root.glob('.publication-receipt-*')), [])

    def test_new_repositories_use_the_installed_sdk_signature_and_verified_commits(self):
        api = self.api()
        created = set()
        def repo_info(repo_id, *, repo_type):
            if repo_id not in created:
                response = SimpleNamespace(status_code=404, headers={}, request=None)
                raise RepositoryNotFoundError('Mock destination does not exist.', response=response)
            return SimpleNamespace(private=False, sdk='static', sha='a' * 40, siblings=[])
        def create_repo(repo_id, *, repo_type, private, space_sdk, exist_ok):
            self.assertFalse(private)
            self.assertFalse(exist_ok)
            self.assertEqual(space_sdk, 'static' if repo_type == 'space' else None)
            created.add(repo_id)
        api.repo_info.side_effect = repo_info
        api.create_repo.side_effect = create_repo
        with mock.patch.object(publisher, 'client', return_value=(api, 'fixture-token')):
            with mock.patch.object(huggingface_hub, 'hf_hub_download', side_effect=self.download) as download:
                with redirect_stdout(io.StringIO()):
                    results = publisher.publish(self.path)
        self.assertEqual(created, {'Dvidia/' + name for name in REPOS.values()})
        self.assertEqual(api.create_repo.call_count, 3)
        self.assertEqual(api.create_commit.call_count, 3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(call.kwargs['revision'] == 'b' * 40 and call.kwargs['token'] is False
                            for call in download.call_args_list))

    def test_selection_skips_unselected_destinations_but_checks_the_entire_plan(self):
        api = self.api()
        with mock.patch.object(publisher, 'client', return_value=(api, 'fixture-token')):
            with mock.patch.object(huggingface_hub, 'hf_hub_download', side_effect=self.download):
                with redirect_stdout(io.StringIO()):
                    results = publisher.publish(self.path, only=['model', 'space'])
        self.assertEqual({row['repo_type'] for row in results}, {'model', 'space'})
        self.assertEqual({call.kwargs['repo_type'] for call in api.repo_info.call_args_list}, {'model', 'space'})
        self.assertEqual(api.create_commit.call_count, 2)
        (self.root / 'dataset' / 'README.md').write_text('unreviewed change')
        with mock.patch.object(publisher, 'client') as client:
            with self.assertRaises(publisher.PublicationError):
                publisher.publish(self.path, only=['model', 'space'])
            client.assert_not_called()

    def test_main_suppresses_raw_sdk_valueerror_containing_a_secret(self):
        api = self.api()
        secret = 'hf_' + 'Y' * 32
        api.whoami.side_effect = ValueError('SDK error echoes ' + secret)
        output, errors = io.StringIO(), io.StringIO()
        with mock.patch.object(publisher, 'client', return_value=(api, 'fixture-token')):
            with mock.patch.object(sys, 'argv', ['publish.py', 'identity']):
                with redirect_stdout(output), redirect_stderr(errors):
                    code = publisher.main()
        self.assertEqual(code, 1)
        self.assertNotIn(secret, output.getvalue() + errors.getvalue())
        self.assertNotIn('SDK error echoes', output.getvalue() + errors.getvalue())
        self.assertIn('ValueError', errors.getvalue())
        self.assertIn('Raw error suppressed', errors.getvalue())
        self.assert_no_remote_mutation(api)


if __name__ == '__main__':
    unittest.main()
