"""Offline fixed-destination publication checks. No real token or network use."""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import publish_robot_sim as publisher
from publish import PublicationError


class RobotSimulationPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.space = self.root/'space'
        self.space.mkdir()
        for name,raw in {
            'README.md':b'---\ntitle: Test\nsdk: static\napp_file: index.html\nlicense: mit\n---\nNative recorded evidence.\n',
            'index.html':b'<!doctype html><p>Native recorded simulation</p>',
            'LICENSE':b'MIT','provenance.json':b'{}',
            'replay/geometry.bin':b'native geometry','replay/motion.bin.gz':b'recorded poses'}.items():
            path = self.space/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
        self.plan = {'kind':publisher.KIND,'schema_version':1,'namespace':'Dvidia','private':False,
                     'expected_remote_commit':None,'repository':{'repo_id':publisher.REPO_ID,'repo_type':'space',
                     'space_sdk':'static','directory':'space','files':[]},'total_bytes':0}
        self.path = self.root/'publication-plan.json'
        self.refresh()
        self.credential_guard = patch.object(publisher,'client',side_effect=AssertionError('Real credentials are forbidden in these tests'))
        self.credential_guard.start()
        self.addCleanup(self.credential_guard.stop)

    def refresh(self):
        self.plan['repository']['files'] = [{'path':path.relative_to(self.space).as_posix(),
            'bytes':path.stat().st_size,'sha256':sha256(path.read_bytes()).hexdigest()}
            for path in sorted(self.space.rglob('*')) if path.is_file()]
        self.plan['total_bytes'] = sum(f['bytes'] for f in self.plan['repository']['files'])
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.plan,indent=2)+'\n')

    def api(self,existing=()):
        api = Mock()
        api.whoami.return_value = {'orgs':[{'name':'Dvidia'}]}
        api.repo_info.return_value = SimpleNamespace(private=False,sdk='static',sha='a'*40,
            siblings=[SimpleNamespace(rfilename=name) for name in existing])
        api.create_commit.return_value = SimpleNamespace(oid='b'*40)
        return api

    def test_accepts_complete_named_static_space_and_native_binaries(self):
        self.assertEqual(publisher.checked_plan(self.path),self.plan)

    def test_rejects_unrelated_destination_private_sdk_and_path_escape(self):
        for field,value in [('repo_id','Dvidia/observation-lab'),('repo_type','model'),
                            ('space_sdk','docker'),('directory','../outside')]:
            changed = json.loads(json.dumps(self.plan))
            changed['repository'][field] = value
            self.path.write_text(json.dumps(changed))
            with self.subTest(field=field),self.assertRaises(PublicationError):
                publisher.checked_plan(self.path)
        self.plan['private'] = True
        self.save()
        with self.assertRaises(PublicationError): publisher.checked_plan(self.path)

    def test_rejects_unlisted_tampered_weight_hidden_secret_and_private_path(self):
        extra = self.space/'extra.json'
        extra.write_bytes(b'{}')
        with self.assertRaises(PublicationError): publisher.checked_plan(self.path)
        extra.unlink()
        (self.space/'index.html').write_bytes(b'changed')
        with self.assertRaises(PublicationError): publisher.checked_plan(self.path)
        for name,raw in [('policy.bin',b'weights'),('.env',b'secret'),('weights.onnx',b'weights'),
                         ('secret.txt',b'hf_'+b'a'*32),('private.txt',b'/Users/roguedev/private')]:
            file = self.space/name
            file.write_bytes(raw)
            self.refresh()
            with self.subTest(name=name),self.assertRaises(PublicationError): publisher.checked_plan(self.path)
            file.unlink()
        self.refresh()

    def test_rejects_symlink_and_malformed_totals_before_credentials(self):
        alias = self.space/'alias.json'
        alias.symlink_to(self.space/'provenance.json')
        with self.assertRaises(PublicationError): publisher.checked_plan(self.path)
        alias.unlink()
        self.plan['total_bytes'] += 1
        self.save()
        with self.assertRaises(PublicationError): publisher.publish_robot_sim(self.path)

    def test_frontmatter_cannot_be_spoofed_by_body_or_duplicate_keys(self):
        cards = [
            '---\nsdk: docker\napp_file: app.py\nlicense: apache-2.0\n---\nsdk: static\napp_file: index.html\nlicense: mit\n',
            '---\nsdk: docker\nsdk: static\napp_file: index.html\nlicense: mit\n---\n',
            '---\nsdk: static\n"sdk": gradio\napp_file: index.html\nlicense: mit\n---\n',
            '---\nsdk: static\n  sdk: gradio\napp_file: index.html\nlicense: mit\n---\n',
            'sdk: static\napp_file: index.html\nlicense: mit\n',
            '---\nsdk: static\napp_file: index.html\nlicense: mit\n']
        for card in cards:
            (self.space/'README.md').write_text(card)
            self.refresh()
            with self.subTest(card=card),self.assertRaises(PublicationError): publisher.checked_plan(self.path)

    def test_byte_bounds_are_checked_without_credential_access(self):
        with patch.object(publisher,'MAX_FILE_BYTES',4),self.assertRaises(PublicationError):
            publisher.checked_plan(self.path)
        with patch.object(publisher,'MAX_TOTAL_BYTES',16),self.assertRaises(PublicationError):
            publisher.checked_plan(self.path)

    def test_receipt_hash_and_upload_bytes_use_original_snapshot_despite_local_edits(self):
        api = self.api()
        original_plan = self.path.read_bytes()
        original_index = (self.space/'index.html').read_bytes()
        def change_local():
            self.path.write_text('{"changed_after_review":true}')
            (self.space/'index.html').write_bytes(b'changed after payload freeze')
            return {'orgs':[{'name':'Dvidia'}]}
        api.whoami.side_effect = change_local
        with patch.object(publisher,'client',return_value=(api,'fixture-token')), \
                patch.object(publisher,'verified_downloads') as verify:
            result = publisher.publish_robot_sim(self.path)
        self.assertEqual(result['plan_sha256'],sha256(original_plan).hexdigest())
        operations = api.create_commit.call_args.kwargs['operations']
        uploaded = {operation.path_in_repo:operation.path_or_fileobj for operation in operations}
        self.assertEqual(uploaded['index.html'],original_index)
        self.assertEqual(api.create_commit.call_args.kwargs['parent_commit'],'a'*40)
        self.assertTrue(all(call.args[2] is False for call in verify.call_args_list))

    def test_same_plan_retry_is_idempotent_but_changed_parent_and_unrelated_files_fail(self):
        all_files = [f['path'] for f in self.plan['repository']['files']]
        api = self.api(all_files)
        with patch.object(publisher,'client',return_value=(api,'fixture-token')),patch.object(publisher,'verified_downloads'):
            self.assertEqual(publisher.publish_robot_sim(self.path)['commit'],'a'*40)
        api.create_commit.assert_not_called()
        for existing,parent in [(['unrelated.txt'],None),(all_files,'c'*40)]:
            self.plan['expected_remote_commit'] = parent
            self.save()
            api = self.api(existing)
            with patch.object(publisher,'client',return_value=(api,'fixture-token')),patch.object(publisher,'verified_downloads'),self.assertRaises(PublicationError):
                publisher.publish_robot_sim(self.path)
            api.create_commit.assert_not_called()

    def test_reviewed_exact_parent_allows_an_explicit_static_space_update(self):
        self.plan['expected_remote_commit'] = 'a'*40
        self.save()
        api = self.api([f['path'] for f in self.plan['repository']['files']])
        with patch.object(publisher,'client',return_value=(api,'fixture-token')),patch.object(publisher,'verified_downloads'):
            self.assertEqual(publisher.publish_robot_sim(self.path)['commit'],'b'*40)
        self.assertEqual(api.create_commit.call_args.kwargs['parent_commit'],'a'*40)
        api.create_repo.assert_not_called()


if __name__=='__main__':
    unittest.main()
