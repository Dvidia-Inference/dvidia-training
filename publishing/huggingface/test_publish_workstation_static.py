"""The hosting migration changes storage for three files, never their bytes."""
import base64
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import prepare_workstation as original
import publish_workstation_static as hosting
import test_prepare_workstation as fixtures


class StaticHostingTests(unittest.TestCase):
    refresh_base = fixtures.WorkstationPublicationTests.refresh_base

    def setUp(self):
        fixtures.WorkstationPublicationTests.setUp(self)
        self.asset_bytes = {name: ('unchanged compressed '+name).encode() for name in hosting.ASSETS}
        self.original.update(self.asset_bytes)
        for name, raw in self.asset_bytes.items():
            path = self.space/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.refresh_base()
        (self.plan.parent/'publication-receipt.json').write_bytes(original.encoded({
            'repo_id': hosting.REPO, 'repo_type': 'space', 'commit': hosting.PARENT,
            'plan_sha256': sha256(self.plan.read_bytes()).hexdigest()}))
        self.attributes = b'*.gz filter=lfs diff=lfs merge=lfs -text\n*.wasm filter=lfs diff=lfs merge=lfs -text\n'
        self.before = {name: hosting.regular_record(raw) for name, raw in self.original.items()}
        self.before['.gitattributes'] = hosting.regular_record(self.attributes)
        for name, raw in self.asset_bytes.items():
            self.before[name] = {'size': len(raw), 'blob_id': '0'*40,
                'lfs': {'sha256': sha256(raw).hexdigest(), 'size': len(raw), 'pointer_size': 128},
                'xet_hash': 'f'*64}
        self.after = {**self.before, **{name: hosting.regular_record(raw) for name, raw in self.asset_bytes.items()},
                      '.gitattributes': hosting.regular_record(hosting.attributes_after(self.attributes))}
        self.api = Mock()
        self.info = SimpleNamespace(private=False, sdk='static', sha=hosting.PARENT, siblings=[])
        self.api.repo_info.return_value = self.info
        self.api.whoami.return_value = {'orgs': [{'name': 'Dvidia'}]}
        self.remote = {hosting.PARENT: self.before, 'b'*40: self.after}
        def entries(*args, **kwargs):
            return [SimpleNamespace(path=name, size=row['size'], blob_id=row['blob_id'],
                    lfs=None if row['lfs'] is None else SimpleNamespace(**row['lfs']), xet_hash=row['xet_hash'])
                    for name, row in self.remote[kwargs['revision']].items()]
        self.api.list_repo_tree.side_effect = entries
        def download(name, revision, limit):
            if name == '.gitattributes':
                return self.attributes if revision == hosting.PARENT else hosting.attributes_after(self.attributes)
            return self.asset_bytes[name]
        for name, value in (('public_api', lambda: self.api), ('public_bytes', download),
                            ('client', lambda _: (self.api, 'test-credential'))):
            guard = patch.object(hosting, name, side_effect=value)
            guard.start()
            self.addCleanup(guard.stop)
        self.output = self.root/'hosting'
        self.hosting_plan = self.output/'hosting-plan.json'

    def prepare(self):
        return hosting.prepare(self.plan, self.output)

    def test_prepare_binds_verified_parent_and_preserves_exact_attributes_and_asset_bytes(self):
        result = self.prepare()
        plan, payload = hosting.checked_plan(self.hosting_plan)
        self.assertEqual(plan['expected_remote_commit'], hosting.PARENT)
        self.assertEqual(plan['prior_tree'], self.before)
        self.assertEqual(plan['source_publication_plan_sha256'], sha256(self.plan.read_bytes()).hexdigest())
        self.assertEqual(payload, {**self.asset_bytes, '.gitattributes': self.attributes+hosting.OVERRIDES})
        self.assertEqual(result['files'], sorted(payload))

    def test_prepare_rejects_wrong_receipt_parent_tree_or_asset_content_hash(self):
        original_receipt = (self.plan.parent/'publication-receipt.json').read_bytes()
        (self.plan.parent/'publication-receipt.json').write_bytes(b'{}')
        with self.assertRaisesRegex(hosting.PublicationError, 'exact verified'):
            self.prepare()
        (self.plan.parent/'publication-receipt.json').write_bytes(original_receipt)
        self.info.sha = 'c'*40
        with self.assertRaisesRegex(hosting.PublicationError, 'parent changed'):
            self.prepare()
        self.info.sha = hosting.PARENT
        self.before['extra.md'] = hosting.regular_record(b'unrelated')
        with self.assertRaisesRegex(hosting.PublicationError, 'inventory differs'):
            self.prepare()
        del self.before['extra.md']
        self.before[hosting.ASSETS[0]]['lfs']['sha256'] = 'f'*64
        with self.assertRaisesRegex(hosting.PublicationError, 'asset differs'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_payload_tampering_and_unrelated_hidden_files_are_refused(self):
        self.prepare()
        (self.output/'payload'/hosting.ASSETS[0]).write_bytes(b'changed')
        with self.assertRaisesRegex(hosting.PublicationError, 'payload bytes'):
            hosting.checked_plan(self.hosting_plan)
        (self.output/'payload'/hosting.ASSETS[0]).write_bytes(self.asset_bytes[hosting.ASSETS[0]])
        (self.output/'payload/.gitconfig').write_bytes(b'bad')
        with self.assertRaisesRegex(hosting.PublicationError, 'inventory changed'):
            hosting.checked_plan(self.hosting_plan)

    def test_exact_attributes_append_and_bounds_are_enforced(self):
        self.assertEqual(hosting.attributes_after(b'old-line'), b'old-line\n'+hosting.OVERRIDES)
        for raw in (b'\0', b'\xff', b'x'*(hosting.MAX_ATTRIBUTES_BYTES+1),
                    self.attributes+hosting.ASSETS[0].encode()+b' filter=lfs\n'):
            with self.subTest(raw=raw[:20]), self.assertRaises(hosting.PublicationError):
                hosting.attributes_after(raw)
        with patch.object(hosting, 'MAX_ASSET_BYTES', 1), self.assertRaises(hosting.PublicationError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_explicit_wire_payload_has_only_four_regular_files_and_parent_guard(self):
        payload = {**self.asset_bytes, '.gitattributes': hosting.attributes_after(self.attributes)}
        opener = MagicMock()
        response = opener.open.return_value.__enter__.return_value = Mock()
        response.read.return_value = original.encoded({'success': True, 'commitOid': 'b'*40})
        with patch.object(hosting.urllib.request, 'build_opener', return_value=opener) as build:
            self.assertEqual(hosting.send_regular_commit(payload, 'test-credential'), 'b'*40)
        args, kwargs = opener.open.call_args
        request = args[0]
        self.assertEqual(request.full_url, 'https://huggingface.co/api/spaces/Dvidia/robot-simulation-lab/commit/main')
        redirect_handler = build.call_args.args[0]
        self.assertIsNone(redirect_handler.redirect_request(None, None, None, None, None, None))
        lines = [json.loads(raw) for raw in request.data.splitlines()]
        self.assertEqual(lines[0]['value']['parentCommit'], hosting.PARENT)
        self.assertEqual([line['key'] for line in lines], ['header', 'file', 'file', 'file', 'file'])
        decoded = {line['value']['path']: base64.b64decode(line['value']['content']) for line in lines[1:]}
        self.assertEqual(decoded, payload)

    def test_publish_verifies_regular_storage_all_other_entries_and_public_bytes(self):
        self.prepare()
        with patch.object(hosting, 'send_regular_commit', return_value='b'*40) as send:
            result = hosting.publish(self.hosting_plan)
        send.assert_called_once()
        self.assertEqual(result['commit'], 'b'*40)
        self.assertTrue(result['runtime_asset_bytes_unchanged'])
        self.assertEqual(result['unchanged_other_entries'], len(self.before)-4)
        self.assertEqual(json.loads((self.output/'hosting-publication-receipt.json').read_bytes()), result)

    def test_changed_parent_tree_or_nonmember_blocks_send(self):
        self.prepare()
        with patch.object(hosting, 'send_regular_commit') as send:
            self.info.sha = 'c'*40
            with self.assertRaisesRegex(hosting.PublicationError, 'remote parent'):
                hosting.publish(self.hosting_plan)
            self.info.sha = hosting.PARENT
            self.before['LICENSE'] = hosting.regular_record(b'changed')
            with self.assertRaisesRegex(hosting.PublicationError, 'remote parent'):
                hosting.publish(self.hosting_plan)
            self.api.whoami.return_value = {'orgs': []}
            with self.assertRaisesRegex(hosting.PublicationError, 'not a member'):
                hosting.publish(self.hosting_plan)
            send.assert_not_called()

    def test_post_commit_storage_or_unrelated_changes_never_report_verified(self):
        self.prepare()
        for variant in ('lfs', 'unrelated', 'missing'):
            after = self.remote['b'*40].copy()
            if variant == 'lfs':
                after[hosting.ASSETS[0]] = self.before[hosting.ASSETS[0]]
            elif variant == 'unrelated':
                after['LICENSE'] = hosting.regular_record(b'changed')
            else:
                del after['LICENSE']
            with patch.object(hosting, 'tree', return_value=after):
                plan, payload = hosting.checked_plan(self.hosting_plan)
                with self.subTest(variant=variant), self.assertRaisesRegex(hosting.PublicationError, 'Published tree'):
                    hosting.verify(self.api, plan, payload, 'b'*40)
        self.assertFalse((self.output/'hosting-publication-receipt.json').exists())

    def test_pending_commit_retry_verifies_without_reposting(self):
        self.prepare()
        with patch.object(hosting, 'send_regular_commit', return_value='b'*40), \
                patch.object(hosting, 'verify', side_effect=hosting.PublicationError('download unavailable')):
            with self.assertRaisesRegex(hosting.PublicationError, 'download unavailable'):
                hosting.publish(self.hosting_plan)
        self.info.sha = 'b'*40
        with patch.object(hosting, 'send_regular_commit') as send:
            result = hosting.publish(self.hosting_plan)
            send.assert_not_called()
        self.assertEqual(result['commit'], 'b'*40)


if __name__ == '__main__':
    unittest.main()
