"""Editorial releases preserve the fixed runtime and every earlier public receipt."""
import json
import unittest
from unittest.mock import patch

import prepare_workstation as original
import prepare_workstation_editorial as prepare
import publish_robot_sim as publisher
import test_prepare_workstation as fixtures


class EditorialTests(unittest.TestCase):
    refresh_base = fixtures.WorkstationPublicationTests.refresh_base

    def setUp(self):
        fixtures.WorkstationPublicationTests.setUp(self)
        self.native_files.update({'index.html': b'<h1>Before</h1><script type="module" src="workstation.bundle.js"></script>',
            'workstation.bundle.js': b'verified mirror loader', 'workstation_worker.bundle.js': b'unchanged physics worker',
            'native-assets.bin.gz': b'unchanged native meshes', 'mujoco.wasm.gz': b'unchanged engine bundle'})
        self.native_files['workstation-integrity.json'] = original.encoded({'schema': 'fixture',
            'scope': 'consistency', 'files': {name: original.record(raw) for name, raw in self.native_files.items()
                                           if name != 'workstation-integrity.json'}})
        self.original.update({original.NAMESPACE+'/'+name: raw for name, raw in self.native_files.items()})
        self.original['provenance.json'] = original.encoded({'prior': 'retained', 'live_workstation': {
            'source_commit': 'a'*40, 'files': {name: original.record(raw) for name, raw in self.native_files.items()},
            'mirror': {'retained': True}, 'download_fix': {'retained': True}, 'evidence': {'retained': True}}})
        for name, raw in self.original.items():
            path = self.space/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.refresh_base()
        self.append = self.root/'append'
        self.append.mkdir()
        self.append_files = {name: b'{}' if name.endswith('.json') else b'Prior public acceptance'
                             for name in sorted(prepare.APPEND_NAMES)}
        for name, raw in self.append_files.items():
            path = self.append/'payload'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.refresh_receipts()
        self.updated = {**self.native_files, 'index.html':
            b'<h1>After</h1><p>Clearer copy.</p><script type="module" src="workstation.bundle.js"></script>'}
        seal = json.loads(self.native_files['workstation-integrity.json'])
        seal['files']['index.html'] = original.record(self.updated['index.html'])
        self.updated['workstation-integrity.json'] = original.encoded(seal)
        for name, raw in self.updated.items():
            (self.workstation/name).write_bytes(raw)

    def refresh_receipts(self):
        digest = original.record(self.plan.read_bytes())['sha256']
        (self.plan.parent/'publication-receipt.json').write_bytes(original.encoded({
            'repo_id': publisher.REPO_ID, 'repo_type': 'space', 'commit': 'b'*40,
            'plan_sha256': digest, 'verified_files': len(self.original)}))
        addition = {'kind': 'dvidia.mirror-public-acceptance-append/v1', 'repo_id': publisher.REPO_ID,
            'parent_commit': 'b'*40, 'source_commit': 'a'*40, 'source_publication_plan_sha256': digest,
            'files': {name: original.record(raw) for name, raw in self.append_files.items()}}
        raw = original.encoded(addition)
        (self.append/'plan.json').write_bytes(raw)
        (self.append/'publication-receipt.json').write_bytes(original.encoded({
            'repo_id': publisher.REPO_ID, 'repo_type': 'space', 'commit': 'c'*40, 'parent_commit': 'b'*40,
            'plan_sha256': original.record(raw)['sha256'], 'runtime_and_sealed_evidence_unchanged': True,
            'unchanged_existing_files': len(self.original)+1,
            'added_files': [{'path': name, **record} for name, record in addition['files'].items()]}))

    def stage(self, **overrides):
        args = dict(workstation=self.workstation, evidence=self.evidence, output=self.root/'output',
                    source_commit='d'*40, expected_remote_commit='c'*40,
                    base_plan=self.plan, append_publication=self.append)
        args.update(overrides)
        with patch('dvidia_training.workstation.inspect_export', return_value={}):
            return prepare.stage(**args)

    def test_reconstructs_complete_parent_and_changes_only_html_seal_provenance(self):
        result = self.stage()
        after = self.root/'output/space'
        permitted = {'provenance.json'} | {original.NAMESPACE+'/'+name for name in prepare.MUTABLE}
        for name, raw in {**self.original, **self.append_files}.items():
            if name not in permitted:
                self.assertEqual((after/name).read_bytes(), raw, name)
        live = json.loads((after/'provenance.json').read_bytes())['live_workstation']
        prior = json.loads(self.original['provenance.json'])['live_workstation']
        for name in ('mirror', 'download_fix', 'evidence'):
            self.assertEqual(live[name], prior[name])
        self.assertEqual(live['files'], {name: original.record(raw) for name, raw in self.updated.items()})
        self.assertEqual(live['editorial']['prior_source_commit'], 'a'*40)
        seal_raw = (after/prepare.EVIDENCE_NAMESPACE/original.EVIDENCE_SEAL).read_bytes()
        self.assertEqual(live['editorial']['integrity'], original.record(seal_raw))
        seal = json.loads(seal_raw)
        self.assertTrue(seal['runtime_and_assets_unchanged'])
        self.assertEqual(seal['files'], {name: original.record(raw) for name, raw in self.evidence_files.items()})
        plan = publisher.checked_plan(self.root/'output/publication-plan.json')
        self.assertEqual(plan['expected_remote_commit'], 'c'*40)
        self.assertEqual(plan['source_commit'], 'd'*40)
        self.assertEqual(plan['remove_files'], [])
        self.assertEqual(result['prior_files_retained'], len(self.original)+3)

    def test_refuses_any_non_html_runtime_change_or_inventory_change(self):
        for name in set(self.native_files)-prepare.MUTABLE:
            (self.workstation/name).write_bytes(b'changed')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'only HTML'):
                self.stage()
            (self.workstation/name).write_bytes(self.updated[name])
        extra = self.workstation/'extra.json'
        extra.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'exact prior runtime inventory'):
            self.stage()
        extra.unlink()
        (self.workstation/'scene.xml').unlink()
        with self.assertRaisesRegex(ValueError, 'exact prior runtime inventory'):
            self.stage()

    def test_refuses_script_and_inline_execution_changes(self):
        for html in (b'<script src="other.js"></script>', b'<button onclick="run()">Go</button>',
                     b'<a href="javascript:run()">Go</a>', b'<script>run()</script>'):
            (self.workstation/'index.html').write_bytes(html)
            with self.subTest(html=html), self.assertRaisesRegex(ValueError, 'scripts|handlers'):
                self.stage()

    def test_seal_can_update_only_exact_html_record(self):
        valid = json.loads(self.updated['workstation-integrity.json'])
        for field, value in (('scope', 'changed'), ('schema', 'changed'), ('files', {})):
            (self.workstation/'workstation-integrity.json').write_bytes(original.encoded({**valid, field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'only the HTML record'):
                self.stage()

    def test_rejects_wrong_remote_parent_and_broken_receipt_chain(self):
        with self.assertRaisesRegex(ValueError, 'reviewed remote parent'):
            self.stage(expected_remote_commit='e'*40)
        for path in (self.plan.parent/'publication-receipt.json', self.append/'plan.json',
                     self.append/'publication-receipt.json'):
            old = path.read_bytes()
            path.write_bytes(b'{}')
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'reviewed remote parent'):
                self.stage()
            path.write_bytes(old)

    def test_rejects_tampered_or_unlisted_acceptance_and_prior_bytes(self):
        name = next(iter(self.append_files))
        path = self.append/'payload'/name
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'published receipt'):
            self.stage()
        path.write_bytes(self.append_files[name])
        extra = self.append/'payload/extra.md'
        extra.write_bytes(b'unlisted')
        with self.assertRaisesRegex(ValueError, 'exact additive inventory'):
            self.stage()
        extra.unlink()
        (self.space/'tasks/prior/report.json').write_bytes(b'changed')
        with self.assertRaises(publisher.PublicationError):
            self.stage()

    def test_refuses_overwriting_historical_editorial_namespace_or_provenance(self):
        for mode in ('namespace', 'provenance'):
            if mode == 'namespace':
                self.original[prepare.EVIDENCE_NAMESPACE+'/README.md'] = b'old evidence'
            else:
                del self.original[prepare.EVIDENCE_NAMESPACE+'/README.md']
                (self.space/prepare.EVIDENCE_NAMESPACE/'README.md').unlink()
                value = json.loads(self.original['provenance.json'])
                value['live_workstation']['editorial'] = {'preserve': True}
                self.original['provenance.json'] = original.encoded(value)
            for name, raw in self.original.items():
                path = self.space/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            self.refresh_base()
            self.refresh_receipts()
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'fresh namespace|historical entries'):
                self.stage()

    def test_rejects_invalid_commits_nested_output_and_private_evidence(self):
        for field in ('source_commit', 'expected_remote_commit'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Exact source'):
                self.stage(**{field: 'main'})
        with self.assertRaisesRegex(ValueError, 'separate'):
            self.stage(evidence=self.workstation)
        for root in (self.workstation, self.evidence, self.plan.parent, self.append):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, 'fresh and outside'):
                self.stage(output=root/'output')
        (self.evidence/'private.json').write_bytes(b'{"path":"/Users/private/item"}')
        with self.assertRaisesRegex(ValueError, 'private path'):
            self.stage()
        self.assertFalse((self.root/'output').exists())

    def test_failed_final_native_inspection_leaves_no_partial_output(self):
        with patch('dvidia_training.workstation.inspect_export', side_effect=[{}, {}, ValueError('native mismatch')]):
            with self.assertRaisesRegex(ValueError, 'native mismatch'):
                prepare.stage(self.workstation, self.evidence, self.root/'output', 'd'*40, 'c'*40,
                              base_plan=self.plan, append_publication=self.append)
        self.assertFalse((self.root/'output').exists())
        self.assertFalse(list(self.root.glob('.workstation-editorial-*')))


if __name__ == '__main__':
    unittest.main()
