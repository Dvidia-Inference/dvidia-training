"""Offline reversible metadata derivation and bounded FakeHub publication."""
from copy import deepcopy
from hashlib import sha1, sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import publish_piper_feedback_dataset_metadata as metadata


def encoded(value):
    return metadata.study._encoded(value)


def entry(raw, *, lfs=False):
    return dict(blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(), size=len(raw),
                lfs_sha256=sha256(raw).hexdigest() if lfs else None)


class DatasetMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.dataset, self.output = self.home/'original-staged-dataset', self.home/'metadata-patch'
        self.dataset.mkdir()
        self.parent = 'a'*40
        report_rows = []
        totals = dict(feedback=6, blind=8, feedforward=5, teacher=24, open_jaw=0)
        for case in range(24):
            for variant in metadata.study.VARIANTS:
                report_rows.append(dict(case_id=f'confirmation-{case:02}', variant=variant,
                                        score=dict(valid=True, success=case < totals[variant])))
        self.report = dict(schema=metadata.study.STUDY_SCHEMA, physical_robot_ready=False,
            rows=report_rows, limitations=['Synthetic simulation only.'],
            variants={v: dict(scheduled=24, valid=24, successes=n) for v, n in totals.items()})
        self.files = {'README.md': b'# DVIDIA original scientific README\n\nOriginal evidence stays unchanged.\n',
            'LICENSE': b'MIT License\nOriginal DVIDIA artifacts.\n',
            'report.json': encoded(self.report), 'protocol.json': encoded({'original': 'protocol'}),
            'frozen.json': encoded({'original': 'frozen'}),
            'confirmation/fixture/native.npz': b'original native LFS fixture bytes'}
        self.seal = encoded(dict(schema=metadata.study.INTEGRITY_SCHEMA,
            files={name: metadata.study._receipt(raw) for name, raw in self.files.items()}))
        self.files.update({'integrity.json': self.seal,
            metadata.CARD: metadata.study._card('dataset', self.report)})
        for name, raw in self.files.items():
            path = self.dataset/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        self.remote = {**self.files, '.gitattributes': b'*.npz filter=lfs diff=lfs merge=lfs -text\n'}
        self.tree = {name: entry(raw, lfs=name.endswith('.npz')) for name, raw in self.remote.items()}
        self.baseline = self.home/'tree.json'
        self.baseline.write_bytes(encoded(dict(schema=metadata.BASELINE_SCHEMA, repo_id=metadata.study.DATASET_REPO,
            repo_type='dataset', commit=self.parent, tree=self.tree)))
        self.receipt = self.home/'completed-publication.json'
        self.initial = dict(schema=metadata.study.SCHEMA+'/receipt', anonymous_readback=True,
            physical_robot_ready=False, source_commit='c'*40, plan_sha256='d'*64,
            source_integrity_sha256=sha256(self.seal).hexdigest(),
            bindings={key+'_sha256': sha256(self.files[key+'.json']).hexdigest() for key in ('report', 'protocol', 'frozen')},
            repositories=[dict(repo_type=kind, repo_id=repo, commit=self.parent if kind == 'dataset' else 'b'*40,
                               verified_files=len(self.files) if kind == 'dataset' else 1)
                          for kind, repo in metadata.study.REPOSITORIES.items()])
        self.receipt.write_bytes(encoded(self.initial))

    def stage(self):
        return metadata.stage(self.dataset, self.receipt, self.parent, self.baseline, self.output)

    def test_stage_and_check_are_offline_and_exactly_restore_the_original_scientific_readme(self):
        original = {name: (self.dataset/name).read_bytes() for name in self.files}
        with patch.object(metadata, 'client', side_effect=AssertionError('credentials forbidden')):
            plan = self.stage()
            checked, tree, payload = metadata.checked_plan(self.output/'metadata-plan.json')
        self.assertEqual(checked, plan)
        self.assertEqual(set(payload), metadata.ALLOWED)
        self.assertEqual(payload[metadata.SOURCE_README], self.files['README.md'])
        self.assertTrue(payload['README.md'].startswith(b'---\nlicense: mit\n'))
        self.assertIn(b'docs/publication/source-README.md', payload['README.md'])
        self.assertIn(b'| feedback | 6 / 24 | 24 |', payload['README.md'])
        overlay = json.loads(payload[metadata.OVERLAY])
        self.assertEqual(overlay['restoration_path_mapping'], {'README.md': metadata.SOURCE_README})
        self.assertEqual(overlay['original_scientific_integrity']['sha256'], sha256(self.seal).hexdigest())
        self.assertEqual(overlay['initial_dataset_commit'], self.parent)
        self.assertEqual(original, {name: (self.dataset/name).read_bytes() for name in self.files})
        self.assertEqual(tree, self.tree)

    def test_only_the_derived_card_is_unsealed_and_the_root_mit_license_is_preserved(self):
        self.assertIn('LICENSE', json.loads(self.seal)['files'])
        self.assertTrue(self.files['LICENSE'].startswith(b'MIT License'))
        self.assertNotIn('docs/publication/LICENSE', self.files)
        original_license = (self.dataset/'LICENSE').read_bytes()
        self.stage()
        self.assertEqual((self.dataset/'LICENSE').read_bytes(), original_license)

    def test_unexpected_initial_extra_is_rejected_even_with_matching_tree_and_file_count(self):
        for name in ('docs/publication/LICENSE', 'docs/publication/unexpected.json'):
            raw = self.files['LICENSE'] if name.endswith('LICENSE') else b'{}'
            path = self.dataset/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            baseline = json.loads(self.baseline.read_bytes()); baseline['tree'][name] = entry(raw)
            self.baseline.write_bytes(encoded(baseline))
            initial = deepcopy(self.initial)
            next(row for row in initial['repositories'] if row['repo_type'] == 'dataset')['verified_files'] += 1
            self.receipt.write_bytes(encoded(initial))
            with self.subTest(name=name), self.assertRaisesRegex(metadata.PublicationError, 'inventory'):
                self.stage()
            self.assertFalse(self.output.exists())
            path.unlink()
            self.baseline.write_bytes(encoded(dict(schema=metadata.BASELINE_SCHEMA, repo_id=metadata.study.DATASET_REPO,
                repo_type='dataset', commit=self.parent, tree=self.tree)))
            self.receipt.write_bytes(encoded(self.initial))

    def test_rehashed_non_mit_root_license_is_rejected(self):
        raw = b'Apache License Version 2.0\n'
        (self.dataset/'LICENSE').write_bytes(raw)
        seal = json.loads(self.seal); seal['files']['LICENSE'] = metadata.study._receipt(raw)
        seal_raw = encoded(seal); (self.dataset/'integrity.json').write_bytes(seal_raw)
        initial = deepcopy(self.initial); initial['source_integrity_sha256'] = sha256(seal_raw).hexdigest()
        self.receipt.write_bytes(encoded(initial))
        baseline = json.loads(self.baseline.read_bytes())
        baseline['tree']['LICENSE'] = entry(raw); baseline['tree']['integrity.json'] = entry(seal_raw)
        self.baseline.write_bytes(encoded(baseline))
        with self.assertRaisesRegex(metadata.PublicationError, 'not MIT'):
            self.stage()
        self.assertFalse(self.output.exists())

    def test_stage_rejects_unfinished_receipt_wrong_parent_tree_license_or_counts(self):
        changes = [dict(anonymous_readback=False), dict(source_integrity_sha256='f'*64), dict(repositories=[])]
        for fields in changes:
            self.receipt.write_bytes(encoded({**self.initial, **fields}))
            with self.subTest(fields=fields), self.assertRaises(metadata.PublicationError): self.stage()
            self.assertFalse(self.output.exists())
        self.receipt.write_bytes(encoded(self.initial))
        with self.assertRaises(metadata.PublicationError):
            metadata.stage(self.dataset, self.receipt, 'main', self.baseline, self.output)
        changed = json.loads(self.baseline.read_bytes()); changed['commit'] = 'e'*40
        self.baseline.write_bytes(encoded(changed))
        with self.assertRaises(metadata.PublicationError): self.stage()
        changed['commit'] = self.parent
        # A rehashed non-MIT card still cannot pass the deterministic derivation.
        bad_card = self.files[metadata.CARD].replace(b'license: mit', b'license: apache-2.0')
        (self.dataset/metadata.CARD).write_bytes(bad_card); changed['tree'][metadata.CARD] = entry(bad_card)
        self.baseline.write_bytes(encoded(changed))
        with self.assertRaisesRegex(metadata.PublicationError, 'MIT YAML'): self.stage()

    def test_rehashed_payload_cannot_change_counts_or_source_readme_even_with_new_plan_receipts(self):
        self.stage()
        plan_path = self.output/'metadata-plan.json'
        changed_path = self.output/'payload'/'README.md'
        raw = changed_path.read_bytes().replace(b'| feedback | 6 / 24', b'| feedback | 24 / 24')
        changed_path.write_bytes(raw)
        plan = json.loads(plan_path.read_bytes())
        next(row for row in plan['files'] if row['path'] == 'README.md')['after'] = metadata.study._receipt(raw)
        plan_path.write_bytes(encoded(plan))
        with self.assertRaisesRegex(metadata.PublicationError, 'exact reversible'): metadata.checked_plan(plan_path)

    def test_staged_scientific_tampering_or_added_unsafe_payload_is_rejected(self):
        self.stage()
        (self.dataset/'confirmation/fixture/native.npz').write_bytes(b'changed native data')
        with self.assertRaises(metadata.PublicationError): metadata.checked_plan(self.output/'metadata-plan.json')
        (self.dataset/'confirmation/fixture/native.npz').write_bytes(self.files['confirmation/fixture/native.npz'])
        (self.output/'payload'/'other.json').write_bytes(b'{"changed":true}')
        with self.assertRaisesRegex(metadata.PublicationError, 'exact reversible'):
            metadata.checked_plan(self.output/'metadata-plan.json')

    def test_rehashed_original_report_must_still_match_all_120_actual_outcomes(self):
        report = deepcopy(self.report); report['variants']['feedback']['successes'] = 24
        raw = encoded(report)
        (self.dataset/'report.json').write_bytes(raw)
        seal = json.loads(self.seal); seal['files']['report.json'] = metadata.study._receipt(raw)
        new_seal = encoded(seal); (self.dataset/'integrity.json').write_bytes(new_seal)
        initial = deepcopy(self.initial)
        initial['bindings']['report_sha256'] = sha256(raw).hexdigest()
        initial['source_integrity_sha256'] = sha256(new_seal).hexdigest()
        self.receipt.write_bytes(encoded(initial))
        baseline = json.loads(self.baseline.read_bytes())
        baseline['tree']['report.json'] = entry(raw); baseline['tree']['integrity.json'] = entry(new_seal)
        self.baseline.write_bytes(encoded(baseline))
        with self.assertRaisesRegex(metadata.PublicationError, 'counts differ'): self.stage()
        self.assertFalse(self.output.exists())

    def test_credential_patterns_unsafe_paths_symlinks_and_commit_size_are_rejected_offline(self):
        with patch.object(metadata, 'PRESERVATION', metadata.PRESERVATION+'\nhf_'+'x'*40), self.assertRaisesRegex(metadata.PublicationError, 'credential'):
            self.stage()
        self.assertFalse(self.output.exists())
        with patch.object(metadata, 'MAX_BYTES', 1), self.assertRaisesRegex(metadata.PublicationError, 'bounded'):
            self.stage()
        self.assertFalse(self.output.exists())
        baseline = json.loads(self.baseline.read_bytes()); baseline['tree']['../escape.json'] = entry(b'escape')
        self.baseline.write_bytes(encoded(baseline))
        with self.assertRaises(metadata.PublicationError): self.stage()
        self.baseline.unlink(); self.baseline.symlink_to(self.receipt)
        with self.assertRaisesRegex(metadata.PublicationError, 'symlink'): self.stage()

    def fake_api(self):
        _, baseline, payload = metadata.checked_plan(self.output/'metadata-plan.json')
        owner = self
        class API:
            head = owner.parent
            mutated = False
            calls = []
            def repo_info(self, repo, repo_type):
                return SimpleNamespace(sha=self.head, private=False, siblings=[])
            def list_repo_tree(self, repo, repo_type, revision, recursive=True):
                return [SimpleNamespace(path=name, blob_id=row['blob'], size=row['size'],
                    lfs={'sha256': row['lfs_sha256']} if row['lfs_sha256'] else None)
                    for name, row in self.current_tree().items()]
            def current_tree(self):
                tree = deepcopy(baseline)
                if self.mutated: tree.update({name: entry(raw) for name, raw in payload.items()})
                return tree
            def create_commit(self, repo, repo_type, parent_commit, operations, commit_message):
                owner.assertTrue((owner.output/'metadata-pending.json').is_file())
                owner.assertEqual(repo, metadata.study.DATASET_REPO); owner.assertEqual(repo_type, 'dataset')
                owner.assertEqual(parent_commit, owner.parent)
                owner.assertEqual({op.path_in_repo for op in operations}, metadata.ALLOWED)
                for op in operations:
                    owner.assertEqual(op._upload_mode, 'regular'); owner.assertFalse(op._should_ignore)
                self.calls.append(operations); self.mutated = True; self.head = 'f'*40
                return SimpleNamespace(oid=self.head)
        reads = []
        def download(repo, name, kind, revision):
            owner.assertEqual(kind, 'dataset'); owner.assertEqual(repo, metadata.study.DATASET_REPO)
            reads.append((revision, name))
            return payload[name] if revision == 'f'*40 and name in payload else owner.remote[name]
        return API(), download, reads

    def publish(self, api, downloader):
        return metadata.publish(self.output/'metadata-plan.json', api=api, downloader=downloader,
                                operation_cls=lambda **kw: SimpleNamespace(**kw))

    def test_publish_is_one_bounded_parent_commit_and_preserves_original_lfs_and_all_outside_paths(self):
        self.stage(); api, download, reads = self.fake_api()
        receipt = self.publish(api, download)
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(receipt['commit'], 'f'*40)
        self.assertEqual(receipt['changed_paths'], sorted(metadata.ALLOWED))
        self.assertEqual(receipt['unchanged_original_lfs_objects'], 1)
        self.assertEqual(receipt['original_dataset_files'], len(self.files))
        self.assertTrue(receipt['scientific_seal_unchanged'])
        self.assertTrue(receipt['anonymous_before_and_after_readback'])
        self.assertTrue(any(commit == self.parent for commit, _ in reads))
        self.assertTrue(any(commit == 'f'*40 for commit, _ in reads))
        self.assertFalse((self.output/'metadata-pending.json').exists())
        with self.assertRaisesRegex(metadata.PublicationError, 'already completed'): self.publish(api, download)

    def test_moved_head_private_repo_or_lfs_identity_changes_prevent_any_commit(self):
        self.stage(); api, download, _ = self.fake_api()
        api.head = 'b'*40
        with self.assertRaisesRegex(metadata.PublicationError, 'head/full tree'): self.publish(api, download)
        self.assertFalse(api.calls)
        api.head = self.parent
        with patch.object(api, 'repo_info', return_value=SimpleNamespace(sha=self.parent, private=True, siblings=[])), self.assertRaises(metadata.PublicationError):
            self.publish(api, download)
        original = api.current_tree
        def changed():
            value = original(); value['confirmation/fixture/native.npz']['lfs_sha256'] = '0'*64; return value
        api.current_tree = changed
        with self.assertRaisesRegex(metadata.PublicationError, 'head/full tree'): self.publish(api, download)
        self.assertFalse(api.calls)

    def test_ambiguous_commit_is_durably_pending_and_cannot_blindly_retry_or_read_credentials(self):
        self.stage(); api, download, _ = self.fake_api()
        api.create_commit = lambda *a, **k: (_ for _ in ()).throw(ConnectionError('secret SDK details'))
        with self.assertRaisesRegex(metadata.PublicationError, 'outcome is uncertain'): self.publish(api, download)
        pending = json.loads((self.output/'metadata-pending.json').read_bytes())
        self.assertEqual(pending['parent_commit'], self.parent)
        self.assertEqual({row['path'] for row in pending['files']}, metadata.ALLOWED)
        with patch.object(metadata, 'client', side_effect=AssertionError('must not read credentials')), self.assertRaisesRegex(metadata.PublicationError, 'uncertain outcome'):
            metadata.publish(self.output/'metadata-plan.json')

    def test_postcommit_unrelated_path_or_lfs_pointer_change_is_rejected_and_pending_preserved(self):
        self.stage(); api, download, _ = self.fake_api()
        original = api.current_tree
        def changed():
            tree = original()
            if api.mutated: tree['confirmation/fixture/native.npz']['blob'] = '0'*40
            return tree
        api.current_tree = changed
        with self.assertRaisesRegex(metadata.PublicationError, 'outside the three allowed'): self.publish(api, download)
        self.assertTrue((self.output/'metadata-pending.json').exists())
        self.assertFalse((self.output/'metadata-publication-receipt.json').exists())

    def test_anonymous_before_or_after_readback_failure_never_claims_completion(self):
        self.stage(); api, download, _ = self.fake_api()
        def corrupt(repo, name, kind, revision):
            raw = download(repo, name, kind, revision)
            return raw+b'changed' if revision == 'f'*40 and name == metadata.SOURCE_README else raw
        with self.assertRaisesRegex(metadata.PublicationError, 'Anonymous metadata readback'): self.publish(api, corrupt)
        self.assertTrue((self.output/'metadata-pending.json').exists())
        self.assertFalse((self.output/'metadata-publication-receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
