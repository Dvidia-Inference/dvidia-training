"""No-network exact YAML correction and immutable scientific-tree contracts."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fix_grip_clock_catalog_v03 as fix
import publish_grip_clock_v03 as pub
from test_publish_grip_clock_v03 import API, Operation


CARD = (b'---\nlicense: mit\nlibrary_name: dvidia-training\npipeline_tag: robotics\n'
        b'tags:\n- robotics\n- simulation-only\n- piper\n---\n\n'
        b'# Original research body\n\nA literal in the body stays untouched:\n- piper\n')


class CatalogCorrectionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.api = API({'.gitattributes': pub.ATTRIBUTES, 'index.html': b'untouched Space'})
        self.original = {}
        rows = []
        plan_sha = 'f'*64
        for index, kind in enumerate(fix.KINDS):
            attrs_head, final_head = str(index*2+1)*40, str(index*2+2)*40
            science = {'README.md': CARD, 'LICENSE': b'MIT License\nPermission is hereby granted.\n',
                       'science.json': b'{"physical_robot_ready":false,"result":44}',
                       'native.npz': b'large scientific binary content is not read again'}
            integrity = dict(schema='dvidia.piper-grip-clock-payload/v0.3', payload=kind,
                             files={name: pub.receipt(raw) for name, raw in science.items()})
            contents = dict(science, **{'integrity.json': pub.encoded(integrity), '.gitattributes': pub.ATTRIBUTES})
            self.api._add(kind, final_head, contents, [])
            value = self.api.repos[kind]
            value['trees'][final_head]['native.npz'] = pub._entry(contents['native.npz'], lfs=True)
            value['histories'][final_head] = [SimpleNamespace(commit_id='0'*40, title='creation'),
                SimpleNamespace(commit_id=attrs_head, title=pub._message(plan_sha, kind, 0)),
                SimpleNamespace(commit_id=final_head, title=pub._message(plan_sha, kind, 1))]
            tree = deepcopy(value['trees'][final_head])
            attrs_entry = {'.gitattributes': tree['.gitattributes']}
            remaining = {name: entry for name, entry in tree.items() if name != '.gitattributes'}
            commits = [dict(index=0, parent='0'*40, commit=attrs_head, paths=['.gitattributes'], entries=attrs_entry,
                            message=pub._message(plan_sha, kind, 0), anonymous_readback=True, reconciled=False),
                       dict(index=1, parent=attrs_head, commit=final_head, paths=sorted(remaining), entries=remaining,
                            message=pub._message(plan_sha, kind, 1), anonymous_readback=True, reconciled=False)]
            rows.append(dict(repo_id=pub.REPOS[kind], repo_type=kind, commit=final_head, verified_files=len(tree),
                             verified_bytes=sum(len(raw) for raw in contents.values()), commits=commits))
            self.original[kind] = dict(contents=contents, tree=tree, head=final_head)
        self.receipt = self.home/'initial-receipt.json'
        self.receipt.write_bytes(pub.encoded(dict(schema=pub.SCHEMA+'/receipt',
            plan=dict(bytes=1234, sha256=plan_sha), repositories=rows,
            anonymous_full_readback=True, physical_robot_ready=False)))
        self.output = self.home/'staged'
        self.downloads = []

    def download_metadata(self, repo, name, kind, revision):
        self.downloads.append((kind, name, revision))
        if name == 'native.npz':
            raise AssertionError('No multi-gigabyte science readback is permitted.')
        return self.api.download(repo, name, kind, revision)

    def stage(self):
        return fix.stage(self.receipt, self.output, api=self.api, downloader=self.download_metadata)

    def publish(self):
        return fix.publish(self.output/fix.PLAN, api=self.api, downloader=self.download_metadata, operation_cls=Operation)

    def test_stage_uses_six_metadata_reads_and_derives_exact_yaml_only_change(self):
        with patch.object(pub, 'client', side_effect=AssertionError('offline review cannot access credentials')):
            plan = self.stage(); checked, trees, payloads = fix.checked_plan(self.output/fix.PLAN)
        self.assertEqual(plan, checked)
        self.assertEqual(len(self.downloads), 6)
        for kind in fix.KINDS:
            payload = payloads[kind]
            self.assertEqual(set(payload), set(fix.ALLOWED))
            original = self.original[kind]
            self.assertEqual(payload[fix.ORIGINAL_CARD], original['contents']['README.md'])
            self.assertEqual(payload[fix.ORIGINAL_SEAL], original['contents']['integrity.json'])
            self.assertEqual(payload['README.md'], CARD.replace(b'- piper\n', b'- agilex-piper\n', 1))
            self.assertEqual(payload['README.md'].split(b'\n---\n', 1)[1], CARD.split(b'\n---\n', 1)[1])
            self.assertIn(b'library_name: dvidia-training\n', payload['README.md'])
            new_inventory = json.loads(payload['integrity.json'])['files']
            self.assertEqual(set(new_inventory), (set(original['tree'])|set(fix.APPENDED))-{'integrity.json'})
            for name in ('science.json', 'native.npz', 'LICENSE'):
                self.assertEqual(new_inventory[name], json.loads(original['contents']['integrity.json'])['files'][name])
            self.assertEqual(new_inventory['.gitattributes'], pub.receipt(pub.ATTRIBUTES))
            self.assertEqual(trees[kind], original['tree'])

    def test_publish_changes_five_metadata_paths_and_preserves_all_science_and_space(self):
        self.stage(); completed = self.publish()
        self.assertEqual(len(self.api.calls), 2)
        self.assertTrue(completed['full_unchanged_git_lfs_identities_preserved'])
        self.assertFalse(completed['scientific_file_bytes_changed'])
        self.assertEqual(self.api.repos['space']['head'], 'a'*40)
        for kind in fix.KINDS:
            value = self.api.repos[kind]; final = value['trees'][value['head']]
            self.assertEqual(len(final), len(self.original[kind]['tree'])+3)
            for name, entry in self.original[kind]['tree'].items():
                if name not in ('README.md', 'integrity.json'):
                    self.assertEqual(final[name], entry)
                    self.assertEqual(value['contents'][value['head']][name], self.original[kind]['contents'][name])
        self.assertEqual(self.publish(), completed)

    def test_missing_duplicate_or_outside_yaml_tags_are_rejected_and_body_is_unchanged(self):
        invalid = [CARD.replace(b'- piper\n', b'- other\n', 1),
                   CARD.replace(b'- piper\n', b'- piper\n- piper\n', 1),
                   CARD.replace(b'tags:\n', b'datasets:\n'),
                   CARD.replace(b'- piper\n', b'- agilex-piper\n', 1),
                   CARD.replace(b'---\n', b'---\r\n', 1)]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(pub.PublicationError):
                fix.replace_tag(value)

    def test_initial_receipt_namespace_denominator_chain_and_complete_readback_are_required(self):
        original = json.loads(self.receipt.read_bytes())
        mutators = [lambda v: v.update(anonymous_full_readback=False),
                    lambda v: v['repositories'][0].update(repo_id='Dvidia/wrong'),
                    lambda v: v['repositories'][0].update(repo_type='space'),
                    lambda v: v['repositories'][0].update(verified_files=99),
                    lambda v: v['repositories'][0]['commits'][1].update(parent='9'*40),
                    lambda v: v['repositories'][0]['commits'][1].update(anonymous_readback=False),
                    lambda v: v['repositories'][0]['commits'][1].update(message='foreign intent')]
        for mutate in mutators:
            value = deepcopy(original); mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(pub.PublicationError):
                fix._initial(pub.encoded(value))

    def test_initial_parent_head_and_full_tree_must_match_receipt(self):
        self.api.foreign('model')
        with self.assertRaisesRegex(pub.PublicationError, 'Initial final head'):
            self.stage()
        self.assertFalse(self.output.exists())

    def test_original_inventory_sha_lfs_map_and_new_metadata_collision_are_guarded(self):
        kind = 'model'; original = self.original[kind]
        integrity = json.loads(original['contents']['integrity.json'])
        # Bind an inconsistent manifest as its own public Git object to isolate
        # map coverage/LFS checks from the outer cryptographic readback guard.
        for mutate in [lambda v: v['files'].pop('science.json'),
                       lambda v: v['files']['native.npz'].update(sha256='f'*64),
                       lambda v: v['files']['README.md'].update(sha256='f'*64)]:
            changed = deepcopy(integrity); mutate(changed); raw = pub.encoded(changed)
            tree = deepcopy(original['tree']); tree['integrity.json'] = pub._entry(raw)
            with self.subTest(mutate=mutate), self.assertRaises(pub.PublicationError):
                fix.derive(kind, original['head'], tree, CARD, raw, pub.ATTRIBUTES, pub.receipt(self.receipt.read_bytes()))
        tree = deepcopy(original['tree']); tree[fix.OVERLAY] = pub._entry(b'existing')
        with self.assertRaises(pub.PublicationError):
            fix.derive(kind, original['head'], tree, CARD, original['contents']['integrity.json'], pub.ATTRIBUTES, pub.receipt(self.receipt.read_bytes()))

    def test_lost_accepted_reply_reconciles_exact_child_without_duplicate_commit(self):
        self.stage(); self.api.drop_after = True
        with self.assertRaises(pub.PublicationError):
            self.publish()
        state = json.loads((self.output/fix.JOURNAL).read_bytes())
        self.assertTrue(state['repositories']['model']['pending'])
        completed = self.publish()
        self.assertEqual(len(self.api.calls), 2)
        self.assertTrue(completed['repositories'][0]['reconciled'])

    def test_lost_before_acceptance_and_interrupted_readback_preserve_exact_intent(self):
        self.stage(); self.api.drop_before = True
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.api.fail_readback = True
        with self.assertRaises(RuntimeError):
            self.publish()
        completed = self.publish()
        self.assertEqual(len(self.api.calls), 3)
        self.assertTrue(completed['repositories'][0]['reconciled'])

    def test_foreign_child_and_unexpected_science_change_cannot_be_reconciled(self):
        self.stage(); self.api.drop_after = True
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.api.foreign('model'); count = len(self.api.calls)
        with self.assertRaisesRegex(pub.PublicationError, 'exact intended immediate child'):
            self.publish()
        self.assertEqual(len(self.api.calls), count)

    def test_same_commit_unplanned_path_fails_full_tree_preservation(self):
        self.stage(); self.api.extra_path = True
        with self.assertRaisesRegex(pub.PublicationError, 'unexpected original scientific or metadata identities'):
            self.publish()
        self.assertFalse((self.output/fix.RECEIPT).exists())
        self.assertEqual(len(self.api.calls), 1)

    def test_staged_extra_paths_rehashed_content_and_source_snapshot_tampering_are_rejected(self):
        self.stage()
        directory = self.output/'model'/'payload'
        (directory/'unplanned.json').write_bytes(b'{}')
        with self.assertRaisesRegex(pub.PublicationError, 'exact five'):
            fix.checked_plan(self.output/fix.PLAN)
        (directory/'unplanned.json').unlink()
        card_path = directory/'README.md'; original = card_path.read_bytes(); card_path.write_bytes(original+b'body change')
        plan_path = self.output/fix.PLAN; plan = json.loads(plan_path.read_bytes()); old_plan = plan_path.read_bytes()
        plan['repositories'][0]['payload']['README.md'] = pub.receipt(card_path.read_bytes())
        plan_path.write_bytes(pub.encoded(plan))
        with self.assertRaisesRegex(pub.PublicationError, 'YAML-only derivation'):
            fix.checked_plan(plan_path)
        card_path.write_bytes(original); plan_path.write_bytes(old_plan)
        snapshot = self.output/'implementation'/'fix_grip_clock_catalog_v03.py'
        snapshot.write_bytes(snapshot.read_bytes()+b'\n')
        with self.assertRaisesRegex(pub.PublicationError, 'snapshot bytes'):
            fix.checked_plan(plan_path)


if __name__ == '__main__':
    unittest.main()
