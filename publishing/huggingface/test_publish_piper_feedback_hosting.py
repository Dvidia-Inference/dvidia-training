"""Offline four-file derivation and fake-Hub publication boundary checks."""
from copy import deepcopy
from hashlib import sha1, sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import publish_piper_feedback_hosting as hosting


def encoded(value):
    return hosting.study._encoded(value)


def entry(raw, *, lfs=None):
    return dict(blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(), size=len(raw), lfs_sha256=lfs)


class HostingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.viewer, self.baseline = self.home/'viewer', self.home/'baseline'
        self.viewer.mkdir(); self.baseline.mkdir()
        self.dataset_commit, self.space_commit = 'd'*40, 'a'*40
        rows = []
        for start in range(6):
            for goal in (0, 1):
                for condition in ('nominal', 'jaw_interruption'):
                    case = f'confirmation-{start:02}-goal{goal}-{condition}'
                    for variant in hosting.study.VARIANTS:
                        name = f'confirmation/{case}/{variant}/native.npz'
                        rows.append(dict(id=case, condition=condition, variant=variant, score={'success': False},
                            evidence={'native.npz': hosting.OLD_NATIVE_ROOT+name,
                                      'native_npz_sha256': sha256(name.encode()).hexdigest(), 'native_npz_bytes': 23}))
        selected = [row for row in rows if row['variant'] == 'feedback'] + [row for row in rows[:5] if row['variant'] != 'feedback']
        motion = dict(schema=hosting.study.VIEWER_SCHEMA, cases=selected, all_attempts=rows,
                      native_dataset=dict(repo_id=hosting.study.DATASET_REPO, revision='main',
                                          publication_readback_verified=False, note='Original configured link targets.'))
        report = dict(variants={name: dict(successes=count, scheduled=24, valid=24)
                               for name, count in [('feedback', 6), ('blind', 8), ('feedforward', 5), ('teacher', 24), ('open_jaw', 0)]})
        self.source = {'motion.json': encoded(motion), 'report.json': encoded(report), 'geometry.bin': b'native fixture geometry'}
        seal = dict(schema=hosting.study.VIEWER_SCHEMA,
                    files={name: hosting._receipt(raw) for name, raw in self.source.items()})
        self.source['viewer-integrity.json'] = encoded(seal)
        for name, raw in self.source.items():
            (self.viewer/name).write_bytes(raw)
        self.parent_seal = sha256(self.source['viewer-integrity.json']).hexdigest()
        self.remote = {hosting.PREFIX+name: raw for name, raw in self.source.items()}
        self.remote.update({'index.html': b'<!doctype html><html><body><main><p>Existing simulations.</p></main></body></html>',
                            'README.md': b'---\nsdk: static\n---\n# Existing DVIDIA simulations\n',
                            'other-study/unchanged.txt': b'preserve this evidence'})
        self.tree = {name: entry(raw) for name, raw in self.remote.items()}
        for name in ('index.html', 'README.md'):
            (self.baseline/name).write_bytes(self.remote[name])
        (self.baseline/'tree.json').write_bytes(encoded(dict(schema=hosting.BASELINE_SCHEMA,
            repo_id=hosting.study.SPACE_REPO, repo_type='space', commit=self.space_commit, tree=self.tree)))
        self.output = self.home/'patch'

    def stage(self):
        return hosting.stage(self.viewer, self.dataset_commit, self.space_commit, self.baseline, self.output,
                             expected_viewer_sha256=self.parent_seal)

    def test_stage_and_check_are_offline_and_only_pin_links_plus_root_exposure(self):
        original = {p.name: p.read_bytes() for p in self.viewer.iterdir()}
        with patch.object(hosting, 'client', side_effect=AssertionError('credentials forbidden')):
            plan = self.stage()
            checked, _, payload = hosting.checked_plan(self.output/'hosting-plan.json')
        self.assertEqual(checked, plan)
        self.assertEqual(set(payload), hosting.ALLOWED)
        self.assertLessEqual(sum(map(len, payload.values())), hosting.MAX_PATCH_BYTES)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.viewer.iterdir()})
        motion = json.loads(payload[hosting.MOTION])
        self.assertEqual(motion['native_dataset']['revision'], self.dataset_commit)
        self.assertFalse(motion['native_dataset']['publication_readback_verified'])
        self.assertEqual(len(plan['native_files']), 120)
        self.assertIn(b'learner without live feedback 8/24', payload['index.html'])
        self.assertIn(b'learned-placement/v02/viewer/index.html', payload['index.html'])
        self.assertTrue(payload['README.md'].startswith(self.remote['README.md'].rstrip()))

    def test_stage_rejects_wrong_commits_seal_and_baseline_bytes(self):
        with self.assertRaises(hosting.PublicationError):
            hosting.stage(self.viewer, 'main', self.space_commit, self.baseline, self.output, expected_viewer_sha256=self.parent_seal)
        with self.assertRaisesRegex(hosting.PublicationError, 'seal differs'):
            hosting.stage(self.viewer, self.dataset_commit, self.space_commit, self.baseline, self.output, expected_viewer_sha256='0'*64)
        (self.baseline/'index.html').write_bytes(b'changed')
        with self.assertRaisesRegex(hosting.PublicationError, 'Root baseline bytes'):
            self.stage()
        self.assertFalse(self.output.exists())

    def test_rehashed_scientific_change_in_motion_cannot_pass_derivation_check(self):
        self.stage()
        path = self.output/'payload'/hosting.MOTION
        motion = json.loads(path.read_bytes()); motion['all_attempts'][0]['score']['success'] = True
        path.write_bytes(encoded(motion))
        plan_path = self.output/'hosting-plan.json'
        plan = json.loads(plan_path.read_bytes())
        for row in plan['files']:
            if row['path'] == hosting.MOTION:
                row['after'] = hosting._receipt(path.read_bytes())
        plan_path.write_bytes(encoded(plan))
        with self.assertRaisesRegex(hosting.PublicationError, 'beyond the exact dataset-link pin'):
            hosting.checked_plan(plan_path)

    def fake_api(self):
        plan, baseline, payload = hosting.checked_plan(self.output/'hosting-plan.json')
        native = {name: dict(blob='0'*40, size=row['bytes'], lfs_sha256=row['sha256']) for name, row in plan['native_files'].items()}
        owner = self
        class API:
            head = owner.space_commit
            mutated = False
            calls = []
            def repo_info(self, repo, repo_type, revision=None):
                return SimpleNamespace(sha=owner.dataset_commit if repo_type == 'dataset' else self.head,
                    private=False, siblings=[], sdk='static')
            def list_repo_tree(self, repo, repo_type, revision, recursive=True):
                tree = native if repo_type == 'dataset' else self.current_tree()
                return [SimpleNamespace(path=name, blob_id=row['blob'], size=row['size'],
                        lfs={'sha256': row['lfs_sha256']} if row['lfs_sha256'] else None) for name, row in tree.items()]
            def current_tree(self):
                tree = deepcopy(baseline)
                if self.mutated:
                    tree.update({name: entry(raw) for name, raw in payload.items()})
                return tree
            def create_commit(self, repo, repo_type, parent_commit, operations, commit_message):
                self.calls.append((repo, parent_commit, operations))
                owner.assertEqual(parent_commit, owner.space_commit)
                owner.assertEqual({op.path_in_repo for op in operations}, hosting.ALLOWED)
                self.mutated = True; self.head = 'b'*40
                return SimpleNamespace(oid=self.head)
        def download(repo, name, kind, revision):
            return payload[name] if revision == 'b'*40 else owner.remote[name]
        return API(), download

    def test_publish_is_one_parent_bound_commit_with_full_tree_and_anonymous_readback(self):
        self.stage()
        api, download = self.fake_api()
        receipt = hosting.publish(self.output/'hosting-plan.json', api=api, downloader=download,
                                  operation_cls=lambda **kwargs: SimpleNamespace(**kwargs))
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(receipt['commit'], 'b'*40)
        self.assertEqual(receipt['dataset_native_lfs_objects_checked'], 120)
        self.assertTrue(receipt['anonymous_before_and_after_readback'])
        self.assertFalse((self.output/'hosting-pending.json').exists())

    def test_moved_head_or_wrong_pinned_native_object_prevents_any_commit(self):
        self.stage()
        api, download = self.fake_api()
        api.head = 'c'*40
        with self.assertRaisesRegex(hosting.PublicationError, 'head/full tree'):
            hosting.publish(self.output/'hosting-plan.json', api=api, downloader=download,
                            operation_cls=lambda **kwargs: SimpleNamespace(**kwargs))
        self.assertFalse(api.calls)
        api.head = self.space_commit
        original = api.list_repo_tree
        def changed(repo, repo_type, revision, recursive=True):
            values = original(repo, repo_type, revision, recursive)
            if repo_type == 'dataset':
                values[0].lfs = {'sha256': 'f'*64}
            return values
        api.list_repo_tree = changed
        with self.assertRaisesRegex(hosting.PublicationError, 'native LFS object'):
            hosting.publish(self.output/'hosting-plan.json', api=api, downloader=download,
                            operation_cls=lambda **kwargs: SimpleNamespace(**kwargs))
        self.assertFalse(api.calls)

    def test_ambiguous_commit_keeps_exact_pending_request_and_never_blindly_retries(self):
        self.stage()
        api, download = self.fake_api()
        api.create_commit = lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError('private SDK details'))
        with self.assertRaisesRegex(hosting.PublicationError, 'outcome is uncertain'):
            hosting.publish(self.output/'hosting-plan.json', api=api, downloader=download,
                            operation_cls=lambda **kwargs: SimpleNamespace(**kwargs))
        self.assertTrue((self.output/'hosting-pending.json').is_file())
        with self.assertRaisesRegex(hosting.PublicationError, 'uncertain outcome'):
            hosting.publish(self.output/'hosting-plan.json', api=api, downloader=download,
                            operation_cls=lambda **kwargs: SimpleNamespace(**kwargs))

    def test_postcommit_tree_change_outside_the_four_paths_is_rejected(self):
        self.stage()
        api, download = self.fake_api()
        original = api.current_tree
        def changed():
            value = original()
            if api.mutated:
                value['other-study/unchanged.txt'] = entry(b'changed unrelated evidence')
            return value
        api.current_tree = changed
        with self.assertRaisesRegex(hosting.PublicationError, 'outside the four allowed'):
            hosting.publish(self.output/'hosting-plan.json', api=api, downloader=download,
                            operation_cls=lambda **kwargs: SimpleNamespace(**kwargs))
        self.assertFalse((self.output/'hosting-publication-receipt.json').exists())
        self.assertTrue((self.output/'hosting-pending.json').is_file())


if __name__ == '__main__':
    unittest.main()
