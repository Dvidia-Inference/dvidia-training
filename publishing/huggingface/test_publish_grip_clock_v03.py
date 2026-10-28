"""No-network fixed-destination, scanner and unknown-reply recovery tests."""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import publish_grip_clock_v03 as pub


class NotFound(Exception):
    response = SimpleNamespace(status_code=404)


class Operation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class API:
    def __init__(self, contents):
        self.sequence = 100
        self.repos = {}
        self.calls, self.creations = [], []
        self.drop_before = self.drop_after = self.drop_create = False
        self.extra_path = False
        self.fail_readback = False
        self._add('space', 'a'*40, contents, [])

    def _add(self, kind, revision, contents, history):
        tree = {name: pub._entry(raw) for name, raw in contents.items()}
        self.repos[kind] = dict(head=revision, trees={revision: tree}, contents={revision: dict(contents)},
                               histories={revision: history+[SimpleNamespace(commit_id=revision, title='baseline')]})

    def repo_info(self, repo, repo_type, revision=None):
        assert repo == pub.REPOS[repo_type]
        if repo_type not in self.repos:
            raise NotFound()
        value = self.repos[repo_type]
        return SimpleNamespace(private=False, siblings=[], sdk='static', sha=revision or value['head'])

    def list_repo_tree(self, repo, repo_type, revision, recursive):
        tree = self.repos[repo_type]['trees'][revision]
        return [SimpleNamespace(path=name, blob_id=entry['blob'], size=entry['size'],
                    lfs={'sha256': entry['lfs_sha256']} if entry['lfs_sha256'] else None)
                for name, entry in tree.items()]

    def list_repo_commits(self, repo, repo_type, revision):
        # Intentionally reorder history; reconciliation must use membership.
        return list(reversed(self.repos[repo_type]['histories'][revision]))

    def create_repo(self, repo, repo_type, private, exist_ok):
        assert repo == pub.REPOS[repo_type] and private is False and exist_ok is False
        if repo_type in self.repos:
            raise RuntimeError('already exists')
        self.sequence += 1
        self.creations.append(repo_type)
        self._add(repo_type, f'{self.sequence:040x}', {'.gitattributes': b'*.zip filter=lfs diff=lfs merge=lfs -text\n'}, [])
        if self.drop_create:
            self.drop_create = False
            raise RuntimeError('lost create reply')

    def create_commit(self, repo, repo_type, parent_commit, operations, commit_message):
        value = self.repos[repo_type]
        assert repo == pub.REPOS[repo_type]
        if parent_commit != value['head']:
            raise RuntimeError('parent guard')
        self.calls.append((repo_type, parent_commit, [op.path_in_repo for op in operations]))
        if self.drop_before:
            self.drop_before = False
            raise RuntimeError('lost reply before acceptance')
        tree = deepcopy(value['trees'][parent_commit]); contents = dict(value['contents'][parent_commit])
        for op in operations:
            assert op._should_ignore is False
            assert op._upload_mode in ('lfs', 'regular')
            contents[op.path_in_repo] = op.path_or_fileobj
            tree[op.path_in_repo] = pub._entry(op.path_or_fileobj, lfs=op._upload_mode == 'lfs')
        if self.extra_path:
            contents['unexpected.txt'] = b'foreign'; tree['unexpected.txt'] = pub._entry(b'foreign')
        self.sequence += 1
        revision = f'{self.sequence:040x}'
        value['trees'][revision] = tree; value['contents'][revision] = contents
        value['histories'][revision] = value['histories'][parent_commit]+[SimpleNamespace(commit_id=revision, title=commit_message)]
        value['head'] = revision
        if self.drop_after:
            self.drop_after = False
            raise RuntimeError('lost accepted reply')
        return SimpleNamespace(oid=revision)

    def download(self, repo, name, kind, revision):
        assert repo == pub.REPOS[kind]
        if self.fail_readback:
            self.fail_readback = False
            raise RuntimeError('readback transport interruption')
        return self.repos[kind]['contents'][revision][name]

    def foreign(self, kind):
        value = self.repos[kind]; parent = value['head']; self.sequence += 1
        revision = f'{self.sequence:040x}'
        value['trees'][revision] = dict(value['trees'][parent], **{'foreign.txt': pub._entry(b'foreign')})
        value['contents'][revision] = dict(value['contents'][parent], **{'foreign.txt': b'foreign'})
        value['histories'][revision] = value['histories'][parent]+[SimpleNamespace(commit_id=revision, title='foreign')]
        value['head'] = revision


class V03PublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.contents = {'.gitattributes': pub.ATTRIBUTES, 'README.md': b'preserve root', 'index.html': b'preserve old index',
                         'learned-placement/v02/viewer/index.html': b'preserve v02'}
        self.api = API(self.contents)
        self.baseline = self.home/'baseline.json'
        pub.capture_space('a'*40, self.baseline, api=self.api, downloader=self.api.download)
        self.audit = self.home/'audit.json'
        self.audit.write_bytes(pub.encoded(dict(physical_robot_ready=False, valid=True, mismatches=[])))
        self.inputs = {}
        for kind in pub.REPOS:
            directory = self.home/kind; directory.mkdir(); self.inputs[kind] = directory
            if kind == 'space':
                nested = directory/pub.NAMESPACE/'viewer'; nested.mkdir(parents=True)
                (nested/'index.html').write_bytes(b'<html>Simulation only, saved native replay</html>')
                (nested/'bundle.js').write_bytes(b'const stable = true;')
            else:
                (directory/'README.md').write_bytes(b'---\nlicense: mit\n---\n\nDVIDIA simulation-only research.\n')
                (directory/'LICENSE').write_bytes(b'MIT License\nPermission is hereby granted, free of charge.\n')
                (directory/'model.json').write_bytes(pub.encoded(dict(physical_robot_ready=False, value=1)))
        native = io.BytesIO(); np.savez_compressed(native, tick=np.arange(10, dtype=float))
        (self.inputs['dataset']/'native.npz').write_bytes(native.getvalue())
        (self.inputs['dataset']/'episode.json.gz').write_bytes(gzip.compress(b'{"physical_robot_ready":false}', mtime=0))
        self.output = self.home/'staged'

    def stage(self, selected=('model', 'dataset', 'space'), output=None):
        return pub.stage(*(self.inputs[kind] if kind in selected else None for kind in pub.REPOS),
                         self.baseline, output or self.output, audit=self.audit)

    def publish(self, output=None):
        return pub.publish((output or self.output)/pub.PLAN, api=self.api, downloader=self.api.download, operation_cls=Operation)

    def test_offline_staging_scans_exact_bytes_and_bounded_fixed_destinations(self):
        with patch.object(pub, 'client', side_effect=AssertionError('no credential access')):
            plan = self.stage()
            checked, baseline = pub.checked_plan(self.output/pub.PLAN)
        self.assertEqual(checked, plan)
        self.assertEqual(baseline['commit'], 'a'*40)
        self.assertEqual([row['repo_id'] for row in plan['repositories']], list(pub.REPOS.values()))
        for row in plan['repositories']:
            items = {item['path']: item for item in row['files']}
            for group in row['batches']:
                self.assertLessEqual(len(group), 100)
                self.assertLessEqual(sum(items[name]['bytes'] for name in group), 64*1024**2)
            if row['repo_type'] != 'space':
                self.assertEqual(row['batches'][0], ['.gitattributes'])
        self.assertFalse((self.inputs['model']/'.gitattributes').exists())

    def test_publication_preserves_all_old_space_objects_and_reads_all_new_bytes(self):
        plan = self.stage(); result = self.publish()
        self.assertTrue(result['anonymous_full_readback'])
        self.assertFalse(result['physical_robot_ready'])
        space = self.api.repos['space']
        for name, raw in self.contents.items():
            self.assertEqual(space['contents'][space['head']][name], raw)
        for row in plan['repositories']:
            current = self.api.repos[row['repo_type']]
            for item in row['files']:
                self.assertEqual(pub.receipt(current['contents'][current['head']][item['path']]),
                                 {key: item[key] for key in ('bytes', 'sha256')})
        self.assertEqual(self.publish(), result)

    def test_two_phase_publication_can_pin_dataset_before_space_staging(self):
        self.stage(('model', 'dataset')); first = self.publish()
        self.assertIsNone(first['viewer_url'])
        self.assertEqual(self.api.repos['space']['head'], 'a'*40)
        final_dataset = next(row['commit'] for row in first['repositories'] if row['repo_type'] == 'dataset')
        (self.inputs['space']/pub.NAMESPACE/'viewer'/'dataset.json').write_bytes(pub.encoded({'revision': final_dataset}))
        second_output = self.home/'space-only'
        self.stage(('space',), second_output); second = self.publish(second_output)
        self.assertEqual([row['repo_type'] for row in second['repositories']], ['space'])
        self.assertEqual(len(self.api.creations), 2)

    def test_lost_accepted_reply_reconciles_exact_child_without_duplicate_commit(self):
        plan = self.stage(); self.api.drop_after = True
        with self.assertRaises(pub.PublicationError):
            self.publish()
        pending = json.loads((self.output/pub.JOURNAL).read_bytes())['repositories']['model']['pending']
        self.assertEqual(pending['kind'], 'commit')
        result = self.publish()
        self.assertEqual(len(self.api.calls), sum(len(row['batches']) for row in plan['repositories']))
        self.assertTrue(result['repositories'][0]['commits'][0]['reconciled'])

    def test_lost_before_acceptance_resumes_same_parent_and_lost_creation_is_reconciled(self):
        self.stage(); self.api.drop_create = True
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertEqual(self.api.creations, ['model'])
        self.api.drop_before = True
        with self.assertRaises(pub.PublicationError):
            self.publish()
        result = self.publish()
        self.assertTrue(result['anonymous_full_readback'])
        self.assertEqual(self.api.creations, ['model', 'dataset'])

    def test_interrupted_readback_is_reconciled_and_foreign_children_are_refused(self):
        self.stage(); self.api.fail_readback = True
        with self.assertRaises(RuntimeError):
            self.publish()
        calls = len(self.api.calls)
        self.publish()
        self.assertEqual(len(self.api.calls), calls+4)
        # A fresh plan with an unknown accepted reply followed by a foreign
        # child must stop; history dates or ordering cannot justify recovery.
        other = self.home/'second'; self.api = API(self.contents); self.stage(output=other)
        self.api.drop_after = True
        with self.assertRaises(pub.PublicationError):
            self.publish(other)
        self.api.foreign('model')
        before = len(self.api.calls)
        with self.assertRaisesRegex(pub.PublicationError, 'exact intended immediate child'):
            self.publish(other)
        self.assertEqual(len(self.api.calls), before)

    def test_unexpected_remote_files_and_preexisting_new_repo_are_never_overwritten(self):
        self.stage(); self.api._add('model', 'b'*40, {'README.md': b'preexisting'}, [])
        with self.assertRaisesRegex(pub.PublicationError, 'already exists'):
            self.publish()
        self.assertEqual(self.api.calls, [])
        self.api = API(self.contents); self.api.extra_path = True
        with self.assertRaisesRegex(pub.PublicationError, 'unexpected paths'):
            self.publish()
        self.assertIsNotNone(json.loads((self.output/pub.JOURNAL).read_bytes())['repositories']['model']['pending'])

    def test_final_readback_failure_resumes_without_any_new_commits(self):
        self.stage()
        counts, failed = {}, []
        def interrupted(repo, name, kind, revision):
            key = kind, name
            counts[key] = counts.get(key, 0)+1
            if key == ('model', 'README.md') and counts[key] == 2 and not failed:
                failed.append(True)
                raise RuntimeError('synthetic final readback failure')
            return self.api.download(repo, name, kind, revision)
        with self.assertRaises(RuntimeError):
            pub.publish(self.output/pub.PLAN, api=self.api, downloader=interrupted, operation_cls=Operation)
        calls = len(self.api.calls)
        self.assertFalse((self.output/pub.RECEIPT).exists())
        self.assertTrue(self.publish()['anonymous_full_readback'])
        self.assertEqual(len(self.api.calls), calls)

    def test_tampered_payload_implementation_and_destination_are_rejected(self):
        self.stage()
        target = self.output/'model'/'model.json'; original = target.read_bytes()
        target.write_bytes(b'{"value":2}')
        with self.assertRaises(pub.PublicationError):
            pub.checked_plan(self.output/pub.PLAN)
        target.write_bytes(original)
        implementation = self.output/'implementation'/'publish_grip_clock_v03.py'
        implementation.write_bytes(implementation.read_bytes()+b'\n')
        with self.assertRaisesRegex(pub.PublicationError, 'snapshot bytes'):
            pub.checked_plan(self.output/pub.PLAN)
        implementation.write_bytes(Path(pub.__file__).read_bytes())
        plan = json.loads((self.output/pub.PLAN).read_bytes()); plan['repositories'][0]['repo_id'] = 'Dvidia/wrong'
        (self.output/pub.PLAN).write_bytes(pub.encoded(plan))
        with self.assertRaises(pub.PublicationError):
            pub.checked_plan(self.output/pub.PLAN)

    def test_namespace_scope_cards_private_paths_secrets_and_symlinks_are_rejected(self):
        (self.inputs['space']/'index.html').write_bytes(b'cannot replace root')
        with self.assertRaisesRegex(pub.PublicationError, 'only append'):
            self.stage()
        (self.inputs['space']/'index.html').unlink()
        for name, raw in [('bad.json', b'{"physical_robot_ready":true}'),
                          ('secret.txt', b'hf_'+b'a'*35), ('private.txt', b'/Users/synthetic/private.txt')]:
            with self.subTest(name=name), self.assertRaises(pub.PublicationError):
                pub.scan(raw, name)
        pub.scan(b'scheme = "file://"', 'safe.py')
        with self.assertRaises(pub.PublicationError):
            pub.scan(b'scheme = "file:///Users/synthetic/private"', 'bad.py')
        (self.inputs['model']/'linked.txt').symlink_to(self.audit)
        with self.assertRaises(pub.PublicationError):
            self.stage(output=self.home/'linked-output')

    def test_native_archive_scanner_rejects_objects_nonfinite_and_secret_unicode(self):
        for value, name in [(np.asarray([float('nan')]), 'native.npz'),
                            (np.asarray([{'x': 1}], dtype=object), 'native.npz'),
                            (np.asarray(['hf_'+'a'*35]), 'samples.npz')]:
            output = io.BytesIO()
            np.savez_compressed(output, **{'groups' if value.dtype.kind == 'U' else 'ticks': value})
            with self.subTest(kind=value.dtype.kind), self.assertRaises(pub.PublicationError):
                pub.scan(output.getvalue(), name)
        valid = io.BytesIO(); np.savez_compressed(valid, groups=np.asarray(['train-01']), features=np.zeros((1, 58)))
        pub.scan(valid.getvalue(), 'samples.npz')

    def test_writer_lock_and_new_namespace_baseline_are_enforced(self):
        self.stage()
        with pub._lock(self.output):
            with self.assertRaisesRegex(pub.PublicationError, 'publisher is active'):
                self.publish()
        value = json.loads(self.baseline.read_bytes())
        value['tree'][pub.NAMESPACE+'viewer/index.html'] = pub._entry(b'already exists')
        self.baseline.write_bytes(pub.encoded(value))
        with self.assertRaisesRegex(pub.PublicationError, 'entirely new'):
            self.stage(output=self.home/'old-namespace')

    def test_anonymous_download_copies_cached_symlink_into_bounded_temporary_bytes(self):
        blob = self.home/'public-blob'; blob.write_bytes(b'anonymous pinned bytes')
        cached = self.home/'hub-snapshot'; cached.symlink_to(blob)
        observed = []
        def hub_download(repo, name, **options):
            self.assertEqual(repo, pub.REPOS['dataset'])
            self.assertEqual(name, 'nested/native.json')
            self.assertEqual(options['repo_type'], 'dataset')
            self.assertEqual(options['revision'], 'b'*40)
            self.assertIs(options['token'], False)
            self.assertEqual(options['endpoint'], 'https://huggingface.co')
            directory = Path(options['local_dir']); observed.append(directory)
            target = directory/name; target.parent.mkdir(parents=True)
            shutil.copyfile(cached, target)  # SDK copies the cached blob's bytes.
            self.assertFalse(target.is_symlink())
            return str(target)
        with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(hf_hub_download=hub_download)}):
            self.assertEqual(pub.download(pub.REPOS['dataset'], 'nested/native.json', 'dataset', 'b'*40), blob.read_bytes())
        self.assertTrue(cached.is_symlink())
        self.assertTrue(all(not directory.exists() for directory in observed))

    def test_anonymous_download_rejects_symlink_outside_path_wrong_revision_and_oversize(self):
        outside = self.home/'outside.json'; outside.write_bytes(b'public')
        def linked(repo, name, **options):
            target = Path(options['local_dir'])/name
            target.symlink_to(outside)
            return str(target)
        with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(hf_hub_download=linked)}):
            with self.assertRaisesRegex(pub.PublicationError, 'symlinks'):
                pub.download(pub.REPOS['model'], 'model.json', 'model', 'b'*40)
        with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(hf_hub_download=lambda *a, **k: str(outside))}):
            with self.assertRaisesRegex(pub.PublicationError, 'exact requested artifact path'):
                pub.download(pub.REPOS['model'], 'model.json', 'model', 'b'*40)
        with self.assertRaisesRegex(pub.PublicationError, 'exact revision'):
            pub.download(pub.REPOS['model'], 'model.json', 'model', 'main')
        with self.assertRaisesRegex(pub.PublicationError, 'fixed repository'):
            pub.download('Dvidia/wrong', 'model.json', 'model', 'b'*40)
        def oversized(repo, name, **options):
            target = Path(options['local_dir'])/name
            with target.open('wb') as stream:
                stream.truncate(pub.MAX_FILE+1)
            return str(target)
        with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(hf_hub_download=oversized)}):
            with self.assertRaisesRegex(pub.PublicationError, 'oversized'):
                pub.download(pub.REPOS['model'], 'model.json', 'model', 'b'*40)


if __name__ == '__main__':
    unittest.main()
