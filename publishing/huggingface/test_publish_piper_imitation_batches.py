"""Offline recovery and batching tests for the PiPER Hub adapter."""
from hashlib import sha1, sha256
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import publish_piper_imitation as study
import publish_piper_imitation_batches as batches
import test_publish_piper_imitation as fixtures


class Add:
    def __init__(self, *, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj
        self._upload_mode = None
        self._should_ignore = None


class FakeHub:
    def __init__(self, plan, stage_root, *, unknown_after_apply_once=False):
        self.repos = {}
        self.events = []
        self.batches = []
        self.unknown_after_apply_once = unknown_after_apply_once
        self.lfs_paths = set()
        for row in plan['repositories']:
            kind = row['repo_type']
            base = {}
            if kind == 'model':
                base = {item['path']: (stage_root / row['directory'] / item['path']).read_bytes()
                        for item in row['files']}
                revision = '1' * 40
            elif kind == 'dataset':
                base = {'.gitattributes': b'* filter=lfs diff=lfs merge=lfs -text\n'}
                revision = '2' * 40
            else:
                base = {'.gitattributes': (b'*.bin filter=lfs diff=lfs merge=lfs -text\n'
                                           b'*.gz filter=lfs diff=lfs merge=lfs -text\n'
                                           b'*.npz filter=lfs diff=lfs merge=lfs -text\n'),
                        'README.md': b'pre-existing Space root\n'}
                revision = batches.EXPECTED_SPACE_BASE
            self.repos[kind] = {'row': row, 'files': base, 'head': revision,
                                'history': {revision: dict(base)}, 'parents': {}}

    def repo_info(self, repo_id, repo_type):
        self.events.append(('repo_info', repo_type))
        repo = self.repos[repo_type]
        return SimpleNamespace(repo_id=repo_id, private=False, siblings=[
            SimpleNamespace(rfilename=name) for name in repo['files']],
            sha=repo['head'], sdk='static' if repo_type == 'space' else None)

    def list_repo_tree(self, repo_id, repo_type, revision, recursive=True):
        repo = self.repos[repo_type]
        data = repo['history'][revision]
        for name, raw in data.items():
            yield SimpleNamespace(path=name,
                blob_id=sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest(),
                size=len(raw),
                lfs=SimpleNamespace(sha256=sha256(raw).hexdigest()) if name in self.lfs_paths else None)

    def create_commit(self, repo_id, repo_type, operations, parent_commit, commit_message):
        repo = self.repos[repo_type]
        self.events.append(('commit', repo_type))
        if parent_commit != repo['head']:
            raise AssertionError('commit did not advance from the live parent')
        assert 0 < len(operations) <= batches.BATCH_MAX_OPERATIONS
        raw_total = sum(len(op.path_or_fileobj) for op in operations)
        assert raw_total <= batches.BATCH_MAX_BYTES
        if repo_type == 'space':
            assert all(op._should_ignore is False for op in operations)
        paths = [op.path_in_repo for op in operations]
        if len(set(paths)) != len(paths) or set(paths) & set(repo['files']):
            raise AssertionError('file overwrite attempted')
        new_files = dict(repo['files'])
        for op in operations:
            new_files[op.path_in_repo] = op.path_or_fileobj
            if repo_type == 'space' and op._upload_mode == 'lfs':
                self.lfs_paths.add(op.path_in_repo)
            elif repo_type == 'space':
                self.lfs_paths.discard(op.path_in_repo)
        revision = sha1((parent_commit + '\0' + '\0'.join(paths)).encode()).hexdigest()
        repo['files'] = new_files
        repo['head'] = revision
        repo['history'][revision] = dict(new_files)
        repo['parents'][revision] = parent_commit
        self.events.append(('commit-finished', repo_type, revision))
        self.batches.append({'kind': repo_type, 'parent': parent_commit, 'revision': revision,
                             'paths': paths, 'bytes': raw_total})
        if self.unknown_after_apply_once:
            self.unknown_after_apply_once = False
            raise RuntimeError('simulated transient ReadError after remote commit landed')
        return SimpleNamespace(oid=revision)

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.events.append(('create_repo', repo_type))
        raise AssertionError('fixture repositories are pre-existing')

    def put_external(self, kind, path, raw):
        repo = self.repos[kind]
        files = dict(repo['files'])
        files[path] = raw
        revision = sha1((repo['head'] + '\0external\0' + path).encode()).hexdigest()
        repo['files'] = files
        repo['head'] = revision
        repo['history'][revision] = dict(files)
        repo['parents'][revision] = next(reversed(repo['history']))

    def download(self, repo_id, path, repo_type, revision):
        self.events.append(('download', repo_type, revision, path))
        return self.repos[repo_type]['history'][revision][path]


def _stage(directory):
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    run, output = base / 'run', base / 'stage'
    run.mkdir()
    fixtures._make_run(run)
    viewer = run / 'viewer'
    viewer_files = {path.relative_to(viewer).as_posix(): batches._receipt(path.read_bytes())
                    for path in viewer.rglob('*') if path.is_file()}
    (viewer / 'viewer-integrity.json').write_text(json.dumps({
        'schema': 'dvidia.piper-imitation-viewer/v1', 'scope': 'Artifact consistency only',
        'files': viewer_files,
    }, sort_keys=True), encoding='utf-8')
    # Enough independent source artifacts to require multiple operation batches.
    for index in range(28):
        path = run / 'sources' / f'batch-fixture-{index:02}.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((f'# source fixture {index}\n').encode())
    fixtures._reseal(run)
    plan = study.stage(run, 'f' * 40, output)
    return output / 'publication-plan.json', plan


class PiperImitationBatchPublisherTests(unittest.TestCase):
    def test_batches_are_bounded_preflighted_and_advance_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan = _stage(directory)
            api = FakeHub(plan, plan_path.parent)
            results = batches.publish(plan_path, api=api, downloader=api.download, operation_cls=Add)
            self.assertEqual({result['repo_type'] for result in results}, {'model', 'dataset', 'space'})
            self.assertTrue((plan_path.parent / 'piper-imitation-batch-journal.json').is_file())
            receipt = study._json((plan_path.parent / 'publication-receipt.json').read_bytes())
            self.assertEqual(receipt['schema'], batches.RECEIPT_SCHEMA)
            self.assertIn('bounded commits', receipt['journal_note'])
            for kind in ('dataset', 'space'):
                commits = [item for item in api.batches if item['kind'] == kind]
                self.assertGreater(len(commits), 1)
                for previous, current in zip(commits, commits[1:]):
                    self.assertEqual(current['parent'], previous['revision'])
            self.assertTrue(all(len(item['paths']) <= 25 and item['bytes'] <= 8 * 1024 * 1024
                                for item in api.batches))
            for batch in api.batches:
                finished = api.events.index(('commit-finished', batch['kind'], batch['revision']))
                next_commit_indexes = [i for i, event in enumerate(api.events)
                                       if i > finished and event[0] == 'commit']
                next_commit = min(next_commit_indexes, default=len(api.events))
                verified_paths = {event[3] for event in api.events[finished + 1:next_commit]
                                  if event[0] == 'download' and event[2] == batch['revision']}
                self.assertTrue(set(batch['paths']) <= verified_paths,
                                'each batch must be publicly verified before the next commit')
            mutations = [i for i, event in enumerate(api.events) if event[0] in {'create_repo', 'commit'}]
            self.assertTrue(mutations)
            first_mutation = mutations[0]
            for kind in ('model', 'dataset', 'space'):
                first_check = api.events.index(('repo_info', kind))
                self.assertLess(first_check, first_mutation)

    def test_unknown_outcome_reconciles_exact_files_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan = _stage(directory)
            api = FakeHub(plan, plan_path.parent, unknown_after_apply_once=True)
            with self.assertRaisesRegex(study.PublicationError, 'uncertain outcome'):
                batches.publish(plan_path, api=api, downloader=api.download, operation_cls=Add)
            first_count = len(api.batches)
            self.assertTrue((plan_path.parent / 'piper-imitation-batch-journal.json').exists())
            results = batches.publish(plan_path, api=api, downloader=api.download, operation_cls=Add)
            self.assertEqual(len(results), 3)
            self.assertGreater(len(api.batches), first_count)
            journal = study._json((plan_path.parent / 'piper-imitation-batch-journal.json').read_bytes())
            dataset_records = journal['repositories']['dataset']['completed_batches']
            self.assertTrue(any(record['reconciled'] for record in dataset_records))
            self.assertEqual(len(api.batches), len({(batch['kind'], batch['revision']) for batch in api.batches}))

    def test_space_outside_tree_change_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan = _stage(directory)
            api = FakeHub(plan, plan_path.parent)
            journal_path = plan_path.parent / 'piper-imitation-batch-journal.json'
            plan_hash = batches._plan_sha(plan_path)
            baseline, _ = batches._initial_preflight(plan['repositories'], api, api.download)
            batches._atomic_json(journal_path, batches._journal_new(plan_hash, baseline))
            api.put_external('space', 'historical/unexpected.txt', b'outside the new namespace')
            with self.assertRaisesRegex(study.PublicationError, 'outside the reviewed append namespace'):
                batches.publish(plan_path, api=api, downloader=api.download, operation_cls=Add)
            self.assertFalse(api.batches)

    def test_existing_planned_model_bytes_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan = _stage(directory)
            api = FakeHub(plan, plan_path.parent)
            model = api.repos['model']
            name = plan['repositories'][0]['files'][0]['path']
            model['files'][name] = b'wrong existing bytes'
            model['head'] = '9' * 40
            model['history'][model['head']] = dict(model['files'])
            with self.assertRaisesRegex(study.PublicationError, 'Published bytes differ'):
                batches.publish(plan_path, api=api, downloader=api.download, operation_cls=Add)
            self.assertFalse(api.batches)

    def test_anonymous_transient_download_retries_with_a_fixed_bound(self):
        raw = b'public artifact bytes'
        item = {'path': 'evidence.json', 'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}
        row = {'repo_id': study.DATASET_REPO, 'repo_type': 'dataset', 'files': [item]}
        attempts = []

        def flaky_download(repo_id, path, repo_type, revision):
            attempts.append((repo_id, path, repo_type, revision))
            if len(attempts) < 3:
                raise RuntimeError('simulated transient ReadError')
            return raw

        batches.verify_public_files(row, 'a' * 40, flaky_download)
        self.assertEqual(len(attempts), 3)

    def test_space_lfs_is_required_only_for_pinned_attribute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan = _stage(directory)
            api = FakeHub(plan, plan_path.parent)
            space_row = next(row for row in plan['repositories'] if row['repo_type'] == 'space')
            text_item = next(item for item in space_row['files'] if item['path'].endswith('.json'))
            baselines, _ = batches._initial_preflight(plan['repositories'], api, api.download)
            api.lfs_paths.add(text_item['path'])
            api.put_external('space', text_item['path'],
                             (plan_path.parent / 'space' / text_item['path']).read_bytes())
            payload = batches._payload(plan, plan_path.parent, None)
            with self.assertRaisesRegex(study.PublicationError, 'Git/LFS representation'):
                batches._validate_live(space_row, baselines['space'],
                    batches._read_info(api, space_row), api, payload, api.download,
                    reconcile=False)

    def test_space_binary_artifacts_use_lfs_and_verify_raw_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan = _stage(directory)
            api = FakeHub(plan, plan_path.parent)
            results = batches.publish(plan_path, api=api, downloader=api.download, operation_cls=Add)
            self.assertEqual(len(results), 3)
            tree = batches._tree(api, study.SPACE_REPO, 'space', api.repos['space']['head'])
            for name, raw in api.repos['space']['files'].items():
                if name.endswith(('.gz', '.bin', '.npz')):
                    self.assertIn(name, api.lfs_paths)
                    self.assertEqual(sha256(raw).hexdigest(), tree[name]['lfs_sha256'])


def _frontend(directory):
    root = Path(directory)
    files = {
        'viewer/piper_imitation_viewer.bundle.js': b'fetch("https://huggingface.co/spaces/Dvidia/robot-simulation-lab/resolve/abc/learned-placement/v01/viewer/motion.bin");',
        'viewer/piper_imitation_viewer.source.js': b'// readable transport source\n',
        'HF-TRANSPORT.md': b'Pinned Hub resolve links are used for LFS-backed viewer assets.\n',
    }
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root


class PiperImitationOverlayTests(unittest.TestCase):
    def test_overlay_derivation_preserves_model_dataset_and_limits_space_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original_path, original = _stage(base / 'source')
            overlay_path = base / 'v03'
            overlay = batches.derive_overlay(original_path, _frontend(base / 'frontend'), overlay_path)
            old_rows = {row['repo_type']: row for row in original['repositories']}
            new_rows = {row['repo_type']: row for row in overlay['repositories']}
            self.assertEqual(new_rows['model'], old_rows['model'])
            self.assertEqual(new_rows['dataset'], old_rows['dataset'])
            old_space = {item['path']: item for item in old_rows['space']['files']}
            new_space = {item['path']: item for item in new_rows['space']['files']}
            changed = {name for name in old_space.keys() | new_space.keys()
                       if old_space.get(name) != new_space.get(name)}
            self.assertEqual(changed, batches.OVERLAY_CHANGE_PATHS)
            self.assertEqual(overlay['overlay']['original_plan_sha256'], batches._plan_sha(original_path))
            self.assertEqual(overlay['overlay']['diff_sha256'],
                             json.loads((overlay_path / 'overlay-derivation.json').read_text())['diff_sha256'])

    def test_overlay_journal_migrates_only_after_exact_live_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original_path, original = _stage(base / 'source')
            overlay_path = base / 'v03'
            batches.derive_overlay(original_path, _frontend(base / 'frontend'), overlay_path)
            api = FakeHub(original, original_path.parent)
            baselines, _ = batches._initial_preflight(original['repositories'], api, api.download)
            journal = batches._journal_new(batches._plan_sha(original_path), baselines)
            space_row = next(row for row in original['repositories'] if row['repo_type'] == 'space')
            first = next(item for item in space_row['files'] if item['path'].endswith('.json'))
            raw = (original_path.parent / 'space' / first['path']).read_bytes()
            op = Add(path_in_repo=first['path'], path_or_fileobj=raw)
            op._upload_mode = 'regular'
            op._should_ignore = False
            commit = api.create_commit(study.SPACE_REPO, 'space', [op], batches.EXPECTED_SPACE_BASE, 'fixture batch')
            journal['repositories']['space']['completed_batches'].append({
                'parent_commit': batches.EXPECTED_SPACE_BASE, 'commit': commit.oid,
                'paths': [first['path']], 'reconciled': False,
            })
            batches._atomic_json(original_path.parent / 'piper-imitation-batch-journal.json', journal)
            batches.migrate_overlay_journal(original_path, overlay_path / 'publication-plan.json', api, api.download)
            migrated = batches._read_journal(overlay_path / 'piper-imitation-batch-journal.json',
                                             batches._plan_sha(overlay_path / 'publication-plan.json'))
            self.assertEqual(migrated['migration']['from_plan_sha256'], batches._plan_sha(original_path))
            self.assertEqual(migrated['repositories']['space']['completed_batches'],
                             journal['repositories']['space']['completed_batches'])
            self.assertFalse((original_path.parent / 'publication-receipt.json').exists())
            results = batches.publish(overlay_path / 'publication-plan.json', api=api,
                                      downloader=api.download, operation_cls=Add)
            self.assertEqual({row['repo_type'] for row in results}, {'model', 'dataset', 'space'})
            self.assertTrue((overlay_path / 'publication-receipt.json').is_file())

    def test_overlay_migration_refuses_changed_already_published_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original_path, original = _stage(base / 'source')
            overlay_path = base / 'v03'
            batches.derive_overlay(original_path, _frontend(base / 'frontend'), overlay_path)
            api = FakeHub(original, original_path.parent)
            baselines, _ = batches._initial_preflight(original['repositories'], api, api.download)
            journal = batches._journal_new(batches._plan_sha(original_path), baselines)
            bundle = 'learned-placement/v01/viewer/piper_imitation_viewer.bundle.js'
            raw = (original_path.parent / 'space' / bundle).read_bytes()
            op = Add(path_in_repo=bundle, path_or_fileobj=raw)
            op._upload_mode = 'regular'
            op._should_ignore = False
            commit = api.create_commit(study.SPACE_REPO, 'space',
                                       [op],
                                       batches.EXPECTED_SPACE_BASE, 'fixture bundle')
            journal['repositories']['space']['completed_batches'].append({
                'parent_commit': batches.EXPECTED_SPACE_BASE, 'commit': commit.oid,
                'paths': [bundle], 'reconciled': False,
            })
            batches._atomic_json(original_path.parent / 'piper-imitation-batch-journal.json', journal)
            with self.assertRaisesRegex(study.PublicationError, 'changes a path already present'):
                batches.migrate_overlay_journal(original_path, overlay_path / 'publication-plan.json', api, api.download)
            self.assertFalse((overlay_path / 'piper-imitation-batch-journal.json').exists())


if __name__ == '__main__':
    unittest.main()
