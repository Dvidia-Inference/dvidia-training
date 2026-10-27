"""Offline recovery and batching tests for the PiPER Hub adapter."""
from hashlib import sha1, sha256
from pathlib import Path
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
                base = {'.gitattributes': b'*.bin filter=lfs diff=lfs merge=lfs -text\n',
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
            assert all(op._upload_mode == 'regular' and op._should_ignore is False for op in operations)
        paths = [op.path_in_repo for op in operations]
        if len(set(paths)) != len(paths) or set(paths) & set(repo['files']):
            raise AssertionError('file overwrite attempted')
        new_files = dict(repo['files'])
        for op in operations:
            new_files[op.path_in_repo] = op.path_or_fileobj
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
    run, output = base / 'run', base / 'stage'
    run.mkdir()
    fixtures._make_run(run)
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

    def test_space_files_must_be_regular_git_blobs(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan = _stage(directory)
            api = FakeHub(plan, plan_path.parent)
            space_row = next(row for row in plan['repositories'] if row['repo_type'] == 'space')
            api.lfs_paths.add(space_row['files'][0]['path'])
            with self.assertRaisesRegex(study.PublicationError, 'regular Git bytes'):
                batches.publish(plan_path, api=api, downloader=api.download, operation_cls=Add)


if __name__ == '__main__':
    unittest.main()
