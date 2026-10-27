"""Offline tests for the exact PiPER viewer URL correction adapter."""
from hashlib import sha1, sha256
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import publish_piper_imitation as study
import publish_piper_imitation_batches as batches
import publish_piper_imitation_linkfix as linkfix


class Add:
    def __init__(self, *, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


def blob(raw):
    return sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


class FakeHub:
    def __init__(self, plan, stage, *, unknown_once=False):
        self.repos = {}
        self.unknown_once = unknown_once
        self.events = []
        self.corrupt_download = False
        for row in plan['repositories']:
            kind = row['repo_type']
            files = {item['path']: (stage / row['directory'] / item['path']).read_bytes()
                     for item in row['files']}
            if kind == 'space':
                files['README.md'] = b'outside namespace remains untouched\n'
            head = {'model': '1' * 40, 'dataset': '2' * 40, 'space': '3' * 40}[kind]
            self.repos[kind] = {'head': head, 'files': files, 'history': {head: dict(files)},
                                'titles': {head: 'original publication'}}

    def repo_info(self, repo_id, repo_type):
        repo = self.repos[repo_type]
        return SimpleNamespace(repo_id=repo_id, private=False, sha=repo['head'], sdk='static' if repo_type == 'space' else None,
                               siblings=[SimpleNamespace(rfilename=name) for name in repo['files']])

    def list_repo_tree(self, repo_id, repo_type, revision, recursive=True):
        for name, raw in self.repos[repo_type]['history'][revision].items():
            yield SimpleNamespace(path=name, blob_id=blob(raw), size=len(raw), lfs=None)

    def list_repo_commits(self, repo_id, repo_type, revision):
        repo = self.repos[repo_type]
        return [SimpleNamespace(commit_id=revision, title=repo['titles'][revision])]

    def create_commit(self, repo_id, repo_type, operations, parent_commit, commit_message):
        repo = self.repos[repo_type]
        self.events.append(('commit', repo_type, [op.path_in_repo for op in operations]))
        expected_count = 1 if repo_type == 'model' else 6
        if parent_commit != repo['head'] or len(operations) != expected_count:
            raise AssertionError('link-fix commit must contain only the exact fixed path set over the pinned parent')
        files = dict(repo['files'])
        for op in operations:
            files[op.path_in_repo] = op.path_or_fileobj
        revision = sha1((parent_commit + '\0' + commit_message).encode()).hexdigest()
        repo['files'] = files
        repo['head'] = revision
        repo['history'][revision] = dict(files)
        repo['titles'][revision] = commit_message
        if self.unknown_once:
            self.unknown_once = False
            raise RuntimeError('response lost after commit applied')
        return SimpleNamespace(oid=revision)

    def download(self, repo_id, path, repo_type, revision):
        raw = self.repos[repo_type]['history'][revision][path]
        if self.corrupt_download and revision == self.repos[repo_type]['head'] and revision not in {'1' * 40, '2' * 40, '3' * 40}:
            return raw + b'corrupt'
        return raw


def make_fixture(root):
    stage = Path(root)
    old_url = b'https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v01/viewer/'
    frontend = {
        'HF-TRANSPORT.md': b'Frontend projection v02\n',
        'viewer/piper_imitation_viewer.bundle.js': b'new hosted bundle',
        'viewer/piper_imitation_viewer.source.js': b'new hosted source',
    }
    frontend_dir = stage.parent / linkfix.FRONTEND_DIRECTORY
    for name, raw in frontend.items():
        path = frontend_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    viewer_files = {
        'HF-TRANSPORT.md': b'old transport note',
        'README.md': b'static Space card',
        'LICENSE': b'license',
        'piper_imitation_viewer.bundle.js': b'old hosted bundle',
        'piper_imitation_viewer.source.js': b'old hosted source',
    }
    viewer_manifest = {'schema': 'dvidia.piper-imitation-viewer/v1',
                       'scope': 'consistency only',
                       'files': {name: study.receipt_bytes(raw) for name, raw in viewer_files.items()}}
    viewer_manifest_raw = (json.dumps(viewer_manifest, indent=2, sort_keys=True) + '\n').encode()
    root_files = {
        'viewer/HF-TRANSPORT.md': viewer_files['HF-TRANSPORT.md'],
        'viewer/README.md': viewer_files['README.md'],
        'viewer/LICENSE': viewer_files['LICENSE'],
        'viewer/piper_imitation_viewer.bundle.js': viewer_files['piper_imitation_viewer.bundle.js'],
        'viewer/piper_imitation_viewer.source.js': viewer_files['piper_imitation_viewer.source.js'],
        'viewer/viewer-integrity.json': viewer_manifest_raw,
        'sources/piper_imitation_viewer.bundle.js': b'preserved original source snapshot',
    }
    root_manifest = {'schema': 'dvidia.piper-imitation-integrity/v1',
                     'scope': 'consistency only',
                     'files': {name: study.receipt_bytes(raw) for name, raw in root_files.items()}}
    root_manifest_raw = (json.dumps(root_manifest, indent=2, sort_keys=True) + '\n').encode()
    content = {
        'model': {'README.md': b'Intro\n[viewer](' + old_url + b')\n'},
        'dataset': {'README.md': b'Dataset card unchanged.\n'},
        'space': {
            'learned-placement/v01/index.html':
                b'<meta http-equiv="refresh" content="0; url=viewer/">'
                b'<a href="viewer/">Open viewer</a>',
            'learned-placement/v01/integrity.json': root_manifest_raw,
            'learned-placement/v01/viewer/viewer-integrity.json': viewer_manifest_raw,
            'learned-placement/v01/viewer/piper_imitation_viewer.bundle.js': viewer_files['piper_imitation_viewer.bundle.js'],
            'learned-placement/v01/viewer/piper_imitation_viewer.source.js': viewer_files['piper_imitation_viewer.source.js'],
            'learned-placement/v01/viewer/HF-TRANSPORT.md': viewer_files['HF-TRANSPORT.md'],
        },
    }
    repo_names = {'model': linkfix.FIXED_REPOS['model'], 'dataset': linkfix.FIXED_REPOS['dataset'],
                  'space': linkfix.FIXED_REPOS['space']}
    rows = []
    for kind, files in content.items():
        directory = stage / kind
        row_files = []
        for name, raw in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            row_files.append({'path': name, **study.receipt_bytes(raw)})
        rows.append({'repo_id': repo_names[kind], 'repo_type': kind, 'directory': kind,
                     'files': row_files})
    plan = {'repositories': rows}
    plan_path = stage / 'publication-plan.json'
    plan_path.write_text('{"fixture":true}\n')
    receipt = {'schema': batches.RECEIPT_SCHEMA,
               'repositories': [{'repo_id': row['repo_id'], 'repo_type': row['repo_type'],
                                 'commit': {'model': '1' * 40, 'dataset': '2' * 40,
                                            'space': '3' * 40}[row['repo_type']],
                                 'verified_files': len(row['files'])}
                                for row in rows]}
    (stage / 'publication-receipt.json').write_text(json.dumps(receipt))
    return plan_path, plan, frontend


def make_frontend(directory):
    path = Path(directory) / linkfix.FRONTEND_DIRECTORY
    return {name: (path / name).read_bytes() for name in linkfix.FRONTEND_FILES}


class PiperImitationLinkFixTests(unittest.TestCase):
    def _run(self, plan_path, plan, api, frontend):
        with patch.object(linkfix, '_load_plan', return_value=plan), \
             patch.object(batches, '_safe_destination_rows', return_value=plan['repositories']), \
             patch.object(linkfix, 'FRONTEND_SHA256',
                          {name: linkfix._hash(raw) for name, raw in frontend.items()}):
            return linkfix.publish(plan_path, api=api, downloader=api.download, operation_cls=Add)

    def test_exactly_two_links_change_and_space_root_remains_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, frontend = make_fixture(directory)
            api = FakeHub(plan, plan_path.parent)
            original_space_readme = api.repos['space']['files']['README.md']
            original_dataset_head = api.repos['dataset']['head']
            result = self._run(plan_path, plan, api, frontend)
            self.assertEqual(result['viewer_url'], linkfix.VIEWER_URL)
            self.assertEqual([(kind, sorted(paths)) for _, kind, paths in api.events], [
                ('model', ['README.md']), ('space', sorted(linkfix.SPACE_PATHS)),
            ])
            self.assertEqual(api.repos['space']['files']['README.md'], original_space_readme)
            self.assertEqual(api.repos['dataset']['head'], original_dataset_head)
            model = api.repos['model']['files']['README.md']
            self.assertIn(linkfix.VIEWER_URL.encode(), model)
            landing = api.repos['space']['files']['learned-placement/v01/index.html']
            self.assertEqual(landing.count(b'viewer/index.html'), 2)
            self.assertEqual(api.repos['space']['files']['learned-placement/v01/viewer/piper_imitation_viewer.bundle.js'],
                             frontend['viewer/piper_imitation_viewer.bundle.js'])
            result_space = next(row for row in result['repositories'] if row['repo_type'] == 'space')
            self.assertEqual({row['path'] for row in result_space['patches']}, linkfix.SPACE_PATHS)
            self.assertFalse((plan_path.parent / 'publication-linkfix-receipt.json').is_symlink())

    def test_unknown_commit_response_reconciles_without_duplicate_write(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, frontend = make_fixture(directory)
            api = FakeHub(plan, plan_path.parent, unknown_once=True)
            with self.assertRaisesRegex(RuntimeError, 'response lost'):
                self._run(plan_path, plan, api, frontend)
            self._run(plan_path, plan, api, frontend)
            model_commits = [event for event in api.events if event[:2] == ('commit', 'model')]
            self.assertEqual(len(model_commits), 1)
            self.assertEqual(len([event for event in api.events if event[0] == 'commit']), 2)

    def test_unknown_head_with_unrelated_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, frontend = make_fixture(directory)
            api = FakeHub(plan, plan_path.parent, unknown_once=True)
            with self.assertRaisesRegex(RuntimeError, 'response lost'):
                self._run(plan_path, plan, api, frontend)
            repo = api.repos['model']
            parent = repo['head']
            files = dict(repo['files'])
            files['unexpected.txt'] = b'external change'
            head = sha1((parent + 'external').encode()).hexdigest()
            repo['head'] = head
            repo['files'] = files
            repo['history'][head] = dict(files)
            repo['titles'][head] = 'external change'
            with self.assertRaisesRegex(study.PublicationError, 'exact link-fix commit'):
                self._run(plan_path, plan, api, frontend)

    def test_missing_or_duplicate_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, frontend = make_fixture(directory)
            row = next(row for row in plan['repositories'] if row['repo_type'] == 'model')
            source = plan_path.parent / 'model/README.md'
            source.write_bytes(source.read_bytes() + source.read_bytes())
            with patch.object(linkfix, 'FRONTEND_SHA256',
                              {name: linkfix._hash(raw) for name, raw in frontend.items()}):
                with self.assertRaisesRegex(study.PublicationError, 'staged patch target'):
                    linkfix._targets(plan_path, plan, plan_path.parent.parent / linkfix.FRONTEND_DIRECTORY)
                row['files'][0].update(study.receipt_bytes(source.read_bytes()))
                with self.assertRaisesRegex(study.PublicationError, 'expected old URL'):
                    linkfix._targets(plan_path, plan, plan_path.parent.parent / linkfix.FRONTEND_DIRECTORY)

    def test_mutated_frontend_input_and_wrong_parent_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, frontend = make_fixture(directory)
            front_path = plan_path.parent.parent / linkfix.FRONTEND_DIRECTORY / 'viewer/piper_imitation_viewer.source.js'
            front_path.write_bytes(frontend['viewer/piper_imitation_viewer.source.js'] + b'mutated')
            with patch.object(linkfix, 'FRONTEND_SHA256',
                              {name: linkfix._hash(raw) for name, raw in frontend.items()}):
                with self.assertRaisesRegex(study.PublicationError, 'fixed reviewed v02'):
                    linkfix._targets(plan_path, plan)

            plan_path, plan, frontend = make_fixture(Path(directory) / 'second')
            api = FakeHub(plan, plan_path.parent)
            api.repos['space']['head'] = '4' * 40
            with self.assertRaisesRegex(study.PublicationError, 'head changed'):
                self._run(plan_path, plan, api, frontend)

    def test_anonymous_readback_mismatch_does_not_issue_final_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path, plan, frontend = make_fixture(directory)
            api = FakeHub(plan, plan_path.parent)
            api.corrupt_download = True
            with self.assertRaisesRegex(study.PublicationError, 'Anonymous file bytes'):
                self._run(plan_path, plan, api, frontend)
            self.assertFalse((plan_path.parent / 'publication-linkfix-receipt.json').exists())

    def test_anonymous_snapshot_cache_symlink_is_read_as_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / 'payload.txt'
            payload.write_bytes(b'cache content')
            snapshot = root / 'snapshot.txt'
            snapshot.symlink_to(payload)
            linkfix._anonymous(lambda *_: snapshot, 'repo', 'model', 'README.md', '1' * 40,
                               b'cache content')


if __name__ == '__main__':
    unittest.main()
