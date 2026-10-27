"""Publish the fixed hosted PiPER frontend and links as two bounded repo commits.

The plan and original publication receipt must already exist. This adapter only
changes the fixed three-file frontend projection, its two derived manifests,
the versioned Space landing page, and the model README. Credentials stay in the
existing publisher client and are never accepted or printed here.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import publish_piper_imitation as study
import publish_piper_imitation_batches as batches
from publish import PublicationError, checked_remote, client

SCHEMA = 'dvidia.piper-imitation-linkfix/v1'
RECEIPT_SCHEMA = SCHEMA + '/receipt-v1'
VIEWER_URL = 'https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v01/viewer/index.html'
COMMIT_MESSAGE_PREFIX = 'PiPER viewer transport and URL correction'
FRONTEND_DIRECTORY = 'piper-imitation-hf-frontend-v02'
FRONTEND_FILES = {'HF-TRANSPORT.md', 'viewer/piper_imitation_viewer.bundle.js',
                  'viewer/piper_imitation_viewer.source.js'}
FRONTEND_SHA256 = {
    'HF-TRANSPORT.md': '5c87d55e3ec29e8da6a9a6c0516fb549f695278abb7b36f294698d588eb52e90',
    'viewer/piper_imitation_viewer.bundle.js': '64ccf3b1d9391cf4c4b6bb258e6b27700c146244f1e9c82b0601117fb83ea87e',
    'viewer/piper_imitation_viewer.source.js': '9358fba0c71f73813f959355b601a294ddff174431617c768e14594d546f6c77',
}
SPACE_PATHS = {
    'learned-placement/v01/index.html',
    'learned-placement/v01/integrity.json',
    'learned-placement/v01/viewer/viewer-integrity.json',
    'learned-placement/v01/viewer/piper_imitation_viewer.bundle.js',
    'learned-placement/v01/viewer/piper_imitation_viewer.source.js',
    'learned-placement/v01/viewer/HF-TRANSPORT.md',
}
MODEL_PATHS = {'README.md'}
MODEL_LINK_OLD = b'https://dvidia-robot-simulation-lab.static.hf.space/learned-placement/v01/viewer/'
MODEL_LINK_NEW = VIEWER_URL.encode()
SPACE_LINK_OLD = b'viewer/'
SPACE_LINK_NEW = b'viewer/index.html'
FIXED_REPOS = {
    'model': 'Dvidia/piper-block-placement-v01',
    'dataset': 'Dvidia/piper-block-placement-demonstrations-v01',
    'space': 'Dvidia/robot-simulation-lab',
}
SHA = re.compile(r'(?:[a-f0-9]{40}|[a-f0-9]{64})\Z')


def _hash(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _read_bytes(path: Path, limit: int = 1024 * 1024) -> bytes:
    path = study._plain(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise PublicationError('A bounded plain link-fix artifact is required.')
    return path.read_bytes()


def _atomic_json(path: Path, value: dict) -> None:
    path = study._plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix='.piper-linkfix-', delete=False) as out:
        staged = Path(out.name)
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    os.replace(staged, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _load_plan(plan_path: Path) -> dict:
    plan_path = study._plain(plan_path)
    plan = batches.checked_overlay_plan(plan_path)
    if {row['repo_type']: row['repo_id'] for row in plan['repositories']} != FIXED_REPOS:
        raise PublicationError('Link correction is restricted to the three fixed PiPER repositories.')
    return plan


def _publication_receipt(plan_path: Path, plan: dict) -> tuple[dict, str]:
    path = plan_path.parent / 'publication-receipt.json'
    raw = _read_bytes(path, 1024 * 1024)
    receipt = json.loads(raw)
    if receipt.get('schema') != batches.RECEIPT_SCHEMA:
        raise PublicationError('A completed bounded PiPER publication receipt is required.')
    expected = {row['repo_type']: row for row in plan['repositories']}
    actual = {row.get('repo_type'): row for row in receipt.get('repositories', [])}
    if set(actual) != set(FIXED_REPOS):
        raise PublicationError('Publication receipt does not cover the fixed PiPER destinations.')
    for kind, row in expected.items():
        item = actual[kind]
        if (item.get('repo_id') != FIXED_REPOS[kind] or item.get('verified_files') != len(row['files'])
                or not isinstance(item.get('commit'), str) or not SHA.fullmatch(item['commit'])):
            raise PublicationError('Publication receipt differs from the reviewed PiPER plan.')
    return receipt, _hash(raw)


def _manifest_update(raw: bytes, schema: str, changes: dict[str, bytes]) -> bytes:
    value = json.loads(raw)
    if value.get('schema') != schema or not isinstance(value.get('files'), dict):
        raise PublicationError('A staged frontend integrity manifest has an unexpected schema.')
    files = dict(value['files'])
    for name, content in changes.items():
        if name not in files:
            raise PublicationError('A frontend manifest is missing a fixed projection path.')
        files[name] = study.receipt_bytes(content)
    value['files'] = files
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _targets(plan_path: Path, plan: dict, frontend_dir: Path | None = None) -> tuple[dict[str, dict[str, dict]], list[dict]]:
    rows = {row['repo_type']: row for row in plan['repositories']}
    frontend_dir = frontend_dir or (plan_path.parent.parent / FRONTEND_DIRECTORY)
    frontend_dir = study._plain(frontend_dir)
    frontend_inventory = set()
    for current in frontend_dir.rglob('*'):
        if current.is_symlink():
            raise PublicationError('The hosted frontend projection may not contain symlinks.')
        if current.is_file():
            frontend_inventory.add(current.relative_to(frontend_dir).as_posix())
    if frontend_inventory != FRONTEND_FILES:
        raise PublicationError('The hosted frontend directory must contain only its three fixed files.')
    frontend = {name: _read_bytes(frontend_dir / name, 2 * 1024 * 1024) for name in FRONTEND_FILES}
    frontend_receipts = [{'path': name, **study.receipt_bytes(frontend[name])} for name in sorted(frontend)]
    if any(_hash(frontend[name]) != FRONTEND_SHA256[name] for name in FRONTEND_FILES):
        raise PublicationError('The frontend input differs from the fixed reviewed v02 projection.')

    def staged(kind: str, name: str) -> bytes:
        row = rows[kind]
        item = next((f for f in row['files'] if f['path'] == name), None)
        if item is None:
            raise PublicationError('A fixed frontend patch target is missing from the reviewed plan.')
        original = _read_bytes(plan_path.parent / row['directory'] / name, 2 * 1024 * 1024)
        if study.receipt_bytes(original) != {'bytes': item['bytes'], 'sha256': item['sha256']}:
            raise PublicationError('A staged patch target differs from its reviewed publication hash.')
        return original

    model_card = staged('model', 'README.md')
    if model_card.count(MODEL_LINK_OLD) != 1:
        raise PublicationError('The model-card viewer link does not have exactly one expected old URL.')
    model_new = model_card.replace(MODEL_LINK_OLD, MODEL_LINK_NEW)
    if model_new.count(VIEWER_URL.encode()) != 1:
        raise PublicationError('The model card does not resolve to the explicit viewer index file.')

    space_path = 'learned-placement/v01/'
    root_manifest_path = space_path + 'integrity.json'
    viewer_manifest_path = space_path + 'viewer/viewer-integrity.json'
    bundle_path = space_path + 'viewer/piper_imitation_viewer.bundle.js'
    source_path = space_path + 'viewer/piper_imitation_viewer.source.js'
    note_path = space_path + 'viewer/HF-TRANSPORT.md'
    landing_path = space_path + 'index.html'
    old_bundle = staged('space', bundle_path)
    old_source = staged('space', source_path)
    old_note = staged('space', note_path)
    if old_bundle == frontend['viewer/piper_imitation_viewer.bundle.js'] or old_source == frontend['viewer/piper_imitation_viewer.source.js']:
        raise PublicationError('The hosted frontend inputs are not a distinct reviewed projection.')
    viewer_manifest = staged('space', viewer_manifest_path)
    viewer_new = _manifest_update(viewer_manifest, 'dvidia.piper-imitation-viewer/v1', {
        'piper_imitation_viewer.bundle.js': frontend['viewer/piper_imitation_viewer.bundle.js'],
        'piper_imitation_viewer.source.js': frontend['viewer/piper_imitation_viewer.source.js'],
        'HF-TRANSPORT.md': frontend['HF-TRANSPORT.md'],
    })
    root_manifest = staged('space', root_manifest_path)
    root_new = _manifest_update(root_manifest, 'dvidia.piper-imitation-integrity/v1', {
        'viewer/piper_imitation_viewer.bundle.js': frontend['viewer/piper_imitation_viewer.bundle.js'],
        'viewer/piper_imitation_viewer.source.js': frontend['viewer/piper_imitation_viewer.source.js'],
        'viewer/HF-TRANSPORT.md': frontend['HF-TRANSPORT.md'],
        'viewer/viewer-integrity.json': viewer_new,
    })
    landing = staged('space', landing_path)
    if landing.count(SPACE_LINK_OLD) != 2:
        raise PublicationError('The versioned landing page does not have the two expected viewer links.')
    landing_new = landing.replace(SPACE_LINK_OLD, SPACE_LINK_NEW)

    outputs = {
        'model': {'README.md': (model_card, model_new)},
        'space': {
            landing_path: (landing, landing_new), root_manifest_path: (root_manifest, root_new),
            viewer_manifest_path: (viewer_manifest, viewer_new), bundle_path:
                (old_bundle, frontend['viewer/piper_imitation_viewer.bundle.js']),
            source_path: (old_source, frontend['viewer/piper_imitation_viewer.source.js']),
            note_path: (old_note, frontend['HF-TRANSPORT.md']),
        },
    }
    if set(outputs['space']) != SPACE_PATHS or set(outputs['model']) != MODEL_PATHS:
        raise PublicationError('Link-fix path set differs from the fixed allowlist.')
    targets = {}
    for kind, files in outputs.items():
        targets[kind] = {}
        for name, (before, after) in files.items():
            # Every original and replacement is tied to the sealed v03 inventory;
            # frontend replacements are the only bytes sourced outside that run.
            if name in {bundle_path, source_path, note_path}:
                pass
            else:
                staged(kind, name)
            targets[kind][name] = {'path': name, 'before_sha256': _hash(before),
                                   'after_sha256': _hash(after), 'before': before, 'after': after,
                                   'bytes': len(after)}
    return targets, frontend_receipts


def _anonymous(downloader, repo_id: str, kind: str, path: str, revision: str, expected: bytes) -> None:
    if downloader is None:
        from huggingface_hub import hf_hub_download

        def downloader(repo_id, path, repo_type, revision):
            return hf_hub_download(repo_id, path, repo_type=repo_type, revision=revision,
                                   token=False, endpoint='https://huggingface.co')
    downloaded = downloader(repo_id, path, kind, revision)
    if isinstance(downloaded, bytes):
        actual = downloaded
    else:
        # Hugging Face's snapshot cache may return a symlink to its content
        # store; this is only an anonymous readback path, not a publish input.
        actual = batches._download_bytes(downloaded, 2 * 1024 * 1024)
    if actual != expected:
        raise PublicationError('Anonymous file bytes differ from the exact staged or patched card.')
    if len(actual) != len(expected) or _hash(actual) != _hash(expected):
        raise PublicationError('Anonymous file digest differs from the fixed expected SHA-256.')


def _tree(api, repo_id: str, kind: str, revision: str) -> dict:
    return batches._tree(api, repo_id, kind, revision)


def _same_except(old: dict, new: dict, paths: set[str]) -> bool:
    return ({k: v for k, v in old.items() if k not in paths}
            == {k: v for k, v in new.items() if k not in paths})


def _journal_path(plan_path: Path) -> Path:
    return plan_path.parent / 'piper-imitation-linkfix-journal.json'


def _journal_new(plan_path: Path, plan: dict, pub_receipt: dict, receipt_sha: str,
                 targets: dict, infos: dict, frontend_receipts: list[dict]) -> dict:
    parents = {row['repo_type']: row['commit'] for row in pub_receipt['repositories']}
    return {
        'schema': SCHEMA,
        'plan_sha256': batches._plan_sha(plan_path),
        'publication_receipt_sha256': receipt_sha,
        'frontend_inputs': frontend_receipts,
        'publication_heads': parents,
        'targets': {kind: {'repo_id': FIXED_REPOS[kind], 'paths': {
                           name: {'before_sha256': target['before_sha256'],
                                  'after_sha256': target['after_sha256']}
                           for name, target in paths.items()},
                           'parent_commit': parents[kind], 'observed_head': infos[kind].sha,
                           'commit': None, 'verified': False}
                    for kind, paths in targets.items()},
    }


def _validate_journal(journal: dict, plan_path: Path, receipt_sha: str, targets: dict,
                      frontend_receipts: list[dict], publication_heads: dict[str, str]) -> None:
    if (journal.get('schema') != SCHEMA or journal.get('plan_sha256') != batches._plan_sha(plan_path)
            or journal.get('publication_receipt_sha256') != receipt_sha
            or journal.get('frontend_inputs') != frontend_receipts
            or journal.get('publication_heads') != publication_heads
            or set(journal.get('targets', {})) != {'model', 'space'}):
        raise PublicationError('Link-fix journal is not bound to this reviewed publication.')
    for kind, paths in targets.items():
        item = journal['targets'][kind]
        expected_paths = {name: {'before_sha256': target['before_sha256'],
                                 'after_sha256': target['after_sha256']}
                          for name, target in paths.items()}
        if (item.get('repo_id') != FIXED_REPOS[kind] or item.get('paths') != expected_paths
                or item.get('parent_commit') != journal['publication_heads'].get(kind)):
            raise PublicationError('Link-fix journal target differs from the exact reviewed patch.')
        commit = item.get('commit')
        if (not isinstance(item.get('verified'), bool)
                or (commit is not None and (not isinstance(commit, str) or not SHA.fullmatch(commit)))):
            raise PublicationError('Link-fix journal has an invalid target commit state.')


def _tree_has_patch(api, kind: str, parent: str, head: str, targets: dict[str, dict], plan_sha: str) -> bool:
    expected_title = f'{COMMIT_MESSAGE_PREFIX} {plan_sha[:12]} {kind}'
    commits = api.list_repo_commits(FIXED_REPOS[kind], repo_type=kind, revision=head)
    if not commits or getattr(commits[0], 'commit_id', None) != head or getattr(commits[0], 'title', None) != expected_title:
        return False
    before = _tree(api, FIXED_REPOS[kind], kind, parent)
    after = _tree(api, FIXED_REPOS[kind], kind, head)
    paths = set(targets)
    if not _same_except(before, after, paths):
        return False
    return all(isinstance(after.get(name), dict) and batches._matches(target['after'], after[name])
               for name, target in targets.items())


def publish(plan_path: Path, *, api=None, downloader=None, operation_cls=None) -> dict:
    plan_path = study._plain(plan_path)
    plan = _load_plan(plan_path)
    pub_receipt, receipt_sha = _publication_receipt(plan_path, plan)
    targets, frontend_receipts = _targets(plan_path, plan)
    rows = {row['repo_type']: row for row in batches._safe_destination_rows(plan)}
    # The root Space README and every dataset artifact are deliberately outside this patch.
    journal_path = study._plain(_journal_path(plan_path))
    receipt_path = study._plain(plan_path.parent / 'publication-linkfix-receipt.json')
    if api is None:
        # The existing client scopes cache, token, logging, and telemetry
        # settings before importing the Hub SDK or doing any downloads.
        api, token = client(plan_path.parent / 'publisher-cache')
        identity = api.whoami()
        if 'Dvidia' not in {org.get('name') for org in identity.get('orgs', [])}:
            raise PublicationError('Authenticated account is not a member of Dvidia.')
    else:
        token = None
    if receipt_path.exists():
        existing = json.loads(_read_bytes(receipt_path, 1024 * 1024))
        if (existing.get('schema') != RECEIPT_SCHEMA
                or existing.get('publication_receipt_sha256') != receipt_sha
                or existing.get('plan_sha256') != batches._plan_sha(plan_path)
                or existing.get('frontend_inputs') != frontend_receipts):
            raise PublicationError('A different link-fix receipt already exists; refusing replacement.')
        existing_rows = {item.get('repo_type'): item for item in existing.get('repositories', [])}
        if set(existing_rows) != {'model', 'space'}:
            raise PublicationError('Existing link-fix receipt does not cover the fixed destinations.')
        if not journal_path.is_file():
            raise PublicationError('Existing link-fix receipt has no bound recovery journal.')
        journal = json.loads(_read_bytes(journal_path, 1024 * 1024))
        original_heads = {row['repo_type']: row['commit'] for row in pub_receipt['repositories']}
        _validate_journal(journal, plan_path, receipt_sha, targets, frontend_receipts, original_heads)
        if journal['targets']['model']['commit'] != existing_rows['model'].get('commit') or \
                journal['targets']['space']['commit'] != existing_rows['space'].get('commit'):
            raise PublicationError('Existing link-fix receipt commits differ from its recovery journal.')
        if not all(journal['targets'][kind]['verified'] for kind in ('model', 'space')):
            raise PublicationError('Existing link-fix receipt is not backed by verified journal targets.')
        for kind, paths in targets.items():
            repo = existing_rows[kind]
            info = api.repo_info(FIXED_REPOS[kind], repo_type=kind)
            checked_remote(info, kind)
            if info.sha != repo.get('commit'):
                raise PublicationError('Live head has changed since the prior link-fix readback receipt.')
            if not _tree_has_patch(api, kind, journal['targets'][kind]['parent_commit'], info.sha,
                                   paths, journal['plan_sha256']):
                raise PublicationError('Existing link-fix commit changed files outside the exact path allowlist.')
            receipt_paths = {row.get('path'): row for row in repo.get('patches', [])}
            if set(receipt_paths) != set(paths):
                raise PublicationError('Existing link-fix receipt path set differs from the fixed allowlist.')
            for name, target in paths.items():
                item = receipt_paths[name]
                if item.get('before_sha256') != target['before_sha256'] or item.get('sha256') != target['after_sha256']:
                    raise PublicationError('Existing link-fix receipt hashes differ from the reviewed patch.')
                _anonymous(downloader, FIXED_REPOS[kind], kind, name, info.sha, target['after'])
        dataset_info = api.repo_info(FIXED_REPOS['dataset'], repo_type='dataset')
        if dataset_info.sha != original_heads['dataset']:
            raise PublicationError('Dataset head changed after the link-fix publication.')
        return existing
    # Recheck the complete reviewed export anonymously at the exact receipt
    # revisions before any destination can be edited. The scoped client above
    # ensures the SDK cannot bind an implicit global token or cache first.
    for kind, row in rows.items():
        revision = next(item['commit'] for item in pub_receipt['repositories'] if item['repo_type'] == kind)
        batches.verify_public_files(row, revision, downloader)
    infos = {}
    for kind, row in rows.items():
        info = api.repo_info(row['repo_id'], repo_type=kind)
        checked_remote(info, kind)
        infos[kind] = info
    if journal_path.exists():
        journal = json.loads(_read_bytes(journal_path, 1024 * 1024))
        publication_heads = {row['repo_type']: row['commit'] for row in pub_receipt['repositories']}
        _validate_journal(journal, plan_path, receipt_sha, targets, frontend_receipts, publication_heads)
        if infos['dataset'].sha != publication_heads['dataset']:
            raise PublicationError('Dataset head changed after the completed publication receipt.')
    else:
        for kind, info in infos.items():
            expected_head = next(row['commit'] for row in pub_receipt['repositories'] if row['repo_type'] == kind)
            if info.sha != expected_head:
                raise PublicationError('A repository head changed after the completed publication receipt.')
            for name, target in targets.get(kind, {}).items():
                _anonymous(downloader, FIXED_REPOS[kind], kind, name, info.sha, target['before'])
        journal = _journal_new(plan_path, plan, pub_receipt, receipt_sha, targets, infos,
                               frontend_receipts)
        batches._atomic_json(journal_path, journal)

    if operation_cls is None:
        from huggingface_hub import CommitOperationAdd
        operation_cls = CommitOperationAdd
    results = []
    for kind, paths in targets.items():
        state = journal['targets'][kind]
        info = api.repo_info(FIXED_REPOS[kind], repo_type=kind)
        checked_remote(info, kind)
        if info.sha == state['parent_commit']:
            if state.get('commit'):
                raise PublicationError('Link-fix head returned to its original parent after a recorded commit.')
            for name, target in paths.items():
                _anonymous(downloader, FIXED_REPOS[kind], kind, name, info.sha, target['before'])
            operations = [operation_cls(path_in_repo=name, path_or_fileobj=target['after'])
                          for name, target in paths.items()]
            if not 0 < len(operations) <= batches.BATCH_MAX_OPERATIONS or \
                    sum(len(target['after']) for target in paths.values()) > batches.BATCH_MAX_BYTES:
                raise PublicationError('Link-fix commit exceeds the fixed operation or byte bound.')
            commit_msg = f'{COMMIT_MESSAGE_PREFIX} {batches._plan_sha(plan_path)[:12]} {kind}'
            commit = api.create_commit(FIXED_REPOS[kind], repo_type=kind, operations=operations,
                                       parent_commit=state['parent_commit'], commit_message=commit_msg)
            revision = getattr(commit, 'oid', None)
            if not isinstance(revision, str) or not SHA.fullmatch(revision):
                raise PublicationError('Link-fix commit returned no valid revision; rerun to reconcile.')
            state['commit'] = revision
            batches._atomic_json(journal_path, journal)
            info = api.repo_info(FIXED_REPOS[kind], repo_type=kind)
            if info.sha != revision:
                raise PublicationError('Repository head changed after the link-fix commit.')
        elif state.get('commit') and info.sha == state['commit']:
            revision = state['commit']
        else:
            # Recover a lost commit response only when the live tree differs from
            # its receipt parent at the exact allowlisted paths and contains
            # all expected bytes. Every other tree entry remains unchanged.
            if not _tree_has_patch(api, kind, state['parent_commit'], info.sha, paths,
                                   journal['plan_sha256']):
                raise PublicationError('Live head is not the exact link-fix commit; refusing another write.')
            revision = info.sha
            state['commit'] = revision
            batches._atomic_json(journal_path, journal)
        if not _tree_has_patch(api, kind, state['parent_commit'], revision, paths,
                               journal['plan_sha256']):
            raise PublicationError('Committed tree differs outside the exact link-fix path allowlist.')
        for name, target in paths.items():
            _anonymous(downloader, FIXED_REPOS[kind], kind, name, revision, target['after'])
        state['verified'] = True
        batches._atomic_json(journal_path, journal)
        results.append({'repo_id': FIXED_REPOS[kind], 'repo_type': kind,
                        'parent_commit': state['parent_commit'], 'commit': revision,
                        'patches': [{'path': name, 'bytes': target['bytes'],
                                     'before_sha256': target['before_sha256'],
                                     'sha256': target['after_sha256'], 'anonymous_verified': True}
                                    for name, target in paths.items()]})

    receipt = {'schema': RECEIPT_SCHEMA, 'publication_receipt_sha256': receipt_sha,
               'plan_sha256': batches._plan_sha(plan_path),
               'repositories': results, 'frontend_inputs': frontend_receipts,
               'viewer_url': VIEWER_URL,
               'scope': 'Fixed hosted frontend projection, derived manifests, and explicit viewer index links.'}
    batches._atomic_json(receipt_path, receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args(argv)
    try:
        result = publish(args.plan)
        print(json.dumps(result, sort_keys=True, indent=2))
    except KeyboardInterrupt:
        print('Link correction interrupted; rerun the same plan to reconcile exact public bytes.', file=sys.stderr)
        return 130
    except Exception as exc:
        if isinstance(exc, PublicationError):
            print(str(exc), file=sys.stderr)
        else:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            print(f'Link correction stopped: {type(exc).__name__}; HTTP status {code}. Raw error suppressed.',
                  file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
