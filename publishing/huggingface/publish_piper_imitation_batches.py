"""Append a reviewed PiPER evidence plan in bounded, journaled Hub commits.

The command uses the same fixed destinations and local plan verifier as
``publish_piper_imitation.py``. It does not accept a token on the command line
or from the environment. The recovery journal is local, atomically replaced,
and bound to the exact reviewed plan and the pinned initial Space tree.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import publish_piper_imitation as study
from publish import PublicationError, checked_remote, client

BATCH_MAX_BYTES = 8 * 1024 * 1024
BATCH_MAX_OPERATIONS = 25
MAX_PLAN_BYTES = 4 * 1024 * 1024
MAX_RETRIES = 3
EXPECTED_SPACE_BASE = 'c635cee2aeec3da8329731580dd7794f7f0e08b2'
JOURNAL_SCHEMA = 'dvidia.piper-imitation-batch-journal/v1'
RECEIPT_SCHEMA = study.KIND + '/receipt-v1'
SHA = re.compile(r'(?:[a-f0-9]{40}|[a-f0-9]{64})\Z')


def _plain(path: Path) -> Path:
    try:
        return study._plain(Path(path))
    except PublicationError as exc:
        raise PublicationError('Batch publication paths may not contain symlinks.') from exc


def _encoded(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _atomic_json(path: Path, value) -> None:
    path = _plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_file():
        raise PublicationError('Batch recovery journal must be a plain file.')
    raw = _encoded(value)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix='.batch-journal-',
                                     delete=False) as stream:
        staged = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Atomic replacement still holds on filesystems that cannot fsync dirs.
        pass


def _plan_sha(path: Path) -> str:
    raw = study._read(path, MAX_PLAN_BYTES)
    return sha256(raw).hexdigest()


def _row_hashes(row: dict) -> dict[str, str]:
    return {item['path']: item['sha256'] for item in row['files']}


def _payload(plan: dict, base: Path, token: str | None) -> dict[tuple[str, str], bytes]:
    result = {}
    for row in plan['repositories']:
        for item in row['files']:
            raw = study._read(base / row['directory'] / study._relative(item['path']))
            if study.receipt_bytes(raw) != {'bytes': item['bytes'], 'sha256': item['sha256']}:
                raise PublicationError('Reviewed artifact changed before upload bytes were frozen.')
            if token and (token.encode() in raw or study.TOKEN.search(raw)):
                raise PublicationError('Possible credential found in a frozen upload artifact.')
            result[row['repo_type'], item['path']] = raw
    return result


def _tree(api, repo_id: str, repo_type: str, revision: str) -> dict:
    tree = {}
    for entry in api.list_repo_tree(repo_id, repo_type=repo_type, revision=revision, recursive=True):
        if not hasattr(entry, 'blob_id'):
            continue
        lfs = getattr(entry, 'lfs', None)
        tree[entry.path] = {
            'blob': entry.blob_id,
            'size': entry.size,
            'lfs_sha256': (lfs.get('sha256') if isinstance(lfs, dict)
                           else getattr(lfs, 'sha256', None)) if lfs else None,
        }
    return tree


def _git_blob(raw: bytes) -> str:
    return sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def _matches(raw: bytes, entry: dict) -> bool:
    return (len(raw) == entry.get('size') and
            (sha256(raw).hexdigest() == entry.get('lfs_sha256') if entry.get('lfs_sha256')
             else _git_blob(raw) == entry.get('blob')))


def _repo_not_found(exc: Exception) -> bool:
    if type(exc).__name__ == 'RepositoryNotFoundError':
        return True
    return getattr(getattr(exc, 'response', None), 'status_code', None) == 404


def _read_info(api, row: dict):
    try:
        info = api.repo_info(row['repo_id'], repo_type=row['repo_type'])
    except Exception as exc:
        if _repo_not_found(exc):
            return None
        raise
    checked_remote(info, row['repo_type'])
    return info


def _download_bytes(path_or_bytes, limit: int) -> bytes:
    if isinstance(path_or_bytes, bytes):
        raw = path_or_bytes
    else:
        path = Path(path_or_bytes)
        if path.stat().st_size > limit:
            raise PublicationError('Anonymous public artifact exceeds its byte bound.')
        with path.open('rb') as stream:
            raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise PublicationError('Anonymous public artifact exceeds its byte bound.')
    return raw


def verify_public_files(row: dict, revision: str, downloader=None) -> None:
    """Verify public bytes at a pinned revision, retrying transient reads only."""
    if downloader is None:
        from huggingface_hub import hf_hub_download

        def downloader(repo_id, path, repo_type, revision):
            return hf_hub_download(repo_id, path, repo_type=repo_type, revision=revision,
                                   token=False, endpoint='https://huggingface.co')

    def verify(item):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                downloaded = downloader(row['repo_id'], item['path'], row['repo_type'], revision)
                raw = _download_bytes(downloaded, item['bytes'])
                if study.receipt_bytes(raw) != {'bytes': item['bytes'], 'sha256': item['sha256']}:
                    raise PublicationError('Published bytes differ from the reviewed plan.')
                return
            except PublicationError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(0.2 * (2 ** attempt))
        raise PublicationError('Anonymous public download failed after bounded retries.') from last_error

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, row['files']))


def _journal_new(plan_hash: str, baselines: dict) -> dict:
    return {'schema': JOURNAL_SCHEMA, 'plan_sha256': plan_hash,
            'space_baseline_commit': EXPECTED_SPACE_BASE,
            'repositories': baselines}


def _read_journal(path: Path, plan_hash: str) -> dict:
    path = _plain(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise PublicationError('Expected the bounded batch recovery journal.')
    try:
        journal = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationError('Batch recovery journal is invalid.') from exc
    if (not isinstance(journal, dict) or journal.get('schema') != JOURNAL_SCHEMA
            or journal.get('plan_sha256') != plan_hash
            or journal.get('space_baseline_commit') != EXPECTED_SPACE_BASE
            or not isinstance(journal.get('repositories'), dict)):
        raise PublicationError('Batch journal belongs to another plan or baseline.')
    return journal


def _baseline_for(row: dict, info, api) -> dict:
    if info is None:
        if row['repo_type'] == 'space':
            raise PublicationError('The fixed append-only Space must exist before publication.')
        return {'repo_id': row['repo_id'], 'repo_type': row['repo_type'],
                'baseline_exists': False, 'baseline_sha': None, 'baseline_tree': {},
                'created_repo_sha': None, 'created_repo_tree': None, 'completed_batches': []}
    if row['repo_type'] == 'space' and info.sha != EXPECTED_SPACE_BASE:
        raise PublicationError('Space base moved from its explicitly reviewed starting commit.')
    tree = _tree(api, row['repo_id'], row['repo_type'], info.sha)
    planned = _row_hashes(row)
    existing = set(tree)
    if row['repo_type'] == 'space':
        if any(path == study.SPACE_NAMESPACE or path.startswith(study.SPACE_NAMESPACE + '/')
               for path in existing):
            raise PublicationError('The fixed Space version namespace already exists.')
    elif existing - set(planned) - {'.gitattributes'}:
        raise PublicationError('Model or dataset repository contains unrelated existing files.')
    return {'repo_id': row['repo_id'], 'repo_type': row['repo_type'],
            'baseline_exists': True, 'baseline_sha': info.sha, 'baseline_tree': tree,
            'created_repo_sha': None, 'created_repo_tree': None, 'completed_batches': []}


def _validate_journal_shape(journal: dict, rows: list[dict]) -> None:
    by_kind = {row['repo_type']: row for row in rows}
    repos = journal['repositories']
    if set(repos) != set(by_kind):
        raise PublicationError('Batch journal repository set differs from the reviewed plan.')
    for kind, row in by_kind.items():
        state = repos[kind]
        if (not isinstance(state, dict) or state.get('repo_id') != row['repo_id']
                or state.get('repo_type') != kind
                or type(state.get('baseline_exists')) is not bool
                or not isinstance(state.get('baseline_tree'), dict)
                or not isinstance(state.get('completed_batches'), list)):
            raise PublicationError('Batch journal repository baseline is invalid.')
        if state['baseline_exists'] and not SHA.fullmatch(state.get('baseline_sha', '') or ''):
            raise PublicationError('Batch journal baseline commit is invalid.')
        if not state['baseline_exists'] and state.get('baseline_sha') is not None:
            raise PublicationError('Missing repository baseline has an unexpected commit.')
        created_sha = state.get('created_repo_sha')
        if created_sha is not None and not SHA.fullmatch(created_sha):
            raise PublicationError('Created repository commit is invalid.')
        if state['baseline_exists'] and created_sha is not None:
            raise PublicationError('Existing repository unexpectedly has a created-repository anchor.')
        if kind == 'space' and (state.get('baseline_sha') != EXPECTED_SPACE_BASE
                                or not state['baseline_exists']):
            raise PublicationError('Batch journal Space baseline differs from the pinned starting commit.')
        known = set(state['baseline_tree'])
        if not state['baseline_exists'] and state.get('created_repo_tree') is not None:
            if not isinstance(state['created_repo_tree'], dict):
                raise PublicationError('Created repository baseline tree is invalid.')
            known = set(state['created_repo_tree'])
        paths = set()
        previous = state.get('created_repo_sha') if not state['baseline_exists'] else state['baseline_sha']
        for entry in state['completed_batches']:
            if (not isinstance(entry, dict) or not SHA.fullmatch(entry.get('commit', '') or '')
                    or entry.get('parent_commit') != previous or type(entry.get('paths')) is not list
                    or not entry['paths'] or type(entry.get('reconciled')) is not bool):
                raise PublicationError('Batch journal commit chain is invalid.')
            if not entry['reconciled'] and len(entry['paths']) > BATCH_MAX_OPERATIONS:
                raise PublicationError('A journaled upload batch exceeds the operation bound.')
            for name in entry['paths']:
                if (not isinstance(name, str) or name not in _row_hashes(row) or name in known or name in paths):
                    raise PublicationError('Batch journal path set is invalid.')
                paths.add(name)
            previous = entry['commit']


def _safe_destination_rows(plan: dict) -> list[dict]:
    rows = plan['repositories']
    if {row.get('repo_type') for row in rows} != {'model', 'dataset', 'space'}:
        raise PublicationError('Plan destinations differ from the fixed model, dataset and Space set.')
    return rows


def _validate_live(row: dict, state: dict, info, api, payload: dict, downloader,
                   *, reconcile: bool, journal: dict | None = None, journal_path: Path | None = None):
    kind = row['repo_type']
    if info is None:
        if state['baseline_exists'] or state.get('created_repo_sha'):
            raise PublicationError('A previously observed repository disappeared.')
        return set(), None
    checked_remote(info, kind)
    current_tree = _tree(api, row['repo_id'], kind, info.sha)
    baseline_tree = state['baseline_tree']
    if state['baseline_exists']:
        baseline_sha = state['baseline_sha']
        if kind == 'space' and baseline_sha != EXPECTED_SPACE_BASE:
            raise PublicationError('Journal Space baseline does not match the pinned commit.')
        if any(current_tree.get(name) != value for name, value in baseline_tree.items()):
            raise PublicationError('A baseline repository path changed or disappeared.')
        if kind == 'space':
            if set(current_tree) - set(baseline_tree) - set(_row_hashes(row)):
                raise PublicationError('Space changed outside the reviewed append namespace.')
        elif set(current_tree) - set(baseline_tree) - set(_row_hashes(row)):
            raise PublicationError('Model or dataset acquired an unrelated path.')
        expected_anchor = baseline_sha
        baseline_paths = set(baseline_tree)
    else:
        created_sha = state.get('created_repo_sha')
        created_tree = state.get('created_repo_tree')
        if not created_sha or not isinstance(created_tree, dict):
            if set(current_tree) != {'.gitattributes'}:
                raise PublicationError('Unjournaled repository creation has unexpected content; manual review required.')
            if reconcile:
                state['created_repo_sha'] = info.sha
                state['created_repo_tree'] = current_tree
                _atomic_json(journal_path, journal)
            return set(), info.sha
        if any(current_tree.get(name) != value for name, value in created_tree.items()):
            raise PublicationError('Created repository metadata changed after its journal baseline.')
        if set(current_tree) - set(created_tree) - set(_row_hashes(row)):
            raise PublicationError('Created model or dataset has unrelated files.')
        expected_anchor = created_sha
        baseline_paths = set(created_tree)

    file_hashes = _row_hashes(row)
    present = set(current_tree) & set(file_hashes)
    if kind == 'space':
        for name in present:
            entry = current_tree.get(name)
            if (not isinstance(entry, dict) or entry.get('lfs_sha256') is not None
                    or not _matches(payload[kind, name], entry)):
                raise PublicationError('New Space files must match regular Git bytes, not LFS pointers.')
    # All present planned paths are read anonymously and hashed at the live pin.
    if present:
        subset = {**row, 'files': [item for item in row['files'] if item['path'] in present]}
        verify_public_files(subset, info.sha, downloader)
    known_paths = {name for entry in state['completed_batches'] for name in entry['paths']}
    added_planned = present - baseline_paths
    unknown_paths = added_planned - known_paths
    if unknown_paths:
        if not reconcile:
            raise PublicationError('Remote planned paths are not journaled; reconcile before continuing.')
        unknown_bytes = sum(len(payload[kind, name]) for name in unknown_paths)
        if len(unknown_paths) > BATCH_MAX_OPERATIONS or unknown_bytes > BATCH_MAX_BYTES:
            raise PublicationError('Unjournaled remote additions exceed one bounded batch; manual review required.')
        prior = state['completed_batches'][-1]['commit'] if state['completed_batches'] else expected_anchor
        state['completed_batches'].append({'parent_commit': prior, 'commit': info.sha,
                                           'paths': sorted(unknown_paths), 'reconciled': True})
        known_paths |= unknown_paths
        _atomic_json(journal_path, journal)
    expected_added = known_paths | (baseline_paths & set(file_hashes))
    if not present <= expected_added:
        raise PublicationError('Remote paths exceed the journaled reviewed artifact set.')
    expected_head = (state['completed_batches'][-1]['commit'] if state['completed_batches']
                     else expected_anchor)
    if added_planned == known_paths and info.sha != expected_head:
        raise PublicationError('Repository head differs from the journaled batch chain.')
    return present, info.sha


def _initial_preflight(rows, api, downloader):
    baselines = {}
    infos = {}
    # Inspect every destination before any create or commit request.
    for row in rows:
        info = _read_info(api, row)
        state = _baseline_for(row, info, api)
        if info is not None:
            present = set(_row_hashes(row)) & set(state['baseline_tree'])
            if present:
                subset = {**row, 'files': [item for item in row['files'] if item['path'] in present]}
                verify_public_files(subset, info.sha, downloader)
        baselines[row['repo_type']] = state
        infos[row['repo_type']] = info
    return baselines, infos


def _batches(row: dict, paths: list[str], payload: dict) -> list[list[str]]:
    batches, current, size = [], [], 0
    for name in paths:
        byte_count = len(payload[row['repo_type'], name])
        if byte_count > BATCH_MAX_BYTES:
            raise PublicationError('A single artifact exceeds the bounded commit size.')
        if current and (len(current) >= BATCH_MAX_OPERATIONS or size + byte_count > BATCH_MAX_BYTES):
            batches.append(current)
            current, size = [], 0
        current.append(name)
        size += byte_count
    if current:
        batches.append(current)
    return batches


def _operations(paths: list[str], row: dict, payload: dict, operation_cls, kind: str):
    result = []
    for name in paths:
        operation = operation_cls(path_in_repo=name, path_or_fileobj=payload[kind, name])
        if kind == 'space':
            operation._upload_mode = 'regular'
            operation._should_ignore = False
        result.append(operation)
    return result


def _verify_space_regular_batch(api, row: dict, revision: str, paths: list[str], payload: dict) -> None:
    tree = _tree(api, row['repo_id'], 'space', revision)
    for name in paths:
        entry = tree.get(name)
        if (not isinstance(entry, dict) or entry.get('lfs_sha256') is not None
                or not _matches(payload['space', name], entry)):
            raise PublicationError('New Space files must match regular Git bytes, not LFS pointers.')


def _publish_with_api(plan_path: Path, plan: dict, api, token: str | None = None,
                      downloader=None, operation_cls=None, reviewed_plan_hash: str | None = None,
                      progress: bool = False) -> list[dict]:
    path = _plain(plan_path)
    plan_hash = _plan_sha(path)
    if reviewed_plan_hash is not None and plan_hash != reviewed_plan_hash:
        raise PublicationError('Reviewed publication plan changed after its offline check.')
    rows = _safe_destination_rows(plan)
    payload = _payload(plan, path.parent, token)
    journal_path = _plain(path.parent / 'piper-imitation-batch-journal.json')
    receipt_path = _plain(path.parent / 'publication-receipt.json')
    if receipt_path.exists():
        raise PublicationError('A publication receipt already exists; refusing to overwrite it.')
    if journal_path.exists():
        journal = _read_journal(journal_path, plan_hash)
        _validate_journal_shape(journal, rows)
        baselines = journal['repositories']
    else:
        baselines, _ = _initial_preflight(rows, api, downloader)
        journal = _journal_new(plan_hash, baselines)
        _validate_journal_shape(journal, rows)
        # Durable before the first remote mutation (repo creation or commit).
        _atomic_json(journal_path, journal)

    # On resume, first reconcile all destinations. A partial remote outcome is
    # accepted only when exact planned bytes are public and the baseline tree
    # remains untouched; no upload is retried blindly.
    current_infos = {}
    present_by_kind = {}
    for row in rows:
        info = _read_info(api, row)
        present, _ = _validate_live(row, baselines[row['repo_type']], info, api, payload, downloader,
                                    reconcile=True, journal=journal, journal_path=journal_path)
        current_infos[row['repo_type']] = info
        present_by_kind[row['repo_type']] = present

    if operation_cls is None:
        from huggingface_hub import CommitOperationAdd
        operation_cls = CommitOperationAdd
    for row in rows:
        kind = row['repo_type']
        info = current_infos[kind]
        state = baselines[kind]
        if info is None:
            # Repo creation was journaled before this first remote write.
            api.create_repo(row['repo_id'], repo_type=kind, private=False, exist_ok=False)
            info = _read_info(api, row)
            if info is None:
                raise PublicationError('Created repository was not visible after creation.')
            if set(_tree(api, row['repo_id'], kind, info.sha)) != {'.gitattributes'}:
                raise PublicationError('New repository contains unexpected starter files.')
            state['created_repo_sha'] = info.sha
            state['created_repo_tree'] = _tree(api, row['repo_id'], kind, info.sha)
            _atomic_json(journal_path, journal)
            current_infos[kind] = info
            present_by_kind[kind] = set()

        # Refresh after repo creation or concurrent writes; compare with the
        # saved baseline before choosing any additions.
        info = _read_info(api, row)
        present, current_head = _validate_live(row, state, info, api, payload, downloader,
                                               reconcile=True, journal=journal,
                                               journal_path=journal_path)
        present_by_kind[kind] = present
        current_infos[kind] = info
        missing = sorted(set(_row_hashes(row)) - present)
        repo_batches = _batches(row, missing, payload)
        for batch_index, batch in enumerate(repo_batches, start=1):
            operations = _operations(batch, row, payload, operation_cls, kind)
            parent = current_head
            try:
                commit = api.create_commit(row['repo_id'], repo_type=kind, operations=operations,
                                           parent_commit=parent,
                                           commit_message='Publish PiPER learned placement evidence (bounded batch)')
            except Exception as exc:
                # The response can be ambiguous. Leave the old journal intact;
                # the next invocation will compare the anonymous tree and hashes.
                raise PublicationError('A batch commit returned an uncertain outcome; rerun the same plan to reconcile.') from exc
            revision = getattr(commit, 'oid', None)
            if not isinstance(revision, str) or not SHA.fullmatch(revision):
                raise PublicationError('Batch commit returned no valid revision; rerun the same plan to reconcile.')
            state['completed_batches'].append({'parent_commit': parent, 'commit': revision,
                                               'paths': batch, 'reconciled': False})
            _atomic_json(journal_path, journal)
            batch_row = {**row, 'files': [item for item in row['files'] if item['path'] in batch]}
            verify_public_files(batch_row, revision, downloader)
            if kind == 'space':
                _verify_space_regular_batch(api, row, revision, batch, payload)
            if progress:
                print(json.dumps({'phase': 'batch-verified', 'repo_type': kind,
                                  'batch': batch_index, 'batch_count': len(repo_batches),
                                  'files': len(batch), 'bytes': sum(len(payload[kind, name]) for name in batch),
                                  'commit': revision}, sort_keys=True), flush=True)
            current_head = revision
            present.update(batch)

    results = []
    for row in rows:
        kind = row['repo_type']
        info = _read_info(api, row)
        if info is None:
            raise PublicationError('A repository disappeared during final verification.')
        present, revision = _validate_live(row, baselines[kind], info, api, payload, downloader,
                                           reconcile=True, journal=journal, journal_path=journal_path)
        if present != set(_row_hashes(row)):
            raise PublicationError('Final repository tree is missing planned publication files.')
        verify_public_files(row, revision, downloader)
        results.append({'repo_id': row['repo_id'], 'repo_type': kind, 'commit': revision,
                        'verified_files': len(row['files']),
                        'url': 'https://huggingface.co/' +
                        ({'model': '', 'dataset': 'datasets/', 'space': 'spaces/'}[kind]) + row['repo_id']})
    receipt = {'schema': RECEIPT_SCHEMA, 'repositories': results,
               'journal_note': ('This release was appended in bounded commits. Its local recovery journal is '
                                'bound to the reviewed plan and pinned Space baseline. Any ambiguous commit '
                                'outcome must be reconciled against the remote tree and anonymous file hashes '
                                'before another batch can proceed.')}
    _atomic_json(receipt_path, receipt)
    return results


def publish(path: Path, *, api=None, downloader=None, operation_cls=None, progress: bool = False):
    path = _plain(path)
    reviewed_plan_hash = _plan_sha(path)
    plan = study.checked_plan(path)
    if _plan_sha(path) != reviewed_plan_hash:
        raise PublicationError('Publication plan changed while its offline check was running.')
    if api is None:
        api, token = client(path.parent / 'publisher-cache')
        identity = api.whoami()
        if 'Dvidia' not in {org.get('name') for org in identity.get('orgs', [])}:
            raise PublicationError('Authenticated account is not a member of Dvidia.')
    else:
        token = None
    return _publish_with_api(plan_path=path, plan=plan, api=api, token=token,
                             downloader=downloader, operation_cls=operation_cls,
                             reviewed_plan_hash=reviewed_plan_hash, progress=progress)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check_parser = commands.add_parser('check')
    check_parser.add_argument('plan', type=Path)
    publish_parser = commands.add_parser('publish')
    publish_parser.add_argument('plan', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'check':
            plan = study.checked_plan(args.plan)
            print(json.dumps({'checked_repositories': [row['repo_id'] for row in plan['repositories']],
                              'network': False, 'credentials_read': False}, indent=2))
        else:
            results = publish(args.plan, progress=True)
            print(json.dumps({'repositories': results}, indent=2, sort_keys=True))
    except KeyboardInterrupt:
        print('Publishing interrupted; rerun the same plan to reconcile exact remote files.', file=sys.stderr)
        return 130
    except Exception as exc:
        if isinstance(exc, PublicationError):
            print(str(exc), file=sys.stderr)
        else:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            print(f'Publishing stopped: {type(exc).__name__}; HTTP status {code}. Raw error suppressed.',
                  file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
