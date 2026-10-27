"""Reviewed continuation of an exited, fully verified initial publisher.

The original publication plan, journal, source seal and payload are immutable.
Only this separate continuation's plan/journal/receipt are written. Logical
batches retain their original order and membership; consecutive batches may
share one <=64 MiB / 100-file parent-bound physical commit. All new bytes are
read back anonymously. Only ``publish`` accesses the named publishing client.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import capture_piper_feedback_baselines as public
import publish_piper_feedback as original
import publish_piper_feedback_hosting as hosting
from publish import PublicationError, checked_remote, client
from publish_piper_imitation_batches import _matches


SCHEMA = 'dvidia.piper-feedback-continuation/v1'
MAX_BYTES, MAX_FILES = 64*1024*1024, 100
JOURNAL = 'continuation-journal.json'
PLAN = 'continuation-plan.json'
IMPLEMENTATION_ROOT = Path(__file__).resolve().parent
IMPLEMENTATION_FILES = (
    'publish_piper_feedback_continuation.py', 'capture_piper_feedback_baselines.py',
    'publish_piper_feedback.py', 'publish_piper_feedback_hosting.py',
    'publish_piper_feedback_dataset_metadata.py', 'publish_piper_imitation_batches.py',
    'publish_piper_imitation.py', 'publish_prerequisites.py', 'publish.py', 'prepare.py',
)


def _implementation():
    """Exact local publisher sources, including the reused local import graph."""
    return dict(schema=SCHEMA+'/implementation', files={
        name: original._receipt(original._read(IMPLEMENTATION_ROOT/name)) for name in IMPLEMENTATION_FILES})


def _check_implementation(expected, snapshot_directory=None):
    if expected != _implementation():
        raise PublicationError('Continuation runtime implementation changed after staging.')
    if snapshot_directory is not None:
        directory = original._plain(snapshot_directory)
        files = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
        if files != set(IMPLEMENTATION_FILES):
            raise PublicationError('Continuation implementation snapshot inventory changed.')
        for name, receipt in expected['files'].items():
            if original._receipt(original._read(directory/name)) != receipt:
                raise PublicationError('Continuation implementation snapshot bytes changed.')


def _dead(pid):
    if type(pid) is not int or not 1 <= pid < 2**31:
        raise PublicationError('An explicit exited original writer PID is required.')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except (PermissionError, OSError):
        raise PublicationError('Original writer exit cannot be established; refusing continuation.') from None
    raise PublicationError('Original writer is still alive; wait for a verified exit boundary.')


def _shape(tree, head):
    hosting._tree_document(dict(schema=hosting.BASELINE_SCHEMA, repo_id=original.SPACE_REPO,
        repo_type='space', commit=head, tree=tree), head)
    return tree


def _head(state):
    return state['batches'][-1]['commit'] if state['batches'] else state.get('created_sha') or state.get('baseline_sha')


def _prefix(row, state):
    if state.get('pending') is not None:
        raise PublicationError('Original journal has a pending outcome; reconcile it before continuation.')
    logical = original._batches(row, {item['path'] for item in row['files']})
    if (not isinstance(_head(state), str) or original.COMMIT.fullmatch(_head(state)) is None
            or len(state['batches']) > len(logical)
            or any(batch['paths'] != logical[index] for index, batch in enumerate(state['batches']))):
        raise PublicationError('Original journal must contain an exact consecutive verified logical-batch prefix.')
    _shape(state['baseline_tree'], state.get('created_sha') or state.get('baseline_sha'))
    return logical, len(state['batches'])


def _original(plan_path, journal_path, pid):
    plan_path, journal_path = map(original._plain, (plan_path, journal_path))
    _dead(pid)
    plan_raw, journal_raw = original._read(plan_path, original.MAX_PLAN), original._read(journal_path, original.MAX_PLAN)
    plan = original.checked_plan(plan_path)
    if journal_path.parent != plan_path.parent or journal_path.name != original.JOURNAL_NAME:
        raise PublicationError('Original journal must be the original publication plan sibling.')
    if (plan_path.parent/'publication-receipt.json').exists():
        raise PublicationError('Initial publication is already complete; continuation is unnecessary.')
    journal = original._load_journal(journal_path, plan, sha256(plan_raw).hexdigest())
    for row in plan['repositories']:
        _prefix(row, journal['repositories'][row['repo_type']])
    if original._read(plan_path, original.MAX_PLAN) != plan_raw or original._read(journal_path, original.MAX_PLAN) != journal_raw:
        raise PublicationError('Original plan or journal changed during continuation review.')
    _dead(pid)
    return plan, journal, plan_raw, journal_raw


def _representation(row, name, raw, entry, old_state):
    if not isinstance(entry, dict) or not _matches(raw, entry):
        raise PublicationError('Public Git/LFS object does not match its original manifest bytes.')
    if row['repo_type'] == 'space' and bool(entry['lfs_sha256']) != any(name.endswith(s) for s in old_state['lfs_suffixes']):
        raise PublicationError('Space representation differs from the original pinned Git/LFS policy.')


def _boundary_check(document, plan, journal, plan_raw, journal_raw, pid, source):
    if (not isinstance(document, dict) or set(document) != {'schema', 'original_plan', 'original_journal',
            'stopped_writer_pid', 'anonymous_public_trees', 'repositories'}
            or document['schema'] != SCHEMA+'/boundary'
            or document['original_plan'] != original._receipt(plan_raw)
            or document['original_journal'] != original._receipt(journal_raw)
            or document['stopped_writer_pid'] != pid or document['anonymous_public_trees'] is not True
            or not isinstance(document['repositories'], dict)
            or set(document['repositories']) != set(original.REPOSITORIES)):
        raise PublicationError('Boundary must bind the exact exited writer, original plan/journal and all public trees.')
    for row in plan['repositories']:
        kind = row['repo_type']; state = journal['repositories'][kind]
        baseline = document['repositories'][kind]
        if (not isinstance(baseline, dict) or set(baseline) != {'repo_id', 'repo_type', 'head', 'tree'}
                or baseline['repo_id'] != row['repo_id'] or baseline['repo_type'] != kind
                or baseline['head'] != _head(state)):
            raise PublicationError('Boundary repository/head differs from the verified original prefix.')
        tree = _shape(baseline['tree'], baseline['head'])
        known = {name for batch in state['batches'] for name in batch['paths']}
        if (set(tree) != set(state['baseline_tree'])|known
                or any(tree.get(name) != entry for name, entry in state['baseline_tree'].items())):
            raise PublicationError('Boundary tree changed outside the original verified prefix.')
        for name in known:
            _representation(row, name, original._read(source/kind/name), tree[name], state)


def capture_boundary(publication_plan, original_journal, stopped_writer_pid, output, *, api=None):
    """Read public trees only, after proving that the old writer has exited."""
    output = original._plain(output)
    if output.exists():
        raise PublicationError('Boundary output must be a fresh directory.')
    plan, journal, plan_raw, journal_raw = _original(publication_plan, original_journal, stopped_writer_pid)
    if api is None:
        api, _ = public.public_client()
    snapshots = {}
    for row in plan['repositories']:
        kind = row['repo_type']; head = _head(journal['repositories'][kind])
        info = api.repo_info(row['repo_id'], repo_type=kind)
        checked_remote(info, kind)
        if info.sha != head:
            raise PublicationError('Public head moved beyond the stopped original writer boundary.')
        tree = public._tree(api, kind, head)
        again = api.repo_info(row['repo_id'], repo_type=kind)
        checked_remote(again, kind)
        if again.sha != head:
            raise PublicationError('Public head changed while capturing the stopped boundary.')
        snapshots[kind] = dict(repo_id=row['repo_id'], repo_type=kind, head=head, tree=tree)
    document = dict(schema=SCHEMA+'/boundary', original_plan=original._receipt(plan_raw),
        original_journal=original._receipt(journal_raw), stopped_writer_pid=stopped_writer_pid,
        anonymous_public_trees=True, repositories=snapshots)
    _boundary_check(document, plan, journal, plan_raw, journal_raw, stopped_writer_pid, Path(publication_plan).parent)
    _original(publication_plan, original_journal, stopped_writer_pid)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'boundary.json').open('xb') as stream:
        stream.write(original._encoded(document))
    return document


def _groups(row, state):
    logical, completed = _prefix(row, state)
    items = {item['path']: item for item in row['files']}
    groups, pending, pending_bytes, pending_count = [], [], 0, 0
    for index, batch in enumerate(logical[completed:], completed):
        size = sum(items[name]['bytes'] for name in batch)
        if pending and (pending_bytes+size > MAX_BYTES or pending_count+len(batch) > MAX_FILES):
            groups.append(pending); pending, pending_bytes, pending_count = [], 0, 0
        pending.append(dict(index=index, paths=batch)); pending_bytes += size; pending_count += len(batch)
    if pending:
        groups.append(pending)
    return [dict(index=index, logical_batches=group, paths=[name for batch in group for name in batch['paths']],
                 bytes=sum(items[name]['bytes'] for batch in group for name in batch['paths']))
            for index, group in enumerate(groups)]


def stage(publication_plan, original_journal, boundary, stopped_writer_pid, output):
    plan_path, journal_path, boundary_path, output = map(original._plain,
        (publication_plan, original_journal, boundary, output))
    if (output.exists() or output == plan_path.parent or output in plan_path.parent.parents
            or plan_path.parent in output.parents or output == boundary_path or output in boundary_path.parents):
        raise PublicationError('Continuation staging must be fresh and separate from original artifacts and boundary.')
    plan, journal, plan_raw, journal_raw = _original(plan_path, journal_path, stopped_writer_pid)
    boundary_raw = original._read(boundary_path, original.MAX_PLAN)
    snapshot = original._json(boundary_raw)
    _boundary_check(snapshot, plan, journal, plan_raw, journal_raw, stopped_writer_pid, plan_path.parent)
    continuation = dict(schema=SCHEMA, original_plan=original._receipt(plan_raw),
        original_journal=original._receipt(journal_raw), boundary=original._receipt(boundary_raw),
        implementation=_implementation(),
        original_directory=os.path.relpath(plan_path.parent, output),
        original_plan_name=plan_path.name, stopped_writer_pid=stopped_writer_pid,
        physical_caps=dict(bytes=MAX_BYTES, files=MAX_FILES, individual_file_bytes=original.MAX_FILE),
        repositories=[dict(repo_id=row['repo_id'], repo_type=row['repo_type'],
            boundary_head=snapshot['repositories'][row['repo_type']]['head'],
            verified_original_logical_batches=len(journal['repositories'][row['repo_type']]['batches']),
            original_manifest=row['files'], groups=_groups(row, journal['repositories'][row['repo_type']]))
            for row in plan['repositories']], physical_robot_ready=False)
    if not any(row['groups'] for row in continuation['repositories']):
        raise PublicationError('No remaining original evidence needs continuation.')
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in [('original-plan.json', plan_raw), ('original-journal.json', journal_raw), ('boundary.json', boundary_raw),
                      (PLAN, original._encoded(continuation))]:
        with (output/name).open('xb') as stream:
            stream.write(raw)
    (output/'implementation').mkdir()
    for name in IMPLEMENTATION_FILES:
        with (output/'implementation'/name).open('xb') as stream:
            stream.write(original._read(IMPLEMENTATION_ROOT/name))
    checked_plan(output/PLAN)
    return continuation


def checked_plan(path):
    path = original._plain(path)
    plan = original._json(original._read(path, original.MAX_PLAN))
    if (not isinstance(plan, dict) or plan.get('schema') != SCHEMA or plan.get('physical_robot_ready') is not False
            or plan.get('physical_caps') != dict(bytes=MAX_BYTES, files=MAX_FILES, individual_file_bytes=original.MAX_FILE)):
        raise PublicationError('Expected the separate bounded continuation plan.')
    _check_implementation(plan.get('implementation'), path.parent/'implementation')
    directory, name = plan.get('original_directory'), plan.get('original_plan_name')
    if (not isinstance(directory, str) or not directory or Path(directory).is_absolute() or '\\' in directory
            or not isinstance(name, str) or Path(name).name != name or not name.endswith('.json')):
        raise PublicationError('Invalid immutable original publication reference.')
    source = original._plain(path.parent/directory)
    original_plan, journal, plan_raw, journal_raw = _original(source/name, source/original.JOURNAL_NAME, plan.get('stopped_writer_pid'))
    boundary_raw = original._read(path.parent/'boundary.json', original.MAX_PLAN)
    if (plan.get('original_plan') != original._receipt(plan_raw) or plan.get('original_journal') != original._receipt(journal_raw)
            or plan.get('boundary') != original._receipt(boundary_raw)
            or original._read(path.parent/'original-plan.json', original.MAX_PLAN) != plan_raw
            or original._read(path.parent/'original-journal.json', original.MAX_PLAN) != journal_raw):
        raise PublicationError('An immutable original plan/journal/boundary changed after review.')
    boundary = original._json(boundary_raw)
    _boundary_check(boundary, original_plan, journal, plan_raw, journal_raw, plan['stopped_writer_pid'], source)
    expected = [dict(repo_id=row['repo_id'], repo_type=row['repo_type'], boundary_head=boundary['repositories'][row['repo_type']]['head'],
        verified_original_logical_batches=len(journal['repositories'][row['repo_type']]['batches']),
        original_manifest=row['files'], groups=_groups(row, journal['repositories'][row['repo_type']]))
        for row in original_plan['repositories']]
    if plan.get('repositories') != expected:
        raise PublicationError('Continuation groups must exactly concatenate consecutive remaining original logical batches.')
    return plan, original_plan, journal, boundary, source


@contextmanager
def _writer_lock(directory):
    import fcntl
    lock = original._plain(directory/'continuation-writer.lock')
    descriptor = os.open(lock, os.O_CREAT|os.O_RDWR|getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise PublicationError('Another continuation writer holds the exclusive local lock.') from None
        os.ftruncate(descriptor, 0)
        os.write(descriptor, (str(os.getpid())+'\n').encode())
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)


def _message(plan_sha, kind, group):
    return f'Continue PiPER evidence {plan_sha} {kind} group {group["index"]}'


def _review_unchanged(plan, plan_raw, path, source):
    _check_implementation(plan['implementation'], path.parent/'implementation')
    _dead(plan['stopped_writer_pid'])
    if (original._receipt(original._read(source/plan['original_plan_name'], original.MAX_PLAN)) != plan['original_plan']
            or original._receipt(original._read(source/original.JOURNAL_NAME, original.MAX_PLAN)) != plan['original_journal']
            or original._read(path, original.MAX_PLAN) != plan_raw
            or (source/'publication-receipt.json').exists()):
        raise PublicationError('Original or continuation review documents changed during publication.')


def _load_progress(path, plan, plan_sha, boundary, source, old_journal):
    if path.exists():
        progress = original._json(original._read(path, original.MAX_PLAN))
    else:
        progress = dict(schema=SCHEMA+'/journal', plan_sha256=plan_sha, repositories={
            kind: dict(head=baseline['head'], groups=[]) for kind, baseline in boundary['repositories'].items()}, pending=None)
    if (not isinstance(progress, dict) or set(progress) != {'schema', 'plan_sha256', 'repositories', 'pending'}
            or progress['schema'] != SCHEMA+'/journal' or progress['plan_sha256'] != plan_sha
            or not isinstance(progress['repositories'], dict) or set(progress['repositories']) != set(original.REPOSITORIES)):
        raise PublicationError('Continuation journal does not bind this exact plan.')
    expected = {kind: dict(value['tree']) for kind, value in boundary['repositories'].items()}
    for row in plan['repositories']:
        kind = row['repo_type']; state = progress['repositories'][kind]
        if (not isinstance(state, dict) or set(state) != {'head', 'groups'} or not isinstance(state['groups'], list)
                or len(state['groups']) > len(row['groups'])):
            raise PublicationError('Malformed continuation progress.')
        head = row['boundary_head']
        for index, receipt in enumerate(state['groups']):
            group = row['groups'][index]
            if (not isinstance(receipt, dict) or set(receipt) != {'index', 'parent_commit', 'commit', 'entries', 'anonymous_readback', 'reconciled', 'commit_round_trip_s', 'verification_s'}
                    or receipt['index'] != index or receipt['parent_commit'] != head
                    or not isinstance(receipt['commit'], str) or original.COMMIT.fullmatch(receipt['commit']) is None
                    or receipt['commit'] == head or receipt['anonymous_readback'] is not True
                    or type(receipt['reconciled']) is not bool
                    or not (receipt['reconciled'] and receipt['commit_round_trip_s'] is None
                            or type(receipt['commit_round_trip_s']) in (float, int) and receipt['commit_round_trip_s'] >= 0)
                    or type(receipt['verification_s']) not in (float, int) or receipt['verification_s'] < 0
                    or not isinstance(receipt['entries'], dict) or set(receipt['entries']) != set(group['paths'])):
                raise PublicationError('Continuation journal group chain is invalid.')
            _shape(receipt['entries'], receipt['commit'])
            for name, entry in receipt['entries'].items():
                if name in expected[kind]:
                    raise PublicationError('Continuation attempted to replace an existing file.')
                _representation(row, name, original._read(source/kind/name), entry, old_journal['repositories'][kind])
            expected[kind].update(receipt['entries']); head = receipt['commit']
        if state['head'] != head:
            raise PublicationError('Continuation journal head differs from its group history.')
    pending = progress['pending']
    if pending is not None:
        if not isinstance(pending, dict) or set(pending) != {'repo_type', 'group_index', 'parent_commit', 'paths'}:
            raise PublicationError('Malformed durable pending continuation request.')
        rows = [row for row in plan['repositories'] if row['repo_type'] == pending['repo_type']]
        if len(rows) != 1:
            raise PublicationError('Pending request has an unknown repository.')
        row = rows[0]; state = progress['repositories'][row['repo_type']]; index = len(state['groups'])
        if (index >= len(row['groups']) or pending['group_index'] != index or pending['parent_commit'] != state['head']
                or pending['paths'] != row['groups'][index]['paths']):
            raise PublicationError('Pending request differs from the exact next reviewed physical group.')
    return progress, expected


def _remote(api, row):
    info = api.repo_info(row['repo_id'], repo_type=row['repo_type'])
    checked_remote(info, row['repo_type'])
    tree = public._tree(api, row['repo_type'], info.sha)
    _shape(tree, info.sha)
    return info.sha, tree


def _history(api, row, revision):
    """ID sets establish ancestry without assuming commit date order."""
    history = {}
    for commit in api.list_repo_commits(row['repo_id'], repo_type=row['repo_type'], revision=revision):
        identity, title = getattr(commit, 'commit_id', None), getattr(commit, 'title', None)
        if (not isinstance(identity, str) or original.COMMIT.fullmatch(identity) is None
                or not isinstance(title, str) or identity in history or len(history) >= public.MAX_TREE_FILES):
            raise PublicationError('Malformed or oversized pinned continuation commit history.')
        history[identity] = title
    if revision not in history:
        raise PublicationError('Pinned continuation commit is absent from its own history.')
    return history


def _group_bytes(row, group, source):
    items = {item['path']: item for item in row['original_manifest']}
    payload = {}
    for name in group['paths']:
        raw = original._read(source/row['repo_type']/name)
        if original._receipt(raw) != {key: items[name][key] for key in ('bytes', 'sha256')}:
            raise PublicationError('An original upload payload changed before its continuation commit.')
        payload[name] = raw
    if len(payload) > MAX_FILES or sum(map(len, payload.values())) > MAX_BYTES:
        raise PublicationError('Continuation physical commit exceeds its reviewed bounds.')
    return payload


def _finish_group(api, downloader, row, group, payload, revision, expected, progress, journal_path,
                  old_state, *, reconciled, duration):
    verification_started = time.monotonic()
    head, tree = _remote(api, row)
    if head != revision or set(tree) != set(expected)|set(payload) or any(tree.get(n) != e for n, e in expected.items()):
        raise PublicationError('Public continuation tree changed outside the exact pending group.')
    entries = {name: tree[name] for name in payload}
    for name, raw in payload.items():
        _representation(row, name, raw, entries[name], old_state)
    original.verify_public(dict(repo_id=row['repo_id'], repo_type=row['repo_type'], files=row['original_manifest']),
                           revision, downloader, set(payload))
    # Readback is not sufficient if another writer changed the head meanwhile.
    final_head, final_tree = _remote(api, row)
    if final_head != revision or final_tree != tree:
        raise PublicationError('Public head/tree moved during anonymous group readback.')
    state = progress['repositories'][row['repo_type']]
    state['groups'].append(dict(index=group['index'], parent_commit=state['head'], commit=revision,
        entries=entries, anonymous_readback=True, reconciled=reconciled,
        commit_round_trip_s=duration, verification_s=time.monotonic()-verification_started))
    state['head'] = revision; progress['pending'] = None
    original._atomic(journal_path, progress)
    expected.update(entries)


def _reconcile(api, downloader, plan, plan_sha, progress, expected, source, old_journal, journal_path):
    pending = progress['pending']
    for row in plan['repositories']:
        kind = row['repo_type']; state = progress['repositories'][kind]
        head, tree = _remote(api, row)
        if pending is None or pending['repo_type'] != kind:
            if head != state['head'] or tree != expected[kind]:
                raise PublicationError('Public repository moved outside the continuation journal.')
            continue
        group = row['groups'][pending['group_index']]
        if head == state['head']:
            if tree != expected[kind]:
                raise PublicationError('Original parent tree changed without the expected commit.')
            # The request has not landed at this observation. A subsequent
            # identical, parent-guarded retry cannot create a second child if
            # the earlier request later wins the atomic parent comparison.
            check_head, check_tree = _remote(api, row)
            if check_head != head or check_tree != tree:
                raise PublicationError('Pending outcome is still changing; refuse a retry.')
            progress['pending'] = None; original._atomic(journal_path, progress)
            continue
        # A lost reply is accepted only for the exact intended immediate child,
        # with its plan-bound message and no later/foreign commit above it.
        previous_history, history = _history(api, row, state['head']), _history(api, row, head)
        if (set(history) != set(previous_history)|{head}
                or history.get(head) != _message(plan_sha, kind, group)):
            raise PublicationError('Unknown outcome is not the exact intended immediate continuation commit.')
        payload = _group_bytes(row, group, source)
        _finish_group(api, downloader, row, group, payload, head, expected[kind], progress,
            journal_path, old_journal['repositories'][kind], reconciled=True, duration=None)


def publish(path, *, api=None, downloader=None, operation_cls=None):
    path = original._plain(path)
    receipt_path = path.parent/'publication-receipt.json'
    if receipt_path.exists():
        raise PublicationError('Continuation publication is complete; refusing to overwrite its receipt.')
    with _writer_lock(path.parent):
        if receipt_path.exists():
            raise PublicationError('Continuation publication is already complete.')
        plan, original_plan, old_journal, boundary, source = checked_plan(path)
        plan_raw = original._read(path, original.MAX_PLAN); plan_sha = sha256(plan_raw).hexdigest()
        journal_path = path.parent/JOURNAL
        progress, expected = _load_progress(journal_path, plan, plan_sha, boundary, source, old_journal)
        token = None
        if api is None:
            api, token = client(path.parent/'publisher-cache')
        if operation_cls is None:
            from huggingface_hub import CommitOperationAdd
            operation_cls = CommitOperationAdd
        if downloader is None:
            from huggingface_hub import hf_hub_download
            def downloader(repo_id, name, kind, revision):
                return hf_hub_download(repo_id, name, repo_type=kind, revision=revision,
                                       token=False, endpoint='https://huggingface.co')
        _reconcile(api, downloader, plan, plan_sha, progress, expected, source, old_journal, journal_path)
        if not journal_path.exists():
            original._atomic(journal_path, progress)
        for row in plan['repositories']:
            kind = row['repo_type']; state = progress['repositories'][kind]
            for group in row['groups'][len(state['groups']):]:
                _review_unchanged(plan, plan_raw, path, source)
                payload = _group_bytes(row, group, source)
                if token and any(token.encode() in raw for raw in payload.values()):
                    raise PublicationError('Credential appeared in an original reviewed payload.')
                head, tree = _remote(api, row)
                if head != state['head'] or tree != expected[kind]:
                    raise PublicationError('Public head/full tree moved before the guarded commit.')
                operations = []
                for name, raw in payload.items():
                    op = operation_cls(path_in_repo=name, path_or_fileobj=raw)
                    if kind == 'space':
                        op._upload_mode = 'lfs' if any(name.endswith(s) for s in old_journal['repositories'][kind]['lfs_suffixes']) else 'regular'
                        op._should_ignore = False
                    operations.append(op)
                progress['pending'] = dict(repo_type=kind, group_index=group['index'], parent_commit=head, paths=group['paths'])
                original._atomic(journal_path, progress)
                started = time.monotonic()
                _check_implementation(plan['implementation'], path.parent/'implementation')
                try:
                    committed = api.create_commit(row['repo_id'], repo_type=kind, parent_commit=head,
                        operations=operations, commit_message=_message(plan_sha, kind, group))
                except Exception:
                    raise PublicationError('Continuation commit outcome is uncertain; preserve its pending request and reconcile the same plan.') from None
                revision = getattr(committed, 'oid', None)
                if not isinstance(revision, str) or original.COMMIT.fullmatch(revision) is None or revision == head:
                    raise PublicationError('Commit did not return a new exact revision; reconcile the durable pending request.')
                _finish_group(api, downloader, row, group, payload, revision, expected[kind], progress,
                    journal_path, old_journal['repositories'][kind], reconciled=False, duration=time.monotonic()-started)
        results = []
        for row in plan['repositories']:
            kind = row['repo_type']; head, tree = _remote(api, row)
            if head != progress['repositories'][kind]['head'] or tree != expected[kind]:
                raise PublicationError('Final public repository differs from its complete verified history.')
            original_names = {item['path'] for item in row['original_manifest']}
            if set(tree) != set(old_journal['repositories'][kind]['baseline_tree'])|original_names:
                raise PublicationError('Final public inventory is incomplete or contains unexpected files.')
            results.append(dict(repo_id=row['repo_id'], repo_type=kind, commit=head, verified_files=len(original_names),
                url='https://huggingface.co/'+{'model': '', 'dataset': 'datasets/', 'space': 'spaces/'}[kind]+row['repo_id']))
        _review_unchanged(plan, plan_raw, path, source)
        receipt = dict(schema=original.SCHEMA+'/receipt', plan_sha256=plan['original_plan']['sha256'],
            source_commit=original_plan['source_commit'], source_integrity_sha256=original_plan['source_integrity_sha256'],
            bindings=original_plan['bindings'], repositories=results,
            space_baseline_commit=old_journal['repositories']['space']['baseline_sha'],
            space_baseline_files_preserved=len(old_journal['repositories']['space']['baseline_tree']),
            anonymous_readback=True, viewer_url=original.VIEWER_URL, physical_robot_ready=False,
            continuation=dict(schema=SCHEMA, plan=original._receipt(plan_raw),
                implementation=plan['implementation'],
                immutable_original_journal=plan['original_journal'], boundary=plan['boundary'],
                journal=original._receipt(original._read(journal_path, original.MAX_PLAN)),
                history={kind: dict(original_prefix_commits=[b['commit'] for b in old_journal['repositories'][kind]['batches']],
                    continuation_commits=[g['commit'] for g in progress['repositories'][kind]['groups']])
                    for kind in original.REPOSITORIES},
                verification='Original publicly verified prefix revalidated by unchanged exact Git/LFS object IDs; every continuation file anonymously read back at its exact committed revision.'))
        original._atomic(receipt_path, receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('capture-boundary', 'stage'):
        prep = sub.add_parser(command)
        prep.add_argument('--publication-plan', required=True, type=Path)
        prep.add_argument('--original-journal', required=True, type=Path)
        prep.add_argument('--stopped-writer-pid', required=True, type=int)
        prep.add_argument('--output', required=True, type=Path)
        if command == 'stage':
            prep.add_argument('--boundary', required=True, type=Path)
    for command in ('check', 'publish'):
        sub.add_parser(command).add_argument('plan', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'capture-boundary':
            result = capture_boundary(args.publication_plan, args.original_journal, args.stopped_writer_pid, args.output)
        elif args.command == 'stage':
            result = stage(args.publication_plan, args.original_journal, args.boundary, args.stopped_writer_pid, args.output)
        elif args.command == 'check':
            result = checked_plan(args.plan)[0]
        else:
            result = publish(args.plan)
        print(json.dumps(dict(schema=result['schema'], command=args.command, completed=True)))
        return 0
    except Exception as exc:
        print('Continuation stopped ('+type(exc).__name__+'); preserve the exact plan and pending journal for reconciliation.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
