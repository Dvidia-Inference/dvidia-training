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
from hashlib import sha1, sha256
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
ATTRIBUTES = '.gitattributes'
WHEEL = 'contracts/dvidia_training-0.1.0-py3-none-any.whl'
WHEEL_RULE = (WHEEL+' filter=lfs diff=lfs merge=lfs -text\n').encode()
MODEL_WHEEL_COMMIT = '4aa5000d76a3eb2647368816ffe494d696839e1a'


def _attribute_entry(raw):
    return dict(blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
                size=len(raw), lfs_sha256=None)


def _attribute_readback(commit, raw):
    return dict(commit=commit, entry=_attribute_entry(raw), receipt=original._receipt(raw), utf8=raw.decode('utf8'))


def _attribute_bytes(value):
    if (not isinstance(value, dict) or set(value) != {'commit', 'entry', 'receipt', 'utf8'}
            or not isinstance(value['commit'], str) or original.COMMIT.fullmatch(value['commit']) is None
            or not isinstance(value['utf8'], str)):
        raise PublicationError('Malformed pinned attributes readback.')
    raw = value['utf8'].encode('utf8')
    if len(raw) > 16384 or value['entry'] != _attribute_entry(raw) or value['receipt'] != original._receipt(raw):
        raise PublicationError('Attributes raw bytes differ from their exact Git/SHA receipts.')
    return raw


def _wheel_receipt(plan, kind):
    row = next(row for row in plan['repositories'] if row['repo_type'] == kind)
    items = [item for item in row['files'] if item['path'] == WHEEL]
    if len(items) != 1 or {k: items[0][k] for k in ('bytes', 'sha256')} != original.CONTRACT_ARCHIVES[WHEEL]:
        raise PublicationError('Attributes exception must bind the exact originally reviewed installation wheel.')
    return items[0]


def _metadata_evidence(value, plan, journal, plan_raw, journal_raw):
    """Only one observed model delta and one prospective identical dataset rule."""
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'schema', 'original_plan', 'original_journal', 'model', 'dataset_before'}
            or value['schema'] != SCHEMA+'/wheel-attributes-evidence'
            or value['original_plan'] != original._receipt(plan_raw)
            or value['original_journal'] != original._receipt(journal_raw)):
        raise PublicationError('Wheel metadata evidence belongs to a different immutable publication prefix.')
    model, state = value['model'], journal['repositories']['model']
    if (not isinstance(model, dict) or set(model) != {'repo_id', 'path', 'wheel', 'added_line_utf8', 'before', 'after', 'provenance', 'anonymous_readbacks'}
            or model['repo_id'] != original.MODEL_REPO or model['path'] != ATTRIBUTES
            or model['wheel'] != _wheel_receipt(plan, 'model')
            or model['added_line_utf8'] != WHEEL_RULE.decode() or model['anonymous_readbacks'] is not True):
        raise PublicationError('Only the explicit reviewed model wheel attributes acknowledgment is supported.')
    before, after = _attribute_bytes(model['before']), _attribute_bytes(model['after'])
    if (model['before']['commit'] != state['created_sha'] or model['after']['commit'] != _head(state)
            or model['before']['entry'] != state['baseline_tree'].get(ATTRIBUTES)
            or not before.endswith(b'\n') or WHEEL_RULE in before or after != before+WHEEL_RULE or len(WHEEL_RULE) != 85):
        raise PublicationError('Model attributes acknowledgment must be exactly the single appended 85-byte wheel rule.')
    if (not isinstance(model['provenance'], list) or len(model['provenance']) != 4 or len(state['batches']) != 4
            or state['batches'][1]['commit'] != MODEL_WHEEL_COMMIT or WHEEL not in state['batches'][1]['paths']
            or any(WHEEL in batch['paths'] for index, batch in enumerate(state['batches']) if index != 1)):
        raise PublicationError('Model wheel rule is not bound to its four known original commits and wheel batch.')
    for index, (proof, batch) in enumerate(zip(model['provenance'], state['batches'])):
        if proof != dict(index=index, commit=batch['commit'], parent_commit=batch['parent_commit'],
                         attributes_entry=model['before' if index == 0 else 'after']['entry']):
            raise PublicationError('Historical wheel attributes provenance differs from the original journal.')
    dataset = value['dataset_before']; dataset_state = journal['repositories']['dataset']
    dataset_raw = _attribute_bytes(dataset)
    if (dataset['commit'] != _head(dataset_state) or dataset['entry'] != dataset_state['baseline_tree'].get(ATTRIBUTES)
            or not dataset_raw.endswith(b'\n') or WHEEL_RULE in dataset_raw
            or any(WHEEL in batch['paths'] for batch in dataset_state['batches'])):
        raise PublicationError('Dataset attributes must remain at their exact unchanged pre-wheel baseline.')
    _wheel_receipt(plan, 'dataset')
    return value


def capture_metadata_evidence(publication_plan, original_journal, stopped_writer_pid, output, *, api=None, downloader=None):
    """Only three pinned public text reads and four pinned metadata observations."""
    plan_path, journal_path, output = map(original._plain, (publication_plan, original_journal, output))
    if output.exists() or journal_path.parent != plan_path.parent or journal_path.name != original.JOURNAL_NAME:
        raise PublicationError('Metadata evidence requires fresh output and the exact original sibling journal.')
    _dead(stopped_writer_pid)
    plan_raw, journal_raw = original._read(plan_path, original.MAX_PLAN), original._read(journal_path, original.MAX_PLAN)
    plan = original._json(plan_raw)
    journal = original._load_journal(journal_path, plan, sha256(plan_raw).hexdigest())
    for row in plan['repositories']:
        _prefix(row, journal['repositories'][row['repo_type']])
    if api is None and downloader is None:
        api, downloader = public.public_client()
    if api is None or downloader is None:
        raise PublicationError('Provide both anonymous metadata-read boundaries.')
    state = journal['repositories']['model']
    def read(repo, kind, commit):
        raw = original._downloaded_bytes(downloader(repo, ATTRIBUTES, kind, commit), 16384)
        return _attribute_readback(commit, raw)
    before = read(original.MODEL_REPO, 'model', state['created_sha'])
    after = read(original.MODEL_REPO, 'model', _head(state))
    provenance = []
    for index, batch in enumerate(state['batches']):
        entries = api.get_paths_info(original.MODEL_REPO, [ATTRIBUTES], repo_type='model', revision=batch['commit'])
        if len(entries) != 1 or entries[0].path != ATTRIBUTES or getattr(entries[0], 'lfs', None):
            raise PublicationError('Pinned model attributes provenance must be one regular Git text object.')
        entry = entries[0]
        provenance.append(dict(index=index, commit=batch['commit'], parent_commit=batch['parent_commit'],
            attributes_entry=dict(blob=entry.blob_id, size=entry.size, lfs_sha256=None)))
    dataset_before = read(original.DATASET_REPO, 'dataset', _head(journal['repositories']['dataset']))
    value = dict(schema=SCHEMA+'/wheel-attributes-evidence', original_plan=original._receipt(plan_raw),
        original_journal=original._receipt(journal_raw),
        model=dict(repo_id=original.MODEL_REPO, path=ATTRIBUTES, wheel=_wheel_receipt(plan, 'model'),
            added_line_utf8=WHEEL_RULE.decode(), before=before, after=after,
            provenance=provenance, anonymous_readbacks=True), dataset_before=dataset_before)
    _metadata_evidence(value, plan, journal, plan_raw, journal_raw)
    if original._read(plan_path, original.MAX_PLAN) != plan_raw or original._read(journal_path, original.MAX_PLAN) != journal_raw:
        raise PublicationError('Original publication documents changed during read-only metadata inspection.')
    _dead(stopped_writer_pid)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'metadata-evidence.json').open('xb') as stream:
        stream.write(original._encoded(value))
    return value


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
            'stopped_writer_pid', 'anonymous_public_trees', 'repositories', 'metadata_evidence'}
            or document['schema'] != SCHEMA+'/boundary'
            or document['original_plan'] != original._receipt(plan_raw)
            or document['original_journal'] != original._receipt(journal_raw)
            or document['stopped_writer_pid'] != pid or document['anonymous_public_trees'] is not True
            or not isinstance(document['repositories'], dict)
            or set(document['repositories']) != set(original.REPOSITORIES)):
        raise PublicationError('Boundary must bind the exact exited writer, original plan/journal and all public trees.')
    evidence = _metadata_evidence(document['metadata_evidence'], plan, journal, plan_raw, journal_raw)
    for row in plan['repositories']:
        kind = row['repo_type']; state = journal['repositories'][kind]
        baseline = document['repositories'][kind]
        if (not isinstance(baseline, dict) or set(baseline) != {'repo_id', 'repo_type', 'head', 'tree'}
                or baseline['repo_id'] != row['repo_id'] or baseline['repo_type'] != kind
                or baseline['head'] != _head(state)):
            raise PublicationError('Boundary repository/head differs from the verified original prefix.')
        tree = _shape(baseline['tree'], baseline['head'])
        known = {name for batch in state['batches'] for name in batch['paths']}
        baseline = dict(state['baseline_tree'])
        if kind == 'model' and evidence is not None:
            baseline[ATTRIBUTES] = evidence['model']['after']['entry']
        if (set(tree) != set(baseline)|known
                or any(tree.get(name) != entry for name, entry in baseline.items())):
            raise PublicationError('Boundary tree changed outside the original verified prefix.')
        for name in known:
            _representation(row, name, original._read(source/kind/name), tree[name], state)


def capture_boundary(publication_plan, original_journal, stopped_writer_pid, output, *, metadata_evidence=None, api=None):
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
    evidence = original._json(original._read(metadata_evidence)) if metadata_evidence is not None else None
    document = dict(schema=SCHEMA+'/boundary', original_plan=original._receipt(plan_raw),
        original_journal=original._receipt(journal_raw), stopped_writer_pid=stopped_writer_pid,
        anonymous_public_trees=True, repositories=snapshots, metadata_evidence=evidence)
    _boundary_check(document, plan, journal, plan_raw, journal_raw, stopped_writer_pid, Path(publication_plan).parent)
    _original(publication_plan, original_journal, stopped_writer_pid)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'boundary.json').open('xb') as stream:
        stream.write(original._encoded(document))
    return document


def _groups(row, state, metadata_bytes=0):
    logical, completed = _prefix(row, state)
    items = {item['path']: item for item in row['files']}
    groups, pending, pending_bytes, pending_count = [], [], 0, 0
    for index, batch in enumerate(logical[completed:], completed):
        reserved = metadata_bytes if WHEEL in batch else 0
        size = sum(items[name]['bytes'] for name in batch)+reserved
        count = len(batch)+int(reserved > 0)
        if pending and (pending_bytes+size > MAX_BYTES or pending_count+count > MAX_FILES):
            groups.append(pending); pending, pending_bytes, pending_count = [], 0, 0
        pending.append(dict(index=index, paths=batch)); pending_bytes += size; pending_count += count
    if pending:
        groups.append(pending)
    return [dict(index=index, logical_batches=group, paths=[name for batch in group for name in batch['paths']],
                 bytes=sum(items[name]['bytes'] for batch in group for name in batch['paths'])+
                       (metadata_bytes if any(WHEEL in batch['paths'] for batch in group) else 0),
                 metadata_reserved_bytes=metadata_bytes if any(WHEEL in batch['paths'] for batch in group) else 0)
            for index, group in enumerate(groups)]


def _planned_repositories(plan, journal, boundary):
    evidence = boundary['metadata_evidence']
    return [dict(repo_id=row['repo_id'], repo_type=row['repo_type'],
        boundary_head=boundary['repositories'][row['repo_type']]['head'],
        verified_original_logical_batches=len(journal['repositories'][row['repo_type']]['batches']),
        original_manifest=row['files'], groups=_groups(row, journal['repositories'][row['repo_type']],
            metadata_bytes=len(_attribute_bytes(evidence['dataset_before'])+WHEEL_RULE)
                if evidence is not None and row['repo_type'] == 'dataset' else 0))
        for row in plan['repositories']]


def _dataset_transition(repositories, evidence):
    if evidence is None:
        return None
    dataset = next(row for row in repositories if row['repo_type'] == 'dataset')
    groups = [group for group in dataset['groups'] if WHEEL in group['paths']]
    if len(groups) != 1:
        raise PublicationError('Dataset wheel metadata permission must apply to exactly one continuation group.')
    before = evidence['dataset_before']; after_raw = _attribute_bytes(before)+WHEEL_RULE
    return dict(repo_id=original.DATASET_REPO, repo_type='dataset', path=ATTRIBUTES,
        group_index=groups[0]['index'], wheel=evidence['model']['wheel'], added_line_utf8=WHEEL_RULE.decode(),
        before=before, after=dict(entry=_attribute_entry(after_raw), receipt=original._receipt(after_raw), utf8=after_raw.decode()))


def _transition_for(plan, row, group):
    transition = plan['dataset_metadata_transition']
    return transition if transition is not None and row['repo_type'] == 'dataset' and transition['group_index'] == group['index'] else None


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
        metadata_evidence=snapshot['metadata_evidence'],
        repositories=_planned_repositories(plan, journal, snapshot), physical_robot_ready=False)
    continuation['dataset_metadata_transition'] = _dataset_transition(continuation['repositories'], snapshot['metadata_evidence'])
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
    expected = _planned_repositories(original_plan, journal, boundary)
    if (plan.get('repositories') != expected or plan.get('metadata_evidence') != boundary['metadata_evidence']
            or plan.get('dataset_metadata_transition') != _dataset_transition(expected, boundary['metadata_evidence'])):
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
            or original._receipt(original._read(path.parent/'original-plan.json', original.MAX_PLAN)) != plan['original_plan']
            or original._receipt(original._read(path.parent/'original-journal.json', original.MAX_PLAN)) != plan['original_journal']
            or original._receipt(original._read(path.parent/'boundary.json', original.MAX_PLAN)) != plan['boundary']
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
            transition = _transition_for(plan, row, group)
            paths = set(group['paths'])|({ATTRIBUTES} if transition is not None else set())
            if (not isinstance(receipt, dict) or set(receipt) != {'index', 'parent_commit', 'commit', 'entries', 'anonymous_readback', 'reconciled', 'commit_round_trip_s', 'verification_s'}
                    or receipt['index'] != index or receipt['parent_commit'] != head
                    or not isinstance(receipt['commit'], str) or original.COMMIT.fullmatch(receipt['commit']) is None
                    or receipt['commit'] == head or receipt['anonymous_readback'] is not True
                    or type(receipt['reconciled']) is not bool
                    or not (receipt['reconciled'] and receipt['commit_round_trip_s'] is None
                            or type(receipt['commit_round_trip_s']) in (float, int) and receipt['commit_round_trip_s'] >= 0)
                    or type(receipt['verification_s']) not in (float, int) or receipt['verification_s'] < 0
                    or not isinstance(receipt['entries'], dict) or set(receipt['entries']) != paths):
                raise PublicationError('Continuation journal group chain is invalid.')
            _shape(receipt['entries'], receipt['commit'])
            for name, entry in receipt['entries'].items():
                if name == ATTRIBUTES and transition is not None:
                    if (expected[kind].get(name) != transition['before']['entry'] or entry != transition['after']['entry']):
                        raise PublicationError('Journal attributes delta differs from the exact declared wheel transition.')
                    continue
                if name in expected[kind]:
                    raise PublicationError('Continuation attempted to replace an existing file.')
                _representation(row, name, original._read(source/kind/name), entry, old_journal['repositories'][kind])
            expected[kind].update(receipt['entries']); head = receipt['commit']
        if state['head'] != head:
            raise PublicationError('Continuation journal head differs from its group history.')
    pending = progress['pending']
    if pending is not None:
        if not isinstance(pending, dict) or set(pending) != {'repo_type', 'group_index', 'parent_commit', 'paths', 'metadata_transition'}:
            raise PublicationError('Malformed durable pending continuation request.')
        rows = [row for row in plan['repositories'] if row['repo_type'] == pending['repo_type']]
        if len(rows) != 1:
            raise PublicationError('Pending request has an unknown repository.')
        row = rows[0]; state = progress['repositories'][row['repo_type']]; index = len(state['groups'])
        if (index >= len(row['groups']) or pending['group_index'] != index or pending['parent_commit'] != state['head']
                or pending['paths'] != row['groups'][index]['paths']
                or pending['metadata_transition'] != _transition_for(plan, row, row['groups'][index])):
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
    reserved = group['metadata_reserved_bytes']
    if len(payload)+int(reserved > 0) > MAX_FILES or sum(map(len, payload.values()))+reserved > MAX_BYTES:
        raise PublicationError('Continuation physical commit exceeds its reviewed bounds.')
    return payload


def _finish_group(api, downloader, row, group, payload, revision, expected, progress, journal_path,
                  old_state, *, reconciled, duration, transition=None):
    verification_started = time.monotonic()
    head, tree = _remote(api, row)
    before = dict(expected)
    if transition is not None:
        if before.get(ATTRIBUTES) != transition['before']['entry'] or group['metadata_reserved_bytes'] != transition['after']['receipt']['bytes']:
            raise PublicationError('Wheel attributes transition has the wrong parent or reserved byte budget.')
        before[ATTRIBUTES] = transition['after']['entry']
    if head != revision or set(tree) != set(expected)|set(payload) or any(tree.get(n) != e for n, e in before.items()):
        raise PublicationError('Public continuation tree changed outside the exact pending group.')
    entries = {name: tree[name] for name in payload}
    for name, raw in payload.items():
        _representation(row, name, raw, entries[name], old_state)
    original.verify_public(dict(repo_id=row['repo_id'], repo_type=row['repo_type'], files=row['original_manifest']),
                           revision, downloader, set(payload))
    if transition is not None:
        raw = original._downloaded_bytes(downloader(row['repo_id'], ATTRIBUTES, row['repo_type'], revision), 16384)
        if raw != transition['after']['utf8'].encode() or not _matches(raw, tree[ATTRIBUTES]):
            raise PublicationError('Anonymous wheel attributes readback differs from the exact declared transition.')
        entries[ATTRIBUTES] = tree[ATTRIBUTES]
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
            journal_path, old_journal['repositories'][kind], reconciled=True, duration=None,
            transition=_transition_for(plan, row, group))


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
                transition = _transition_for(plan, row, group)
                progress['pending'] = dict(repo_type=kind, group_index=group['index'], parent_commit=head,
                    paths=group['paths'], metadata_transition=transition)
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
                    journal_path, old_journal['repositories'][kind], reconciled=False,
                    duration=time.monotonic()-started, transition=transition)
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
                model_metadata_acknowledgment=plan['metadata_evidence']['model'] if plan['metadata_evidence'] else None,
                dataset_metadata_transition=plan['dataset_metadata_transition'],
                immutable_original_journal=plan['original_journal'], boundary=plan['boundary'],
                journal=original._receipt(original._read(journal_path, original.MAX_PLAN)),
                history={kind: dict(original_prefix_commits=[b['commit'] for b in old_journal['repositories'][kind]['batches']],
                    continuation_commits=[g['commit'] for g in progress['repositories'][kind]['groups']])
                    for kind in original.REPOSITORIES},
                verification='Original publicly verified scientific payload prefix revalidated by unchanged exact Git/LFS object IDs; the single historical model LFS-rule metadata delta is explicitly acknowledged when present. Every continuation payload and the sole declared dataset LFS-rule delta, when present, were anonymously read back at their exact committed revision.'))
        original._atomic(receipt_path, receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('capture-boundary', 'stage', 'capture-metadata-evidence'):
        prep = sub.add_parser(command)
        prep.add_argument('--publication-plan', required=True, type=Path)
        prep.add_argument('--original-journal', required=True, type=Path)
        prep.add_argument('--stopped-writer-pid', required=True, type=int)
        prep.add_argument('--output', required=True, type=Path)
        if command == 'stage':
            prep.add_argument('--boundary', required=True, type=Path)
        if command == 'capture-boundary':
            prep.add_argument('--metadata-evidence', type=Path)
    for command in ('check', 'publish'):
        sub.add_parser(command).add_argument('plan', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'capture-boundary':
            result = capture_boundary(args.publication_plan, args.original_journal, args.stopped_writer_pid, args.output,
                metadata_evidence=args.metadata_evidence)
        elif args.command == 'capture-metadata-evidence':
            result = capture_metadata_evidence(args.publication_plan, args.original_journal, args.stopped_writer_pid, args.output)
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
