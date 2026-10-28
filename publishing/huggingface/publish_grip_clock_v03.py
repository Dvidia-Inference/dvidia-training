"""Stage/check offline, then explicitly publish the reviewed v0.3 artifacts.

Only publish reads the named macOS Keychain credential, through the existing
memory-only client. Fresh model/dataset repositories and one append-only static
Space namespace are fixed. Unknown commit replies retain a durable intent and
are reconciled against the exact immediate child, never blindly replayed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha1, sha256
import gzip
import io
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile

from publish import PublicationError, checked_remote, client


SCHEMA = 'dvidia.grip-clock-publication/v0.3'
BASELINE_SCHEMA = SCHEMA+'/space-baseline'
REPOS = {'model': 'Dvidia/piper-grip-clock-v03',
         'dataset': 'Dvidia/piper-grip-clock-demonstrations-v03',
         'space': 'Dvidia/robot-simulation-lab'}
NAMESPACE = 'learned-placement/v03/'
PLAN = 'publication-plan.json'
JOURNAL = 'publication-journal.json'
RECEIPT = 'publication-receipt.json'
MAX_FILE = 8*1024*1024
MAX_TOTAL = 8*1024**3
MAX_MODEL, MAX_SPACE = 64*1024**2, 256*1024**2
MAX_FILES, MAX_PLAN, MAX_TREE = 20000, 16*1024**2, 30000
MAX_DECODED = 128*1024**2
BATCH_BYTES, BATCH_FILES = 64*1024**2, 100
LFS = {'.bin', '.gz', '.npz'}
ATTRIBUTES = ''.join(f'*{suffix} filter=lfs diff=lfs merge=lfs -text\n' for suffix in sorted(LFS)).encode()
SUFFIXES = {'.json', '.gz', '.npz', '.py', '.md', '.txt', '.html', '.js', '.css',
            '.bin', '.png', '.svg', '.webp', '.toml'}
TOKEN = re.compile(rb'(?:hf_[A-Za-z0-9]{30,}|apikey_[a-f0-9]{20,}_[a-f0-9]{20,}|sk-[A-Za-z0-9_-]{30,}|Bearer\s+[A-Za-z0-9_-]{20,})')
PRIVATE = re.compile(rb'(?:/Users/[^/\s]+/|/private/tmp/|file://)')
SHA = re.compile(r'[a-f0-9]{64}\Z')
OID = re.compile(r'[a-f0-9]{40}\Z')
PRIVATE_KEYS = {'authorization', 'api_key', 'apikey', 'credential', 'credentials', 'token', 'raw_response'}
IMPLEMENTATION = ('publish_grip_clock_v03.py', 'publish.py', 'prepare.py')


def plain(path):
    path = Path(path).absolute()
    if any(item.is_symlink() and item not in {Path('/var'), Path('/tmp')} for item in (path, *path.parents)):
        raise PublicationError('Publication paths cannot traverse artifact-owned symlinks.')
    return path.resolve()


def relative(name, *, attributes=False):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\0' in name:
        raise PublicationError('Unsafe public path.')
    path = PurePosixPath(name)
    if attributes and name == '.gitattributes':
        return path
    if (path.is_absolute() or path.as_posix() != name or any(part.startswith('.') for part in path.parts)
            or '__pycache__' in path.parts or path.suffix not in SUFFIXES and path.name != 'LICENSE'):
        raise PublicationError('Unsafe or unexpected public artifact type.')
    return path


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def decoded(raw):
    try:
        return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError):
        raise PublicationError('Expected finite UTF-8 JSON.') from None


def receipt(raw):
    return dict(bytes=len(raw), sha256=sha256(raw).hexdigest())


def read(path, limit=MAX_FILE):
    path = plain(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise PublicationError('Missing or oversized publication file.')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise PublicationError('Publication file exceeded its bounded read.')
    return raw


def atomic(path, value, *, exclusive=False):
    path = plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.publication-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def scan(raw, name):
    """Bounded plaintext and decoded native-evidence credential/scope checks."""
    private_raw = re.sub(rb"(['\"])file://\1", b'bare-protocol-literal', raw) if name.endswith(('.py', '.js', '.html')) else raw
    if TOKEN.search(raw) or PRIVATE.search(private_raw):
        raise PublicationError('Possible credential or private path in staged artifacts.')
    if name.endswith('.json'):
        pending = [decoded(raw)]
        while pending:
            item = pending.pop()
            if type(item) is dict:
                if any(str(key).lower() in PRIVATE_KEYS for key in item):
                    raise PublicationError('Credential fields cannot be published.')
                for key in ('physical_robot_ready', 'hardware_qualified', 'sim2real_validated'):
                    if key in item and item[key] is not False:
                        raise PublicationError('Publication must retain its simulation-only qualification scope.')
                pending.extend(item.values())
            elif type(item) is list:
                pending.extend(item)
    if name.endswith('.gz'):
        if not name.endswith('.json.gz'):
            raise PublicationError('Only bounded compressed JSON episodes are supported.')
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                body = stream.read(MAX_DECODED+1)
        except (OSError, EOFError):
            raise PublicationError('Invalid compressed JSON evidence.') from None
        if len(body) > MAX_DECODED:
            raise PublicationError('Compressed JSON exceeds its decoded size bound.')
        scan(body, name[:-3])
    if name.endswith('.npz'):
        import numpy as np
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                rows = archive.infolist()
                if (not 1 <= len(rows) <= 128 or len({row.filename for row in rows}) != len(rows)
                        or sum(row.file_size for row in rows) > 2*MAX_DECODED):
                    raise PublicationError('Native archive has duplicate/excessive members or expansion.')
                for row in rows:
                    path = PurePosixPath(row.filename)
                    if (len(path.parts) != 1 or path.suffix != '.npy' or path.name.startswith('.')
                            or row.file_size > MAX_DECODED or row.flag_bits & 1
                            or row.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                        raise PublicationError('Unsafe native archive member.')
                    body = archive.read(row)
                    if TOKEN.search(body) or PRIVATE.search(body):
                        raise PublicationError('Possible credential or private path in a native array.')
                    value = np.load(io.BytesIO(body), allow_pickle=False)
                    if value.dtype.kind in 'fiu':
                        if not np.isfinite(value).all():
                            raise PublicationError('Native arrays must contain finite numeric values.')
                    elif value.dtype.kind == 'U' and path.stem == 'groups' and PurePosixPath(name).name == 'samples.npz':
                        if value.ndim != 1 or any(not item or len(item) > 100 for item in value.tolist()):
                            raise PublicationError('Invalid whole-episode sample groups.')
                        for group in value.tolist():
                            if TOKEN.search(group.encode()) or PRIVATE.search(group.encode()):
                                raise PublicationError('Possible credential or private path in a sample group.')
                    else:
                        raise PublicationError('Only finite numeric arrays and sample group strings are supported.')
        except PublicationError:
            raise
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            raise PublicationError('Invalid or object-bearing native archive.') from None


def _inventory(directory, kind):
    directory = plain(directory)
    if not directory.is_dir():
        raise PublicationError('Expected a staged payload directory.')
    result, total = [], 0
    for path in sorted(directory.rglob('*')):
        plain(path)
        if not path.is_file():
            continue
        name = path.relative_to(directory).as_posix()
        relative(name, attributes=kind != 'space')
        if kind == 'space' and not name.startswith(NAMESPACE):
            raise PublicationError('Space publication may only append learned-placement/v03 paths.')
        raw = read(path)
        if name == '.gitattributes':
            if raw != ATTRIBUTES:
                raise PublicationError('New repository attributes must be the exact declared native LFS rules.')
        else:
            scan(raw, name)
        total += len(raw)
        result.append(dict(path=name, **receipt(raw), representation='lfs' if path.suffix in LFS else 'git'))
    if not 1 <= len(result) <= MAX_FILES or total > {'model': MAX_MODEL, 'dataset': MAX_TOTAL, 'space': MAX_SPACE}[kind]:
        raise PublicationError('Staged inventory exceeds its declared file or byte bound.')
    names = {item['path'] for item in result}
    if kind == 'space' and NAMESPACE+'viewer/index.html' not in names:
        raise PublicationError('Space payload must contain its declared v0.3 viewer entry point.')
    if kind != 'space':
        if not {'.gitattributes', 'README.md', 'LICENSE'} <= names:
            raise PublicationError('Model/dataset require the generated attributes, MIT license and scoped Hub card.')
        card, license_raw = read(directory/'README.md'), read(directory/'LICENSE')
        if (not re.search(rb'^license:\s*mit\s*$', card, re.M)
                or not re.search(rb'simulation[ -]only', card, re.I)
                or b'MIT License' not in license_raw or b'Permission is hereby granted' not in license_raw):
            raise PublicationError('Public cards and licenses must explicitly retain MIT and simulation-only scope.')
    return result


def _entry(raw, *, lfs=False):
    return dict(blob=sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
                size=len(raw), lfs_sha256=sha256(raw).hexdigest() if lfs else None)


def _matches(raw, entry):
    return (len(raw) == entry.get('size') and (sha256(raw).hexdigest() == entry.get('lfs_sha256')
        if entry.get('lfs_sha256') else _entry(raw)['blob'] == entry.get('blob')))


def _tree_valid(tree):
    if type(tree) is not dict or len(tree) > MAX_TREE:
        raise PublicationError('Invalid or excessive pinned public tree.')
    for name, item in tree.items():
        path = PurePosixPath(name)
        if (not isinstance(name, str) or not name or path.is_absolute() or path.as_posix() != name
                or '\\' in name or ':' in name or '..' in path.parts or '__pycache__' in path.parts
                or type(item) is not dict or set(item) != {'blob', 'size', 'lfs_sha256'}
                or not isinstance(item['blob'], str) or OID.fullmatch(item['blob']) is None
                or type(item['size']) is not int or item['size'] < 0
                or item['lfs_sha256'] is not None and (not isinstance(item['lfs_sha256'], str) or SHA.fullmatch(item['lfs_sha256']) is None)):
            raise PublicationError('Malformed pinned Git/LFS object.')


def _baseline(value):
    if (type(value) is not dict or value.get('schema') != BASELINE_SCHEMA
            or value.get('repo_id') != REPOS['space'] or value.get('repo_type') != 'space'
            or value.get('anonymous_readback') is not True
            or not isinstance(value.get('commit'), str) or OID.fullmatch(value['commit']) is None
            or type(value.get('attributes_utf8')) is not str):
        raise PublicationError('Expected the exact anonymous pinned Space baseline.')
    _tree_valid(value.get('tree'))
    raw = value['attributes_utf8'].encode()
    if len(raw) > MAX_FILE or not _matches(raw, value['tree'].get('.gitattributes', {})):
        raise PublicationError('Pinned Space attributes differ from their Git object.')
    if any(name.startswith(NAMESPACE) for name in value['tree']):
        raise PublicationError('The v0.3 Space namespace must be entirely new.')
    return value


def _implementation():
    home = Path(__file__).resolve().parent
    return {name: receipt(read(home/name)) for name in IMPLEMENTATION}


def _batches(files):
    # Declare attributes first, before any new-repository LFS payload. Keep all
    # remaining membership/order deterministic and bounded.
    ordered = sorted(files, key=lambda item: (item['path'] != '.gitattributes', item['path']))
    groups, group, size = [], [], 0
    for item in ordered:
        if group and (len(group) >= BATCH_FILES or size+item['bytes'] > BATCH_BYTES
                      or group == ['.gitattributes']):
            groups.append(group); group, size = [], 0
        group.append(item['path']); size += item['bytes']
    if group:
        groups.append(group)
    return groups


def stage(model, dataset, space, space_baseline, output, *, audit):
    output = plain(output)
    if output.exists():
        raise PublicationError('Publication output must be a fresh directory.')
    baseline_raw, audit_raw = read(space_baseline, MAX_PLAN), read(audit, MAX_PLAN)
    baseline = _baseline(decoded(baseline_raw))
    scan(audit_raw, 'audit.json')
    directories = {kind: plain(value) for kind, value in [('model', model), ('dataset', dataset), ('space', space)] if value is not None}
    if not directories:
        raise PublicationError('Stage at least one fixed publication payload.')
    if any(output == directory or output in directory.parents or directory in output.parents for directory in directories.values()):
        raise PublicationError('Staged output must be separate from every input payload.')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for kind, directory in directories.items():
        # Validate caller inputs before copying; generated root attributes are
        # the sole explicitly declared metadata addition, never science edits.
        for source in directory.rglob('*'):
            plain(source)
            if source.is_file():
                relative(source.relative_to(directory).as_posix())
                scan(read(source), source.name)
        shutil.copytree(directory, output/kind)
        if kind != 'space':
            with (output/kind/'.gitattributes').open('xb') as stream:
                stream.write(ATTRIBUTES)
        files = _inventory(output/kind, kind)
        if kind == 'space':
            attrs = baseline['attributes_utf8']
            for suffix in {PurePosixPath(item['path']).suffix for item in files if item['representation'] == 'lfs'}:
                if f'*{suffix} filter=lfs diff=lfs merge=lfs -text' not in attrs.splitlines():
                    raise PublicationError('Space native assets require an existing exact global LFS suffix rule.')
        rows.append(dict(repo_type=kind, repo_id=REPOS[kind], directory=kind, files=files, batches=_batches(files),
                         expected_parent=baseline['commit'] if kind == 'space' else None))
    if sum(item['bytes'] for row in rows for item in row['files']) > MAX_TOTAL:
        raise PublicationError('Combined publication exceeds its total size bound.')
    (output/'inputs').mkdir()
    (output/'inputs'/'space-baseline.json').write_bytes(baseline_raw)
    (output/'inputs'/'audit.json').write_bytes(audit_raw)
    (output/'implementation').mkdir()
    implementation = _implementation()
    for name in implementation:
        shutil.copyfile(Path(__file__).with_name(name), output/'implementation'/name)
    plan = dict(schema=SCHEMA, public=True, physical_robot_ready=False,
        repositories=rows, space_namespace=NAMESPACE,
        baseline=receipt(baseline_raw), audit=receipt(audit_raw), implementation=implementation,
        scope='Simulation-only research artifacts; no physical skill qualification.',
        new_repository_attributes=receipt(ATTRIBUTES), unchanged_space_baseline_files=len(baseline['tree']))
    atomic(output/PLAN, plan, exclusive=True)
    checked_plan(output/PLAN)
    return plan


def checked_plan(path):
    path = plain(path)
    raw = read(path, MAX_PLAN)
    plan = decoded(raw)
    if (type(plan) is not dict or plan.get('schema') != SCHEMA or plan.get('public') is not True
            or plan.get('physical_robot_ready') is not False or plan.get('space_namespace') != NAMESPACE
            or plan.get('implementation') != _implementation()):
        raise PublicationError('Plan scope, namespace or publisher implementation changed.')
    baseline_raw, audit_raw = read(path.parent/'inputs'/'space-baseline.json', MAX_PLAN), read(path.parent/'inputs'/'audit.json', MAX_PLAN)
    if plan.get('baseline') != receipt(baseline_raw) or plan.get('audit') != receipt(audit_raw):
        raise PublicationError('Reviewed baseline or audit bytes changed.')
    baseline = _baseline(decoded(baseline_raw)); scan(audit_raw, 'audit.json')
    if plan.get('new_repository_attributes') != receipt(ATTRIBUTES) or plan.get('unchanged_space_baseline_files') != len(baseline['tree']):
        raise PublicationError('Pinned metadata or preserved-baseline count changed.')
    if {entry.name for entry in (path.parent/'implementation').iterdir()} != set(IMPLEMENTATION):
        raise PublicationError('Publisher snapshot inventory changed.')
    for name, expected in plan['implementation'].items():
        if receipt(read(path.parent/'implementation'/name)) != expected:
            raise PublicationError('Publisher snapshot bytes changed.')
    rows = plan.get('repositories')
    selected = [row.get('repo_type') for row in rows] if type(rows) is list else []
    if not selected or selected != [kind for kind in REPOS if kind in selected]:
        raise PublicationError('Only an ordered nonempty subset of the fixed v0.3 destinations is authorized.')
    for row in rows:
        kind = row['repo_type']
        if (row.get('repo_id') != REPOS[kind] or row.get('directory') != kind
                or row.get('expected_parent') != (baseline['commit'] if kind == 'space' else None)
                or row.get('files') != _inventory(path.parent/kind, kind) or row.get('batches') != _batches(row['files'])):
            raise PublicationError('Reviewed payload bytes, groups or destinations changed.')
    if sum(item['bytes'] for row in rows for item in row['files']) > MAX_TOTAL:
        raise PublicationError('Combined publication exceeds its total size bound.')
    return plan, baseline


def public_client():
    os.environ.update(HF_DEBUG='0', HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_UPDATE_CHECK='1')
    logging.disable(logging.CRITICAL)
    from huggingface_hub import HfApi
    return HfApi(token=False, endpoint='https://huggingface.co')


def download(repo, name, kind, revision):
    if kind not in REPOS or repo != REPOS[kind] or not isinstance(revision, str) or OID.fullmatch(revision) is None:
        raise PublicationError('Anonymous downloads require a fixed repository and exact revision.')
    relative(name, attributes=True)
    from huggingface_hub import hf_hub_download
    # The Hub's ordinary snapshot cache intentionally contains symlinks. Ask
    # the SDK for a separate local copy, then consume bytes before its private
    # temporary directory is removed. Never weaken plain() for staged evidence
    # or accept a downloader-returned path outside this exact directory.
    with tempfile.TemporaryDirectory(prefix='dvidia-v03-public-readback-') as temporary:
        directory = plain(temporary)
        copied = hf_hub_download(repo, name, repo_type=kind, revision=revision, token=False,
                                 endpoint='https://huggingface.co', local_dir=str(directory))
        path = plain(copied)
        if path != directory/relative(name, attributes=True):
            raise PublicationError('Anonymous local copy differs from the exact requested artifact path.')
        return read(path)


def _downloaded(value, limit=MAX_FILE):
    raw = value if isinstance(value, bytes) else read(value, limit)
    if len(raw) > limit:
        raise PublicationError('Anonymous readback exceeds its declared bound.')
    return raw


def _remote(api, kind, revision=None):
    info = api.repo_info(REPOS[kind], repo_type=kind, revision=revision)
    checked_remote(info, kind)
    if OID.fullmatch(info.sha) is None or revision is not None and info.sha != revision:
        raise PublicationError('Public head or pinned revision differs.')
    tree = {}
    for item in api.list_repo_tree(REPOS[kind], repo_type=kind, revision=info.sha, recursive=True):
        if not hasattr(item, 'blob_id'):
            continue
        if item.path in tree or len(tree) >= MAX_TREE:
            raise PublicationError('Duplicate or excessive public tree.')
        lfs = getattr(item, 'lfs', None)
        tree[item.path] = dict(blob=item.blob_id, size=item.size,
            lfs_sha256=(lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None)
    _tree_valid(tree)
    return info.sha, tree


def capture_space(revision, output, *, api=None, downloader=None):
    if not isinstance(revision, str) or OID.fullmatch(revision) is None:
        raise PublicationError('Capture requires an exact 40-hex Space revision.')
    output = plain(output)
    if output.exists():
        raise PublicationError('Space capture output must be fresh.')
    api, downloader = api or public_client(), downloader or download
    head, tree = _remote(api, 'space', revision)
    if _remote(api, 'space') != (head, tree):
        raise PublicationError('Space head moved before baseline capture.')
    raw = _downloaded(downloader(REPOS['space'], '.gitattributes', 'space', head))
    value = dict(schema=BASELINE_SCHEMA, repo_id=REPOS['space'], repo_type='space', commit=head,
                 tree=tree, attributes_utf8=raw.decode('utf8'), anonymous_readback=True)
    _baseline(value)
    if _remote(api, 'space') != (head, tree):
        raise PublicationError('Space moved during anonymous baseline capture.')
    atomic(output, value, exclusive=True)
    return value


def _not_found(error):
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    return status == 404 if status is not None else type(error).__name__ == 'RepositoryNotFoundError'


def _state(plan, baseline):
    return dict(schema=SCHEMA+'/journal', plan_sha256=sha256(encoded(plan)).hexdigest(), repositories={
        kind: dict(head=baseline['commit'] if kind == 'space' else None,
                   baseline_head=baseline['commit'] if kind == 'space' else None,
                   baseline_tree=baseline['tree'] if kind == 'space' else None,
                   commits=[], pending=None) for kind in [row['repo_type'] for row in plan['repositories']]})


def _journal(path, plan, baseline):
    state = decoded(read(path, MAX_PLAN)) if path.exists() else _state(plan, baseline)
    if (type(state) is not dict or state.get('schema') != SCHEMA+'/journal'
            or state.get('plan_sha256') != sha256(encoded(plan)).hexdigest()
            or type(state.get('repositories')) is not dict
            or set(state['repositories']) != {row['repo_type'] for row in plan['repositories']}):
        raise PublicationError('Recovery journal differs from its exact reviewed plan.')
    expected = {}
    for row in plan['repositories']:
        kind, value = row['repo_type'], state['repositories'][row['repo_type']]
        if type(value) is not dict or set(value) != {'head', 'baseline_head', 'baseline_tree', 'commits', 'pending'}:
            raise PublicationError('Malformed repository recovery state.')
        tree = value['baseline_tree']
        if kind == 'space':
            if tree != baseline['tree'] or value['baseline_head'] != baseline['commit']:
                raise PublicationError('Shared Space recovery baseline changed.')
        elif tree is not None:
            _tree_valid(tree)
            if set(tree)-{'.gitattributes'}:
                raise PublicationError('Created repository baseline contains unrelated files.')
        if tree is None and value['baseline_head'] is not None or tree is not None and OID.fullmatch(value['baseline_head'] or '') is None:
            raise PublicationError('Created repository baseline/head mismatch.')
        wanted = dict(tree or {})
        parent = value['baseline_head']
        if type(value['commits']) is not list or len(value['commits']) > len(row['batches']):
            raise PublicationError('Invalid recovery commit prefix.')
        items = {item['path']: item for item in row['files']}
        for index, entry in enumerate(value['commits']):
            if (type(entry) is not dict or entry.get('index') != index or entry.get('parent') != parent
                    or OID.fullmatch(entry.get('commit', '')) is None or entry.get('paths') != row['batches'][index]
                    or set(entry.get('entries', {})) != set(entry['paths'])
                    or entry.get('message') != _message(state['plan_sha256'], kind, index)):
                raise PublicationError('Recovery commits differ from the exact declared batch prefix.')
            _tree_valid(entry['entries'])
            for name, remote in entry['entries'].items():
                item = items[name]
                if remote['size'] != item['bytes'] or bool(remote['lfs_sha256']) != (item['representation'] == 'lfs') or remote['lfs_sha256'] and remote['lfs_sha256'] != item['sha256']:
                    raise PublicationError('Recovery entry differs from reviewed payload identity.')
            wanted.update(entry['entries']); parent = entry['commit']
        if value['head'] != parent:
            raise PublicationError('Recovery head differs from its exact commit chain.')
        pending = value['pending']
        if pending is not None:
            if pending == {'kind': 'create'}:
                if tree is not None or value['commits'] or kind == 'space':
                    raise PublicationError('Creation intent cannot modify an existing repository.')
            elif len(value['commits']) >= len(row['batches']) or pending != dict(kind='commit', index=len(value['commits']), parent=value['head'],
                                 message=_message(state['plan_sha256'], kind, len(value['commits']))):
                raise PublicationError('Pending commit differs from the exact next batch intent.')
        expected[kind] = wanted
    return state, expected


def _message(plan_sha, kind, index):
    return f'DVIDIA grip clock v0.3 {plan_sha} {kind} batch {index:04}'


def _history(api, kind, revision):
    result = {}
    for item in api.list_repo_commits(REPOS[kind], repo_type=kind, revision=revision):
        oid = getattr(item, 'commit_id', None)
        if OID.fullmatch(oid or '') is None or oid in result or len(result) >= MAX_TREE:
            raise PublicationError('Invalid or excessive public commit history.')
        result[oid] = getattr(item, 'title', None)
    return result


def _finish(api, downloader, plan, state, expected, row, payload, revision, journal, *, reconciled):
    kind = row['repo_type']; value = state['repositories'][kind]
    index = len(value['commits']); paths = row['batches'][index]
    if OID.fullmatch(revision or '') is None or revision == value['head']:
        raise PublicationError('Commit reply lacks a new exact revision; retain pending intent.')
    head, tree = _remote(api, kind)
    old = expected[kind]
    if head != revision or set(tree) != set(old)|set(paths) or any(tree.get(name) != entry for name, entry in old.items() if name not in paths):
        raise PublicationError('Commit changed unexpected paths or moved the guarded parent.')
    def verify(name):
        raw, item = payload[name], next(item for item in row['files'] if item['path'] == name)
        remote = tree[name]
        if bool(remote['lfs_sha256']) != (item['representation'] == 'lfs') or not _matches(raw, remote):
            raise PublicationError('Public Git/LFS representation differs from reviewed bytes.')
        if _downloaded(downloader(REPOS[kind], name, kind, head)) != raw:
            raise PublicationError('Anonymous pinned readback differs from reviewed payload.')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, paths))
    if _remote(api, kind) != (head, tree):
        raise PublicationError('Public repository moved during anonymous batch readback.')
    value['commits'].append(dict(index=index, parent=value['head'], commit=head, paths=paths,
        entries={name: tree[name] for name in paths}, message=_message(state['plan_sha256'], kind, index),
        anonymous_readback=True, reconciled=reconciled))
    value['head'], value['pending'] = head, None
    expected[kind] = tree
    atomic(journal, state)


def _created(api, kind, state, expected, journal):
    head, tree = _remote(api, kind)
    if set(tree)-{'.gitattributes'}:
        raise PublicationError('Fresh repository creation produced unrelated existing content.')
    value = state['repositories'][kind]
    value.update(head=head, baseline_head=head, baseline_tree=tree, pending=None)
    expected[kind] = tree
    atomic(journal, state)


def _reconcile(api, downloader, plan, state, expected, row, payload, journal):
    kind = row['repo_type']; value = state['repositories'][kind]; pending = value['pending']
    if pending == {'kind': 'create'}:
        try:
            _created(api, kind, state, expected, journal)
        except Exception as error:
            if not _not_found(error):
                raise
            value['pending'] = None; atomic(journal, state)
        return
    if value['head'] is None:
        try:
            _remote(api, kind)
        except Exception as error:
            if _not_found(error):
                return
            raise
        raise PublicationError('A new destination already exists without this publication creation intent.')
    head, tree = _remote(api, kind)
    if pending is None:
        if head != value['head'] or tree != expected[kind]:
            raise PublicationError('Repository moved outside the exact reviewed journal.')
        return
    if head == value['head']:
        if tree != expected[kind] or _remote(api, kind) != (head, tree):
            raise PublicationError('Pending outcome is still changing; retain its journal.')
        # Atomic identical-parent commits allow at most one of a late original
        # request and this exact resumed request to become the parent's child.
        value['pending'] = None; atomic(journal, state)
        return
    before, after = _history(api, kind, value['head']), _history(api, kind, head)
    if set(after) != set(before)|{head} or after.get(head) != pending['message']:
        raise PublicationError('Unknown reply is not the exact intended immediate child.')
    _finish(api, downloader, plan, state, expected, row, payload, head, journal, reconciled=True)


@contextmanager
def _lock(directory):
    import fcntl
    descriptor = os.open(plain(directory/'publisher.lock'), os.O_CREAT|os.O_RDWR|getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise PublicationError('Another v0.3 publisher is active.') from None
        yield
    finally:
        os.close(descriptor)


def publish(path, *, api=None, downloader=None, operation_cls=None):
    path = plain(path)
    with _lock(path.parent):
        plan, baseline = checked_plan(path)
        plan_raw = read(path, MAX_PLAN)
        if (path.parent/RECEIPT).exists():
            completed = decoded(read(path.parent/RECEIPT, MAX_PLAN))
            if type(completed) is not dict:
                raise PublicationError('Existing final receipt is not a publication record.')
            rows = completed.get('repositories') if type(completed) is dict else None
            if (completed.get('schema') != SCHEMA+'/receipt' or completed.get('plan') != receipt(plan_raw)
                    or completed.get('anonymous_full_readback') is not True or completed.get('physical_robot_ready') is not False
                    or type(rows) is not list or len(rows) != len(plan['repositories'])
                    or any(row.get('repo_type') != planned['repo_type'] or row.get('repo_id') != planned['repo_id']
                           or OID.fullmatch(row.get('commit', '')) is None
                           or row.get('verified_files') != len(planned['files'])
                           or row.get('verified_bytes') != sum(item['bytes'] for item in planned['files'])
                           for row, planned in zip(rows, plan['repositories']))):
                raise PublicationError('Existing final receipt differs from this plan.')
            return completed
        journal = path.parent/JOURNAL
        state, expected = _journal(journal, plan, baseline)
        token = None
        if api is None:
            api, token = client(path.parent/'publisher-cache')
            identity = api.whoami()
            if 'dvidia' not in {str(item.get('name', '')).lower() for item in identity.get('orgs', [])}:
                raise PublicationError('Authenticated account is not a member of the fixed Dvidia organization.')
        downloader = downloader or download
        if operation_cls is None:
            from huggingface_hub import CommitOperationAdd
            operation_cls = CommitOperationAdd
        # Preflight every destination before the first remote mutation.
        for row in plan['repositories']:
            value = state['repositories'][row['repo_type']]
            pending = value['pending']
            paths = row['batches'][len(value['commits'])] if pending is not None and pending.get('kind') == 'commit' else []
            payload = {name: read(path.parent/row['repo_type']/name) for name in paths}
            _reconcile(api, downloader, plan, state, expected, row, payload, journal)
        if not journal.exists():
            atomic(journal, state, exclusive=True)
        for row in plan['repositories']:
            kind = row['repo_type']; value = state['repositories'][kind]
            if value['head'] is None:
                value['pending'] = {'kind': 'create'}; atomic(journal, state)
                try:
                    api.create_repo(REPOS[kind], repo_type=kind, private=False, exist_ok=False)
                except Exception:
                    raise PublicationError('Repository creation reply is uncertain; preserve the exact creation intent.') from None
                _created(api, kind, state, expected, journal)
            for index in range(len(value['commits']), len(row['batches'])):
                if _implementation() != plan['implementation'] or read(path, MAX_PLAN) != plan_raw:
                    raise PublicationError('Publisher or reviewed plan changed during publication.')
                paths = row['batches'][index]
                items = {item['path']: item for item in row['files']}
                payload = {name: read(path.parent/kind/name) for name in paths}
                for name, raw in payload.items():
                    if receipt(raw) != {key: items[name][key] for key in ('bytes', 'sha256')} or token and token.encode() in raw:
                        raise PublicationError('Frozen upload artifact changed or contains the publishing credential.')
                if _remote(api, kind) != (value['head'], expected[kind]):
                    raise PublicationError('Destination moved before the next guarded commit.')
                operations = []
                for name in paths:
                    operation = operation_cls(path_in_repo=name, path_or_fileobj=payload[name])
                    operation._upload_mode = 'lfs' if items[name]['representation'] == 'lfs' else 'regular'
                    operation._should_ignore = False
                    operations.append(operation)
                message = _message(state['plan_sha256'], kind, index)
                value['pending'] = dict(kind='commit', index=index, parent=value['head'], message=message)
                atomic(journal, state)
                try:
                    result = api.create_commit(REPOS[kind], repo_type=kind, parent_commit=value['head'],
                                               operations=operations, commit_message=message)
                except Exception:
                    raise PublicationError('Commit reply is uncertain; preserve the exact pending intent for reconciliation.') from None
                _finish(api, downloader, plan, state, expected, row, payload, getattr(result, 'oid', None), journal, reconciled=False)
                print(json.dumps(dict(phase='verified-batch', repo_type=kind, index=index, batches=len(row['batches']))), flush=True)
        # Read every staged byte anonymously at each final pinned commit. Older
        # Space objects are preserved by full-tree Git/LFS identity equality.
        results = []
        for row in plan['repositories']:
            kind = row['repo_type']; value = state['repositories'][kind]
            if _remote(api, kind) != (value['head'], expected[kind]):
                raise PublicationError('Final repository differs from the completed journal.')
            def verify(item):
                raw = _downloaded(downloader(REPOS[kind], item['path'], kind, value['head']))
                if receipt(raw) != {key: item[key] for key in ('bytes', 'sha256')}:
                    raise PublicationError('Final anonymous full readback differs from reviewed bytes.')
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(verify, row['files']))
            if _remote(api, kind) != (value['head'], expected[kind]):
                raise PublicationError('Repository moved during final anonymous full readback.')
            results.append(dict(repo_type=kind, repo_id=REPOS[kind], commit=value['head'],
                verified_files=len(row['files']), verified_bytes=sum(item['bytes'] for item in row['files']),
                preserved_baseline_files=len(baseline['tree']) if kind == 'space' else 0,
                commits=value['commits']))
        has_space = 'space' in state['repositories']
        completed = dict(schema=SCHEMA+'/receipt', plan=receipt(plan_raw), repositories=results,
            anonymous_full_readback=True, physical_robot_ready=False,
            preserved_space_baseline=plan['baseline'] if has_space else None,
            viewer_url='https://dvidia-robot-simulation-lab.static.hf.space/'+NAMESPACE+'viewer/index.html' if has_space else None)
        atomic(path.parent/RECEIPT, completed, exclusive=True)
        return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capture = sub.add_parser('capture-space')
    capture.add_argument('--revision', required=True); capture.add_argument('--output', required=True)
    prepare = sub.add_parser('stage')
    for name in ('model', 'dataset', 'space'):
        prepare.add_argument('--'+name)
    for name in ('space-baseline', 'audit', 'output'):
        prepare.add_argument('--'+name, required=True)
    for name in ('check', 'publish'):
        sub.add_parser(name).add_argument('plan')
    args = parser.parse_args(argv)
    try:
        if args.command == 'capture-space':
            value = capture_space(args.revision, args.output)
        elif args.command == 'stage':
            value = stage(args.model, args.dataset, args.space, args.space_baseline, args.output, audit=args.audit)
        elif args.command == 'check':
            value = checked_plan(args.plan)[0]
        else:
            value = publish(args.plan)
        print(json.dumps(dict(command=args.command, schema=value['schema'], complete=True)), flush=True)
        return 0
    except Exception as error:
        # Never emit SDK responses, request headers, tracebacks or credentials.
        message = str(error) if isinstance(error, PublicationError) else type(error).__name__
        print('v0.3 publication stopped: '+message, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
