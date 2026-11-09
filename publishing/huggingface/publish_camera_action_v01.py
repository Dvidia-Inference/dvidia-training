"""Publish one reviewed, byte-bound camera/action evidence dataset.

Stage and verify never read credentials. Publish copies the reviewed payload
into a private temporary directory before Keychain access, then makes one
parent-bound commit to the named empty public dataset. No Space is modified.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

from publish import PublicationError, client
from publish_task_environments_v02 import encoded, plain, public_client, relative, scan
from camera_action_hardware import MAX_BYTES as MAX_HARDWARE_BYTES, validate_hardware
from camera_action_model_cards import RECEIPT_FILE, MAX_RECEIPT_BYTES, validate_receipt

REPO = 'Dvidia/piper-camera-action-v01'
SCHEMA = 'dvidia.camera-action-publication/v1'
EVIDENCE_SCHEMA = 'dvidia.camera-action-evidence/v1'
MAX_JSON = 32 * 1024 ** 2
MAX_FILE = 1024 ** 3
MAX_TOTAL = 120 * 1024 ** 3
RESERVE = 30 * 1024 ** 3
OID = re.compile(r'(?:[a-f0-9]{40}|[a-f0-9]{64})\Z')
REQUIRED = {'README.md', 'LICENSE', 'manifest.json', 'SHA256SUMS', 'report.json',
            'protocol.json', 'registration.json', 'independent-audit.json', 'study-metadata.zip'}


def identity(path, *, maximum=MAX_FILE):
    path = plain(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise PublicationError('Publication file is missing, nonregular or oversized.')
        h, count, tail = sha256(), 0, b''
        for raw in iter(lambda: stream.read(1024 * 1024), b''):
            count += len(raw)
            if count > maximum:
                raise PublicationError('Publication payload grew beyond its bound.')
            h.update(raw)
            scan(tail + raw)
            tail = raw[-256:]
        after = os.fstat(stream.fileno())
        if count != before.st_size or any(getattr(before, key) != getattr(after, key)
                for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise PublicationError('Publication bytes changed during inspection.')
    return dict(bytes=count, sha256=h.hexdigest())


def read(path):
    path = plain(path)
    identity(path, maximum=MAX_JSON)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PublicationError('Duplicate JSON field in reviewed publication artifact.')
            result[key] = value
        return result
    def invalid(_value):
        raise PublicationError('Nonfinite reviewed publication artifact.')
    try:
        return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise PublicationError('Malformed reviewed publication artifact.') from exc


def allowed(name):
    relative(name)
    return (name in REQUIRED or name in ('diagnostic-phase-coverage.json', 'full-study-audit.json', 'hardware.json', RECEIPT_FILE)
        or re.fullmatch(r'episodes/[a-z-]+/[a-z0-9-]+\.zip', name) is not None
        # The frozen v1 roster has 16 negative-control archives with these
        # underscore-bearing fault names. Do not broaden other episode paths.
        or re.fullmatch(r'episodes/negative/negative-confirmation-0[01]-goal[01]-student-'
                        r'(initial_missing|initial_stale|altered_calibration|open_gripper)\.zip', name) is not None
        or re.fullmatch(r'models/(initial|corrected)/(full|blind)/model\.json', name) is not None
        or re.fullmatch(r'model-cards/(initial|corrected)/(full|blind)/README\.md', name) is not None
        or re.fullmatch(r'source/[A-Za-z0-9_.+-]+\.(whl|tar\.gz)', name) is not None)


def _record(value):
    if (not isinstance(value, dict) or set(value) != {'bytes', 'sha256'}
            or type(value['bytes']) is not int or not 0 <= value['bytes'] <= MAX_FILE
            or not isinstance(value['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', value['sha256'])):
        raise PublicationError('Malformed reviewed file identity.')
    return value


def files(root):
    root = plain(root)
    manifest = read(root / 'manifest.json')
    if (manifest.get('schema') != EVIDENCE_SCHEMA or manifest.get('repo_id') != REPO
            or manifest.get('repo_type') != 'dataset' or manifest.get('physical_robot_ready') is not False
            or manifest.get('original_footage_included') is not False
            or manifest.get('status') not in ('blocked', 'complete', 'incomplete')
            or manifest.get('pipeline_status', manifest.get('status')) not in ('blocked', 'complete')
            or type(manifest.get('qualified')) is not bool):
        raise PublicationError('Expected the reviewed simulation-only camera/action evidence manifest.')
    expected = manifest.get('files')
    if not isinstance(expected, dict) or not REQUIRED - {'manifest.json', 'SHA256SUMS'} <= set(expected):
        raise PublicationError('Required evidence files are missing.')
    if any(not allowed(name) or name in ('manifest.json', 'SHA256SUMS') for name in expected):
        raise PublicationError('Unrecognized public evidence path.')
    if manifest.get('hardware_context_file') != ('hardware.json' if 'hardware.json' in expected else None):
        raise PublicationError('Hardware context file differs from its manifest binding.')
    if 'hardware.json' in expected:
        identity(root / 'hardware.json', maximum=MAX_HARDWARE_BYTES)
        try:
            validate_hardware(read(root / 'hardware.json'))
        except ValueError as exc:
            raise PublicationError('Hardware context contains unknown, private or unbounded fields.') from exc
    observed = []
    for path in sorted(root.rglob('*')):
        plain(path)
        if not (path.is_file() or path.is_dir()):
            raise PublicationError('Public evidence contains a special file.')
        if path.is_file():
            observed.append(path.relative_to(root).as_posix())
    if not 10 <= len(observed) <= 1024 or set(observed) != set(expected) | {'manifest.json', 'SHA256SUMS'}:
        raise PublicationError('Public file roster differs from the reviewed manifest.')
    # Match the writer's POSIX-name order, including model-cards/ versus
    # model-cards-receipt.json (Path component ordering differs here).
    observed.sort()
    source_files = [name for name in expected if name.startswith('source/')]
    if len(source_files) != 2 or sum(name.endswith('.whl') for name in source_files) != 1 or sum(name.endswith('.tar.gz') for name in source_files) != 1:
        raise PublicationError('Exact source archive and installable wheel are required.')
    episodes = manifest.get('episodes')
    if not isinstance(episodes, list) or len(episodes) != manifest.get('episode_count'):
        raise PublicationError('Episode shard roster is malformed.')
    if {r.get('archive') for r in episodes} != {name for name in expected if name.startswith('episodes/')}:
        raise PublicationError('Episode archives differ from the declared roster.')
    if set(manifest.get('trained_model_files', [])) != {name for name in expected if name.startswith('models/')}:
        raise PublicationError('Model artifacts differ from the actual training roster.')
    report = read(root / 'report.json')
    if (report.get('qualification') != manifest.get('qualification') or report.get('qualified') is not manifest['qualified']
            or report.get('physical_robot_ready') is not False or report.get('status') != manifest['status']):
        raise PublicationError('Public report and qualification manifest disagree.')
    card_files = {name for name in expected if name.startswith('model-cards/')}
    declared_cards = manifest.get('model_card_files', [])
    card_receipt = manifest.get('model_card_receipt_file')
    if card_receipt is None:
        if declared_cards or card_files or RECEIPT_FILE in expected:
            raise PublicationError('Model cards require their explicit receipt and manifest bindings.')
    else:
        if card_receipt != RECEIPT_FILE or RECEIPT_FILE not in expected:
            raise PublicationError('Model-card receipt path differs from its manifest binding.')
        identity(root / RECEIPT_FILE, maximum=MAX_RECEIPT_BYTES)
        receipt = read(root / RECEIPT_FILE)
        try:
            required_cards = validate_receipt(receipt, source_commit=manifest.get('source_commit'),
                protocol_sha256=expected['protocol.json']['sha256'], report_sha256=expected['report.json']['sha256'],
                report=report, models={name: expected[name] for name in manifest['trained_model_files']})
        except (ValueError, KeyError, TypeError) as exc:
            raise PublicationError('Model-card receipt differs from the public models or report.') from exc
        if (declared_cards != required_cards or card_files != set(required_cards)
                or any(receipt['files'][name] != expected[name] for name in ['README.md', *required_cards])
                or any(receipt['study_bindings'][name] != expected[name]
                    for name in ('protocol.json', 'registration.json', 'report.json'))
                or any(receipt['source_inputs'].get(name) != expected.get(name)
                    for name in ('hardware.json', 'full-study-audit.json', 'diagnostic-phase-coverage.json'))
                or ('independent-audit.json' in receipt['source_inputs'] and
                    receipt['source_inputs']['independent-audit.json'] != expected.get('independent-audit.json'))
                or receipt['full_study_audited'] is not ('full-study-audit.json' in expected)):
            raise PublicationError('Model-card files or provenance differ from the publication manifest.')
    if manifest['qualified']:
        if 'full-study-audit.json' not in expected:
            raise PublicationError('A qualified skill requires the full-study independent audit.')
        proof = read(root / 'full-study-audit.json')
        if (proof.get('schema') != 'dvidia.camera-action-study-audit/v1' or proof.get('passed') is not True
                or proof.get('errors') != [] or (proof.get('qualification') or {}).get('qualified') is not True):
            raise PublicationError('Full-study audit does not support the qualified-skill label.')
    inventory = []
    for name in observed:
        record = identity(root / name)
        if name in expected and record != _record(expected[name]):
            raise PublicationError('Payload bytes differ from the reviewed evidence manifest.')
        inventory.append(dict(path=name, **record,
            representation='lfs' if name.endswith(('.zip', '.gz', '.whl')) else 'git'))
    if sum(row['bytes'] for row in inventory) > MAX_TOTAL:
        raise PublicationError('Public evidence exceeds the bounded dataset size.')
    sums = ''.join(f'{row["sha256"]}  {row["path"]}\n' for row in inventory if row['path'] != 'SHA256SUMS')
    if (root / 'SHA256SUMS').read_text() != sums:
        raise PublicationError('Published checksum list does not match all payload bytes.')
    return inventory


def stage(root, plan):
    root, plan = plain(root), plain(plan)
    value = dict(schema=SCHEMA, repo_id=REPO, repo_type='dataset', private=False,
                 root=str(root), files=files(root))
    with plan.open('xb') as stream:
        stream.write(encoded(value))
    return dict(plan_sha256=identity(plan, maximum=MAX_JSON)['sha256'], files=len(value['files']),
                bytes=sum(r['bytes'] for r in value['files']))


def checked(plan, expected_digest):
    if identity(plan, maximum=MAX_JSON)['sha256'] != expected_digest:
        raise PublicationError('Reviewed publication plan digest changed.')
    value = read(plan)
    if (set(value) != {'schema', 'repo_id', 'repo_type', 'private', 'root', 'files'}
            or value['schema'] != SCHEMA or value['repo_id'] != REPO or value['repo_type'] != 'dataset'
            or value['private'] is not False or type(value['root']) is not str or type(value['files']) is not list):
        raise PublicationError('Reviewed publication destination or scope changed.')
    if files(value['root']) != value['files']:
        raise PublicationError('Reviewed publication inventory changed.')
    return value


def _remote(info):
    if (getattr(info, 'private', None) is not False or not OID.fullmatch(getattr(info, 'sha', '') or '')
            or getattr(info, 'siblings', None) is None):
        raise PublicationError('Expected an exact initialized public dataset head.')


def _space_guard(directory, additional=0):
    if shutil.disk_usage(directory).free - additional < RESERVE:
        raise PublicationError('Freezing the publication would consume the protected disk reserve.')


def _clone_from_fd(source_fd, directory_fd, name):
    """Darwin clonefile(2): new inode/attributes; subsequent writes are private.

    Source and destination directory are already open without following links.
    No hardlink fallback is permitted. Local SDK clonefile.h defines the
    CLONE_NOOWNERCOPY flag as 0x0002; source is a regular-file descriptor.
    """
    if sys.platform != 'darwin':
        return False
    library = ctypes.CDLL(None, use_errno=True)
    try:
        clone = library.fclonefileat
    except AttributeError:
        return False
    clone.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
    clone.restype = ctypes.c_int
    if clone(source_fd, directory_fd, os.fsencode(name), 0x0002) == 0:
        return True
    number = ctypes.get_errno()
    if number in {errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.ENOSYS}:
        return False
    raise OSError(number, 'Private copy-on-write freeze failed.')


def freeze_file(source, target, expected):
    """Create and verify one private snapshot, with bounded streaming fallback."""
    source, target = plain(source), plain(target)
    _record(expected)
    if target.exists():
        raise PublicationError('Private freeze destination already exists.')
    target.parent.mkdir(parents=True, exist_ok=True)
    plain(target.parent)
    _space_guard(target.parent, 1024 * 1024)  # metadata headroom; no full-data reservation for a clone
    created = False
    source_fd = directory_fd = None
    try:
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected['bytes']:
            raise PublicationError('Freeze source is not the reviewed ordinary file.')
        cloned = _clone_from_fd(source_fd, directory_fd, target.name)
        if cloned:
            created = True
        else:
            _space_guard(target.parent, expected['bytes'])
            fd = os.open(target.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
            created = True
            with os.fdopen(fd, 'wb') as destination:
                copied = 0
                while True:
                    chunk = os.read(source_fd, 1024 * 1024)
                    if not chunk: break
                    copied += len(chunk)
                    if copied > expected['bytes']:
                        raise PublicationError('Freeze source grew beyond its reviewed byte count.')
                    _space_guard(target.parent, len(chunk))
                    destination.write(chunk)
                    destination.flush()
                    _space_guard(target.parent)
                if copied != expected['bytes']:
                    raise PublicationError('Freeze source length changed while copying.')
                os.fsync(destination.fileno())
        after = os.fstat(source_fd)
        if any(getattr(before, k) != getattr(after, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise PublicationError('Source changed while freezing reviewed publication bytes.')
        frozen_fd = os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            frozen_stat = os.fstat(frozen_fd)
            if (not stat.S_ISREG(frozen_stat.st_mode) or frozen_stat.st_nlink != 1
                    or (frozen_stat.st_dev, frozen_stat.st_ino) == (before.st_dev, before.st_ino)):
                raise PublicationError('Freeze must create an independent ordinary inode, never a hardlink.')
            os.fchmod(frozen_fd, 0o400)
        finally:
            os.close(frozen_fd)
        _space_guard(target.parent)
        if identity(target) != expected:
            raise PublicationError('Payload changed while freezing reviewed publication bytes.')
        return 'apfs_clone' if cloned else 'stream_copy'
    except BaseException:
        if created:
            os.unlink(target.name, dir_fd=directory_fd)
        raise
    finally:
        if source_fd is not None: os.close(source_fd)
        if directory_fd is not None: os.close(directory_fd)


def publish(plan, expected_digest, cache):
    value = checked(plan, expected_digest)
    cache = plain(cache)
    cache.mkdir(parents=True, exist_ok=True)
    _space_guard(cache, 1024 * 1024)
    with tempfile.TemporaryDirectory(prefix='camera-action-frozen-', dir=cache) as temporary:
        frozen = Path(temporary)
        freeze_methods = dict(apfs_clone=0, stream_copy=0)
        for row in value['files']:
            source = plain(Path(value['root']) / row['path'])
            target = frozen / relative(row['path'])
            method = freeze_file(source, target, {key: row[key] for key in ('bytes', 'sha256')})
            freeze_methods[method] += 1
        # Originals are never reopened after the credential boundary. The
        # private immutable copy bounds memory while preserving reviewed bytes.
        api, token = client(cache)
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.errors import RepositoryNotFoundError
        try:
            info = api.repo_info(REPO, repo_type='dataset', files_metadata=True)
        except RepositoryNotFoundError:
            api.create_repo(REPO, repo_type='dataset', private=False, exist_ok=False)
            info = api.repo_info(REPO, repo_type='dataset', files_metadata=True)
        _remote(info)
        if len(info.siblings) != 1 or {item.rfilename for item in info.siblings} != {'.gitattributes'}:
            raise PublicationError('Dataset already contains content; verify it instead of overwriting it.')
        operations = [CommitOperationAdd(path_in_repo=row['path'], path_or_fileobj=str(frozen / row['path']))
                      for row in value['files']]
        commit = api.create_commit(REPO, repo_type='dataset', operations=operations,
            parent_commit=info.sha, commit_message='Publish camera-to-action v0.1 reproducible simulation evidence')
        del token, api
        return dict(repo_id=REPO, commit=commit.oid, files_added=len(operations), freeze_methods=freeze_methods)


def _http_readback():
    # Hub constants are initialized at import time. Fail closed if a caller
    # pre-imported an SDK that would route data into a separate Xet cache.
    os.environ.update(HF_HUB_DISABLE_XET='1', HF_DEBUG='0', HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DISABLE_UPDATE_CHECK='1')
    from huggingface_hub import constants
    if getattr(constants, 'HF_HUB_DISABLE_XET', None) is not True:
        raise PublicationError('Start verification in a fresh process with HF_HUB_DISABLE_XET=1 before SDK import.')


def download(name, revision, cache):
    _http_readback()
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, name, repo_type='dataset', revision=revision,
        token=False, endpoint='https://huggingface.co', cache_dir=str(cache), force_download=True)).resolve()


def verify(plan, expected_digest, cache):
    value = checked(plan, expected_digest)
    cache = plain(cache)
    cache.mkdir(parents=True, exist_ok=True)
    _http_readback()
    api = public_client()
    info = api.repo_info(REPO, repo_type='dataset', files_metadata=True, token=False)
    _remote(info)
    expected = {row['path'] for row in value['files']}
    if len(info.siblings) != len(expected) + 1 or {item.rfilename for item in info.siblings} != expected | {'.gitattributes'}:
        raise PublicationError('Public dataset roster differs from the reviewed plan.')
    remote_files = {item.rfilename: item for item in info.siblings}
    for row in value['files']:
        size = getattr(remote_files[row['path']], 'size', None)
        if type(size) is not int or size != row['bytes']:
            raise PublicationError('Revision-pinned remote size differs from the reviewed per-file download budget.')
    for row in value['files']:
        _space_guard(cache, row['bytes'] + 1024 * 1024)
        with tempfile.TemporaryDirectory(prefix='camera-action-readback-', dir=cache) as temporary:
            owned = Path(temporary)
            path = download(row['path'], info.sha, owned)
            if not path.resolve().is_relative_to(owned.resolve()):
                raise PublicationError('Anonymous readback escaped its owned temporary cache.')
            if identity(path) != {key: row[key] for key in ('bytes', 'sha256')}:
                raise PublicationError('Anonymous public bytes differ from reviewed evidence.')
            _space_guard(cache)
    return dict(repo_id=REPO, commit=info.sha, anonymous_files_verified=len(value['files']),
        anonymous_bytes_verified=sum(r['bytes'] for r in value['files']),
        maximum_readback_file_bytes=max(r['bytes'] for r in value['files']),
        reviewed_plan_sha256=expected_digest,
        readback_inventory_sha256=sha256(encoded(value['files'])).hexdigest(),
        cache_policy='One file per owned temporary cache, removed after full pinned-revision byte verification; shared caches untouched.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('stage')
    create.add_argument('root')
    create.add_argument('--plan', required=True)
    for name in ('publish', 'verify'):
        command = commands.add_parser(name)
        command.add_argument('--plan', required=True)
        command.add_argument('--sha256', required=True)
        command.add_argument('--cache', required=True)
    args = parser.parse_args(argv)
    try:
        result = stage(args.root, args.plan) if args.command == 'stage' else globals()[args.command](args.plan, args.sha256, args.cache)
        print(json.dumps(result, indent=2))
        return 0
    except PublicationError as exc:
        print(str(exc), file=sys.stderr)
    except Exception:
        print('Publication operation failed; no credential or remote response details displayed.', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
