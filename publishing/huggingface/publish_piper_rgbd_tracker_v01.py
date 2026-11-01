"""Stage or publish the reviewed PiPER RGB-D tracker v0.1 evidence dataset.

This publisher is intentionally inert until its explicit ``publish`` command.
It freezes exactly six reviewed files and writes one commit only to an empty,
public dataset head. It does not modify any Space.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

from publish import PublicationError, client
from publish_task_environments_v02 import digest, encoded, plain, public_client, download, scan


REPO = 'Dvidia/piper-rgbd-tracker-v01'
SCHEMA = 'dvidia.piper-incremental-hold-publication/v1'
ARCHIVE = 'piper-rgbd-tracker-v01.zip'
NAMES = {'README.md', 'LICENSE', ARCHIVE, 'report.json', 'protocol.json', 'inspection.json'}
PLAN_LIMIT = 4 * 1024 * 1024
LIMITS = {name: (192 * 1024 * 1024 if name == ARCHIVE else
                 16 * 1024 * 1024 if name == 'report.json' else 4 * 1024 * 1024)
          for name in NAMES}
OID = re.compile(r'(?:[a-f0-9]{40}|[a-f0-9]{64})\Z')


def read(path, limit=None):
    path = plain(path)
    limit = LIMITS.get(path.name, PLAN_LIMIT) if limit is None else limit
    if not path.is_file() or path.stat().st_size > limit:
        raise PublicationError('Missing or oversized reviewed publication file.')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise PublicationError('Publication file changed beyond its byte limit.')
    return raw


def files(root):
    root = plain(root)
    if not root.is_dir():
        raise PublicationError('Expected a plain publication directory.')
    paths = sorted(root.iterdir())
    if {path.name for path in paths} != NAMES:
        raise PublicationError('Dataset must contain exactly the reviewed six files.')
    inventory = []
    for path in paths:
        raw = read(path)
        scan(raw)
        inventory.append({'path': path.name, **digest(raw),
                          'representation': 'lfs' if path.name == ARCHIVE else 'git'})
    return inventory


def stage(root, plan):
    root, plan = plain(root), plain(plan)
    value = {'schema': SCHEMA, 'repo_id': REPO, 'repo_type': 'dataset',
             'private': False, 'root': str(root), 'files': files(root)}
    with plan.open('xb') as stream:
        stream.write(encoded(value))
    return {'plan_sha256': sha256(read(plan, PLAN_LIMIT)).hexdigest(),
            'files': len(value['files']), 'bytes': sum(row['bytes'] for row in value['files'])}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise PublicationError('Duplicate keys in reviewed publication plan.')
        result[key] = value
    return result


def _constant(_value):
    raise PublicationError('Nonfinite values in reviewed publication plan.')


def checked(plan, expected_digest):
    raw = read(plan, PLAN_LIMIT)
    if sha256(raw).hexdigest() != expected_digest:
        raise PublicationError('Reviewed plan digest changed.')
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise PublicationError('Malformed reviewed publication plan.') from exc
    if (type(value) is not dict
            or set(value) != {'schema', 'repo_id', 'repo_type', 'private', 'root', 'files'}
            or value['schema'] != SCHEMA or value['repo_id'] != REPO
            or value['repo_type'] != 'dataset' or value['private'] is not False
            or type(value['root']) is not str or type(value['files']) is not list
            or len(value['files']) != len(NAMES)):
        raise PublicationError('Reviewed dataset identity changed.')
    for row in value['files']:
        if (type(row) is not dict or set(row) != {'path', 'bytes', 'sha256', 'representation'}
                or type(row['path']) is not str or row['path'] not in NAMES
                or type(row['bytes']) is not int or row['bytes'] < 0
                or type(row['sha256']) is not str or not re.fullmatch(r'[a-f0-9]{64}', row['sha256'])
                or row['representation'] != ('lfs' if row['path'] == ARCHIVE else 'git')):
            raise PublicationError('Reviewed dataset inventory is malformed.')
    if ({row['path'] for row in value['files']} != NAMES
            or files(value['root']) != value['files']):
        raise PublicationError('Reviewed dataset identity or bytes changed.')
    return value


def _remote(info):
    if (getattr(info, 'private', None) is not False
            or not OID.fullmatch(getattr(info, 'sha', '') or '')
            or not getattr(info, 'siblings', None)):
        raise PublicationError('Expected an initialized public dataset with an exact commit.')


def publish(plan, expected_digest, cache):
    value = checked(plan, expected_digest)
    payloads = []
    for row in value['files']:
        raw = read(Path(value['root']) / row['path'])
        if digest(raw) != {key: row[key] for key in ('bytes', 'sha256')}:
            raise PublicationError('Payload changed while freezing reviewed publication bytes.')
        payloads.append((row['path'], raw))
    # No payload file is reopened after credential loading or remote access.
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
        raise PublicationError('Dataset is not empty. Verify the existing publication; do not overwrite it.')
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=raw) for name, raw in payloads]
    commit = api.create_commit(REPO, repo_type='dataset', operations=operations,
                               parent_commit=info.sha,
                               commit_message='Publish PiPER RGB-D tracker v0.1 simulator evidence')
    del token, api
    return {'repo_id': REPO, 'commit': commit.oid, 'files_added': len(operations)}


def verify(plan, expected_digest, cache):
    value = checked(plan, expected_digest)
    api = public_client()
    info = api.repo_info(REPO, repo_type='dataset', files_metadata=True, token=False)
    _remote(info)
    if (len(info.siblings) != len(NAMES) + 1
            or {item.rfilename for item in info.siblings} != NAMES | {'.gitattributes'}):
        raise PublicationError('Public dataset roster differs from the reviewed files.')
    for row in value['files']:
        raw = download(REPO, row['path'], 'dataset', info.sha, cache)
        if digest(raw) != {key: row[key] for key in ('bytes', 'sha256')}:
            raise PublicationError('Anonymous public bytes differ from the reviewed artifacts.')
    return {'repo_id': REPO, 'commit': info.sha, 'anonymous_files_verified': len(value['files'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('stage')
    command.add_argument('root'); command.add_argument('--plan', required=True)
    for name in ('publish', 'verify'):
        command = commands.add_parser(name)
        command.add_argument('--plan', required=True)
        command.add_argument('--sha256', required=True)
        command.add_argument('--cache', required=True)
    args = parser.parse_args()
    try:
        result = (stage(args.root, args.plan) if args.command == 'stage' else
                  globals()[args.command](args.plan, args.sha256, Path(args.cache)))
        print(json.dumps(result, indent=2))
        return 0
    except PublicationError as exc:
        print(str(exc), file=sys.stderr)
    except Exception:
        print('Publication operation failed; no credential or remote response details displayed.', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
