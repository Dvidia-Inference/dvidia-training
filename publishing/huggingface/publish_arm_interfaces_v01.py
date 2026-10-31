"""Add one bounded public benchmark dataset; never modify shared Spaces.

Uses the existing named Keychain client only during publish. Credentials remain
in process memory; remote failures produce fixed diagnostics, not raw responses.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from publish import PublicationError, client
from publish_task_environments_v02 import digest, encoded, files, plain, read, public_client, download

REPO = 'Dvidia/arm-interfaces-v01'
SCHEMA = 'dvidia.arm-interfaces-publication/v1'
NAMES = {'README.md', 'LICENSE', 'arm-interfaces-v01-evidence.zip',
         'summary.json', 'independent-audit.json', 'installation-smoke.json'}


def stage(root, plan):
    root, plan = plain(root), plain(plan)
    inventory = files(root)
    if {r['path'] for r in inventory} != NAMES:
        raise PublicationError('Dataset must contain exactly the reviewed six files.')
    value = {'schema': SCHEMA, 'repo_id': REPO, 'repo_type': 'dataset',
             'private': False, 'root': str(root), 'files': inventory}
    with plan.open('xb') as stream:
        stream.write(encoded(value))
    return {'plan_sha256': sha256(read(plan)).hexdigest(), 'files': len(inventory),
            'bytes': sum(r['bytes'] for r in inventory)}


def checked(plan, expected_digest):
    raw = read(plan)
    if sha256(raw).hexdigest() != expected_digest:
        raise PublicationError('Reviewed plan digest changed.')
    value = json.loads(raw)
    if (set(value) != {'schema', 'repo_id', 'repo_type', 'private', 'root', 'files'}
            or value['schema'] != SCHEMA or value['repo_id'] != REPO
            or value['repo_type'] != 'dataset' or value['private'] is not False
            or {r['path'] for r in value['files']} != NAMES
            or files(value['root']) != value['files']):
        raise PublicationError('Reviewed dataset identity or bytes changed.')
    return value


def publish(plan, expected_digest, cache):
    value = checked(plan, expected_digest)
    payloads = []
    for row in value['files']:
        raw = read(Path(value['root'])/row['path'])
        if digest(raw) != {key: row[key] for key in ('bytes', 'sha256')}:
            raise PublicationError('Payload changed while freezing reviewed publication bytes.')
        payloads.append((row['path'], raw))
    api, token = client(cache)
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = api.repo_info(REPO, repo_type='dataset', files_metadata=True)
    except RepositoryNotFoundError:
        api.create_repo(REPO, repo_type='dataset', private=False, exist_ok=False)
        info = api.repo_info(REPO, repo_type='dataset', files_metadata=True)
    if info.private is not False or not info.sha or not info.siblings:
        raise PublicationError('Expected an initialized public dataset.')
    existing = {item.rfilename for item in info.siblings}
    if existing != {'.gitattributes'}:
        raise PublicationError('Dataset is not empty. Verify the existing publication; do not overwrite it.')
    # Use the exact immutable bytes verified before credential/remote access.
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=raw)
                  for name, raw in payloads]
    commit = api.create_commit(REPO, repo_type='dataset', operations=operations,
                              parent_commit=info.sha,
                              commit_message='Publish three-arm native interface benchmark and installation evidence')
    del token, api
    return {'repo_id': REPO, 'commit': commit.oid, 'files_added': len(operations)}


def verify(plan, expected_digest, cache):
    value = checked(plan, expected_digest)
    api = public_client()
    info = api.repo_info(REPO, repo_type='dataset', files_metadata=True, token=False)
    if info.private is not False or {r.rfilename for r in info.siblings} != NAMES | {'.gitattributes'}:
        raise PublicationError('Public dataset visibility or roster differs.')
    for row in value['files']:
        raw = download(REPO, row['path'], 'dataset', info.sha, cache)
        if digest(raw) != {key: row[key] for key in ('bytes', 'sha256')}:
            raise PublicationError('Anonymous public bytes differ from the reviewed artifacts.')
    return {'repo_id': REPO, 'commit': info.sha, 'anonymous_files_verified': len(value['files'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    s = commands.add_parser('stage'); s.add_argument('root'); s.add_argument('--plan', required=True)
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
