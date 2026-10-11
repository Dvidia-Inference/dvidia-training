"""Publish only the reviewed, public DVIDIA Observation Lab static Space.

Credentials use the existing in-memory Keychain adapter. No footage discovery,
weight upload, token cache, paid compute request or unrelated repository update.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

from prepare import safe_relative, receipt
from publish import PublicationError, checked_remote, client, verified_downloads

REPO_ID = 'Dvidia/observation-lab'
ALLOWED_SUFFIXES = {'.md', '.html', '.js', '.json', '.svg', '.png', '.mp4', '.txt', '.py'}


def checked_plan(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 1024 * 1024:
        raise PublicationError('Expected a bounded plain Observation Lab plan.')
    plan = json.loads(path.read_text())
    if (plan.get('kind') != 'dvidia.observation-lab-publication'
            or plan.get('schema_version') != 1 or plan.get('namespace') != 'Dvidia'
            or plan.get('private') is not False):
        raise PublicationError('Only the public Dvidia Observation Lab plan is accepted.')
    row = plan.get('repository', {})
    expected = {'repo_id': REPO_ID, 'repo_type': 'space', 'space_sdk': 'static', 'directory': 'space'}
    if any(row.get(k) != v for k, v in expected.items()):
        raise PublicationError('Only the named static Observation Lab Space may be published.')
    base = path.parent / 'space'
    if base.is_symlink() or not base.is_dir():
        raise PublicationError('Expected a plain staged Space directory.')
    files = row.get('files', [])
    if (not isinstance(files, list) or not 1 <= len(files) <= 60
            or any(not isinstance(f, dict) for f in files)
            or len({f.get('path') for f in files}) != len(files)):
        raise PublicationError('Invalid Observation Lab file inventory.')
    actual = set()
    for current in base.rglob('*'):
        if current.is_symlink():
            raise PublicationError('Publication files may not contain symlinks.')
        if current.is_file():
            actual.add(current.relative_to(base).as_posix())
    if actual != {f.get('path') for f in files} or not {'README.md', 'index.html', 'LICENSE'} <= actual:
        raise PublicationError('Staged files differ from the complete reviewed inventory.')
    total = 0
    for item in files:
        relative = safe_relative(item['path'])
        if (any(p.startswith('.') for p in relative.parts)
                or (relative.suffix not in ALLOWED_SUFFIXES and relative.name != 'LICENSE')):
            raise PublicationError('Unexpected Observation Lab artifact type.')
        source = base / relative
        if source.stat().st_size > 8 * 1024 * 1024:
            raise PublicationError('Space file exceeds its publication cap.')
        if receipt(source) != {'bytes': item.get('bytes'), 'sha256': item.get('sha256')}:
            raise PublicationError('Staged artifact differs from the reviewed plan.')
        raw = source.read_bytes()
        if re.search(rb'hf_[A-Za-z0-9]{30,}', raw) or b'/Users/roguedev/' in raw:
            raise PublicationError('Possible credential or private local path in staged files.')
        total += len(raw)
    if total > 20 * 1024 * 1024:
        raise PublicationError('Space exceeds its total publication cap.')
    readme = (base / 'README.md').read_text()
    if not re.search(r'^sdk:\s*static\s*$', readme, re.M) or not re.search(r'^app_file:\s*index.html\s*$', readme, re.M):
        raise PublicationError('Space card must declare the reviewed static entry point.')
    return plan


def publish_observation(path):
    path = Path(path).absolute()
    plan = checked_plan(path)
    row = plan['repository']
    # Freeze the reviewed bytes before credentials or network operations.
    payload = {}
    for item in row['files']:
        raw = (path.parent / 'space' / safe_relative(item['path'])).read_bytes()
        if {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()} != {'bytes': item['bytes'], 'sha256': item['sha256']}:
            raise PublicationError('Publication changed before freezing.')
        payload[item['path']] = raw
    api, token = client(path.parent / 'publisher-cache')
    if any(token.encode() in raw for raw in payload.values()):
        raise PublicationError('Credential found in frozen upload bytes.')
    identity = api.whoami()
    if 'Dvidia' not in {o.get('name') for o in identity.get('orgs', [])}:
        raise PublicationError('The publishing account is not in Dvidia.')
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = api.repo_info(REPO_ID, repo_type='space')
    except RepositoryNotFoundError:
        info = None
    if info is not None:
        checked_remote(info, 'space')
        existing = {x.rfilename for x in info.siblings} - {'.gitattributes'}
        if existing - set(payload):
            raise PublicationError('Destination has unrelated files; refusing to modify it.')
        verified_downloads({**row, 'files': [f for f in row['files'] if f['path'] in existing]}, info.sha, False)
    else:
        api.create_repo(REPO_ID, repo_type='space', private=False, space_sdk='static', exist_ok=False)
        info = api.repo_info(REPO_ID, repo_type='space')
        checked_remote(info, 'space')
        existing = set()
    if existing == set(payload):
        revision = info.sha
    else:
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=raw) for name, raw in payload.items()]
        result = api.create_commit(REPO_ID, repo_type='space', operations=operations, parent_commit=info.sha,
                                   commit_message='Publish DVIDIA Observation Lab review demo and local installation guide')
        revision = result.oid
    verified_downloads(row, revision, False)
    result = {'repo_id': REPO_ID, 'repo_type': 'space', 'commit': revision,
              'url': 'https://huggingface.co/spaces/' + REPO_ID,
              'verified_files': len(row['files']), 'scope': 'synthetic review replay; local inference installed separately'}
    target = path.parent / 'publication-receipt.json'
    if target.is_symlink():
        raise PublicationError('Receipt may not be a symlink.')
    target.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'publish'))
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    try:
        result = checked_plan(args.plan) if args.command == 'check' else publish_observation(args.plan)
        print(json.dumps({'checked_repository': result['repository']['repo_id']} if args.command == 'check' else result))
    except Exception as exc:
        if isinstance(exc, PublicationError):
            print(str(exc), file=sys.stderr)
        else:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            print(f'Publication stopped: {type(exc).__name__}; HTTP status {code}. Raw error suppressed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
