"""Publish a reviewed hash-bound plan using a token held only in process memory.

The macOS credential is read from one named Keychain entry. There is no token
argument, login/cache write, environment token injection, or raw API error output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from prepare import REPOS, receipt, safe_relative


class PublicationError(ValueError):
    """A fixed publisher diagnostic safe to display without API response details."""


def checked_plan(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 1024 * 1024:
        raise PublicationError('Expected a bounded plain publication plan.')
    plan = json.loads(path.read_text())
    org = plan.get('namespace', '')
    if (plan.get('kind') != 'dvidia.huggingface-publication' or plan.get('schema_version') != 1
            or plan.get('private') is not False
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,95}', org)):
        raise PublicationError('Expected a reviewed public publication plan.')
    rows = plan.get('repositories', [])
    if len(rows) != 3 or {r.get('repo_type') for r in rows} != set(REPOS):
        raise PublicationError('Only the three named pilot repositories may be published.')
    total = 0
    for row in rows:
        kind = row['repo_type']
        if row.get('repo_id') != org + '/' + REPOS[kind] or row.get('directory') != kind:
            raise PublicationError('Repository destination differs from the approved pilot layout.')
        if kind == 'space' and row.get('space_sdk') != 'static':
            raise PublicationError('Only a static Space is authorized by this plan.')
        base = path.parent / kind
        if base.is_symlink():
            raise PublicationError('Publication roots may not be symlinks.')
        files = row.get('files', [])
        if not 1 <= len(files) <= 200 or len({f['path'] for f in files}) != len(files):
            raise PublicationError('Invalid publication inventory.')
        inventory = set()
        for current in base.rglob('*'):
            if current.is_symlink():
                raise PublicationError('Publication trees may not contain symlinks.')
            if current.is_file():
                inventory.add(current.relative_to(base).as_posix())
        if inventory != {f['path'] for f in files}:
            raise PublicationError('Publication files changed after staging.')
        for item in files:
            relative = safe_relative(item['path'])
            if any(p.startswith('.') for p in relative.parts):
                raise PublicationError('Hidden files are never published.')
            if relative.suffix not in {'.md', '.html', '.json', '.png', '.mp4', '.zip', '.npz', '.py'} and relative.name != 'LICENSE':
                raise PublicationError('Unexpected public artifact type.')
            source = base / relative
            if not source.is_file() or source.stat().st_size > 64 * 1024 * 1024:
                raise PublicationError('Missing or oversized public artifact.')
            if receipt(source) != {'bytes': item['bytes'], 'sha256': item['sha256']}:
                raise PublicationError('Public artifact differs from the reviewed plan.')
            raw = source.read_bytes()
            if re.search(rb'hf_[A-Za-z0-9]{30,}', raw):
                raise PublicationError('Possible credential found in a publication artifact.')
            total += len(raw)
    if total > 100 * 1024 * 1024:
        raise PublicationError('Pilot publication exceeds its total size limit.')
    return plan


def keychain_token():
    if sys.platform != 'darwin':
        raise PublicationError('This maintainer publisher uses macOS Keychain.')
    result = subprocess.run(['/usr/bin/security', 'find-generic-password', '-a', 'dvidia-publisher',
                             '-s', 'dvidia-huggingface', '-w'], capture_output=True, text=True, timeout=30)
    if result.returncode or not result.stdout.strip():
        raise PublicationError('The named Hugging Face Keychain credential is unavailable.')
    return result.stdout.strip()


def checked_remote(info, kind):
    if (getattr(info, 'private', None) is not False
            or getattr(info, 'siblings', None) is None
            or not re.fullmatch(r'(?:[a-f0-9]{40}|[a-f0-9]{64})', getattr(info, 'sha', '') or '')):
        raise PublicationError('Repository visibility, inventory and commit must be explicit before publishing.')
    if kind == 'space' and getattr(info, 'sdk', None) != 'static':
        raise PublicationError('Existing Space does not have the planned static SDK.')


def client(cache):
    # Configure before importing the SDK; never let debug curl logs expose auth.
    cache = Path(cache).absolute()
    if any(p.is_symlink() for p in (cache, *cache.parents)) or (cache.exists() and not cache.is_dir()):
        raise PublicationError('Publisher cache must be a plain local directory.')
    cache.mkdir(parents=True, exist_ok=True)
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_CACHE'):
        os.environ.pop(key, None)
    os.environ.update(HF_DEBUG='0', HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                      HF_HUB_DISABLE_UPDATE_CHECK='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      HF_ENDPOINT='https://huggingface.co', HF_HOME=str(cache),
                      HF_HUB_CACHE=str(cache / 'hub'), HF_ASSETS_CACHE=str(cache / 'assets'),
                      HF_XET_CACHE=str(cache / 'xet'), HF_TOKEN_PATH=str(cache / 'unused-token'))
    logging.disable(logging.CRITICAL)
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    token = keychain_token()
    return HfApi(endpoint='https://huggingface.co', token=token), token


def verified_downloads(row, revision, token):
    """Read every selected public file at a fixed commit with bounded concurrency."""
    from huggingface_hub import hf_hub_download
    def verify(item):
        remote = Path(hf_hub_download(row['repo_id'], item['path'], repo_type=row['repo_type'],
                      revision=revision, token=token, endpoint='https://huggingface.co'))
        if receipt(remote) != {'bytes': item['bytes'], 'sha256': item['sha256']}:
            raise PublicationError('Published artifact differs from the reviewed bytes.')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, row['files']))


def publish(path, only=None):
    path = Path(path).absolute()
    plan = checked_plan(path)
    selected = set(only or REPOS)
    if not selected <= set(REPOS):
        raise PublicationError('Choose only the planned dataset, model or Space.')
    repositories = [row for row in plan['repositories'] if row['repo_type'] in selected]
    receipt_path = path.parent / 'publication-receipt.json'
    if receipt_path.is_symlink() or (receipt_path.exists() and not receipt_path.is_file()):
        raise PublicationError('Publication receipt must be a plain local file.')
    api, token = client(path.parent / 'publisher-cache')
    # Freeze the reviewed payload before remote calls. Upload these exact bytes,
    # even if a local file changes while the API preflight is in progress.
    payload = {}
    for row in repositories:
        for item in row['files']:
            raw = (path.parent / row['directory'] / safe_relative(item['path'])).read_bytes()
            if {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()} != {'bytes': item['bytes'], 'sha256': item['sha256']}:
                raise PublicationError('Public artifact changed before freezing the upload.')
            if token.encode() in raw or re.search(rb'hf_[A-Za-z0-9]{30,}', raw):
                raise PublicationError('Possible credential found in an upload artifact.')
            payload[row['repo_type'], item['path']] = raw
    identity = api.whoami()
    if plan['namespace'] not in {o.get('name') for o in identity.get('orgs', [])}:
        raise PublicationError('Authenticated account is not a member of the planned organization.')
    # Check every destination before creating or modifying any repository.
    from huggingface_hub.errors import RepositoryNotFoundError
    from huggingface_hub import CommitOperationAdd
    destinations = []
    for row in repositories:
        try:
            info = api.repo_info(row['repo_id'], repo_type=row['repo_type'])
        except RepositoryNotFoundError:
            info = None
        if info is not None:
            checked_remote(info, row['repo_type'])
            existing = {x.rfilename for x in info.siblings}
            expected = {f['path'] for f in row['files']}
            if existing - expected - {'.gitattributes'}:
                raise PublicationError('Destination contains unrelated existing files.')
            matching = {**row, 'files': [item for item in row['files'] if item['path'] in existing]}
            print(json.dumps({'phase': 'checking-existing-public-files', 'repo_id': row['repo_id']}), flush=True)
            verified_downloads(matching, info.sha, False)
        destinations.append((row, info))
    results = []
    for row, info in destinations:
        print(json.dumps({'phase': 'publishing', 'repo_id': row['repo_id']}), flush=True)
        if info is None:
            api.create_repo(row['repo_id'], repo_type=row['repo_type'], private=False,
                            space_sdk='static' if row['repo_type'] == 'space' else None, exist_ok=False)
            info = api.repo_info(row['repo_id'], repo_type=row['repo_type'])
            checked_remote(info, row['repo_type'])
        operations = [CommitOperationAdd(path_in_repo=f['path'],
                      path_or_fileobj=payload[row['repo_type'], f['path']]) for f in row['files']]
        commit = api.create_commit(row['repo_id'], repo_type=row['repo_type'], operations=operations,
                                  parent_commit=info.sha, commit_message='Publish synthetic DVIDIA training pilot and evidence')
        # Verify every published byte at the exact commit, using unauthenticated reads.
        print(json.dumps({'phase': 'verifying-public-downloads', 'repo_id': row['repo_id']}), flush=True)
        verified_downloads(row, commit.oid, False)
        result = {'repo_id': row['repo_id'], 'repo_type': row['repo_type'], 'commit': commit.oid,
                  'url': 'https://huggingface.co/' + ({'dataset': 'datasets/', 'space': 'spaces/', 'model': ''}[row['repo_type']]) + row['repo_id'],
                  'verified_files': len(row['files'])}
        results.append(result)
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.publication-receipt-', delete=False) as staged:
            json.dump({'repositories': results}, staged, indent=2)
            staged.write('\n')
        os.replace(staged.name, receipt_path)
        print(json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('identity')
    for name in ('check', 'publish'):
        sub = commands.add_parser(name)
        sub.add_argument('plan', type=Path)
        if name == 'publish':
            sub.add_argument('--only', nargs='+', choices=tuple(REPOS), help='Publish selected reviewed artifacts independently.')
    args = parser.parse_args()
    try:
        if args.command == 'identity':
            api, _ = client(Path.cwd() / 'runs' / 'huggingface-tools' / 'cache')
            data = api.whoami()
            print(json.dumps({'user': data.get('name'), 'organizations': [o.get('name') for o in data.get('orgs', [])]}))
        elif args.command == 'check':
            plan = checked_plan(args.plan)
            print(json.dumps({'checked_repositories': [r['repo_id'] for r in plan['repositories']]}))
        else:
            publish(args.plan, only=args.only)
    except KeyboardInterrupt:
        print('Publishing interrupted; completed public commits remain available for the same-plan retry.', file=sys.stderr)
        return 130
    except Exception as exc:
        # API responses, request headers and tracebacks may contain credentials.
        if isinstance(exc, PublicationError):
            print(str(exc), file=sys.stderr)
        else:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            print(f'Publishing stopped: {type(exc).__name__}; HTTP status {code}. Raw error suppressed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
