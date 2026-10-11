"""Publish only the reviewed public DVIDIA Robot Simulation Lab static Space.

The complete bounded payload is frozen before the existing in-memory Keychain
adapter is used. No token argument/cache, neural weights, compute deployment,
unrelated repository update or implicit overwrite is supported.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

from prepare import safe_relative
from publish import PublicationError, checked_remote, client, verified_downloads

REPO_ID = 'Dvidia/robot-simulation-lab'
KIND = 'dvidia.robot-simulation-lab-publication'
MAX_FILE_BYTES = 32*1024*1024
MAX_TOTAL_BYTES = 150*1024*1024
ALLOWED_SUFFIXES = {'.md', '.html', '.js', '.json', '.txt', '.png', '.bin', '.gz'}
SHA = re.compile(r'(?:[a-f0-9]{40}|[a-f0-9]{64})')
MAX_STARTER_STYLESHEET_BYTES = 64*1024


def _plain(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise PublicationError('Publication paths may not contain symlinks.')


def _bytes(path, limit):
    _plain(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise PublicationError('Missing or oversized plain publication file.')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise PublicationError('Publication file grew beyond its byte bound.')
    return raw


def _artifact_name(name):
    try:
        relative = safe_relative(name)
    except ValueError as exc:
        raise PublicationError('Invalid Robot Simulation Lab artifact path.') from exc
    if (any(p.startswith('.') for p in relative.parts)
            or (relative.suffix not in ALLOWED_SUFFIXES and relative.name != 'LICENSE')):
        raise PublicationError('Unexpected Robot Simulation Lab artifact type.')
    if relative.suffix in ('.bin', '.gz') and relative.name not in (
            'geometry.bin', 'motion.bin', 'geometry.bin.gz', 'motion.bin.gz'):
        raise PublicationError('Only native geometry and recorded motion binaries may be published.')
    return relative


def _no_private_bytes(raw):
    if re.search(rb'hf_[A-Za-z0-9]{30,}', raw) or b'/Users/roguedev/' in raw:
        raise PublicationError('Possible credential or private local path in publication bytes.')


def _reviewed_removals(plan):
    """Only an explicitly reviewed HF static starter stylesheet can be removed."""
    removals = plan.get('remove_files',[])
    if type(removals) is not list or len(removals)>1:
        raise PublicationError('Only one reviewed static starter stylesheet removal is supported.')
    for item in removals:
        if (type(item) is not dict or set(item)!={'path','bytes','sha256'}
                or item['path']!='style.css' or type(item['bytes']) is not int
                or not 0<=item['bytes']<=MAX_STARTER_STYLESHEET_BYTES
                or type(item['sha256']) is not str
                or re.fullmatch(r'[a-f0-9]{64}',item['sha256']) is None):
            raise PublicationError('Starter stylesheet removal needs its exact bounded bytes and SHA-256.')
    if removals and plan.get('expected_remote_commit') is None:
        raise PublicationError('Starter stylesheet removal requires an exact reviewed remote parent commit.')
    return removals


def _checked_plan_bytes(path):
    path = Path(path).absolute()
    plan_bytes = _bytes(path, 1024*1024)
    plan = json.loads(plan_bytes)
    if (type(plan) is not dict or plan.get('kind') != KIND or plan.get('schema_version') != 1
            or plan.get('namespace') != 'Dvidia' or plan.get('private') is not False):
        raise PublicationError('Only the public Dvidia Robot Simulation Lab plan is accepted.')
    parent = plan.get('expected_remote_commit')
    if parent is not None and (type(parent) is not str or SHA.fullmatch(parent) is None):
        raise PublicationError('A reviewed update must name its exact remote parent commit.')
    _reviewed_removals(plan)
    row = plan.get('repository')
    expected = {'repo_id': REPO_ID, 'repo_type': 'space', 'space_sdk': 'static', 'directory': 'space'}
    if type(row) is not dict or any(row.get(key) != value for key,value in expected.items()):
        raise PublicationError('Only the named static Robot Simulation Lab Space may be published.')
    base = path.parent/'space'
    _plain(base)
    if not base.is_dir():
        raise PublicationError('Expected a plain staged Robot Simulation Lab directory.')
    files = row.get('files')
    if (type(files) is not list or not 3 <= len(files) <= 600 or any(type(f) is not dict for f in files)
            or any(type(f.get('path')) is not str for f in files)
            or len({f['path'] for f in files}) != len(files)):
        raise PublicationError('Invalid complete Robot Simulation Lab inventory.')
    actual = set()
    for current in base.rglob('*'):
        _plain(current)
        if current.is_file():
            actual.add(current.relative_to(base).as_posix())
    if actual != {f['path'] for f in files} or not {'README.md','index.html','LICENSE','provenance.json'} <= actual:
        raise PublicationError('Staged files differ from the complete reviewed inventory.')
    total = 0
    for item in files:
        source = base/_artifact_name(item['path'])
        raw = _bytes(source, MAX_FILE_BYTES)
        if (type(item.get('bytes')) is not int or item['bytes'] != len(raw)
                or type(item.get('sha256')) is not str or item['sha256'] != sha256(raw).hexdigest()):
            raise PublicationError('Staged artifact differs from the reviewed plan.')
        _no_private_bytes(raw)
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise PublicationError('Robot Simulation Lab exceeds its total publication cap.')
    if type(plan.get('total_bytes')) is not int or plan['total_bytes'] != total:
        raise PublicationError('Publication total bytes disagree with its inventory.')
    lines = _bytes(base/'README.md', MAX_FILE_BYTES).decode('utf-8').splitlines()
    if not lines or lines[0] != '---' or '---' not in lines[1:101]:
        raise PublicationError('Space card needs bounded opening YAML frontmatter.')
    header = lines[1:lines.index('---',1)]
    fields = {}
    previous_key = None
    for line in header:
        if not line.strip():
            continue
        if previous_key=='tags' and re.fullmatch(r'  - [A-Za-z0-9_-]+',line):
            continue
        match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_-]*):\s*(.*)',line)
        if match is None:
            raise PublicationError('Space card uses unsupported frontmatter syntax.')
        key,value = match.groups()
        if key in fields:
            raise PublicationError('Space card frontmatter has duplicate keys.')
        fields[key] = value
        previous_key = key
    if any(fields.get(key) != value for key,value in {'sdk':'static','app_file':'index.html','license':'mit'}.items()):
        raise PublicationError('Space card must declare the reviewed static entry point and MIT software license.')
    return plan,plan_bytes


def checked_plan(path):
    return _checked_plan_bytes(path)[0]


def publish_robot_sim(path):
    path = Path(path).absolute()
    plan,plan_bytes = _checked_plan_bytes(path)
    plan_digest = sha256(plan_bytes).hexdigest()
    row = plan['repository']
    removals = _reviewed_removals(plan)
    target = path.parent/'publication-receipt.json'
    _plain(target)
    if target.exists() and not target.is_file():
        raise PublicationError('Publication receipt must be a plain local file.')
    # Take a final immutable snapshot before credentials and all remote calls.
    payload = {}
    for item in row['files']:
        raw = _bytes(path.parent/'space'/_artifact_name(item['path']), MAX_FILE_BYTES)
        if {'bytes':len(raw),'sha256':sha256(raw).hexdigest()} != {'bytes':item['bytes'],'sha256':item['sha256']}:
            raise PublicationError('Publication changed before freezing.')
        _no_private_bytes(raw)
        payload[item['path']] = raw
    api, token = client(path.parent/'publisher-cache')
    if any(token.encode() in raw for raw in payload.values()):
        raise PublicationError('Credential found in frozen upload bytes.')
    identity = api.whoami()
    if 'Dvidia' not in {o.get('name') for o in identity.get('orgs',[])}:
        raise PublicationError('The publishing account is not in Dvidia.')
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = api.repo_info(REPO_ID,repo_type='space')
    except RepositoryNotFoundError:
        info = None
    parent = plan.get('expected_remote_commit')
    if info is None:
        if parent is not None:
            raise PublicationError('Reviewed update destination is missing.')
        api.create_repo(REPO_ID,repo_type='space',private=False,space_sdk='static',exist_ok=False)
        info = api.repo_info(REPO_ID,repo_type='space')
        checked_remote(info,'space')
        existing = set()
    else:
        checked_remote(info,'space')
        existing = {x.rfilename for x in info.siblings}- {'.gitattributes'}
        removal_names = {item['path'] for item in removals}
        if existing-set(payload)-removal_names:
            raise PublicationError('Destination has unrelated files; refusing to modify it.')
        if parent is not None:
            if info.sha != parent:
                raise PublicationError('Remote parent changed; prepare a fresh reviewed plan.')
            if not removal_names<=existing:
                raise PublicationError('Reviewed starter stylesheet is missing at the remote parent.')
            # Verify the exact public bytes at the locked parent before queuing
            # the one permitted deletion alongside the frozen upload.
            if removals:
                verified_downloads({**row,'files':removals},info.sha,False)
        else:
            # Same-plan retries must prove existing files already match. Changed
            # content requires an explicitly reviewed exact-parent update plan.
            verified_downloads({**row,'files':[f for f in row['files'] if f['path'] in existing]},info.sha,False)
    if existing == set(payload) and parent is None:
        revision = info.sha
    else:
        result = api.create_commit(REPO_ID,repo_type='space',parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=raw) for name,raw in payload.items()]
                       +[CommitOperationDelete(path_in_repo=item['path'],is_folder=False) for item in removals],
            commit_message='Publish reviewed DVIDIA native robot simulation evidence and local tools')
        revision = result.oid
    if type(revision) is not str or SHA.fullmatch(revision) is None:
        raise PublicationError('Publisher did not return an explicit immutable revision.')
    verified_downloads(row,revision,False)
    result = {'repo_id':REPO_ID,'repo_type':'space','commit':revision,
              'url':'https://huggingface.co/spaces/'+REPO_ID,'verified_files':len(row['files']),
              'scope':'Recorded native simulation evidence; no live physics, learned policy or physical robot qualification',
              'plan_sha256':plan_digest,'removed_files':removals}
    _plain(target)
    target.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','publish'))
    parser.add_argument('plan',type=Path)
    args = parser.parse_args()
    try:
        result = checked_plan(args.plan) if args.command=='check' else publish_robot_sim(args.plan)
        print(json.dumps({'checked_repository':result['repository']['repo_id']} if args.command=='check' else result))
    except Exception as exc:
        if isinstance(exc,PublicationError):
            print(str(exc),file=sys.stderr)
        else:
            code = getattr(getattr(exc,'response',None),'status_code',None)
            print(f'Publication stopped: {type(exc).__name__}; HTTP status {code}. Raw error suppressed.',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
