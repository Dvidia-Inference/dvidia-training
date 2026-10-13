"""Move exactly three reviewed workstation assets to regular Hub Git blobs.

The explicit NDJSON `file` operations use the Hub commit API's regular-file
format, rather than the SDK's automatic LFS/Xet selection. No runtime bytes,
other repository entries, remote URL, token argument or Git credential cache
may be changed by this one-parent hosting migration.
"""
from __future__ import annotations

import argparse
import base64
from hashlib import sha1, sha256
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.request

import prepare_workstation as original
import publish_robot_sim as publisher
from publish import PublicationError, checked_remote, client

KIND = 'dvidia.workstation-regular-git-hosting/v1'
PARENT = '826df278c6904a755cd4e1aaab7d233d1b843fbc'
REPO = publisher.REPO_ID
ASSETS = ('workstation/v01/geometry.bin.gz', 'workstation/v01/mujoco.wasm.gz',
          'workstation/v01/native-assets.bin.gz')
MAX_ASSET_BYTES = 8*1024*1024
MAX_ATTRIBUTES_BYTES = 64*1024
MAX_PLAN_BYTES = 1024*1024
OVERRIDES = ''.join(path+' -filter -diff -merge\n' for path in ASSETS).encode()


def git_blob(raw):
    return sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def attributes_after(raw):
    if len(raw) > MAX_ATTRIBUTES_BYTES or b'\0' in raw:
        raise PublicationError('Expected bounded UTF-8 Git attributes.')
    try:
        raw.decode('utf-8')
    except UnicodeError as exc:
        raise PublicationError('Expected bounded UTF-8 Git attributes.') from exc
    if any(path.encode() in raw for path in ASSETS):
        raise PublicationError('Explicit workstation attributes already exist; review a new migration.')
    return raw+(b'' if not raw or raw.endswith(b'\n') else b'\n')+OVERRIDES


def public_api():
    from huggingface_hub import HfApi
    return HfApi(endpoint='https://huggingface.co', token=False)


def public_bytes(path, revision, limit):
    if path not in (*ASSETS, '.gitattributes') or re.fullmatch('[a-f0-9]{40}', revision) is None:
        raise PublicationError('Unexpected immutable public download.')
    url = f'https://huggingface.co/spaces/{REPO}/resolve/{revision}/{path}'
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = bytearray()
        while chunk := response.read(min(64*1024, limit+1-len(raw))):
            raw.extend(chunk)
            if len(raw) > limit:
                raise PublicationError('Public hosting asset exceeded its byte bound.')
    return bytes(raw)


def tree(api, revision):
    result = {}
    for item in api.list_repo_tree(REPO, repo_type='space', revision=revision, recursive=True):
        if not hasattr(item, 'blob_id'):
            continue
        if len(result) >= publisher.MAX_FILES+1 or item.path in result:
            raise PublicationError('Remote hosting inventory is unbounded or duplicated.')
        publisher._artifact_name(item.path) if item.path != '.gitattributes' else None
        lfs = getattr(item, 'lfs', None)
        result[item.path] = {'size': item.size, 'blob_id': item.blob_id,
            'lfs': None if lfs is None else {'sha256': lfs.sha256, 'size': lfs.size,
                                            'pointer_size': lfs.pointer_size},
            'xet_hash': getattr(item, 'xet_hash', None)}
    return result


def regular_record(raw):
    return {'size': len(raw), 'blob_id': git_blob(raw), 'lfs': None, 'xet_hash': None}


def matches_bytes(metadata, raw):
    if metadata['size'] != len(raw):
        return False
    if metadata['lfs'] is not None:
        return (metadata['lfs']['size'] == len(raw)
                and metadata['lfs']['sha256'] == sha256(raw).hexdigest())
    return metadata['blob_id'] == git_blob(raw)


def prepare(source_plan, output):
    source_plan, output = map(lambda p: Path(p).absolute(), (source_plan, output))
    publisher._plain(output)
    if output.exists() or source_plan.parent in output.parents:
        raise PublicationError('Hosting migration output must be fresh and outside the source release.')
    source = publisher.checked_plan(source_plan)
    plan_raw = publisher._bytes(source_plan, MAX_PLAN_BYTES)
    prior_receipt = json.loads(publisher._bytes(source_plan.parent/'publication-receipt.json', MAX_PLAN_BYTES))
    if (prior_receipt.get('repo_id') != REPO or prior_receipt.get('repo_type') != 'space'
            or prior_receipt.get('commit') != PARENT
            or prior_receipt.get('plan_sha256') != sha256(plan_raw).hexdigest()):
        raise PublicationError('Hosting migration requires the exact verified download-fix publication.')
    files = {item['path']: item for item in source['repository']['files']}
    frozen = {}
    for path in ASSETS:
        raw = publisher._bytes(source_plan.parent/'space'/path, MAX_ASSET_BYTES)
        if original.record(raw) != {key: files[path][key] for key in ('bytes', 'sha256')}:
            raise PublicationError('Hosting migration source changed after validation.')
        frozen[path] = raw
    api = public_api()
    info = api.repo_info(REPO, repo_type='space')
    checked_remote(info, 'space')
    if info.sha != PARENT:
        raise PublicationError('Hosting migration remote parent changed.')
    before = tree(api, PARENT)
    if set(before) != set(files)|{'.gitattributes'}:
        raise PublicationError('Remote inventory differs from the verified source publication.')
    for path, raw in frozen.items():
        if not matches_bytes(before[path], raw):
            raise PublicationError('Remote compressed asset differs from the verified source bytes.')
    prior_attributes = public_bytes('.gitattributes', PARENT, MAX_ATTRIBUTES_BYTES)
    if before['.gitattributes'] != regular_record(prior_attributes):
        raise PublicationError('Remote Git attributes do not match their immutable Git blob.')
    frozen['.gitattributes'] = attributes_after(prior_attributes)
    for raw in frozen.values():
        publisher._no_private_bytes(raw)
    plan = {'kind': KIND, 'repo_id': REPO, 'repo_type': 'space', 'private': False,
            'expected_remote_commit': PARENT, 'source_commit': source['source_commit'],
            'source_publication_plan_sha256': sha256(plan_raw).hexdigest(),
            'source_publication_receipt': original.record(original.encoded(prior_receipt)),
            'prior_attributes': original.record(prior_attributes), 'prior_tree': before,
            'files': {path: original.record(raw) for path, raw in sorted(frozen.items())}}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.workstation-static-', dir=output.parent))
    try:
        for path, raw in frozen.items():
            target = staging/'payload'/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        (staging/'original-gitattributes.txt').write_bytes(prior_attributes)
        (staging/'hosting-plan.json').write_bytes(original.encoded(plan))
        checked_plan(staging/'hosting-plan.json')
        if output.exists() or output.is_symlink():
            raise PublicationError('Hosting migration output must remain fresh.')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'plan': str(output/'hosting-plan.json'), 'files': sorted(frozen),
            'payload_bytes': sum(map(len, frozen.values())), 'expected_remote_commit': PARENT}


def checked_plan(path):
    path = Path(path).absolute()
    plan = json.loads(publisher._bytes(path, MAX_PLAN_BYTES))
    if (plan.get('kind') != KIND or plan.get('repo_id') != REPO or plan.get('repo_type') != 'space'
            or plan.get('private') is not False or plan.get('expected_remote_commit') != PARENT
            or re.fullmatch('[a-f0-9]{40}', plan.get('source_commit', '')) is None
            or set(plan.get('files', {})) != set(ASSETS)|{'.gitattributes'}):
        raise PublicationError('Unexpected hosting migration plan or destination.')
    inventory = set()
    for current in (path.parent/'payload').rglob('*'):
        publisher._plain(current)
        if current.is_file():
            inventory.add(current.relative_to(path.parent/'payload').as_posix())
    if inventory != set(plan['files']):
        raise PublicationError('Hosting migration payload inventory changed.')
    frozen = {}
    for name, expected in plan['files'].items():
        raw = publisher._bytes(path.parent/'payload'/name,
                               MAX_ATTRIBUTES_BYTES+len(OVERRIDES)+1 if name == '.gitattributes' else MAX_ASSET_BYTES)
        if expected != original.record(raw):
            raise PublicationError('Hosting migration payload bytes changed.')
        publisher._no_private_bytes(raw)
        frozen[name] = raw
    prior_attributes = publisher._bytes(path.parent/'original-gitattributes.txt', MAX_ATTRIBUTES_BYTES)
    if (plan['prior_attributes'] != original.record(prior_attributes)
            or frozen['.gitattributes'] != attributes_after(prior_attributes)):
        raise PublicationError('Git attributes must retain all prior bytes and add only the three explicit overrides.')
    before = plan.get('prior_tree')
    if type(before) is not dict or not set(frozen) <= set(before) or len(before) > publisher.MAX_FILES+1:
        raise PublicationError('Invalid prior hosting tree.')
    if before['.gitattributes'] != regular_record(prior_attributes):
        raise PublicationError('Prior Git attributes metadata changed.')
    for name in ASSETS:
        if not matches_bytes(before[name], frozen[name]):
            raise PublicationError('Frozen assets differ from their prior public content hashes.')
    return plan, frozen


def send_regular_commit(payload, token):
    # This is the official Hub commit wire format used by huggingface_hub's
    # _prepare_commit_payload for regular files. Explicit file operations avoid
    # preupload choosing LFS/Xet again for these already-reviewed small assets.
    if set(payload) != set(ASSETS)|{'.gitattributes'} or any(
            len(raw) > (MAX_ATTRIBUTES_BYTES+len(OVERRIDES)+1 if path == '.gitattributes' else MAX_ASSET_BYTES)
            for path, raw in payload.items()):
        raise PublicationError('Only the four bounded reviewed hosting files may be committed.')
    operations = [{'key': 'header', 'value': {
        'summary': 'Serve live workstation compressed assets as regular Git files',
        'description': 'Hosting-only migration; runtime asset bytes and all other evidence unchanged.',
        'parentCommit': PARENT}}]
    operations.extend({'key': 'file', 'value': {'path': path, 'encoding': 'base64',
                      'content': base64.b64encode(raw).decode('ascii')}} for path, raw in sorted(payload.items()))
    content = b''.join(json.dumps(item, separators=(',', ':')).encode()+b'\n' for item in operations)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            return None
    request = urllib.request.Request(f'https://huggingface.co/api/spaces/{REPO}/commit/main',
        headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/x-ndjson'},
        data=content, method='POST')
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=120) as response:
        response_bytes = response.read(MAX_ATTRIBUTES_BYTES+1)
    if len(response_bytes) > MAX_ATTRIBUTES_BYTES:
        raise PublicationError('Hub returned an oversized hosting commit response.')
    result = json.loads(response_bytes)
    revision = result.get('commitOid')
    if (result.get('success') is not True or type(revision) is not str
            or re.fullmatch('[a-f0-9]{40}', revision) is None):
        raise PublicationError('Hub did not return an immutable regular-file commit.')
    return revision


def verify(api, plan, payload, revision):
    after = tree(api, revision)
    expected = {**plan['prior_tree'], **{name: regular_record(raw) for name, raw in payload.items()}}
    if after != expected:
        raise PublicationError('Published tree did not preserve all other entries and regular Git asset storage.')
    for name, raw in payload.items():
        if public_bytes(name, revision, len(raw)) != raw:
            raise PublicationError('Published public download differs from original frozen bytes.')
    return {'repo_id': REPO, 'repo_type': 'space', 'commit': revision,
            'expected_remote_commit': PARENT, 'verified_regular_git_files': sorted(payload),
            'unchanged_other_entries': len(after)-len(payload), 'runtime_asset_bytes_unchanged': True,
            'source_commit': plan['source_commit']}


def publish(path):
    path = Path(path).absolute()
    plan, payload = checked_plan(path)
    plan_digest = sha256(publisher._bytes(path, MAX_PLAN_BYTES)).hexdigest()
    pending = path.parent/'hosting-publication-pending.json'
    receipt = path.parent/'hosting-publication-receipt.json'
    for target in (pending, receipt):
        publisher._plain(target)
        if target.exists() and not target.is_file():
            raise PublicationError('Hosting migration receipts must be plain files.')
    api, token = client(path.parent/'publisher-cache')
    if any(token.encode() in raw for raw in payload.values()):
        raise PublicationError('Credential found in hosting migration payload.')
    identity = api.whoami()
    if 'Dvidia' not in {org.get('name') for org in identity.get('orgs', [])}:
        raise PublicationError('Publishing account is not a member of Dvidia.')
    info = api.repo_info(REPO, repo_type='space')
    checked_remote(info, 'space')
    if pending.exists():
        prior = json.loads(publisher._bytes(pending, MAX_PLAN_BYTES))
        if prior.get('plan_sha256') != plan_digest or prior.get('commit') != info.sha:
            raise PublicationError('Pending hosting publication differs from the current plan or remote head.')
        revision = prior['commit']
    else:
        if info.sha != PARENT or tree(api, PARENT) != plan['prior_tree']:
            raise PublicationError('Hosting migration remote parent or tree changed before publishing.')
        revision = send_regular_commit(payload, token)
        pending.write_bytes(original.encoded({'plan_sha256': plan_digest, 'commit': revision}))
    result = {**verify(api, plan, payload, revision), 'plan_sha256': plan_digest}
    receipt.write_bytes(original.encoded(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('source_plan', type=Path)
    prepare_parser.add_argument('output', type=Path)
    for command in ('check', 'publish'):
        commands.add_parser(command).add_argument('plan', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.source_plan, args.output)
        elif args.command == 'check':
            plan, payload = checked_plan(args.plan)
            result = {'repo_id': REPO, 'files': sorted(payload), 'expected_remote_commit': plan['expected_remote_commit']}
        else:
            result = publish(args.plan)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        if isinstance(exc, PublicationError):
            print(str(exc), file=sys.stderr)
        else:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            print(f'Hosting publication stopped: {type(exc).__name__}; HTTP status {code}. Raw error suppressed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
