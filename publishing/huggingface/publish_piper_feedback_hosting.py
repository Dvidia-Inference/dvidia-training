"""Derive a four-file hosting patch without modifying sealed study evidence.

Stage/check are offline and never access credentials. Stage requires explicit
local Space baseline files/tree and exact reviewed commits. Only publish reads
the named Keychain credential; it performs one parent-bound text-only commit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import publish_piper_feedback as study
from publish import PublicationError, checked_remote, client
from publish_piper_imitation_batches import _matches, _tree


SCHEMA = 'dvidia.piper-feedback-hosting-patch/v1'
BASELINE_SCHEMA = 'dvidia.piper-feedback-space-baseline/v1'
PREFIX = study.SPACE_NAMESPACE+'/viewer/'
MOTION = PREFIX+'motion.json'
SEAL = PREFIX+'viewer-integrity.json'
ALLOWED = {MOTION, SEAL, 'index.html', 'README.md'}
DATASET_ROOT = 'https://huggingface.co/datasets/'+study.DATASET_REPO+'/resolve/'
OLD_NATIVE_ROOT = DATASET_ROOT+'main/'
MAX_PATCH_BYTES = 8*1024*1024
MAX_FILES = 25
SHA = re.compile(r'[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
NATIVE_PATH = re.compile(r'confirmation/confirmation-[0-9]{2}-goal[01]-(?:nominal|jaw_interruption)/(?:feedback|blind|feedforward|teacher|open_jaw)/native\.npz\Z')


def _read(path, limit=study.MAX_FILE):
    return study._read(path, limit)


def _json(path):
    return study._json(_read(path))


def _receipt(raw):
    return study._receipt(raw)


def _commit(value, name):
    if not isinstance(value, str) or COMMIT.fullmatch(value) is None:
        raise PublicationError(name+' must be an exact 40-hex commit.')


def _relative(name):
    return study._relative(name)


def _tree_document(value, expected):
    if (not isinstance(value, dict) or value.get('schema') != BASELINE_SCHEMA
            or value.get('repo_id') != study.SPACE_REPO or value.get('repo_type') != 'space'
            or value.get('commit') != expected or not isinstance(value.get('tree'), dict)
            or not 1 <= len(value['tree']) <= 20000):
        raise PublicationError('Baseline snapshot must identify the exact public Space commit and full tree.')
    for name, row in value['tree'].items():
        if (not isinstance(name, str) or name.startswith('/') or '\\' in name or any(p in ('', '.', '..') for p in name.split('/'))
                or not isinstance(row, dict) or set(row) != {'blob', 'size', 'lfs_sha256'}
                or not isinstance(row['blob'], str) or COMMIT.fullmatch(row['blob']) is None
                or type(row['size']) is not int or row['size'] < 0
                or row['lfs_sha256'] is not None and (not isinstance(row['lfs_sha256'], str) or SHA.fullmatch(row['lfs_sha256']) is None)):
            raise PublicationError('Malformed baseline tree entry.')
    return value['tree']


def _source_viewer(viewer, expected_seal):
    viewer = study._plain(viewer)
    if not isinstance(expected_seal, str) or SHA.fullmatch(expected_seal) is None:
        raise PublicationError('An explicit parent viewer-integrity SHA256 is required.')
    seal_raw = _read(viewer/'viewer-integrity.json')
    if sha256(seal_raw).hexdigest() != expected_seal:
        raise PublicationError('Parent viewer seal differs from the reviewed artifact.')
    seal = study._json(seal_raw)
    inventory = seal.get('files') if isinstance(seal, dict) else None
    if (not isinstance(seal, dict) or seal.get('schema') != study.VIEWER_SCHEMA or not isinstance(inventory, dict)
            or not 1 <= len(inventory) <= 2000):
        raise PublicationError('Expected a sealed feedback viewer projection.')
    paths = {}
    for path in viewer.rglob('*'):
        study._plain(path)
        if path.is_file():
            paths[path.relative_to(viewer).as_posix()] = path
    if set(paths) != set(inventory) | {'viewer-integrity.json'}:
        raise PublicationError('Viewer file inventory differs from its original seal.')
    files = {'viewer-integrity.json': seal_raw}
    for name, expected in inventory.items():
        _relative(name)
        raw = _read(paths[name])
        if _receipt(raw) != expected:
            raise PublicationError('A sealed viewer file changed.')
        files[name] = raw
    return files, seal


def _pin_motion(raw, dataset_commit):
    motion = study._json(raw)
    if (not isinstance(motion, dict) or motion.get('schema') != study.VIEWER_SCHEMA
            or not isinstance(motion.get('cases'), list) or len(motion['cases']) != 28
            or not isinstance(motion.get('all_attempts'), list) or len(motion['all_attempts']) != 120
            or motion.get('native_dataset', {}).get('repo_id') != study.DATASET_REPO
            or motion['native_dataset'].get('revision') != 'main'):
        raise PublicationError('Expected the original unpinned 28-replay/120-attempt metadata.')
    derived = deepcopy(motion)
    native = {}
    for row in derived['all_attempts']+derived['cases']:
        evidence = row.get('evidence')
        href = evidence.get('native.npz') if isinstance(evidence, dict) else None
        if not isinstance(href, str) or not href.startswith(OLD_NATIVE_ROOT):
            raise PublicationError('Native evidence URL differs from the fixed source dataset.')
        path = href.removeprefix(OLD_NATIVE_ROOT)
        digest, size = evidence.get('native_npz_sha256'), evidence.get('native_npz_bytes')
        if (NATIVE_PATH.fullmatch(path) is None or not isinstance(digest, str) or SHA.fullmatch(digest) is None
                or type(size) is not int or not 0 < size <= study.MAX_FILE):
            raise PublicationError('Native evidence path or original hash receipt is invalid.')
        entry = dict(bytes=size, sha256=digest)
        if path in native and native[path] != entry:
            raise PublicationError('A native link has conflicting original receipts.')
        native[path] = entry
        evidence['native.npz'] = DATASET_ROOT+dataset_commit+'/'+path
    if len(native) != 120:
        raise PublicationError('All 120 distinct native evidence files must remain linked.')
    derived['native_dataset']['revision'] = dataset_commit
    # Verification is performed by publish, not claimed during offline staging.
    derived['native_dataset']['publication_readback_verified'] = False
    derived['native_dataset']['note'] = 'Exact dataset commit configured by a derived hosting patch; consult its publication receipt for public-tree checks and anonymous hosting-file readback.'
    return study._encoded(derived), native


def _root_links(index_raw, readme_raw, report):
    index, readme = index_raw.decode('utf8'), readme_raw.decode('utf8')
    target = study.SPACE_NAMESPACE+'/viewer/index.html'
    if target in index or study.VIEWER_URL in readme:
        raise PublicationError('The root already exposes v0.2; inspect it instead of appending duplicate links.')
    variants = report.get('variants', {})
    counts = []
    for variant, name in [('feedback', 'learned jaw feedback'), ('blind', 'learner without live feedback'),
                          ('feedforward', 'frozen primitive'), ('teacher', 'authored teacher'), ('open_jaw', 'open jaw')]:
        row = variants.get(variant)
        if (not isinstance(row, dict) or type(row.get('successes')) is not int or row.get('scheduled') != 24
                or not 0 <= row['successes'] <= 24):
            raise PublicationError('Root links require the original complete study counts.')
        counts.append(f"{name} {row['successes']}/24")
    summary = '; '.join(counts)+'.'
    fragment = ('\n<section aria-labelledby="piper-learned-grip-v02">\n'
        '<h2 id="piper-learned-grip-v02">Testing grip feedback</h2>\n'
        '<p>A learned jaw adapter on a frozen six-axis arm skill. Simulation only.</p>\n'
        '<p>'+summary+'</p>\n'
        '<p><a href="'+target+'">Open PiPER grip feedback v0.2</a></p>\n</section>\n')
    endings = list(re.finditer(r'</main\s*>', index, flags=re.I))
    if len(endings) != 1:
        raise PublicationError('Root index requires one explicit closing main element for the bounded link insertion.')
    point = endings[0].start()
    index = index[:point]+fragment+index[point:]
    readme = readme.rstrip()+'\n\n## PiPER learned grip feedback v0.2\n\n'+summary+'\n\nA learned jaw adapter on a frozen six-axis arm skill. These simulation results do not establish contact recovery or hardware transfer. [Open the full replay and evidence]('+study.VIEWER_URL+').\n'
    return index.encode(), readme.encode()


def stage(viewer, dataset_commit, space_commit, baseline, output, *, expected_viewer_sha256):
    """Offline derivation: baseline holds tree.json, index.html and README.md."""
    _commit(dataset_commit, 'Dataset revision'); _commit(space_commit, 'Space parent')
    viewer, baseline, output = map(study._plain, (viewer, baseline, output))
    if output.exists():
        raise PublicationError('Hosting patch output must be fresh.')
    source, seal = _source_viewer(viewer, expected_viewer_sha256)
    baseline_raw = _read(baseline/'tree.json')
    tree = _tree_document(study._json(baseline_raw), space_commit)
    viewer_tree = {name.removeprefix(PREFIX): entry for name, entry in tree.items() if name.startswith(PREFIX)}
    if set(viewer_tree) != set(source) or any(not _matches(raw, viewer_tree[name]) for name, raw in source.items()):
        raise PublicationError('Pinned Space viewer bytes differ from the original sealed projection.')
    before = {MOTION: source['motion.json'], SEAL: source['viewer-integrity.json']}
    for name in ('index.html', 'README.md'):
        before[name] = _read(baseline/name)
        if name not in tree or not _matches(before[name], tree[name]):
            raise PublicationError('Root baseline bytes differ from the full pinned Space tree.')
    motion_raw, native = _pin_motion(source['motion.json'], dataset_commit)
    derived_seal = deepcopy(seal)
    derived_seal['files']['motion.json'] = _receipt(motion_raw)
    derived_seal['hosting_derivation'] = dict(schema=SCHEMA, parent_viewer_integrity_sha256=expected_viewer_sha256,
        dataset_commit=dataset_commit, changed_viewer_paths=['motion.json'],
        scope='Dataset link pin only; every model, scientific result, trace, native metadata, frame and pose binary remains unchanged.')
    index, readme = _root_links(before['index.html'], before['README.md'], study._json(source['report.json']))
    after = {MOTION: motion_raw, SEAL: study._encoded(derived_seal), 'index.html': index, 'README.md': readme}
    if sum(map(len, after.values())) > MAX_PATCH_BYTES or len(after) > MAX_FILES:
        raise PublicationError('Hosting patch exceeds its single bounded commit.')
    for name, raw in after.items():
        _relative(name); study._scan(raw, json_value=name.endswith('.json'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'baseline-tree.json').write_bytes(baseline_raw)
    (output/'parent-viewer-integrity.json').write_bytes(source['viewer-integrity.json'])
    (output/'parent-report.json').write_bytes(source['report.json'])
    for name, raw in before.items():
        path = output/'baseline-files'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for name, raw in after.items():
        path = output/'payload'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    plan = dict(schema=SCHEMA, repo_id=study.SPACE_REPO, repo_type='space', parent_commit=space_commit,
                dataset_repo=study.DATASET_REPO, dataset_commit=dataset_commit,
                parent_viewer_integrity_sha256=expected_viewer_sha256,
                baseline_tree=_receipt(baseline_raw),
                files=[dict(path=name, before=_receipt(before[name]), after=_receipt(raw)) for name, raw in sorted(after.items())],
                native_files=native, physical_robot_ready=False,
                scope='Four text files only: dataset links pinned in derived viewer metadata/seal and a root Space exposure link. Original sealed viewer and scientific evidence are untouched.')
    (output/'hosting-plan.json').write_bytes(study._encoded(plan))
    return plan


def checked_plan(path):
    path = study._plain(path)
    plan = _json(path)
    if (not isinstance(plan, dict) or plan.get('schema') != SCHEMA or plan.get('repo_id') != study.SPACE_REPO
            or plan.get('repo_type') != 'space' or plan.get('dataset_repo') != study.DATASET_REPO
            or plan.get('physical_robot_ready') is not False):
        raise PublicationError('Only the fixed DVIDIA Space hosting patch is supported.')
    _commit(plan.get('parent_commit'), 'Space parent'); _commit(plan.get('dataset_commit'), 'Dataset revision')
    baseline_raw = _read(path.parent/'baseline-tree.json')
    if _receipt(baseline_raw) != plan.get('baseline_tree'):
        raise PublicationError('The reviewed full baseline tree changed.')
    tree = _tree_document(study._json(baseline_raw), plan['parent_commit'])
    parent_raw = _read(path.parent/'parent-viewer-integrity.json')
    if sha256(parent_raw).hexdigest() != plan.get('parent_viewer_integrity_sha256'):
        raise PublicationError('The parent viewer seal changed.')
    parent = study._json(parent_raw)
    report_raw = _read(path.parent/'parent-report.json')
    if not isinstance(parent, dict) or _receipt(report_raw) != parent.get('files', {}).get('report.json'):
        raise PublicationError('Original report differs from its sealed viewer receipt.')
    rows = plan.get('files')
    if not isinstance(rows, list) or len(rows) != 4 or {row.get('path') for row in rows if isinstance(row, dict)} != ALLOWED:
        raise PublicationError('Only the four reviewed hosting paths may change.')
    total, payload, original = 0, {}, {}
    for row in rows:
        name = row['path']; raw = _read(path.parent/'payload'/_relative(name))
        if _receipt(raw) != row.get('after') or name not in tree:
            raise PublicationError('A reviewed hosting payload changed.')
        before = _read(path.parent/'baseline-files'/_relative(name))
        if _receipt(before) != row.get('before') or not _matches(before, tree[name]):
            raise PublicationError('Original hosting bytes differ from the reviewed baseline tree.')
        original[name] = before
        study._scan(raw, json_value=name.endswith('.json'))
        total += len(raw); payload[name] = raw
    actual = {p.relative_to(path.parent/'payload').as_posix() for p in (path.parent/'payload').rglob('*') if p.is_file()}
    if actual != ALLOWED or total > MAX_PATCH_BYTES:
        raise PublicationError('Hosting payload roster or commit size changed.')
    if original[SEAL] != parent_raw or _receipt(original[MOTION]) != parent['files'].get('motion.json'):
        raise PublicationError('Original viewer metadata differs from the sealed parent.')
    expected_motion, expected_links = _pin_motion(original[MOTION], plan['dataset_commit'])
    if expected_motion != payload[MOTION]:
        raise PublicationError('Derived metadata changed beyond the exact dataset-link pin.')
    expected_index, expected_readme = _root_links(original['index.html'], original['README.md'], study._json(report_raw))
    if payload['index.html'] != expected_index or payload['README.md'] != expected_readme:
        raise PublicationError('Root exposure changed beyond the exact bounded insertion.')
    motion = study._json(payload[MOTION])
    expected_native = plan.get('native_files')
    if not isinstance(expected_native, dict) or expected_native != expected_links or len(expected_native) != 120:
        raise PublicationError('All 120 original native receipts must be retained.')
    for name, entry in expected_native.items():
        if NATIVE_PATH.fullmatch(name) is None or not isinstance(entry, dict) or set(entry) != {'bytes', 'sha256'} or type(entry['bytes']) is not int or not 0 < entry['bytes'] <= study.MAX_FILE or SHA.fullmatch(entry.get('sha256', '')) is None:
            raise PublicationError('Malformed native dataset receipt.')
    # Reversing the pin must reproduce the exact original metadata hash.
    if motion.get('native_dataset', {}).get('revision') != plan['dataset_commit']:
        raise PublicationError('Derived motion metadata does not pin the reviewed dataset commit.')
    for row in motion.get('cases', [])+motion.get('all_attempts', []):
        evidence = row.get('evidence', {})
        href = evidence.get('native.npz', '')
        prefix = DATASET_ROOT+plan['dataset_commit']+'/'
        name = href.removeprefix(prefix) if isinstance(href, str) and href.startswith(prefix) else None
        if name not in expected_native or expected_native[name] != dict(bytes=evidence.get('native_npz_bytes'), sha256=evidence.get('native_npz_sha256')):
            raise PublicationError('A derived native link or original receipt changed.')
    derived = study._json(payload[SEAL])
    expected = deepcopy(parent)
    expected['files']['motion.json'] = _receipt(payload[MOTION])
    expected['hosting_derivation'] = dict(schema=SCHEMA, parent_viewer_integrity_sha256=plan['parent_viewer_integrity_sha256'],
        dataset_commit=plan['dataset_commit'], changed_viewer_paths=['motion.json'],
        scope='Dataset link pin only; every model, scientific result, trace, native metadata, frame and pose binary remains unchanged.')
    if derived != expected:
        raise PublicationError('Derived viewer seal changed beyond the metadata pin.')
    return plan, tree, payload


def _public_download(repo, path, kind, revision):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type=kind, revision=revision, token=False, endpoint='https://huggingface.co')


def publish(path, *, api=None, downloader=None, operation_cls=None):
    path = study._plain(path)
    plan, baseline, payload = checked_plan(path)
    token = None
    if api is None:
        api, token = client(path.parent/'publisher-cache')
    if operation_cls is None:
        from huggingface_hub import CommitOperationAdd
        operation_cls = CommitOperationAdd
    downloader = downloader or _public_download
    if token and any(token.encode() in raw for raw in payload.values()):
        raise PublicationError('Credential appeared in a frozen hosting payload.')
    info = api.repo_info(study.SPACE_REPO, repo_type='space')
    checked_remote(info, 'space')
    if info.sha != plan['parent_commit'] or _tree(api, study.SPACE_REPO, 'space', info.sha) != baseline:
        raise PublicationError('Space head/full tree differs from the exact reviewed baseline.')
    # The public dataset commit must expose each original LFS object identity.
    dataset = api.repo_info(study.DATASET_REPO, repo_type='dataset', revision=plan['dataset_commit'])
    checked_remote(dataset, 'dataset')
    if dataset.sha != plan['dataset_commit']:
        raise PublicationError('Dataset revision did not resolve exactly.')
    dataset_tree = _tree(api, study.DATASET_REPO, 'dataset', dataset.sha)
    for name, entry in plan['native_files'].items():
        remote = dataset_tree.get(name)
        if not remote or remote.get('size') != entry['bytes'] or remote.get('lfs_sha256') != entry['sha256']:
            raise PublicationError('Pinned dataset native LFS object differs from the original evidence receipt.')
    for row in plan['files']:
        old = study._downloaded_bytes(downloader(study.SPACE_REPO, row['path'], 'space', info.sha), study.MAX_FILE)
        if _receipt(old) != row['before'] or not _matches(old, baseline[row['path']]):
            raise PublicationError('Anonymous baseline bytes differ from the reviewed hosting patch.')
    pending = path.parent/'hosting-pending.json'
    if pending.exists():
        raise PublicationError('An earlier hosting commit has an uncertain outcome; inspect it before any retry.')
    study._atomic(pending, dict(schema=SCHEMA+'/pending', plan_sha256=sha256(_read(path)).hexdigest(), parent_commit=info.sha, paths=sorted(payload)))
    operations = []
    for name, raw in sorted(payload.items()):
        operation = operation_cls(path_in_repo=name, path_or_fileobj=raw)
        operation._upload_mode, operation._should_ignore = 'regular', False
        operations.append(operation)
    try:
        result = api.create_commit(study.SPACE_REPO, repo_type='space', parent_commit=info.sha,
            operations=operations, commit_message='Pin PiPER grip feedback evidence and expose the v0.2 study')
    except Exception as exc:
        raise PublicationError('Hosting commit outcome is uncertain; retained the exact pending request. Inspect before retrying.') from exc
    revision = getattr(result, 'oid', None)
    _commit(revision, 'Published Space revision')
    after = _tree(api, study.SPACE_REPO, 'space', revision)
    if set(after) != set(baseline) or any(after[name] != entry for name, entry in baseline.items() if name not in ALLOWED) or any(not _matches(raw, after.get(name, {})) for name, raw in payload.items()):
        raise PublicationError('Published Space changed outside the four allowed hosting paths.')
    for name, raw in payload.items():
        public = study._downloaded_bytes(downloader(study.SPACE_REPO, name, 'space', revision), study.MAX_FILE)
        if public != raw:
            raise PublicationError('Anonymous hosting-file readback differed from the reviewed bytes.')
    receipt = dict(schema=SCHEMA+'/receipt', plan_sha256=sha256(_read(path)).hexdigest(),
        parent_commit=plan['parent_commit'], commit=revision, dataset_commit=plan['dataset_commit'],
        parent_viewer_integrity_sha256=plan['parent_viewer_integrity_sha256'],
        derived_viewer_integrity_sha256=sha256(payload[SEAL]).hexdigest(), changed_paths=sorted(payload),
        unchanged_space_files=len(baseline)-len(payload), dataset_native_lfs_objects_checked=120,
        anonymous_before_and_after_readback=True, viewer_url=study.VIEWER_URL, physical_robot_ready=False)
    study._atomic(path.parent/'hosting-publication-receipt.json', receipt)
    pending.unlink()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('stage')
    for name in ('viewer', 'dataset-commit', 'space-commit', 'baseline', 'output', 'expected-viewer-sha256'):
        prep.add_argument('--'+name, required=True)
    for command in ('check', 'publish'):
        sub.add_parser(command).add_argument('plan')
    args = parser.parse_args(argv)
    try:
        if args.command == 'stage':
            result = stage(args.viewer, args.dataset_commit, args.space_commit, args.baseline, args.output,
                           expected_viewer_sha256=args.expected_viewer_sha256)
            result = dict(schema=result['schema'], changed_paths=[r['path'] for r in result['files']], network=False, credentials=False)
        elif args.command == 'check':
            plan, _, _ = checked_plan(args.plan)
            result = dict(schema=plan['schema'], changed_paths=[r['path'] for r in plan['files']], network=False, credentials=False)
        else:
            result = publish(args.plan)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        # Never print raw SDK exceptions or signed CDN URLs.
        print('PiPER hosting patch stopped: '+(str(exc) if isinstance(exc, PublicationError) else type(exc).__name__), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
