"""Stage an HTML-only workstation refresh, retaining the published mirror runtime.

Defaults reconstruct the last complete mirror release plus its additive public
acceptance receipt. Staging is offline; publishing uses publish_robot_sim.py.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import tempfile

import prepare_workstation as original
import publish_robot_sim as publisher

NAMESPACE = original.NAMESPACE
EVIDENCE_NAMESPACE = 'workstation-editorial/v01'
MUTABLE = {'index.html', 'workstation-integrity.json'}
APPEND_NAMESPACE = 'workstation-mirror/v01/public/'
APPEND_NAMES = {APPEND_NAMESPACE+name for name in ('README.md', 'acceptance.json', 'test-provenance.json')}
DEFAULT_BASE = original.ROOT/'runs/robot-simulation-lab-publication-mirror-v01/publication-plan.json'
DEFAULT_APPEND = original.ROOT/'runs/hf-download-fix/public-evidence-publication'


class _ExecutableHTML(HTMLParser):
    """Record executable HTML so editorial changes cannot silently add scripts."""
    def __init__(self, raw):
        super().__init__(convert_charrefs=True)
        self.scripts, self.active = [], False
        self.feed(raw.decode('utf-8'))
        self.close()

    def handle_starttag(self, tag, attrs):
        if any(key.lower().startswith('on') or (value or '').lstrip().lower().startswith('javascript:')
               for key, value in attrs):
            raise ValueError('Editorial HTML may not contain event handlers or javascript URLs')
        if tag == 'script':
            self.scripts.append([sorted(attrs), ''])
            self.active = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.scripts[-1][1] += data


def _json(path):
    return json.loads(publisher._bytes(path, 1024*1024))


def _reconstruct(base_path, append_root, expected_remote_commit):
    base = publisher.checked_plan(base_path)
    base_raw = publisher._bytes(base_path, 1024*1024)
    receipt = _json(base_path.parent/'publication-receipt.json')
    addition_raw = publisher._bytes(append_root/'plan.json', 1024*1024)
    addition, published = json.loads(addition_raw), _json(append_root/'publication-receipt.json')
    if (receipt.get('repo_id') != publisher.REPO_ID or receipt.get('repo_type') != 'space'
            or receipt.get('plan_sha256') != original.record(base_raw)['sha256']
            or receipt.get('verified_files') != len(base['repository']['files'])
            or addition.get('kind') != 'dvidia.mirror-public-acceptance-append/v1'
            or addition.get('repo_id') != publisher.REPO_ID
            or addition.get('source_publication_plan_sha256') != receipt['plan_sha256']
            or addition.get('parent_commit') != receipt.get('commit')
            or addition.get('source_commit') != base.get('source_commit')
            or published.get('repo_id') != publisher.REPO_ID or published.get('repo_type') != 'space'
            or published.get('parent_commit') != receipt.get('commit')
            or published.get('commit') != expected_remote_commit
            or published.get('plan_sha256') != original.record(addition_raw)['sha256']
            or published.get('runtime_and_sealed_evidence_unchanged') is not True):
        raise ValueError('Baseline and acceptance receipts must bind the exact reviewed remote parent')
    entries = addition.get('files')
    if (type(entries) is not dict or set(entries) != APPEND_NAMES
            or published.get('added_files') != [{'path': name, **record} for name, record in entries.items()]
            or published.get('unchanged_existing_files') != len(base['repository']['files'])+1):
        raise ValueError('Acceptance append must retain the complete prior inventory and add its three known files')
    payload = {}
    for entry in base['repository']['files']:
        raw = publisher._bytes(base_path.parent/'space'/entry['path'], publisher.MAX_FILE_BYTES)
        if original.record(raw) != {key: entry[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Prior artifact changed during reconstruction')
        payload[entry['path']] = raw
    append_payload = append_root/'payload'
    actual = set()
    for path in append_payload.rglob('*'):
        publisher._plain(path)
        if path.is_file():
            actual.add(path.relative_to(append_payload).as_posix())
    if actual != APPEND_NAMES or set(payload)&APPEND_NAMES:
        raise ValueError('Acceptance append payload must match its exact additive inventory')
    for name, expected in entries.items():
        raw = publisher._bytes(append_payload/name, publisher.MAX_FILE_BYTES)
        if original.record(raw) != expected:
            raise ValueError('Acceptance append differs from the published receipt')
        publisher._no_private_artifact_bytes(name, raw)
        payload[name] = raw
    return base, payload


def stage(workstation, evidence, output, source_commit, expected_remote_commit,
          *, base_plan=DEFAULT_BASE, append_publication=DEFAULT_APPEND):
    from dvidia_training.workstation import inspect_export
    if any(type(value) is not str or re.fullmatch('[a-f0-9]{40}', value) is None
           for value in (source_commit, expected_remote_commit)):
        raise ValueError('Exact source and remote parent commits are required')
    root, evidence_root, destination, base_path, append_root = map(lambda p: Path(p).absolute(),
        (workstation, evidence, output, base_plan, append_publication))
    for path in (root, evidence_root, destination, base_path, append_root):
        publisher._plain(path)
    if root == evidence_root or root in evidence_root.parents or evidence_root in root.parents:
        raise ValueError('Workstation and evidence require separate directory trees')
    if destination.exists() or any(destination == p or p in destination.parents for p in
                                  (root, evidence_root, base_path.parent, append_root)):
        raise ValueError('Editorial output must be fresh and outside input trees')
    base, payload = _reconstruct(base_path, append_root, expected_remote_commit)
    inspect_export(root)
    inspect_export(base_path.parent/'space'/NAMESPACE)
    old = {name[len(NAMESPACE)+1:]: raw for name, raw in payload.items() if name.startswith(NAMESPACE+'/')}
    updated = {}
    for path in root.rglob('*'):
        publisher._plain(path)
        if path.is_file():
            updated[path.relative_to(root).as_posix()] = publisher._bytes(path, publisher.MAX_FILE_BYTES)
    if not old or set(old) != set(updated):
        raise ValueError('Editorial update must retain the exact prior runtime inventory')
    changed = {name for name in old if old[name] != updated[name]}
    if changed != MUTABLE:
        raise ValueError('Editorial update must change only HTML and its integrity seal')
    if _ExecutableHTML(old['index.html']).scripts != _ExecutableHTML(updated['index.html']).scripts:
        raise ValueError('Editorial update must preserve executable scripts exactly')
    before, after = [json.loads(data['workstation-integrity.json']) for data in (old, updated)]
    expected = json.loads(original.encoded(before))
    expected['files']['index.html'] = original.record(updated['index.html'])
    if after != expected:
        raise ValueError('Editorial integrity seal may update only the HTML record')
    if any(name.startswith(EVIDENCE_NAMESPACE+'/') for name in payload):
        raise ValueError('Editorial evidence requires a fresh namespace')
    proof = original.read_evidence(evidence_root)
    for name in MUTABLE:
        publisher._no_private_artifact_bytes(NAMESPACE+'/'+name, updated[name])
        payload[NAMESPACE+'/'+name] = updated[name]
    seal = {'schema': 'dvidia.workstation-editorial-evidence/v1', 'source_commit': source_commit,
            'prior_publication_commit': expected_remote_commit,
            'scope': 'HTML layout and copy only; no new physics or performance benchmark',
            'prior_workstation_integrity': original.record(old['workstation-integrity.json']),
            'workstation_integrity': original.record(updated['workstation-integrity.json']),
            'changed_workstation_files': sorted(changed), 'runtime_and_assets_unchanged': True,
            'files': {name: original.record(raw) for name, raw in sorted(proof.items())}}
    proof[original.EVIDENCE_SEAL] = original.encoded(seal)
    payload.update({EVIDENCE_NAMESPACE+'/'+name: raw for name, raw in proof.items()})
    provenance = json.loads(payload['provenance.json'])
    live = provenance['live_workstation']
    if live['files'] != {name: original.record(raw) for name, raw in old.items()} or 'editorial' in live:
        raise ValueError('Prior workstation provenance must match and retain its historical entries')
    live['editorial'] = {'path': EVIDENCE_NAMESPACE, 'source_commit': source_commit,
        'prior_source_commit': live['source_commit'], 'prior_publication_commit': expected_remote_commit,
        'integrity': original.record(proof[original.EVIDENCE_SEAL])}
    live['source_commit'] = source_commit
    live['files'] = {name: original.record(raw) for name, raw in updated.items()}
    payload['provenance.json'] = original.encoded(provenance)
    plan = {**base, 'source_commit': source_commit, 'expected_remote_commit': expected_remote_commit,
            'remove_files': [], 'total_bytes': sum(map(len, payload.values())),
            'repository': {**base['repository'], 'files': [
                {'path': name, **original.record(raw)} for name, raw in sorted(payload.items())]}}
    if len(payload) > publisher.MAX_FILES or plan['total_bytes'] > publisher.MAX_TOTAL_BYTES:
        raise ValueError('Editorial publication exceeds its bounded budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.workstation-editorial-', dir=destination.parent))
    try:
        for name, raw in payload.items():
            path = staging/'space'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw)
        (staging/'publication-plan.json').write_bytes(original.encoded(plan))
        inspect_export(staging/'space'/NAMESPACE)
        publisher.checked_plan(staging/'publication-plan.json')
        if destination.exists() or destination.is_symlink():
            raise ValueError('Editorial output must remain fresh')
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'plan': str(destination/'publication-plan.json'), 'files': len(payload),
            'evidence_files': len(proof), 'total_bytes': plan['total_bytes'],
            'changed_workstation_files': sorted(changed), 'prior_files_retained': len(payload)-len(proof)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workstation', 'evidence', 'output', 'source-commit', 'expected-remote-commit'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--base-plan', default=DEFAULT_BASE)
    parser.add_argument('--append-publication', default=DEFAULT_APPEND)
    print(json.dumps(stage(**vars(parser.parse_args())), indent=2))
