"""Stage workstation delivery/branding fixes while retaining original evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

import prepare_workstation as original
import publish_robot_sim as publisher

NAMESPACE = original.NAMESPACE
EVIDENCE_NAMESPACE = 'workstation-delivery/v01'
MUTABLE = {'index.html', 'workstation.json', 'workstation-integrity.json', 'workstation_worker.bundle.js',
           'workstation.bundle.js', 'README.md'}
ADDED = {'native-assets.bin.gz'}


def stage(base_plan, workstation, evidence, output, source_commit, expected_remote_commit):
    from dvidia_training.workstation import inspect_export
    for value in (source_commit, expected_remote_commit):
        if type(value) is not str or re.fullmatch('[a-f0-9]{40}', value) is None:
            raise ValueError('Exact source and remote parent commits are required')
    base_plan, root, destination = map(lambda p: Path(p).absolute(), (base_plan, workstation, output))
    publisher._plain(root)
    publisher._plain(destination)
    if destination.exists() or root == destination or root in destination.parents or base_plan.parent in destination.parents:
        raise ValueError('Delivery publication output must be fresh and outside inputs')
    base = publisher.checked_plan(base_plan)
    inspect_export(root)
    proof = original.read_evidence(evidence)
    payload = {}
    for entry in base['repository']['files']:
        raw = publisher._bytes(base_plan.parent/'space'/entry['path'], publisher.MAX_FILE_BYTES)
        if original.record(raw) != {key: entry[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Prior artifact changed during staging')
        payload[entry['path']] = raw
    old = {name[len(NAMESPACE)+1:]: raw for name, raw in payload.items() if name.startswith(NAMESPACE+'/')}
    if not old or any(name.startswith(EVIDENCE_NAMESPACE+'/') for name in payload):
        raise ValueError('Requires the first published workstation and a fresh delivery evidence namespace')
    inspect_export(base_plan.parent/'space'/NAMESPACE)
    updated = {p.relative_to(root).as_posix(): publisher._bytes(p, publisher.MAX_FILE_BYTES)
               for p in root.rglob('*') if p.is_file()}
    if not set(old) <= set(updated) or set(updated)-set(old) != ADDED:
        raise ValueError('Delivery update must retain the original inventory and add only its transport bundle')
    changed = {name for name in old if old[name] != updated[name]}
    if changed-MUTABLE:
        raise ValueError('Delivery update changed physics, renderer geometry, native engine, meshes or notices')
    before, after = (json.loads(data['workstation.json']) for data in (old, updated))
    if 'native_asset_bundle' in before or 'native_asset_bundle' not in after:
        raise ValueError('Delivery update must introduce its compressed transport descriptor')
    if before != {key: value for key, value in after.items() if key != 'native_asset_bundle'}:
        raise ValueError('Delivery update changed native metadata or controller configuration')
    for name, raw in updated.items():
        publisher._artifact_name(NAMESPACE+'/'+name)
        publisher._no_private_artifact_bytes(NAMESPACE+'/'+name, raw)
        payload[NAMESPACE+'/'+name] = raw
    seal = {'schema': 'dvidia.workstation-delivery-evidence/v1', 'source_commit': source_commit,
            'prior_publication_commit': expected_remote_commit,
            'old_workstation_integrity': original.record(old['workstation-integrity.json']),
            'workstation_integrity': original.record(updated['workstation-integrity.json']),
            'changed_files': sorted(changed), 'added_files': sorted(ADDED),
            'native_metadata_unchanged': True,
            'files': {name: original.record(raw) for name, raw in sorted(proof.items())}}
    proof[original.EVIDENCE_SEAL] = original.encoded(seal)
    payload.update({EVIDENCE_NAMESPACE+'/'+name: raw for name, raw in proof.items()})
    provenance = json.loads(payload['provenance.json'])
    live = provenance['live_workstation']
    if live['files'] != {name: original.record(raw) for name, raw in old.items()}:
        raise ValueError('Prior workstation provenance differs from its published inventory')
    live['original_source_commit'] = live['source_commit']
    live['source_commit'] = source_commit
    live['files'] = {name: original.record(raw) for name, raw in updated.items()}
    live['delivery'] = {'path': EVIDENCE_NAMESPACE, 'source_commit': source_commit,
                        'integrity': original.record(proof[original.EVIDENCE_SEAL])}
    provenance['publication_byte_bounds'] = {
        'stored_payload_bytes': publisher.MAX_TOTAL_BYTES,
        'per_file_bytes': publisher.MAX_FILE_BYTES,
        'scope': '768 MiB complete stored Space cap; prior evidence retained; original per-file and attempt decoding bounds unchanged'}
    payload['provenance.json'] = original.encoded(provenance)
    payload['README.md'] += (f'\n\nThe live workstation now loads one compressed native mesh bundle. '
        f'Original meshes, physics parameters and original benchmark evidence are retained. '
        f'[Delivery validation]({EVIDENCE_NAMESPACE}/README.md).\n').encode()
    plan = {**base, 'source_commit': source_commit, 'expected_remote_commit': expected_remote_commit,
            'remove_files': [], 'total_bytes': sum(map(len, payload.values())),
            'repository': {**base['repository'], 'files': [
                {'path': name, **original.record(raw)} for name, raw in sorted(payload.items())]}}
    if len(payload) > publisher.MAX_FILES or plan['total_bytes'] > publisher.MAX_TOTAL_BYTES:
        raise ValueError('Delivery publication exceeds its file or byte budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.workstation-delivery-', dir=destination.parent))
    try:
        for name, raw in payload.items():
            path = staging/'space'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (staging/'publication-plan.json').write_bytes(original.encoded(plan))
        inspect_export(staging/'space'/NAMESPACE)
        publisher.checked_plan(staging/'publication-plan.json')
        if destination.exists() or destination.is_symlink():
            raise ValueError('Delivery output must remain fresh')
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'plan': str(destination/'publication-plan.json'), 'files': len(payload),
            'total_bytes': plan['total_bytes'], 'changed_workstation_files': sorted(changed)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-plan', 'workstation', 'evidence', 'output', 'source-commit', 'expected-remote-commit'):
        parser.add_argument('--'+name, required=True)
    print(json.dumps(stage(**vars(parser.parse_args())), indent=2))
