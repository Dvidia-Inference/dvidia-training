"""Stage a verified workstation asset mirror while preserving prior evidence."""
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
EVIDENCE_NAMESPACE = 'workstation-mirror/v01'
MUTABLE = {'workstation.bundle.js', 'workstation_worker.bundle.js', 'model.json',
           'workstation-integrity.json', 'README.md'}


def _check_metadata(old, updated):
    before, after = (json.loads(data['model.json']) for data in (old, updated))
    if type(before) is not dict or type(after) is not dict:
        raise ValueError('Mirror update requires model dictionaries')
    old_geometry, new_geometry = before.get('geometry_binary'), after.get('geometry_binary')
    if type(old_geometry) is not dict or type(new_geometry) is not dict:
        raise ValueError('Mirror update requires geometry descriptors')
    checksum = original.record(old['geometry.bin.gz'])['sha256']
    if ('compressed_sha256' in old_geometry and old_geometry['compressed_sha256'] != checksum):
        raise ValueError('Prior compressed geometry checksum does not match unchanged bytes')
    if new_geometry.get('compressed_sha256') != checksum:
        raise ValueError('Mirror compressed geometry checksum must match unchanged bytes')
    before = {**before, 'geometry_binary': {key: value for key, value in old_geometry.items()
                                           if key != 'compressed_sha256'}}
    after = {**after, 'geometry_binary': {key: value for key, value in new_geometry.items()
                                         if key != 'compressed_sha256'}}
    if json.dumps(before, sort_keys=True, allow_nan=False) != json.dumps(after, sort_keys=True, allow_nan=False):
        raise ValueError('Mirror update may add only the compressed geometry checksum to model metadata')


def stage(base_plan, workstation, evidence, output, source_commit, expected_remote_commit):
    from dvidia_training.workstation import inspect_export
    if any(type(value) is not str or re.fullmatch('[a-f0-9]{40}', value) is None
           for value in (source_commit, expected_remote_commit)):
        raise ValueError('Exact source and remote parent commits are required')
    base_plan, root, evidence_root, destination = map(lambda p: Path(p).absolute(),
                                                    (base_plan, workstation, evidence, output))
    for path in (root, evidence_root, destination):
        publisher._plain(path)
    if root == evidence_root or root in evidence_root.parents or evidence_root in root.parents:
        raise ValueError('Workstation and evidence inputs require separate directory trees')
    inputs = (root, evidence_root, base_plan.parent)
    if destination.exists() or any(destination == p or p in destination.parents for p in inputs):
        raise ValueError('Mirror update output must be fresh and outside input trees')
    base = publisher.checked_plan(base_plan)
    inspect_export(root)
    proof = original.read_evidence(evidence_root)
    payload = {}
    for entry in base['repository']['files']:
        raw = publisher._bytes(base_plan.parent/'space'/entry['path'], publisher.MAX_FILE_BYTES)
        if original.record(raw) != {key: entry[key] for key in ('bytes', 'sha256')}:
            raise ValueError('Prior artifact changed during staging')
        payload[entry['path']] = raw
    old = {name[len(NAMESPACE)+1:]: raw for name, raw in payload.items() if name.startswith(NAMESPACE+'/')}
    if not old or any(name.startswith(EVIDENCE_NAMESPACE+'/') for name in payload):
        raise ValueError('Requires a published workstation and a fresh mirror evidence namespace')
    inspect_export(base_plan.parent/'space'/NAMESPACE)
    updated = {}
    for path in root.rglob('*'):
        publisher._plain(path)
        if path.is_file():
            updated[path.relative_to(root).as_posix()] = publisher._bytes(path, publisher.MAX_FILE_BYTES)
    if set(old) != set(updated):
        raise ValueError('Mirror update must retain the exact prior runtime inventory')
    changed = {name for name in old if old[name] != updated[name]}
    if changed-MUTABLE:
        raise ValueError('Mirror update changed physics, renderer geometry, native engine, meshes or notices')
    _check_metadata(old, updated)
    for name, raw in updated.items():
        publisher._artifact_name(NAMESPACE+'/'+name)
        publisher._no_private_artifact_bytes(NAMESPACE+'/'+name, raw)
        payload[NAMESPACE+'/'+name] = raw
    seal = {'schema': 'dvidia.workstation-mirror-evidence/v1', 'source_commit': source_commit,
            'prior_publication_commit': expected_remote_commit,
            'scope': 'SHA-256-verified DVIDIA research mirror delivery; original binary, physics and evidence bytes retained',
            'old_workstation_integrity': original.record(old['workstation-integrity.json']),
            'workstation_integrity': original.record(updated['workstation-integrity.json']),
            'changed_files': sorted(changed), 'added_files': [],
            'native_engine_bytes_unchanged': True, 'prior_native_metadata_unchanged': True,
            'compressed_resources_unchanged': True,
            'files': {name: original.record(raw) for name, raw in sorted(proof.items())}}
    proof[original.EVIDENCE_SEAL] = original.encoded(seal)
    payload.update({EVIDENCE_NAMESPACE+'/'+name: raw for name, raw in proof.items()})
    provenance = json.loads(payload['provenance.json'])
    live = provenance['live_workstation']
    if live['files'] != {name: original.record(raw) for name, raw in old.items()}:
        raise ValueError('Prior workstation provenance differs from its published inventory')
    if 'mirror' in live:
        raise ValueError('Prior mirror provenance must not be replaced')
    live['mirror'] = {'path': EVIDENCE_NAMESPACE, 'source_commit': source_commit,
                      'prior_source_commit': live['source_commit'],
                      'prior_publication_commit': expected_remote_commit,
                      'integrity': original.record(proof[original.EVIDENCE_SEAL])}
    live['source_commit'] = source_commit
    live['files'] = {name: original.record(raw) for name, raw in updated.items()}
    payload['provenance.json'] = original.encoded(provenance)
    payload['README.md'] += (f'\n\nThe Hugging Face workstation can load its three compressed simulator resources from '
        f'DVIDIA’s research mirror, with exact byte bounds and SHA-256 verification. '
        f'Original engine, mesh, geometry and benchmark bytes are retained. '
        f'[Mirror validation]({EVIDENCE_NAMESPACE}/README.md).\n').encode()
    plan = {**base, 'source_commit': source_commit, 'expected_remote_commit': expected_remote_commit,
            'remove_files': [], 'total_bytes': sum(map(len, payload.values())),
            'repository': {**base['repository'], 'files': [
                {'path': name, **original.record(raw)} for name, raw in sorted(payload.items())]}}
    if len(payload) > publisher.MAX_FILES or plan['total_bytes'] > publisher.MAX_TOTAL_BYTES:
        raise ValueError('Mirror publication exceeds its file or byte budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.workstation-mirror-', dir=destination.parent))
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
            raise ValueError('Mirror output must remain fresh')
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'plan': str(destination/'publication-plan.json'), 'files': len(payload),
            'evidence_files': len(proof), 'total_bytes': plan['total_bytes'],
            'changed_workstation_files': sorted(changed), 'added_workstation_files': []}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-plan', 'workstation', 'evidence', 'output', 'source-commit', 'expected-remote-commit'):
        parser.add_argument('--'+name, required=True)
    print(json.dumps(stage(**vars(parser.parse_args())), indent=2))
