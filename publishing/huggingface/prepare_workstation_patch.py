"""Stage a bounded workstation download fix without rewriting published evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile
import zlib

import prepare_workstation as original
import publish_robot_sim as publisher

NAMESPACE = original.NAMESPACE
EVIDENCE_NAMESPACE = 'workstation-download-fix/v01'
MUTABLE = {'index.html', 'workstation.bundle.js', 'workstation_worker.bundle.js',
           'workstation.json', 'workstation-integrity.json', 'README.md'}
ADDED = {'mujoco.wasm.gz'}


def _check_metadata(old, updated):
    before, after = (json.loads(data['workstation.json']) for data in (old, updated))
    descriptors = {'native_wasm', 'wasm_bundle'}
    if (type(before) is not dict or type(after) is not dict or descriptors & before.keys()
            or not descriptors <= after.keys()
            or before != {key: value for key, value in after.items() if key not in descriptors}):
        raise ValueError('Download fix must preserve all prior metadata and add only native WASM descriptors')
    native, compressed = old['mujoco.wasm'], updated['mujoco.wasm.gz']
    if after['native_wasm'] != {'path': 'mujoco.wasm', **original.record(native)}:
        raise ValueError('Native WASM descriptor must match unchanged published engine bytes')
    if after['wasm_bundle'] != {'path': 'mujoco.wasm.gz', 'compression': 'gzip',
                              **original.record(compressed), 'uncompressed_bytes': len(native)}:
        raise ValueError('WASM bundle descriptor must match compressed and native byte bounds')
    decoder = zlib.decompressobj(16+zlib.MAX_WBITS)
    try:
        decoded = decoder.decompress(compressed, len(native)+1)
    except zlib.error as exc:
        raise ValueError('WASM bundle is not valid bounded gzip') from exc
    if (decoded != native or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data):
        raise ValueError('WASM bundle must decode exactly to the unchanged engine with no trailing data')


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
        raise ValueError('Download fix output must be fresh and outside input trees')
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
        raise ValueError('Requires a published workstation and a fresh download-fix evidence namespace')
    inspect_export(base_plan.parent/'space'/NAMESPACE)
    updated = {}
    for path in root.rglob('*'):
        publisher._plain(path)
        if path.is_file():
            updated[path.relative_to(root).as_posix()] = publisher._bytes(path, publisher.MAX_FILE_BYTES)
    if not set(old) <= set(updated) or set(updated)-set(old) != ADDED:
        raise ValueError('Download fix must retain the prior inventory and add only mujoco.wasm.gz')
    changed = {name for name in old if old[name] != updated[name]}
    if changed-MUTABLE:
        raise ValueError('Download fix changed physics, renderer geometry, native engine, meshes or notices')
    _check_metadata(old, updated)
    for name, raw in updated.items():
        publisher._artifact_name(NAMESPACE+'/'+name)
        publisher._no_private_artifact_bytes(NAMESPACE+'/'+name, raw)
        payload[NAMESPACE+'/'+name] = raw
    seal = {'schema': 'dvidia.workstation-download-fix-evidence/v1', 'source_commit': source_commit,
            'prior_publication_commit': expected_remote_commit,
            'scope': 'Download transport and browser validation; original physics and evidence retained',
            'old_workstation_integrity': original.record(old['workstation-integrity.json']),
            'workstation_integrity': original.record(updated['workstation-integrity.json']),
            'changed_files': sorted(changed), 'added_files': sorted(ADDED),
            'native_engine_bytes_unchanged': True, 'prior_native_metadata_unchanged': True,
            'files': {name: original.record(raw) for name, raw in sorted(proof.items())}}
    proof[original.EVIDENCE_SEAL] = original.encoded(seal)
    payload.update({EVIDENCE_NAMESPACE+'/'+name: raw for name, raw in proof.items()})
    provenance = json.loads(payload['provenance.json'])
    live = provenance['live_workstation']
    if live['files'] != {name: original.record(raw) for name, raw in old.items()}:
        raise ValueError('Prior workstation provenance differs from its published inventory')
    if 'download_fix' in live:
        raise ValueError('Prior download-fix provenance must not be replaced')
    live['download_fix'] = {'path': EVIDENCE_NAMESPACE, 'source_commit': source_commit,
                            'prior_source_commit': live['source_commit'],
                            'prior_publication_commit': expected_remote_commit,
                            'integrity': original.record(proof[original.EVIDENCE_SEAL])}
    live['source_commit'] = source_commit
    live['files'] = {name: original.record(raw) for name, raw in updated.items()}
    payload['provenance.json'] = original.encoded(provenance)
    payload['README.md'] += (f'\n\nThe workstation download fix adds compressed WASM transport and '
        f'bounded download recovery. Original engine, mesh and benchmark bytes are retained. '
        f'[Download-fix validation]({EVIDENCE_NAMESPACE}/README.md).\n').encode()
    plan = {**base, 'source_commit': source_commit, 'expected_remote_commit': expected_remote_commit,
            'remove_files': [], 'total_bytes': sum(map(len, payload.values())),
            'repository': {**base['repository'], 'files': [
                {'path': name, **original.record(raw)} for name, raw in sorted(payload.items())]}}
    if len(payload) > publisher.MAX_FILES or plan['total_bytes'] > publisher.MAX_TOTAL_BYTES:
        raise ValueError('Download-fix publication exceeds its file or byte budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.workstation-download-fix-', dir=destination.parent))
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
            raise ValueError('Download-fix output must remain fresh')
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'plan': str(destination/'publication-plan.json'), 'files': len(payload),
            'evidence_files': len(proof), 'total_bytes': plan['total_bytes'],
            'changed_workstation_files': sorted(changed), 'added_workstation_files': sorted(ADDED)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-plan', 'workstation', 'evidence', 'output', 'source-commit', 'expected-remote-commit'):
        parser.add_argument('--'+name, required=True)
    print(json.dumps(stage(**vars(parser.parse_args())), indent=2))
