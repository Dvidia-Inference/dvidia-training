"""Hashed, inspectable simulation skill packs. No automatic code execution."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import zipfile
import mujoco
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def export_pack(path, payload, policy):
    root = Path(__file__).resolve().parent
    files = {f'code/simlab/{p.name}': p.read_bytes() for p in sorted(root.glob('*.py'))}
    files['requirements.lock.txt'] = (root / 'requirements.lock.txt').read_bytes()
    files['code/simlab/requirements.lock.txt'] = files['requirements.lock.txt']
    files['LICENSE'] = (root/'LICENSE').read_bytes()
    files['code/simlab/LICENSE'] = files['LICENSE']
    files['run.json'] = encode(payload)
    files['policy.json'] = encode(policy)
    files['README.md'] = (
        '# DVIDIA simulation-only skill pack\n\n'
        'Task: CableEndReach-v0. Actuation: ideal Cartesian force attachment at the cable tip. '
        'Observations: privileged simulator state. No physical robot or sensor qualification.\n\n'
        'Verify this archive with `python -m simlab verify-pack skill-pack.zip`. '
        'After inspecting and unpacking it, create a Python 3.12 environment, install '
        '`requirements.lock.txt`, and use `PYTHONPATH=code python -m simlab run '
        '--config run.json --policy-file policy.json --seeds 100,101,102 --output replay`. '
        'Dependencies must be staged separately for an air-gapped installation.\n'
    ).encode()
    episodes = payload['episodes']
    manifest = {
        'schema_version': 1, 'task': 'CableEndReach-v0', 'status': 'simulation-only',
        'actuation_profile': 'cartesian_endpoint_force_v0',
        'observation_profile': 'privileged_simulator_state_v0',
        'physical_evidence': None,
        'config': payload['config'],
        'dependency_versions': {'python': platform.python_version(), 'mujoco': mujoco.__version__, 'numpy': np.__version__},
        'qualification': {
            'scope': 'fixed simulated seed suite; no physical or GPU qualification',
            'seeds': [e['seed'] for e in episodes],
            'successes': sum(e['success'] for e in episodes), 'episodes': len(episodes),
            'training_seeds': payload.get('training_seeds', []),
        },
        'files': {name: {'sha256': digest(data), 'bytes': len(data)} for name, data in files.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
        archive.writestr('manifest.json', encode(manifest))
    return manifest


def verify_pack(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 64:
            raise ValueError('Duplicate entries or oversized file list.')
        for item in archive.infolist():
            p = PurePosixPath(item.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in item.filename or item.file_size > 50_000_000:
                raise ValueError('Unsafe archive entry.')
        if sum(item.file_size for item in archive.infolist()) > 100_000_000:
            raise ValueError('Archive exceeds verification budget.')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('schema_version') != 1 or manifest.get('status') != 'simulation-only':
            raise ValueError('Unsupported skill manifest.')
        if set(names) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Manifest file list mismatch.')
        for name, expected in manifest['files'].items():
            data = archive.read(name)
            if len(data) != expected['bytes'] or digest(data) != expected['sha256']:
                raise ValueError(f'Artifact hash mismatch: {name}')
        payload = json.loads(archive.read('run.json'))
        if payload['config'] != manifest['config']:
            raise ValueError('Configuration mismatch.')
        seeds = [episode['seed'] for episode in payload['episodes']]
        if seeds != manifest['qualification']['seeds']:
            raise ValueError('Qualification seed mismatch.')
        qualification = manifest['qualification']
        if len(seeds) != qualification['episodes'] or sum(e['success'] for e in payload['episodes']) != qualification['successes']:
            raise ValueError('Qualification count mismatch.')
        return manifest
