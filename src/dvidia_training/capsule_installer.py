"""Install data-only movement capsules and qualify one exact simulation scene."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from .installer import runtime_contract
from .skill_capsule import encode_capsule, inspect_capsule, qualification_receipt, source_bytes, scene_sha256
from .skillspace import load_skillspace

EXTRA_RUNTIME_FILES = ('arm_distill.py', 'skill_capsule.py', 'capsule_installer.py')
_LOADED_HASHES = {name: sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                 for name in EXTRA_RUNTIME_FILES}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sidecar(capsule):
    return None if 'simulation_grounding' in json.loads(source_bytes(capsule)) else capsule['grounding']


def capsule_runtime():
    base = runtime_contract()
    files = {name: sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
             for name in EXTRA_RUNTIME_FILES}
    if files != _LOADED_HASHES:
        raise ValueError('Capsule runtime changed. Restart the lab before importing or executing a capsule.')
    return {**base, 'kind': 'hybrid-distilled-placement-runtime',
            'files_sha256': {**base['files_sha256'], **files}}


def scene_digest(scene):
    return scene_sha256(scene)


def _root(directory, identity):
    if not isinstance(identity, str) or not re.fullmatch('[a-f0-9]{24}', identity):
        raise ValueError('Choose an imported skill capsule.')
    return Path(directory)/identity


def install_capsule(document, directory):
    runtime = capsule_runtime()
    capsule = inspect_capsule(document, expected_runtime=runtime)
    raw = encode_capsule(capsule)
    identity = sha256(canonical({'capsule': capsule['payload_sha256'], 'runtime': runtime})).hexdigest()[:24]
    skill = load_skillspace(source_bytes(capsule), _sidecar(capsule))
    record = {'kind': 'skill_capsule_installation', 'schema_version': 1,
              'installation_id': identity, 'status': 'candidate', 'scope': 'simulation-only',
              'physical_robot_ready': False, 'source_url': None,
              'capsule_sha256': capsule['payload_sha256'], 'capsule_file_sha256': sha256(raw).hexdigest(),
              'runtime': runtime, 'skill': skill.to_dict(),
              'policy_origin': 'learned joint-movement head; authored phases, grasp and recovery'}
    target = _root(directory, identity)
    target.mkdir(parents=True, exist_ok=True)
    (target/'capsule.json').write_bytes(raw)
    (target/'installation.json').write_bytes(canonical(record))
    return record


def read_capsule(directory, identity):
    root = _root(directory, identity)
    raw = (root/'capsule.json').read_bytes()
    runtime = capsule_runtime()
    capsule = inspect_capsule(raw, expected_runtime=runtime)
    record = json.loads((root/'installation.json').read_text())
    expected_id = sha256(canonical({'capsule': capsule['payload_sha256'], 'runtime': runtime})).hexdigest()[:24]
    expected = {'kind': 'skill_capsule_installation', 'schema_version': 1,
              'installation_id': expected_id, 'status': 'candidate', 'scope': 'simulation-only',
              'physical_robot_ready': False, 'source_url': None,
              'capsule_sha256': capsule['payload_sha256'], 'capsule_file_sha256': sha256(raw).hexdigest(),
              'runtime': runtime, 'skill': load_skillspace(source_bytes(capsule), _sidecar(capsule)).to_dict(),
              'policy_origin': 'learned joint-movement head; authored phases, grasp and recovery'}
    if identity != expected_id or canonical(record) != canonical(expected):
        raise ValueError('Capsule installation changed. Import the candidate again.')
    return record, capsule


def _qualification(directory, identity, scene, capsule):
    folder = _root(directory, identity)/'qualifications'/scene_digest(scene)
    raw = (folder/'run.json').read_bytes()
    controls_raw = (folder/'controls.json').read_bytes()
    run, controls = json.loads(raw), json.loads(controls_raw)
    expected = qualification_receipt(capsule, scene, run, runtime=capsule_runtime(), controls=controls)
    stored = json.loads((folder/'qualification.json').read_text())
    if canonical(stored) != canonical(expected):
        raise ValueError('Stored scene qualification changed. Validate this scene again.')
    return stored


def read_capsule_installation(directory, identity, scene=None, *, require_qualified=False):
    record, capsule = read_capsule(directory, identity)
    if scene is not None:
        try:
            receipt = _qualification(directory, identity, scene, capsule)
        except (FileNotFoundError, ValueError, KeyError):
            if require_qualified:
                raise ValueError('Validate this imported capsule in the current scene before execution.')
        else:
            record = {**record, 'status': 'validated_simulation_scene', 'qualification': receipt}
    elif require_qualified:
        raise ValueError('Execution requires the exact qualified scene.')
    return record


def record_qualification(directory, identity, scene, run, controls):
    _, capsule = read_capsule(directory, identity)
    receipt = qualification_receipt(capsule, scene, run, runtime=capsule_runtime(), controls=controls)
    raw, controls_raw = canonical(run), canonical(controls)
    folder = _root(directory, identity)/'qualifications'/scene_digest(scene)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'run.json').write_bytes(raw)
    (folder/'controls.json').write_bytes(controls_raw)
    (folder/'qualification.json').write_bytes(canonical(receipt))
    return receipt


def capsule_worker(directory, identity, scene, output, *, qualify=False):
    from .arm_runner import config_for, payload_for, rollout, run_skill, write_run
    from .cli import NetworkGuard
    if not qualify:
        read_capsule_installation(directory, identity, scene, require_qualified=True)
    record, capsule = read_capsule(directory, identity)
    source = source_bytes(capsule)
    grounding = _sidecar(capsule)
    run = run_skill(source, grounding, scene, movement_head=capsule['movement_head'], offline=True)
    run.update(runtime=capsule_runtime(), installation_id=identity, source_url=None,
               capsule_sha256=capsule['payload_sha256'], qualification_scope='one exact simulated scene')
    Path(output).mkdir(parents=True, exist_ok=True)
    if qualify:
        skill = load_skillspace(source, grounding)
        config = config_for(skill, scene)
        with NetworkGuard(True) as guard:
            control = rollout(config, scene['seed'], 'replay_open_jaw', action_tape=run['episodes'][0]['actions'])
        controls = payload_for(skill, config, [control], guard.record())
        controls['runtime'] = capsule_runtime()
        (Path(output)/'controls.json').write_bytes(canonical(controls))
        try:
            receipt = record_qualification(directory, identity, scene, run, controls)
        except ValueError as exc:
            run['local_qualification'] = {'status': 'failed', 'reason': str(exc), 'scope': 'one exact simulated scene'}
        else:
            run['local_qualification'] = receipt
        (Path(output)/'qualification.json').write_bytes(canonical(run['local_qualification']))
    write_run(output, run, source=source, grounding=grounding)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Package a learned movement head with its authored task contract.')
    build.add_argument('--model', type=Path, required=True)
    build.add_argument('--source', type=Path, default=Path(__file__).with_name('place_cup.skill.json'))
    build.add_argument('--grounding', type=Path)
    build.add_argument('--output', type=Path, required=True)
    install = commands.add_parser('install', help='Import candidate data; no native qualification is implied.')
    install.add_argument('capsule', type=Path)
    install.add_argument('--directory', type=Path, default=Path('runs/capsule-installs'))
    for name in ('qualify', 'run'):
        action = commands.add_parser(name)
        action.add_argument('installation_id')
        action.add_argument('--directory', type=Path, default=Path('runs/capsule-installs'))
        action.add_argument('--scene', type=Path)
        action.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        from .arm_runner import authored_grounding
        from .skill_capsule import build_capsule
        source = args.source.read_bytes()
        grounding = json.loads(args.grounding.read_text()) if args.grounding else (
            None if 'simulation_grounding' in json.loads(source) else authored_grounding(source))
        capsule = build_capsule(source, grounding, json.loads(args.model.read_text()), runtime=capsule_runtime())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encode_capsule(capsule))
        print(json.dumps({'output': str(args.output), 'capsule_sha256': capsule['payload_sha256'], 'status': 'candidate'}))
    elif args.command == 'install':
        record = install_capsule(args.capsule.read_bytes(), args.directory)
        print(json.dumps({'installation_id': record['installation_id'], 'status': record['status']}))
    else:
        from .arm_runner import validate_scene
        scene = validate_scene(json.loads(args.scene.read_text()) if args.scene else None)
        result = capsule_worker(args.directory, args.installation_id, scene, args.output,
                                qualify=args.command=='qualify')
        print(json.dumps({'summary': result['summary'], 'qualification': result.get('local_qualification')}))


if __name__ == '__main__':
    main()
