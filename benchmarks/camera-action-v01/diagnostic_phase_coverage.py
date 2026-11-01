"""Read-only phase coverage characterization of the four audited diagnostics.

Teacher phase is used only for this offline analysis. It is never exported as
a learner input. Motor decisions reuse a 25 Hz camera at 50 Hz, so decision
counts are correlated observations, not independent tracking trials.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path


MANIPULATION_PHASES = ('close', 'lift', 'transfer', 'place', 'release')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024**2:
        raise ValueError('Expected an ordinary bounded JSON artifact.')
    return json.loads(path.read_bytes())


def hash_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sidecar(root, reference, inventory):
    name = reference['path']; relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid source path.')
    path = root/relative
    if path.is_symlink():
        raise ValueError('Source cannot be a symlink.')
    raw = path.read_bytes()
    identity = {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}
    if (inventory.get(name) != identity
            or {key: reference[key] for key in ('bytes', 'sha256')} != identity):
        raise ValueError('Teacher/perception sidecar identity changed.')
    return json.loads(raw)


def analyze(study, audited):
    root = Path(study)
    evidence = read(audited)
    if (evidence.get('schema') != 'dvidia.camera-action-diagnostic-audit/v1'
            or evidence.get('passed') is not True or evidence.get('attempts_verified') != 4):
        raise ValueError('A passing four-diagnostic independent audit is required.')
    stage = read(root/'stages/diagnostic.json')
    protocol = read(root/'protocol.json')
    registration = read(root/'registration.json')
    if hash_file(root/'protocol.json') != registration['protocol_sha256']:
        raise ValueError('Registered protocol changed.')
    expected = {row['id']: row for row in protocol['rosters']['diagnostic']}
    if len(stage['rows']) != 4 or {row['id'] for row in stage['rows']} != set(expected):
        raise ValueError('Expected the exact four diagnostic rows.')
    audited_cases = {row['case_id'] for row in evidence['results']}
    if audited_cases != {row['case']['id'] for row in stage['rows']}:
        raise ValueError('Audit and diagnostic case identities differ.')
    total = defaultdict(lambda: {'decisions': 0, 'available': 0})
    results = []
    for row in stage['rows']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Episode path is not study-relative.')
        episode = root/relative
        if (hash_file(episode/'integrity.json') != row['integrity_sha256']
                or hash_file(episode/'rollout.json') != row['receipt_sha256']):
            raise ValueError('Episode binding changed after stage sealing.')
        inventory = read(episode/'integrity.json')['files']
        raw = (episode/'recording.json').read_bytes()
        if inventory['recording.json'] != {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}:
            raise ValueError('Recording bytes differ from the episode seal.')
        recording = json.loads(raw)
        counts = defaultdict(lambda: {'decisions': 0, 'available': 0})
        for step in recording['steps']:
            query = sidecar(episode, step['teacher_query'], inventory)
            perception = sidecar(episode, step['perception'], inventory)
            if (query['snapshot_sha256'] != digest(step['before'])
                    or perception['snapshot_sha256'] != digest(step['before'])
                    or query['index'] != step['index'] or perception['index'] != step['index']):
                raise ValueError('Phase and perception are not aligned to the same decision.')
            phase = query.get('phase')
            if type(phase) is not str or len(phase) > 80:
                raise ValueError('Expected a bounded offline teacher phase label.')
            available = int(perception['estimate']['status'] == 'available')
            counts[phase]['decisions'] += 1; counts[phase]['available'] += available
            total[phase]['decisions'] += 1; total[phase]['available'] += available
        results.append(dict(case_id=row['case']['id'], source_group=row['case']['group_id'],
            phases=dict(counts), integrity_sha256=row['integrity_sha256']))
    contact_count = sum(total[phase]['decisions'] for phase in MANIPULATION_PHASES)
    contact_available = sum(total[phase]['available'] for phase in MANIPULATION_PHASES)
    return dict(schema='dvidia.camera-action-phase-coverage/v1',
        protocol_sha256=hash_file(root/'protocol.json'), diagnostic_audit_sha256=hash_file(audited),
        analysis_source_sha256=hash_file(__file__), cases=4, starting_position_groups=2,
        total_motor_decisions=sum(v['decisions'] for v in total.values()), phases=dict(total),
        manipulation_phases=list(MANIPULATION_PHASES), manipulation_decisions=contact_count,
        manipulation_available=contact_available, results=results,
        evidence_scope='Offline teacher phase analysis only; phases are excluded from learner inputs. Repeated 50 Hz decisions share 25 Hz camera frames.',
        physics_executed=False, estimator_executed=False, physical_robot_ready=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study'); parser.add_argument('--audit', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.study, args.audit), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
