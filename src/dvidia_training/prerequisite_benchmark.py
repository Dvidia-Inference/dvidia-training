"""Reproducible native experiments in skill prerequisites, with every trial retained.

The web export replays measured body poses. Policies are authored and observations
are simulator truth. Neither the viewer nor this benchmark trains a robot policy.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
import time

from . import asset_preview, reference_replay, reference_sim as sim

SCHEMA = 'dvidia.skill-prerequisites/v1'
SCOPE = ('Authored PiPER policies in native MuJoCo with synthetic rigid objects and '
         'privileged simulator observations. Browser replay uses recorded poses; '
         'no learned policy, hardware safety qualification or real-world transfer result.')
TASKS = ('hinged-box', 'tool-reach')
POLICIES = ('baseline', 'recovery')
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_EXPORT_BYTES = 192 * 1024 * 1024


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def modules():
    from . import hinged_shoebox, tool_reach
    return {'hinged-box': hinged_shoebox, 'tool-reach': tool_reach}


def implementation_hashes():
    root = Path(__file__).parent
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in (
        'prerequisite_benchmark.py', 'hinged_shoebox.py', 'tool_reach.py',
        'workstation.py', 'asset_library.py', 'asset_catalog.json', 'robot_profiles.py',
        'prerequisite_lab.html', 'prerequisite_lab.js', 'prerequisite_lab.bundle.js')}


def protocol(split='heldout', task='all'):
    if split not in ('development', 'heldout') or task not in (*TASKS, 'all'):
        raise ValueError('Choose a declared split and task')
    cases = []
    for name, module in modules().items():
        if task not in (name, 'all'):
            continue
        for case in module.cases(split):
            if not isinstance(case, dict) or not re.fullmatch(r'[a-z0-9_-]{1,80}', case.get('id', '')):
                raise ValueError('Every case needs a bounded portable ID')
            for policy in POLICIES:
                cases.append({'id': name + '-' + case['id'] + '-' + policy,
                              'task': name, 'policy': policy, 'case': case})
    if len({item['id'] for item in cases}) != len(cases) or not 1 <= len(cases) <= 100:
        raise ValueError('Protocol requires distinct, bounded attempts')
    return {'schema': SCHEMA, 'scope': SCOPE, 'split': split, 'attempts': cases,
            'comparison': 'Matched declared scene conditions. Tool baseline is a no-tool capability control; it is not a policy-only ablation.',
            'split_scope': 'Explicit parameter holdouts, not independent real-world reliability trials. No held-out tuning.',
            'success_scope': 'Task-specific completion. Safe abort, retrieval, and retained lift are reported separately.',
            'timing_scope': 'CPU wall time for native execution including initialization and recording; rendering, downloads and real sensor latency excluded.'}


def _write(root, name, content):
    if len(content) > MAX_FILE_BYTES:
        raise ValueError('Export file exceeds byte limit: ' + name)
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError('Export path collision: ' + name)
        return
    with target.open('xb') as stream:
        stream.write(content)


def _binary(root, raw, prefix):
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('Decoded binary exceeds byte limit')
    digest = sha256(raw).hexdigest()
    packed = gzip.compress(raw, mtime=0)
    name = 'assets/' + prefix + '-' + digest[:20] + '.bin.gz'
    _write(root, name, packed)
    return {'href': name, 'compression': 'gzip', 'bytes': len(raw),
            'compressed_bytes': len(packed), 'sha256': digest}


def export_trial(root, trial_id, result, source):
    """Export native poses without recreating or interpolating a physics result."""
    _, np = sim._native_imports()
    model, frames = result['model'], result['frames']
    source = {**source, 'compiled_model_sha256': sim.model_digest(model)}
    if not 2 <= len(frames) <= 10000:
        raise ValueError('A replay needs between two and ten thousand recorded frames')
    poses = np.asarray([frame['body_poses'] for frame in frames], dtype='<f4')
    times = [float(frame['t']) for frame in frames]
    if poses.shape != (len(frames), model.nbody, 7) or not np.isfinite(poses).all():
        raise ValueError('Invalid recorded native body poses')
    if times[0] < 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Replay time must be strictly increasing')
    if not np.all(np.abs(np.linalg.norm(poses[:, :, 3:], axis=2) - 1) < .001):
        raise ValueError('Invalid native body quaternion')
    scene, geometry = asset_preview._scene(model, source)
    scene.update(schema='dvidia.prerequisite-scene/v1', scope=SCOPE)
    scene['geometry_binary'] = _binary(root, geometry, 'geometry')
    name = 'trials/' + trial_id + '.model.json'
    _write(root, name, encode(scene))
    motion = {'schema': 'dvidia.prerequisite-motion/v1', 'source': source,
              'body_count': model.nbody, 'frame_count': len(frames), 'times': times,
              'rows': [{k: v for k, v in row.items() if k != 'body_poses'} for row in frames],
              'motion_binary': _binary(root, poses.tobytes(order='C'), 'motion')}
    motion_name = 'trials/' + trial_id + '.motion.json'
    _write(root, motion_name, encode(motion))
    return {'model': name, 'motion': motion_name}


def summarize(records):
    groups = []
    for task in TASKS:
        for policy in POLICIES:
            rows = [row for row in records if row['task'] == task and row['policy'] == policy]
            if not rows:
                continue
            valid = [row for row in rows if row['valid']]
            outcomes = {}
            for row in rows:
                outcome = (row['metrics'].get('outcome', 'completed' if row['success'] else 'failed')
                           if row['valid'] else 'invalid_attempt')
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
            groups.append({'task': task, 'policy': policy, 'attempts': len(rows),
                           'valid': len(valid), 'successes': sum(row['success'] and row['valid'] for row in rows),
                           'outcomes': outcomes,
                           'wall_seconds': sum(row['wall_seconds'] for row in rows),
                           'metrics_note': 'See every individual trial for contact, retrieval, recovery and failure details.'})
    return groups


def paired_comparisons(records):
    pairs = []
    for baseline in (row for row in records if row['policy'] == 'baseline'):
        recovery = next((row for row in records if row['task'] == baseline['task']
                         and row['case'] == baseline['case'] and row['policy'] == 'recovery'), None)
        if recovery is None:
            raise ValueError('Every baseline needs its matched recovery attempt')
        a, b = baseline['metrics'], recovery['metrics']
        row = {'task': baseline['task'], 'case_id': baseline['case']['id'],
               'valid_pair': baseline['valid'] and recovery['valid'],
               'baseline_success': baseline['success'], 'recovery_success': recovery['success']}
        if row['valid_pair'] and baseline['task'] == 'hinged-box':
            row['lid_contact_avoided'] = a['lid_robot_collision'] and not b['lid_robot_collision']
            row['nuisance_interruption'] = baseline['success'] and b['intervention_count'] > 0
            row['nuisance_scope'] = 'Baseline completed without disallowed contact; conservative withdrawal is not proven necessary.'
            t0, t1 = a['completion_time_s'], b['completion_time_s']
            row['completion_delay_s'] = None if t0 is None or t1 is None else t1 - t0
        if row['valid_pair'] and baseline['task'] == 'tool-reach':
            row['baseline_retrieved'] = a['retrieval_success']
            row['with_tool_retrieved'] = b['retrieval_success']
            row['with_tool_retained_lift'] = b['retained_lift_success']
            row['comparison_scope'] = 'No-tool capability control versus mounted rake plus recovery; not an isolated policy effect.'
        pairs.append(row)
    return pairs


def seal(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Export cannot contain symlinks')
        if path.is_file() and path.name != 'integrity.json':
            content = path.read_bytes()
            files[path.relative_to(root).as_posix()] = {'bytes': len(content), 'sha256': sha256(content).hexdigest()}
    if any(item['bytes'] > MAX_FILE_BYTES for item in files.values()) or sum(item['bytes'] for item in files.values()) > MAX_EXPORT_BYTES:
        raise ValueError('Export exceeds size limits')
    _write(root, 'integrity.json', encode({'schema': 'dvidia.prerequisite-integrity/v1',
                                         'scope': 'Artifact consistency, not authenticity or physical validation.', 'files': files}))
    return files


def benchmark(assets, output, *, split='heldout', task='all', source_commit=None):
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Benchmark output must be a fresh directory')
    if source_commit is not None and re.fullmatch(r'[a-f0-9]{40}', source_commit) is None:
        raise ValueError('Source commit must be a full Git SHA')
    plan = protocol(split, task)
    plan.update(source_commit=source_commit, implementation_sha256=implementation_hashes())
    mj, _ = sim._native_imports()
    provenance, notices = asset_preview._licenses('piper', assets)
    destination.mkdir(parents=True)
    _write(destination, 'protocol.json', encode(plan))
    source = {'commit': source_commit, 'protocol_sha256': sha256(encode(plan)).hexdigest(),
              'mujoco': mj.__version__, 'robot_assets': provenance}
    records = []
    for attempt in plan['attempts']:
        start = time.perf_counter()
        row = {'id': attempt['id'], 'task': attempt['task'], 'policy': attempt['policy'],
               'case': attempt['case'], 'valid': False, 'success': False, 'metrics': {}, 'events': [], 'errors': []}
        try:
            result = modules()[attempt['task']].run_case(assets, attempt['case'], policy=attempt['policy'], record=True)
            row['wall_seconds'] = time.perf_counter() - start
            if type(result['success']) is not bool or type(result['valid']) is not bool:
                raise ValueError('Attempt validity and success must be explicit booleans')
            row.update(valid=result['valid'], success=result['success'] and result['valid'],
                       metrics=result['metrics'], events=result['events'])
            for key in ('native_trace', 'native_trace_columns', 'trace_columns', 'control_compute_ms', 'warnings'):
                if key in result:
                    row[key] = result[key]
            row['replay'] = export_trial(destination, attempt['id'], result, source)
            encode(row)
        except Exception as exc:
            row.update(valid=False, success=False)
            # Published failures use a category; exception messages can contain
            # private asset paths. Do not leak those paths into public evidence.
            row['errors'].append(type(exc).__name__ + ': attempt or export failed')
            try:
                encode(row)
            except (ValueError, TypeError):
                row.update(metrics={}, events=[])
                for key in ('native_trace', 'native_trace_columns', 'trace_columns', 'control_compute_ms', 'warnings'):
                    row.pop(key, None)
        row.setdefault('wall_seconds', time.perf_counter() - start)
        row['export_seconds'] = max(0., time.perf_counter() - start - row['wall_seconds'])
        # A failed attempt is evidence too; write it before starting the next one.
        _write(destination, 'trials/' + attempt['id'] + '.json', encode(row))
        # Keep the landing page small. Full-rate evidence remains in the exact
        # per-trial receipt; the index retains every trial and all other fields.
        records.append({key: value for key, value in row.items() if key != 'native_trace'})
        print(json.dumps({'attempt': row['id'], 'success': row['success'], 'valid': row['valid'],
                          'outcome': row['metrics'].get('outcome') if row['valid'] else 'invalid_attempt',
                          'errors': row['errors']}), flush=True)
    report = {'schema': SCHEMA, 'scope': SCOPE, 'source': source, 'split': split,
              'host': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version(),
                       'compute': 'CPU native simulation; no GPU required for benchmark. Browser replay needs WebGL2.'},
              'learned_policy': False, 'hardware_validated': False, 'attempts': records,
              'groups': summarize(records), 'paired_comparisons': paired_comparisons(records)}
    _write(destination, 'benchmark.json', encode(report))
    resource = Path(__file__).parent
    for name in ('prerequisite_lab.html', 'prerequisite_lab.bundle.js', 'reference_replay.third-party.txt'):
        _write(destination, 'index.html' if name.endswith('.html') else name, (resource / name).read_bytes())
    for name, content in notices.items():
        _write(destination, name, content)
    _write(destination, 'LICENSE', reference_replay.MIT_LICENSE.encode())
    _write(destination, 'README.md', ('# DVIDIA skill prerequisites v0.1\n\n' + SCOPE + '\n\n'
        'Open the replay by serving this folder: `python -m http.server 8000`. All visual assets are local. '
        'The browser plays recorded native poses; it does not run physics.\n\n'
        'Reproduce native experiments using the matching Git source commit, install `.[arm]`, '
        'fetch pinned PiPER assets with `dvidia-robot-sim fetch piper --output robot-assets`, then run '
        '`dvidia-prerequisites --assets robot-assets --output fresh-results --split ' + split + '`.\n\n'
        'DVIDIA code and synthetic objects: MIT. Upstream arm and viewer notices are retained under licenses/ '
        'and reference_replay.third-party.txt. Minimum RAM and cross-device throughput are not established.\n').encode())
    seal(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('development', 'heldout'), default='heldout')
    parser.add_argument('--task', choices=(*TASKS, 'all'), default='all')
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    report = benchmark(args.assets, args.output, split=args.split, task=args.task, source_commit=args.source_commit)
    print(json.dumps({'output': args.output, 'groups': report['groups']}))


if __name__ == '__main__':
    main()
