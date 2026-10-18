"""A frozen 2 x 2 diagnostic separates jaw capability from task time budget.

This follows the one-factor grip diagnostic on the same development scene.
Even a successful higher-capability/longer-budget result is not a repair of
the original hardware or its 18-second qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

from .arm_runner import config_for, payload_for, save_replay
from .capsule_installer import capsule_runtime
from .cli import NetworkGuard
from .footage_benchmark import freeze_protocol, verify_protocol
from .footage_pipeline import _fresh, _json, _canonical
from .grip_diagnostic import _same_trajectory, _write, rollout_diagnostic, summarize_episode
from .skill_capsule import inspect_capsule, source_bytes
from .skillspace import load_skillspace

ARMS = (
    {'id': 'jaw030-horizon18', 'scene_overrides': {}, 'horizon': 18., 'loaded_goal_cap_m': None},
    {'id': 'jaw060-horizon18', 'scene_overrides': {'jaw_force_limit': .6}, 'horizon': 18., 'loaded_goal_cap_m': None},
    {'id': 'jaw030-horizon24', 'scene_overrides': {}, 'horizon': 24., 'loaded_goal_cap_m': None},
    {'id': 'jaw060-horizon24', 'scene_overrides': {'jaw_force_limit': .6}, 'horizon': 24., 'loaded_goal_cap_m': None},
    {'id': 'loaded-4000um', 'scene_overrides': {}, 'horizon': 18., 'loaded_goal_cap_m': .004},
)
LIMITATIONS = [
    'Same development scene and frozen weights as phase one; this factorial is not independent generalization evidence.',
    'Higher jaw capability and a longer horizon are changed conditions, not improved original qualification.',
    'Jaw-capability intervention retains existing supervisor adaptation to that cap.',
    'All native success predicates, arm torque limits, phase budgets and recovery rules remain unchanged.',
    'The fifth, separate exploratory condition changes only the loaded Cartesian goal cap to 4 mm, not physical velocity/acceleration.',
    'No retraining, hardware promotion, premium-plan readiness or physical-robot claim.',
]


def paired_factorial(results):
    """Retain failed and missing attempts; never collapse them out of a rate."""
    if len(results) != 10 or len({(row['arm_id'], row['policy']) for row in results}) != 10:
        raise ValueError('Factorial requires four conditions plus one exploratory condition for both policies.')
    summary = {}
    for policy in ('student', 'teacher'):
        cells = {row['arm_id']: row for row in results if row['policy'] == policy}
        if set(cells) != {arm['id'] for arm in ARMS}:
            raise ValueError('Missing factorial policy/condition cell.')
        summary[policy] = {name: {'status': row['status'], 'success': row.get('measurement', {}).get('success', False),
                                 'simulated_seconds': row.get('measurement', {}).get('simulated_seconds'),
                                 'control_success': row.get('control', {}).get('success'),
                                 'control_status': row.get('control_status')}
                           for name, row in cells.items()}
    return summary


def run(diagnostic, output):
    started = time.perf_counter()
    diagnostic = Path(diagnostic).resolve()
    prior_protocol = _json(diagnostic/'protocol.json')
    prior_report = _json(diagnostic/'report.json')
    prior_hash = sha256((diagnostic/'protocol.json').read_bytes()).hexdigest()
    if (prior_protocol.get('kind') != 'dvidia.grip-recovery-diagnostic'
            or prior_report.get('kind') != 'dvidia.grip-recovery-diagnostic-result'
            or prior_report.get('protocol_sha256') != prior_hash
            or prior_report.get('completed_native_episodes') != 28 or prior_report.get('episode_errors') != 0):
        raise ValueError('A complete hash-bound first-phase diagnostic is required.')
    runtime = capsule_runtime()
    if runtime != prior_protocol['runtime']:
        raise ValueError('The diagnostic runtime changed.')
    capsule_raw = (diagnostic/'baseline.skill-capsule.json').read_bytes()
    if sha256(capsule_raw).hexdigest() != prior_protocol['capsule_file_sha256']:
        raise ValueError('Frozen baseline weights changed.')
    capsule = inspect_capsule(capsule_raw, expected_runtime=runtime)
    source, grounding = source_bytes(capsule), capsule['grounding']
    if 'simulation_grounding' in json.loads(source):
        grounding = None
    skill = load_skillspace(source, grounding)
    output = _fresh(output)
    (output/'baseline.skill-capsule.json').write_bytes(capsule_raw)
    source_hashes = {**runtime['files_sha256'], **{name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                               for name in ('grip_diagnostic.py', 'grip_followup.py')}}
    (output/'sources').mkdir()
    for name in source_hashes:
        (output/'sources'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    original_raw = {name: (diagnostic/'originals'/f'{name}.json').read_bytes() for name in ('student', 'teacher')}
    if any(sha256(raw).hexdigest() != prior_protocol['original_run_sha256'][name] for name, raw in original_raw.items()):
        raise ValueError('Frozen original trajectories changed.')
    originals = {name: json.loads(raw)['episodes'][0] for name, raw in original_raw.items()}
    (output/'originals').mkdir()
    for name, raw in original_raw.items():
        (output/'originals'/f'{name}.json').write_bytes(raw)
    protocol = {'schema_version': 1, 'kind': 'dvidia.grip-recovery-diagnostic', 'scope': 'simulation-only',
                'experiment_phase': 'factorial-followup',
                'physical_robot_ready': False, 'prior_diagnostic_protocol_sha256': prior_hash,
                'prior_diagnostic_report_sha256': sha256((diagnostic/'report.json').read_bytes()).hexdigest(),
                'parent_report_sha256': sha256((diagnostic/'report.json').read_bytes()).hexdigest(),
                'benchmark_report_sha256': prior_protocol['benchmark_report_sha256'],
                'benchmark_protocol_sha256': prior_protocol['benchmark_protocol_sha256'],
                'capsule_file_sha256': prior_protocol['capsule_file_sha256'], 'runtime': runtime,
                'source_sha256': source_hashes, 'original_run_sha256': prior_protocol['original_run_sha256'],
                'case': prior_protocol['case'], 'arms': list(ARMS), 'policies': ['student', 'teacher'],
                'planned_native_episodes': 20, 'limits': LIMITATIONS,
                'rules': {'same_model': True, 'retrain': False, 'qualification': False,
                          'factorial': 'First four arms cross jaw capability and horizon; fifth arm is a separate one-factor loaded-goal-cap exploration.',
                          'negative_control': 'Each policy/condition uses its own recorded arm tape with open jaws.'}}
    protocol_hash = freeze_protocol(output, protocol)
    def verify():
        verify_protocol(output, protocol)
        if capsule_runtime() != runtime or any(sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected
                                               for name, expected in source_hashes.items()):
            raise ValueError('Frozen factorial runtime or source changed.')
        if sha256((output/'baseline.skill-capsule.json').read_bytes()).hexdigest() != protocol['capsule_file_sha256']:
            raise ValueError('Frozen factorial weights changed.')
    results, completed, errors = [], 0, 0
    with NetworkGuard(True) as guard:
        for arm in ARMS:
            for name in protocol['policies']:
                verify()
                scene = {**protocol['case']['scene'], **arm['scene_overrides']}
                config = replace(config_for(skill, scene), horizon=arm['horizon'])
                directory = output/arm['id']/name
                directory.mkdir(parents=True)
                row = {'arm_id': arm['id'], 'policy': name, 'scene': scene, 'horizon': arm['horizon'],
                       'loaded_goal_cap_m': arm['loaded_goal_cap_m'],
                       'status': 'completed', 'qualification': 'diagnostic-only',
                       'run_path': str(directory.relative_to(output)/'run.json')}
                episode = None
                try:
                    episode = rollout_diagnostic(config, scene['seed'],
                        movement_head=capsule['movement_head'] if name == 'student' else None,
                        loaded_goal_cap_m=arm['loaded_goal_cap_m'])
                    payload = payload_for(skill, config, [episode], {'offline': True, 'scope': 'parent NetworkGuard'})
                    payload['diagnostic_only'] = True
                    _write(directory/'run.json', payload)
                    save_replay(payload, directory/'replay.html')
                    row['measurement'] = summarize_episode(episode)
                    if arm['id'] == ARMS[0]['id']:
                        row['exact_original_trajectory_reproduced'] = _same_trajectory(episode, originals[name])
                        if not row['exact_original_trajectory_reproduced']:
                            raise ValueError('Factorial baseline failed exact original trajectory reproduction.')
                    completed += 1
                except (ValueError, RuntimeError) as exc:
                    errors += 1
                    row.update(status='error', error=str(exc))
                    if arm['id'] == ARMS[0]['id']:
                        _write(directory/'error.json', row)
                        raise
                try:
                    if episode is None:
                        raise ValueError('No candidate tape for paired negative control.')
                    verify()
                    control = rollout_diagnostic(config, scene['seed'], action_tape=episode['actions'])
                    payload = payload_for(skill, config, [control], {'offline': True, 'scope': 'parent NetworkGuard'})
                    payload['diagnostic_only'] = True
                    _write(directory/'control.json', payload)
                    row.update(control=summarize_episode(control), control_status='completed')
                    completed += 1
                except (ValueError, RuntimeError) as exc:
                    errors += 1
                    row.update(control_status='error', control_error=str(exc))
                results.append(row)
                _write(output/'progress.json', {'protocol_sha256': protocol_hash, 'results': results})
                print(json.dumps({'arm': arm['id'], 'policy': name, 'status': row['status'],
                                  'success': row.get('measurement', {}).get('success'),
                                  'simulated_seconds': row.get('measurement', {}).get('simulated_seconds')}), flush=True)
    verify()
    result = {'schema_version': 1, 'kind': 'dvidia.grip-recovery-diagnostic-result', 'scope': 'simulation-only',
              'experiment_phase': 'factorial-followup',
              'physical_robot_ready': False, 'protocol_sha256': protocol_hash, 'results': results,
              'factorial': paired_factorial(results), 'completed_native_episodes': completed, 'episode_errors': errors,
              'elapsed_seconds': time.perf_counter()-started, 'network_guard': guard.record(),
              'cost': {'external_api_usd': 0., 'local_compute': 'unpriced'}, 'limits': LIMITATIONS,
              'timing_scope': 'Input checks, snapshots, all rollouts, telemetry and replay output; excludes startup and final report write.'}
    _write(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.diagnostic, args.output)
    print(json.dumps({key: result[key] for key in ('completed_native_episodes', 'episode_errors', 'elapsed_seconds', 'cost')}))


if __name__ == '__main__':
    main()
