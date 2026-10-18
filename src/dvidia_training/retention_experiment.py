"""Frozen pre-recovery retention experiment at unchanged force caps and deadline."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from .arm_distill import MovementHead
from .arm_env import ArmEnv
from .arm_policy import ArmPickPlacePolicy
from .arm_runner import config_for, payload_for, save_replay
from .capsule_installer import capsule_runtime
from .cli import NetworkGuard
from .footage_benchmark import freeze_protocol, verify_protocol
from .footage_pipeline import _fresh, _json, _canonical
from .grip_diagnostic import GoalProxy, _same_trajectory, _write, summarize_episode
from .improvement import load_benchmark
from .skill_capsule import inspect_capsule, source_bytes
from .skillspace import load_skillspace
from .retention_policy import SmoothedLoadedGoalProxy, SMOOTHING_ALPHA, LOW_HEADROOM_MARGIN
from .controlled_recovery_release_experiment import measurement as recovery_measurement

SETTINGS = {
    'smoothing_alpha': SMOOTHING_ALPHA, 'low_headroom_margin': LOW_HEADROOM_MARGIN,
    'phases': ['lift', 'transfer'], 'control_dt_seconds': .02,
    'filter': 'convex smoothing of requested Cartesian increments; reset outside loaded phases',
    'maximum_increment': 'original controller cap preserved; physical speed not guaranteed',
    'force_caps': 'each original case unchanged', 'horizon_seconds': 18.,
    'phase_budgets': 'all original phase budgets unchanged',
    'movement_mapping': 'same frozen student head / authored teacher IK; no fallback',
    'recovery': 'original recovery unchanged to isolate the pre-recovery intervention',
}
LIMITATIONS = [
    'One frozen controller experiment on 16 previously inspected simulation cases; regression, not held-out generalization.',
    'Same original force caps, model weights, object states/material and 18-second deadline; no retraining or predicate relaxation.',
    'A smoother requested increment is not a guaranteed physical acceleration or speed limit.',
    'Control observations are sampled at 50 Hz; native physics and retained-lift predicate operate at 1 kHz.',
    'Simplified rigid box, privileged simulator state and authored phases; no physical-robot qualification.',
    'No automatic promotion. Fresh confirmation requires a strict student task gain, zero regressions/errors and failed open-jaw controls.',
]


def rollout_retention(config, seed, *, variant='baseline', movement_head=None, action_tape=None):
    if variant not in ('baseline', 'loaded_goal_smoothing'):
        raise ValueError('Unknown frozen retention variant.')
    started = time.perf_counter()
    env = ArmEnv(config)
    obs, reset_info = env.reset(seed=seed)
    setup = time.perf_counter()-started
    proxy_type = SmoothedLoadedGoalProxy if variant == 'loaded_goal_smoothing' else GoalProxy
    proxy = proxy_type(env, MovementHead(movement_head) if movement_head is not None else None)
    policy = ArmPickPlacePolicy(proxy)
    proxy.policy = policy
    frames = [{**obs, 'time': 0., 'stage': 'observe'}]
    tape, latencies, rewards = [], [], []
    done = False
    info = reset_info
    while not done:
        began = time.perf_counter()
        if action_tape is not None:
            if not action_tape:
                raise ValueError('An open-jaw control needs its paired arm tape.')
            entry = action_tape[min(len(tape), len(action_tape)-1)]
            action, stage = {**entry['action'], 'gripper_width': .08}, entry['stage']
            pre, diagnostics = None, None
        else:
            held = policy._held_offset
            offset = (float(np.linalg.norm(np.asarray(obs['object_position'])-
                       np.asarray(obs['end_effector_position'])-held)) if held is not None else None)
            pre = {'time': float(env._elapsed), 'stage': policy.stage, 'contacts': dict(obs['grasp_contacts']),
                   'table_contacts': obs['table_contacts'], 'object_position': list(obs['object_position']),
                   'object_velocity': list(obs['object_velocity']),
                   'object_angular_velocity': list(obs['object_angular_velocity']),
                   'tcp_position': list(obs['end_effector_position']), 'gripper_width': obs['gripper_width'],
                   'pad_normal_forces_n': dict(obs['pad_normal_forces']),
                   'held_offset_m': held.tolist() if held is not None else None,
                   'relative_offset_error_m': offset,
                   'previous_retention_failure_ticks': policy._contact_loss_ticks}
            proxy.observation = obs
            action = policy(obs)
            diagnostics = {**policy.diagnostics(), 'retention_intervention': getattr(proxy, 'last_intervention', None)}
            stage = diagnostics['action_phase']
            pre['next_stage'] = policy.stage
            pre['retention_failure_ticks_after_check'] = policy._contact_loss_ticks
            pre['requested_gripper_width'] = action['gripper_width']
        obs, reward, terminated, truncated, info = env.step(action)
        latencies.append(time.perf_counter()-began)
        tape.append({'stage': stage, 'action': action})
        rewards.append(reward)
        if info['valid'] and len(obs['object_position']) == 3:
            frames.append({**obs, 'time': info['simulation_time'], 'stage': stage,
                           'target_distance': info['target_distance'], 'dwell_elapsed': info['dwell_elapsed'],
                           'lift_dwell_elapsed': info['lift_dwell_elapsed'], 'lift_seen': info['lift_seen'],
                           'controller': diagnostics, 'pre_action': pre})
        done = terminated or truncated
    elapsed = time.perf_counter()-started
    label = ('replay_open_jaw' if action_tape is not None else
             'distilled_placement' if movement_head is not None else 'placement')
    return {'seed': seed, 'policy': label, 'experiment_variant': variant,
            'success': info['success'], 'reason': info['reason'], 'final_distance': info['target_distance'],
            'simulated_seconds': info['simulation_time'], 'episode_wall_seconds': elapsed, 'setup_seconds': setup,
            'control_latency_p50_seconds': float(np.median(latencies)),
            'control_latency_p95_seconds': float(np.percentile(latencies, 95)),
            'simulated_seconds_per_wall_second': info['simulation_time']/elapsed,
            'control_steps': len(tape), 'reward': float(sum(rewards)), 'reset_info': reset_info,
            'final_info': info, 'actions': tape, 'trace': frames,
            'controller_failure': policy.failure_reason, 'controller_diagnostics': policy.diagnostics(),
            'diagnostic_loaded_goal_cap_m': None, 'diagnostic_intervention_calls': proxy.intervention_calls,
            'first_intervention_time': getattr(proxy, 'first_intervention_time', None),
            'movement_head_sha256': sha256(_canonical(movement_head)).hexdigest() if movement_head else None}


def measurement(episode):
    result = recovery_measurement(episode)
    events, curves = [], []
    seen, previous_stage = set(), None
    max_error = 0.
    first_recovery = False
    for frame in episode['trace']:
        pre = frame.get('pre_action')
        if pre is None:
            continue
        stage = frame['stage']
        if stage != previous_stage:
            events.append({'id': 'phase:' + stage, 'time': pre['time'],
                           'basis': 'pre-action observation at first command of phase',
                           'frame_time': frame['time'], 'stage': stage})
        previous_stage = stage
        if stage.startswith('recover_'):
            first_recovery = True
        error = pre.get('relative_offset_error_m')
        if not first_recovery and stage in ('lift', 'transfer') and error is not None:
            max_error = max(max_error, error)
            for threshold in (.002, .006, .012):
                identity = f'offset-{int(threshold * 1000)}mm'
                if error > threshold and identity not in seen:
                    seen.add(identity)
                    events.append({'id': identity, 'time': pre['time'],
                                   'basis': 'first sampled pre-action offset above threshold',
                                   'frame_time': frame['time'], 'stage': stage,
                                   'relative_offset_error_m': error,
                                   'contacts': pre['contacts']})
        if frame.get('lift_seen') and 'native-lift-seen' not in seen:
            seen.add('native-lift-seen')
            events.append({'id': 'native-lift-seen', 'time': frame['time'],
                           'basis': 'first post-control observation with native lift predicate true',
                           'frame_time': frame['time'], 'stage': stage})
        # An explicit 10 Hz view of 50 Hz observations; no interpolated evidence.
        if stage in ('lift', 'transfer', 'place') and round(pre['time'] / .02) % 5 == 0:
            curves.append({'time': pre['time'], 'stage': stage,
                           'relative_offset_error_m': error, 'contacts': pre['contacts'],
                           'object_z_m': pre['object_position'][2], 'tcp_z_m': pre['tcp_position'][2],
                           'table_contacts': pre['table_contacts']})
    result['retention'] = {'first_intervention_time': episode.get('first_intervention_time'),
                           'maximum_pre_recovery_loaded_offset_m': max_error,
                           'events': sorted(events, key=lambda item: item['time']),
                           'curve': curves, 'curve_cadence_seconds': .1,
                           'curve_scope': '10 Hz subset of actual 50 Hz pre-action observations'}
    return result


def summarize(results):
    student = [row for row in results if row['policy'] == 'student']
    groups = {}
    paired = {'improved': 0, 'regressed': 0, 'unchanged': 0, 'errors': 0}
    for variant in ('baseline', 'loaded_goal_smoothing'):
        groups[variant] = {}
        for group in ('nominal', 'actuator_stress'):
            rows = [row for row in student if row['variant'] == variant and row['group'] == group]
            groups[variant][group] = {
                'planned': len(rows), 'successes': sum(row.get('measurement', {}).get('success') is True for row in rows),
                'errors': sum(row['status'] != 'completed' or row.get('control_status') != 'completed' for row in rows),
                'control_successes': sum(row.get('control', {}).get('success') is True for row in rows)}
    for identity in sorted({row['case_id'] for row in student}):
        pair = {row['variant']: row for row in student if row['case_id'] == identity}
        if set(pair) != {'baseline', 'loaded_goal_smoothing'}:
            raise ValueError('Every planned regression needs baseline and candidate results.')
        if any(row['status'] != 'completed' or row.get('control_status') != 'completed' for row in pair.values()):
            paired['errors'] += 1
            continue
        before, after = (pair[v]['measurement']['success'] for v in ('baseline', 'loaded_goal_smoothing'))
        paired['improved' if after and not before else 'regressed' if before and not after else 'unchanged'] += 1
    controls_pass = any(row.get('control', {}).get('success') is True for row in results)
    errors = any(row['status'] != 'completed' or row.get('control_status') != 'completed' for row in results)
    gate = paired['improved'] > 0 and paired['regressed'] == 0 and not controls_pass and not errors
    return {'student': {**groups, 'paired': paired}, 'teacher': [
        {'variant': row['variant'], 'status': row['status'], 'success': row.get('measurement', {}).get('success')}
        for row in results if row['policy'] == 'teacher'],
        'fresh_confirmation_gate_met': gate,
        'promotion': 'not promoted; independent confirmation required' if gate else 'not promoted; no demonstrated task-completion improvement'}


def run(benchmark, output):
    started = time.perf_counter()
    benchmark = Path(benchmark).resolve()
    bundle = load_benchmark(benchmark)
    benchmark_protocol, benchmark_report = bundle['protocol'], bundle['report']
    cases = benchmark_protocol['scenes']
    if len(cases) != 16 or {row['id'] for row in cases} != {f'scene-{i:02}' for i in range(16)}:
        raise ValueError('Expected the complete original 16-case improvement regression suite.')
    candidate = next(row for row in benchmark_report['comparisons'] if row['cohort_id'] == 'baseline4')
    raw_capsule = (benchmark/candidate['capsule_path']).read_bytes()
    if sha256(raw_capsule).hexdigest() != candidate['capsule_file_sha256']:
        raise ValueError('Frozen baseline movement weights changed.')
    runtime = capsule_runtime()
    capsule = inspect_capsule(raw_capsule, expected_runtime=runtime)
    source, grounding = source_bytes(capsule), capsule['grounding']
    if 'simulation_grounding' in json.loads(source):
        grounding = None
    skill = load_skillspace(source, grounding)
    original_raw = {f"student-{case['id']}": (benchmark/'qualification'/'baseline4'/case['id']/'run.json').read_bytes()
                    for case in cases}
    original_raw['teacher-scene-15'] = (benchmark/'teacher'/'scene-15'/'run.json').read_bytes()
    originals = {key: json.loads(raw)['episodes'][0] for key, raw in original_raw.items()}
    output = _fresh(output)
    (output/'baseline.skill-capsule.json').write_bytes(raw_capsule)
    (output/'originals').mkdir()
    for key, raw in original_raw.items():
        (output/'originals'/f'{key}.json').write_bytes(raw)
    names = ('retention_policy.py', 'retention_experiment.py', 'grip_diagnostic.py',
             'controlled_recovery_release_experiment.py', 'controlled_recovery_policy.py',
             'controlled_recovery_release_policy.py')
    source_hashes = {**runtime['files_sha256'], **{name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}}
    (output/'sources').mkdir()
    for name in source_hashes:
        (output/'sources'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    # Known failing baseline reproduction comes first; the full suite was frozen
    # before this ordering executes any model or candidate condition.
    attempts = [
        {'case_id': 'scene-15', 'policy': name, 'variant': variant}
        for variant in ('baseline', 'loaded_goal_smoothing') for name in ('student', 'teacher')]
    attempts += [{'case_id': case['id'], 'policy': 'student', 'variant': variant}
                 for case in cases if case['id'] != 'scene-15' for variant in ('baseline', 'loaded_goal_smoothing')]
    protocol = {'schema_version': 1, 'kind': 'dvidia.retention-experiment', 'scope': 'simulation-only',
                'physical_robot_ready': False, 'retention_revision': 'increment-smoothing-v1',
                'benchmark_report_sha256': sha256((benchmark/'report.json').read_bytes()).hexdigest(),
                'benchmark_protocol_sha256': sha256((benchmark/'protocol.json').read_bytes()).hexdigest(),
                'capsule_file_sha256': sha256(raw_capsule).hexdigest(), 'runtime': runtime,
                'source_sha256': source_hashes,
                'original_run_sha256': {key: sha256(raw).hexdigest() for key, raw in original_raw.items()},
                'cases': cases, 'variants': ['baseline', 'loaded_goal_smoothing'], 'attempts': attempts,
                'planned_native_episodes': len(attempts)*2, 'settings': SETTINGS,
                'confirmation_gate': {'strict_task_success_gain': True, 'allowed_regressions': 0,
                                     'allowed_errors': 0, 'allowed_control_successes': 0,
                                     'confirmation_required_before_promotion': True},
                'limits': LIMITATIONS}
    protocol_hash = freeze_protocol(output, protocol)
    def verify():
        verify_protocol(output, protocol)
        if capsule_runtime() != runtime or any(sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected
                                               for name, expected in source_hashes.items()):
            raise ValueError('Frozen experiment source or runtime changed.')
        if sha256((output/'baseline.skill-capsule.json').read_bytes()).hexdigest() != protocol['capsule_file_sha256']:
            raise ValueError('Frozen movement head changed.')
        for key, expected in protocol['original_run_sha256'].items():
            if sha256((output/'originals'/f'{key}.json').read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen original regression trajectory changed.')
    index = {case['id']: case for case in cases}
    results, completed, errors = [], 0, 0
    with NetworkGuard(True) as guard:
        for attempt in attempts:
            verify()
            case, name, variant = index[attempt['case_id']], attempt['policy'], attempt['variant']
            scene = case['scene']
            config = replace(config_for(skill, scene), horizon=18.)
            directory = output/variant/name/case['id']
            directory.mkdir(parents=True)
            row = {**attempt, 'group': case['group'], 'status': 'completed',
                   'horizon': 18., 'scene_sha256': case['scene_sha256'],
                   'run_path': str(directory.relative_to(output)/'run.json'),
                   'exact_original_trajectory_reproduced': None}
            episode = None
            try:
                episode = rollout_retention(config, scene['seed'], variant=variant,
                    movement_head=capsule['movement_head'] if name == 'student' else None)
                payload = payload_for(skill, config, [episode], {'offline': True, 'scope': 'parent NetworkGuard'})
                payload.update(diagnostic_only=True, experiment_variant=variant)
                _write(directory/'run.json', payload)
                save_replay(payload, directory/'replay.html')
                row['measurement'] = measurement(episode)
                if variant == 'baseline':
                    row['exact_original_trajectory_reproduced'] = _same_trajectory(episode, originals[f"{name}-{case['id']}"])
                    if not row['exact_original_trajectory_reproduced']:
                        raise ValueError('Baseline does not reproduce its original actions/outcomes.')
                else:
                    original = originals[f"{name}-{case['id']}"]
                    first = next((i for i, frame in enumerate(episode['trace'][1:])
                                  if ((frame.get('controller') or {}).get('retention_intervention') or {}).get('active')), len(episode['actions']))
                    row['identical_before_intervention'] = episode['actions'][:first] == original['actions'][:first]
                    if not row['identical_before_intervention']:
                        raise ValueError('Candidate changed commands before its declared intervention.')
                completed += 1
            except (ValueError, RuntimeError) as exc:
                errors += 1
                row.update(status='error', error=str(exc))
                if variant == 'baseline':
                    _write(directory/'error.json', row)
                    raise
            try:
                if episode is None:
                    raise ValueError('No action tape for paired open-jaw control.')
                verify()
                control = rollout_retention(config, scene['seed'], action_tape=episode['actions'])
                payload = payload_for(skill, config, [control], {'offline': True, 'scope': 'parent NetworkGuard'})
                payload['diagnostic_only'] = True
                _write(directory/'control.json', payload)
                row.update(control=measurement(control), control_status='completed')
                completed += 1
            except (ValueError, RuntimeError) as exc:
                errors += 1
                row.update(control_status='error', control_error=str(exc))
            results.append(row)
            _write(output/'progress.json', {'protocol_sha256': protocol_hash, 'results': results})
            print(json.dumps({'case': case['id'], 'policy': name, 'variant': variant, 'status': row['status'],
                              'success': row.get('measurement', {}).get('success'),
                              'offset_m': row.get('measurement', {}).get('retention', {}).get('maximum_pre_recovery_loaded_offset_m')}), flush=True)
    verify()
    summary = summarize(results)
    report = {'schema_version': 1, 'kind': 'dvidia.retention-report', 'scope': 'simulation-only',
              'physical_robot_ready': False, 'retention_revision': 'increment-smoothing-v1', 'protocol_sha256': protocol_hash,
              'results': results, 'summary': summary, 'completed_native_episodes': completed,
              'episode_errors': errors, 'elapsed_seconds': time.perf_counter()-started,
              'cost': {'external_api_usd': 0., 'local_compute': 'unpriced'}, 'network_guard': guard.record(),
              'timing_scope': 'Input validation, snapshots, all rollouts and receipt/replay writes; excludes startup and final report write.',
              'limits': LIMITATIONS}
    _write(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.benchmark, args.output)
    print(json.dumps({key: result[key] for key in ('summary', 'completed_native_episodes', 'episode_errors', 'elapsed_seconds', 'cost')}))


if __name__ == '__main__':
    main()
