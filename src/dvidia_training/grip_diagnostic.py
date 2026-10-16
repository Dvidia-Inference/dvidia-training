"""Frozen, offline one-factor contact/recovery diagnostics; never qualification.

Reuse a measured movement head and exact scene. The optional loaded-goal cap
is explicit diagnostic-only instrumentation. Production controller defaults,
success predicates, source benchmark receipts and learned weights are unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math
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
from .improvement import load_benchmark
from .skill_capsule import inspect_capsule, source_bytes
from .skillspace import load_skillspace


ARMS = (
    {'id': 'original', 'scene_overrides': {}, 'horizon': 18., 'loaded_goal_cap_m': None},
    {'id': 'jaw-060', 'scene_overrides': {'jaw_force_limit': .6}, 'horizon': 18., 'loaded_goal_cap_m': None},
    {'id': 'jaw-080', 'scene_overrides': {'jaw_force_limit': .8}, 'horizon': 18., 'loaded_goal_cap_m': None},
    {'id': 'torque-40', 'scene_overrides': {'joint_torque_limit': 40.}, 'horizon': 18., 'loaded_goal_cap_m': None},
    {'id': 'loaded-2500um', 'scene_overrides': {}, 'horizon': 18., 'loaded_goal_cap_m': .0025},
    {'id': 'loaded-1250um', 'scene_overrides': {}, 'horizon': 18., 'loaded_goal_cap_m': .00125},
    {'id': 'horizon-30', 'scene_overrides': {}, 'horizon': 30., 'loaded_goal_cap_m': None},
)
LIMITATIONS = [
    'Exact previously inspected simulated scene: diagnosis, not an independent generalization test.',
    'No retraining; privileged state, simplified rigid box, authored phases/grip/recovery and collision-excluded arm links.',
    'Jaw-force arms change actuator capability and the existing supervisor response to that capability.',
    'Loaded-goal cap changes requested Cartesian tracking distance; it is not a physical speed or acceleration guarantee.',
    'A longer-horizon success does not repair the original 18-second benchmark failure.',
    'Open-jaw controls replay each intervention arm trajectory; no physical-robot or broad causal qualification is issued.',
    'Control observations are 50 Hz. Bilateral contact in these samples does not establish uninterrupted 1-kHz native contact or the required lift dwell.',
]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b'\n')


def validate_arm(arm):
    if type(arm) is not dict or set(arm) != {'id', 'scene_overrides', 'horizon', 'loaded_goal_cap_m'}:
        raise ValueError('A diagnostic arm needs exactly its identity, scene overrides, horizon and loaded-goal cap.')
    if arm['id'] not in {row['id'] for row in ARMS} or arm != next(row for row in ARMS if row['id'] == arm['id']):
        raise ValueError('Use only a frozen one-factor diagnostic arm.')
    return deepcopy(arm)


class GoalProxy:
    """Apply one loaded-motion intervention before unchanged IK/head mapping."""
    def __init__(self, environment, head=None, loaded_goal_cap_m=None):
        if loaded_goal_cap_m is not None and (type(loaded_goal_cap_m) not in (int, float)
                or not math.isfinite(loaded_goal_cap_m) or not .0005 <= loaded_goal_cap_m <= .006):
            raise ValueError('Loaded-goal cap must be finite and within [0.0005, 0.006] m.')
        self.environment, self.head, self.loaded_goal_cap_m = environment, head, loaded_goal_cap_m
        self.policy, self.observation = None, None
        self.intervention_calls = 0

    def __getattr__(self, key):
        return getattr(self.environment, key)

    def joint_targets_for_pose(self, goal):
        if self.loaded_goal_cap_m is not None and self.policy._action_phase in ('lift', 'transfer', 'place'):
            tcp = np.asarray(self.observation['end_effector_position'])
            delta = np.asarray(goal) - tcp
            distance = float(np.linalg.norm(delta))
            if distance > self.loaded_goal_cap_m:
                goal = tcp + delta * (self.loaded_goal_cap_m / distance)
                self.intervention_calls += 1
        return (self.head.predict(self.observation, goal) if self.head is not None
                else self.environment.joint_targets_for_pose(goal))


def rollout_diagnostic(config, seed, *, movement_head=None, loaded_goal_cap_m=None, action_tape=None):
    """Record full 50-Hz observations plus native 1-kHz force/velocity peaks."""
    started = time.perf_counter()
    env = ArmEnv(config)
    obs, reset_info = env.reset(seed=seed)
    setup = time.perf_counter() - started
    head = MovementHead(movement_head) if movement_head is not None else None
    proxy = GoalProxy(env, head, loaded_goal_cap_m)
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
                raise ValueError('Open-jaw controls require a nonempty recorded action tape.')
            entry = action_tape[min(len(tape), len(action_tape)-1)]
            action, stage = {**entry['action'], 'gripper_width': .08}, entry['stage']
            diagnostic = None
            pre_action = None
        else:
            held = policy._held_offset
            offset_error = (float(np.linalg.norm(np.asarray(obs['object_position']) -
                              np.asarray(obs['end_effector_position']) - held)) if held is not None else None)
            pre_action = {'time': float(env._elapsed), 'stage': policy.stage,
                          'contacts': dict(obs['grasp_contacts']), 'table_contacts': obs['table_contacts'],
                          'object_position': list(obs['object_position']),
                          'object_velocity': list(obs['object_velocity']),
                          'tcp_position': list(obs['end_effector_position']),
                          'pad_normal_forces_n': dict(obs['pad_normal_forces']),
                          'held_offset_m': held.tolist() if held is not None else None,
                          'relative_offset_error_m': offset_error,
                          'previous_retention_failure_ticks': policy._contact_loss_ticks}
            proxy.observation = obs
            action = policy(obs)
            diagnostic = policy.diagnostics()
            pre_action['next_stage'] = policy.stage
            pre_action['retention_failure_ticks_after_check'] = policy._contact_loss_ticks
            stage = diagnostic['action_phase']
        obs, reward, terminated, truncated, info = env.step(action)
        latencies.append(time.perf_counter() - began)
        rewards.append(reward)
        tape.append({'stage': stage, 'action': action})
        if info['valid'] and len(obs['object_position']) == 3:
            frames.append({**obs, 'time': info['simulation_time'], 'stage': stage,
                           'target_distance': info['target_distance'], 'dwell_elapsed': info['dwell_elapsed'],
                           'lift_dwell_elapsed': info['lift_dwell_elapsed'], 'lift_seen': info['lift_seen'],
                           'controller': diagnostic, 'pre_action': pre_action})
        done = terminated or truncated
    elapsed = time.perf_counter() - started
    label = ('replay_open_jaw' if action_tape is not None else
             'distilled_placement' if movement_head is not None else 'placement')
    return {'seed': seed, 'policy': label, 'success': info['success'], 'reason': info['reason'],
            'final_distance': info['target_distance'], 'simulated_seconds': info['simulation_time'],
            'episode_wall_seconds': elapsed, 'setup_seconds': setup,
            'control_latency_p50_seconds': float(np.median(latencies)),
            'control_latency_p95_seconds': float(np.percentile(latencies, 95)),
            'simulated_seconds_per_wall_second': info['simulation_time'] / elapsed,
            'control_steps': len(tape), 'reward': float(sum(rewards)), 'reset_info': reset_info,
            'final_info': info, 'actions': tape, 'trace': frames,
            'controller_failure': policy.failure_reason, 'controller_diagnostics': policy.diagnostics(),
            'diagnostic_loaded_goal_cap_m': loaded_goal_cap_m,
            'diagnostic_intervention_calls': proxy.intervention_calls,
            'movement_head_sha256': sha256(_canonical(movement_head)).hexdigest() if movement_head else None}


def summarize_episode(episode):
    frames = episode['trace']
    events, previous = [], None
    reference = None
    max_offset = 0.
    first_loss = None
    retention_triggers = []
    first_threshold_breach = None
    loaded_bilateral_samples = 0
    loaded_samples = 0
    for frame in frames:
        stage = frame['stage']
        if stage != previous:
            events.append({'stage': stage, 'time': frame['time'], 'object_position': frame['object_position'],
                           'contacts': frame['grasp_contacts'], 'target_distance': frame.get('target_distance')})
            previous = stage
        loaded = stage in ('lift', 'transfer')
        pre = frame.get('pre_action')
        if pre and pre['stage'] in ('lift', 'transfer'):
            loaded_samples += 1
            both = pre['contacts']['left'] > 0 and pre['contacts']['right'] > 0
            loaded_bilateral_samples += bool(both)
            exceeds = pre['relative_offset_error_m'] is not None and pre['relative_offset_error_m'] > .012
            if first_threshold_breach is None and (not both or exceeds):
                first_threshold_breach = pre
            if pre['next_stage'] in ('recover_wait', 'failed'):
                retention_triggers.append(pre)
        if stage == 'lift' and (reference is None or (len(events) > 1 and events[-1]['time'] == frame['time'])):
            reference = np.asarray(frame['object_position']) - np.asarray(frame['end_effector_position'])
        if loaded and reference is not None:
            offset = float(np.linalg.norm(np.asarray(frame['object_position']) - np.asarray(frame['end_effector_position']) - reference))
            max_offset = max(max_offset, offset)
            contacts = frame['grasp_contacts']
            if first_loss is None and (contacts['left'] == 0 or contacts['right'] == 0 or offset > .012):
                first_loss = {'time': frame['time'], 'stage': stage, 'offset_change_m': offset,
                              'contacts': contacts, 'pad_normal_forces_n': frame['pad_normal_forces'],
                              'object_position': frame['object_position']}
    final = episode['final_info']
    diagnostics = episode['controller_diagnostics']
    return {'success': episode['success'], 'reason': episode['reason'],
            'final_target_error_m': episode['final_distance'], 'simulated_seconds': episode['simulated_seconds'],
            'episode_wall_seconds': episode['episode_wall_seconds'], 'control_steps': episode['control_steps'],
            'first_loaded_contact_or_offset_loss': first_loss, 'maximum_loaded_offset_change_m': max_offset,
            'first_pre_action_retention_threshold_breach': first_threshold_breach,
            'retention_transitions': retention_triggers,
            'loaded_pre_action_samples': loaded_samples,
            'loaded_pre_action_bilateral_samples': loaded_bilateral_samples,
            'recovery_count': diagnostics['recovery_count'], 'recovery_reason': diagnostics['recovery_reason'],
            'controller_failure': episode['controller_failure'], 'stage_events': events,
            'placement_dwell_seconds': final['dwell_elapsed'], 'lift_seen': final['lift_seen'],
            'valid_final_state': final['valid'],
            'maximum_sampled_lift_dwell_seconds': max(frame.get('lift_dwell_elapsed', 0.) for frame in frames),
            'contact_observation_cadence_seconds': final['config']['control_dt'],
            'maximum_object_lift_m': final['max_object_lift'], 'native_tick_peaks': final['native_tick_peaks'],
            'grip_demand_n': diagnostics['grip_normal_demand_n'], 'grip_target_n': diagnostics['grip_normal_target_n'],
            'intervention_calls': episode['diagnostic_intervention_calls']}


def _same_trajectory(episode, original):
    """Reject a diagnostic harness whose no-op arm changes the original motion."""
    fields = ('success', 'reason', 'final_distance', 'simulated_seconds', 'control_steps', 'actions')
    return all(episode[key] == original[key] for key in fields)


def run(benchmark, output):
    started = time.perf_counter()
    benchmark = Path(benchmark).resolve()
    bundle = load_benchmark(benchmark)
    # The first pilot intentionally freezes one named cohort and failed case.
    protocol = bundle['protocol']
    case = next(row for row in protocol['scenes'] if row['id'] == 'scene-15')
    if case['group'] != 'actuator_stress':
        raise ValueError('This diagnostic targets the recorded actuator-stress failure only.')
    report = bundle['report']
    candidate = next(row for row in report['comparisons'] if row['cohort_id'] == 'baseline4')
    capsule_path = benchmark / candidate['capsule_path']
    capsule_raw = capsule_path.read_bytes()
    if sha256(capsule_raw).hexdigest() != candidate['capsule_file_sha256']:
        raise ValueError('Existing baseline capsule differs from its frozen receipt.')
    runtime = capsule_runtime()
    capsule = inspect_capsule(capsule_raw, expected_runtime=runtime)
    source, grounding = source_bytes(capsule), capsule['grounding']
    if 'simulation_grounding' in json.loads(source):
        grounding = None
    skill = load_skillspace(source, grounding)
    original_paths = {'student': 'qualification/baseline4/scene-15/run.json', 'teacher': 'teacher/scene-15/run.json'}
    original_raw = {name: (benchmark / relative).read_bytes() for name, relative in original_paths.items()}
    originals = {name: json.loads(raw)['episodes'][0] for name, raw in original_raw.items()}
    if any(row['success'] for row in originals.values()):
        raise ValueError('Both recorded policies must have failed the source scene.')
    arms = [validate_arm(row) for row in ARMS]
    output = _fresh(output)
    (output / 'baseline.skill-capsule.json').write_bytes(capsule_raw)
    source_hashes = {**runtime['files_sha256'], 'grip_diagnostic.py': sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'sources').mkdir()
    for name in source_hashes:
        (output/'sources'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (output/'originals').mkdir()
    for name, raw in original_raw.items():
        (output/'originals'/f'{name}.json').write_bytes(raw)
    frozen = {'schema_version': 1, 'kind': 'dvidia.grip-recovery-diagnostic', 'scope': 'simulation-only',
              'physical_robot_ready': False, 'benchmark_protocol_sha256': sha256((benchmark/'protocol.json').read_bytes()).hexdigest(),
              'benchmark_report_sha256': sha256((benchmark/'report.json').read_bytes()).hexdigest(),
              'capsule_file_sha256': sha256(capsule_raw).hexdigest(), 'runtime': runtime,
              'original_run_sha256': {name: sha256(raw).hexdigest() for name, raw in original_raw.items()},
              'source_sha256': source_hashes, 'case': case, 'arms': arms,
              'policies': ['student', 'teacher'], 'planned_native_episodes': len(arms) * 4,
              'rules': {'same_model': True, 'retrain': False, 'qualification': False,
                        'negative_control': 'Each policy/arm tape is replayed with jaws forced open.',
                        'completion': 'Record every planned policy and control attempt, including errors.',
                        'baseline_reproduction': 'Require exact action tape and outcome parity for original teacher and student.'},
              'limits': LIMITATIONS}
    protocol_hash = freeze_protocol(output, frozen)
    def verify():
        verify_protocol(output, frozen)
        if capsule_runtime() != runtime or any(sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest
                                               for name, digest in source_hashes.items()):
            raise ValueError('Frozen diagnostic source or runtime changed.')
        if sha256((output/'baseline.skill-capsule.json').read_bytes()).hexdigest() != frozen['capsule_file_sha256']:
            raise ValueError('Frozen movement head changed.')
        for name, expected in frozen['original_run_sha256'].items():
            if sha256((output/'originals'/f'{name}.json').read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen original trajectory changed.')
    results, episodes_completed, episode_errors = [], 0, 0
    with NetworkGuard(True) as guard:
        for arm in arms:
            for name in frozen['policies']:
                verify()
                scene = {**case['scene'], **arm['scene_overrides']}
                config = replace(config_for(skill, scene), horizon=arm['horizon'])
                directory = output / arm['id'] / name
                directory.mkdir(parents=True)
                row = {'arm_id': arm['id'], 'policy': name, 'scene': scene,
                       'horizon': arm['horizon'], 'loaded_goal_cap_m': arm['loaded_goal_cap_m'],
                       'status': 'completed', 'qualification': 'diagnostic-only', 'run_path': str(directory.relative_to(output)/'run.json')}
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
                    episodes_completed += 1
                    if arm['id'] == 'original':
                        row['exact_original_trajectory_reproduced'] = _same_trajectory(episode, originals[name])
                        if not row['exact_original_trajectory_reproduced']:
                            raise ValueError('No-op diagnostic failed to reproduce original actions/outcomes; stop before interventions.')
                except (ValueError, RuntimeError) as exc:
                    episode_errors += 1
                    row.update(status='error', error=str(exc))
                    if arm['id'] == 'original':
                        _write(directory/'error.json', row)
                        raise
                try:
                    if episode is None:
                        raise ValueError('No candidate action tape; dependent open-jaw control cannot execute.')
                    verify()
                    control = rollout_diagnostic(config, scene['seed'], action_tape=episode['actions'])
                    control_payload = payload_for(skill, config, [control], {'offline': True, 'scope': 'parent NetworkGuard'})
                    control_payload['diagnostic_only'] = True
                    _write(directory/'control.json', control_payload)
                    row['control'] = summarize_episode(control)
                    row['control_status'] = 'completed'
                    episodes_completed += 1
                except (ValueError, RuntimeError) as exc:
                    episode_errors += 1
                    row.update(control_status='error', control_error=str(exc))
                results.append(row)
                _write(output/'progress.json', {'protocol_sha256': protocol_hash, 'results': results})
                print(json.dumps({'arm': arm['id'], 'policy': name, 'status': row['status'],
                                  'success': row.get('measurement', {}).get('success'),
                                  'reason': row.get('measurement', {}).get('reason'),
                                  'simulated_seconds': row.get('measurement', {}).get('simulated_seconds'),
                                  'recoveries': row.get('measurement', {}).get('recovery_count')}), flush=True)
    verify()
    result = {'schema_version': 1, 'kind': 'dvidia.grip-recovery-diagnostic-result', 'scope': 'simulation-only',
              'physical_robot_ready': False, 'protocol_sha256': protocol_hash,
              'results': results, 'completed_native_episodes': episodes_completed, 'episode_errors': episode_errors,
              'elapsed_seconds': time.perf_counter()-started, 'network_guard': guard.record(),
              'cost': {'external_api_usd': 0., 'local_compute': 'unpriced'},
              'timing_scope': 'Validation, snapshot, execution, telemetry/replay writes; excludes process startup and final report write.',
              'limits': LIMITATIONS}
    _write(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.benchmark, args.output)
    print(json.dumps({key: result[key] for key in ('completed_native_episodes', 'episode_errors', 'elapsed_seconds', 'cost')}))


if __name__ == '__main__':
    main()
