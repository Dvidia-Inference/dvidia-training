"""Passive native-tick replay audit; no controller, physics or scoring changes.

This diagnoses the already inspected retention tapes. It neither generates new
actions nor qualifies a learned or physical robot policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import io
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from .arm_env import ArmConfig, ArmEnv
from .arm_policy import PHASE_TIMEOUTS
from .cli import NetworkGuard
from .installer import runtime_contract

SCHEMA = 'dvidia.native-contact-audit/v1'
LOADED_PHASES = ('lift', 'transfer', 'place')
CONTACT_THRESHOLD_N = .01
LIFT_HEIGHT_M = .055
LIFT_DWELL_SECONDS = .1
LIMITATIONS = [
    'Replays of previously inspected synthetic cases, not new policy trials or held-out generalization.',
    'Fixed recorded actions; no controller improvement, model promotion or physical robot qualification.',
    'Simplified rigid box and authored arm; this is separate from PiPER and shoe geometry studies.',
    'Collision candidates, positive force, and the unchanged per-contact 0.01 N scoring threshold are distinct.',
    'Native evidence describes this exact simulator and does not validate physical contact parameters.',
    'Loaded-phase and held-offset labels are taken from the frozen original controller tape; no new supervisor runs.',
    'Recorded open-jaw controls use paired arm commands and different dynamics, not counterfactual identical states.',
    'Wall time includes passive instrumentation and file export; it is not uninstrumented simulation throughput.',
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b'\n')


def read_frozen_input(root, plan):
    expected_path = 'inputs/' + plan['id'] + '.json.gz'
    if plan['input_path'] != expected_path or Path(plan['id']).name != plan['id']:
        raise ValueError('Frozen input path must be its declared local episode ID.')
    compressed = (Path(root) / expected_path).read_bytes()
    assert_equal(sha256(compressed).hexdigest(), plan['compressed_input_sha256'], 'compressed input ' + plan['id'])
    if len(compressed) > 32 * 1024 * 1024:
        raise ValueError('A frozen compressed input exceeds the 32 MiB audit bound.')
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError('A decompressed input exceeds the 32 MiB audit bound.')
    assert_equal(sha256(raw).hexdigest(), plan['input_sha256'], 'decompressed input ' + plan['id'])
    return raw


def assert_equal(actual, expected, context):
    """Exact JSON-value equality; no tolerance can hide a changed replay."""
    if canonical(actual) != canonical(expected):
        raise ValueError(f'Exact reproduction mismatch: {context}')


def contact_classification(raw_normal_forces, separation_m):
    """Describe force threshold and geometry separately, without rescoring."""
    if any(force > CONTACT_THRESHOLD_N for force in raw_normal_forces):
        return 'scored_contact'
    if any(force > 0 for force in raw_normal_forces):
        return 'positive_force_below_threshold'
    if separation_m > 0:
        return 'geometrically_separated'
    if raw_normal_forces:
        return 'candidate_without_positive_force'
    return 'no_scored_contact_unclassified'


class TickRecorder:
    """Reject gaps/reversals and ignore repeat reads of the same native tick."""
    def __init__(self, timestep):
        self.timestep = float(timestep)
        self.last_time = 0.
        self.rows = []
        self.dwell = 0.
        self.lift_seen = False
        self.sink = None

    def wants(self, elapsed):
        if elapsed == self.last_time:
            return False
        if elapsed < self.last_time or not math.isclose(
                elapsed - self.last_time, self.timestep, rel_tol=0, abs_tol=1e-10):
            raise ValueError('Native contact timestamps reverse or skip a physics tick.')
        return True

    def append(self, row):
        if not self.wants(row['time']):
            return False
        failures = []
        for side in ('left', 'right'):
            if row['pads'][side]['scored_count'] == 0:
                failures.append(side + '_contact')
        if row['table']['scored_count']:
            failures.append('table_support')
        if row['object_lift_m'] < LIFT_HEIGHT_M:
            failures.append('lift_height')
        previous = self.dwell
        self.dwell = self.dwell + self.timestep if not failures else 0.
        self.lift_seen |= self.dwell + 1e-12 >= LIFT_DWELL_SECONDS
        row.update(native_tick=len(self.rows) + 1,
                   retained_lift_predicate=not failures,
                   retained_lift_failure_reasons=failures,
                   retained_lift_dwell_seconds=self.dwell,
                   retained_lift_seen=self.lift_seen,
                   interrupted_dwell_seconds=previous if failures else 0.)
        self.rows.append(row)
        self.last_time = row['time']
        if self.sink is not None:
            self.sink(row)
        return True


class AuditedArmEnv(ArmEnv):
    """Observe the existing pure contact-read seam after each native step.

    No integration, actuator, observation, success or controller method is
    replaced. The logger never writes model/data arrays or issues commands.
    """
    def __init__(self, config):
        self.recorder = None
        self.action_context = None
        super().__init__(config)

    def reset(self, *args, **kwargs):
        self.recorder = None  # exclude authored reset/settling and its reads
        result = super().reset(*args, **kwargs)
        self.recorder = TickRecorder(self.config.timestep)
        return result

    def _contact_state(self):
        counts, forces = super()._contact_state()
        recorder = self.recorder
        if recorder is not None and recorder.wants(float(self._elapsed)):
            if self.action_context is None:
                raise ValueError('A recorded action context is required before native replay.')
            recorder.append(self._native_record(counts, forces))
        return counts, forces

    def _native_record(self, counts, forces):
        object_geom = self.geom_ids['object_geom']
        mapping = {'left': self.geom_ids['left_pad_geom'],
                   'right': self.geom_ids['right_pad_geom'], 'table': self.geom_ids['table']}
        contacts = {side: [] for side in mapping}
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            if object_geom not in pair:
                continue
            for side, geom in mapping.items():
                if geom in pair:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, index, force)
                    contacts[side].append({
                        'distance_m': float(contact.dist),
                        'constraint_address': int(contact.efc_address),
                        'force_contact_frame': force.tolist(),
                        'force_units': 'N,N,N,N*m,N*m,N*m',
                    })
        measured = {}
        for side, geom in mapping.items():
            separation = float(mujoco.mj_geomDistance(
                self.model, self.data, object_geom, geom, 10., np.zeros(6)))
            raw = [contact['force_contact_frame'][0] for contact in contacts[side]]
            measured[side] = {
                'candidate_count': len(raw), 'positive_force_count': sum(value > 0 for value in raw),
                'scored_count': int(counts[side]), 'scored_normal_force_n': float(forces[side]),
                'raw_positive_normal_force_n': sum(max(0., value) for value in raw),
                'minimum_geometry_distance_m': separation,
                'classification': contact_classification(raw, separation), 'contacts': contacts[side],
            }
            assert_equal(sum(value > CONTACT_THRESHOLD_N for value in raw), counts[side],
                         'native contact threshold agrees with unchanged scorer')
        obj = self.data.xpos[self.object_id].copy()
        tcp = self.data.site_xpos[self.tcp_id].copy()
        relative = obj - tcp
        held = self.action_context.get('held_offset_m')
        error = None if held is None else relative - np.asarray(held)
        return {
            'time': float(self._elapsed), 'native_engine_time': float(self.data.time),
            **self.action_context,
            'pads': {side: measured[side] for side in ('left', 'right')}, 'table': measured['table'],
            'object_position_m': obj.tolist(), 'tcp_position_m': tcp.tolist(),
            'object_quaternion': self.data.xquat[self.object_id].tolist(),
            'object_velocity_world': self._velocity(mujoco.mjtObj.mjOBJ_BODY, self.object_id).tolist(),
            'tcp_velocity_world': self._velocity(mujoco.mjtObj.mjOBJ_SITE, self.tcp_id).tolist(),
            'velocity_order': 'angular_rad_s_then_linear_m_s',
            'object_lift_m': float(obj[2] - self.initial_object_position[2]),
            'object_to_tcp_m': relative.tolist(),
            'relative_offset_error_m': None if error is None else error.tolist(),
            'relative_offset_error_norm_m': None if error is None else float(np.linalg.norm(error)),
            'jaw_position_m': self.data.qpos[self.jaw_qpos].tolist(),
            'jaw_velocity_m_s': self.data.qvel[self.model.jnt_dofadr[self.jaw_ids]].tolist(),
            'jaw_actuator_force_n': self.data.actuator_force[6:].tolist(),
            'joint_actuator_torque_nm': self.data.actuator_force[:6].tolist(),
        }


def compare_sample(observation, info, frame, index):
    # Every physical observation field is compared, including forces and geometry.
    for key, value in observation.items():
        if key not in frame:
            raise ValueError(f'Missing frozen physical field at sample {index}: {key}')
        assert_equal(value, frame[key], f'sample {index} / {key}')
    assert_equal(info['simulation_time'], frame['time'], f'sample {index} / time')
    if index:
        for key in ('target_distance', 'dwell_elapsed', 'lift_dwell_elapsed', 'lift_seen'):
            if key not in frame:
                raise ValueError(f'Missing frozen scoring field at sample {index}: {key}')
            assert_equal(info[key], frame[key], f'sample {index} / {key}')


def replay_episode(original, *, trace_sink=None):
    config = ArmConfig(**original['final_info']['config'])
    if config.timestep != .001 or config.control_dt != .02 or config.horizon != 18.:
        raise ValueError('This audit is frozen to the original 1 kHz / 50 Hz / 18 second contract.')
    if len(original['trace']) != len(original['actions']) + 1:
        raise ValueError('Frozen tape requires one initial and one physical frame per command.')
    env = AuditedArmEnv(config)
    observation, info = env.reset(seed=original['seed'])
    env.recorder.sink = trace_sink
    assert_equal(info, original['reset_info'], 'reset information')
    compare_sample(observation, info, original['trace'][0], 0)
    executed, rewards = [], []
    phase, phase_start = None, 0.
    for index, entry in enumerate(original['actions']):
        frame = original['trace'][index + 1]
        if entry['stage'] != phase:
            phase, phase_start = entry['stage'], float(env._elapsed)
        assert_equal(frame['stage'], phase, f'action {index} phase')
        pre = frame.get('pre_action') or {}
        env.action_context = {
            'action_index': index, 'phase': phase, 'phase_command_started_at': phase_start,
            'held_offset_m': pre.get('held_offset_m'),
            'requested_gripper_width_m': entry['action']['gripper_width'],
        }
        observation, reward, terminated, truncated, info = env.step(entry['action'])
        executed.append(entry)
        rewards.append(reward)
        compare_sample(observation, info, frame, index + 1)
        assert_equal(env.recorder.dwell, info['lift_dwell_elapsed'], f'native dwell at sample {index + 1}')
        assert_equal(env.recorder.lift_seen, info['lift_seen'], f'native lift at sample {index + 1}')
        if (terminated or truncated) != (index == len(original['actions']) - 1):
            raise ValueError('Frozen tape and native episode termination disagree.')
    assert_equal(executed, original['actions'], 'complete action tape')
    assert_equal(info, original['final_info'], 'complete final information')
    for key, value in {'success': info['success'], 'reason': info['reason'],
                       'final_distance': info['target_distance'], 'simulated_seconds': info['simulation_time'],
                       'control_steps': len(executed), 'reward': float(sum(rewards))}.items():
        assert_equal(value, original[key], 'episode ' + key)
    return env.recorder.rows, {
        'exact_actions': True, 'exact_all_sampled_physical_fields': True,
        'exact_initial_and_final_info': True, 'exact_outcome_and_rewards': True,
        'independent_native_dwell_agrees_at_every_control_boundary': True,
        'sampled_frames': len(original['trace']), 'native_ticks': len(env.recorder.rows),
        'success': info['success'], 'reason': info['reason'], 'simulated_seconds': info['simulation_time'],
    }


def sampling_misses(rows, phase_scope=None):
    """Count 50 Hz *observed* contact windows that hide native contact losses.

    Each window requires its previous and final stored native boundary to show
    bilateral contact. No initial t=0 native record is invented.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row['action_index'], []).append(row)
    previous, eligible, missed = None, 0, []
    for index, group in groups.items():
        bilateral = lambda row: all(row['pads'][side]['scored_count'] > 0 for side in ('left', 'right'))
        phase_eligible = phase_scope is None or all(row['phase'] in phase_scope for row in group)
        if previous is not None and phase_eligible and bilateral(previous) and bilateral(group[-1]):
            eligible += 1
            missing = [row for row in group if not bilateral(row)]
            if missing:
                missed.append({'action_index': index, 'phase': group[-1]['phase'],
                               'interval_start': previous['time'], 'interval_end': group[-1]['time'],
                               'missing_native_ticks': len(missing),
                               'first_missing_time': missing[0]['time'], 'last_missing_time': missing[-1]['time']})
        previous = group[-1]
    return {'eligible_bilateral_boundary_intervals': eligible,
            'intervals_hiding_native_contact_loss': len(missed), 'intervals': missed}


def _runs(rows, predicate, timestep):
    spans, active = [], []
    for row in rows:
        if predicate(row):
            active.append(row)
        elif active:
            spans.append(active)
            active = []
    if active:
        spans.append(active)
    return [{'first_tick_time': part[0]['time'], 'last_tick_time': part[-1]['time'],
             'sampled_duration_seconds': len(part) * timestep, 'native_ticks': len(part),
             'start_phase': part[0]['phase'], 'end_phase': part[-1]['phase']}
            for part in spans]


def summarize_native(rows, original):
    dt = original['final_info']['config']['timestep']
    loaded = [row for row in rows if row['phase'] in LOADED_PHASES]
    # Include time/phase predicates instead of concatenating disjoint phases
    # when finding spans, so retries cannot silently join contact intervals.
    dropouts = _runs(rows, lambda row: row['phase'] in LOADED_PHASES and any(
        row['pads'][side]['scored_count'] == 0 for side in ('left', 'right')), dt)
    streaks = _runs(rows, lambda row: row['retained_lift_predicate'], dt)
    reasons = Counter(reason for row in loaded for reason in row['retained_lift_failure_reasons'])
    interruptions = [{'time': row['time'], 'phase': row['phase'],
                      'prior_dwell_seconds': row['interrupted_dwell_seconds'],
                      'failure_reasons': row['retained_lift_failure_reasons'],
                      'pad_classifications': {side: row['pads'][side]['classification'] for side in ('left', 'right')}}
                     for row in rows if row['interrupted_dwell_seconds'] > 0]
    phases = []
    for row in rows:
        if not phases or row['phase_command_started_at'] != phases[-1]['command_started_at']:
            phases.append({'phase': row['phase'], 'command_started_at': row['phase_command_started_at'],
                           'first_native_time': row['time'], 'last_native_time': row['time'], 'native_ticks': 0,
                           'declared_phase_timeout_seconds': PHASE_TIMEOUTS.get(row['phase']),
                           'deadline_remaining_at_start_seconds': max(0., 18. - row['phase_command_started_at'])})
        phases[-1]['last_native_time'] = row['time']
        phases[-1]['native_ticks'] += 1
    for phase in phases:
        phase['duration_seconds'] = phase['native_ticks'] * dt
        phase['deadline_remaining_at_end_seconds'] = max(0., 18. - phase['last_native_time'])
    forces = {side: {
        'classification_native_ticks': dict(Counter(row['pads'][side]['classification'] for row in loaded)),
        'scored_normal_force_min_n': min((row['pads'][side]['scored_normal_force_n'] for row in loaded), default=None),
        'scored_normal_force_max_n': max((row['pads'][side]['scored_normal_force_n'] for row in loaded), default=None),
        'raw_positive_normal_force_max_n': max((row['pads'][side]['raw_positive_normal_force_n'] for row in loaded), default=None),
        'geometric_separation_max_m': max((row['pads'][side]['minimum_geometry_distance_m'] for row in loaded), default=None),
    } for side in ('left', 'right')}
    native_lift = next((row['time'] for row in rows if row['retained_lift_seen']), None)
    return {
        'native_ticks': len(rows), 'loaded_native_ticks': len(loaded),
        'loaded_phase_scope': list(LOADED_PHASES), 'native_cadence_seconds': dt,
        'loaded_failure_reason_ticks': dict(reasons), 'loaded_pad_evidence': forces,
        'loaded_contact_dropout_spans': dropouts,
        'maximum_loaded_contact_dropout_seconds': max((span['sampled_duration_seconds'] for span in dropouts), default=0.),
        'retained_lift_streaks': streaks,
        'maximum_retained_lift_streak_seconds': max((span['sampled_duration_seconds'] for span in streaks), default=0.),
        'first_native_retained_lift_time': native_lift, 'dwell_interruptions': interruptions,
        'sampling_misses': sampling_misses(rows),
        'loaded_sampling_misses': sampling_misses(rows, LOADED_PHASES), 'phase_budgets': phases,
        'maximum_loaded_relative_offset_error_m': max((row['relative_offset_error_norm_m'] for row in loaded
                                                       if row['relative_offset_error_norm_m'] is not None), default=None),
        'maximum_absolute_jaw_actuator_force_n': max((abs(value) for row in rows for value in row['jaw_actuator_force_n']), default=None),
    }


def planned_inputs(smoothing, compliance):
    result = []
    for variant, directory, source_variant in (
            ('baseline', smoothing, 'baseline'), ('smoothing', smoothing, 'loaded_goal_smoothing'),
            ('compliance', compliance, 'vertical_compliance')):
        for policy in ('student', 'teacher'):
            result.append((f'{variant}-{policy}-scene15', Path(directory) / source_variant / policy / 'scene-15'))
    result.append(('baseline-student-scene14', Path(smoothing) / 'baseline' / 'student' / 'scene-14'))
    return [(identity + ('-open-jaw' if name == 'control.json' else ''), folder / name,
             'recorded_open_jaw_control' if name == 'control.json' else 'recorded_controller')
            for identity, folder in result for name in ('run.json', 'control.json')]


def run_audit(smoothing, compliance, output, *, from_audit=None):
    started = time.perf_counter()
    root = Path(output).resolve()
    if root.exists():
        raise ValueError('Output must be a fresh directory; completed evidence is never overwritten.')
    runtime = runtime_contract()
    plans, sources = [], {}
    for name, expected in runtime['files_sha256'].items():
        raw = Path(__file__).with_name(name).read_bytes()
        assert_equal(sha256(raw).hexdigest(), expected, 'loaded runtime source ' + name)
        sources[name] = raw
    sources[Path(__file__).name] = Path(__file__).read_bytes()
    snapshots = {}
    original_protocol_hash = None
    if from_audit is not None:
        bundle = Path(from_audit)
        previous_raw = (bundle / 'protocol.json').read_bytes()
        original_protocol_hash = sha256(previous_raw).hexdigest()
        assert_equal(original_protocol_hash, (bundle / 'protocol.sha256').read_text().strip(), 'published source protocol')
        previous = json.loads(previous_raw)
        assert_equal(previous['schema'], SCHEMA, 'published source schema')
        assert_equal({name: sha256(raw).hexdigest() for name, raw in sources.items()},
                     previous['sources_sha256'], 'published audit and runtime source hashes')
        declared = [(identity, kind) for identity, _, kind in planned_inputs('.', '.')]
        assert_equal([(row['id'], row['role']) for row in previous['attempts']], declared, 'complete published attempt list')
        input_bytes = [(row['id'], read_frozen_input(bundle, row), row['role']) for row in previous['attempts']]
        metadata = {name: (bundle / 'upstream' / name).read_bytes() for name in (
            'smoothing-protocol.json', 'smoothing-report.json', 'compliance-protocol.json', 'compliance-report.json')}
        assert_equal({name: sha256(raw).hexdigest() for name, raw in metadata.items()},
                     previous['upstream_metadata_sha256'], 'published upstream metadata')
    else:
        input_bytes = [(identity, path.read_bytes(), kind) for identity, path, kind in planned_inputs(smoothing, compliance)]
        metadata = {}
        for label, directory in (('smoothing', smoothing), ('compliance', compliance)):
            for name in ('protocol.json', 'report.json'):
                metadata[label + '-' + name] = (Path(directory) / name).read_bytes()
    for identity, raw, kind in input_bytes:
        payload = json.loads(raw)
        assert_equal(payload['runtime'], runtime, 'frozen episode runtime ' + identity)
        if len(payload['episodes']) != 1:
            raise ValueError('Each frozen input must contain exactly one episode.')
        episode = payload['episodes'][0]
        assert_equal(payload['environment_config'], episode['final_info']['config'], 'episode configuration ' + identity)
        if not episode['final_info']['valid']:
            raise ValueError('An invalid original episode cannot be replay evidence.')
        if identity == 'baseline-student-scene14' and episode['success'] is not True:
            raise ValueError('The declared scene-14 comparator must be the known successful original.')
        if kind == 'recorded_open_jaw_control' and episode['success'] is not False:
            raise ValueError('A declared negative control unexpectedly succeeded.')
        snapshots[identity] = gzip.compress(raw, mtime=0)
        plans.append({'id': identity, 'role': kind, 'input_sha256': sha256(raw).hexdigest(),
                      'compressed_input_sha256': sha256(snapshots[identity]).hexdigest(),
                      'input_path': 'inputs/' + identity + '.json.gz', 'seed': episode['seed'],
                      'scene_sha256': episode['final_info']['scene_sha256'],
                      'config': episode['final_info']['config'], 'original_success': episode['success'],
                      'action_count': len(episode['actions']), 'scope': 'previously inspected frozen replay'})
    protocol = {
        'schema': SCHEMA, 'scope': 'passive_replay_of_inspected_simulation_cases',
        'runtime': runtime, 'sources_sha256': {name: sha256(raw).hexdigest() for name, raw in sources.items()},
        'upstream_metadata_sha256': {name: sha256(raw).hexdigest() for name, raw in metadata.items()},
        'attempts': plans, 'planned_native_episodes': len(plans),
        'reproduced_from_audit_protocol_sha256': original_protocol_hash,
        'scoring': {'per_contact_normal_threshold_n': CONTACT_THRESHOLD_N, 'lift_height_m': LIFT_HEIGHT_M,
                    'bilateral_contact': True, 'table_contact_absent': True, 'uninterrupted_lift_seconds': LIFT_DWELL_SECONDS,
                    'scope': 'independent read-only reconstruction of unchanged original native predicate'},
        'reproduction': 'exact JSON equality for all observations, actions, initial/final info, outcomes and rewards',
        'limits': LIMITATIONS,
    }
    root.mkdir(parents=True)
    for directory in ('inputs', 'sources', 'upstream', 'episodes'):
        (root / directory).mkdir()
    for identity, raw in snapshots.items():
        (root / 'inputs' / (identity + '.json.gz')).write_bytes(raw)
    for name, raw in sources.items():
        (root / 'sources' / name).write_bytes(raw)
    for name, raw in metadata.items():
        (root / 'upstream' / name).write_bytes(raw)
    write_json(root / 'protocol.json', protocol)
    protocol_hash = sha256((root / 'protocol.json').read_bytes()).hexdigest()
    (root / 'protocol.sha256').write_text(protocol_hash + '\n')

    def verify():
        assert_equal(sha256((root / 'protocol.json').read_bytes()).hexdigest(), protocol_hash, 'frozen audit protocol')
        assert_equal(runtime_contract(), runtime, 'current loaded runtime')
        for name, expected in protocol['sources_sha256'].items():
            assert_equal(sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(), expected, 'current audit source ' + name)
            assert_equal(sha256((root / 'sources' / name).read_bytes()).hexdigest(), expected, 'frozen audit source ' + name)
        for plan in plans:
            read_frozen_input(root, plan)
        for name, expected in protocol['upstream_metadata_sha256'].items():
            assert_equal(sha256((root / 'upstream' / name).read_bytes()).hexdigest(), expected, 'frozen upstream ' + name)

    results = []
    with NetworkGuard(True) as guard:
        for plan in plans:
            verify()
            episode = json.loads(read_frozen_input(root, plan))['episodes'][0]
            began = time.perf_counter()
            folder = root / 'episodes' / plan['id']
            folder.mkdir()
            trace = folder / 'native-trace.jsonl.gz'
            try:
                with trace.open('wb') as stream:
                    with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as zipped:
                        rows, reproduction = replay_episode(episode,
                            trace_sink=lambda row: zipped.write(canonical(row) + b'\n'))
                measured = summarize_native(rows, episode)
                receipt = {
                    'id': plan['id'], 'role': plan['role'], 'status': 'verified',
                    'input_sha256': plan['input_sha256'], 'protocol_sha256': protocol_hash,
                    'reproduction': reproduction, 'measurement': measured,
                    'dense_trace_path': str(trace.relative_to(root)),
                    'dense_trace_sha256': sha256(trace.read_bytes()).hexdigest(),
                    'dense_trace_bytes': trace.stat().st_size,
                    'wall_seconds': time.perf_counter() - began,
                }
                write_json(folder / 'receipt.json', receipt)
                results.append(receipt)
            except Exception as exc:
                failed = {'status': 'invalid_audit', 'id': plan['id'], 'role': plan['role'],
                    'input_sha256': plan['input_sha256'], 'protocol_sha256': protocol_hash, 'error': str(exc),
                    'dense_trace_path': str(trace.relative_to(root)),
                    'dense_trace_sha256': sha256(trace.read_bytes()).hexdigest() if trace.exists() else None,
                    'trace_scope': 'partial native evidence retained up to reproduction failure; not valid benchmark evidence',
                    'completed_receipts': [row['id'] for row in results]}
                write_json(folder / 'receipt.json', failed)
                write_json(root / 'failure.json', failed)
                raise
            write_json(root / 'progress.json', {'completed': len(results), 'planned': len(plans), 'last_id': plan['id']})
        verify()
    report = {
        'schema': SCHEMA, 'protocol_sha256': protocol_hash, 'status': 'verified',
        'planned_native_episodes': len(plans), 'verified_native_episodes': len(results),
        'unique_scene_count': 2, 'new_policy_trials': 0,
        'total_native_ticks': sum(row['reproduction']['native_ticks'] for row in results),
        'simulated_seconds': sum(row['reproduction']['simulated_seconds'] for row in results),
        'elapsed_seconds': time.perf_counter() - started,
        'network_guard': {'python_socket_attempts': len(guard.attempts), 'attempted_calls': guard.attempts,
                          'scope': 'Python socket guard; native dependency networking not audited'},
        'promotion': 'not applicable: passive instrumentation, unchanged recorded controllers',
        'results': results, 'limits': LIMITATIONS,
    }
    write_json(root / 'report.json', report)
    files = {str(path.relative_to(root)): {'sha256': sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
             for path in sorted(root.rglob('*')) if path.is_file()}
    write_json(root / 'integrity.json', {'schema': SCHEMA, 'files': files})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoothing', type=Path, help='Completed retention-smoothing-20261016 directory')
    parser.add_argument('--compliance', type=Path, help='Completed retention-compliance-20261016 directory')
    parser.add_argument('--from-audit', type=Path, help='Published audit bundle with frozen compressed inputs; no original runs needed')
    parser.add_argument('--output', type=Path, required=True, help='Fresh audit output directory')
    args = parser.parse_args()
    if args.from_audit is not None:
        if args.smoothing is not None or args.compliance is not None:
            parser.error('--from-audit cannot be combined with original experiment directories')
    elif args.smoothing is None or args.compliance is None:
        parser.error('Supply --from-audit or both --smoothing and --compliance')
    report = run_audit(args.smoothing, args.compliance, args.output, from_audit=args.from_audit)
    print(json.dumps({key: report[key] for key in ('status', 'verified_native_episodes', 'total_native_ticks', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
