"""Versioned, data-only control-boundary tapes for the authored simulator.

This is an evidence recorder, not a learner, hardware adapter or extension of
the existing movement sidecar. Validation checks the contract and consistency;
it does not establish that sensor readings or task labels are true.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re

KIND = 'dvidia.simulation-episode-tape'
ARM_PROFILE = 'dvidia-authored-6dof-parallel-jaw-v0'
TASK_ID = 'ArmPickPlace-v0'
MAX_BYTES = 64 * 1024 * 1024
MAX_STEPS = 10000
MAX_NATIVE_TICKS_PER_INTERVAL = 1000000
JOINT_LIMITS = ((-2.6, 2.6), (-2.2, 1.6), (-2.7, 2.7), (-3., 3.), (-2.8, 2.8), (-3.1, 3.1))
PHASES = {'observe', 'approach', 'descend', 'close', 'lift', 'transfer', 'place',
          'release', 'retreat', 'verify', 'recover_wait', 'recover_up', 'failed'}
CASES = ('nominal', 'open-jaw', 'release-after-lift')
SIGNALS = {
    'joint_position': {'shape': [6], 'units': 'rad', 'frame': 'joint_coordinates', 'privileged': True},
    'joint_velocity': {'shape': [6], 'units': 'rad/s', 'frame': 'joint_coordinates', 'privileged': True},
    'joint_actuator_torque': {'shape': [6], 'units': 'N*m', 'frame': 'joint_coordinates', 'privileged': True},
    'jaw_actuator_force': {'shape': [2], 'units': 'N', 'frame': 'jaw_slide_coordinates_left_right', 'privileged': True},
    'gripper_width': {'shape': [], 'units': 'm', 'frame': 'jaw_gap', 'privileged': True},
    'pad_normal_forces': {'shape': [2], 'units': 'N', 'frame': 'contact_normal_left_right', 'privileged': True},
    'grasp_contacts': {'shape': [2], 'units': 'count', 'frame': 'left_right_object_contact', 'privileged': True},
    'table_contacts': {'shape': [], 'units': 'count', 'frame': 'table_object_contact', 'privileged': True},
    'end_effector_position': {'shape': [3], 'units': 'm', 'frame': 'simulation_world', 'privileged': True},
    'end_effector_quaternion': {'shape': [4], 'units': 'unit_quaternion_wxyz', 'frame': 'simulation_world', 'privileged': True},
    'end_effector_velocity': {'shape': [3], 'units': 'm/s', 'frame': 'simulation_world', 'privileged': True},
    'object_position': {'shape': [3], 'units': 'm', 'frame': 'simulation_world', 'privileged': True},
    'object_quaternion': {'shape': [4], 'units': 'unit_quaternion_wxyz', 'frame': 'simulation_world', 'privileged': True},
    'object_velocity': {'shape': [3], 'units': 'm/s', 'frame': 'simulation_world', 'privileged': True},
    'object_angular_velocity': {'shape': [3], 'units': 'rad/s', 'frame': 'simulation_world', 'privileged': True},
    'target_position': {'shape': [3], 'units': 'm', 'frame': 'simulation_world', 'privileged': True},
}
PROTOCOL = {
    'revision': 'control-boundary-v1',
    'sampling': 'before decision and after each control step; intermediate native ticks are not retained',
    'application': 'native targets ramp during the step; applied values are the final native tick only',
    'sensor_age_limit_seconds': 0.1,
    'labels': 'authored controller action phase and next phase; evidence only, not deployment inputs',
    'qualification': 'simulation-only; privileged state; no physical robot qualification',
}
CONFIG_FIELDS = {'timestep', 'control_dt', 'horizon', 'settle_time', 'object_position',
                 'target_position', 'object_half_size', 'object_mass', 'object_friction',
                 'table_height', 'position_tolerance', 'dwell_seconds', 'scene_jitter',
                 'joint_torque_limit', 'jaw_force_limit', 'joint_target_rate_limit',
                 'jaw_width_rate_limit', 'gravity'}
SOURCE_FILES = {'arm_env.py', 'arm_policy.py', 'episode_tape.py'}


def canonical(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def digest(document):
    return sha256(canonical(document)).hexdigest()


def _keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f'{label}: unknown or missing fields.')


def _number(value, label, *, minimum=None):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (minimum is not None and value < minimum):
        raise ValueError(f'{label}: finite numeric value required.')
    return value


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label}: integer >= {minimum} required.')
    return value


def _ticks(numerator, denominator, label, *, maximum, tolerance=1e-9, boundary_roundoff=False):
    """Check finite bounded ratios before rounding; extreme JSON is not a crash."""
    try:
        ratio = numerator/denominator
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f'{label}: invalid tick ratio.') from exc
    lower = 1-tolerance if boundary_roundoff else 1
    if not math.isfinite(ratio) or not lower <= ratio <= maximum+tolerance:
        raise ValueError(f'{label}: tick count must lie in [1, {maximum}].')
    ticks = round(ratio)
    if not 1 <= ticks <= maximum or not math.isclose(ratio, ticks, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f'{label}: an integer number of ticks is required.')
    return ticks


def _identifier(value, label):
    if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}', value):
        raise ValueError(f'{label}: short source identifier required.')


def _sha(value, label):
    if type(value) is not str or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError(f'{label}: lowercase SHA256 required.')


def _vector(value, size, label):
    if type(value) is not list or len(value) != size:
        raise ValueError(f'{label}: {size} entries required.')
    for item in value:
        _number(item, label)


def _decode(document):
    if type(document) is dict:
        try:
            raw = canonical(document)
        except (TypeError, ValueError, RecursionError, OverflowError) as exc:
            raise ValueError('Tape must be finite JSON data.') from exc
    elif isinstance(document, (str, bytes)):
        raw = document.encode('utf8') if isinstance(document, str) else document
    else:
        raise ValueError('Tape must be a plain dict or UTF-8 JSON.')
    if len(raw) > MAX_BYTES:
        raise ValueError('Episode tape exceeds 64 MiB.')
    def pairs(items):
        row = {}
        for key, value in items:
            if key in row:
                raise ValueError('Duplicate episode-tape field.')
            row[key] = value
        return row
    def constant(value):
        raise ValueError('Nonfinite episode-tape value.')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError('Tape must be finite UTF-8 JSON.') from exc


def _snapshot(snapshot, label):
    _keys(snapshot, {'timestamp_seconds', 'signals'}, label)
    timestamp = _number(snapshot['timestamp_seconds'], label, minimum=0)
    _keys(snapshot['signals'], SIGNALS, f'{label}.signals')
    for name, contract in SIGNALS.items():
        sensor = snapshot['signals'][name]
        _keys(sensor, {'status', 'captured_at_seconds', 'age_seconds', 'value'}, f'{label}.{name}')
        if sensor['status'] == 'missing':
            if any(sensor[key] is not None for key in ('captured_at_seconds', 'age_seconds', 'value')):
                raise ValueError('Missing signals must have null time, age and value, never synthetic zeros.')
            continue
        if sensor['status'] not in ('available', 'stale'):
            raise ValueError('Signal status must be available, stale or missing.')
        captured = _number(sensor['captured_at_seconds'], name, minimum=0)
        age = _number(sensor['age_seconds'], name, minimum=0)
        if captured > timestamp + 1e-9 or not math.isclose(timestamp-captured, age, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Signal acquisition time and age disagree with observation time.')
        if (sensor['status'] == 'available') != (age <= PROTOCOL['sensor_age_limit_seconds']):
            raise ValueError('Signal freshness disagrees with its declared age.')
        value = sensor['value']
        if contract['shape']:
            _vector(value, contract['shape'][0], name)
        else:
            _number(value, name)
        values = value if type(value) is list else [value]
        if contract['units'] == 'count':
            for item in values:
                _integer(item, name)
        if name in ('gripper_width', 'pad_normal_forces') and any(x < -1e-9 for x in values):
            raise ValueError('Measured gap and normal forces cannot be negative.')
        if contract['units'] == 'unit_quaternion_wxyz' and not .99 <= math.hypot(*value) <= 1.01:
            raise ValueError('Measured orientation must be a unit quaternion.')
    return timestamp


def _action(action, label):
    _keys(action, {'joint_targets', 'gripper_width'}, label)
    _vector(action['joint_targets'], 6, label)
    for value, (low, high) in zip(action['joint_targets'], JOINT_LIMITS):
        if not low <= value <= high:
            raise ValueError('Requested joint targets exceed the authored arm limits.')
    if not 0 <= _number(action['gripper_width'], label) <= .08:
        raise ValueError('Requested gap exceeds the authored jaw limits.')


def validate_tape(document):
    """Return a detached JSON tape; consistency and hashes are not qualification."""
    tape = _decode(document)
    _keys(tape, {'schema_version', 'kind', 'scope', 'physical_robot_ready', 'task_id',
                 'arm_profile', 'provenance', 'clock', 'signal_contract', 'protocol',
                 'protocol_sha256', 'configuration', 'initial', 'steps', 'outcome'}, 'tape')
    expected = {'schema_version': 1, 'kind': KIND, 'scope': 'simulation-only',
                'physical_robot_ready': False, 'task_id': TASK_ID, 'arm_profile': ARM_PROFILE}
    if any(type(tape[key]) is not type(value) or tape[key] != value for key, value in expected.items()):
        raise ValueError('Unsupported episode-tape version, arm, task or scope.')
    if canonical(tape['signal_contract']) != canonical(SIGNALS) or canonical(tape['protocol']) != canonical(PROTOCOL) or tape['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('Unsupported signal contract or recording protocol.')
    provenance = tape['provenance']
    _keys(provenance, {'episode_id', 'source_recording_id', 'session_id', 'source_kind',
                       'seed', 'case', 'policy_origin', 'policy_revision', 'engine',
                       'scene_sha256', 'configuration_sha256', 'source_files_sha256'}, 'provenance')
    for key in ('episode_id', 'source_recording_id', 'session_id', 'policy_revision'):
        _identifier(provenance[key], key)
    _integer(provenance['seed'], 'seed')
    if provenance['source_kind'] != 'native-simulation-telemetry' or provenance['policy_origin'] != 'authored controller' or provenance['engine'] != 'mujoco-3.15.0' or provenance['case'] not in CASES:
        raise ValueError('Unsupported simulation provenance.')
    _sha(provenance['scene_sha256'], 'scene_sha256')
    _sha(provenance['configuration_sha256'], 'configuration_sha256')
    _keys(provenance['source_files_sha256'], SOURCE_FILES, 'source_files_sha256')
    for name, value in provenance['source_files_sha256'].items():
        _sha(value, name)
    config = tape['configuration']
    _keys(config, CONFIG_FIELDS, 'configuration')
    for name, value in config.items():
        if name in ('object_position', 'target_position', 'object_half_size'):
            _vector(value, 3, name)
        else:
            _number(value, name, minimum=0)
    if config['timestep'] <= 0 or config['control_dt'] < config['timestep'] or config['horizon'] <= 0:
        raise ValueError('Invalid simulator timing configuration.')
    control_ticks = _ticks(config['control_dt'], config['timestep'], 'Control interval', maximum=MAX_NATIVE_TICKS_PER_INTERVAL)
    _ticks(config['horizon'], config['control_dt'], 'Episode horizon', maximum=MAX_STEPS)
    settle_ticks = _ticks(config['settle_time'], config['timestep'], 'Reset settling interval', maximum=MAX_NATIVE_TICKS_PER_INTERVAL)
    if any(config[name] <= 0 for name in ('joint_torque_limit', 'jaw_force_limit', 'joint_target_rate_limit', 'jaw_width_rate_limit', 'gravity')):
        raise ValueError('Native actuator and rate limits must be positive.')
    for name, low, high in (('object_mass', .02, .08), ('object_friction', .4, 1.5),
                            ('position_tolerance', .01, .035), ('dwell_seconds', .15, .6), ('scene_jitter', 0., .01)):
        if not low <= config[name] <= high:
            raise ValueError('Configuration exceeds the authored simulation envelope.')
    if config['table_height'] != .29 or any(not .015 <= x <= .025 for x in config['object_half_size']):
        raise ValueError('Unsupported table height or rigid-box dimensions.')
    for name in ('object_position', 'target_position'):
        x, y, z = config[name]
        if not .25 <= x <= .60 or not -.20 <= y <= .20 or math.hypot(x, y) > .56 or abs(z-(.29+config['object_half_size'][2])) > .01000001:
            raise ValueError('Placement exceeds the authored table and reach envelope.')
    if math.dist(config['object_position'][:2], config['target_position'][:2]) < .075:
        raise ValueError('Placement must require nontrivial movement.')
    if provenance['configuration_sha256'] != digest(config):
        raise ValueError('Configuration hash mismatch.')
    _keys(tape['clock'], {'kind', 'origin', 'native_time_offset_seconds'}, 'clock')
    if tape['clock']['kind'] != 'simulation-elapsed-seconds' or tape['clock']['origin'] != 'after-reset-settle':
        raise ValueError('Unsupported clock; wall time and device-clock mappings require a new contract.')
    clock_offset = _number(tape['clock']['native_time_offset_seconds'], 'native clock offset', minimum=0)
    expected_offset = settle_ticks*config['timestep']
    if not math.isfinite(expected_offset) or not math.isclose(clock_offset, expected_offset, rel_tol=0, abs_tol=1e-9):
        raise ValueError('Native clock origin does not match the declared reset settling ticks.')
    if _snapshot(tape['initial'], 'initial') != 0:
        raise ValueError('Tape must begin at the declared zero-time origin.')
    if type(tape['steps']) is not list or not 1 <= len(tape['steps']) <= MAX_STEPS:
        raise ValueError('Tape requires 1–10000 control steps.')
    previous = tape['initial']
    terminal = None
    for index, step in enumerate(tape['steps']):
        _keys(step, {'index', 'before', 'command', 'applied', 'after', 'labels', 'result'}, f'step {index}')
        if type(step['index']) is not int or step['index'] != index or step['before'] != previous:
            raise ValueError('Steps must be sequential and preserve the preceding observed state.')
        start = _snapshot(step['before'], 'before')
        end = _snapshot(step['after'], 'after')
        if not start < end or end-start > config['control_dt']+1e-9:
            raise ValueError('Control timestamps must advance without gaps or reordered samples.')
        if end > config['horizon']+1e-9:
            raise ValueError('Recorded transition exceeds the declared episode horizon.')
        _ticks(end-start, config['timestep'], 'Recorded transition', maximum=control_ticks, tolerance=1e-7, boundary_roundoff=True)
        command = step['command']
        _keys(command, {'issued_at_seconds', 'policy_requested', 'issued', 'intervention'}, 'command')
        if _number(command['issued_at_seconds'], 'command issue time', minimum=0) != start:
            raise ValueError('Command issue time must match its input observation.')
        _action(command['policy_requested'], 'policy request')
        _action(command['issued'], 'issued command')
        if command['intervention'] not in ('none', 'open-jaw', 'release-after-lift'):
            raise ValueError('Unknown recording intervention.')
        if provenance['case'] == 'open-jaw' and command['intervention'] != 'open-jaw':
            raise ValueError('Open-jaw case must declare every jaw-command override.')
        if command['intervention'] == 'none' and command['issued'] != command['policy_requested']:
            raise ValueError('Unmodified command must equal policy intent.')
        if command['intervention'] != 'none':
            if command['intervention'] != provenance['case'] or command['issued']['gripper_width'] != .08 or command['issued']['joint_targets'] != command['policy_requested']['joint_targets']:
                raise ValueError('Jaw intervention must preserve authored joint intent and declare its case.')
        applied = step['applied']
        _keys(applied, {'interval_start_seconds', 'interval_end_seconds', 'ramped_joint_targets',
                        'ramped_gripper_width', 'native_position_controls', 'rate_limited'}, 'applied')
        if _number(applied['interval_start_seconds'], 'application start', minimum=0) != start or _number(applied['interval_end_seconds'], 'application end', minimum=0) != end:
            raise ValueError('Applied-command interval must match the recorded transition.')
        _action({'joint_targets': applied['ramped_joint_targets'], 'gripper_width': applied['ramped_gripper_width']}, 'ramped command')
        _vector(applied['native_position_controls'], 8, 'native_position_controls')
        _action({'joint_targets': applied['native_position_controls'][:6], 'gripper_width': sum(applied['native_position_controls'][6:])}, 'native servo controls')
        if any(not 0 <= x <= .04 for x in applied['native_position_controls'][6:]) or type(applied['rate_limited']) is not bool:
            raise ValueError('Invalid jaw servo controls or rate-limit flag.')
        expected_jaw_control = applied['ramped_gripper_width']/2
        if any(not math.isclose(x, expected_jaw_control, rel_tol=0, abs_tol=1e-12) for x in applied['native_position_controls'][6:]):
            raise ValueError('Both native jaw controls must equal half the recorded ramped gap.')
        labels = step['labels']
        _keys(labels, {'origin', 'action_phase', 'next_phase'}, 'labels')
        if labels['origin'] != 'authored controller' or type(labels['action_phase']) is not str or type(labels['next_phase']) is not str or labels['action_phase'] not in PHASES or labels['next_phase'] not in PHASES:
            raise ValueError('Unsupported authored phase evidence.')
        result = step['result']
        _keys(result, {'terminated', 'truncated', 'success', 'valid', 'reason'}, 'result')
        if any(type(result[key]) is not bool for key in ('terminated', 'truncated', 'success', 'valid')):
            raise ValueError('Result flags must be booleans.')
        reason = result['reason']
        if reason not in ('running', 'success', 'horizon', 'invalid_state', 'solver_warning', 'object_dropped'):
            raise ValueError('Unsupported result reason.')
        expected_flags = {'terminated': reason not in ('running', 'horizon'), 'truncated': reason == 'horizon',
                          'success': reason == 'success', 'valid': reason not in ('invalid_state', 'solver_warning', 'object_dropped')}
        if any(result[key] != value for key, value in expected_flags.items()):
            raise ValueError('Terminal, success and validity flags contradict the result reason.')
        if reason == 'horizon' and not math.isclose(end, config['horizon'], rel_tol=0, abs_tol=1e-8):
            raise ValueError('Horizon termination must reach the declared episode horizon.')
        terminal = result['terminated'] or result['truncated']
        if terminal != (index == len(tape['steps'])-1):
            raise ValueError('Only the final transition must be terminal.')
        if not terminal and not math.isclose(end-start, config['control_dt'], rel_tol=0, abs_tol=1e-9):
            raise ValueError('Nonterminal transition must cover one control interval.')
        previous = step['after']
    outcome = tape['outcome']
    _keys(outcome, {'result', 'simulation_seconds', 'control_steps', 'policy_failure_reason', 'recovery_count'}, 'outcome')
    if canonical(outcome['result']) != canonical(tape['steps'][-1]['result']) or _number(outcome['simulation_seconds'], 'outcome time', minimum=0) != previous['timestamp_seconds'] or type(outcome['control_steps']) is not int or outcome['control_steps'] != len(tape['steps']):
        raise ValueError('Final outcome does not match the terminal transition.')
    _integer(outcome['recovery_count'], 'recovery_count')
    failure = outcome['policy_failure_reason']
    if failure is not None and (type(failure) is not str or not failure or len(failure) > 160):
        raise ValueError('Invalid authored policy failure reason.')
    return tape


def snapshot(observation, timestamp_seconds):
    """Convert one native observation; no unavailable sensor is filled with zero."""
    signals = {}
    for name in SIGNALS:
        value = observation.get(name)
        if name in ('pad_normal_forces', 'grasp_contacts') and type(value) is dict:
            value = [value.get('left'), value.get('right')]
        missing = value is None or value == [] or (type(value) is list and any(x is None for x in value))
        signals[name] = {'status': 'missing' if missing else 'available',
                         'captured_at_seconds': None if missing else timestamp_seconds,
                         'age_seconds': None if missing else 0., 'value': None if missing else deepcopy(value)}
    return {'timestamp_seconds': timestamp_seconds, 'signals': signals}


def record_episode(*, episode_id, source_recording_id, session_id, seed=0, case='nominal', config=None):
    """Record a fresh authored simulation. Requires the optional arm dependencies.

    No renderer, network, video conversion, training or frozen benchmark is run.
    Case interventions change issued jaw commands, never object dynamics.
    """
    for key, value in (('episode_id', episode_id), ('source_recording_id', source_recording_id), ('session_id', session_id)):
        _identifier(value, key)
    _integer(seed, 'seed')
    if case not in CASES:
        raise ValueError(f'Use one of: {", ".join(CASES)}.')
    from .arm_env import ArmConfig, ArmEnv
    from .arm_policy import ArmPickPlacePolicy, POLICY_REVISION
    environment = ArmEnv(config if config is not None else ArmConfig())
    observation, info = environment.reset(seed=seed)
    policy = ArmPickPlacePolicy(environment)
    configuration = json.loads(canonical(asdict(environment.config)))
    tape = {'schema_version': 1, 'kind': KIND, 'scope': 'simulation-only', 'physical_robot_ready': False,
            'task_id': TASK_ID, 'arm_profile': ARM_PROFILE,
            'provenance': {'episode_id': episode_id, 'source_recording_id': source_recording_id,
                           'session_id': session_id, 'source_kind': 'native-simulation-telemetry',
                           'seed': seed, 'case': case, 'policy_origin': 'authored controller',
                           'policy_revision': POLICY_REVISION, 'engine': f'mujoco-{info["engine"]}',
                           'scene_sha256': info['scene_sha256'], 'configuration_sha256': digest(configuration),
                           'source_files_sha256': {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_FILES}},
            'clock': {'kind': 'simulation-elapsed-seconds', 'origin': 'after-reset-settle',
                      'native_time_offset_seconds': float(environment.data.time)},
            'signal_contract': deepcopy(SIGNALS), 'protocol': deepcopy(PROTOCOL),
            'protocol_sha256': digest(PROTOCOL), 'configuration': configuration,
            'initial': snapshot(observation, 0.), 'steps': []}
    before = tape['initial']
    release_at = None
    while True:
        policy_requested = policy(observation)
        issued = deepcopy(policy_requested)
        diagnostics = policy.diagnostics()
        timestamp = before['timestamp_seconds']
        if case == 'release-after-lift' and release_at is None and info['lift_seen']:
            release_at = timestamp
        intervention = 'none'
        if case == 'open-jaw' or (release_at is not None and timestamp < release_at+.35):
            issued['gripper_width'] = .08
            intervention = case
        observation, _, terminated, truncated, info = environment.step(issued)
        end = float(info['simulation_time'])
        after = snapshot(observation, end)
        if info['reason'] == 'invalid_state':
            for sensor in after['signals'].values():
                sensor.update(status='missing', captured_at_seconds=None, age_seconds=None, value=None)
        result = {'terminated': bool(terminated), 'truncated': bool(truncated), 'success': info['success'],
                  'valid': info['valid'], 'reason': info['reason']}
        tape['steps'].append({'index': len(tape['steps']), 'before': before,
                             'command': {'issued_at_seconds': timestamp, 'policy_requested': deepcopy(policy_requested),
                                         'issued': issued, 'intervention': intervention},
                             'applied': {'interval_start_seconds': timestamp, 'interval_end_seconds': end,
                                         'ramped_joint_targets': info['commanded_joint_targets'],
                                         'ramped_gripper_width': info['commanded_gripper_width'],
                                         'native_position_controls': info['native_position_controls'],
                                         'rate_limited': info['command_rate_limited']},
                             'after': after, 'labels': {'origin': 'authored controller',
                                                        'action_phase': diagnostics['action_phase'], 'next_phase': diagnostics['phase']},
                             'result': result})
        before = after
        if terminated or truncated:
            break
        if len(tape['steps']) >= MAX_STEPS:
            raise ValueError('Recording exceeds the episode-tape step limit.')
    diagnostics = policy.diagnostics()
    tape['outcome'] = {'result': result, 'simulation_seconds': end, 'control_steps': len(tape['steps']),
                       'policy_failure_reason': diagnostics['failure_reason'], 'recovery_count': diagnostics['recovery_count']}
    return validate_tape(tape)


def load_tape(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Episode tape exceeds 64 MiB.')
    return validate_tape(path.read_bytes())


def write_tape(path, document):
    """Create a new tape without overwriting an existing file or benchmark."""
    tape = validate_tape(document)
    raw = canonical(tape)
    if len(raw)+1 > MAX_BYTES:
        raise ValueError('Episode tape exceeds 64 MiB.')
    path = Path(path)
    with path.open('xb') as output:
        output.write(raw + b'\n')
    return {'path': str(path), 'sha256': sha256(raw + b'\n').hexdigest(), 'bytes': len(raw)+1}


def inspect_tape(document):
    """Summarize evidence and missingness without treating authored labels as truth."""
    tape = validate_tape(document)
    counts = {name: {'available': 0, 'stale': 0, 'missing': 0} for name in SIGNALS}
    phases = {}
    max_age = {name: None for name in SIGNALS}
    interventions = 0
    observations = [tape['initial']] + [step['after'] for step in tape['steps']]
    for observation in observations:
        for name, sensor in observation['signals'].items():
            counts[name][sensor['status']] += 1
            if sensor['age_seconds'] is not None:
                max_age[name] = max(max_age[name] or 0., sensor['age_seconds'])
    for step in tape['steps']:
        phase = step['labels']['action_phase']
        phases[phase] = phases.get(phase, 0)+1
        interventions += step['command']['intervention'] != 'none'
    return {'kind': 'dvidia.episode-tape-inspection', 'schema_version': 1,
            'scope': 'simulation-only', 'physical_robot_ready': False,
            'episode_id': tape['provenance']['episode_id'], 'source_recording_id': tape['provenance']['source_recording_id'],
            'session_id': tape['provenance']['session_id'], 'case': tape['provenance']['case'],
            'tape_content_sha256': digest(tape), 'protocol_sha256': tape['protocol_sha256'],
            'control_steps': len(tape['steps']), 'unique_observations': len(observations),
            'simulation_seconds': tape['outcome']['simulation_seconds'], 'outcome': deepcopy(tape['outcome']),
            'authored_action_phase_samples': phases, 'command_intervention_samples': interventions,
            'signal_status_counts': counts, 'max_signal_age_seconds': max_age,
            'limitation': 'Contract validation only; privileged simulator state and authored phase labels; no learned supervisor or hardware qualification.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    recorder = sub.add_parser('record', help='Record a fresh native simulation; requires the arm extra.')
    recorder.add_argument('--output', required=True)
    recorder.add_argument('--episode-id', required=True)
    recorder.add_argument('--source-recording-id', required=True)
    recorder.add_argument('--session-id', required=True)
    recorder.add_argument('--seed', type=int, default=0)
    recorder.add_argument('--case', choices=CASES, default='nominal')
    for name in ('validate', 'inspect'):
        reader = sub.add_parser(name, help='Read data-only tape without MuJoCo or a GPU.')
        reader.add_argument('tape')
    args = parser.parse_args(argv)
    try:
        if args.operation == 'record':
            path = Path(args.output)
            if path.exists():
                raise ValueError('Output already exists; use a fresh path.')
            tape = record_episode(episode_id=args.episode_id, source_recording_id=args.source_recording_id,
                                  session_id=args.session_id, seed=args.seed, case=args.case)
            result = write_tape(path, tape)
            result['inspection'] = inspect_tape(tape)
        else:
            tape = load_tape(args.tape)
            result = inspect_tape(tape) if args.operation == 'inspect' else {
                'status': 'valid', 'scope': 'simulation-only', 'physical_robot_ready': False,
                'tape_content_sha256': digest(tape), 'control_steps': len(tape['steps'])}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(json.dumps({'status': 'error', 'error': str(exc)}, allow_nan=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
