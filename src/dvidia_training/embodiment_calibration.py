"""Original paired replay fidelity measurements (MIT), without qualification.

Inputs pair observations at explicitly shared elapsed times in a declared robot
base frame. There is no interpolation, inferred alignment, controller, contact
model or automatic pass threshold. Physical provenance and reviewed alignment
are declarations; a file and its hashes cannot authenticate a physical trial.
The scorer and bounded data loader use only Python's standard library.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
from statistics import median

SCHEMA = 'dvidia.paired-fidelity/v1'
REPORT_SCHEMA = 'dvidia.paired-fidelity-report/v1'
MAX_TRACE_BYTES = 4 * 1024 * 1024
MAX_SAMPLES = 100000
MAX_JOINTS = 128
MAX_NUMERIC_MAGNITUDE = 1000000
QUATERNION_NORM_TOLERANCE = 1e-6
_TRACE_KEYS = {
    'schema', 'reference_profile', 'model_sha256', 'physical_hardware_id',
    'protocol_id', 'source_group', 'command_replay_sha256', 'provenance',
    'alignment', 'time_basis', 'coordinate_frame', 'joint_names', 'samples',
}
_STATE_KEYS = {
    'joint_position_rad', 'tcp_position_m', 'tcp_quaternion_wxyz',
    'gripper_gap_m',
}
_ALIGNMENT_KEYS = {'time_alignment_reviewed', 'coordinate_alignment_reviewed'}


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f'{name}: unknown or missing fields.')


def _identifier(value, name):
    if type(value) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,199}', value) is None:
        raise ValueError(f'{name}: a bounded nonempty identifier is required.')
    return value


def _hash(value, name):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name}: a lowercase SHA256 declaration is required.')
    return value


def _number(value, name, minimum=None):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and abs(value) <= MAX_NUMERIC_MAGNITUDE
    except OverflowError:
        valid = False
    if not valid or (minimum is not None and value < minimum):
        raise ValueError(f'{name}: a bounded finite numeric value is required.')
    return float(value)


def _vector(value, length, name):
    if type(value) is not list or len(value) != length:
        raise ValueError(f'{name}: exactly {length} explicit numeric values are required.')
    return [_number(item, name) for item in value]


def _state(value, joint_count, name):
    _keys(value, _STATE_KEYS, name)
    joint = _vector(value['joint_position_rad'], joint_count, f'{name}.joint_position_rad')
    position = _vector(value['tcp_position_m'], 3, f'{name}.tcp_position_m')
    quaternion = _vector(value['tcp_quaternion_wxyz'], 4, f'{name}.tcp_quaternion_wxyz')
    norm = math.hypot(*quaternion)
    if abs(norm - 1.) > QUATERNION_NORM_TOLERANCE:
        raise ValueError(f'{name}.tcp_quaternion_wxyz: a normalized quaternion is required.')
    gap = _number(value['gripper_gap_m'], f'{name}.gripper_gap_m', minimum=0)
    return {'joint_position_rad': joint, 'tcp_position_m': position,
            'tcp_quaternion_wxyz': quaternion, 'gripper_gap_m': gap}


def validate_trace(payload):
    """Validate and detach a complete trace; never fill missing observations.

    Revolute joint positions are compared directly as supplied, without angle
    wrapping. A jaw gap remains a single initial-reference gripper channel;
    extra hand contact/tactile channels require a future schema and protocol.
    """
    _keys(payload, _TRACE_KEYS, 'trace')
    if payload['schema'] != SCHEMA:
        raise ValueError(f'trace.schema: expected {SCHEMA}.')
    if payload['provenance'] not in ('synthetic_fixture', 'declared_physical_replay'):
        raise ValueError('trace.provenance: synthetic_fixture or declared_physical_replay is required.')
    if payload['time_basis'] != 'shared_elapsed_seconds' or payload['coordinate_frame'] != 'robot_base':
        raise ValueError('Trace requires shared elapsed seconds and the declared robot_base frame.')
    alignment = payload['alignment']
    _keys(alignment, _ALIGNMENT_KEYS, 'trace.alignment')
    if any(type(alignment[key]) is not bool for key in _ALIGNMENT_KEYS):
        raise ValueError('Trace alignment reviews must be explicit booleans.')
    if payload['provenance'] == 'declared_physical_replay' and not all(alignment.values()):
        raise ValueError('Declared physical replay requires reviewed time and coordinate alignment.')
    names = payload['joint_names']
    if type(names) is not list or not 1 <= len(names) <= MAX_JOINTS:
        raise ValueError(f'Trace requires 1–{MAX_JOINTS} ordered joint names.')
    names = [_identifier(name, 'joint name') for name in names]
    if len(set(names)) != len(names):
        raise ValueError('Joint names must be unique.')
    samples = payload['samples']
    if type(samples) is not list or not 2 <= len(samples) <= MAX_SAMPLES:
        raise ValueError(f'Trace requires 2–{MAX_SAMPLES} explicit paired samples.')
    result = {'schema': SCHEMA, 'provenance': payload['provenance'],
              'time_basis': payload['time_basis'], 'coordinate_frame': payload['coordinate_frame'],
              'alignment': dict(alignment), 'joint_names': names, 'samples': []}
    for name in ('reference_profile', 'physical_hardware_id', 'protocol_id', 'source_group'):
        result[name] = _identifier(payload[name], name)
    for name in ('model_sha256', 'command_replay_sha256'):
        result[name] = _hash(payload[name], name)
    previous_time = -1.
    for index, sample in enumerate(samples):
        _keys(sample, {'t_seconds', 'simulator', 'physical'}, f'sample {index}')
        time = _number(sample['t_seconds'], f'sample {index}.t_seconds', minimum=0)
        if time <= previous_time:
            raise ValueError('Sample timestamps must be strictly increasing; no repeated or reordered samples.')
        result['samples'].append({
            't_seconds': time,
            'simulator': _state(sample['simulator'], len(names), f'sample {index}.simulator'),
            'physical': _state(sample['physical'], len(names), f'sample {index}.physical'),
        })
        previous_time = time
    return result


def _duplicate_checked_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'Duplicate JSON key: {name}.')
        result[name] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}.')


def load_trace(path):
    """Load at most 4 MiB of UTF-8 JSON from a regular, non-symlink file.

    Identity checks catch swaps where filesystem identifiers are available;
    Windows cannot provide the POSIX no-follow guarantee on every filesystem.
    Paths are never executed, remote references are never fetched, and all
    duplicate JSON keys are rejected before schema validation.
    """
    path = Path(path)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_TRACE_BYTES:
            raise ValueError('Trace must be a regular non-symlink file at most 4 MiB.')
        flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
        descriptor = os.open(path, flags)
    except (OSError, ValueError) as exc:
        raise ValueError('Trace cannot be opened as a regular non-symlink file at most 4 MiB.') from exc
    try:
        with os.fdopen(descriptor, 'rb') as source:
            opened = os.fstat(source.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_size > MAX_TRACE_BYTES:
                raise ValueError('Trace must be a regular file at most 4 MiB.')
            before_ino, opened_ino = getattr(before, 'st_ino', None), getattr(opened, 'st_ino', None)
            before_dev, opened_dev = getattr(before, 'st_dev', None), getattr(opened, 'st_dev', None)
            if before_ino not in (None, 0) and opened_ino not in (None, 0) and before_dev is not None and opened_dev is not None:
                if (before_dev, before_ino) != (opened_dev, opened_ino):
                    raise ValueError('Trace file identity changed during opening.')
            raw = source.read(MAX_TRACE_BYTES + 1)
    except OSError as exc:
        raise ValueError('Trace regular-file read failed.') from exc
    if len(raw) > MAX_TRACE_BYTES:
        raise ValueError('Trace exceeds the 4 MiB limit.')
    try:
        payload = json.loads(raw.decode('utf-8'), object_pairs_hook=_duplicate_checked_object,
                             parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f'Trace must be bounded, finite UTF-8 JSON with unique keys: {exc}') from exc
    return validate_trace(payload)


def _errors(values):
    count = len(values)
    return {'count': count, 'rmse': math.sqrt(math.fsum(value * value for value in values) / count),
            'mae': math.fsum(abs(value) for value in values) / count,
            'max_abs': max(abs(value) for value in values)}


def _orientation_error(first, second):
    # Unit rescaling only removes tolerated floating-point roundoff. Choosing
    # the nearest quaternion sign gives the same SO(3) orientation for q/-q.
    # The chord formula remains stable for very small angles and near pi.
    first = [value / math.hypot(*first) for value in first]
    second = [value / math.hypot(*second) for value in second]
    if math.fsum(a * b for a, b in zip(first, second)) < 0:
        second = [-value for value in second]
    difference = math.hypot(*(a - b for a, b in zip(first, second)))
    addition = math.hypot(*(a + b for a, b in zip(first, second)))
    return 4. * math.atan2(difference, addition)


def score_trace(payload):
    """Measure equal-weight paired observation errors, never hardware readiness.

    Many timestamps in one source group are correlated. These error summaries
    describe this single replay, without an independence claim, confidence
    interval, contact accuracy score or task success/safety verdict.
    """
    trace = validate_trace(payload)
    joint_errors = [[] for _ in trace['joint_names']]
    position_errors = [[], [], []]
    distances, orientations, gap_errors = [], [], []
    for sample in trace['samples']:
        simulator, physical = sample['simulator'], sample['physical']
        for axis, (simulated, observed) in enumerate(zip(simulator['joint_position_rad'], physical['joint_position_rad'])):
            joint_errors[axis].append(observed - simulated)
        displacement = [observed - simulated for simulated, observed in
                        zip(simulator['tcp_position_m'], physical['tcp_position_m'])]
        for axis, value in enumerate(displacement):
            position_errors[axis].append(value)
        distances.append(math.hypot(*displacement))
        orientations.append(_orientation_error(simulator['tcp_quaternion_wxyz'], physical['tcp_quaternion_wxyz']))
        gap_errors.append(physical['gripper_gap_m'] - simulator['gripper_gap_m'])
    samples = trace['samples']
    intervals = [later['t_seconds'] - earlier['t_seconds'] for earlier, later in zip(samples, samples[1:])]
    canonical = json.dumps(trace, allow_nan=False, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return {
        'schema': REPORT_SCHEMA,
        'scope': 'offline-paired-replay-only',
        'physical_robot_ready': False,
        'input_provenance': trace['provenance'],
        'declared_provenance_authenticated': False,
        'trace_sha256': sha256(canonical).hexdigest(),
        'reference_profile': trace['reference_profile'],
        'model_sha256': trace['model_sha256'],
        'physical_hardware_id': trace['physical_hardware_id'],
        'protocol_id': trace['protocol_id'],
        'command_replay_sha256': trace['command_replay_sha256'],
        'source_group': trace['source_group'],
        'source_group_count': 1,
        'sample_count': len(samples),
        'joint_count': len(trace['joint_names']),
        'alignment': {
            'time_basis': trace['time_basis'], 'coordinate_frame': trace['coordinate_frame'],
            **trace['alignment'], 'status': 'review-declaration-only',
            'interpolation_performed': False, 'time_shift_fitted': False,
        },
        'sampling': {
            'start_seconds': samples[0]['t_seconds'], 'end_seconds': samples[-1]['t_seconds'],
            'duration_seconds': samples[-1]['t_seconds'] - samples[0]['t_seconds'],
            'interval_min_seconds': min(intervals), 'interval_median_seconds': median(intervals),
            'interval_max_seconds': max(intervals),
            'weighting': 'equal per supplied sample; not duration weighted',
        },
        'metrics': {
            'joint_position': {
                'unit': 'rad', 'difference': 'physical minus simulator; no angle wrapping',
                'per_joint': [{'name': name, **_errors(values)} for name, values in zip(trace['joint_names'], joint_errors)],
                'pooled': _errors([value for values in joint_errors for value in values]),
            },
            'tcp_position': {
                'unit': 'm',
                'per_axis': [{'name': axis, **_errors(values)} for axis, values in zip(('x', 'y', 'z'), position_errors)],
                'euclidean': _errors(distances),
            },
            'tcp_orientation': {
                'unit': 'rad', 'definition': 'sign-invariant SO(3) geodesic angle in [0, pi]',
                'geodesic': _errors(orientations),
            },
            'gripper_gap': {'unit': 'm', **_errors(gap_errors)},
        },
        'limitations': [
            'Synthetic fixtures are authored values, not measured physical hardware evidence.'
            if trace['provenance'] == 'synthetic_fixture' else
            'Physical replay provenance is a supplied declaration, not authenticated by this scorer.',
            'Alignment review booleans are declarations; shared timestamps and frames are not independently verified.',
            'Many sample rows from this single source group are correlated; independent trial coverage is not established.',
            'Model, command and trace hashes establish declared identity or internal consistency, not authenticity.',
            'Offline replay error does not establish contact fidelity, task success, closed-loop behavior or physical safety.',
            'No acceptance threshold, pass decision, hardware qualification or actuator command is produced.',
        ],
    }
