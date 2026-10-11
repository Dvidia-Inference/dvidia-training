"""Original compact, action-conditioned boundary predictor (MIT).

This learns seven *20 ms state deltas*, not a task policy, visual representation,
contact predictor or robot controller. The present training observations are
privileged native simulation values in sensor-shaped fields. The data-only
artifact and prediction functions need only Python's standard library; fitting
imports NumPy locally. Internal hashes establish consistency, not authenticity.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat

from .episode_tape import JOINT_LIMITS, PROTOCOL, canonical, digest, validate_tape

KIND = 'dvidia.compact-action-transition-model'
CONTROL_DT_SECONDS = .02
MAX_MODEL_BYTES = 256 * 1024
MAX_TRAINING_ROWS = 200000
MAX_NUMERIC_MAGNITUDE = 1e6
FEATURE_NAMES = tuple([f'before.joint_position[{i}]' for i in range(6)] +
                      [f'before.joint_velocity[{i}]' for i in range(6)] +
                      ['before.gripper_width'] +
                      [f'command.issued.joint_targets[{i}]' for i in range(6)] +
                      ['command.issued.gripper_width'])
OUTPUT_NAMES = tuple([f'delta.joint_position[{i}]' for i in range(6)] +
                     ['delta.gripper_width'])
FEATURE_CONTRACT = {
    'names': list(FEATURE_NAMES),
    'units': ['rad']*6 + ['rad/s']*6 + ['m'] + ['rad']*6 + ['m'],
    'availability': 'before decision plus issued command; no future or applied control',
    'acquisition': 'required signals contemporaneous with boundary; no delayed-sensor propagation',
    'origin': 'sensor-shaped privileged native simulation; hardware sensors unqualified',
}
OUTPUT_CONTRACT = {
    'names': list(OUTPUT_NAMES), 'units': ['rad']*6 + ['m'],
    'definition': 'after minus before at exactly one 20 ms control boundary',
    'acquisition': 'required after signals contemporaneous with the after boundary',
    'contact_and_task_outcomes': 'not predicted',
}
_MODEL_KEYS = {'schema_version', 'kind', 'license', 'scope', 'physical_robot_ready',
               'is_controller', 'predicts_task_success', 'sensor_origin', 'feature_mode',
               'feature_contract', 'output_contract', 'control_dt_seconds', 'alpha',
               'normalization', 'coefficients', 'intercept', 'training', 'model_sha256'}
_PROVENANCE_KEYS = {'episode_id', 'session_id', 'source_recording_id', 'seed', 'case',
                    'index', 'before_time_seconds'}
_REQUIRED_BEFORE = ('joint_position', 'joint_velocity', 'gripper_width')
_REQUIRED_AFTER = ('joint_position', 'gripper_width')


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f'{name}: unknown or missing fields.')


def _number(value, name, *, minimum=None, maximum=MAX_NUMERIC_MAGNITUDE):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and abs(value) <= maximum
    except OverflowError:
        valid = False
    if not valid or (minimum is not None and value < minimum):
        raise ValueError(f'{name}: bounded finite numeric value required.')
    return float(value)


def _vector(value, length, name, **limits):
    if type(value) not in (list, tuple) or len(value) != length:
        raise ValueError(f'{name}: exactly {length} values required.')
    return [_number(item, name, **limits) for item in value]


def _sha(value, name):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name}: lowercase SHA256 required.')


def _sensor_values(snapshot, required):
    """Use only explicitly fresh required inputs; extras never become features."""
    _keys(snapshot, {'timestamp_seconds', 'signals'}, 'observation')
    timestamp = _number(snapshot['timestamp_seconds'], 'observation timestamp', minimum=0)
    signals = snapshot['signals']
    if type(signals) is not dict:
        raise ValueError('Observation signals must be a dictionary.')
    values = []
    for name in required:
        sensor = signals.get(name)
        if sensor is None:
            raise ValueError(f'missing:{name}')
        _keys(sensor, {'status', 'captured_at_seconds', 'age_seconds', 'value'}, name)
        status = sensor['status']
        if status != 'available':
            if status in ('missing', 'stale'):
                raise ValueError(f'{status}:{name}')
            raise ValueError(f'unsupported_status:{name}')
        captured = _number(sensor['captured_at_seconds'], name, minimum=0)
        age = _number(sensor['age_seconds'], name, minimum=0)
        if captured > timestamp+1e-9 or not math.isclose(timestamp-captured, age, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f'inconsistent_acquisition_time:{name}')
        if age > PROTOCOL['sensor_age_limit_seconds']:
            raise ValueError(f'stale:{name}')
        if age > 1e-9 or not math.isclose(captured, timestamp, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f'noncontemporaneous:{name}')
        if name == 'gripper_width':
            values.append(_number(sensor['value'], name, minimum=0))
        else:
            values.extend(_vector(sensor['value'], 6, name))
    return values


def _issued_values(issued_command):
    _keys(issued_command, {'joint_targets', 'gripper_width'}, 'issued command')
    targets = _vector(issued_command['joint_targets'], 6, 'issued joint targets')
    if any(not low <= target <= high for target, (low, high) in zip(targets, JOINT_LIMITS)):
        raise ValueError('Issued target exceeds the authored arm limits.')
    gap = _number(issued_command['gripper_width'], 'issued gripper width', minimum=0)
    if gap > .08:
        raise ValueError('Issued gap exceeds the authored jaw limits.')
    return targets + [gap]


def extract_rows(tape):
    """Validate a tape, then expose causal features and explicitly exclude gaps.

Provenance, phases, time, intervention/case labels, contacts, object state,
future observations and applied servo controls are never learner inputs.
"""
    tape = validate_tape(tape)
    provenance = tape['provenance']
    rows, exclusions = [], []
    configured_dt = tape['configuration']['control_dt']
    for step in tape['steps']:
        before, after = step['before'], step['after']
        interval = after['timestamp_seconds']-before['timestamp_seconds']
        reason = None
        if not math.isclose(configured_dt, CONTROL_DT_SECONDS, rel_tol=0, abs_tol=1e-9):
            reason = 'unsupported_control_interval'
        elif not math.isclose(interval, CONTROL_DT_SECONDS, rel_tol=0, abs_tol=1e-9):
            reason = 'incomplete_20ms_boundary'
        try:
            if reason is None:
                state = _sensor_values(before, _REQUIRED_BEFORE)
                next_state = _sensor_values(after, _REQUIRED_AFTER)
                features = state + _issued_values(step['command']['issued'])
        except ValueError as exc:
            reason = f'required_signal_unavailable:{exc}'
        if reason is not None:
            exclusions.append({'index': step['index'], 'reason': reason,
                               'episode_id': provenance['episode_id']})
            continue
        current_targets = state[:6]+[state[12]]
        target = [later-now for later, now in zip(next_state, current_targets)]
        _vector(target, 7, 'target delta')
        rows.append({'features': features, 'target_delta': target,
                     'episode_id': provenance['episode_id'], 'session_id': provenance['session_id'],
                     'source_recording_id': provenance['source_recording_id'],
                     'seed': provenance['seed'], 'case': provenance['case'],
                     'index': step['index'], 'before_time_seconds': before['timestamp_seconds']})
    return {'rows': rows, 'exclusions': exclusions, 'attempted_steps': len(tape['steps'])}


def _training_rows(rows):
    if type(rows) not in (list, tuple) or not 2 <= len(rows) <= MAX_TRAINING_ROWS:
        raise ValueError('Training requires 2–200000 explicit rows.')
    detached = []
    for row in rows:
        if type(row) is not dict or set(row) not in (
                {'features', 'target_delta'}, {'features', 'target_delta'} | _PROVENANCE_KEYS):
            raise ValueError('Training row must contain causal features, target deltas and optional complete provenance.')
        item = {'features': _vector(row['features'], 20, 'features'),
                'target_delta': _vector(row['target_delta'], 7, 'target delta')}
        if _PROVENANCE_KEYS <= set(row):
            for name in ('episode_id', 'session_id', 'source_recording_id'):
                value = row[name]
                if type(value) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}', value) is None:
                    raise ValueError('Training provenance requires short source identifiers.')
            for name in ('seed', 'index'):
                if type(row[name]) is not int or not 0 <= row[name] <= 10**9:
                    raise ValueError('Training provenance requires bounded nonnegative integers.')
            if row['case'] not in ('nominal', 'open-jaw', 'release-after-lift'):
                raise ValueError('Unsupported simulation case.')
            _number(row['before_time_seconds'], 'training timestamp', minimum=0)
            item.update({name: row[name] for name in _PROVENANCE_KEYS})
        detached.append(item)
    return detached


def fit_model(rows, alpha=.01, feature_mode='action', sample_weights=None):
    """Fit weighted ridge from the supplied training partition only.

The caller owns episode/session splits and must exclude development and test
rows. Weighted centering makes the intercept unpenalized. Constant training
features have scale one and zero coefficients. No NumPy import occurs at load
or inference time.
"""
    rows = _training_rows(rows)
    alpha = _number(alpha, 'ridge alpha', minimum=0)
    if feature_mode not in ('action', 'state'):
        raise ValueError('Use action or state feature mode.')
    if sample_weights is None:
        weights = [1.]*len(rows)
    else:
        weights = _vector(sample_weights, len(rows), 'sample weights', minimum=0)
    if sum(weight > 0 for weight in weights) < 2:
        raise ValueError('At least two positive-weight training rows are required.')
    import numpy as np
    active = 20 if feature_mode == 'action' else 13
    features = np.asarray([row['features'][:active] for row in rows], dtype=np.float64)
    targets = np.asarray([row['target_delta'] for row in rows], dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    weight_sum = float(weight.sum())
    means = np.sum(features*weight[:, None], axis=0)/weight_sum
    variance = np.sum((features-means)**2*weight[:, None], axis=0)/weight_sum
    constants = np.sqrt(variance) <= 1e-12
    scales = np.where(constants, 1., np.sqrt(variance))
    normalized = (features-means)/scales
    normalized[:, constants] = 0.
    target_mean = np.sum(targets*weight[:, None], axis=0)/weight_sum
    weighted_x = normalized*np.sqrt(weight[:, None])
    weighted_y = (targets-target_mean)*np.sqrt(weight[:, None])
    matrix = np.vstack((weighted_x, np.sqrt(alpha)*np.eye(active)))
    rhs = np.vstack((weighted_y, np.zeros((active, 7))))
    fitted, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    fitted[constants] = 0.
    if not all(np.isfinite(value).all() for value in (means, scales, fitted, target_mean)):
        raise ValueError('Ridge fitting must produce finite coefficients.')
    model = {
        'schema_version': 1, 'kind': KIND, 'license': 'MIT', 'scope': 'simulation-only',
        'physical_robot_ready': False, 'is_controller': False, 'predicts_task_success': False,
        'sensor_origin': 'privileged-native-simulation', 'feature_mode': feature_mode,
        'feature_contract': FEATURE_CONTRACT, 'output_contract': OUTPUT_CONTRACT,
        'control_dt_seconds': CONTROL_DT_SECONDS, 'alpha': alpha,
        'normalization': {'means': means.tolist(), 'scales': scales.tolist(),
                          'constant_features': constants.tolist(), 'scope': 'fit-training-only'},
        'coefficients': fitted.T.tolist(), 'intercept': target_mean.tolist(),
        'training': {'rows': len(rows), 'positive_weight_rows': sum(x > 0 for x in weights),
                     'weight_sum': weight_sum, 'rows_sha256': digest(rows),
                     'sample_weights_sha256': digest(weights)},
    }
    model['model_sha256'] = digest(model)
    return validate_model(model)


def _decode_model(document):
    if type(document) is dict:
        try:
            raw = canonical(document)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError('Model must be finite JSON data.') from exc
    elif isinstance(document, (str, bytes)):
        try:
            raw = document.encode('utf8') if isinstance(document, str) else document
        except UnicodeError as exc:
            raise ValueError('Model must be finite UTF-8 JSON.') from exc
    else:
        raise ValueError('Model must be a plain dictionary or UTF-8 JSON.')
    if len(raw) > MAX_MODEL_BYTES:
        raise ValueError('Model exceeds 256 KiB.')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate model field.')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Model must contain finite values.')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError('Model must be finite UTF-8 JSON.') from exc


def validate_model(document):
    """Return detached bounded data, rejecting unknown fields and false scope."""
    model = _decode_model(document)
    _keys(model, _MODEL_KEYS, 'model')
    expected = {'schema_version': 1, 'kind': KIND, 'license': 'MIT', 'scope': 'simulation-only',
                'physical_robot_ready': False, 'is_controller': False, 'predicts_task_success': False,
                'sensor_origin': 'privileged-native-simulation', 'control_dt_seconds': CONTROL_DT_SECONDS}
    for key, value in expected.items():
        if type(model[key]) is not type(value) or model[key] != value:
            raise ValueError('Unsupported model kind, version, license, interval or qualification scope.')
    if canonical(model['feature_contract']) != canonical(FEATURE_CONTRACT) or canonical(model['output_contract']) != canonical(OUTPUT_CONTRACT):
        raise ValueError('Unsupported causal feature or output contract.')
    if model['feature_mode'] not in ('action', 'state'):
        raise ValueError('Unsupported feature mode.')
    active = 20 if model['feature_mode'] == 'action' else 13
    _number(model['alpha'], 'ridge alpha', minimum=0)
    normalization = model['normalization']
    _keys(normalization, {'means', 'scales', 'constant_features', 'scope'}, 'normalization')
    if normalization['scope'] != 'fit-training-only':
        raise ValueError('Normalization must be fitted on training rows only.')
    _vector(normalization['means'], active, 'normalization means')
    scales = _vector(normalization['scales'], active, 'normalization scales')
    if any(value <= 1e-12 for value in scales):
        raise ValueError('Normalization scales must exceed 1e-12.')
    constants = normalization['constant_features']
    if type(constants) is not list or len(constants) != active or any(type(value) is not bool for value in constants):
        raise ValueError('Constant feature mask must contain explicit booleans.')
    coefficients = model['coefficients']
    if type(coefficients) is not list or len(coefficients) != 7:
        raise ValueError('Coefficients require seven output rows.')
    for row in coefficients:
        _vector(row, active, 'coefficients', maximum=1e12)
        if any(constant and (coefficient != 0 or scale != 1.)
               for constant, coefficient, scale in zip(constants, row, scales)):
            raise ValueError('Constant training features must have scale one and zero coefficients.')
    _vector(model['intercept'], 7, 'intercept')
    training = model['training']
    _keys(training, {'rows', 'positive_weight_rows', 'weight_sum', 'rows_sha256', 'sample_weights_sha256'}, 'training')
    if any(type(training[name]) is not int or not 2 <= training[name] <= MAX_TRAINING_ROWS
           for name in ('rows', 'positive_weight_rows')) or training['positive_weight_rows'] > training['rows']:
        raise ValueError('Training counts must be bounded and consistent.')
    if _number(training['weight_sum'], 'weight sum', minimum=0, maximum=MAX_TRAINING_ROWS*MAX_NUMERIC_MAGNITUDE) <= 0:
        raise ValueError('Positive training weight required.')
    _sha(training['rows_sha256'], 'training hash')
    _sha(training['sample_weights_sha256'], 'sample weights hash')
    _sha(model['model_sha256'], 'model hash')
    if model['model_sha256'] != digest({key: value for key, value in model.items() if key != 'model_sha256'}):
        raise ValueError('Model hash mismatch.')
    return model


def _predict_validated(model, features):
    active = 20 if model['feature_mode'] == 'action' else 13
    normalization = model['normalization']
    normalized = [0. if constant else (value-mean)/scale
                  for value, mean, scale, constant in zip(features[:active], normalization['means'],
                                                         normalization['scales'], normalization['constant_features'])]
    try:
        delta = [math.fsum([offset] + [coefficient*value for coefficient, value in zip(row, normalized)])
                 for row, offset in zip(model['coefficients'], model['intercept'])]
    except OverflowError as exc:
        raise ValueError('Prediction arithmetic must remain finite.') from exc
    return _vector(delta, 7, 'predicted delta', maximum=1e12)


def predict_features(model, features20):
    """Strict standard-library inference; state ablation ignores command values."""
    model = validate_model(model)
    features = _vector(features20, 20, 'prediction features')
    return _predict_validated(model, features)


def predict(model, before_snapshot, issued_command):
    """Predict a boundary delta, abstaining on absent/stale/invalid observations.

This function emits no actuator command. It does not verify latency, the scene,
hardware compatibility, stopping distance, task completion or contact dynamics.
"""
    model = validate_model(model)
    try:
        features = _sensor_values(before_snapshot, _REQUIRED_BEFORE) + _issued_values(issued_command)
        delta = _predict_validated(model, features)
    except (ValueError, OverflowError) as exc:
        return {'status': 'abstained', 'reason': str(exc)}
    return {'status': 'predicted', 'delta': delta, 'control_dt_seconds': CONTROL_DT_SECONDS}


def write_model(path, model):
    """Create a bounded JSON artifact without overwriting any existing path."""
    model = validate_model(model)
    raw = canonical(model)+b'\n'
    if len(raw) > MAX_MODEL_BYTES:
        raise ValueError('Model exceeds 256 KiB.')
    with Path(path).open('xb') as output:
        output.write(raw)
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def load_model(path):
    """Load a regular, non-symlink bounded data file; no executable format."""
    path = Path(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0))
    except OSError as exc:
        raise ValueError('Model cannot be opened as a regular non-symlink file.') from exc
    with os.fdopen(descriptor, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MODEL_BYTES:
            raise ValueError('Model must be a regular file at most 256 KiB.')
        raw = source.read(MAX_MODEL_BYTES+1)
    return validate_model(raw)
