"""CPU behavior cloning of synthetic authored PiPER block demonstrations.

The student predicts all six native arm targets and the jaw command from a
fixed whitelist of privileged simulator measurements. It has no environment,
teacher, phase machine, outcome labels, IK call, or fallback at inference.
Supervised error alone does not establish closed-loop placement competence.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np


SCHEMA = 'dvidia.piper-imitation-mlp/v1'
ABSOLUTE_SCHEMA = 'dvidia.piper-imitation-mlp/v2'
FEATURE_CONTRACT = 'piper-privileged-measurements-43-v1'
SCOPE = ('Simulation-only behavioral cloning of synthetic authored-controller demonstrations; '
         'privileged native state, uncalibrated contact, no camera learning or physical robot qualification.')
JOINT_LIMITS = ((-2.618, 2.618), (0., 3.14), (-2.697, 0.),
                (-1.832, 1.832), (-1.22, 1.22), (-3.14, 3.14))
JOINT_DELTA_RAD = .06
JAW_MAX_M = .035
MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_WEIGHT = 64.
FEATURE_COUNT = 43
ACTION_COUNT = 7
FEATURE_NAMES = (tuple(f'joint_position_rad_{i}/3' for i in range(1, 7))
                 + tuple(f'joint_velocity_rad_s_{i}/3' for i in range(1, 7))
                 + ('left_jaw_m/.035', 'right_jaw_m/.035')
                 + tuple(f'tcp_{axis}_m/.5' for axis in 'xyz')
                 + tuple(f'tcp_quaternion_{axis}' for axis in 'wxyz')
                 + tuple(f'object_minus_tcp_{axis}_m/.2' for axis in 'xyz')
                 + tuple(f'target_minus_tcp_{axis}_m/.2' for axis in 'xyz')
                 + tuple(f'object_quaternion_{axis}' for axis in 'wxyz')
                 + tuple(f'object_velocity_{axis}_m_s/.2' for axis in 'xyz')
                 + tuple(f'object_angular_velocity_{axis}_rad_s/3' for axis in 'xyz')
                 + ('log1p_left_pad_force_n/3', 'log1p_right_pad_force_n/3',
                    'left_finger_contacts/2', 'right_finger_contacts/2',
                    'table_object_contacts/2', 'episode_elapsed_seconds/20'))
MODEL_FIELDS = {'schema', 'scope', 'physical_robot_ready', 'observation_origin',
                'supervision_origin', 'feature_contract', 'feature_names', 'joint_limits_rad',
                'joint_delta_limit_rad', 'jaw_limit_m', 'hidden_units',
                'feature_mean', 'feature_scale', 'weights', 'biases',
                'dataset_sha256', 'train_cases', 'training'}
ABSOLUTE_MODEL_FIELDS = MODEL_FIELDS | {'action_representation', 'action_mean', 'action_scale'}


def canonical(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def digest(document):
    return sha256(canonical(document)).hexdigest()


def _number(value, name, low, high):
    if (not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_))
            or not low <= value <= high or not math.isfinite(value)):
        raise ValueError(f'{name} must be a finite number in [{low},{high}].')
    return float(value)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low},{high}].')
    return value


def _array(value, shape, name, maximum=1e6):
    def finite_numbers(row):
        if isinstance(row, np.ndarray):
            return row.dtype.kind in 'iuf' and bool(np.isfinite(row).all())
        if type(row) in (list, tuple):
            return all(finite_numbers(item) for item in row)
        return (isinstance(row, (int, float, np.integer, np.floating)) and not isinstance(row, (bool, np.bool_))
                and -maximum <= row <= maximum and math.isfinite(row))
    try:
        if not finite_numbers(value):
            raise ValueError(f'{name} must contain finite numeric values.')
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError(f'{name} must contain finite numeric values.') from exc
    if result.shape != shape or not np.isfinite(result).all() or np.any(np.abs(result) > maximum):
        raise ValueError(f'{name} has invalid dimensions or range.')
    return result


def _quaternion(value, name):
    quat = _array(value, (4,), name, 1.01)
    norm = float(np.linalg.norm(quat))
    if not .99 <= norm <= 1.01:
        raise ValueError(f'{name} must be a unit quaternion.')
    quat = quat / norm
    # q and -q describe the same rotation; choose one deterministically.
    nonzero = np.flatnonzero(np.abs(quat) > 1e-12)
    return quat if quat[nonzero[0]] >= 0 else -quat


def features(observation):
    """43 bounded measurements; extra diagnostics/outcome/phase fields are ignored.

    Order: q6, qdot6, jaw2, TCP xyz3, TCP wxyz4, object-minus-TCP3,
    target-minus-TCP3, object wxyz4, object linear3/angular3 velocity,
    log pad forces2, finger contacts2, table contacts1, elapsed seconds1.
    Time is a measured episode clock, never the teacher's phase elapsed time.
    """
    if type(observation) is not dict:
        raise ValueError('A PiPER observation dictionary is required.')
    try:
        q = _array(observation['joint_position_rad'], (6,), 'joint_position_rad', 10)
        qdot = _array(observation['joint_velocity_rad_s'], (6,), 'joint_velocity_rad_s', 1e4)
        jaw = _array(observation['gripper_joint_position_m'], (2,), 'gripper_joint_position_m', 1)
        tcp = _array(observation['tcp_position_m'], (3,), 'tcp_position_m', 10)
        obj = _array(observation['object_position_m'], (3,), 'object_position_m', 10)
        target = _array(observation['target_position_m'], (3,), 'target_position_m', 10)
        tcp_quat = _quaternion(observation['tcp_quaternion_wxyz'], 'tcp_quaternion_wxyz')
        obj_quat = _quaternion(observation['object_quaternion_wxyz'], 'object_quaternion_wxyz')
        linear = _array(observation['object_linear_velocity_m_s'], (3,), 'object_linear_velocity_m_s', 1e4)
        angular = _array(observation['object_angular_velocity_rad_s'], (3,), 'object_angular_velocity_rad_s', 1e4)
        forces = [_number(observation['pad_normal_forces_n'][side], 'pad force', 0, 1e6)
                  for side in ('left', 'right')]
        contacts = [_integer(observation['grasp_contact_counts'][side], 'finger contacts', 0, 10000)
                    for side in ('left', 'right')]
        table = _integer(observation['table_object_contact_count'], 'table contacts', 0, 10000)
        elapsed = _number(observation['t_seconds'], 't_seconds', 0, 1e4)
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or invalid PiPER measurement.') from exc
    result = np.concatenate((q / 3., qdot / 3., jaw / JAW_MAX_M, tcp / .5,
                             tcp_quat, (obj-tcp) / .2, (target-tcp) / .2,
                             obj_quat, linear / .2, angular / 3.,
                             np.log1p(forces) / 3., np.asarray(contacts) / 2.,
                             [table / 2., elapsed / 20.]))
    return np.clip(result, -10., 10.).astype(np.float32)


def training_sample(observation, action, case_id):
    """Encode a causal pre-action observation and its teacher-issued command."""
    _case_ids([case_id])
    if type(action) is not dict or set(action) != {'joint_targets_rad', 'gripper_position_m'}:
        raise ValueError('Teacher action must use native arm/jaw targets.')
    q = _array(observation['joint_position_rad'], (6,), 'joint_position_rad', 10)
    targets = _array(action['joint_targets_rad'], (6,), 'joint_targets_rad', 10)
    limits = np.asarray(JOINT_LIMITS)
    if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
        raise ValueError('Teacher command exceeds pinned native joint limits.')
    jaw = _number(action['gripper_position_m'], 'gripper_position_m', 0, JAW_MAX_M)
    normalized = np.concatenate((np.clip((targets-q) / JOINT_DELTA_RAD, -1., 1.), [jaw/JAW_MAX_M]))
    return {'features': features(observation).tolist(), 'targets': normalized.tolist(), 'case_id': case_id}


def _case_ids(value):
    if (type(value) is not list or not 1 <= len(value) <= 10000
            or any(type(item) is not str or not item or len(item) > 100 for item in value)
            or len(set(value)) != len(value)):
        raise ValueError('Case IDs must be distinct nonempty short strings.')
    return value


def _decode(document):
    if type(document) is dict:
        try:
            document = canonical(document)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError('Model must be finite data-only JSON.') from exc
    if not isinstance(document, (bytes, str)):
        raise ValueError('Model must be a JSON dictionary, bytes, or string.')
    try:
        raw = document.encode('utf8') if isinstance(document, str) else document
    except UnicodeError as exc:
        raise ValueError('Model must be finite UTF-8 JSON.') from exc
    if len(raw) > MAX_MODEL_BYTES:
        raise ValueError('PiPER imitation model exceeds 8 MiB.')
    def pairs(items):
        row = {}
        for key, value in items:
            if key in row:
                raise ValueError('Duplicate model field.')
            row[key] = value
        return row
    def nonfinite(value):
        raise ValueError('Nonfinite JSON value.')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Model must be finite UTF-8 JSON.') from exc


def validate_model(document):
    """Strict bounded data validation; accepts no executable serialization."""
    model = _decode(document)
    if type(model) is dict and model.get('schema') in ('dvidia.piper-motor-primitive/v1', 'dvidia.piper-motor-primitive/v2'):
        from .piper_motor_primitive import validate_model as validate_primitive
        return validate_primitive(model)
    absolute = type(model) is dict and model.get('schema') == ABSOLUTE_SCHEMA
    if type(model) is not dict or set(model) != (ABSOLUTE_MODEL_FIELDS if absolute else MODEL_FIELDS):
        raise ValueError('Unknown or missing PiPER imitation model field.')
    expected = {'schema': ABSOLUTE_SCHEMA if absolute else SCHEMA, 'scope': SCOPE, 'physical_robot_ready': False,
                'observation_origin': 'privileged_native_simulator_state',
                'supervision_origin': 'synthetic_authored_controller',
                'feature_contract': FEATURE_CONTRACT, 'joint_delta_limit_rad': JOINT_DELTA_RAD,
                'jaw_limit_m': JAW_MAX_M}
    if any(type(model[name]) is not type(value) or model[name] != value for name, value in expected.items()):
        raise ValueError('Unsupported model scope, observation, or native action contract.')
    if model['feature_names'] != list(FEATURE_NAMES):
        raise ValueError('Unsupported input feature order.')
    if absolute:
        if model['action_representation'] != 'absolute':
            raise ValueError('V2 requires learned absolute native arm targets.')
        _array(model['action_mean'], (ACTION_COUNT,), 'action_mean', 2.)
        action_scale = _array(model['action_scale'], (ACTION_COUNT,), 'action_scale', 2.)
        if np.any(action_scale < .0001):
            raise ValueError('Action scales must be at least .0001.')
    limits = _array(model['joint_limits_rad'], (6, 2), 'joint_limits_rad', 10)
    if not np.array_equal(limits, np.asarray(JOINT_LIMITS)):
        raise ValueError('Model cannot change pinned native joint limits.')
    hidden = _integer(model['hidden_units'], 'hidden_units', 8, 128)
    _array(model['feature_mean'], (FEATURE_COUNT,), 'feature_mean', 10)
    scale = _array(model['feature_scale'], (FEATURE_COUNT,), 'feature_scale', 20)
    if np.any(scale < .01):
        raise ValueError('Feature scales must be at least .01.')
    if type(model['weights']) is not list or type(model['biases']) is not list or len(model['weights']) != 3 or len(model['biases']) != 3:
        raise ValueError('Exactly two hidden layers and one action layer are required.')
    for index, shape in enumerate(((FEATURE_COUNT, hidden), (hidden, hidden), (hidden, ACTION_COUNT))):
        _array(model['weights'][index], shape, f'weights[{index}]', MAX_WEIGHT)
        _array(model['biases'][index], (shape[1],), f'biases[{index}]', MAX_WEIGHT)
    ds = model['dataset_sha256']
    if type(ds) is not str or len(ds) != 64 or any(c not in '0123456789abcdef' for c in ds):
        raise ValueError('A dataset SHA256 is required.')
    _case_ids(model['train_cases'])
    receipt = model['training']
    receipt_fields = {'algorithm', 'sample_count', 'epochs', 'batch_size', 'learning_rate', 'seed', 'initial_loss', 'final_loss'}
    if absolute:
        receipt_fields |= {'label_shuffle_seed', 'source_dataset_sha256'}
    if type(receipt) is not dict or set(receipt) != receipt_fields:
        raise ValueError('Invalid training receipt.')
    if receipt['algorithm'] != ('numpy-tanh-mlp-adam-standardized-mse' if absolute else 'numpy-tanh-mlp-adam-mse'):
        raise ValueError('Unsupported training algorithm.')
    _integer(receipt['sample_count'], 'sample_count', 16, 1000000)
    _integer(receipt['epochs'], 'epochs', 1, 10000)
    _integer(receipt['batch_size'], 'batch_size', 8, 8192)
    _integer(receipt['seed'], 'seed', 0, 999999)
    _number(receipt['learning_rate'], 'learning_rate', 1e-5, .1)
    for name in ('initial_loss', 'final_loss'):
        _number(receipt[name], name, 0, 1e6)
    if absolute:
        if receipt['label_shuffle_seed'] is not None:
            _integer(receipt['label_shuffle_seed'], 'label_shuffle_seed', 0, 999999)
        source_ds = receipt['source_dataset_sha256']
        if type(source_ds) is not str or len(source_ds) != 64 or any(c not in '0123456789abcdef' for c in source_ds):
            raise ValueError('A source dataset SHA256 is required.')
    return model


def _forward(x, weights, biases):
    h1 = np.tanh(x @ weights[0] + biases[0])
    h2 = np.tanh(h1 @ weights[1] + biases[1])
    return h1, h2, h2 @ weights[2] + biases[2]


def train(dataset, *, hidden_units=64, epochs=500, batch_size=256, learning_rate=.003, seed=0,
          action_representation='delta', label_shuffle_seed=None):
    """Deterministic CPU mini-batch Adam fit, using only training sample data.

    Dataset: ``{'scene_ids': [id, ...], 'samples': [training_sample(...), ...]}``.
    No success/phase labels or held-out scenes are consumed. Any sample balancing
    or data aggregation must be declared by the caller in its study receipt.
    ``absolute`` reconstructs teacher targets from the existing delta labels,
    then learns qtarget/3 and jaw/.035 with training-only output standardization.
    V1 ``delta`` behavior and loading remain unchanged.
    Optional V2 label_shuffle_seed permutes complete converted absolute action
    vectors across training states; the transformed supervision hash is saved.
    """
    hidden = _integer(hidden_units, 'hidden_units', 8, 128)
    epochs = _integer(epochs, 'epochs', 1, 10000)
    batch_size = _integer(batch_size, 'batch_size', 8, 8192)
    seed = _integer(seed, 'seed', 0, 999999)
    learning_rate = _number(learning_rate, 'learning_rate', 1e-5, .1)
    if type(action_representation) is not str or action_representation not in ('delta', 'absolute'):
        raise ValueError('action_representation must be delta or absolute.')
    absolute = action_representation == 'absolute'
    if label_shuffle_seed is not None:
        _integer(label_shuffle_seed, 'label_shuffle_seed', 0, 999999)
        if not absolute:
            raise ValueError('Internal full-label shuffle is supported only for absolute V2 training.')
    if type(dataset) is not dict or set(dataset) != {'scene_ids', 'samples'}:
        raise ValueError('Dataset requires scene_ids and causal samples only.')
    cases = _case_ids(dataset['scene_ids'])
    rows = dataset['samples']
    if type(rows) is not list or not 16 <= len(rows) <= 1000000:
        raise ValueError('Training requires 16–1000000 samples.')
    case_set = set(cases)
    seen = set()
    x, y = [], []
    for row in rows:
        if type(row) is not dict or set(row) != {'features', 'targets', 'case_id'}:
            raise ValueError('Samples require features, targets, and case_id only.')
        if type(row['case_id']) is not str or row['case_id'] not in case_set:
            raise ValueError('Sample case is not declared in scene_ids.')
        seen.add(row['case_id'])
        x.append(_array(row['features'], (FEATURE_COUNT,), 'features', 10))
        target = _array(row['targets'], (ACTION_COUNT,), 'targets', 1)
        if target[6] < 0:
            raise ValueError('Normalized jaw targets must lie in [0,1].')
        y.append(target)
    if seen != case_set:
        raise ValueError('Every declared training scene must contain samples.')
    # Hash the validated numeric dataset so list and NumPy row inputs agree.
    dataset_hash = digest({'scene_ids': list(cases), 'samples': [
        {'features': a.tolist(), 'targets': b.tolist(), 'case_id': row['case_id']}
        for a, b, row in zip(x, y, rows)]})
    raw_x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if absolute:
        source_dataset_hash = dataset_hash
        # The dataset retains causal state and q-relative labels. No teacher or
        # trajectory lookup is needed to recover each issued native target.
        joint_targets = raw_x[:, :6]*3. + y[:, :6]*JOINT_DELTA_RAD
        limits = np.asarray(JOINT_LIMITS, dtype=np.float32)
        if np.any(joint_targets < limits[:, 0]-1e-5) or np.any(joint_targets > limits[:, 1]+1e-5):
            raise ValueError('Reconstructed absolute targets exceed pinned native joint limits.')
        y = y.copy()
        y[:, :6] = np.clip(joint_targets, limits[:, 0], limits[:, 1])/3.
        if label_shuffle_seed is not None:
            y = y[np.random.default_rng(label_shuffle_seed).permutation(len(y))]
        dataset_hash = digest({'scene_ids': list(cases), 'samples': [
            {'features': a.tolist(), 'targets': b.tolist(), 'case_id': row['case_id']}
            for a, b, row in zip(raw_x, y, rows)]})
        action_mean = y.mean(axis=0, dtype=np.float64).astype(np.float32)
        action_scale = np.maximum(y.std(axis=0, dtype=np.float64), .00010001).astype(np.float32)
        y = (y-action_mean)/action_scale
    mean = raw_x.mean(axis=0)
    scale = np.maximum(raw_x.std(axis=0), .010001)
    x = np.clip((raw_x-mean)/scale, -10., 10.)
    rng = np.random.default_rng(seed)
    shapes = ((FEATURE_COUNT, hidden), (hidden, hidden), (hidden, ACTION_COUNT))
    weights = [rng.normal(0, math.sqrt(2/(a+b)), (a, b)).astype(np.float32) for a, b in shapes]
    biases = [np.zeros(b, dtype=np.float32) for _, b in shapes]
    parameters = weights + biases
    first = [np.zeros_like(p) for p in parameters]
    second = [np.zeros_like(p) for p in parameters]
    initial_loss = float(np.mean((_forward(x, weights, biases)[2]-y)**2))
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), batch_size):
            indices = order[start:start+batch_size]
            xb, yb = x[indices], y[indices]
            h1, h2, prediction = _forward(xb, weights, biases)
            d3 = (prediction-yb) * (2/(len(indices)*ACTION_COUNT))
            d2 = (d3 @ weights[2].T) * (1-h2*h2)
            d1 = (d2 @ weights[1].T) * (1-h1*h1)
            gradients = [xb.T @ d1, h1.T @ d2, h2.T @ d3,
                         d1.sum(axis=0), d2.sum(axis=0), d3.sum(axis=0)]
            step += 1
            for param, grad, m, v in zip(parameters, gradients, first, second):
                np.clip(grad, -1., 1., out=grad)
                m *= .9
                m += .1*grad
                v *= .999
                v += .001*grad*grad
                param -= learning_rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
                np.clip(param, -MAX_WEIGHT, MAX_WEIGHT, out=param)
    final_loss = float(np.mean((_forward(x, weights, biases)[2]-y)**2))
    model = {'schema': ABSOLUTE_SCHEMA if absolute else SCHEMA, 'scope': SCOPE, 'physical_robot_ready': False,
             'observation_origin': 'privileged_native_simulator_state',
             'supervision_origin': 'synthetic_authored_controller',
             'feature_contract': FEATURE_CONTRACT, 'feature_names': list(FEATURE_NAMES),
             'joint_limits_rad': [list(row) for row in JOINT_LIMITS],
             'joint_delta_limit_rad': JOINT_DELTA_RAD, 'jaw_limit_m': JAW_MAX_M,
             'hidden_units': hidden, 'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
             'weights': [value.tolist() for value in weights], 'biases': [value.tolist() for value in biases],
             'dataset_sha256': dataset_hash, 'train_cases': list(cases),
             'training': {'algorithm': 'numpy-tanh-mlp-adam-standardized-mse' if absolute else 'numpy-tanh-mlp-adam-mse', 'sample_count': len(rows),
                          'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
                          'seed': seed, 'initial_loss': initial_loss, 'final_loss': final_loss}}
    if absolute:
        model.update(action_representation='absolute', action_mean=action_mean.tolist(), action_scale=action_scale.tolist())
        model['training'].update(label_shuffle_seed=label_shuffle_seed, source_dataset_sha256=source_dataset_hash)
    return validate_model(model)


def save_model(document, path):
    """Save validated JSON; no pickle, imports, callbacks, or executable payloads."""
    model = validate_model(document)
    Path(path).write_bytes(canonical(model) + b'\n')
    return digest(model)


def load_model(path):
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_MODEL_BYTES+1)
    return validate_model(raw)


class LearnedPiperPolicy:
    """A stateless learned action policy; instantiate afresh for each episode."""
    def __init__(self, document):
        self.model = validate_model(document)
        self.primitive = None
        if self.model['schema'] in ('dvidia.piper-motor-primitive/v1', 'dvidia.piper-motor-primitive/v2'):
            from .piper_motor_primitive import MotorPrimitivePolicy
            self.primitive = MotorPrimitivePolicy(self.model)
            return
        self.mean = np.asarray(self.model['feature_mean'], dtype=np.float32)
        self.scale = np.asarray(self.model['feature_scale'], dtype=np.float32)
        self.weights = [np.asarray(value, dtype=np.float32) for value in self.model['weights']]
        self.biases = [np.asarray(value, dtype=np.float32) for value in self.model['biases']]
        self.absolute = self.model['schema'] == ABSOLUTE_SCHEMA
        if self.absolute:
            self.action_mean = np.asarray(self.model['action_mean'], dtype=np.float32)
            self.action_scale = np.asarray(self.model['action_scale'], dtype=np.float32)
        self.calls = 0
        self.inference_seconds = 0.

    def normalized_actions(self, observations):
        """Batch outputs: V1 deltas/.06 or V2 absolute targets/3; jaw/.035."""
        if self.primitive is not None:
            raise ValueError('A time-conditioned primitive requires sequential native calls starting at t=0.')
        if type(observations) not in (list, tuple) or not observations:
            raise ValueError('A nonempty observation batch is required.')
        x = np.asarray([features(row) for row in observations], dtype=np.float32)
        x = np.clip((x-self.mean)/self.scale, -10., 10.)
        output = _forward(x, self.weights, self.biases)[2]
        if self.absolute:
            output = output*self.action_scale+self.action_mean
            limits = np.asarray(JOINT_LIMITS)/3.
            return np.clip(output, np.concatenate((limits[:, 0], [0.])), np.concatenate((limits[:, 1], [1.])))
        return np.clip(output, [-1.]*6+[0.], [1.]*7)

    def predict(self, observation):
        if self.primitive is not None:
            return self.primitive.predict(observation)
        started = time.perf_counter()
        output = self.normalized_actions([observation])[0]
        q = _array(observation['joint_position_rad'], (6,), 'joint_position_rad', 10)
        limits = np.asarray(JOINT_LIMITS)
        delta = np.clip(3.*output[:6]-q, -JOINT_DELTA_RAD, JOINT_DELTA_RAD) if self.absolute else JOINT_DELTA_RAD*output[:6]
        targets = np.clip(q + delta, limits[:, 0]+.02, limits[:, 1]-.02)
        action = {'joint_targets_rad': targets.tolist(), 'gripper_position_m': float(JAW_MAX_M*output[6])}
        self.calls += 1
        self.inference_seconds += time.perf_counter()-started
        return action

    __call__ = predict

    def diagnostics(self):
        if self.primitive is not None:
            return self.primitive.diagnostics()
        return {'policy_origin': 'learned_numpy_mlp', 'observations': 'privileged simulator measurements',
                'supervision_origin': 'synthetic authored controller', 'student_inference_calls': self.calls,
                'student_inference_seconds': self.inference_seconds, 'teacher_fallback_calls': 0,
                'runtime_teacher_available': False, 'authored_task_phase_machine': False,
                'action_representation': 'absolute' if self.absolute else 'delta',
                'physical_robot_ready': False, 'model_sha256': digest(self.model)}
