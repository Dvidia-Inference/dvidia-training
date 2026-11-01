"""Small causal RGB-D/encoder policy, trained on explicit native action labels.

The network never receives a simulator, teacher, task phase, current target,
contact truth or a future snapshot. Its visual anchor is historical context,
not a held-forward current measurement. Missing live estimates stay missing.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import time

import numpy as np

from . import piper_imitation as core
from . import piper_rgbd_tracker as tracker


SCHEMA = 'dvidia.camera-action-mlp/v1'
DATASET_SCHEMA = 'dvidia.camera-action-learning/v1'
FEATURE_CONTRACT = 'camera-encoder-absolute-causal160/v1'
SCOPE = ('Simulation-only camera/encoder behavioral cloning of a privileged synthetic '
         'teacher; known upright block, no human-video motor recovery or physical qualification.')
LAGS = (0., .04, .08, .12, .16)
SENSOR_SPECS = (
    ('joint_position_rad', 6, 3.), ('joint_velocity_rad_s', 6, 3.),
    ('gripper_joint_position_m', 2, .035), ('gripper_joint_velocity_m_s', 2, .035),
    ('tcp_position_estimate_m', 3, .5), ('tcp_quaternion_estimate_wxyz', 4, 1.),
)
CONTEXT_NAMES = tuple(f'initial_visual_{a}_m/.5' for a in 'xyz') + (
    'initial_visual_captured_at_s/20', 'initial_visual_uncertainty_m/.02',
) + tuple(f'goal_{a}_m/.5' for a in 'xyz') + ('elapsed_s/20',)
COMMAND_NAMES = tuple(f'previous_applied_joint_{i}_rad/3' for i in range(1, 7)) + (
    'previous_applied_jaw_m/.035', 'history_present', 'history_age_s/.16',)
SENSOR_NAMES = tuple(name for field, size, scale in SENSOR_SPECS
    for name in tuple(f'{field}_{i}/{scale:g}' for i in range(size))
    + (field+'_available', field+'_age_s/.1')) + (
        'visual_x_m/.5', 'visual_y_m/.5', 'visual_z_m/.5',
        'visual_uncertainty_m/.02', 'visual_available', 'visual_age_s/.1',
        'history_present', 'history_age_s/.16',)
FEATURE_NAMES = CONTEXT_NAMES + tuple(f'lag{round(lag*1000)}:{name}'
    for lag in LAGS for name in COMMAND_NAMES) + tuple(f'lag{round(lag*1000)}:{name}'
    for lag in LAGS for name in SENSOR_NAMES)
FEATURE_COUNT = len(FEATURE_NAMES)
LIVE_FEATURE_START = len(CONTEXT_NAMES) + len(LAGS)*len(COMMAND_NAMES)
ACTION_SCALE_NATIVE = np.asarray([3.]*6 + [core.JAW_MAX_M], dtype=np.float64)
MODEL_FIELDS = {'schema', 'scope', 'physical_robot_ready', 'observation_origin',
    'supervision_origin', 'feature_contract', 'feature_names', 'feature_contract_sha256',
    'blind', 'hidden_units', 'joint_limits_rad', 'jaw_limit_m', 'action_representation',
    'feature_mean', 'feature_scale', 'action_mean', 'action_scale', 'weights', 'biases', 'training'}
TRAINING_FIELDS = {'algorithm', 'stage', 'seed', 'epochs', 'batch_size', 'learning_rate',
    'sample_count', 'train_groups', 'dataset_sha256', 'provenance', 'provenance_sha256',
    'model_source_sha256', 'initial_loss', 'final_loss', 'validation'}


def canonical(document):
    return core.canonical(document)


def digest(document):
    return sha256(canonical(document)).hexdigest()


FEATURE_CONTRACT_SHA256 = digest({'contract': FEATURE_CONTRACT, 'names': FEATURE_NAMES,
    'lags': LAGS, 'history_seconds': .16, 'camera_maximum_age_seconds': .1,
    'blind_live_start': LIVE_FEATURE_START,
    'previous_applied': 'preceding interval final native servo references; never measured q',
    'initial_anchor': 'fresh t=0 visual xyz and heuristic uncertainty; historical thereafter'})


def _hash(value, label):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(label+' must be a lowercase SHA-256 digest.')
    return value


def action_vector(action):
    """Validate seven *absolute* issued/query targets, without interpreting authority."""
    if type(action) is not dict or set(action) != {'joint_targets_rad', 'gripper_position_m'}:
        raise ValueError('An action must contain only six joint targets and one jaw target.')
    q = core._array(action['joint_targets_rad'], (6,), 'joint targets', 10.)
    limits = np.asarray(core.JOINT_LIMITS)
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError('Action exceeds the original native joint bounds.')
    jaw = core._number(action['gripper_position_m'], 'jaw target', 0., core.JAW_MAX_M)
    return np.concatenate((q, [jaw]))


def _time(value, name):
    return core._number(value, name, 0., 20.)


def _sample(sample, size, now, name):
    if type(sample) is not dict or sample.get('status') not in ('available', 'missing', 'stale', 'refused'):
        raise ValueError(name+' has an invalid availability state.')
    if sample['status'] != 'available':
        if sample.get('value') is not None:
            raise ValueError(name+' must not forward-fill unavailable values.')
        return np.zeros(size), 0., 0.
    captured = _time(sample.get('captured_at_seconds'), name+' capture time')
    available = _time(sample.get('available_at_seconds'), name+' availability time')
    age = _time(sample.get('age_seconds'), name+' age')
    if (captured > available+1e-10 or available > now+1e-10
            or abs(age-(now-captured)) > 1e-8 or age > .1+1e-10):
        raise ValueError(name+' is future, stale or incorrectly age-stamped.')
    return core._array(sample.get('value'), (size,), name, 1e4), 1., age


def _visual(estimate, now, snapshot):
    if type(estimate) is not dict or estimate.get('schema') != tracker.SCHEMA:
        raise ValueError('A frozen tracker result is required.')
    if estimate.get('status') not in ('available', 'missing', 'stale', 'refused'):
        raise ValueError('Invalid tracker status.')
    decision = _time(estimate.get('decision_at_seconds'), 'tracker decision time')
    if abs(decision-now) > 1e-8:
        raise ValueError('Tracker decision time must match this motor boundary.')
    camera = snapshot.get('camera')
    if type(camera) is not dict or camera.get('status') != 'available' or type(camera.get('value')) is not dict:
        raise ValueError('Visual estimates require their available source camera frame.')
    index = camera['value'].get('index')
    if (type(index) is not int or index < 0 or estimate.get('frame_index') != index
            or snapshot.get('vision_frame_index') != index):
        raise ValueError('Tracker frame index differs from the source camera snapshot.')
    captured = _time(camera.get('captured_at_seconds'), 'camera capture time')
    available = _time(camera.get('available_at_seconds'), 'camera availability time')
    camera_age = _time(camera.get('age_seconds'), 'camera age')
    if (captured > available+1e-10 or available > now+1e-10
            or abs(camera_age-(now-captured)) > 1e-8):
        raise ValueError('Source camera timestamps are noncausal or incorrectly aged.')
    if any(estimate.get(field) != camera.get(field)
            for field in ('captured_at_seconds', 'available_at_seconds', 'age_seconds')):
        raise ValueError('Tracker source timestamps differ from the camera snapshot.')
    signals = estimate.get('signals')
    if type(signals) is not dict:
        raise ValueError('Tracker signals are required.')
    p, present, age = _sample(signals.get('object_position_estimate_m'), 3, now, 'visual position')
    u, uncertainty_present, uncertainty_age = _sample(signals.get('object_position_uncertainty_m'), 1, now, 'visual uncertainty')
    if present != uncertainty_present or abs(age-uncertainty_age) > 1e-8:
        raise ValueError('Visual position and uncertainty must share availability and age.')
    if bool(present) != (estimate['status'] == 'available'):
        raise ValueError('Tracker status and visual signal availability disagree.')
    if present:
        if u[0] < 0. or u[0] > 1.:
            raise ValueError('Visual uncertainty must be a nonnegative bounded scale.')
        top = core._array(estimate.get('object_position_estimate_m'), (3,), 'tracker position', 1e4)
        if not np.array_equal(p, top) or estimate.get('object_position_uncertainty_m') != float(u[0]):
            raise ValueError('Tracker position/uncertainty receipt disagrees with signals.')
        for name in ('captured_at_seconds', 'available_at_seconds', 'age_seconds'):
            if (estimate.get(name) != signals['object_position_estimate_m'].get(name)
                    or estimate.get(name) != signals['object_position_uncertainty_m'].get(name)):
                raise ValueError('Tracker top-level and signal times disagree.')
    else:
        if estimate.get('object_position_estimate_m') is not None or estimate.get('object_position_uncertainty_m') is not None:
            raise ValueError('Unavailable visual estimates must be null.')
    # An abstention has no pose age, but its source image still has a known
    # acquisition age. Preserve that age next to the zero position and mask.
    return p, float(u[0]), present, camera_age


class CameraActionFeatures:
    """Fresh per episode. Five causal slots span 160 ms; warmup slots are absent.

    The blind comparator retains the same initial camera anchor and previous
    applied command history. It receives no live sensor values, masks or ages.
    Its command history is the preceding native reference, not measured joints.
    """
    def __init__(self, *, blind=False):
        if type(blind) is not bool:
            raise ValueError('blind must be boolean.')
        self.blind = blind
        self.history = deque()
        self.initial = self.goal = None
        self.last_time = None

    def features(self, context):
        if type(context) is not dict or type(context.get('snapshot')) is not dict:
            raise ValueError('Camera-action context and pre-action snapshot are required.')
        snapshot = context['snapshot']
        now = _time(snapshot.get('timestamp_seconds'), 'snapshot time')
        if self.last_time is None and now != 0.:
            raise ValueError('Camera-action episodes must start at t=0.')
        if self.last_time is not None and not math.isclose(now-self.last_time, .02, abs_tol=1e-8, rel_tol=0.):
            raise ValueError('Camera-action observations require increasing 20 ms boundaries.')
        goal = core._array(context.get('goal_position_m'), (3,), 'goal position', 1.)
        if self.goal is not None and not np.array_equal(goal, self.goal):
            raise ValueError('The destination cannot change during an episode.')
        command = action_vector(context.get('previous_applied'))/ACTION_SCALE_NATIVE
        visual = None
        if self.initial is None or not self.blind:
            visual = _visual(context.get('estimate'), now, snapshot)
        initial = self.initial
        if initial is None:
            p, uncertainty, present, age = visual
            if not present or age > 1e-10:
                raise ValueError('A fresh initial visual anchor is required before the first student action.')
            initial = np.concatenate((p/.5, [0., uncertainty/.02]))
        if self.blind:
            sensed = np.zeros(len(SENSOR_NAMES)-2)
        else:
            signals = snapshot.get('signals')
            if type(signals) is not dict:
                raise ValueError('Measured encoder/FK signals are required.')
            values = []
            for name, size, scale in SENSOR_SPECS:
                value, present, age = _sample(signals.get(name), size, now, name)
                if present and name == 'tcp_quaternion_estimate_wxyz':
                    value = core._quaternion(value, name)
                values.extend(np.clip(value/scale, -10., 10.))
                values.extend((present, age/.1))
            p, uncertainty, present, age = visual
            values.extend(np.clip(p/.5, -10., 10.))
            values.extend((uncertainty/.02, present, age/.1))
            sensed = np.asarray(values)
        pending = [row for row in self.history if row[0] >= now-.16-1e-8]
        pending.append((now, command.copy(), sensed.copy()))
        selected = []
        for lag in LAGS:
            eligible = [row for row in pending if row[0] <= now-lag+1e-8]
            selected.append(eligible[-1] if eligible else None)
        command_slots, sensed_slots = [], []
        for row in selected:
            if row is None:
                command_slots.extend(np.zeros(len(COMMAND_NAMES)))
                sensed_slots.extend(np.zeros(len(SENSOR_NAMES)))
            else:
                age = max(0., now-row[0])/.16
                command_slots.extend([*row[1], 1., age])
                sensed_slots.extend(np.zeros(len(SENSOR_NAMES)) if self.blind else [*row[2], 1., age])
        result = np.concatenate((initial, goal/.5, [now/20.], command_slots, sensed_slots))
        if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all() or np.any(np.abs(result) > 10.+1e-8):
            raise ValueError('Camera-action features exceed the bounded finite contract.')
        self.history = deque(pending)
        self.initial, self.goal, self.last_time = initial.copy(), goal.copy(), now
        return result.astype(np.float32)

    __call__ = features


def _dataset(dataset, *, blind):
    if type(dataset) is not dict or set(dataset) != {'schema', 'features', 'targets', 'groups', 'provenance'} or dataset['schema'] != DATASET_SCHEMA:
        raise ValueError('Use a camera-action mask-aware dataset with provenance.')
    groups = dataset['groups']
    if type(groups) not in (list, tuple) or not 1 <= len(groups) <= 1_000_000:
        raise ValueError('Dataset requires 1 to 1000000 grouped samples.')
    groups = list(groups)
    if any(type(group) is not str for group in groups):
        raise ValueError('Every training sample requires a string source-group ID.')
    core._case_ids(list(dict.fromkeys(groups)))
    x = core._array(dataset['features'], (len(groups), FEATURE_COUNT), 'features', 10.+1e-7)
    y = core._array(dataset['targets'], (len(groups), 7), 'absolute action targets', 10.)
    limits = np.asarray(core.JOINT_LIMITS)
    if (np.any(y[:, :6] < limits[:, 0]) or np.any(y[:, :6] > limits[:, 1])
            or np.any(y[:, 6] < 0.) or np.any(y[:, 6] > core.JAW_MAX_M)):
        raise ValueError('Training targets exceed original native bounds.')
    if blind and np.any(x[:, LIVE_FEATURE_START:] != 0.):
        raise ValueError('Blind datasets must zero all live/history feedback features.')
    provenance = dataset['provenance']
    if type(provenance) is not dict or not provenance:
        raise ValueError('Dataset provenance is required.')
    provenance = core._decode(canonical(provenance))
    h = sha256(canonical({'schema': DATASET_SCHEMA, 'features': list(x.shape),
        'targets': list(y.shape), 'groups': groups, 'provenance': provenance}))
    for value in (x, y):
        h.update(np.asarray(value, dtype='<f8').tobytes(order='C'))
    return x, y, groups, provenance, h.hexdigest()


def _forward(x, weights, biases):
    a = np.tanh(x@weights[0]+biases[0])
    b = np.tanh(a@weights[1]+biases[1])
    return a, b, b@weights[2]+biases[2]


def train(dataset, *, blind=False, stage='initial', seed=17, epochs=50,
          batch_size=512, learning_rate=.003, validation_dataset=None):
    """Fixed CPU 2×64 Adam recipe. Validation never influences normalization.

    Targets are absolute native issued actions, or separately identified teacher
    query labels on mixed trajectories. There is no residual/reference input.
    Training normalization includes explicit missing zeros and their masks;
    it is not advertised as available-only sensor standardization.
    """
    if type(blind) is not bool or stage not in ('initial', 'corrected'):
        raise ValueError('Choose a boolean blind flag and initial/corrected stage.')
    if (type(seed) is not int or seed != 17 or type(epochs) is not int or epochs != 50
            or type(batch_size) is not int or batch_size != 512
            or type(learning_rate) not in (int, float) or learning_rate != .003):
        raise ValueError('The registered recipe is seed17, 50 epochs, batch512, Adam0.003.')
    x, native_y, groups, provenance, dataset_hash = _dataset(dataset, blind=blind)
    validation = None
    if validation_dataset is not None:
        vx, vy, vg, vp, vh = _dataset(validation_dataset, blind=blind)
        if set(groups) & set(vg):
            raise ValueError('Training and development groups must be disjoint.')
        validation = (vx, vy, vg, vh)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-4)
    action_y = native_y/ACTION_SCALE_NATIVE
    action_mean, action_scale = action_y.mean(axis=0), np.maximum(action_y.std(axis=0), .01)
    x = np.clip((x-mean)/scale, -10., 10.).astype(np.float32)
    y = ((action_y-action_mean)/action_scale).astype(np.float32)
    rng = np.random.default_rng(seed)
    dims = (FEATURE_COUNT, 64, 64, 7)
    weights = [rng.normal(0., math.sqrt(2./(a+b)), (a, b)).astype(np.float32) for a, b in zip(dims[:-1], dims[1:])]
    biases = [np.zeros(b, dtype=np.float32) for b in dims[1:]]
    parameters = weights+biases
    first = [np.zeros_like(p) for p in parameters]
    second = [np.zeros_like(p) for p in parameters]
    initial_loss = float(np.mean((_forward(x, weights, biases)[2]-y)**2))
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            indices = order[start:start+batch_size]
            xx, yy = x[indices], y[indices]
            a, b, prediction = _forward(xx, weights, biases)
            output_grad = 2.*(prediction-yy)/(len(indices)*7.)
            b_grad = (output_grad@weights[2].T)*(1.-b*b)
            a_grad = (b_grad@weights[1].T)*(1.-a*a)
            gradients = [xx.T@a_grad, a.T@b_grad, b.T@output_grad,
                         a_grad.sum(axis=0), b_grad.sum(axis=0), output_grad.sum(axis=0)]
            step += 1
            for p, g, m, v in zip(parameters, gradients, first, second):
                np.clip(g, -1., 1., out=g)
                m *= .9; m += .1*g
                v *= .999; v += .001*g*g
                p -= learning_rate*(m/(1.-.9**step))/(np.sqrt(v/(1.-.999**step))+1e-8)
                np.clip(p, -core.MAX_WEIGHT, core.MAX_WEIGHT, out=p)
    final_loss = float(np.mean((_forward(x, weights, biases)[2]-y)**2))
    validation_record = None
    if validation is not None:
        vx, vy, vg, vh = validation
        predictions = _forward(np.clip((vx-mean)/scale, -10., 10.).astype(np.float32), weights, biases)[2]
        target = (vy/ACTION_SCALE_NATIVE-action_mean)/action_scale
        native_predictions = (predictions*action_scale+action_mean)*ACTION_SCALE_NATIVE
        validation_record = dict(sample_count=len(vx), groups=list(dict.fromkeys(vg)), dataset_sha256=vh,
            normalized_mse=float(np.mean((predictions-target)**2)),
            action_scale_mse=float(np.mean(((native_predictions-vy)/ACTION_SCALE_NATIVE)**2)),
            native_rmse=np.sqrt(np.mean((native_predictions-vy)**2, axis=0)).tolist())
    model = dict(schema=SCHEMA, scope=SCOPE, physical_robot_ready=False,
        observation_origin='rendered_rgbd_and_simulated_encoders', supervision_origin='synthetic_authored_teacher',
        feature_contract=FEATURE_CONTRACT, feature_names=list(FEATURE_NAMES),
        feature_contract_sha256=FEATURE_CONTRACT_SHA256, blind=blind, hidden_units=[64, 64],
        joint_limits_rad=[list(pair) for pair in core.JOINT_LIMITS], jaw_limit_m=core.JAW_MAX_M,
        action_representation='absolute_native_targets', feature_mean=mean.tolist(), feature_scale=scale.tolist(),
        action_mean=action_mean.tolist(), action_scale=action_scale.tolist(),
        weights=[p.tolist() for p in weights], biases=[p.tolist() for p in biases],
        training=dict(algorithm='numpy-tanh-2x64-adam-standardized-absolute-mse', stage=stage,
            seed=seed, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            sample_count=len(x), train_groups=list(dict.fromkeys(groups)), dataset_sha256=dataset_hash,
            provenance=provenance, provenance_sha256=digest(provenance),
            model_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            initial_loss=initial_loss, final_loss=final_loss, validation=validation_record))
    return validate_model(model)


def validate_model(document):
    model = core._decode(document)
    if type(model) is not dict or set(model) != MODEL_FIELDS or model.get('schema') != SCHEMA:
        raise ValueError('Unsupported camera-action model schema or fields.')
    expected = dict(scope=SCOPE, physical_robot_ready=False,
        observation_origin='rendered_rgbd_and_simulated_encoders', supervision_origin='synthetic_authored_teacher',
        feature_contract=FEATURE_CONTRACT, feature_names=list(FEATURE_NAMES),
        feature_contract_sha256=FEATURE_CONTRACT_SHA256, hidden_units=[64, 64],
        joint_limits_rad=[list(pair) for pair in core.JOINT_LIMITS], jaw_limit_m=core.JAW_MAX_M,
        action_representation='absolute_native_targets')
    if any(model.get(key) != value for key, value in expected.items()) or model['physical_robot_ready'] is not False:
        raise ValueError('Camera-action bounds, provenance or feature contract changed.')
    if type(model['blind']) is not bool:
        raise ValueError('Model blind flag must be boolean.')
    for key, size in (('feature_mean', FEATURE_COUNT), ('feature_scale', FEATURE_COUNT), ('action_mean', 7), ('action_scale', 7)):
        value = core._array(model[key], (size,), key, 100.)
        if key.endswith('_scale') and np.any(value <= 0.):
            raise ValueError('Normalization scales must be positive.')
    if model['blind'] and (np.any(np.asarray(model['feature_mean'])[LIVE_FEATURE_START:] != 0.)
            or not np.allclose(np.asarray(model['feature_scale'])[LIVE_FEATURE_START:], 1e-4, atol=0., rtol=1e-12)):
        raise ValueError('Blind model normalization must preserve zeroed feedback.')
    dims = (FEATURE_COUNT, 64, 64, 7)
    if type(model['weights']) is not list or type(model['biases']) is not list or len(model['weights']) != 3 or len(model['biases']) != 3:
        raise ValueError('Exactly three learned affine layers are required.')
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        core._array(model['weights'][i], (a, b), 'weights', core.MAX_WEIGHT)
        core._array(model['biases'][i], (b,), 'biases', core.MAX_WEIGHT)
    training = model['training']
    if type(training) is not dict or set(training) != TRAINING_FIELDS:
        raise ValueError('Exact training provenance fields are required.')
    if (training['algorithm'] != 'numpy-tanh-2x64-adam-standardized-absolute-mse'
            or training['stage'] not in ('initial', 'corrected') or training['seed'] != 17
            or training['epochs'] != 50 or training['batch_size'] != 512 or training['learning_rate'] != .003):
        raise ValueError('Model does not describe the fixed registered recipe.')
    for name in ('seed', 'epochs', 'batch_size'):
        if type(training[name]) is not int:
            raise ValueError('Training integer metadata must not use booleans or floats.')
    core._integer(training['sample_count'], 'sample count', 1, 1_000_000)
    core._case_ids(training['train_groups'])
    for field in ('dataset_sha256', 'provenance_sha256', 'model_source_sha256'):
        _hash(training[field], field)
    if type(training['provenance']) is not dict or not training['provenance'] or digest(training['provenance']) != training['provenance_sha256']:
        raise ValueError('Training provenance hash mismatch.')
    for name in ('initial_loss', 'final_loss'):
        core._number(training[name], name, 0., 1e12)
    val = training['validation']
    if val is not None:
        if type(val) is not dict or set(val) != {'sample_count', 'groups', 'dataset_sha256', 'normalized_mse', 'action_scale_mse', 'native_rmse'}:
            raise ValueError('Invalid development metrics.')
        core._integer(val['sample_count'], 'validation sample count', 1, 1_000_000)
        core._case_ids(val['groups'])
        if set(val['groups']) & set(training['train_groups']):
            raise ValueError('Model training and validation groups overlap.')
        _hash(val['dataset_sha256'], 'validation dataset hash')
        core._number(val['normalized_mse'], 'validation MSE', 0., 1e12)
        core._number(val['action_scale_mse'], 'validation shared-scale MSE', 0., 1e12)
        if np.any(core._array(val['native_rmse'], (7,), 'validation RMSE', 1e6) < 0.):
            raise ValueError('RMSE cannot be negative.')
    return model


def save_model(document, path):
    model = validate_model(document)
    with Path(path).open('xb') as stream:
        stream.write(canonical(model)+b'\n')
    return digest(model)


def load_model(path):
    with Path(path).open('rb') as stream:
        return validate_model(stream.read(core.MAX_MODEL_BYTES+1))


class ModelPolicy:
    """Sequential local inference; generic native rate limiting stays in the arm runtime."""
    def __init__(self, document):
        self.model = validate_model(document)
        self.encoder = CameraActionFeatures(blind=self.model['blind'])
        self.mean = np.asarray(self.model['feature_mean'], dtype=np.float32)
        self.scale = np.asarray(self.model['feature_scale'], dtype=np.float32)
        self.action_mean = np.asarray(self.model['action_mean'], dtype=np.float32)
        self.action_scale = np.asarray(self.model['action_scale'], dtype=np.float32)
        self.weights = [np.asarray(w, dtype=np.float32) for w in self.model['weights']]
        self.biases = [np.asarray(b, dtype=np.float32) for b in self.model['biases']]
        self.inference_seconds = []

    def __call__(self, context):
        began = time.perf_counter()
        raw = self.encoder(context)
        x = np.clip((raw-self.mean)/self.scale, -10., 10.)[None, :]
        normalized = _forward(x, self.weights, self.biases)[2][0]*self.action_scale+self.action_mean
        target = normalized*ACTION_SCALE_NATIVE
        limits = np.asarray(core.JOINT_LIMITS)
        result = dict(joint_targets_rad=np.clip(target[:6], limits[:, 0], limits[:, 1]).tolist(),
            gripper_position_m=float(np.clip(target[6], 0., core.JAW_MAX_M)))
        self.inference_seconds.append(time.perf_counter()-began)
        return result

    def diagnostics(self):
        return dict(policy_origin='camera_encoder_absolute_numpy_mlp', blind=self.model['blind'],
            student_inference_calls=len(self.inference_seconds), student_inference_seconds=sum(self.inference_seconds),
            student_inference_p95_seconds=float(np.quantile(self.inference_seconds, .95)) if self.inference_seconds else None,
            teacher_fallback_calls=0, runtime_teacher_available=False, authored_task_phase_machine=False,
            physical_robot_ready=False, model_sha256=digest(self.model), feature_contract=FEATURE_CONTRACT,
            timing_scope='Feature encoding and NumPy inference only; excludes camera/render/actuation.')
