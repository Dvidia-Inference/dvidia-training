"""Unregistered reference-endpoint learning bridge; no hardware/skill qualification.

Only independently pinned, train-only reference exports are training inputs.
The callable policy fits the unchanged camera/action recorder's four-key context.
Exact prior servo references are decoder state, never reconstructed from features.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import time

import numpy as np

from . import camera_action_model as base
from . import camera_action_export as export
from . import camera_action_reference_export as ref

SCHEMA = 'dvidia.camera-action-reference-mlp/v1'
SCOPE = 'Unregistered simulation-only reference-endpoint research candidate; no skill or hardware qualification.'
VARIANTS = ('absolute', 'delta')
R0 = [0., 1.57, -1.3485, 0., 0., 0., .035]
D = np.asarray([.03]*6+[.001], dtype=np.float64)
S = np.asarray([3.]*6+[.035], dtype=np.float64)
CONTRACT = dict(feature_contract=base.FEATURE_CONTRACT,
    feature_contract_sha256=base.FEATURE_CONTRACT_SHA256,
    reference_kind='previous_applied_normal_servo_reference', reset_reference=R0,
    joint_order=[f'joint{i}' for i in range(1, 7)]+['single_finger_jaw'],
    action_units=['rad']*6+['single_finger_m'], increment_scale=D.tolist(), loss_scale=S.tolist(),
    absolute_decoder='r0 + D*g(x)', delta_decoder='r + D*g(x)',
    loss='mean(((unprojected_endpoint-feasible_target)/S)**2)',
    projection='native bounds then exact original 20 normal reference updates',
    network_dtype='float32', decoder_loss_and_gradients_dtype='float64', adam_moments_dtype='float32',
    history_seconds=.16, control_period_s=.02, hidden_units=[64, 64], blind=False)
RECIPE = dict(seed=17, epochs=50, batch_size=512, learning_rate=.003,
    beta1=.9, beta2=.999, epsilon=1e-8, gradient_clip=1., parameter_clip=base.core.MAX_WEIGHT)
MODEL_FIELDS = {'schema', 'scope', 'variant', 'physical_robot_ready', 'task_qualified',
    'scientific_registration', 'contract', 'recipe', 'source_identity', 'feature_mean',
    'feature_scale', 'weights', 'biases', 'training'}
TRAINING_FIELDS = {'algorithm', 'samples', 'groups', 'parent_dataset_sha256',
    'parent_provenance', 'parent_provenance_sha256', 'reference_dataset_sha256',
    'reference_inputs', 'initial_parameters_sha256', 'initial_loss', 'final_loss',
    'optimizer_updates', 'normalization'}
PIN_FIELDS = {'expected_manifest_sha256', 'expected_parent_manifest_sha256',
    'expected_parent_source_sha256', 'expected_protocol_sha256', 'expected_stage',
    'expected_split', 'expected_source_sha256'}


def source_identity():
    """Fail closed on the frozen parent, limiter or reset-source contract."""
    identity = ref.source_identity()
    tree = ast.parse(Path(base.__file__).with_name('piper_task.py').read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PiperTaskEnv')
    reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'reset')
    assignments = {ast.unparse(n.targets[0]): n.value for n in ast.walk(reset)
        if isinstance(n, ast.Assign) and len(n.targets) == 1}
    initial = ast.literal_eval(assignments['self.data.qpos[:8]'])[:6]
    initial += [ast.literal_eval(assignments['self.command_gripper'])]
    if initial != R0 or ast.unparse(assignments['self.command_targets']) != 'self.data.qpos[:6].copy()':
        raise ValueError('The original constant reset-reference assignment changed.')
    return dict(reference_export=identity, reset_reference=initial,
        model_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())


def _variant(variant):
    if type(variant) is not str or variant not in VARIANTS:
        raise ValueError('Choose absolute or delta reference-endpoint decoding.')


def _input_pins(value):
    if type(value) is not dict or set(value) != PIN_FIELDS:
        raise ValueError('Require all independent reference and parent pins.')
    if value['expected_split'] != 'train' or value['expected_stage'] not in ('demonstrations', 'corrections'):
        raise ValueError('Only sealed training reference exports may enter learning.')
    for field in PIN_FIELDS-{'expected_split', 'expected_stage'}:
        base._hash(value[field], field)
    return deepcopy(value)


def _load_training(sources):
    """Load/check derived bytes only; do not open source recordings or query bodies."""
    if type(sources) not in (list, tuple) or not 1 <= len(sources) <= 2:
        raise ValueError('Use demonstrations, optionally followed by corrections.')
    datasets, previous, endpoints, inputs = [], [], [], []
    combined_rows = 0
    for index, source in enumerate(sources):
        if type(source) is not dict or set(source) != {'directory', 'parent_cache', 'pins'}:
            raise ValueError('Reference inputs require directories and independent pins.')
        pins = _input_pins(source['pins'])
        if pins['expected_stage'] != ('demonstrations', 'corrections')[index]:
            raise ValueError('Preserve demonstration then correction stage order.')
        if inputs and pins['expected_protocol_sha256'] != inputs[0]['pins']['expected_protocol_sha256']:
            raise ValueError('Training stages require the same registered parent protocol.')
        loaded = ref.load_reference_export(source['directory'], source['parent_cache'], **pins)
        combined_rows += len(loaded['previous_references'])
        if not 1 <= combined_rows <= ref.MAX_SAMPLES:
            raise ValueError('Combined training rows exceed the reference-model sample budget.')
        offset = 0
        for episode in loaded['parent_dataset']['provenance']['episodes']:
            if not np.array_equal(loaded['previous_references'][offset], np.asarray(R0)):
                raise ValueError('Every training episode must retain the source-bound reset reference.')
            offset += episode['sample_count']
        if offset != len(loaded['previous_references']):
            raise ValueError('Episode offsets must cover every exact reference.')
        datasets.append(loaded['parent_dataset'])
        previous.append(loaded['previous_references']); endpoints.append(loaded['feasible_targets'])
        inputs.append(dict(pins=pins, samples=len(previous[-1])))
    parent = export.combine_datasets(datasets)
    x, _, groups, provenance, parent_hash = base._dataset(parent, blind=False)
    r = np.ascontiguousarray(np.concatenate(previous), dtype='<f8')
    y = np.ascontiguousarray(np.concatenate(endpoints), dtype='<f8')
    ref._matrix(r, 'Previous references'); ref._matrix(y, 'Feasible targets')
    if len(x) != len(r) or len(y) != len(r):
        raise ValueError('Parent features and exact-reference labels must stay row-aligned.')
    identity = dict(parent_dataset_sha256=parent_hash, inputs=inputs,
        previous_sha256=sha256(r.tobytes()).hexdigest(), endpoint_sha256=sha256(y.tobytes()).hexdigest())
    return x, r, y, groups, provenance, parent_hash, base.digest(identity), inputs


def _parameters():
    rng = np.random.default_rng(RECIPE['seed'])
    dims = (base.FEATURE_COUNT, 64, 64, 7)
    weights = [rng.normal(0., math.sqrt(2./(a+b)), (a, b)).astype(np.float32)
        for a, b in zip(dims[:-1], dims[1:])]
    biases = [np.zeros(b, dtype=np.float32) for b in dims[1:]]
    return weights, biases, rng


def decode(head, previous, variant):
    """Unprojected float64 endpoints. Both heads have the same Jacobian D."""
    _variant(variant)
    r = ref._matrix(previous, 'Exact previous references')
    g = base.core._array(head, (len(r), 7), 'Network head', 1e6)
    return (np.asarray(R0, dtype=np.float64) if variant == 'absolute' else r) + D*g


def project(endpoint, previous):
    """Identical native-bound and 20-tick reference projection for both variants."""
    r = ref._matrix(previous, 'Exact previous references')
    target = base.core._array(endpoint, (len(r), 7), 'Decoded endpoint', 1e6)
    bounds = np.asarray(ref.CONTRACT['limits_native'], dtype=np.float64)
    return ref.feasible_targets(r, np.ascontiguousarray(np.clip(target, bounds[:, 0], bounds[:, 1])))


def _loss_gradients(x, previous, target, weights, biases, variant):
    """Full-network gradients of the unprojected physical-scale endpoint loss."""
    a, b, head = base._forward(x, weights, biases)
    prediction = decode(head, previous, variant)
    error = prediction-target
    loss = float(np.mean((error/S)**2))
    output_grad = 2.*error*D/(len(x)*7.*S*S)
    b_grad = (output_grad@weights[2].T)*(1.-b.astype(np.float64)**2)
    a_grad = (b_grad@weights[1].T)*(1.-a.astype(np.float64)**2)
    gradients = [x.astype(np.float64).T@a_grad, a.astype(np.float64).T@b_grad,
        b.astype(np.float64).T@output_grad, a_grad.sum(axis=0), b_grad.sum(axis=0), output_grad.sum(axis=0)]
    return loss, gradients


def _adam_update(parameters, gradients, first, second, step):
    # Keep the original float32 parameter/moment storage and Adam constants;
    # the endpoint chain rule/backprop above is explicitly float64.
    for p, gradient, m, v in zip(parameters, gradients, first, second):
        g = np.clip(gradient, -RECIPE['gradient_clip'], RECIPE['gradient_clip'])
        m *= .9; m += .1*g
        v *= .999; v += .001*g*g
        p -= RECIPE['learning_rate']*(m/(1.-.9**step))/(np.sqrt(v/(1.-.999**step))+1e-8)
        np.clip(p, -base.core.MAX_WEIGHT, base.core.MAX_WEIGHT, out=p)


def _loss(x, previous, target, weights, biases, variant):
    prediction = decode(base._forward(x, weights, biases)[2], previous, variant)
    return float(np.mean(((prediction-target)/S)**2))


def train(sources, *, variant):
    """Fit the fixed recipe to validated training exports; no evaluation-data API.

    This implementation is unregistered research. Calling it neither runs a
    simulator nor qualifies a skill. Real experiments need separate registration.
    """
    _variant(variant)
    identity = source_identity()
    x, r, y, groups, provenance, parent_hash, data_hash, inputs = _load_training(sources)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-4)
    x = np.clip((x-mean)/scale, -10., 10.).astype(np.float32)
    weights, biases, rng = _parameters()
    initial_parameters = base.digest(dict(weights=[w.tolist() for w in weights], biases=[b.tolist() for b in biases]))
    parameters = weights+biases
    first, second = ([np.zeros_like(p) for p in parameters] for _ in range(2))
    initial_loss = _loss(x, r, y, weights, biases, variant)
    step = 0
    for _ in range(RECIPE['epochs']):
        order = rng.permutation(len(x))
        for start in range(0, len(x), RECIPE['batch_size']):
            batch = order[start:start+RECIPE['batch_size']]
            _, gradients = _loss_gradients(x[batch], r[batch], y[batch], weights, biases, variant)
            step += 1; _adam_update(parameters, gradients, first, second, step)
    final_loss = _loss(x, r, y, weights, biases, variant)
    if source_identity() != identity:
        raise ValueError('Source contract changed while fitting.')
    document = dict(schema=SCHEMA, scope=SCOPE, variant=variant, physical_robot_ready=False,
        task_qualified=False, scientific_registration=None, contract=deepcopy(CONTRACT), recipe=deepcopy(RECIPE),
        source_identity=identity, feature_mean=mean.tolist(), feature_scale=scale.tolist(),
        weights=[w.tolist() for w in weights], biases=[b.tolist() for b in biases],
        training=dict(algorithm='numpy-tanh-2x64-adam-native-endpoint-mse/v1', samples=len(x),
            groups=list(dict.fromkeys(groups)), parent_dataset_sha256=parent_hash,
            parent_provenance=provenance, parent_provenance_sha256=base.digest(provenance),
            reference_dataset_sha256=data_hash, reference_inputs=inputs,
            initial_parameters_sha256=initial_parameters, initial_loss=initial_loss, final_loss=final_loss,
            optimizer_updates=step, normalization='training features only; no label whitening'))
    return validate_model(document)


def validate_model(document):
    value = base.core._decode(document)
    if type(value) is not dict or set(value) != MODEL_FIELDS or value['schema'] != SCHEMA:
        raise ValueError('Unsupported reference-model schema or fields.')
    _variant(value['variant'])
    if (value['scope'] != SCOPE or value['physical_robot_ready'] is not False
            or value['task_qualified'] is not False or value['scientific_registration'] is not None
            or base.canonical(value['contract']) != base.canonical(CONTRACT)
            or base.canonical(value['recipe']) != base.canonical(RECIPE)
            or value['source_identity'] != source_identity()):
        raise ValueError('Reference-model contract, source, recipe or research scope changed.')
    for name in ('feature_mean', 'feature_scale'):
        array = base.core._array(value[name], (base.FEATURE_COUNT,), name, 20.)
        if name == 'feature_scale' and np.any(array < 1e-4):
            raise ValueError('Feature scales must retain the training-only floor.')
    dims = (base.FEATURE_COUNT, 64, 64, 7)
    if (type(value['weights']) is not list or type(value['biases']) is not list
            or len(value['weights']) != 3 or len(value['biases']) != 3):
        raise ValueError('Exactly three affine layers are required.')
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        for name, shape in (('weights', (a, b)), ('biases', (b,))):
            p = base.core._array(value[name][i], shape, name, base.core.MAX_WEIGHT)
            if not np.array_equal(p, p.astype(np.float32).astype(np.float64)):
                raise ValueError('Serialized network parameters must preserve exact float32 values.')
    training = value['training']
    if type(training) is not dict or set(training) != TRAINING_FIELDS:
        raise ValueError('Exact reference-training provenance is required.')
    if (training['algorithm'] != 'numpy-tanh-2x64-adam-native-endpoint-mse/v1'
            or training['normalization'] != 'training features only; no label whitening'):
        raise ValueError('Unknown endpoint learning or normalization recipe.')
    count = base.core._integer(training['samples'], 'training rows', 1, ref.MAX_SAMPLES)
    base.core._case_ids(training['groups'])
    if len(training['groups']) > count:
        raise ValueError('Training groups cannot exceed samples.')
    for name in ('parent_dataset_sha256', 'parent_provenance_sha256', 'reference_dataset_sha256', 'initial_parameters_sha256'):
        base._hash(training[name], name)
    if (type(training['parent_provenance']) is not dict or not training['parent_provenance']
            or base.digest(training['parent_provenance']) != training['parent_provenance_sha256']):
        raise ValueError('Parent provenance identity changed.')
    items = training['reference_inputs']
    if type(items) is not list or not 1 <= len(items) <= 2:
        raise ValueError('Exact reference-input provenance is required.')
    total = 0
    for index, item in enumerate(items):
        if type(item) is not dict or set(item) != {'pins', 'samples'}:
            raise ValueError('Unknown reference-input fields.')
        pins = _input_pins(item['pins'])
        if (pins['expected_stage'] != ('demonstrations', 'corrections')[index]
                or pins['expected_protocol_sha256'] != items[0]['pins']['expected_protocol_sha256']
                or pins['expected_source_sha256'] != ref.source_digest()
                or pins['expected_parent_source_sha256'] != ref.cache.source_digest()):
            raise ValueError('Training input source, protocol or stage changed.')
        total += base.core._integer(item['samples'], 'input samples', 1, ref.MAX_SAMPLES)
    if total != count or type(training['optimizer_updates']) is not int or training['optimizer_updates'] != 50*math.ceil(count/512):
        raise ValueError('Training sample count or fixed optimizer budget changed.')
    for name in ('initial_loss', 'final_loss'):
        base.core._number(training[name], name, 0., 1e12)
    return value


def save_model(document, path):
    value = validate_model(document)
    with Path(path).open('xb') as stream:
        stream.write(base.canonical(value)+b'\n')
    return base.digest(value)


def load_model(path):
    with Path(path).open('rb') as stream:
        return validate_model(stream.read(base.core.MAX_MODEL_BYTES+1))


def _previous(context):
    """Trust the recorder's explicit field; reject already reduced-precision arrays.

    Python floats preserve JSON doubles but cannot authenticate their provenance.
    No arbitrary caller can prove reference authority just by naming this field.
    """
    value = context.get('previous_applied')
    if type(value) is not dict or set(value) != {'joint_targets_rad', 'gripper_position_m'}:
        raise ValueError('Require the recorder\'s exact previous_applied reference.')
    joints = value['joint_targets_rad']; jaw = value['gripper_position_m']
    if isinstance(joints, np.ndarray):
        if joints.dtype != np.dtype('float64'):
            raise ValueError('Previous references must not be reduced-precision arrays.')
    elif type(joints) not in (list, tuple) or any(type(x) not in (float, int, np.float64) for x in joints):
        raise ValueError('Previous references require ordinary double-precision coordinates.')
    if type(jaw) not in (float, int, np.float64):
        raise ValueError('The exact single-finger reference is required.')
    return np.ascontiguousarray(base.action_vector(value)[None, :], dtype='<f8')


class ModelPolicy:
    """Fresh per episode; callable by unchanged camera_action_capture.run_rollout."""
    def __init__(self, document):
        self.model = validate_model(document)
        self.encoder = base.CameraActionFeatures(blind=False)
        self.mean = np.asarray(self.model['feature_mean'], dtype=np.float64)
        self.scale = np.asarray(self.model['feature_scale'], dtype=np.float64)
        self.weights = [np.asarray(w, dtype=np.float32) for w in self.model['weights']]
        self.biases = [np.asarray(b, dtype=np.float32) for b in self.model['biases']]
        self.inference_seconds = []

    def __call__(self, context):
        began = time.perf_counter()
        if type(context) is not dict:
            raise ValueError('A causal camera/action context is required.')
        previous = _previous(context)
        if self.encoder.last_time is None and not np.array_equal(previous[0], np.asarray(R0)):
            raise ValueError('The first reference must equal the exact source-bound reset.')
        raw = self.encoder(context)
        x = np.clip((raw.astype(np.float64)-self.mean)/self.scale, -10., 10.).astype(np.float32)[None, :]
        endpoint = decode(base._forward(x, self.weights, self.biases)[2], previous, self.model['variant'])
        target = project(endpoint, previous)[0]
        self.inference_seconds.append(time.perf_counter()-began)
        return dict(joint_targets_rad=target[:6].tolist(), gripper_position_m=float(target[6]))

    def diagnostics(self):
        return dict(policy_origin='camera_encoder_reference_endpoint_numpy_mlp', variant=self.model['variant'],
            model_sha256=base.digest(self.model), student_inference_calls=len(self.inference_seconds),
            student_inference_seconds=sum(self.inference_seconds),
            student_inference_p95_seconds=float(np.quantile(self.inference_seconds, .95)) if self.inference_seconds else None,
            teacher_fallback_calls=0, runtime_teacher_available=False, physical_robot_ready=False, task_qualified=False,
            timing_scope='Causal feature encoding, NumPy inference and reference projection; excludes camera/render/actuation.')
