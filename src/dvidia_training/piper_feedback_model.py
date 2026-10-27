"""Compact causal residual learning over a frozen PiPER motion primitive.

All inference inputs are whitelisted present/past native measurements.  The
learner has no simulator handle, teacher, IK, task phase, intervention ID,
success flag, or access to a future frame.  This is privileged simulation state,
not a camera policy or a qualified physical-robot controller.

FEATURE_NAMES is the exact order and normalization contract.  Its first 15
values are reset object XY/.5, fixed goal XYZ/.5, elapsed seconds/20, the raw
frozen nominal joint6/3 and jaw/.035, then the actual ages of two past snapshots
/.12.  The remaining 168 values are current, 40 ms-old and 120 ms-old snapshots:
the first 42 piper_imitation.FEATURE_NAMES (the elapsed term is omitted), native
actuator controls joint6/3 and jaw/.035, and native actuator forces/10. Native
arm forces are torques in Nm; the coupled jaw force is N. Snapshot age targets
use the latest available frame at or before each target time, or the reset
frame during warmup. Each lag is capped at 120 ms; at irregular cadences a
missing in-window lag falls back to the oldest retained in-window frame.

Blind models have the same architecture and feature dimensions; every live
snapshot slot is zero. The two age features depend only on observation timing.
The nominal reference is calculated BEFORE live-joint target caps, so nominal
features cannot leak joint feedback into the blind model. Live joints are
read only by its generic final native command limiter.
"""
from __future__ import annotations

from collections import deque
from hashlib import sha256
import math
import time

import numpy as np

from . import piper_imitation as core
from . import piper_motor_primitive as primitive


SCHEMA = 'dvidia.piper-feedback-residual/v1'
FEATURE_CONTRACT = 'causal-native-183-current-lag40-lag120/v1'
SCOPE = ('Simulation-only learned residual correction over a frozen synthetic-demo motion primitive; '
         'privileged native measurements and at most 120 ms history. No camera policy, '
         'teacher, IK, authored task phase or physical robot qualification at inference.')
RESIDUAL_LIMITS = np.asarray([1.]*6 + [core.JAW_MAX_M])
SNAPSHOT_NAMES = core.FEATURE_NAMES[:-1] + tuple(
    f'native_control_joint_{i}_rad/3' for i in range(1, 7)) + (
    'native_control_jaw_m/.035',) + tuple(
    f'native_actuator_{i}_torque_nm/10' for i in range(1, 7)) + (
    'native_actuator_jaw_force_n/10',)
CONTEXT_NAMES = (tuple(f'reset_object_{a}_m/.5' for a in 'xy')
                 + tuple(f'fixed_goal_{a}_m/.5' for a in 'xyz')
                 + ('elapsed_seconds/20',)
                 + tuple(f'raw_nominal_joint_{i}_rad/3' for i in range(1, 7))
                 + ('raw_nominal_jaw_m/.035', 'lag40_actual_age_s/.12',
                    'lag120_actual_age_s/.12'))
FEATURE_NAMES = CONTEXT_NAMES + tuple(
    f'{lag}:{name}' for lag in ('current', 'lag40', 'lag120') for name in SNAPSHOT_NAMES)
FEATURE_COUNT = len(FEATURE_NAMES)
LIVE_FEATURE_START = len(CONTEXT_NAMES)
MODEL_FIELDS = {'schema', 'scope', 'physical_robot_ready', 'observation_origin',
                'feature_contract', 'feature_names', 'blind', 'base_sha256',
                'residual_limits_native', 'joint_limits_rad', 'joint_delta_limit_rad',
                'jaw_limit_m', 'feature_mean', 'feature_scale', 'frequencies',
                'phases', 'weights', 'training'}
TRAINING_FIELDS = {'algorithm', 'seed', 'basis', 'lengthscale', 'ridge',
                   'sample_count', 'source_sample_count', 'train_groups',
                   'validation_groups', 'dataset_sha256', 'source_dataset_sha256',
                   'weighted_normalized_mse', 'native_rmse', 'validation_native_rmse',
                   'clipped_target_components', 'weight_sum', 'random_basis_sha256'}


def _native_action(action, name='action'):
    if type(action) is not dict or set(action) != {'joint_targets_rad', 'gripper_position_m'}:
        raise ValueError(f'{name} must contain six joint targets and one jaw target.')
    q = core._array(action['joint_targets_rad'], (6,), name+' joints', 100.)
    jaw = core._number(action['gripper_position_m'], name+' jaw', 0., core.JAW_MAX_M)
    return np.concatenate((q, [jaw]))


def _hash(value, name):
    primitive._hash(value, name)


class NominalReference:
    """Raw frozen time/context reference, independent of every live state after reset.

    Fresh per episode; exactly t=0, .02, ... seconds. No live-q cap is applied
    here. Its jaw is bounded as in MotorPrimitivePolicy; joint caps apply only
    once after adding the learned residual in FeedbackPolicy.
    """
    def __init__(self, base_document):
        self.model = primitive.validate_model(base_document)
        self.weights = np.asarray(self.model['weights'])
        self.high_resolution = self.model['schema'] == primitive.HIGH_RESOLUTION_SCHEMA
        self.calls, self.initial_xy, self.goal_index = 0, None, None

    def __call__(self, observation):
        if type(observation) is not dict:
            raise ValueError('Nominal reference requires an observation dictionary.')
        elapsed = core._number(observation.get('t_seconds'), 't_seconds', 0., 20.)
        if not math.isclose(elapsed, self.calls*.02, rel_tol=0., abs_tol=1e-8):
            raise ValueError('Nominal reference requires t=0 and 20 ms action cadence.')
        goal = primitive._goal(observation.get('target_position_m'))
        initial = primitive._start(observation.get('object_position_m')) if not self.calls else self.initial_xy
        if self.calls and goal != self.goal_index:
            raise ValueError('Nominal reference goal cannot change after reset.')
        reference = np.einsum('s,t,stj->j', primitive._spatial(initial, self.high_resolution),
                              primitive._time_basis(elapsed, self.high_resolution),
                              self.weights[goal], optimize=True)
        self.calls += 1
        self.initial_xy, self.goal_index = initial.copy(), goal
        return {'joint_targets_rad': (reference[:6]*3.).tolist(),
                'gripper_position_m': float(core.JAW_MAX_M*np.clip(reference[6], 0., 1.))}


class FeedbackFeatures:
    """Deterministic per-episode current and bounded causal history encoder.

    ``features(obs, nominal_action)`` accepts a raw NominalReference action,
    never the live-q-capped MotorPrimitivePolicy output. Timestamp order is
    strictly increasing. Extra observation fields are deliberately ignored.
    Validation completes before any history is committed.
    """
    def __init__(self, *, blind=False):
        if type(blind) is not bool:
            raise ValueError('blind must be a boolean.')
        self.blind = blind
        self.history = deque()
        self.initial_xy = self.goal = None
        self.last_time = None

    def features(self, observation, nominal_action):
        if type(observation) is not dict:
            raise ValueError('Feedback features require an observation dictionary.')
        elapsed = core._number(observation.get('t_seconds'), 't_seconds', 0., 20.)
        if self.last_time is None and elapsed != 0.:
            raise ValueError('Feedback features must initialize at t=0.')
        if self.last_time is not None and elapsed <= self.last_time:
            raise ValueError('Feedback features require strictly increasing timestamps.')
        goal = core._array(observation.get('target_position_m'), (3,), 'target_position_m', 1.)
        if self.goal is not None and not np.array_equal(goal, self.goal):
            raise ValueError('Feedback destination cannot change within an episode.')
        initial = primitive._start(observation.get('object_position_m')) if self.initial_xy is None else self.initial_xy
        nominal = _native_action(nominal_action, 'raw nominal reference')
        if self.blind:
            snapshot = np.zeros(len(SNAPSHOT_NAMES))
        else:
            measured = core.features(observation).astype(np.float64)[:-1]
            controls = core._array(observation.get('native_actuator_controls'), (7,), 'native_actuator_controls', 1e4)
            forces = core._array(observation.get('actuator_force_native'), (7,), 'actuator_force_native', 1e6)
            snapshot = np.concatenate((measured, controls/np.asarray([3.]*6+[core.JAW_MAX_M]), forces/10.))
            snapshot = np.clip(snapshot, -10., 10.)
        pending = list(self.history) + [(elapsed, snapshot)]
        # Drop all observations older than the allowed history horizon. Exact
        # regular-cadence 120 ms snapshots survive floating-point timestamp noise.
        pending = [row for row in pending if row[0] >= elapsed-.12-1e-8]
        selected = []
        for lag in (.04, .12):
            eligible = [row for row in pending if row[0] <= elapsed-lag+1e-8]
            selected.append(eligible[-1] if eligible else pending[0])
        context = np.concatenate((initial/.5, goal/.5, [elapsed/20.],
                                  nominal/np.asarray([3.]*6+[core.JAW_MAX_M]),
                                  [min(.12, elapsed-row[0])/.12 for row in selected]))
        result = np.concatenate((context, snapshot, selected[0][1], selected[1][1]))
        if not np.isfinite(result).all() or np.any(np.abs(result) > 10.):
            raise ValueError('Feedback features exceed their finite bounded contract.')
        if len(pending) > 4096:
            raise ValueError('Too many observations within the 120 ms history window.')
        self.history = deque(pending)
        self.initial_xy, self.goal, self.last_time = initial.copy(), goal.copy(), elapsed
        return result

    __call__ = features


def residual_target(expert_action, nominal_action):
    """Native expert-minus-raw-nominal labels, bounded to ±1 rad and ±.035 m."""
    return np.clip(_native_action(expert_action, 'expert action')
                   - _native_action(nominal_action, 'raw nominal reference'),
                   -RESIDUAL_LIMITS, RESIDUAL_LIMITS)


def _random_basis(seed, basis, lengthscale):
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(size=(FEATURE_COUNT, basis))/(math.sqrt(FEATURE_COUNT)*lengthscale)
    phases = rng.uniform(0., 2.*np.pi, size=basis)
    return frequencies, phases


def _design(x, mean, scale, frequencies, phases):
    z = np.clip((x-mean)/scale, -8., 8.)
    return np.concatenate((np.ones((len(z), 1)), z,
                           np.cos(z@frequencies+phases)*math.sqrt(2./len(phases))), axis=1)


def _dataset_hash(x, y, groups, weights):
    h = sha256(core.canonical({'features': list(x.shape), 'targets': list(y.shape), 'groups': groups}))
    for a in (x, y, weights):
        h.update(np.asarray(a, dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def train(dataset, *, base_sha256, blind=False, seed=17, basis=256,
          lengthscale=1., ridge=1e-3, validation_groups=(), sample_weights=None):
    """Weighted deterministic ridge on linear terms plus Gaussian Fourier features.

    Dataset keys: ``features`` (N×183), ``targets`` (N×7 native residuals),
    ``groups`` (N whole-episode string IDs); optional ``sample_weights`` (N).
    Every supplied validation group is wholly excluded from standardization,
    fitting, and the training dataset hash. Hyperparameters are selected by the
    caller using development episodes, never confirmation outcomes. The same
    seed/basis/recipe serves independently fitted full and blind controls.
    """
    _hash(base_sha256, 'base_sha256')
    if type(blind) is not bool:
        raise ValueError('blind must be a boolean.')
    core._integer(seed, 'seed', 0, 2**32-1)
    if type(basis) is not int or basis not in (128, 256, 512):
        raise ValueError('basis must be 128, 256 or 512.')
    lengthscale = core._number(lengthscale, 'lengthscale', .05, 20.)
    ridge = core._number(ridge, 'ridge', 1e-9, 100.)
    if type(dataset) is not dict or not {'features', 'targets', 'groups'} <= set(dataset) <= {'features', 'targets', 'groups', 'sample_weights'}:
        raise ValueError('Feedback dataset needs features, targets and episode groups.')
    groups = dataset['groups']
    if type(groups) not in (list, tuple) or not 2 <= len(groups) <= 1_000_000:
        raise ValueError('Feedback dataset requires 2 to 1000000 grouped samples.')
    groups = list(groups)
    if any(type(group) is not str for group in groups):
        raise ValueError('Every sample needs a whole-episode group ID.')
    unique = list(dict.fromkeys(groups))
    core._case_ids(unique)
    x = core._array(dataset['features'], (len(groups), FEATURE_COUNT), 'features', 10.)
    y = core._array(dataset['targets'], (len(groups), 7), 'native residual labels', 100.)
    if blind and np.any(x[:, LIVE_FEATURE_START:] != 0.):
        raise ValueError('Blind training must zero every live/history measurement feature.')
    if sample_weights is not None and 'sample_weights' in dataset:
        raise ValueError('Provide sample weights only once.')
    raw_weights = sample_weights if sample_weights is not None else dataset.get('sample_weights', np.ones(len(groups)))
    w = core._array(raw_weights, (len(groups),), 'sample_weights', 100.)
    if np.any(w <= 0.):
        raise ValueError('All sample weights must be strictly positive.')
    if type(validation_groups) not in (list, tuple):
        raise ValueError('validation_groups must be episode IDs.')
    held = list(validation_groups)
    if held:
        core._case_ids(held)
    if any(group not in unique for group in held):
        raise ValueError('Validation IDs must identify whole supplied episodes.')
    fit = np.asarray([group not in held for group in groups])
    if fit.sum() < 2:
        raise ValueError('At least two training samples must remain after whole-episode holdout.')
    source_hash = _dataset_hash(x, y, groups, w)
    train_hash = _dataset_hash(x[fit], y[fit], [g for g, use in zip(groups, fit) if use], w[fit])
    clipped = np.clip(y, -RESIDUAL_LIMITS, RESIDUAL_LIMITS)
    target = clipped/RESIDUAL_LIMITS
    mean = np.average(x[fit], weights=w[fit], axis=0)
    scale = np.maximum(np.sqrt(np.average((x[fit]-mean)**2, weights=w[fit], axis=0)), 1e-4)
    frequencies, phases = _random_basis(seed, basis, lengthscale)
    width = 1+FEATURE_COUNT+basis
    gram, rhs = np.zeros((width, width)), np.zeros((width, 7))
    for start in range(0, len(x), 4096):
        use = fit[start:start+4096]
        if not use.any():
            continue
        matrix = _design(x[start:start+4096][use], mean, scale, frequencies, phases)
        weight = w[start:start+4096][use]
        gram += matrix.T@(matrix*weight[:, None])
        rhs += matrix.T@(target[start:start+4096][use]*weight[:, None])
    gram.flat[::width+1] += ridge
    coefficients = np.linalg.solve(gram, rhs)
    prediction = np.empty_like(target)
    for start in range(0, len(x), 4096):
        prediction[start:start+4096] = _design(x[start:start+4096], mean, scale, frequencies, phases)@coefficients
    errors = prediction-target
    rmse = lambda mask: (np.sqrt(np.average(errors[mask]**2, weights=w[mask], axis=0))*RESIDUAL_LIMITS).tolist()
    model = {'schema': SCHEMA, 'scope': SCOPE, 'physical_robot_ready': False,
             'observation_origin': 'privileged_native_simulator_state',
             'feature_contract': FEATURE_CONTRACT, 'feature_names': list(FEATURE_NAMES),
             'blind': blind, 'base_sha256': base_sha256,
             'residual_limits_native': RESIDUAL_LIMITS.tolist(),
             'joint_limits_rad': [list(b) for b in core.JOINT_LIMITS],
             'joint_delta_limit_rad': core.JOINT_DELTA_RAD, 'jaw_limit_m': core.JAW_MAX_M,
             'feature_mean': mean.tolist(), 'feature_scale': scale.tolist(),
             'frequencies': frequencies.tolist(), 'phases': phases.tolist(),
             'weights': coefficients.tolist(), 'training': {
                 'algorithm': 'numpy-weighted-ridge-linear-gaussian-rff', 'seed': seed, 'basis': basis,
                 'lengthscale': lengthscale, 'ridge': ridge, 'sample_count': int(fit.sum()),
                 'source_sample_count': len(groups),
                 'train_groups': [g for g in unique if g not in held], 'validation_groups': held,
                 'dataset_sha256': train_hash, 'source_dataset_sha256': source_hash,
                 'weighted_normalized_mse': float(np.average(np.mean(errors[fit]**2, axis=1), weights=w[fit])),
                 'native_rmse': rmse(fit), 'validation_native_rmse': rmse(~fit) if held else None,
                 'clipped_target_components': int(np.count_nonzero(y[fit] != clipped[fit])),
                 'weight_sum': float(w[fit].sum()),
                 'random_basis_sha256': sha256(frequencies.astype('<f8').tobytes()+phases.astype('<f8').tobytes()).hexdigest()}}
    return validate_model(model)


def validate_model(document):
    """Strict bounded JSON only: no executable deserialization or extra fields."""
    model = core._decode(document)
    if type(model) is not dict or set(model) != MODEL_FIELDS:
        raise ValueError('Unknown or missing feedback-model field.')
    fixed = {'schema': SCHEMA, 'scope': SCOPE, 'physical_robot_ready': False,
             'observation_origin': 'privileged_native_simulator_state',
             'feature_contract': FEATURE_CONTRACT, 'feature_names': list(FEATURE_NAMES),
             'joint_delta_limit_rad': core.JOINT_DELTA_RAD, 'jaw_limit_m': core.JAW_MAX_M}
    if any(type(model[k]) is not type(v) or model[k] != v for k, v in fixed.items()):
        raise ValueError('Unsupported feedback scope or feature/native contract.')
    if type(model['blind']) is not bool:
        raise ValueError('blind must be a boolean.')
    _hash(model['base_sha256'], 'base_sha256')
    for key, value in (('residual_limits_native', RESIDUAL_LIMITS), ('joint_limits_rad', core.JOINT_LIMITS)):
        if not np.array_equal(core._array(model[key], np.asarray(value).shape, key, 10.), value):
            raise ValueError('Feedback model cannot change '+key+'.')
    mean = core._array(model['feature_mean'], (FEATURE_COUNT,), 'feature_mean', 10.)
    scale = core._array(model['feature_scale'], (FEATURE_COUNT,), 'feature_scale', 10.)
    if np.any(scale < 1e-4):
        raise ValueError('Feature scales must be positive and bounded.')
    receipt = model['training']
    if type(receipt) is not dict or set(receipt) != TRAINING_FIELDS or receipt['algorithm'] != 'numpy-weighted-ridge-linear-gaussian-rff':
        raise ValueError('Unsupported feedback training receipt.')
    seed = core._integer(receipt['seed'], 'seed', 0, 2**32-1)
    basis = receipt['basis']
    if type(basis) is not int or basis not in (128, 256, 512):
        raise ValueError('Unsupported random-feature capacity.')
    lengthscale = core._number(receipt['lengthscale'], 'lengthscale', .05, 20.)
    core._number(receipt['ridge'], 'ridge', 1e-9, 100.)
    frequencies = core._array(model['frequencies'], (FEATURE_COUNT, basis), 'frequencies', 100.)
    phases = core._array(model['phases'], (basis,), 'phases', 2.*np.pi)
    expected_f, expected_p = _random_basis(seed, basis, lengthscale)
    if not np.array_equal(frequencies, expected_f) or not np.array_equal(phases, expected_p):
        raise ValueError('Random basis differs from the declared deterministic recipe.')
    actual_hash = sha256(frequencies.astype('<f8').tobytes()+phases.astype('<f8').tobytes()).hexdigest()
    if receipt['random_basis_sha256'] != actual_hash:
        raise ValueError('Random-basis hash does not match its data.')
    core._array(model['weights'], (1+FEATURE_COUNT+basis, 7), 'weights', 1e4)
    count = core._integer(receipt['sample_count'], 'sample_count', 2, 1_000_000)
    source_count = core._integer(receipt['source_sample_count'], 'source_sample_count', count, 1_000_000)
    train_groups = core._case_ids(receipt['train_groups'])
    held = receipt['validation_groups']
    if type(held) is not list or (held and not core._case_ids(held)) or set(held)&set(train_groups):
        raise ValueError('Training and validation episode groups must be disjoint.')
    if len(train_groups) > count:
        raise ValueError('Training group count exceeds sample count.')
    if bool(held) != (source_count > count) or len(held) > source_count-count:
        raise ValueError('Whole-episode validation groups and held-out sample counts disagree.')
    for key in ('dataset_sha256', 'source_dataset_sha256'):
        _hash(receipt[key], key)
    core._number(receipt['weighted_normalized_mse'], 'weighted_normalized_mse', 0., 1e6)
    for name in ('native_rmse', 'validation_native_rmse'):
        if name == 'validation_native_rmse' and not held:
            if receipt[name] is not None:
                raise ValueError('Validation error needs held-out episodes.')
            continue
        if np.any(core._array(receipt[name], (7,), name, 1e3) < 0.):
            raise ValueError('Fit errors cannot be negative.')
    core._integer(receipt['clipped_target_components'], 'clipped_target_components', 0, count*7)
    core._number(receipt['weight_sum'], 'weight_sum', 0., count*100.)
    if receipt['weight_sum'] <= 0.:
        raise ValueError('Training weight sum must be positive.')
    if model['blind'] and (np.any(mean[LIVE_FEATURE_START:] != 0.) or np.any(scale[LIVE_FEATURE_START:] != 1e-4)):
        raise ValueError('Blind model normalization contains live measurements.')
    return model


class FeedbackPolicy:
    """Fresh per episode: frozen reference + learned residual + generic caps."""
    def __init__(self, base_document, feedback_document, *, blind=False):
        if type(blind) is not bool:
            raise ValueError('blind must be a boolean.')
        self.model = validate_model(feedback_document)
        self.nominal = NominalReference(base_document)
        if self.model['base_sha256'] != core.digest(self.nominal.model):
            raise ValueError('Feedback model is bound to a different frozen base model.')
        if self.model['blind'] != blind:
            raise ValueError('Feedback/blind model and runtime mode must match.')
        self.encoder = FeedbackFeatures(blind=blind)
        self.mean = np.asarray(self.model['feature_mean'])
        self.scale = np.asarray(self.model['feature_scale'])
        self.frequencies = np.asarray(self.model['frequencies'])
        self.phases = np.asarray(self.model['phases'])
        self.weights = np.asarray(self.model['weights'])
        self.calls, self.inference_seconds, self.residual_clip_components = 0, 0., 0

    def __call__(self, observation):
        began = time.perf_counter()
        # Validate measurements before advancing the strict nominal clock.
        if type(observation) is not dict:
            raise ValueError('Feedback policy requires an observation dictionary.')
        q = core._array(observation.get('joint_position_rad'), (6,), 'joint_position_rad', 10.)
        before = self.nominal.calls, self.nominal.initial_xy, self.nominal.goal_index
        try:
            raw = self.nominal(observation)
            feature = self.encoder.features(observation, raw)
        except ValueError:
            self.nominal.calls, self.nominal.initial_xy, self.nominal.goal_index = before
            raise
        prediction = (_design(feature[None, :], self.mean, self.scale,
                              self.frequencies, self.phases)@self.weights)[0]
        self.residual_clip_components += int(np.count_nonzero(np.abs(prediction) > 1.))
        residual = np.clip(prediction, -1., 1.)*RESIDUAL_LIMITS
        requested = _native_action(raw)+residual
        limits = np.asarray(core.JOINT_LIMITS)
        target = np.clip(q+np.clip(requested[:6]-q, -core.JOINT_DELTA_RAD, core.JOINT_DELTA_RAD),
                         limits[:, 0]+.02, limits[:, 1]-.02)
        result = {'joint_targets_rad': target.tolist(),
                  'gripper_position_m': float(np.clip(requested[6], 0., core.JAW_MAX_M))}
        self.calls += 1
        self.inference_seconds += time.perf_counter()-began
        return result

    predict = __call__

    def diagnostics(self):
        return {'policy_origin': 'learned_causal_feedback_residual', 'blind': self.model['blind'],
                'student_inference_calls': self.calls, 'student_inference_seconds': self.inference_seconds,
                'residual_clipped_components': self.residual_clip_components,
                'teacher_fallback_calls': 0, 'runtime_teacher_available': False,
                'authored_task_phase_machine': False, 'runtime_ik_available': False,
                'reactive_contact_feedback': not self.model['blind'], 'maximum_history_seconds': .12,
                'physical_robot_ready': False, 'base_model_sha256': self.model['base_sha256'],
                'feedback_model_sha256': core.digest(self.model)}
