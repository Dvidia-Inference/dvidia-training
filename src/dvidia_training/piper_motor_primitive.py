"""A learned, time-conditioned PiPER feed-forward motion primitive.

Regression coefficients combine initial observed XY, one of two known goals and
elapsed time. They are fitted jointly across demonstrations, not replayed from a
case/action tape. Contact correction and recovery are not learned. Native servos
and the generic joint-target bounds remain the only live state feedback.
"""
from __future__ import annotations

from hashlib import sha256
import math
import time

import numpy as np

from .piper_imitation import (JOINT_LIMITS, JOINT_DELTA_RAD, JAW_MAX_M,
                             _array, _case_ids, _decode, _integer, _number,
                             canonical, digest)


SCHEMA = 'dvidia.piper-motor-primitive/v1'
HIGH_RESOLUTION_SCHEMA = 'dvidia.piper-motor-primitive/v2'
SCOPE = ('Simulation-only time-conditioned trajectory distillation from synthetic authored-controller demonstrations. '
         'Initial privileged object XY, two known goals and measured clock condition learned feed-forward references; '
         'native servos remain authored. No learned contact feedback, recovery, camera learning or physical robot qualification.')
GOALS = ((.35, .08), (.40, -.08))
START_BOUNDS = ((.284, .314), (-.045, -.015))
SPATIAL_TERMS = ('1', 'y', 'y^2', 'x', 'x*y', 'x*y^2', 'x^2', 'x^2*y', 'x^2*y^2')
CENTERS = np.linspace(0., 20., 201)
WIDTH = .075
RIDGE = 1e-5
SHUFFLE_SEED = 19217
HIGH_CENTERS = np.linspace(0., 20., 401)
HIGH_WIDTH = .025
HIGH_RIDGE = 1e-7
HIGH_SPATIAL_TERMS = tuple('*'.join(piece for piece in (
    '' if x == 0 else 'x' if x == 1 else f'x^{x}',
    '' if y == 0 else 'y' if y == 1 else f'y^{y}') if piece) or '1'
    for x in range(4) for y in range(4))
MODEL_FIELDS = {'schema', 'scope', 'physical_robot_ready', 'supervision_origin',
                'feature_contract', 'reactive_contact_feedback', 'known_goals_xy_m',
                'start_bounds_xy_m', 'spatial_terms', 'time_centers_s', 'time_width_s',
                'joint_limits_rad', 'joint_delta_limit_rad', 'jaw_limit_m', 'output_encoding',
                'weights', 'dataset_sha256', 'source_dataset_sha256', 'supervision_sha256', 'train_cases', 'training'}
FEATURE_CONTRACT = 'initial_object_xy_known_goal_elapsed_time__live_q_only_for_native_cap/v1'
OUTPUT_ENCODING = 'absolute_joint_targets_rad_div3__jaw_target_m_div.035'


def _time_basis(seconds, high_resolution=False):
    centers, width = (HIGH_CENTERS, HIGH_WIDTH) if high_resolution else (CENTERS, WIDTH)
    distance = (np.asarray(seconds)[..., None]-centers)/width
    logits = -.5*distance*distance
    logits -= np.max(logits, axis=-1, keepdims=True)
    basis = np.exp(logits)
    return basis/basis.sum(axis=-1, keepdims=True)


def _spatial(xy, high_resolution=False):
    xy = np.asarray(xy)
    normalized = (xy-np.asarray([.299, -.03]))/.015
    x, y = normalized[..., 0], normalized[..., 1]
    if high_resolution:
        xp, yp = (np.ones_like(x), x, x*x, x*x*x), (np.ones_like(y), y, y*y, y*y*y)
        return np.stack(tuple(a*b for a in xp for b in yp), axis=-1)
    return np.stack((np.ones_like(x), y, y*y, x, x*y, x*y*y, x*x, x*x*y, x*x*y*y), axis=-1)


def _goal(point, tolerance=1e-8):
    point = _array(point, (3,), 'target_position_m', 1.)
    if abs(point[2]-.08) > tolerance:
        raise ValueError('Primitive supports only the declared .08 m destination height.')
    for index, known in enumerate(GOALS):
        if np.max(np.abs(point[:2]-known)) <= tolerance:
            return index
    raise ValueError('Primitive supports only its two known training destinations.')


def _start(point):
    point = _array(point, (3,), 'initial object position', 1.)
    bounds = np.asarray(START_BOUNDS)
    # Native settling shifts the declared XY by less than a micrometre. This
    # tolerance is for that measured reset residual, not workspace extrapolation.
    if np.any(point[:2] < bounds[:, 0]-1e-5) or np.any(point[:2] > bounds[:, 1]+1e-5):
        raise ValueError('Primitive initial XY is outside the bounded training envelope.')
    return point[:2]


def _hash(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a SHA256 digest.')


def validate_model(document):
    model = _decode(document)
    if type(model) is not dict or set(model) != MODEL_FIELDS:
        raise ValueError('Unknown or missing motor-primitive field.')
    high_resolution = model.get('schema') == HIGH_RESOLUTION_SCHEMA
    centers, width, ridge, terms = ((HIGH_CENTERS, HIGH_WIDTH, HIGH_RIDGE, HIGH_SPATIAL_TERMS)
                                    if high_resolution else (CENTERS, WIDTH, RIDGE, SPATIAL_TERMS))
    expected = {'schema': HIGH_RESOLUTION_SCHEMA if high_resolution else SCHEMA, 'scope': SCOPE, 'physical_robot_ready': False,
                'supervision_origin': 'synthetic_authored_controller', 'feature_contract': FEATURE_CONTRACT,
                'reactive_contact_feedback': False, 'time_width_s': width,
                'joint_delta_limit_rad': JOINT_DELTA_RAD, 'jaw_limit_m': JAW_MAX_M,
                'output_encoding': OUTPUT_ENCODING}
    if any(type(model[key]) is not type(value) or model[key] != value for key, value in expected.items()):
        raise ValueError('Unsupported motor-primitive scope or native contract.')
    for key, expected_value in (('known_goals_xy_m', GOALS), ('start_bounds_xy_m', START_BOUNDS),
                                ('joint_limits_rad', JOINT_LIMITS)):
        actual = _array(model[key], np.asarray(expected_value).shape, key, 10.)
        if not np.array_equal(actual, np.asarray(expected_value)):
            raise ValueError(f'Primitive cannot change {key}.')
    actual_centers = _array(model['time_centers_s'], (len(centers),), 'time_centers_s', 20.)
    if not np.array_equal(actual_centers, centers) or model['spatial_terms'] != list(terms):
        raise ValueError('Unsupported fixed regression basis.')
    _array(model['weights'], (2, len(terms), len(centers), 7), 'weights', 64.)
    _hash(model['dataset_sha256'], 'dataset_sha256')
    _hash(model['source_dataset_sha256'], 'source_dataset_sha256')
    _hash(model['supervision_sha256'], 'supervision_sha256')
    if len(_case_ids(model['train_cases'])) != 32:
        raise ValueError('This bounded primitive requires its 32 training episodes.')
    receipt = model['training']
    expected_fields = {'algorithm', 'sample_count', 'scene_count', 'spatial_ridge', 'time_ridge',
                       'shuffled_labels', 'label_shuffle_seed', 'normalized_mse', 'native_rmse',
                       'label_conversion', 'weak_control_notice'}
    if type(receipt) is not dict or set(receipt) != expected_fields:
        raise ValueError('Invalid primitive training receipt.')
    if (receipt['algorithm'] != ('numpy-separable-ridge-cubic-gaussian' if high_resolution else 'numpy-separable-ridge-quadratic-gaussian')
            or type(receipt['sample_count']) is not int or receipt['sample_count'] != 32000
            or type(receipt['scene_count']) is not int or receipt['scene_count'] != 32
            or type(receipt['shuffled_labels']) is not bool
            or type(receipt['spatial_ridge']) is not float or receipt['spatial_ridge'] != ridge
            or type(receipt['time_ridge']) is not float or receipt['time_ridge'] != ridge
            or receipt['label_shuffle_seed'] != (SHUFFLE_SEED if receipt['shuffled_labels'] else None)
            or receipt['label_conversion'] != 'q_feature*3 + clipped_delta_label*.06; float32 source precision'
            or receipt['weak_control_notice'] != 'Full frame-wise label shuffle; weak temporal negative control, not whole-trajectory shuffle.'):
        raise ValueError('Primitive training recipe differs from the fixed candidate.')
    if receipt['label_shuffle_seed'] is not None:
        _integer(receipt['label_shuffle_seed'], 'label_shuffle_seed', SHUFFLE_SEED, SHUFFLE_SEED)
    _number(receipt['normalized_mse'], 'normalized_mse', 0., 100.)
    rmse = _array(receipt['native_rmse'], (7,), 'native_rmse', 10.)
    if np.any(rmse < 0):
        raise ValueError('Native fit errors cannot be negative.')
    return model


def train(dataset, *, shuffled=False, high_resolution=False):
    """Fit the one fixed candidate from 32 complete 1000-frame train episodes.

    high_resolution selects the one predeclared V2 capacity change: 16 cubic
    spatial terms,401 Gaussian time centers,width .025s and ridge 1e-7.
    Takes the existing 43-feature/delta-label dataset. Episode grouping recovers
    initial observed XY; neither grouping IDs nor a demonstration tape are used
    at runtime. The shuffled control permutes whole seven-dimensional converted
    action labels across all frames with the predeclared seed.
    """
    if type(shuffled) is not bool:
        raise ValueError('shuffled must be a boolean.')
    if type(high_resolution) is not bool:
        raise ValueError('high_resolution must be a boolean.')
    centers, width, ridge, terms = ((HIGH_CENTERS, HIGH_WIDTH, HIGH_RIDGE, HIGH_SPATIAL_TERMS)
                                    if high_resolution else (CENTERS, WIDTH, RIDGE, SPATIAL_TERMS))
    if type(dataset) is not dict or set(dataset) != {'scene_ids', 'samples'}:
        raise ValueError('Primitive requires scene_ids and causal samples.')
    cases = _case_ids(dataset['scene_ids'])
    rows = dataset['samples']
    if len(cases) != 32 or type(rows) is not list or len(rows) != 32000:
        raise ValueError('Fixed primitive requires 32 complete 1000-frame training episodes.')
    index = {case: i for i, case in enumerate(cases)}
    x, target, scene_index, ticks = [], [], [], []
    for row in rows:
        if type(row) is not dict or set(row) != {'features', 'targets', 'case_id'} or row['case_id'] not in index:
            raise ValueError('Malformed primitive training sample.')
        feature = _array(row['features'], (43,), 'features', 10.)
        label = _array(row['targets'], (7,), 'targets', 1.)
        if label[6] < 0:
            raise ValueError('Jaw labels must lie in [0,1].')
        tick = round(float(feature[-1])*1000.)
        if not 0 <= tick < 1000 or abs(float(feature[-1])-tick/1000.) > 1e-7:
            raise ValueError('Training episode time grid must contain ordered 20 ms samples.')
        x.append(feature); target.append(label); scene_index.append(index[row['case_id']]); ticks.append(tick)
    raw_x, target = np.asarray(x), np.asarray(target)
    source_hash = digest({'scene_ids': list(cases), 'samples': [
        {'case_id': row['case_id'], 'features': a.tolist(), 'targets': b.tolist()}
        for row, a, b in zip(rows, raw_x, target)]})
    target[:, :6] = (raw_x[:, :6]*3.+target[:, :6]*JOINT_DELTA_RAD)/3.
    limits = np.asarray(JOINT_LIMITS)/3.
    if np.any(target[:, :6] < limits[:, 0]-1e-5) or np.any(target[:, :6] > limits[:, 1]+1e-5):
        raise ValueError('Converted target exceeds native joint limits.')
    if shuffled:
        target = target[np.random.default_rng(SHUFFLE_SEED).permutation(len(target))]
    supervision_hash = sha256(np.asarray(target, dtype='<f8').tobytes(order='C')).hexdigest()
    structured = np.empty((32, 1000, 7))
    filled = np.zeros((32, 1000), dtype=bool)
    contexts, goals = np.zeros((32, 2)), np.full(32, -1, dtype=int)
    for sample, (scene, tick) in enumerate(zip(scene_index, ticks)):
        if filled[scene, tick]:
            raise ValueError('Duplicate training frame within an episode.')
        filled[scene, tick] = True
        structured[scene, tick] = target[sample]
        f = raw_x[sample]
        tcp, destination = f[14:17]*.5, f[24:27]*.2+f[14:17]*.5
        goal = _goal(destination, tolerance=1e-5)
        if goals[scene] not in (-1, goal):
            raise ValueError('Training destination changed within an episode.')
        goals[scene] = goal
        if tick == 0:
            contexts[scene] = _start(np.concatenate((tcp[:2]+f[21:23]*.2, [tcp[2]+f[23]*.2])))
    if not filled.all() or any(np.count_nonzero(goals == goal) != 16 for goal in range(2)):
        raise ValueError('Each known goal requires 16 complete training episodes.')
    basis = _time_basis(np.arange(1000)*.02, high_resolution)
    time_projection = np.linalg.solve(basis.T@basis+ridge*np.eye(len(centers)), basis.T)
    weights, residuals = [], []
    for goal in range(2):
        selected = goals == goal
        spatial, y = _spatial(contexts[selected], high_resolution), structured[selected]
        if np.linalg.matrix_rank(spatial) != len(terms):
            raise ValueError('Spatial demonstration design is rank deficient.')
        spatial_projection = np.linalg.solve(spatial.T@spatial+ridge*np.eye(len(terms)), spatial.T)
        intermediate = np.tensordot(spatial_projection, y, axes=(1, 0))
        coefficients = np.einsum('stj,kt->skj', intermediate, time_projection, optimize=True)
        weights.append(coefficients)
        residuals.append(np.einsum('ns,skj,tk->ntj', spatial, coefficients, basis, optimize=True)-y)
    residuals = np.asarray(residuals)
    native_rmse = np.sqrt(np.mean(residuals**2, axis=(0, 1, 2)))*np.asarray([3.]*6+[JAW_MAX_M])
    model = {'schema': HIGH_RESOLUTION_SCHEMA if high_resolution else SCHEMA, 'scope': SCOPE, 'physical_robot_ready': False,
             'supervision_origin': 'synthetic_authored_controller', 'feature_contract': FEATURE_CONTRACT,
             'reactive_contact_feedback': False, 'known_goals_xy_m': [list(g) for g in GOALS],
             'start_bounds_xy_m': [list(b) for b in START_BOUNDS], 'spatial_terms': list(terms),
             'time_centers_s': centers.tolist(), 'time_width_s': width,
             'joint_limits_rad': [list(b) for b in JOINT_LIMITS], 'joint_delta_limit_rad': JOINT_DELTA_RAD,
             'jaw_limit_m': JAW_MAX_M, 'output_encoding': OUTPUT_ENCODING,
             'weights': np.asarray(weights).tolist(),
             'dataset_sha256': digest({'source_dataset_sha256': source_hash, 'supervision_sha256': supervision_hash,
                                       'output_encoding': OUTPUT_ENCODING}), 'source_dataset_sha256': source_hash,
             'supervision_sha256': supervision_hash, 'train_cases': list(cases),
             'training': {'algorithm': 'numpy-separable-ridge-cubic-gaussian' if high_resolution else 'numpy-separable-ridge-quadratic-gaussian', 'sample_count': 32000,
                          'scene_count': 32, 'spatial_ridge': ridge, 'time_ridge': ridge,
                          'shuffled_labels': shuffled, 'label_shuffle_seed': SHUFFLE_SEED if shuffled else None,
                          'normalized_mse': float(np.mean(residuals**2)), 'native_rmse': native_rmse.tolist(),
                          'label_conversion': 'q_feature*3 + clipped_delta_label*.06; float32 source precision',
                          'weak_control_notice': 'Full frame-wise label shuffle; weak temporal negative control, not whole-trajectory shuffle.'}}
    return validate_model(model)


class MotorPrimitivePolicy:
    """Fresh per episode: latch initial XY at t=0 and follow learned references."""
    def __init__(self, document):
        self.model = validate_model(document)
        self.weights = np.asarray(self.model['weights'])
        self.high_resolution = self.model['schema'] == HIGH_RESOLUTION_SCHEMA
        self.calls, self.inference_seconds = 0, 0.
        self.initial_xy, self.goal_index = None, None

    def predict(self, observation):
        began = time.perf_counter()
        if type(observation) is not dict:
            raise ValueError('Primitive requires an observation dictionary.')
        elapsed = _number(observation.get('t_seconds'), 't_seconds', 0., 20.)
        if not math.isclose(elapsed, self.calls*.02, rel_tol=0, abs_tol=1e-8):
            raise ValueError('Primitive requires t=0 initialization followed by 20 ms action cadence.')
        q = _array(observation.get('joint_position_rad'), (6,), 'joint_position_rad', 10.)
        goal = _goal(observation.get('target_position_m'))
        initial_xy = _start(observation.get('object_position_m')) if self.calls == 0 else self.initial_xy
        if self.calls and goal != self.goal_index:
            raise ValueError('Primitive destination cannot change after initialization.')
        reference = np.einsum('s,t,stj->j', _spatial(initial_xy, self.high_resolution),
                              _time_basis(elapsed, self.high_resolution), self.weights[goal], optimize=True)
        limits = np.asarray(JOINT_LIMITS)
        targets = np.clip(q+np.clip(reference[:6]*3.-q, -JOINT_DELTA_RAD, JOINT_DELTA_RAD),
                          limits[:, 0]+.02, limits[:, 1]-.02)
        self.initial_xy, self.goal_index = initial_xy, goal
        self.calls += 1
        self.inference_seconds += time.perf_counter()-began
        return {'joint_targets_rad': targets.tolist(), 'gripper_position_m': float(JAW_MAX_M*np.clip(reference[6], 0., 1.))}

    __call__ = predict

    def diagnostics(self):
        return {'policy_origin': 'learned_conditional_motion_primitive', 'action_representation': 'absolute_time_conditioned',
                'student_inference_calls': self.calls, 'student_inference_seconds': self.inference_seconds,
                'teacher_fallback_calls': 0, 'runtime_teacher_available': False, 'authored_task_phase_machine': False,
                'reactive_contact_feedback': False, 'physical_robot_ready': False, 'model_sha256': digest(self.model)}
