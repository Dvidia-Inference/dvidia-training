"""A learned absolute-jaw adapter preserving the frozen arm policy exactly.

This is a new architecture, not a repair of a frozen residual-model result.
It requires a new protocol and fresh confirmation. Training may read the 64
teacher episodes from its parent, including failed prefixes; DAgger and parent
confirmation are excluded. At inference a compact data-only classification tree
reads causal grip measurements, while MotorPrimitivePolicy supplies all six
arm targets unchanged. No teacher, IK, task gate or intervention ID is present.
"""
from __future__ import annotations

from collections import deque
from hashlib import sha256
import gzip
import io
import json
from pathlib import Path
import time

import numpy as np

from . import piper_imitation as core
from . import piper_motor_primitive as primitive
from .reference_replay import _read


SCHEMA = 'dvidia.piper-learned-grip-adapter/v1'
FEATURE_CONTRACT = 'grip56-context-current-lag40-lag120-no-joint-control/v1'
SCOPE = ('New simulation-only learned absolute-jaw adapter, preserving frozen v0.1 arm commands exactly. '
         'Synthetic authored-expert labels and privileged causal grip measurements. '
         'Requires a new protocol and fresh confirmation; no camera or physical robot qualification.')
SNAPSHOT_NAMES = (tuple(f'object_minus_tcp_{a}_m/.05' for a in 'xyz')
                  + tuple(f'target_minus_object_{a}_m/.2' for a in 'xyz')
                  + tuple(f'object_linear_velocity_{a}_m_s/.2' for a in 'xyz')
                  + ('jaw_left_m/.035', 'jaw_right_m/.035',
                     'log1p_left_force_n/3', 'log1p_right_force_n/3',
                     'left_finger_contacts/2', 'right_finger_contacts/2', 'table_contacts/2'))
CONTEXT_NAMES = (tuple(f'reset_object_{a}_m/.5' for a in 'xy')
                 + tuple(f'fixed_goal_{a}_m/.5' for a in 'xy')
                 + ('elapsed_seconds/20', 'frozen_nominal_jaw_m/.035',
                    'lag40_actual_age_s/.12', 'lag120_actual_age_s/.12'))
FEATURE_NAMES = CONTEXT_NAMES + tuple(f'{lag}:{name}' for lag in ('current', 'lag40', 'lag120') for name in SNAPSHOT_NAMES)
FEATURE_COUNT = len(FEATURE_NAMES)
LIVE_START = len(CONTEXT_NAMES)
MODEL_FIELDS = {'schema', 'scope', 'physical_robot_ready', 'feature_contract', 'feature_names',
                'blind', 'base_sha256', 'arm_command_contract', 'jaw_classes_m', 'nodes', 'training'}
NODE_FIELDS = {'feature', 'threshold', 'left', 'right', 'open_probability', 'samples', 'weight_sum'}
TRAINING_FIELDS = {'algorithm', 'max_depth', 'min_leaf', 'max_bins', 'sample_count', 'scene_count',
                   'train_groups', 'dataset_sha256', 'weight_sum', 'weighted_classification_error',
                   'classification_error', 'brier_mse', 'jaw_mae_m', 'parent_artifacts'}


class GripFeatures:
    """56 bounded features, with no q/qdot/absolute-TCP/actuator-control inputs.

    Only the jaw component of ``nominal_action`` is read. Current and lagged
    state is privileged simulation sensing. Past snapshots are selected at or
    before t−40/t−120 ms, or the earliest retained in-window sample during
    warmup/irregular cadence. No snapshot older than 120 ms is retained.
    """
    def __init__(self, *, blind=False):
        if type(blind) is not bool:
            raise ValueError('blind must be a boolean.')
        self.blind, self.history = blind, deque()
        self.last_time = self.initial_xy = self.goal = None

    def features(self, observation, nominal_action):
        if type(observation) is not dict or type(nominal_action) is not dict:
            raise ValueError('Grip features require an observation and nominal action mapping.')
        elapsed = core._number(observation.get('t_seconds'), 't_seconds', 0., 20.)
        if self.last_time is None and elapsed != 0. or self.last_time is not None and elapsed <= self.last_time:
            raise ValueError('Grip features require t=0 followed by strictly increasing timestamps.')
        goal = core._array(observation.get('target_position_m'), (3,), 'target_position_m', 1.)
        if self.goal is not None and not np.array_equal(goal, self.goal):
            raise ValueError('Grip destination cannot change within an episode.')
        initial = primitive._start(observation.get('object_position_m')) if self.initial_xy is None else self.initial_xy
        jaw = core._number(nominal_action.get('gripper_position_m'), 'nominal jaw', 0., core.JAW_MAX_M)
        if self.blind:
            current = np.zeros(len(SNAPSHOT_NAMES))
        else:
            try:
                obj = core._array(observation['object_position_m'], (3,), 'object_position_m', 10.)
                tcp = core._array(observation['tcp_position_m'], (3,), 'tcp_position_m', 10.)
                velocity = core._array(observation['object_linear_velocity_m_s'], (3,), 'object_linear_velocity_m_s', 1e4)
                opening = core._array(observation['gripper_joint_position_m'], (2,), 'gripper_joint_position_m', 1.)
                forces = [core._number(observation['pad_normal_forces_n'][side], 'pad force', 0., 1e6) for side in ('left', 'right')]
                contacts = [core._integer(observation['grasp_contact_counts'][side], 'finger contacts', 0, 10000) for side in ('left', 'right')]
                table = core._integer(observation['table_object_contact_count'], 'table contacts', 0, 10000)
            except (KeyError, TypeError) as exc:
                raise ValueError('Missing or invalid causal grip measurement.') from exc
            current = np.clip(np.r_[(obj-tcp)/.05, (goal-obj)/.2, velocity/.2,
                                    opening/core.JAW_MAX_M, np.log1p(forces)/3.,
                                    np.asarray(contacts)/2., table/2.], -10., 10.)
        history = [(t, x) for t, x in (*self.history, (elapsed, current)) if t >= elapsed-.12-1e-8]
        if len(history) > 4096:
            raise ValueError('Too many observations in the bounded grip history.')
        past = []
        for lag in (.04, .12):
            earlier = [row for row in history if row[0] <= elapsed-lag+1e-8]
            past.append(earlier[-1] if earlier else history[0])
        result = np.r_[initial/.5, goal[:2]/.5, elapsed/20., jaw/core.JAW_MAX_M,
                       [min(.12, elapsed-row[0])/.12 for row in past], current, past[0][1], past[1][1]]
        if not np.isfinite(result).all() or np.any(np.abs(result) > 10.):
            raise ValueError('Grip features must be finite and bounded.')
        self.history = deque(history)
        self.initial_xy, self.goal, self.last_time = initial.copy(), goal.copy(), elapsed
        return result

    __call__ = features


def _dataset_hash(x, labels, groups, weights):
    h = sha256(core.canonical({'shape': list(x.shape), 'groups': groups}))
    for value in (x, labels, weights):
        h.update(np.asarray(value, dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def _probability(nodes, features):
    index = 0
    while nodes[index]['feature'] != -1:
        node = nodes[index]
        index = node['left'] if features[node['feature']] <= node['threshold'] else node['right']
    return nodes[index]['open_probability']


def train(dataset, *, base_sha256, blind=False, max_depth=6, min_leaf=64, max_bins=31,
          sample_weights=None, parent_artifacts=None):
    """Deterministic weighted binary CART; both jaw labels remain native metres.

    Dataset has features (N×56), targets (N; exactly 0 or .035 m), groups (N
    episode IDs). All supplied finite labels participate, including unsuccessful
    expert prefixes. Ties choose the first feature/threshold and closed class.
    No success flags, phase or fault identifiers are features.
    """
    primitive._hash(base_sha256, 'base_sha256')
    if type(blind) is not bool:
        raise ValueError('blind must be a boolean.')
    core._integer(max_depth, 'max_depth', 1, 10)
    core._integer(min_leaf, 'min_leaf', 2, 10000)
    core._integer(max_bins, 'max_bins', 3, 127)
    if type(dataset) is not dict or set(dataset) != {'features', 'targets', 'groups'}:
        raise ValueError('Grip dataset needs features, absolute jaw targets and episode groups.')
    groups = dataset['groups']
    if type(groups) not in (list, tuple) or not 2 <= len(groups) <= 1_000_000 or any(type(g) is not str for g in groups):
        raise ValueError('Grip samples need whole-episode string group IDs.')
    groups = list(groups)
    roster = core._case_ids(list(dict.fromkeys(groups)))
    x = core._array(dataset['features'], (len(groups), FEATURE_COUNT), 'features', 10.)
    y = core._array(dataset['targets'], (len(groups),), 'absolute jaw labels', core.JAW_MAX_M)
    if np.any(np.minimum(abs(y), abs(y-core.JAW_MAX_M)) > 1e-12):
        raise ValueError('This binary adapter requires expert jaw labels 0 or .035 metres.')
    if blind and np.any(x[:, LIVE_START:] != 0.):
        raise ValueError('Blind grip features must zero every live/history slot.')
    labels = (y > core.JAW_MAX_M/2.).astype(float)
    weights = core._array(np.ones(len(y)) if sample_weights is None else sample_weights, (len(y),), 'sample weights', 100.)
    if np.any(weights <= 0.):
        raise ValueError('Sample weights must be strictly positive.')
    nodes = []

    def grow(indices, depth):
        weight = weights[indices]
        positive = float(np.dot(weight, labels[indices])); total = float(weight.sum())
        probability = positive/total
        index = len(nodes)
        node = dict(feature=-1, threshold=0., left=-1, right=-1,
                    open_probability=probability, samples=len(indices), weight_sum=total)
        nodes.append(node)
        if depth >= max_depth or len(indices) < 2*min_leaf or probability in (0., 1.):
            return index
        impurity = 2.*positive*(total-positive)/total
        best_gain, best = 1e-12, None
        for feature in range(FEATURE_COUNT):
            values = x[indices, feature]
            distinct = np.unique(values)
            if len(distinct) < 2:
                continue
            if len(distinct) <= max_bins+1:
                thresholds = (distinct[:-1]+distinct[1:])/2.
            else:
                thresholds = np.unique(np.quantile(values, np.arange(1, max_bins+1)/(max_bins+1)))
                thresholds = thresholds[(thresholds >= distinct[0]) & (thresholds < distinct[-1])]
            bins = np.searchsorted(thresholds, values, side='left')
            count = np.cumsum(np.bincount(bins, minlength=len(thresholds)+1))[:-1]
            left_w = np.cumsum(np.bincount(bins, weights=weight, minlength=len(thresholds)+1))[:-1]
            left_p = np.cumsum(np.bincount(bins, weights=weight*labels[indices], minlength=len(thresholds)+1))[:-1]
            right_w, right_p = total-left_w, positive-left_p
            eligible = (count >= min_leaf) & (len(indices)-count >= min_leaf) & (left_w > 0.) & (right_w > 0.)
            gain = np.full(len(thresholds), -np.inf)
            gain[eligible] = impurity-(2.*left_p[eligible]*(left_w[eligible]-left_p[eligible])/left_w[eligible]
                                         +2.*right_p[eligible]*(right_w[eligible]-right_p[eligible])/right_w[eligible])
            if len(gain):
                choice = int(np.argmax(gain))
                if gain[choice] > best_gain:
                    best_gain, best = float(gain[choice]), (feature, float(thresholds[choice]))
        if best is not None:
            feature, threshold = best
            left_mask = x[indices, feature] <= threshold
            node.update(feature=feature, threshold=threshold,
                        left=grow(indices[left_mask], depth+1), right=grow(indices[~left_mask], depth+1))
        return index

    grow(np.arange(len(y)), 0)
    probabilities = np.asarray([_probability(nodes, row) for row in x])
    predicted = probabilities > .5
    incorrect = predicted != labels
    model = dict(schema=SCHEMA, scope=SCOPE, physical_robot_ready=False, feature_contract=FEATURE_CONTRACT,
                 feature_names=list(FEATURE_NAMES), blind=blind, base_sha256=base_sha256,
                 arm_command_contract='original MotorPrimitivePolicy six joint targets unchanged',
                 jaw_classes_m=[0., core.JAW_MAX_M], nodes=nodes,
                 training=dict(algorithm='numpy-weighted-binned-binary-cart', max_depth=max_depth,
                     min_leaf=min_leaf, max_bins=max_bins, sample_count=len(y), scene_count=len(roster),
                     train_groups=roster, dataset_sha256=_dataset_hash(x, y, groups, weights), weight_sum=float(weights.sum()),
                     weighted_classification_error=float(np.average(incorrect, weights=weights)),
                     classification_error=float(np.mean(incorrect)), brier_mse=float(np.average((probabilities-labels)**2, weights=weights)),
                     jaw_mae_m=float(np.mean(incorrect)*core.JAW_MAX_M), parent_artifacts=parent_artifacts))
    return validate_model(model)


def validate_model(document):
    """Bounded finite JSON tree only; reject executable fields, cycles and drift."""
    model = core._decode(document)
    if type(model) is not dict or set(model) != MODEL_FIELDS:
        raise ValueError('Unknown or missing grip-adapter field.')
    fixed = dict(schema=SCHEMA, scope=SCOPE, physical_robot_ready=False,
                 feature_contract=FEATURE_CONTRACT, feature_names=list(FEATURE_NAMES),
                 arm_command_contract='original MotorPrimitivePolicy six joint targets unchanged',
                 jaw_classes_m=[0., core.JAW_MAX_M])
    if any(type(model[k]) is not type(v) or model[k] != v for k, v in fixed.items()) or type(model['blind']) is not bool:
        raise ValueError('Unsupported grip-adapter scope, feature or native action contract.')
    core._array(model['jaw_classes_m'], (2,), 'jaw_classes_m', core.JAW_MAX_M)
    primitive._hash(model['base_sha256'], 'base_sha256')
    training = model['training']
    if type(training) is not dict or set(training) != TRAINING_FIELDS or training['algorithm'] != 'numpy-weighted-binned-binary-cart':
        raise ValueError('Unsupported grip training receipt.')
    depth = core._integer(training['max_depth'], 'max_depth', 1, 10)
    core._integer(training['min_leaf'], 'min_leaf', 2, 10000)
    core._integer(training['max_bins'], 'max_bins', 3, 127)
    count = core._integer(training['sample_count'], 'sample_count', 2, 1_000_000)
    groups = core._case_ids(training['train_groups'])
    if training['scene_count'] != len(groups) or type(training['scene_count']) is not int or len(groups) > count:
        raise ValueError('Invalid training episode roster.')
    primitive._hash(training['dataset_sha256'], 'dataset_sha256')
    core._number(training['weight_sum'], 'weight_sum', 1e-12, count*100.)
    for key in ('weighted_classification_error', 'classification_error', 'brier_mse'):
        core._number(training[key], key, 0., 1.)
    core._number(training['jaw_mae_m'], 'jaw_mae_m', 0., core.JAW_MAX_M)
    parent = training['parent_artifacts']
    if parent is not None:
        if type(parent) is not dict or set(parent) != {'protocol_sha256', 'base_model_raw_sha256', 'episodes'}:
            raise ValueError('Invalid parent training lineage.')
        for key in ('protocol_sha256', 'base_model_raw_sha256'):
            primitive._hash(parent[key], key)
        if type(parent['episodes']) is not list or len(parent['episodes']) != len(groups):
            raise ValueError('Parent episode roster differs from training groups.')
        total = 0
        for row, group in zip(parent['episodes'], groups):
            if type(row) is not dict or set(row) != {'case_id', 'episode_sha256', 'samples'} or row['case_id'] != group:
                raise ValueError('Invalid parent training episode binding.')
            primitive._hash(row['episode_sha256'], 'episode_sha256')
            total += core._integer(row['samples'], 'samples', 1, 1000)
        if total != count:
            raise ValueError('Parent sample counts do not match the trained dataset.')
    nodes = model['nodes']
    if type(nodes) is not list or not 1 <= len(nodes) <= 2**(depth+1)-1:
        raise ValueError('Tree exceeds its bounded capacity.')
    visited = set()

    def check(index, level):
        if type(index) is not int or not 0 <= index < len(nodes) or index in visited or level > depth:
            raise ValueError('Invalid, cyclic or over-deep grip tree.')
        visited.add(index)
        node = nodes[index]
        if type(node) is not dict or set(node) != NODE_FIELDS:
            raise ValueError('Unknown grip tree node field.')
        feature = core._integer(node['feature'], 'feature', -1, FEATURE_COUNT-1)
        core._number(node['threshold'], 'threshold', -10., 10.)
        core._number(node['open_probability'], 'open_probability', 0., 1.)
        core._integer(node['samples'], 'samples', 1, count)
        core._number(node['weight_sum'], 'node weight_sum', 1e-12, count*100.)
        if feature == -1:
            if node['left'] != -1 or node['right'] != -1 or type(node['left']) is not int or type(node['right']) is not int:
                raise ValueError('Leaf nodes must have no executable branch.')
        else:
            if model['blind'] and feature >= LIVE_START:
                raise ValueError('Blind tree cannot read live features.')
            check(node['left'], level+1); check(node['right'], level+1)
            left, right = nodes[node['left']], nodes[node['right']]
            if (left['samples']+right['samples'] != node['samples']
                    or not np.isclose(left['weight_sum']+right['weight_sum'], node['weight_sum'], rtol=1e-10, atol=1e-9)):
                raise ValueError('Tree sample or weight accounting is inconsistent.')
    check(0, 0)
    if len(visited) != len(nodes) or nodes[0]['samples'] != count or not np.isclose(nodes[0]['weight_sum'], training['weight_sum']):
        raise ValueError('Tree contains unreachable data or inconsistent training counts.')
    return model


def train_parent_data(parent, *, blind=False, max_depth=6, min_leaf=64, max_bins=31):
    """Fit only the parent's 64 literal teacher episodes; never DAgger/confirmation."""
    parent = Path(parent)
    protocol_raw = _read(parent/'protocol.json', 1024*1024)
    protocol = json.loads(protocol_raw)
    rows = protocol['cases']['train']
    if type(rows) is not list or len(rows) != 64:
        raise ValueError('Grip lineage requires its 64 scheduled parent teacher episodes.')
    base_raw = _read(parent/'base-model.json', core.MAX_MODEL_BYTES)
    base = primitive.validate_model(base_raw)
    features, targets, groups, weights, bindings = [], [], [], [], []
    for case in rows:
        path = parent/'demonstrations'/case['id']/'episode.json.gz'
        raw = _read(path, 32*1024*1024)
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            decoded = stream.read(128*1024*1024+1)
        if len(decoded) > 128*1024*1024:
            raise ValueError('Parent teacher episode exceeds its bounded decoded size.')
        episode = json.loads(decoded)
        if (episode.get('case_id') != case['id'] or episode.get('policy_kind') != 'teacher'
                or episode.get('seed') != case['seed'] or episode.get('pulses') != case['pulses']
                or episode.get('base_model_sha256') != core.digest(base)):
            raise ValueError('Parent teacher episode does not match its scheduled training identity.')
        actions, states = episode['actions'], episode['trajectory']
        if type(actions) is not list or type(states) is not list or not 1 <= len(actions) <= 1000 or len(states) != len(actions)+1:
            raise ValueError('Parent teacher episode must retain its complete causal action prefix.')
        nominal, encoder = primitive.MotorPrimitivePolicy(base), GripFeatures(blind=blind)
        for obs, action in zip(states[:-1], actions):
            expert = action.get('expert_action')
            if type(expert) is not dict or set(expert) != {'joint_targets_rad', 'gripper_position_m'}:
                raise ValueError('Parent collection must preserve its original expert action label.')
            label = core._number(expert['gripper_position_m'], 'expert jaw target', 0., core.JAW_MAX_M)
            features.append(encoder(obs, nominal(obs))); targets.append(label); groups.append(case['id'])
            # Training-only weighting from the existing collector is retained;
            # its authored phase never enters the learned feature contract.
            weights.append(5. if action.get('expert_phase') in ('close', 'lift', 'transfer', 'place', 'release') else 1.)
        bindings.append(dict(case_id=case['id'], episode_sha256=sha256(raw).hexdigest(), samples=len(actions)))
    return train(dict(features=np.asarray(features), targets=np.asarray(targets), groups=groups),
                 base_sha256=core.digest(base), blind=blind, max_depth=max_depth,
                 min_leaf=min_leaf, max_bins=max_bins, sample_weights=np.asarray(weights),
                 parent_artifacts=dict(protocol_sha256=sha256(protocol_raw).hexdigest(),
                                       base_model_raw_sha256=sha256(base_raw).hexdigest(), episodes=bindings))


class GripPolicy:
    """Original six arm targets unchanged; only the jaw is learned."""
    def __init__(self, base_document, document, *, blind=None):
        self.model = validate_model(document)
        self.nominal = primitive.MotorPrimitivePolicy(base_document)
        if self.model['base_sha256'] != core.digest(self.nominal.model):
            raise ValueError('Grip adapter is bound to a different frozen arm model.')
        if blind is not None and (type(blind) is not bool or blind != self.model['blind']):
            raise ValueError('Grip runtime mode differs from its trained feature mode.')
        self.encoder = GripFeatures(blind=self.model['blind'])
        self.calls, self.inference_seconds, self.last_open_probability = 0, 0., None

    def __call__(self, observation):
        began = time.perf_counter()
        before = (self.nominal.calls, self.nominal.initial_xy, self.nominal.goal_index, self.nominal.inference_seconds)
        try:
            action = self.nominal(observation)
            feature = self.encoder(observation, action)
        except ValueError:
            self.nominal.calls, self.nominal.initial_xy, self.nominal.goal_index, self.nominal.inference_seconds = before
            raise
        probability = _probability(self.model['nodes'], feature)
        self.last_open_probability = probability
        self.calls += 1
        self.inference_seconds += time.perf_counter()-began
        return {'joint_targets_rad': action['joint_targets_rad'],
                'gripper_position_m': core.JAW_MAX_M if probability > .5 else 0.}

    predict = __call__

    def diagnostics(self):
        return dict(policy_origin='learned_absolute_jaw_adapter', blind=self.model['blind'],
                    original_arm_geometry_preserved=True, nominal_clock='physical_elapsed_time',
                    student_inference_calls=self.calls, student_inference_seconds=self.inference_seconds,
                    last_open_probability=self.last_open_probability, maximum_history_seconds=.12,
                    teacher_fallback_calls=0, runtime_teacher_available=False,
                    authored_task_phase_machine=False, runtime_ik_available=False,
                    reactive_contact_feedback=not self.model['blind'], physical_robot_ready=False,
                    base_model_sha256=self.model['base_sha256'], grip_model_sha256=core.digest(self.model))
