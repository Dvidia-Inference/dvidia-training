"""A data-only learned hold/advance clock for the frozen PiPER path.

The clock predicts one binary progress increment from causal pre-action state.
V1 retains the frozen physical-time jaw adapter. V2 coordinates the original
primitive's arm and jaw references with the same virtual progress. Neither
version learns new path geometry or a new jaw shape. Privileged simulator
features do not establish camera or hardware competence.
"""
from __future__ import annotations

from hashlib import sha256
import re

import numpy as np

from . import piper_grip_adapter as grip
from . import piper_imitation as core
from . import piper_motor_primitive as primitive


SCHEMA = 'dvidia.piper-grip-progress-clock/v1'
COORDINATED_SCHEMA = 'dvidia.piper-grip-progress-clock/v2'
FEATURE_CONTRACT = 'grip56-physical-time-plus-causal-progress-and-debt/v1'
SCOPE = ('Simulation-only learned binary hold/advance clock for a frozen PiPER six-joint path; '
         'synthetic authored-expert supervision and privileged causal grip sensing. '
         'Frozen jaw adapter uses physical elapsed time. No camera, new path geometry, '
         'runtime teacher, IK or physical robot qualification.')
COORDINATED_SCOPE = ('Simulation-only learned binary hold/advance clock coordinating all seven frozen PiPER '
                     'primitive arm/jaw references at shared virtual progress; synthetic authored-expert '
                     'supervision and privileged causal grip sensing. Physical elapsed time and its nominal '
                     'jaw reference remain clock features. No new jaw policy or path geometry, camera, '
                     'runtime teacher, IK or physical robot qualification.')
INTERVAL_S = .02
HORIZON_S = 20.
FEATURE_NAMES = grip.FEATURE_NAMES + ('virtual_progress_seconds/20', 'physical_minus_virtual_seconds/20')
FEATURE_COUNT = len(FEATURE_NAMES)
LIVE_START = grip.LIVE_START
LIVE_STOP = grip.FEATURE_COUNT
ELAPSED_INDEX = grip.CONTEXT_NAMES.index('elapsed_seconds/20')
ARM_CONTRACT = 'binary hold/advance of frozen v0.1 path; no learned geometry correction'
JAW_CONTRACT = 'frozen v0.2 jaw adapter evaluated at physical elapsed time, unchanged'
COORDINATED_JAW_CONTRACT = ('Original primitive jaw reference evaluated at shared virtual progress; '
                          'physical time retained in features')
JAW_MODES = ('physical_adapter', 'coordinated_reference')
MODEL_FIELDS = {'schema', 'scope', 'physical_robot_ready', 'feature_contract', 'feature_names',
                'blind', 'base_sha256', 'jaw_sha256', 'arm_command_contract', 'jaw_command_contract',
                'control_interval_s', 'progress_horizon_s', 'progress_increment_classes_s',
                'decision_rule', 'nodes', 'training'}
NODE_FIELDS = {'feature', 'threshold', 'left', 'right', 'advance_probability', 'samples', 'weight_sum'}
TRAINING_FIELDS = {'algorithm', 'max_depth', 'min_leaf', 'max_bins', 'sample_count', 'scene_count',
                   'train_groups', 'group_samples', 'groups_sha256', 'dataset_sha256', 'weight_sum',
                   'weighted_classification_error', 'classification_error', 'brier_mse',
                   'progress_mae_s', 'parent_artifacts'}
_TOL = 1e-8


class ClockFeatures:
    """The frozen 56 grip features followed by causal progress and clock debt.

    ``progress_s`` is the progress *before* the action now being chosen. Physical
    observations start at zero and arrive every 20 ms. Progress starts at zero
    and may hold or advance one 20 ms tick per observation, never ahead of time.
    The frozen encoder retains at most 120 ms of actual observed history. Blind
    mode zeros only its 48 live/history slots; it retains context and both clocks.
    """
    def __init__(self, *, blind=False):
        self.grip = grip.GripFeatures(blind=blind)
        self.blind = blind
        self.last_time = self.last_progress_s = None

    def features(self, observation, nominal_physical_action, progress_s):
        if type(observation) is not dict:
            raise ValueError('Clock features require an observation mapping.')
        elapsed = core._number(observation.get('t_seconds'), 't_seconds', 0., HORIZON_S)
        progress = core._number(progress_s, 'progress_s', 0., HORIZON_S)
        if self.last_time is None:
            if elapsed != 0. or progress != 0.:
                raise ValueError('Clock features require physical time and progress to start at zero.')
        else:
            if abs(elapsed-self.last_time-INTERVAL_S) > _TOL:
                raise ValueError('Clock observations must advance by exactly 20 ms.')
            increment = progress-self.last_progress_s
            if min(abs(increment), abs(increment-INTERVAL_S)) > _TOL or increment < -_TOL:
                raise ValueError('Virtual progress must hold or advance one 20 ms tick, never backtrack.')
        if progress > elapsed+_TOL:
            raise ValueError('Virtual progress cannot use future physical time.')
        # Validate progress before invoking the stateful frozen encoder. Rejected
        # clock inputs cannot consume a history sample or poison a later retry.
        original = self.grip.features(observation, nominal_physical_action)
        result = np.r_[original, progress/HORIZON_S, max(0., elapsed-progress)/HORIZON_S]
        self.last_time, self.last_progress_s = elapsed, progress
        return result

    __call__ = features


def _features(value, shape, *, blind):
    x = core._array(value, shape, 'clock features', 10.)
    rows = x.reshape((-1, FEATURE_COUNT))
    clocks = rows[:, [ELAPSED_INDEX, FEATURE_COUNT-2, FEATURE_COUNT-1]]
    if (np.any(clocks < -_TOL) or np.any(clocks > 1.+_TOL)
            or np.any(np.abs(clocks[:, 0]-clocks[:, 1]-clocks[:, 2]) > _TOL)):
        raise ValueError('Clock features require causal progress plus debt equal to physical elapsed time.')
    if blind and np.any(rows[:, LIVE_START:LIVE_STOP] != 0.):
        raise ValueError('Blind clock features must zero every frozen live/history slot.')
    return x


def _dataset_hash(x, targets, groups, weights):
    h = sha256(core.canonical({'shape': list(x.shape), 'groups': groups}))
    for value in (x, targets, weights):
        h.update(np.asarray(value, dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def _parent_artifacts(value):
    if value is not None:
        if (type(value) is not dict or not 1 <= len(value) <= 32
                or any(type(key) is not str or re.fullmatch(r'[a-z][a-z0-9_]{0,79}_sha256', key) is None
                       for key in value)):
            raise ValueError('Parent artifacts must be a bounded mapping of named SHA256 digests.')
        for key, digest in value.items():
            primitive._hash(digest, key)
    return value


def _probability(nodes, features):
    index = 0
    while nodes[index]['feature'] != -1:
        node = nodes[index]
        index = node['left'] if features[node['feature']] <= node['threshold'] else node['right']
    return nodes[index]['advance_probability']


def train(dataset, *, base_sha256, jaw_sha256, blind=False, max_depth=4, min_leaf=64,
          max_bins=31, sample_weights=None, parent_artifacts=None, jaw_mode='physical_adapter'):
    """Fit deterministic weighted binned CART to native 0/.02-second labels.

    Dataset fields are exactly ``features`` (N x 58), ``targets`` (N native
    seconds) and ``groups`` (N whole-episode IDs). All supplied samples remain in
    the receipt. Split ties use the first feature/threshold, and class ties hold.
    No task outcomes, phase or future labels become inference features.

    ``physical_adapter`` preserves the V1 clock/jaw architecture. In V2,
    ``coordinated_reference`` binds ``jaw_sha256`` to the strict reference-jaw
    manifest rather than to a v0.2 GripPolicy. The runtime verifies that binding;
    this trainer neither evaluates nor changes either frozen reference.
    """
    primitive._hash(base_sha256, 'base_sha256')
    primitive._hash(jaw_sha256, 'jaw_sha256')
    if type(jaw_mode) is not str or jaw_mode not in JAW_MODES:
        raise ValueError('Clock jaw_mode must be physical_adapter or coordinated_reference.')
    coordinated = jaw_mode == 'coordinated_reference'
    if type(blind) is not bool:
        raise ValueError('blind must be a boolean.')
    if type(max_depth) is not int or max_depth not in (4, 6):
        raise ValueError('Clock max_depth must be 4 or 6.')
    core._integer(min_leaf, 'min_leaf', 2, 10000)
    core._integer(max_bins, 'max_bins', 3, 127)
    _parent_artifacts(parent_artifacts)
    if type(dataset) is not dict or set(dataset) != {'features', 'targets', 'groups'}:
        raise ValueError('Clock dataset needs features, native progress increments and episode groups.')
    groups = dataset['groups']
    if type(groups) not in (list, tuple) or not 2 <= len(groups) <= 1_000_000 or any(type(g) is not str for g in groups):
        raise ValueError('Clock samples need whole-episode string group IDs.')
    groups = list(groups)
    roster = core._case_ids(list(dict.fromkeys(groups)))
    x = _features(dataset['features'], (len(groups), FEATURE_COUNT), blind=blind)
    targets = core._array(dataset['targets'], (len(groups),), 'progress increments', INTERVAL_S)
    if not np.isin(targets, (0., INTERVAL_S)).all():
        raise ValueError('Clock labels must be exactly 0 or .02 seconds.')
    labels = (targets > 0.).astype(float)
    weights = core._array(np.ones(len(targets)) if sample_weights is None else sample_weights,
                          (len(targets),), 'sample weights', 100.)
    if np.any(weights <= 0.):
        raise ValueError('Sample weights must be strictly positive.')
    nodes = []

    def grow(indices, depth):
        weight = weights[indices]
        total = float(weight.sum())
        # dot and sum can round differently for fractional weights. Their
        # mathematical positive mass remains inside [0,total], including pure
        # leaves, so do not serialize 1+epsilon as a probability.
        positive = min(total, max(0., float(np.dot(weight, labels[indices]))))
        probability = positive/total
        index = len(nodes)
        node = dict(feature=-1, threshold=0., left=-1, right=-1,
                    advance_probability=probability, samples=len(indices), weight_sum=total)
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
            counts = np.cumsum(np.bincount(bins, minlength=len(thresholds)+1))[:-1]
            left_w = np.cumsum(np.bincount(bins, weights=weight, minlength=len(thresholds)+1))[:-1]
            left_p = np.cumsum(np.bincount(bins, weights=weight*labels[indices], minlength=len(thresholds)+1))[:-1]
            right_w, right_p = total-left_w, positive-left_p
            eligible = ((counts >= min_leaf) & (len(indices)-counts >= min_leaf)
                        & (left_w > 0.) & (right_w > 0.))
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

    grow(np.arange(len(targets)), 0)
    probabilities = np.asarray([_probability(nodes, row) for row in x])
    incorrect = (probabilities > .5) != labels
    group_counts = {group: 0 for group in roster}
    for group in groups:
        group_counts[group] += 1
    model = dict(schema=COORDINATED_SCHEMA if coordinated else SCHEMA,
                 scope=COORDINATED_SCOPE if coordinated else SCOPE, physical_robot_ready=False,
                 feature_contract=FEATURE_CONTRACT, feature_names=list(FEATURE_NAMES), blind=blind,
                 base_sha256=base_sha256, jaw_sha256=jaw_sha256,
                 arm_command_contract=ARM_CONTRACT,
                 jaw_command_contract=COORDINATED_JAW_CONTRACT if coordinated else JAW_CONTRACT,
                 control_interval_s=INTERVAL_S, progress_horizon_s=HORIZON_S,
                 progress_increment_classes_s=[0., INTERVAL_S], decision_rule='advance_probability > 0.5; ties hold',
                 nodes=nodes, training=dict(algorithm='numpy-weighted-binned-binary-cart', max_depth=max_depth,
                     min_leaf=min_leaf, max_bins=max_bins, sample_count=len(targets), scene_count=len(roster),
                     train_groups=roster, group_samples=[dict(case_id=g, samples=group_counts[g]) for g in roster],
                     groups_sha256=core.digest({'groups': groups}),
                     dataset_sha256=_dataset_hash(x, targets, groups, weights), weight_sum=float(weights.sum()),
                     weighted_classification_error=float(np.average(incorrect, weights=weights)),
                     classification_error=float(np.mean(incorrect)),
                     brier_mse=float(np.average((probabilities-labels)**2, weights=weights)),
                     progress_mae_s=float(np.mean(incorrect)*INTERVAL_S), parent_artifacts=parent_artifacts))
    return validate_model(model)


def validate_model(document, *, base_sha256=None, jaw_sha256=None):
    """Validate exact V1/V2 contracts and optional expected dependency hashes.

    Hash syntax alone does not authenticate an external dependency. Runtimes
    must verify the actual base and jaw document/manifest against these hashes.
    Expected hashes here are useful for data-only binding checks without loading
    or executing a policy.
    """
    model = core._decode(document)
    if type(model) is not dict or set(model) != MODEL_FIELDS:
        raise ValueError('Unknown or missing clock-model field.')
    if model['schema'] not in (SCHEMA, COORDINATED_SCHEMA):
        raise ValueError('Unsupported clock schema/version.')
    coordinated = model['schema'] == COORDINATED_SCHEMA
    fixed = dict(schema=COORDINATED_SCHEMA if coordinated else SCHEMA,
                 scope=COORDINATED_SCOPE if coordinated else SCOPE, physical_robot_ready=False,
                 feature_contract=FEATURE_CONTRACT, feature_names=list(FEATURE_NAMES),
                 arm_command_contract=ARM_CONTRACT,
                 jaw_command_contract=COORDINATED_JAW_CONTRACT if coordinated else JAW_CONTRACT,
                 control_interval_s=INTERVAL_S, progress_horizon_s=HORIZON_S,
                 progress_increment_classes_s=[0., INTERVAL_S], decision_rule='advance_probability > 0.5; ties hold')
    if any(type(model[k]) is not type(v) or model[k] != v for k, v in fixed.items()) or type(model['blind']) is not bool:
        raise ValueError('Unsupported clock scope, features or native progress contract.')
    core._array(model['progress_increment_classes_s'], (2,), 'progress increment classes', INTERVAL_S)
    primitive._hash(model['base_sha256'], 'base_sha256')
    primitive._hash(model['jaw_sha256'], 'jaw_sha256')
    for name, expected in (('base_sha256', base_sha256), ('jaw_sha256', jaw_sha256)):
        if expected is not None:
            primitive._hash(expected, 'expected '+name)
            if model[name] != expected:
                raise ValueError('Clock model differs from the expected '+name+' dependency.')
    training = model['training']
    if (type(training) is not dict or set(training) != TRAINING_FIELDS
            or training['algorithm'] != 'numpy-weighted-binned-binary-cart'):
        raise ValueError('Unsupported clock training receipt.')
    depth = training['max_depth']
    if type(depth) is not int or depth not in (4, 6):
        raise ValueError('Clock max_depth must be 4 or 6.')
    min_leaf = core._integer(training['min_leaf'], 'min_leaf', 2, 10000)
    core._integer(training['max_bins'], 'max_bins', 3, 127)
    count = core._integer(training['sample_count'], 'sample_count', 2, 1_000_000)
    groups = core._case_ids(training['train_groups'])
    if type(training['scene_count']) is not int or training['scene_count'] != len(groups) or len(groups) > count:
        raise ValueError('Invalid clock training episode roster.')
    receipts = training['group_samples']
    if type(receipts) is not list or len(receipts) != len(groups):
        raise ValueError('Clock sample receipts must match the training episode roster.')
    total_samples = 0
    for receipt, group in zip(receipts, groups):
        if type(receipt) is not dict or set(receipt) != {'case_id', 'samples'} or receipt['case_id'] != group:
            raise ValueError('Invalid clock episode sample receipt.')
        total_samples += core._integer(receipt['samples'], 'group samples', 1, count)
    if total_samples != count:
        raise ValueError('Clock episode sample receipts do not sum to the trained dataset.')
    for key in ('dataset_sha256', 'groups_sha256'):
        primitive._hash(training[key], key)
    total_weight = core._number(training['weight_sum'], 'weight_sum', 1e-12, count*100.)
    for key in ('weighted_classification_error', 'classification_error', 'brier_mse'):
        core._number(training[key], key, 0., 1.)
    mae = core._number(training['progress_mae_s'], 'progress_mae_s', 0., INTERVAL_S)
    if not np.isclose(mae, training['classification_error']*INTERVAL_S, rtol=1e-10, atol=1e-12):
        raise ValueError('Clock progress error does not match binary classification accounting.')
    _parent_artifacts(training['parent_artifacts'])
    nodes = model['nodes']
    if type(nodes) is not list or not 1 <= len(nodes) <= 2**(depth+1)-1:
        raise ValueError('Clock tree exceeds its bounded capacity.')
    visited = set()

    def check(index, level):
        if type(index) is not int or not 0 <= index < len(nodes) or index in visited or level > depth:
            raise ValueError('Invalid, cyclic or over-deep clock tree.')
        visited.add(index)
        node = nodes[index]
        if type(node) is not dict or set(node) != NODE_FIELDS:
            raise ValueError('Unknown clock tree node field.')
        feature = core._integer(node['feature'], 'feature', -1, FEATURE_COUNT-1)
        threshold = core._number(node['threshold'], 'threshold', -10., 10.)
        probability = core._number(node['advance_probability'], 'advance_probability', 0., 1.)
        samples = core._integer(node['samples'], 'samples', 1, count)
        weight = core._number(node['weight_sum'], 'node weight_sum', 1e-12, count*100.)
        if feature == -1:
            if (type(node['left']) is not int or node['left'] != -1
                    or type(node['right']) is not int or node['right'] != -1 or threshold != 0.):
                raise ValueError('Clock leaves require canonical empty branches and zero threshold.')
            if level > 0 and samples < min_leaf:
                raise ValueError('Clock split contains a leaf below its declared minimum size.')
        else:
            if model['blind'] and LIVE_START <= feature < LIVE_STOP:
                raise ValueError('Blind clock tree cannot read live/history features.')
            check(node['left'], level+1); check(node['right'], level+1)
            left, right = nodes[node['left']], nodes[node['right']]
            positive_weight = (left['weight_sum']*left['advance_probability']
                               +right['weight_sum']*right['advance_probability'])
            if (left['samples']+right['samples'] != samples
                    or not np.isclose(left['weight_sum']+right['weight_sum'], weight, rtol=1e-10, atol=1e-9)
                    or not np.isclose(positive_weight, weight*probability, rtol=1e-10, atol=1e-9)):
                raise ValueError('Clock tree sample, weight or class accounting is inconsistent.')
    check(0, 0)
    if (len(visited) != len(nodes) or nodes[0]['samples'] != count
            or not np.isclose(nodes[0]['weight_sum'], total_weight, rtol=1e-10, atol=1e-9)):
        raise ValueError('Clock tree contains unreachable data or inconsistent training counts.')
    return model


def jaw_mode(document):
    """Return architecture mode only after the complete model is validated."""
    model = validate_model(document)
    return 'coordinated_reference' if model['schema'] == COORDINATED_SCHEMA else 'physical_adapter'


def predict(document, features):
    """Return a hold/advance decision and probability; a probability tie holds."""
    model = validate_model(document)
    x = _features(features, (FEATURE_COUNT,), blind=model['blind'])
    probability = float(_probability(model['nodes'], x))
    return {'advance': probability > .5, 'advance_probability': probability}
