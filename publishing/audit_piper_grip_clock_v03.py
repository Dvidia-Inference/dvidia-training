"""Independent, read-only numeric audit of a frozen PiPER progress-clock study.

No environment construction, physics, policy rollout, fit or lifecycle report is
used. Raw references, causal features, readiness and tree decisions are rebuilt
here; the unchanged strict numeric scorer is reused only after source checks.
"""
from __future__ import annotations

import argparse
from collections import deque
import gzip
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


SCHEMA = 'dvidia.piper-grip-clock-independent-audit/v1'
SIDES = ('left', 'right')
VARIANTS = ('clock', 'blind_clock', 'fixed', 'teacher')
TOL = 2e-10


def _raw(path, limit=128*1024*1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Missing, linked or oversized audit input: '+path.name)
    return path.read_bytes()


def _hash(path):
    return sha256(_raw(path)).hexdigest()


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError('Duplicate JSON key.')
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON.')))


def _read(path):
    return _json(_raw(path))


def _canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _episode(path):
    with gzip.GzipFile(fileobj=io.BytesIO(_raw(path, 32*1024*1024))) as stream:
        raw = stream.read(128*1024*1024+1)
    if len(raw) > 128*1024*1024:
        raise ValueError('Decoded episode exceeds the audit limit.')
    return _json(raw)


def _npz(path):
    raw = _raw(path, 32*1024*1024)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (len(entries) > 64 or len({entry.filename for entry in entries}) != len(entries)
                or sum(info.file_size for info in entries) > 64*1024*1024):
            raise ValueError('Decoded numeric sidecar exceeds the audit limit.')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    if any(value.dtype.kind not in 'fiuU' for value in result.values()):
        raise ValueError('Numeric sidecar contains an unsupported array type.')
    return result


def local_offset(position, tcp, quaternion):
    offset = np.asarray(position, dtype=float)-np.asarray(tcp, dtype=float)
    quat = np.asarray(quaternion, dtype=float)
    if offset.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[offset, quat]).all():
        raise ValueError('Invalid measured pose for grip-local coordinates.')
    if not math.isclose(float(quat@quat), 1., abs_tol=1e-6, rel_tol=0.):
        raise ValueError('Measured tool quaternion is not normalized.')
    w, x, y, z = quat
    rotation = np.asarray(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                           (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                           (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))
    return rotation.T@offset


class References:
    """Evaluate the serialized fixed basis, not a policy or action tape."""
    def __init__(self, base, initial, target):
        self.base = base
        self.initial = np.asarray(initial, dtype=float)[:2]
        self.goal = next((i for i, goal in enumerate(base['known_goals_xy_m'])
                          if np.allclose(goal, np.asarray(target)[:2], atol=1e-8, rtol=0)), None)
        if self.goal is None:
            raise ValueError('Unknown destination in raw reference audit.')
        x, y = (self.initial-np.asarray([.299, -.03]))/.015
        if base['schema'] == 'dvidia.piper-motor-primitive/v2':
            self.spatial = np.asarray([x**i*y**j for i in range(4) for j in range(4)])
        elif base['schema'] == 'dvidia.piper-motor-primitive/v1':
            self.spatial = np.asarray([1., y, y*y, x, x*y, x*y*y, x*x, x*x*y, x*x*y*y])
        else:
            raise ValueError('Unsupported raw-reference schema.')
        self.weights = np.asarray(base['weights'], dtype=float)[self.goal]
        self.centers = np.asarray(base['time_centers_s'], dtype=float)
        self.width = base['time_width_s']
        self.cache = {}

    def raw(self, seconds):
        if not math.isfinite(seconds) or not 0. <= seconds <= 20.:
            raise ValueError('Raw reference time exceeds its frozen domain.')
        tick = round(seconds/.02)
        if not math.isclose(seconds, tick*.02, abs_tol=1e-8, rel_tol=0):
            raise ValueError('Raw reference requires complete virtual 20 ms ticks.')
        if tick not in self.cache:
            logits = -.5*((tick*.02-self.centers)/self.width)**2
            basis = np.exp(logits-np.max(logits))
            basis /= basis.sum()
            self.cache[tick] = np.einsum('s,t,stj->j', self.spatial, basis, self.weights, optimize=True)
        return self.cache[tick].copy()

    def action(self, state, seconds):
        reference = self.raw(seconds)
        q, limits = np.asarray(state['joint_position_rad']), np.asarray(self.base['joint_limits_rad'])
        targets = np.clip(q+np.clip(reference[:6]*3.-q, -self.base['joint_delta_limit_rad'], self.base['joint_delta_limit_rad']),
                          limits[:, 0]+.02, limits[:, 1]-.02)
        return targets, float(self.base['jaw_limit_m']*np.clip(reference[6], 0., 1.))

    def window(self, offset):
        jaws = [self.base['jaw_limit_m']*np.clip(self.raw(i*.02)[6], 0., 1.) for i in range(1001)]
        closed = next((i for i, value in enumerate(jaws) if value < .01), None)
        if closed is None:
            raise ValueError('Frozen reference has no closing interval.')
        release = next((i for i in range(closed+1, len(jaws)) if jaws[i] > .03), None)
        gate = closed+round(offset/.02)
        if release is None or not 0 <= gate < release:
            raise ValueError('Readiness window differs from its bounded frozen path.')
        return gate, release


class CausalMeasurements:
    """Rebuild causal 58-input rows and sampled teacher readiness from states."""
    def __init__(self, initial, target, config):
        self.initial, self.target = np.asarray(initial)[:2], np.asarray(target)
        self.config = config
        self.feature_history, self.ready_history = deque(), deque()
        self.last_time = None
        self.acquisition_released = False

    def observe(self, state, nominal_jaw, progress, gate, release):
        t = state['t_seconds']
        if self.last_time is None and t != 0. or self.last_time is not None and not math.isclose(t-self.last_time, .02, abs_tol=1e-8, rel_tol=0):
            raise ValueError('Causal measurements need t=0 and exact 20 ms cadence.')
        if not np.array_equal(np.asarray(state['target_position_m']), self.target):
            raise ValueError('Destination changes inside an episode.')
        obj, tcp = np.asarray(state['object_position_m']), np.asarray(state['tcp_position_m'])
        force = np.asarray([state['pad_normal_forces_n'][s] for s in SIDES])
        contacts = np.asarray([state['grasp_contact_counts'][s] for s in SIDES])
        snapshot = np.clip(np.r_[(obj-tcp)/.05, (self.target-obj)/.2,
            np.asarray(state['object_linear_velocity_m_s'])/.2,
            np.asarray(state['gripper_joint_position_m'])/.035, np.log1p(force)/3.,
            contacts/2., state['table_object_contact_count']/2.], -10., 10.)
        self.feature_history.append((t, snapshot))
        while self.feature_history and self.feature_history[0][0] < t-.12-1e-8:
            self.feature_history.popleft()
        past = []
        for lag in (.04, .12):
            eligible = [row for row in self.feature_history if row[0] <= t-lag+1e-8]
            past.append(eligible[-1] if eligible else self.feature_history[0])
        feature = np.r_[self.initial/.5, self.target[:2]/.5, t/20., nominal_jaw/.035,
                        [min(.12, t-row[0])/.12 for row in past], snapshot, past[0][1], past[1][1],
                        progress/20., max(0., t-progress)/20.]
        if feature.shape != (58,) or not np.isfinite(feature).all():
            raise ValueError('Invalid reconstructed causal features.')
        good = bool(np.all(contacts > 0) and np.all(force >= self.config['minimum_pad_force_n']))
        local = local_offset(obj, tcp, state['tcp_quaternion_wxyz'])
        if not good:
            self.ready_history.clear()
        self.ready_history.append((t, local, good))
        dwell = self.config['secure_dwell_s']
        while len(self.ready_history) > 1 and self.ready_history[1][0] <= t-dwell+1e-8:
            self.ready_history.popleft()
        duration = t-self.ready_history[0][0]
        drift = max(float(np.linalg.norm(value-local)) for _, value, _ in self.ready_history)
        secure = bool(good and all(row[2] for row in self.ready_history)
                      and duration+1e-8 >= dwell and drift <= self.config['maximum_relative_drift_m'])
        in_window = gate*.02-1e-8 <= progress < release*.02-1e-8
        if in_window and secure:
            self.acquisition_released = True
        advance = bool(not in_window or secure or self.config['acquisition_only'] and self.acquisition_released)
        self.last_time = t
        return feature, dict(secure=secure, duration_s=duration, drift_m=drift,
            in_window=in_window, acquisition_released=self.acquisition_released, teacher_advance=advance,
            supported=state['table_object_contact_count'] > 0, local_offset_m=local)


def tree_probability(document, feature):
    index, seen = 0, set()
    nodes = document['nodes']
    for _ in range(128):
        if type(index) is not int or not 0 <= index < len(nodes) or index in seen:
            raise ValueError('Invalid or cyclic data-only decision tree.')
        seen.add(index)
        node = nodes[index]
        if node['feature'] == -1:
            probability = node.get('advance_probability', node.get('open_probability'))
            if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid leaf probability.')
            return probability
        position = node['feature']
        if type(position) is not int or not 0 <= position < len(feature):
            raise ValueError('Invalid tree feature index.')
        index = node['left'] if feature[position] <= node['threshold'] else node['right']
    raise ValueError('Decision tree exceeds its bounded depth.')


def _local_many(arrays, indices):
    quat = np.asarray(arrays['tcp_quaternion_wxyz'])[indices]
    offset = np.asarray(arrays['object_position_m'])[indices]-np.asarray(arrays['tcp_position_m'])[indices]
    if quat.shape != (len(indices), 4) or offset.shape != (len(indices), 3) or not np.isfinite(np.c_[quat, offset]).all():
        raise ValueError('Invalid dense pose rows.')
    if not np.allclose(np.sum(quat*quat, axis=1), 1., atol=1e-6, rtol=0):
        raise ValueError('Invalid dense tool quaternions.')
    w, x, y, z = quat.T
    rotation = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape((-1, 3, 3))
    return np.einsum('nji,nj->ni', rotation, offset)


def native_readiness_window(arrays, t, config, current_local, *, local_offsets=None):
    """Diagnostic only: check all already completed 1 ms rows in a sampled dwell."""
    start = t-config['secure_dwell_s']
    times = np.asarray(arrays.get('t_seconds', []), dtype=float)
    first = int(np.searchsorted(times, start+1e-8, side='right'))
    last = int(np.searchsorted(times, t+1e-8, side='right'))
    selected = np.arange(first, last)
    expected = round(config['secure_dwell_s']/.001)
    if (len(selected) != expected or expected == 0
            or not np.allclose(times[selected], start+np.arange(1, expected+1)*.001, atol=1e-8, rtol=0)):
        return dict(complete=False, rows=len(selected), expected_rows=expected, secure=False,
                    force_or_contact_insecure_rows=None, drift_insecure_rows=None,
                    supported_rows=None, airborne_rows=None)
    contacts = arrays['grasp_contact_counts'][selected]
    force = arrays['pad_normal_forces_n'][selected]
    good = np.all(contacts > 0, axis=1) & np.all(force >= config['minimum_pad_force_n'], axis=1)
    local = _local_many(arrays, selected) if local_offsets is None else local_offsets[selected]
    drifting = np.linalg.norm(local-current_local, axis=1) > config['maximum_relative_drift_m']
    supported = arrays['table_object_contact_count'][selected] > 0
    return dict(complete=True, rows=len(selected), expected_rows=expected,
                secure=bool(np.all(good) and not np.any(drifting)),
                force_or_contact_insecure_rows=int(np.count_nonzero(~good)),
                drift_insecure_rows=int(np.count_nonzero(drifting)),
                supported_rows=int(np.count_nonzero(supported)), airborne_rows=int(np.count_nonzero(~supported)))


def inspect_trace(episode, arrays, base, jaw, clock_document, config, jaw_mode):
    states, actions = episode['trajectory'], episode['actions']
    if len(states) < len(actions) or not states:
        raise ValueError('A retained action lacks its causal pre-issue state.')
    references = References(base, states[0]['object_position_m'], states[0]['target_position_m'])
    gate, release = references.window(config['gate_offset_s'])
    measurements = CausalMeasurements(states[0]['object_position_m'], states[0]['target_position_m'], config)
    variant = episode['policy_kind']
    counts = dict(actions=len(actions), feature_mismatches=0, progress_mismatches=0,
        decision_mismatches=0, command_mismatches=0, teacher_metadata_mismatches=0,
        clock_metadata_mismatches=0,
        hold_decisions=0, advance_decisions=0, premature_advances_before_readiness=0,
        holds_when_sampled_teacher_would_advance=0, sampled_secure_observations=0,
        sampled_secure_supported_observations=0, sampled_secure_airborne_observations=0,
        sampled_secure_native_window_missing=0, sampled_secure_native_insecure=0,
        maximum_joint_command_error_rad=0., maximum_jaw_command_error_m=0.,
        first_sampled_secure_s=None, first_acquisition_release_s=None)
    expected_progress, examples = 0., []
    dense_local = _local_many(arrays, np.arange(len(arrays.get('t_seconds', []))))
    for index, (state, action) in enumerate(zip(states, actions)):
        decision = action['clock_decision']
        t = index*.02
        if not math.isclose(state['t_seconds'], t, abs_tol=1e-8, rel_tol=0) or not math.isclose(action['t_seconds'], t, abs_tol=1e-8, rel_tol=0):
            raise ValueError('Action/state physical timestamps differ from the causal grid.')
        before = decision['progress_before_s']
        nominal_jaw = references.action(state, t)[1]
        feature, readiness = measurements.observe(state, nominal_jaw, before, gate, release)
        if variant == 'blind_clock':
            feature[8:56] = 0.
        failure = []
        expected_origin = {'fixed': 'fixed_unit_progress', 'teacher': 'authored_causal_grip_teacher',
                           'clock': 'learned_grip_clock', 'blind_clock': 'learned_blind_clock'}[variant]
        if (decision.get('decision_origin') != expected_origin or decision.get('jaw_mode') != jaw_mode
                or not math.isclose(decision.get('frozen_jaw_target_m', float('nan')),
                                    action['gripper_position_m'], abs_tol=TOL, rel_tol=0)):
            counts['clock_metadata_mismatches'] += 1
            failure.append('clock_metadata')
        if not np.allclose(decision['features'], feature, atol=TOL, rtol=0):
            counts['feature_mismatches'] += 1
            failure.append('causal_features')
        increment = decision['increment_s']
        progress_ok = (math.isclose(before, expected_progress, abs_tol=1e-8, rel_tol=0)
            and increment in (0., .02) and before <= t+1e-8
            and math.isclose(decision['physical_time_s'], t, abs_tol=1e-8, rel_tol=0)
            and math.isclose(decision['progress_after_s'], min(20., before+increment), abs_tol=1e-8, rel_tol=0))
        if not progress_ok:
            counts['progress_mismatches'] += 1
            failure.append('clock_progress')
        expected_progress = decision['progress_after_s']
        probability = None
        if variant == 'fixed':
            advance = True
        elif variant == 'teacher':
            advance = readiness['teacher_advance']
            meta = dict(secure_grip=readiness['secure'], grip_dwell_s=readiness['duration_s'],
                relative_drift_m=readiness['drift_m'], in_grasp_window=readiness['in_window'],
                acquisition_released=readiness['acquisition_released'], gate_progress_s=gate*.02,
                release_progress_s=release*.02)
            if any(not math.isclose(decision.get(key, float('nan')), value, abs_tol=TOL, rel_tol=0)
                   if type(value) is float else decision.get(key) is not value for key, value in meta.items()):
                counts['teacher_metadata_mismatches'] += 1
                failure.append('sampled_readiness')
        else:
            probability = tree_probability(clock_document, feature)
            advance = probability > .5
        if ((increment > 0.) != advance
                or probability is not None and not math.isclose(decision.get('advance_probability', float('nan')), probability, abs_tol=TOL, rel_tol=0)):
            counts['decision_mismatches'] += 1
            failure.append('decision')
        target, reference_jaw = references.action(state, before)
        if jaw_mode == 'physical_adapter':
            # The frozen jaw receives the first56 original physical-time rows.
            jaw_feature = feature[:56].copy()
            if variant == 'blind_clock':
                # Clock blindness must not mask the independent common jaw.
                snapshot = measurements.feature_history[-1][1]
                past = []
                for lag in (.04, .12):
                    eligible = [row for row in measurements.feature_history if row[0] <= t-lag+1e-8]
                    past.append(eligible[-1][1] if eligible else measurements.feature_history[0][1])
                jaw_feature[8:56] = np.r_[snapshot, *past]
            reference_jaw = .035 if tree_probability(jaw, jaw_feature) > .5 else 0.
        joint_error = float(np.max(np.abs(np.asarray(action['joint_targets_rad'])-target)))
        jaw_error = abs(action['gripper_position_m']-reference_jaw)
        counts['maximum_joint_command_error_rad'] = max(counts['maximum_joint_command_error_rad'], joint_error)
        counts['maximum_jaw_command_error_m'] = max(counts['maximum_jaw_command_error_m'], jaw_error)
        if joint_error > TOL or jaw_error > TOL or not math.isclose(decision['nominal_physical_jaw_m'], nominal_jaw, abs_tol=TOL, rel_tol=0):
            counts['command_mismatches'] += 1
            failure.append('frozen_reference_command')
        counts['advance_decisions' if increment > 0 else 'hold_decisions'] += 1
        counts['premature_advances_before_readiness'] += bool(increment > 0 and readiness['in_window'] and not readiness['teacher_advance'])
        counts['holds_when_sampled_teacher_would_advance'] += bool(increment == 0 and readiness['teacher_advance'])
        if readiness['secure']:
            counts['sampled_secure_observations'] += 1
            counts['sampled_secure_supported_observations' if readiness['supported'] else 'sampled_secure_airborne_observations'] += 1
            if counts['first_sampled_secure_s'] is None:
                counts['first_sampled_secure_s'] = t
            native = native_readiness_window(arrays, t, config, readiness['local_offset_m'], local_offsets=dense_local)
            counts['sampled_secure_native_window_missing'] += not native['complete']
            counts['sampled_secure_native_insecure'] += bool(native['complete'] and not native['secure'])
        if readiness['acquisition_released'] and counts['first_acquisition_release_s'] is None:
            counts['first_acquisition_release_s'] = t
        if failure and len(examples) < 25:
            examples.append(dict(action_index=index, physical_time_s=t, mismatches=failure))
    counts.update(final_progress_s=expected_progress, complete_20s_action_trace=len(actions) == 1000,
                  timeout_with_incomplete_path=bool(len(actions) == 1000 and expected_progress < 20.-1e-8),
                  mismatch_examples=examples, mismatch_example_limit=25)
    lift = (arrays['object_position_m'][:, 2]-episode.get('initial_object_position_m', states[0]['object_position_m'])[2])
    qualifying = ((lift >= episode.get('config', {}).get('min_lift_m', .06))
                  & np.all(arrays['grasp_contact_counts'] > 0, axis=1)
                  & (arrays['table_object_contact_count'] == 0))
    padded = np.r_[False, qualifying, False].astype(int)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    dwell_ticks = round(episode.get('config', {}).get('min_lift_dwell_s', .1)/.001)
    first = next((int(start)+dwell_ticks-1 for start, end in zip(starts, ends) if end-start >= dwell_ticks), None)
    counts['first_qualifying_lift_s'] = float(arrays['t_seconds'][first]) if first is not None else None
    return counts


def _verify_attempt(folder, case, variant, protocol_hash, expected_model_hash, collect_labels):
    expected = {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'score.json', 'post-grip.json'}
    if collect_labels:
        expected.add('samples.npz')
    if folder.is_symlink() or not folder.is_dir() or {p.name for p in folder.iterdir()} != expected|{'receipt.json'}:
        raise ValueError('Attempt does not have its exact immutable evidence inventory.')
    receipt, started = _read(folder/'receipt.json'), _read(folder/'started.json')
    binding = receipt['binding']
    if (binding != started or binding['case'] != case or binding['variant'] != variant
            or binding['collect_labels'] is not collect_labels or binding['protocol_sha256'] != protocol_hash
            or binding['model_raw_sha256'] != expected_model_hash
            or set(receipt['files']) != expected or receipt['case_id'] != case['id'] or receipt['variant'] != variant):
        raise ValueError('Attempt request identity differs from its scheduled case/model.')
    for name, expected_hash in receipt['files'].items():
        if _hash(folder/name) != expected_hash:
            raise ValueError('Attempt evidence hash differs: '+name)
    return binding


def _requests(root, protocol, split):
    if split in ('all', 'train'):
        for case in protocol['cases']['train']:
            yield 'train', case, 'teacher', root/'demonstrations'/case['id'], None, True
    if split in ('all', 'development'):
        for case in protocol['cases']['development']:
            for variant in ('fixed', 'teacher'):
                yield 'development', case, variant, root/'development'/variant/case['id'], None, False
            for name in ('full4', 'full6', 'blind4', 'blind6'):
                variant = 'blind_clock' if name.startswith('blind') else 'clock'
                yield 'development', case, variant, root/'candidates'/name/'development'/case['id'], root/'candidates'/name/'model.json', False
    if split in ('all', 'confirmation'):
        for case in protocol['cases']['confirmation']:
            for variant in VARIANTS:
                path = root/('blind.json' if variant == 'blind_clock' else 'selected.json') if variant in ('clock', 'blind_clock') else None
                yield 'confirmation', case, variant, root/'confirmation'/case['id']/variant, path, False


def audit(study, output, *, split='all', source_root=None):
    root, output = Path(study).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('Use a fresh immutable audit output outside the frozen study.')
    if split not in ('all', 'train', 'development', 'confirmation'):
        raise ValueError('Unknown audit partition.')
    protocol = _read(root/'protocol.json')
    protocol_hash = _hash(root/'protocol.json')
    prepared = _read(root/'prepared.json')
    if prepared['protocol_sha256'] != protocol_hash or protocol['source_sha256'] != prepared['source_sha256']:
        raise ValueError('Prepared protocol/source identity differs.')
    source_root = Path(source_root) if source_root else Path(__file__).resolve().parents[1]/'src'/'dvidia_training'
    sources = protocol['source_sha256']
    for name, expected in sources.items():
        if Path(name).name != name or _hash(root/'sources'/name) != expected or _hash(source_root/name) != expected:
            raise ValueError('Frozen source snapshot or installed numeric source differs: '+name)
    base, jaw = _read(root/'base-model.json'), _read(root/'jaw-model.json')
    for filename, key in (('base-model.json', 'base_model_raw_sha256'), ('jaw-model.json', 'jaw_model_raw_sha256')):
        if _hash(root/filename) != protocol[key] or prepared[key] != protocol[key]:
            raise ValueError('Declared frozen base/jaw bytes differ.')
    if protocol['jaw_mode'] == 'coordinated_reference':
        if jaw != dict(schema='dvidia.piper-reference-jaw/v1', base_sha256=_canonical_hash(base), mode='coordinated_reference', physical_robot_ready=False):
            raise ValueError('Coordinated jaw manifest differs from the frozen base.')
    elif protocol['jaw_mode'] != 'physical_adapter':
        raise ValueError('Unknown explicit jaw architecture.')
    frozen_hash = None
    if split in ('all', 'confirmation'):
        frozen = _read(root/'frozen.json')
        frozen_hash = _hash(root/'frozen.json')
        if (frozen['protocol_sha256'] != protocol_hash or frozen['lifecycle_source_sha256'] != sources
                or frozen['runtime_sha256'] != sources or frozen['jaw_mode'] != protocol['jaw_mode']
                or frozen['selected'] not in ('full4', 'full6') or frozen['blind'] not in ('blind4', 'blind6')
                or frozen['selected'][-1] != frozen['blind'][-1]
                or frozen['selected_sha256'] != _hash(root/'selected.json')
                or frozen['blind_sha256'] != _hash(root/'blind.json')
                or frozen['selected_sha256'] != _hash(root/'candidates'/frozen['selected']/'model.json')
                or frozen['blind_sha256'] != _hash(root/'candidates'/frozen['blind']/'model.json')):
            raise ValueError('Confirmation freeze identity differs.')
        for name, expected in frozen['files'].items():
            path = root/name
            if path.resolve().is_relative_to(root) is not True or _hash(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
                raise ValueError('Frozen pre-confirmation evidence changed: '+name)
        actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()
                  and path.relative_to(root).parts[0] != 'confirmation'
                  and path.relative_to(root).as_posix() not in ('.lifecycle.lock', 'frozen.json', 'report.json')}
        if actual != set(frozen['files']):
            raise ValueError('Pre-confirmation evidence has unbound extra or missing files.')
    from dvidia_training import piper_feedback_score as strict_scorer
    from dvidia_training import piper_grip_clock_env as post_grip_scorer
    if (_hash(strict_scorer.__file__) != sources['piper_feedback_score.py']
            or _hash(post_grip_scorer.__file__) != sources['piper_grip_clock_env.py']):
        raise ValueError('Imported numeric scorers differ from the frozen source.')
    rows = []
    for partition, case, variant, folder, model_path, labels in _requests(root, protocol, split):
        document = _read(model_path) if model_path else None
        if document is not None and (document.get('base_sha256') != _canonical_hash(base)
                or document.get('jaw_sha256') != _canonical_hash(jaw)
                or document.get('blind') is not (variant == 'blind_clock')):
            raise ValueError('Clock model dependency or blind/full identity differs.')
        binding = _verify_attempt(folder, case, variant, protocol_hash, _hash(model_path) if model_path else None, labels)
        if (binding['runtime_sha256'] != sources
                or binding['split'] != partition
                or binding['base_model_raw_sha256'] != protocol['base_model_raw_sha256']
                or binding['jaw_model_raw_sha256'] != protocol['jaw_model_raw_sha256']
                or binding['jaw_mode'] != protocol['jaw_mode']
                or binding['teacher_config'] != protocol['teacher_config']
                or binding['post_grip_parameters'] != protocol['post_grip_parameters']):
            raise ValueError('Attempt runtime differs from the source snapshot.')
        episode, arrays, metadata = _episode(folder/'episode.json.gz'), _npz(folder/'native.npz'), _read(folder/'native.json')
        if (episode['case_id'] != case['id'] or episode['seed'] != case['seed'] or episode['condition'] != case['condition']
                or episode['policy_kind'] != variant or episode['jaw_mode'] != protocol['jaw_mode']
                or episode['base_model_sha256'] != _canonical_hash(base) or episode['jaw_model_sha256'] != _canonical_hash(jaw)
                or episode['model_sha256'] != (_canonical_hash(document) if document else None)
                or not np.allclose(episode['config']['object_position_m'], [*case['start_xy'], .082], atol=1e-12, rtol=0)
                or not np.allclose(episode['config']['target_position_m'], [*case['goal_xy'], .08], atol=1e-12, rtol=0)
                or episode['implementation_sha256'] != binding['runtime_sha256']):
            raise ValueError('Episode identity differs from its receipt/dependencies.')
        numeric = strict_scorer.score_feedback_episode(episode, arrays, metadata, case['goal_xy'])
        post = post_grip_scorer.analyze_post_grip(arrays, metadata)
        saved = _read(folder/'score.json')
        numeric_matches = all(saved.get(key) == value for key, value in numeric.items())
        post_matches = post == _read(folder/'post-grip.json') == saved.get('post_grip_analysis')
        trace = inspect_trace(episode, arrays, base, jaw, document, protocol['teacher_config'], protocol['jaw_mode'])
        label_mismatches = 0
        if labels:
            samples = _npz(folder/'samples.npz')
            expected_features = np.asarray([a['clock_decision']['features'] for a in episode['actions']])
            blind = expected_features.copy()
            blind[:, 8:56] = 0.
            targets = np.asarray([a['clock_decision']['increment_s'] for a in episode['actions']])
            expected = dict(features=expected_features, blind_features=blind, targets=targets,
                weights=np.where(targets == 0., 20., 1.), groups=np.asarray([case['id']]*len(targets)))
            label_mismatches = int(set(samples) != set(expected) or any(not np.array_equal(samples.get(key), value) for key, value in expected.items()))
        failures = (sum(trace[key] for key in ('feature_mismatches', 'progress_mismatches', 'decision_mismatches',
                    'command_mismatches', 'teacher_metadata_mismatches', 'clock_metadata_mismatches'))+label_mismatches
                    +int(not numeric_matches)+int(not post_matches))
        candidate = model_path.parent.name if model_path is not None and model_path.parent.parent.name == 'candidates' else variant
        rows.append(dict(partition=partition, case_id=case['id'], variant=variant, candidate=candidate,
            relative_path=folder.relative_to(root).as_posix(), receipt_sha256=_hash(folder/'receipt.json'),
            numeric_score_matches=numeric_matches, intervention_matches=post_matches,
            training_label_mismatches=label_mismatches, audit_mismatches=failures,
            valid=numeric['valid'], strict_success=numeric['success'], terminal_only=numeric.get('terminal_only'),
            grip_summary=numeric.get('grip_summary'), grip_events=numeric.get('grip_events'),
            native_task=numeric.get('native_recomputed'), intervention=post, trace=trace))
    for name, expected in sources.items():
        if _hash(source_root/name) != expected:
            raise ValueError('Source changed during independent numeric audit.')
    summaries = {}
    diagnostic_keys = ('hold_decisions', 'advance_decisions', 'premature_advances_before_readiness',
        'holds_when_sampled_teacher_would_advance', 'sampled_secure_observations',
        'sampled_secure_supported_observations', 'sampled_secure_airborne_observations',
        'sampled_secure_native_window_missing', 'sampled_secure_native_insecure')
    for row in rows:
        key = row['partition']+'/'+row['candidate']
        summary = summaries.setdefault(key, dict(attempts=0, valid=0, strict_successes=0,
            audit_mismatches=0, attempts_with_premature_advances=0, attempts_with_native_readiness_insecurity=0,
            **{name: 0 for name in diagnostic_keys}))
        summary['attempts'] += 1
        summary['valid'] += row['valid'] is True
        summary['strict_successes'] += row['strict_success'] is True
        summary['audit_mismatches'] += row['audit_mismatches']
        summary['attempts_with_premature_advances'] += row['trace']['premature_advances_before_readiness'] > 0
        summary['attempts_with_native_readiness_insecurity'] += row['trace']['sampled_secure_native_insecure'] > 0
        for name in diagnostic_keys:
            summary[name] += row['trace'][name]
    result = dict(schema=SCHEMA, audit_implementation_sha256=_hash(__file__), protocol_sha256=protocol_hash,
        frozen_sha256=frozen_hash, source_sha256=sources, jaw_mode=protocol['jaw_mode'], split=split,
        audited_attempts=len(rows), audit_mismatches=sum(row['audit_mismatches'] for row in rows),
        interpretation='Reconstructed numeric evidence, not simulator authenticity, physics rerun or hardware qualification.',
        sampled_readiness='Six 20 ms observations spanning 100 ms; 1 ms interval checks are separately reported diagnostics.',
        premature_advance_scope='Advance where the separately reconstructed sampled acquisition teacher would still hold.',
        false_waiting_scope='Hold where the same sampled teacher would advance; not proof of an unnecessary physical wait.',
        summaries=summaries, rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.audit-writing-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # Exclusive, atomic final result; never replace.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=('all', 'train', 'development', 'confirmation'), default='all')
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args(argv)
    result = audit(args.study, args.output, split=args.split, source_root=args.source_root)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'source_sha256')}, indent=2))
    return int(result['audit_mismatches'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
