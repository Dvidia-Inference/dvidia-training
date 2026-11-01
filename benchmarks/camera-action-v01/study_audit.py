"""Independent saved-evidence audit across the camera/action study lifecycle.

Imports numeric scoring and the diagnostic auditor's byte readers only. No
environment, runner, controller, tracker or study-report function is imported.
The audit does not initialize physics, infer poses or execute learned actions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import random

import numpy as np

from independent_audit import (canonical, digest, read, identity, relative, require,
    inventory, _reference, _close, _transform, scoring)

SCHEMA = 'dvidia.camera-action-study-audit/v1'
STAGES = ('diagnostic', 'demonstrations', 'train-initial', 'development-initial',
          'corrections', 'train-corrected', 'development-corrected', 'confirmation', 'negative')
VISION = ('object_position_estimate_m', 'object_quaternion_estimate_wxyz',
          'object_position_uncertainty_m', 'object_orientation_uncertainty_rad')
ENCODERS = ('joint_position_rad', 'joint_velocity_rad_s', 'gripper_joint_position_m',
            'gripper_joint_velocity_m_s', 'tcp_position_estimate_m', 'tcp_quaternion_estimate_wxyz')


def _action(value, limits):
    require(type(value) is dict and set(value) == {'joint_targets_rad', 'gripper_position_m'}, 'Unknown or missing action fields.')
    q = np.asarray(value['joint_targets_rad'], dtype=float)
    bounds = np.asarray(limits['joint_limits_rad'], dtype=float)
    require(q.shape == (6,) and bounds.shape == (6, 2) and np.isfinite(q).all()
        and np.all(q >= bounds[:, 0]) and np.all(q <= bounds[:, 1]), 'Action joint targets violate native bounds.')
    jaw = value['gripper_position_m']; low, high = limits['gripper_position_limits_m']
    require(type(jaw) in (int, float) and math.isfinite(jaw) and low <= jaw <= high, 'Action gripper target violates native bounds.')


def command_check(step, query, mode, negative, rng, limits):
    command = step['command']
    require(command.get('operator_correction') is False, 'Synthetic control cannot claim operator correction.')
    for name in ('requested', 'issued'):
        _action(command[name], limits)
    expected_authority = ('teacher' if mode == 'teacher' else
        ('teacher' if rng.random() < .5 else 'student') if mode == 'mixed' else mode)
    require(command['authority'] == expected_authority, 'Command authority differs from registered controller or seeded mixture.')
    expected_issued = deepcopy(command['requested'])
    if negative == 'open_gripper':
        expected_issued['gripper_position_m'] = .035
    require(command['issued'] == expected_issued, 'Issued command differs from its explicitly registered intervention.')
    require(command['issue_reason'] == ('open_gripper_negative_control' if expected_issued != command['requested'] else 'unchanged'),
        'Command intervention reason is inconsistent.')
    if mode in ('teacher', 'mixed'):
        require(type(query) is dict and query.get('origin') == 'privileged_simulator_teacher', 'Missing or mislabeled teacher query.')
        _action(query['action'], limits)
        if expected_authority == 'teacher':
            require(query['action'] == command['requested'], 'Executed teacher request differs from its separately sealed query.')
    else:
        require(query is None, 'A student or fixed rollout contains teacher intervention.')


def _native(root, files, case, summary):
    required = {'native_episode.json', 'native_metadata.json', 'native.npz'}
    if not required <= set(files):
        require(not required.intersection(files) and summary.get('recorded_complete_intervals') == 0,
            'Partial native evidence cannot justify completed intervals.')
        require(summary['score'].get('valid') is False and summary['score'].get('success') is False,
            'No-native-evidence attempt claims a valid native score.')
        return None, None, None, dict(valid=False, success=False, reason='native_evidence_unavailable')
    episode = read(root / 'native_episode.json'); metadata = read(root / 'native_metadata.json')
    with np.load(root / 'native.npz', allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    require(set(arrays) == {name for name, width in scoring.FIELDS}, 'Unknown or missing native arrays.')
    count = metadata.get('rows')
    require(type(count) is int and 0 <= count <= 20000, 'Native trace exceeds the registered time budget.')
    for name, width in scoring.FIELDS:
        values = arrays[name]
        require(values.dtype.kind in 'iuf' and values.shape == ((count,) if width == 1 else (count, width))
            and np.isfinite(values).all(), 'Native array shape or finite values differ: ' + name)
    require(np.array_equal(arrays['native_tick'], np.arange(1, count + 1)), 'Native tick ordering differs.')
    require(np.allclose(arrays['t_seconds'], np.arange(1, count + 1) * .001, atol=1e-8, rtol=0), 'Native 1 ms cadence differs.')
    require(episode['config']['object_position_m'] == case['object_position_m']
        and episode['config']['target_position_m'] == case['goal_position_m'], 'Native scene differs from registered start or goal.')
    score = scoring.score_feedback_episode(episode, arrays, metadata, case['goal_position_m'][:2])
    require(score == summary['score'], 'Saved native score differs from independent numeric reconstruction.')
    ticks = int(np.count_nonzero(arrays['unexpected_contact_count'] > 0))
    require(summary.get('critical_contact_ticks') == ticks, 'Critical contact count differs from native ticks.')
    return episode, arrays, metadata, score


def _snapshot(snapshot, index, files, cameras, native_state=None):
    _close(snapshot['timestamp_seconds'], index * .02, 'Measured snapshot is not on the 50 Hz clock.')
    require(set(snapshot['signals']) == set(VISION) | set(ENCODERS), 'Raw sensor channels differ from the registered whitelist.')
    for name in VISION:
        signal = snapshot['signals'][name]
        require(signal['status'] == 'missing' and signal['value'] is None, 'Raw snapshot substitutes native object truth for perception.')
    for name in ENCODERS:
        signal = snapshot['signals'][name]
        require(signal['status'] == 'available', 'Registered native encoder channel is unavailable.')
        for stamp in ('captured_at_seconds', 'available_at_seconds'):
            _close(signal[stamp], snapshot['timestamp_seconds'], 'Encoder signal timestamp differs.')
        _close(signal['age_seconds'], 0., 'Encoder signal age differs.')
        if native_state and name in native_state:
            require(signal['value'] == native_state[name], 'Encoder signal differs from saved native measurement: ' + name)
    camera = snapshot['camera']; frame = camera['value']; expected_index = index // 2
    require(frame['index'] == expected_index and snapshot['vision_frame_index'] == expected_index, 'Camera source index differs from 25 Hz cadence.')
    _close(camera['captured_at_seconds'], expected_index * .04, 'Camera capture time differs.')
    _close(camera['available_at_seconds'], expected_index * .04, 'Synthetic camera availability differs from its captured frame.')
    _close(camera['age_seconds'], index * .02 - expected_index * .04, 'Camera age was reset or changed.')
    if expected_index in cameras:
        require(frame == cameras[expected_index], 'Reused camera frame changed its identity.')
    cameras[expected_index] = frame
    for channel in ('rgb', 'depth'):
        reference = frame[channel]
        require(files.get(reference['path']) == {key: reference[key] for key in ('bytes', 'sha256')}, 'Camera frame is not byte-bound to the recording.')


def _rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    require(q.shape == (4,) and np.isfinite(q).all() and abs(float(q @ q)-1.) < 1e-7,
        'Encoder pose quaternion is not a finite unit quaternion.')
    w, x, y, z = q
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _encoder_pose(snapshot, native_state, world_from_base):
    signals = snapshot['signals']
    require(signals['gripper_joint_velocity_m_s']['value'] == native_state['qvel'][6:8],
        'Gripper encoder velocity differs from saved native state.')
    # Frozen capture names the link6 body origin "tcp_position_estimate_m".
    # The task scorer uses its distal authored site at link6 [0,0,.12003].
    # Reconstruct this explicit site-to-body offset; do not execute kinematics.
    world_rotation = _rotation(native_state['tcp_quaternion_wxyz'])
    world_link6 = np.asarray(native_state['tcp_position_m'])-world_rotation @ np.asarray([0., 0., .12003])
    expected = world_from_base[:3, :3].T @ (world_link6-world_from_base[:3, 3])
    require(np.allclose(signals['tcp_position_estimate_m']['value'], expected, atol=1e-8, rtol=0),
        'Encoder tool position differs from saved native pose and base transform.')
    expected_rotation = world_from_base[:3, :3].T @ world_rotation
    require(np.allclose(_rotation(signals['tcp_quaternion_estimate_wxyz']['value']), expected_rotation, atol=1e-8, rtol=0),
        'Encoder tool orientation differs from saved native pose and base transform.')


def _perception(document, snapshot, recording, negative, index):
    require(document['index'] == index and document['snapshot_sha256'] == digest(snapshot), 'Perception is bound to a different motor snapshot.')
    _close(document['decision_at_seconds'], index * .02, 'Perception decision time differs.')
    require(document['frame'] == snapshot['camera']['value']
        and document['calibration_sha256'] == recording['binding']['calibration_sha256']
        and document['tracker_sha256'] == recording['tracker_sha256'], 'Perception frame/calibration/tracker identity differs.')
    expected_fault = negative if snapshot['camera']['value']['index'] == 0 else None
    require(document.get('negative_control') == expected_fault, 'Perception intervention differs from its registered initial fault.')
    calibration = deepcopy(recording['calibration'])
    if expected_fault == 'altered_calibration':
        calibration['base_from_camera'][0] *= 1.1
    require(document['input_calibration_sha256'] == digest(calibration), 'Actual estimator input calibration differs from the declared intervention.')
    estimate = document['estimate']; camera = snapshot['camera']
    require(estimate['frame_index'] == camera['value']['index'], 'Estimate source frame differs from the camera.')
    captured = -.2 if expected_fault == 'initial_stale' else camera['captured_at_seconds']
    available = -.2 if expected_fault == 'initial_stale' else camera['available_at_seconds']
    _close(estimate['captured_at_seconds'], captured, 'Estimate capture time differs from its declared source.')
    _close(estimate['available_at_seconds'], available, 'Estimate availability differs from its declared source.')
    _close(estimate['decision_at_seconds'], index * .02, 'Estimate decision time differs.')
    _close(estimate['age_seconds'], max(0., index * .02 - captured), 'Estimate age differs from its source.')
    if estimate['status'] == 'available':
        require(available <= index * .02 + 1e-8 and index * .02 - captured <= .1 + 1e-8, 'Available pose is stale or from the future.')
        for field in VISION:
            signal = estimate['signals'][field]
            expected = estimate[field]
            if field in ('object_position_uncertainty_m', 'object_orientation_uncertainty_rad'):
                expected = [expected]
            require(signal['status'] == 'available' and signal['value'] == expected, 'Available pose and sensor values disagree.')
            for stamp in ('captured_at_seconds', 'available_at_seconds', 'age_seconds'):
                _close(signal[stamp], estimate[stamp], 'Estimate signal age or source differs.')
    else:
        for field in VISION:
            require(estimate[field] is None and estimate['signals'][field]['status'] == 'missing'
                and estimate['signals'][field]['value'] is None, 'Unavailable perception forwards a live object value.')
    return estimate


def audit_episode(root, attempt, binding, *, tracker_sha256=None):
    root = Path(root); files = inventory(root); summary = read(root / 'rollout.json')
    mode = 'student' if attempt['controller'] == 'blind' else attempt['controller']
    case = {**attempt['case'], 'policy_artifact_sha256': binding}
    negative = attempt['negative_control']
    require(summary['case'] == case and summary['mode'] == mode and summary['negative_control'] == negative,
        'Episode controller, scene or intervention differs from registration.')
    if tracker_sha256 is not None:
        require(summary['tracker_sha256'] == tracker_sha256, 'Episode used a different tracker implementation.')
    started = read(root / 'started.json')
    require(started['case'] == case and started['mode'] == mode and started['negative_control'] == negative
        and started['config']['duration_s'] == 20. and started['config']['seed'] == case['seed']
        and started['config']['width'] == 320 and started['config']['height'] == 240
        and started['config']['camera_every_controls'] == 2, 'Started rollout differs from its registered scene, controller or capture budget.')
    episode, arrays, metadata, score = _native(root, files, case, summary)
    result = dict(id=attempt['id'], case=attempt['case'], controller=attempt['controller'], negative_control=negative,
        status=summary['status'], complete=False, strict_success=False, score=score,
        initial_anchor_available=False, initial_anchor_binding_valid=False, initial_anchor_error_m=None,
        recorded_complete_intervals=0, critical_contact_ticks=summary.get('critical_contact_ticks'),
        provenance_ok=mode in ('student', 'fixed') and summary.get('teacher_queries') == 0,
        files_verified=len(files), native_ticks=metadata['rows'] if metadata else 0,
        native_score_recomputed=episode is not None, available_position_error_p95_m=None,
        integrity_sha256=identity(root / 'integrity.json')['sha256'], receipt_sha256=identity(root / 'rollout.json')['sha256'])
    require(summary['no_privileged_features_or_teacher_intervention'] is result['provenance_ok'], 'Reported no-teacher provenance differs from query evidence.')
    if 'recording.json' not in files:
        require(summary['status'] == 'refused' and summary['recorded_complete_intervals'] == 0
            and summary['strict_success'] is False, 'Setup failure claims completed task evidence.')
        return result
    recording = read(root / 'recording.json'); steps = recording['steps']; initial = recording['initial']
    require(recording['tracker_sha256'] == summary['tracker_sha256'], 'Recording and receipt tracker identities differ.')
    require(recording['schema'] == 'dvidia.camera-action-rollout/v2' and recording['control_mode'] == mode,
        'Recording schema or authority differs.')
    require(recording['provenance']['case_id'] == case['id'] and recording['provenance']['source_group'] == case['group_id']
        and recording['goal_position_m'] == case['goal_position_m'], 'Recording group, goal or case differs.')
    require(len(steps) == summary['recorded_complete_intervals'] and len(steps) <= 1000, 'Completed interval count differs.')
    if initial is None:
        require(not steps and summary['status'] == 'refused' and not summary['initial_anchor_available'], 'Missing initial snapshot cannot support motion or a visual anchor.')
        return result
    context = read(root / 'context.json'); transform = _transform(context['metadata'])
    require(context['case'] == case and context['initial'] == initial, 'Initial context differs from the sealed recording.')
    require(digest(recording['calibration']) == recording['binding']['calibration_sha256'], 'Camera calibration binding differs.')
    require(digest(recording['limits']) == recording['binding']['limits_sha256'], 'Recorded native action limits changed.')
    cameras = {}; snapshots = [initial] + [step['after'] for step in steps]
    for index, snapshot in enumerate(snapshots):
        state = episode['trajectory'][index] if episode and len(episode['trajectory']) > index else None
        _snapshot(snapshot, index, files, cameras, state)
        if state is not None:
            _encoder_pose(snapshot, state, transform)
    require(summary['camera_pairs'] == len(cameras), 'Unique camera pair count differs.')
    prior = initial; rng = random.Random(case['seed'])
    for index, step in enumerate(steps):
        require(step['index'] == index and step['before'] == prior, 'Transition sequence is discontinuous.')
        _close(step['command']['issued_at_seconds'], index * .02, 'Command issue clock differs.')
        query = _reference(root, step['teacher_query'], files) if step['teacher_query'] is not None else None
        if query:
            require(query['index'] == index and query['snapshot_sha256'] == digest(step['before']), 'Teacher query uses a different input snapshot.')
            _close(query['computed_at_seconds'], index * .02, 'Teacher query timing differs.')
        command_check(step, query, mode, negative, rng, recording['limits'])
        perception = _reference(root, step['perception'], files)
        _perception(perception, step['before'], recording, negative, index)
        require(episode is not None and arrays is not None, 'Complete interval has no independently reconstructable native trace.')
        native_action = episode['actions'][index]
        require({key: native_action[key] for key in ('joint_targets_rad', 'gripper_position_m')} == step['command']['issued'],
            'Command label differs from actual issued native evidence.')
        tick = (index + 1) * 20 - 1
        require(tick < metadata['rows'], 'Complete interval lacks its native endpoint.')
        expected_controls = [*arrays['arm_actuator_controls_rad'][tick].tolist(), float(arrays['applied_jaw_target_m'][tick])]
        require(step['applied']['native_position_controls'] == expected_controls, 'Actual native position control differs.')
        require(step['applied']['ramped_targets'] == dict(joint_targets_rad=expected_controls[:6],
            gripper_position_m=float(arrays['normal_rate_limited_jaw_target_m'][tick])), 'Previous applied references differ from native controls.')
        prior = step['after']
    perception_files = sorted(name for name in files if name.startswith('perception/') and name.endswith('.json'))
    teacher_files = sorted(name for name in files if name.startswith('teacher/') and name.endswith('.json'))
    require(perception_files == [f'perception/{i:04}.json' for i in range(len(perception_files))]
        and len(steps) <= len(perception_files) <= len(steps) + 1, 'Perception decision prefix is discontinuous.')
    require(teacher_files == [f'teacher/{i:04}.json' for i in range(len(teacher_files))], 'Teacher query prefix is discontinuous.')
    if mode in ('teacher', 'mixed'):
        require(len(steps) <= len(teacher_files) <= len(steps) + 1, 'Teacher queries are missing or repeated.')
    else:
        require(not teacher_files, 'Student or fixed controller contains teacher-query evidence.')
    require(summary['teacher_queries'] == len(teacher_files), 'Teacher query counter differs from sealed evidence.')
    for index, filename in enumerate(teacher_files):
        query = read(root / filename)
        require(query['index'] == index and query['snapshot_sha256'] == digest(snapshots[index])
            and query['origin'] == 'privileged_simulator_teacher', 'Teacher query prefix has an invalid origin or source snapshot.')
        _close(query['computed_at_seconds'], index * .02, 'Teacher query prefix clock differs.')
        _action(query['action'], recording['limits'])
    require(summary['student_calls'] == 0 if mode == 'teacher' else len(steps) <= summary['student_calls'] <= len(steps) + 1,
        'Student inference count differs from the completed decision prefix.')
    require(summary['no_privileged_features_or_teacher_intervention'] is result['provenance_ok'], 'Reported no-teacher provenance differs from query evidence.')
    estimates, visible_errors = [], []
    initial_world = np.asarray(episode['trajectory'][0]['object_position_m']) if episode and episode['trajectory'] else None
    for index, filename in enumerate(perception_files):
        document = read(root / filename)
        estimate = _perception(document, snapshots[index], recording, negative, index)
        estimates.append(estimate)
        if estimate['status'] == 'available' and initial_world is not None:
            tick = round(estimate['captured_at_seconds'] / .001)
            truth_world = initial_world if tick == 0 else arrays['object_position_m'][tick - 1]
            truth_base = transform[:3, :3].T @ (truth_world - transform[:3, 3])
            visible_errors.append(float(np.linalg.norm(np.asarray(estimate['object_position_estimate_m']) - truth_base)))
    anchor_available = bool(estimates and estimates[0]['status'] == 'available')
    anchor_bound = anchor_available and negative not in ('initial_stale', 'altered_calibration')
    require(summary['initial_anchor_available'] is anchor_available and summary['initial_anchor_binding_valid'] is anchor_bound,
        'Initial visual anchor availability or binding differs.')
    anchor_error = visible_errors[0] if anchor_available and visible_errors else None
    if anchor_error is None:
        require(summary['initial_anchor_error_m'] is None, 'Unavailable anchor claims native diagnostic accuracy.')
    else:
        _close(summary['initial_anchor_error_m'], anchor_error, 'Initial anchor error differs from native reconstruction.')
    missing = [estimate['status'] != 'available' for estimate in estimates]
    longest = streak = 0
    for absent in missing:
        streak = streak + 1 if absent else 0; longest = max(longest, streak)
    perception = dict(decisions=len(estimates), available=sum(not value for value in missing),
        missing=sum(missing), longest_missing_seconds=longest * .02)
    require(summary['perception'] == perception, 'Perception coverage or longest missing interval differs.')
    complete = summary['status'] == 'complete' and len(steps) == 1000
    strict = complete and score.get('success') is True
    require(summary['strict_success'] is strict, 'Reported strict success differs from complete native scoring.')
    result.update(complete=complete, strict_success=strict, initial_anchor_available=anchor_available,
        initial_anchor_binding_valid=anchor_bound, initial_anchor_error_m=anchor_error,
        recorded_complete_intervals=len(steps), perception=perception,
        available_position_error_p95_m=float(np.quantile(visible_errors, .95)) if visible_errors else None)
    return result


def _stage(root, name, protocol_hash, inputs):
    path = root / 'stages' / (name + '.json')
    if not path.exists():
        return None
    seal_path = path.with_name(name + '.integrity.json')
    record = identity(path)
    require(read(seal_path) == dict(sha256=record['sha256'], protocol_sha256=protocol_hash), 'Stage receipt seal differs: ' + name)
    receipt = read(path)
    require(receipt['stage'] == name and receipt['protocol_sha256'] == protocol_hash, 'Stage receipt names a different stage or protocol.')
    started_path = path.with_name(name + '.started.json')
    started = read(started_path)
    require(started['stage'] == name and started['protocol_sha256'] == protocol_hash
        and started.get('model_bindings') == receipt.get('model_bindings'), 'Started stage and completed receipt bindings differ.')
    inputs[str(started_path.relative_to(root))] = identity(started_path)
    inputs[str(path.relative_to(root))] = record
    inputs[str(seal_path.relative_to(root))] = identity(seal_path)
    return receipt


def validate_roster(protocol):
    groups, points = {}, {}
    for split, expected in (('train', 64), ('development', 16), ('confirmation', 64)):
        cases = protocol['cases'][split]
        require(len(cases) == expected and len({case['id'] for case in cases}) == expected, 'Case denominator or identity differs.')
        by_group = {}
        for case in cases:
            require(case['split'] == split and case['goal_id'] in (0, 1), 'Case split or goal differs.')
            require(groups.setdefault(case['group_id'], split) == split, 'Starting group leaks between splits.')
            point = tuple(case['object_position_m'])
            require(len(point) == 3 and protocol['start_domain']['x_m'][0] <= point[0] <= protocol['start_domain']['x_m'][1]
                and protocol['start_domain']['y_m'][0] <= point[1] <= protocol['start_domain']['y_m'][1]
                and point[2] == .082 and case['goal_position_m'] == protocol['goals'][case['goal_id']], 'Case position or destination differs from the registered domain.')
            require(list(point[:2]) not in protocol['historical_start_exclusion'], 'Fresh start duplicates an explicitly excluded historical start.')
            require(points.setdefault(point, split) == split, 'Exact starting position leaks between splits.')
            by_group.setdefault(case['group_id'], []).append(case)
        require(len(by_group) == expected // 2 and all({c['goal_id'] for c in row} == {0, 1}
            and len({tuple(c['object_position_m']) for c in row}) == 1 for row in by_group.values()),
            'Both goals must share one starting-position group.')
    require(len(points) == 72, 'Expected 72 unique starting positions.')
    expected_counts = dict(diagnostic=4, demonstrations=80, corrections=32,
        **{'development-initial': 32, 'development-corrected': 32}, confirmation=256, negative=16)
    require({name: len(rows) for name, rows in protocol['rosters'].items()} == expected_counts, 'Registered attempt budgets differ.')
    for name, roster in protocol['rosters'].items():
        require(len({r['id'] for r in roster}) == len(roster), 'Duplicate attempt identity.')
        for attempt in roster:
            split = attempt['case']['split']
            require(attempt['case'] in protocol['cases'][split], 'Attempt contains an unregistered scene.')
            expected_split = ('train' if name == 'corrections' else 'confirmation' if name in ('confirmation', 'negative') else
                'development' if name == 'diagnostic' or name.startswith('development-') else None)
            require(expected_split is None or split == expected_split, 'Attempt crossed its registered split.')
        expected_controllers = ({'teacher': 4} if name == 'diagnostic' else {'teacher': 80} if name == 'demonstrations'
            else {'mixed': 32} if name == 'corrections' else {'student': 16, 'blind': 16} if name.startswith('development-')
            else {'student': 64, 'blind': 64, 'fixed': 64, 'teacher': 64} if name == 'confirmation' else {'student': 16})
        require({c: sum(a['controller'] == c for a in roster) for c in {a['controller'] for a in roster}} == expected_controllers,
            'Controller budgets differ from the approved experiment.')
        require(all(a['negative_control'] is None for a in roster) if name != 'negative' else
            {c: sum(a['negative_control'] == c for a in roster) for c in {a['negative_control'] for a in roster}}
                == dict(initial_missing=4, initial_stale=4, altered_calibration=4, open_gripper=4), 'Negative fault budgets differ.')
        pairs = [(a['case']['id'], a['controller'], a['negative_control']) for a in roster]
        require(len(set(pairs)) == len(pairs), 'Same scene/controller/fault appears more than once in a stage.')
        if name == 'corrections':
            require(len({a['case']['group_id'] for a in roster}) == 32
                and all(a['case']['goal_id'] == i % 2 for i, a in enumerate(roster)), 'Corrections must cover each training group once with alternating destinations.')
        if name == 'negative':
            require(len({a['case']['id'] for a in roster}) == 4
                and all(sum(a['case']['id'] == case_id for a in roster) == 4 for case_id, _, _ in pairs), 'Every negative scene must receive all four faults.')


def data_quality(rows, protocol):
    successful = [r for r in rows if r['case']['split'] == 'train' and r['strict_success']
        and r['initial_anchor_available'] and r['initial_anchor_binding_valid']]
    expected = {case['group_id'] for case in protocol['cases']['train']}
    covered = {row['case']['group_id'] for row in successful}
    counts = {str(goal): sum(r['case']['goal_id'] == goal for r in successful) for goal in (0, 1)}
    minimum = protocol['data_quality']['minimum_successful_episodes_per_goal']
    return dict(passed=covered == expected and all(n >= minimum for n in counts.values()),
        covered_training_groups=len(covered), required_training_groups=len(expected),
        missing_training_groups=sorted(expected - covered), successful_episodes_per_goal=counts,
        minimum_successful_episodes_per_goal=minimum,
        successful_with_available_bound_initial_camera=len(successful))


def admitted(row):
    return row.get('initial_anchor_available') is True and row.get('initial_anchor_binding_valid') is True and row.get('recorded_complete_intervals', 0) > 0


def bootstrap(rows, comparator, acceptance, *, admitted_only=False):
    groups = {}
    for row in rows:
        if row['controller'] in ('student', comparator):
            key = (row['controller'], row['case']['goal_id'])
            group = groups.setdefault(row['case']['group_id'], {})
            require(key not in group, 'Duplicate paired outcome.')
            group[key] = row
    require(len(groups) == 32 and all(set(group) == {(c, g) for c in ('student', comparator) for g in (0, 1)} for group in groups.values()),
        'Paired comparison lacks complete starting groups.')
    retained = {key: group for key, group in groups.items() if not admitted_only or all(admitted(row) for row in group.values())}
    differences = np.asarray([sum(int(group[('student', goal)]['strict_success']) - int(group[(comparator, goal)]['strict_success'])
        for goal in (0, 1)) / 2 for key, group in sorted(retained.items())])
    n = len(differences); confidence = acceptance['confidence']; alpha = (1 - confidence) / 2
    if n:
        rng = np.random.default_rng(acceptance['bootstrap_seed'])
        sample = differences[rng.integers(0, n, size=(acceptance['bootstrap_resamples'], n))].mean(axis=1)
        low, high = np.quantile(sample, [alpha, 1 - alpha]).tolist()
    else:
        low = high = None
    return dict(comparison='student-minus-' + comparator, difference=float(differences.mean()) if n else None,
        confidence=confidence, lower=low, upper=high, resamples=acceptance['bootstrap_resamples'],
        seed=acceptance['bootstrap_seed'], groups=n, excluded_groups=32 - n, admitted_only=admitted_only,
        pairing='both goals within exact starting-position group', positive=low is not None and low > 0.)


def negative_correct(row):
    if row['negative_control'] == 'open_gripper':
        return (row['complete'] is True and row['strict_success'] is False and row['score'].get('valid') is True
            and row['score'].get('strict_v01', {}).get('trajectory_complete') is True)
    return row['status'] == 'refused' and row['recorded_complete_intervals'] == 0 and row['strict_success'] is False


def final_gates(rows, negatives, protocol, *, stages_complete):
    acceptance = protocol['acceptance']; controllers = {}
    for controller in ('student', 'blind', 'fixed', 'teacher'):
        selected = [row for row in rows if row['controller'] == controller]
        contact = [row['critical_contact_ticks'] for row in selected]
        controllers[controller] = dict(scheduled=len(selected), strict_successes=sum(r['strict_success'] for r in selected),
            complete=sum(r['complete'] for r in selected), admitted_cases=sum(admitted(r) for r in selected),
            initial_refusals=sum(r['status'] in ('refused', 'interrupted') and r['recorded_complete_intervals'] == 0 for r in selected),
            critical_contact_episodes=sum(n > 0 for n in contact if n is not None),
            critical_contact_ticks=sum(n for n in contact if n is not None), contact_counts_complete=all(n is not None for n in contact))
    complete = stages_complete and len(rows) == 256 and len(negatives) == 16
    comparisons = [bootstrap(rows, other, acceptance) for other in ('fixed', 'blind')] if complete else []
    common = [bootstrap(rows, other, acceptance, admitted_only=True) for other in ('fixed', 'blind')] if complete else []
    student, fixed = controllers['student'], controllers['fixed']
    gates = dict(all_registered_attempts_accounted=complete,
        at_least_58_of_64=student['scheduled'] == acceptance['student_attempts'] and student['strict_successes'] >= acceptance['student_successes_minimum'],
        positive_paired_comparisons=len(comparisons) == 2 and all(c['positive'] for c in comparisons),
        positive_common_admitted_comparisons=len(common) == 2 and all(c['groups'] >= acceptance['minimum_common_admitted_groups'] and c['positive'] for c in common),
        no_critical_contact_increase=student['contact_counts_complete'] and fixed['contact_counts_complete']
            and student['critical_contact_episodes'] <= fixed['critical_contact_episodes'],
        all_negative_controls_correct=len(negatives) == 16 and all(negative_correct(row) for row in negatives),
        no_privileged_features_or_teacher_intervention=all(row['provenance_ok'] is True for row in rows if row['controller'] in ('student', 'blind', 'fixed')))
    return dict(controllers=controllers, comparisons=comparisons, admitted_comparisons=common, gates=gates,
        qualified=all(gates.values()), status='complete' if complete else 'incomplete')


def _model_hash(root, stage, controller, stages, inputs):
    require(stage in ('initial', 'corrected') and stages.get('train-' + stage) is not None, 'Model has no sealed fitting stage.')
    variant = 'blind' if controller == 'blind' else 'full'
    path = f'models/{stage}/{variant}/model.json'
    record = identity(root / path); inputs[path] = record
    require(stages['train-' + stage]['models'][variant]['model_sha256'] == record['sha256'], 'Model differs from its sealed training receipt.')
    return record['sha256']


def _eligible(rows, *, queried=False):
    return [row for row in rows if row.get('complete') is True and row.get('initial_anchor_available') is True
        and row.get('initial_anchor_binding_valid') is True and (queried or row.get('strict_success') is True)]


def _export_source(root, row, label):
    directory = relative(root, row['path']); recording = read(directory / 'recording.json'); steps = recording['steps']
    require(len(steps) == row['recorded_complete_intervals'], 'Training source sample count differs from completed intervals.')
    require(recording['provenance']['source_group'] == row['case']['group_id']
        and recording['provenance']['case_id'] == row['case']['id'], 'Training source group differs from its registered scene.')
    require(recording['control_mode'] == ('teacher' if label == 'issued' else 'mixed'), 'Training label origin differs from actual control mode.')
    return dict(source_group=row['case']['group_id'], case_id=row['case']['id'],
        recording_sha256=identity(directory / 'recording.json')['sha256'],
        integrity_sha256=identity(directory / 'integrity.json')['sha256'],
        calibration_sha256=recording['binding']['calibration_sha256'], tracker_sha256=recording['tracker_sha256'],
        label_source=label, sample_count=len(steps),
        perception_references_sha256=digest([s['perception']['sha256'] for s in steps]),
        teacher_query_references_sha256=digest([s['teacher_query']['sha256'] for s in steps] if label == 'teacher_query' else []),
        actual_authority_sha256=digest([s['command']['authority'] for s in steps]))


def _training(root, stages, protocol, inputs):
    training_groups = {case['group_id'] for case in protocol['cases']['train']}
    development_groups = {case['group_id'] for case in protocol['cases']['development']}
    source_cache = {}
    for stage in ('initial', 'corrected'):
        receipt = stages.get('train-' + stage)
        if receipt is None:
            continue
        require((stages.get('demonstrations') or {}).get('data_quality', {}).get('passed') is True,
            'Training proceeded without the demonstration quality gate.')
        demos = stages['demonstrations']['rows']
        selected = _eligible([row for row in demos if row['case']['split'] == 'train'])
        development = _eligible([row for row in demos if row['case']['split'] == 'development'])
        tagged = [(row, 'issued') for row in selected]
        if stage == 'corrected':
            require(stages.get('corrections') is not None, 'Corrected fitting has no correction collection.')
            corrections = _eligible(stages['corrections']['rows'], queried=True)
            selected = selected + corrections
            tagged += [(row, 'teacher_query') for row in corrections]
        require(bool(selected) and bool(development), 'Fitting lacks eligible training or development recordings.')
        expected_sources = []
        for row, label in tagged:
            key = (row['path'], label)
            if key not in source_cache: source_cache[key] = _export_source(root, row, label)
            expected_sources.append(source_cache[key])
        require(len({s['recording_sha256'] for s in expected_sources}) == len(expected_sources), 'A training recording is duplicated.')
        source_identities = []
        for controller, variant in (('student', 'full'), ('blind', 'blind')):
            _model_hash(root, stage, controller, stages, inputs)
            model = read(root / f'models/{stage}/{variant}/model.json'); training = model['training']
            receipt_entry = receipt['models'][variant]
            expected_path = f'models/{stage}/{variant}/model.json'
            require(receipt_entry['path'] == expected_path and receipt_entry['model_bytes'] == inputs[expected_path]['bytes'], 'Model receipt path or byte count differs.')
            require(receipt_entry['training_episodes'] == [r['id'] for r in selected]
                and receipt_entry['development_episodes'] == [r['id'] for r in development], 'Model fitting or validation episode roster differs from all eligible recordings.')
            for name in ('receipt.json', 'started.json'):
                path = f'models/{stage}/{variant}/{name}'; inputs[path] = identity(root / path)
            require(read(root / f'models/{stage}/{variant}/receipt.json') == receipt_entry, 'Individual model receipt differs from sealed training stage.')
            require(read(root / f'models/{stage}/{variant}/started.json') == dict(stage=stage, variant=variant,
                protocol_sha256=identity(root / 'protocol.json')['sha256']), 'Model fitting start names a different stage or protocol.')
            require(receipt_entry['training'] == training and receipt_entry['dataset_provenance'] == training['provenance'], 'Model metadata differs from its sealed fitting receipt.')
            require(model['schema'] == 'dvidia.camera-action-mlp/v1' and model['blind'] is (variant == 'blind')
                and model['observation_origin'] == 'rendered_rgbd_and_simulated_encoders'
                and model['feature_contract'] == 'camera-encoder-absolute-causal160/v1'
                and model['action_representation'] == 'absolute_native_targets'
                and model['hidden_units'] == protocol['fitting']['hidden_units'], 'Learned model observation, architecture or action contract differs.')
            require(training['stage'] == stage and training['model_source_sha256'] == protocol['source_sha256']['camera_action_model.py'], 'Training source/stage differs.')
            for key in ('epochs', 'batch_size', 'learning_rate', 'seed'):
                require(training[key] == protocol['fitting'][key], 'Training hyperparameter differs from registration.')
            require(set(training['train_groups']) <= training_groups
                and set(training['validation']['groups']) <= development_groups, 'Model fitting or validation crosses its split.')
            require(training['train_groups'] == list(dict.fromkeys(r['case']['group_id'] for r in selected))
                and training['validation']['groups'] == list(dict.fromkeys(r['case']['group_id'] for r in development)), 'Model training or validation group roster differs.')
            require(training['sample_count'] == sum(r['recorded_complete_intervals'] for r in selected)
                and training['validation']['sample_count'] == sum(r['recorded_complete_intervals'] for r in development), 'Model training or validation sample count differs.')
            provenance = training['provenance']
            require(training['provenance_sha256'] == digest(provenance)
                and provenance['source_identity_sha256'] == digest(provenance['episodes']), 'Training provenance identity differs.')
            require(provenance['blind'] is (variant == 'blind')
                and provenance['export_source_sha256'] == protocol['source_sha256']['camera_action_export.py']
                and provenance['feature_contract_sha256'] == model['feature_contract_sha256'], 'Training export, feature contract or feedback masking identity differs.')
            require(provenance['episodes'] == expected_sources, 'Training provenance omits, duplicates or changes eligible source recordings or supervision labels.')
            source_identities.append(provenance['source_identity_sha256'])
        require(source_identities[0] == source_identities[1], 'Full and blind models used different source demonstrations.')


def audit(directory):
    root = Path(directory); errors = []; results = []; inputs = {}; stages = {}; stage_results = {}; qualification = None
    protocol_hash = None
    try:
        require(not (root / 'worker.lock').exists(), 'Refusing to audit a running or interrupted study worker.')
        protocol = read(root / 'protocol.json'); registration = read(root / 'registration.json')
        protocol_hash = identity(root / 'protocol.json')['sha256']
        require(registration['protocol_sha256'] == protocol_hash and registration['source_sha256'] == protocol['source_sha256'], 'Registration identity differs.')
        require(identity(root / 'fixed-model.json')['sha256'] == registration['fixed_model_file_sha256'], 'Pinned comparator identity differs.')
        for name in ('protocol.json', 'registration.json', 'fixed-model.json'):
            inputs[name] = identity(root / name)
        for name, expected in protocol['source_sha256'].items():
            require(Path(name).name == name and identity(Path(scoring.__file__).with_name(name))['sha256'] == expected, 'Registered source bytes changed: ' + name)
        validate_roster(protocol)
        stages = {name: _stage(root, name, protocol_hash, inputs) for name in STAGES}
        gap = blocked = False
        for name in STAGES:
            receipt = stages[name]
            if receipt is None:
                gap = True
                continue
            require(not gap and not blocked, 'Study continued after a missing or blocked prerequisite stage.')
            blocked = receipt['status'] == 'blocked'
            start = read(root / 'stages' / (name + '.started.json'))
            require(start['source_sha256'] == protocol['source_sha256'], 'Stage started with different registered runtime bytes.')
        frozen = read(root / 'freeze.json') if (root / 'freeze.json').exists() else None
        if frozen:
            inputs['freeze.json'] = identity(root / 'freeze.json')
            require(frozen['protocol_sha256'] == protocol_hash and frozen['source_sha256'] == protocol['source_sha256'], 'Frozen protocol/source binding differs.')
            for name, expected in frozen['artifacts'].items():
                record = identity(relative(root, name)); inputs[name] = record
                require(record['sha256'] == expected, 'Frozen selected artifact changed.')
        for stage in STAGES:
            receipt = stages[stage]
            if receipt is None or stage.startswith('train-'):
                continue
            roster = protocol['rosters'][stage]; rows = receipt['rows']; by_id = {row['id']: row for row in rows}
            require(len(rows) == len(by_id) == len(roster) and set(by_id) == {a['id'] for a in roster}, 'Stage denominator or attempt IDs differ: ' + stage)
            model_stage = ('initial' if stage in ('corrections', 'development-initial') else 'corrected' if stage == 'development-corrected'
                else frozen['selected_stage'] if frozen and stage in ('confirmation', 'negative') else None)
            expected_bindings = {}
            for controller in {a['controller'] for a in roster}:
                expected_bindings[controller] = (digest(protocol['source_sha256']) if controller == 'teacher' else
                    registration['fixed_model_file_sha256'] if controller == 'fixed' else
                    _model_hash(root, model_stage, controller, stages, inputs))
            require(receipt['model_bindings'] == expected_bindings, 'Stage model identities differ: ' + stage)
            if stage in ('confirmation', 'negative'):
                require(frozen is not None and receipt['freeze_sha256'] == inputs['freeze.json']['sha256'], 'Evaluation is not bound to the frozen model selection.')
            audited = []
            for attempt in roster:
                row = by_id[attempt['id']]
                try:
                    require(all(row.get(key) == attempt[key] for key in ('case', 'controller', 'negative_control')), 'Stage row substitutes an unregistered scene or controller.')
                    path = f'episodes/{stage}/{attempt["id"]}'
                    if row['status'] in ('unknown', 'unattempted'):
                        require(row.get('strict_success') is False and row.get('complete') is False, 'Unknown/unattempted outcome claims success.')
                        require((root / path).exists() is (row['status'] == 'unknown'), 'Unknown/unattempted directory presence differs.')
                        result = dict(**attempt, status=row['status'], strict_success=False, complete=False,
                            initial_anchor_available=False, initial_anchor_binding_valid=False, initial_anchor_error_m=None,
                            recorded_complete_intervals=0, critical_contact_ticks=None, provenance_ok=False,
                            score=dict(valid=False, success=False), files_verified=0, native_score_recomputed=False)
                    else:
                        require(row['path'] == path, 'Sealed episode path differs from its attempt identity.')
                        require(row['policy_artifact_sha256'] == expected_bindings[attempt['controller']], 'Stage row model identity differs from its registered controller.')
                        result = audit_episode(root / path, attempt, expected_bindings[attempt['controller']],
                            tracker_sha256=protocol['source_sha256']['piper_rgbd_tracker.py'])
                        for key in ('status', 'strict_success', 'complete', 'initial_anchor_available', 'initial_anchor_binding_valid',
                                    'recorded_complete_intervals', 'critical_contact_ticks', 'provenance_ok', 'receipt_sha256', 'integrity_sha256', 'score'):
                            require(row.get(key) == result[key], 'Stage summary differs from independent evidence: ' + key)
                        if result['initial_anchor_error_m'] is None:
                            require(row.get('initial_anchor_error_m') is None, 'Stage summary invents an anchor error.')
                        else:
                            _close(row.get('initial_anchor_error_m'), result['initial_anchor_error_m'], 'Stage initial anchor error differs.')
                        for name in ('integrity.json', 'rollout.json', 'recording.json'):
                            if (root / path / name).exists():
                                inputs[path + '/' + name] = identity(root / path / name)
                    result['stage'] = stage; audited.append(result); results.append(result)
                except (ValueError, OSError, KeyError, TypeError, IndexError, AssertionError) as exc:
                    errors.append(dict(stage=stage, attempt=attempt['id'], reason=str(exc)))
            stage_results[stage] = audited
            if len(audited) != len(roster):
                continue
            require(receipt['strict_successes'] == sum(r['strict_success'] for r in audited), 'Stage success count differs.')
            if stage == 'diagnostic':
                threshold = protocol['diagnostic']['maximum_initial_camera_error_m']
                passed = all(r['strict_success'] and r['initial_anchor_available'] and r['initial_anchor_binding_valid']
                    and type(r['initial_anchor_error_m']) in (int, float) and math.isfinite(r['initial_anchor_error_m'])
                    and 0 <= r['initial_anchor_error_m'] <= threshold for r in audited)
                require(receipt['diagnostic_passed'] is passed, 'Diagnostic gate differs from reconstructed evidence.')
            if stage == 'demonstrations':
                require(receipt['data_quality'] == data_quality(audited, protocol), 'Demonstration quality gate differs from reconstructed evidence.')
        _training(root, stages, protocol, inputs)
        if frozen:
            candidates = []
            for order, name in enumerate(('initial', 'corrected')):
                rows = [row for row in stage_results['development-' + name] if row['controller'] == 'student']
                require(len(rows) == 16 and all(r['critical_contact_ticks'] is not None for r in rows), 'Development selection lacks complete outcomes.')
                model = read(root / f'models/{name}/full/model.json')
                candidates.append(dict(stage=name, strict_successes=sum(r['strict_success'] for r in rows),
                    critical_contact_ticks=sum(r['critical_contact_ticks'] for r in rows),
                    prediction_loss=model['training']['validation']['action_scale_mse'], order=order))
            winner = min(candidates, key=lambda c: (-c['strict_successes'], c['critical_contact_ticks'], c['prediction_loss'], c['order']))
            require(frozen['candidates'] == candidates and frozen['selected_stage'] == winner['stage'], 'Frozen selection differs from independent development comparison.')
        if stages.get('confirmation') and stages.get('negative') and not errors:
            qualification = final_gates(stage_results['confirmation'], stage_results['negative'], protocol,
                stages_complete=all(stages[name]['status'] == 'complete' for name in ('confirmation', 'negative')))
            report = read(root / 'report.json'); inputs['report.json'] = identity(root / 'report.json')
            for key, value in qualification.items():
                require(_equivalent(report[key], value), 'Final benchmark report differs from independent reconstruction: ' + key)
            require(report['qualification'] == ('bounded_simulation_skill' if qualification['qualified'] else 'research_candidate')
                and report['selected_stage'] == frozen['selected_stage'], 'Report qualification label or selected model differs.')
        elif (root / 'report.json').exists():
            report = read(root / 'report.json'); inputs['report.json'] = identity(root / 'report.json')
            blocked = [name for name in STAGES if stages[name] and stages[name]['status'] == 'blocked']
            require(report.get('qualified') is False and report.get('qualification') == 'research_candidate'
                and report.get('status') == ('blocked' if blocked else 'incomplete'), 'Incomplete study claims qualification or an inconsistent blocker.')
    except (ValueError, OSError, KeyError, TypeError, IndexError, AssertionError) as exc:
        errors.append(dict(stage=None, attempt=None, reason=str(exc)))
    return dict(schema=SCHEMA, passed=not errors, errors=errors, protocol_sha256=protocol_hash,
        stages_verified=[name for name in STAGES if name in stage_results],
        training_stages_verified=[name for name in STAGES if name.startswith('train-') and stages.get(name)],
        attempts_verified=len(results), native_scores_recomputed=sum(r.get('native_score_recomputed') is True for r in results),
        files_verified=sum(r['files_verified'] for r in results), results=results, qualification=qualification,
        evidence_inventory=inputs, evidence_inventory_sha256=digest(inputs),
        audit_source_sha256=identity(Path(__file__))['sha256'],
        diagnostic_helper_source_sha256=identity(Path(__file__).with_name('independent_audit.py'))['sha256'],
        scorer_source_sha256=identity(Path(scoring.__file__))['sha256'],
        physics_executed=False, estimator_executed=False, model_executed=False,
        source_authenticated=False, physical_robot_ready=False,
        encoder_pose_reference='Capture estimates use the link6 body origin in robot_base. Native scoring uses the authored site at link6 local [0,0,0.12003] m; this audit explicitly reconstructs that fixed offset.',
        scope='Independent byte, roster, clock, control-authority and numeric outcome reconstruction for all sealed completed stages. Missing stages remain unrun; no physical qualification.')


def _equivalent(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equivalent(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, abs_tol=1e-12, rel_tol=0)
    return type(left) is type(right) and left == right


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study')
    args = parser.parse_args(argv)
    result = audit(args.study)
    print(canonical(result).decode())
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
