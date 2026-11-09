"""Read-only diagnostic audit; imports no controller, renderer or environment.

Recomputes numeric evidence and identities, not the simulator itself. Passing
this audit means the reported failures or successes agree with saved evidence;
it does not imply that the diagnostic or learned-skill gates passed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from dvidia_training import piper_feedback_score as scoring


SCHEMA = 'dvidia.camera-action-diagnostic-audit/v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024**2:
        raise ValueError('Expected a bounded ordinary JSON artifact: '+path.name)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field.')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value.')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Artifact is not an ordinary file.')
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def relative(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or str(path) != name:
        raise ValueError('Artifact paths must be canonical relative paths.')
    result = root/path
    if any(p.is_symlink() for p in (result, *result.parents) if str(p) not in ('/tmp', '/var')):
        raise ValueError('Artifact path traverses a symlink.')
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(root):
    manifest = read(root/'integrity.json')
    require(manifest.get('schema') == 'dvidia.camera-action-rollout/v2/integrity', 'Unexpected episode integrity schema.')
    files = manifest['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'integrity.json'}
    require(set(files) == actual, 'Episode inventory does not match its seal.')
    for name, expected in files.items():
        require(identity(relative(root, name)) == expected, 'Artifact identity changed: '+name)
    return files


def _reference(root, ref, files):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'}, 'Malformed artifact reference.')
    require(files.get(ref['path']) == {key: ref[key] for key in ('bytes', 'sha256')}, 'Artifact reference does not match sealed bytes.')
    return read(relative(root, ref['path']))


def _close(a, b, name, tolerance=1e-8):
    require(type(a) in (int, float) and type(b) in (int, float)
        and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=tolerance, rel_tol=0.), name)


def _transform(metadata):
    raw = metadata.get('world_from_robot_base')
    require(metadata.get('world_from_robot_base_sha256') == digest(raw), 'Fixed-base transform hash differs.')
    value = np.asarray(raw, dtype=float)
    if value.shape == (16,):
        value = value.reshape(4, 4)
    require(value.shape == (4, 4) and np.isfinite(value).all(), 'Actual world-from-base transform is absent.')
    require(np.allclose(value[3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
        and np.allclose(value[:3, :3].T@value[:3, :3], np.eye(3), atol=1e-8, rtol=0)
        and math.isclose(float(np.linalg.det(value[:3, :3])), 1., abs_tol=1e-8), 'Invalid fixed-base transform.')
    return value


def audit_episode(root, expected_case):
    root = Path(root)
    files = inventory(root)
    summary = read(root/'rollout.json')
    require(summary.get('case') == expected_case and summary.get('mode') == 'teacher'
        and summary.get('negative_control') is None, 'Diagnostic identity differs from registration.')
    result = dict(case_id=expected_case['id'], files_verified=len(files), status=summary['status'],
        strict_success=False, complete_intervals=0, native_ticks=0, camera_pairs=0,
        initial_anchor_error_m=None, available_position_error_p95_m=None,
        maximum_native_object_displacement_m=None)
    if 'recording.json' not in files:
        require(summary['status'] == 'refused' and summary['recorded_complete_intervals'] == 0
            and summary['strict_success'] is False, 'Setup failure claimed completed data or success.')
        return result
    recording = read(root/'recording.json')
    require(recording.get('schema') == 'dvidia.camera-action-rollout/v2'
        and recording.get('control_mode') == 'teacher', 'Diagnostic recording schema/control changed.')
    require(recording['provenance']['source_group'] == expected_case['group_id']
        and recording['provenance']['case_id'] == expected_case['id']
        and recording['goal_position_m'] == expected_case['goal_position_m'], 'Recording group/goal identity changed.')
    steps = recording['steps']
    require(len(steps) == summary['recorded_complete_intervals'], 'Recorded interval count differs.')
    context = read(root/'context.json')
    transform = _transform(context['metadata'])
    native_episode = read(root/'native_episode.json')
    native_metadata = read(root/'native_metadata.json')
    with np.load(root/'native.npz', allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    require(all(value.dtype.kind in 'iuf' for value in arrays.values()), 'Native arrays are not numeric-only.')
    score = scoring.score_feedback_episode(native_episode, arrays, native_metadata, expected_case['goal_position_m'][:2])
    require(score == summary['score'], 'Saved independent score differs from recomputation.')
    require(len(arrays['native_tick']) == native_metadata['rows'], 'Native row count differs.')
    require(np.array_equal(arrays['native_tick'], np.arange(1, native_metadata['rows']+1)), 'Native tick ordering differs.')
    require(np.allclose(arrays['t_seconds'], np.arange(1, native_metadata['rows']+1)*.001, atol=1e-8, rtol=0), 'Native 1 ms cadence differs.')
    initial_world = np.asarray(native_episode['trajectory'][0]['object_position_m'], dtype=float)
    initial_base = transform[:3, :3].T@(initial_world-transform[:3, 3])
    native_position = np.asarray(arrays['object_position_m'])
    prior = recording['initial']; estimates = []; visible_errors = []; cameras = {}
    for index, snapshot in enumerate([prior]+[s['after'] for s in steps]):
        _close(snapshot['timestamp_seconds'], index*.02, 'Encoder 50 Hz cadence differs.')
        camera = snapshot['camera']; frame = camera['value']
        require(frame['index'] == index//2 and snapshot['vision_frame_index'] == index//2, 'Camera source index differs from 25 Hz cadence.')
        _close(camera['captured_at_seconds'], (index//2)*.04, 'Camera capture time differs.')
        _close(camera['age_seconds'], index*.02-(index//2)*.04, 'Camera age was reset or changed.')
        require(camera['captured_at_seconds'] <= camera['available_at_seconds'] <= snapshot['timestamp_seconds']+1e-8,
            'Camera timestamps are noncausal.')
        if frame['index'] in cameras:
            require(cameras[frame['index']] == frame, 'A reused camera frame changed its bytes.')
        cameras[frame['index']] = frame
        for channel in ('rgb', 'depth'):
            reference = frame[channel]
            require(files.get(reference['path']) == {key: reference[key] for key in ('bytes', 'sha256')}, 'Frame source identity changed.')
    require(digest(recording['calibration']) == recording['binding']['calibration_sha256'], 'Camera calibration binding differs.')
    for index, step in enumerate(steps):
        require(step['index'] == index and step['before'] == prior, 'Transition prefix continuity differs.')
        require(step['command']['authority'] == 'teacher' and step['command']['operator_correction'] is False,
            'Diagnostic issued authority is not its teacher.')
        perception = _reference(root, step['perception'], files)
        query = _reference(root, step['teacher_query'], files)
        for sidecar in (perception, query):
            require(sidecar['index'] == index and sidecar['snapshot_sha256'] == digest(step['before']), 'Sidecar is bound to a different decision snapshot.')
        require(query['action'] == step['command']['issued'] == step['command']['requested']
            and query['origin'] == 'privileged_simulator_teacher', 'Teacher query/issued targets differ.')
        require(perception['frame'] == step['before']['camera']['value']
            and perception['calibration_sha256'] == recording['binding']['calibration_sha256']
            and perception['tracker_sha256'] == recording['tracker_sha256'], 'Perception frame/calibration/tracker binding differs.')
        estimate = perception['estimate']; estimates.append(estimate)
        camera = step['before']['camera']
        require(estimate['frame_index'] == camera['value']['index'], 'Estimate frame differs from camera.')
        for field in ('captured_at_seconds', 'available_at_seconds', 'age_seconds'):
            _close(estimate[field], camera[field], 'Estimate and source camera timestamps differ.')
        _close(estimate['decision_at_seconds'], index*.02, 'Estimate decision clock differs.')
        tick = (index+1)*20-1
        require(tick < native_metadata['rows'], 'Complete motor interval has no native endpoint.')
        np.testing.assert_allclose(step['applied']['ramped_targets']['joint_targets_rad'],
            arrays['arm_actuator_controls_rad'][tick], atol=1e-9, rtol=0)
        _close(step['applied']['ramped_targets']['gripper_position_m'], float(arrays['applied_jaw_target_m'][tick]), 'Applied gripper endpoint differs from native data.')
        if estimate['status'] == 'available':
            position = np.asarray(estimate['object_position_estimate_m'])
            captured_tick = round(estimate['captured_at_seconds']/.001)
            world = initial_world if captured_tick == 0 else native_position[captured_tick-1]
            truth = transform[:3, :3].T@(world-transform[:3, 3])
            visible_errors.append(float(np.linalg.norm(position-truth)))
            for field in ('object_position_estimate_m', 'object_position_uncertainty_m'):
                signal = estimate['signals'][field]
                require(signal['status'] == 'available', 'Available estimate has unavailable signals.')
                for stamp in ('captured_at_seconds', 'available_at_seconds', 'age_seconds'):
                    _close(signal[stamp], estimate[stamp], 'Estimate signal age/source differs.')
        else:
            for field in ('object_position_estimate_m', 'object_position_uncertainty_m'):
                require(estimate[field] is None and estimate['signals'][field]['value'] is None,
                    'Unavailable estimate forwards a position or uncertainty.')
        prior = step['after']
    query_count = sum(name.startswith('teacher/') and name.endswith('.json') for name in files)
    require(summary['teacher_queries'] == query_count and len(steps) <= query_count <= len(steps)+1 and summary['student_calls'] == 0,
        'Diagnostic teacher query count or student intervention differs.')
    require(summary['camera_pairs'] == len(cameras), 'Unique camera pair count differs.')
    expected_success = summary['status'] == 'complete' and len(steps) == 1000 and score.get('success') is True
    require(summary['strict_success'] is expected_success, 'Diagnostic strict success differs from complete native scoring.')
    initial_error = None
    if estimates and estimates[0]['status'] == 'available':
        initial_error = float(np.linalg.norm(np.asarray(estimates[0]['object_position_estimate_m'])-initial_base))
        _close(summary['initial_anchor_error_m'], initial_error, 'Initial camera anchor error differs from native reconstruction.')
    else:
        require(summary['initial_anchor_error_m'] is None, 'Unavailable anchor has a fabricated error.')
    missing = [e['status'] != 'available' for e in estimates]
    longest = streak = 0
    for value in missing:
        streak = streak+1 if value else 0; longest = max(longest, streak)
    expected_perception = dict(decisions=len(estimates), available=sum(not m for m in missing),
        missing=sum(missing), longest_missing_seconds=longest*.02)
    require(summary['perception'] == expected_perception, 'Perception coverage/missing-duration summary differs.')
    result.update(strict_success=expected_success, complete_intervals=len(steps), native_ticks=native_metadata['rows'],
        camera_pairs=len(cameras), initial_anchor_error_m=initial_error, perception=expected_perception,
        available_position_error_p95_m=float(np.quantile(visible_errors, .95)) if visible_errors else None,
        maximum_native_object_displacement_m=float(np.max(np.linalg.norm(native_position-initial_world, axis=1))) if len(native_position) else 0.,
        score=score)
    return result


def audit(directory):
    root = Path(directory)
    errors, results = [], []
    try:
        protocol, registration = read(root/'protocol.json'), read(root/'registration.json')
        require(identity(root/'protocol.json')['sha256'] == registration['protocol_sha256'], 'Study protocol changed after registration.')
        require(protocol['source_sha256'] == registration['source_sha256'], 'Registration/runtime source lists disagree.')
        for name, expected in protocol['source_sha256'].items():
            require(identity(Path(scoring.__file__).with_name(name))['sha256'] == expected, 'Registered source changed: '+name)
        roster = protocol['rosters']['diagnostic']
        require(len(roster) == 4 and len({r['id'] for r in roster}) == 4, 'Expected exactly four diagnostic attempts.')
        stage = read(root/'stages/diagnostic.json')
        seal = read(root/'stages/diagnostic.integrity.json')
        require(seal == dict(sha256=identity(root/'stages/diagnostic.json')['sha256'],
            protocol_sha256=identity(root/'protocol.json')['sha256']), 'Diagnostic stage seal changed.')
        require({row['id'] for row in stage['rows']} == {row['id'] for row in roster}
            and len(stage['rows']) == 4, 'Diagnostic denominator differs from its roster.')
        teacher_binding = digest(protocol['source_sha256'])
        require(stage.get('model_bindings') == {'teacher': teacher_binding}, 'Diagnostic teacher runtime binding differs.')
        for attempt in roster:
            try:
                row = next(r for r in stage['rows'] if r['id'] == attempt['id'])
                require(row['case'] == attempt['case'], 'Stage row is associated with a different case.')
                bound_case = {**attempt['case'], 'policy_artifact_sha256': teacher_binding}
                result = audit_episode(root/'episodes/diagnostic'/attempt['id'], bound_case)
                require(row['strict_success'] is result['strict_success'], 'Stage strict success differs from raw evidence.')
                results.append(result)
            except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
                errors.append(dict(attempt=attempt['id'], reason=str(exc)))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append(dict(attempt=None, reason=str(exc)))
    return dict(schema=SCHEMA, passed=not errors and len(results) == 4, errors=errors,
        attempts_verified=len(results), strict_successes=sum(r['strict_success'] for r in results), results=results,
        audit_source_sha256=identity(Path(__file__))['sha256'], scorer_source_sha256=identity(Path(scoring.__file__))['sha256'],
        physics_executed=False, estimator_executed=False, source_authenticated=False, physical_robot_ready=False,
        scope='Independent byte/clock/control and numeric task reconstruction; synthetic depth accuracy and hardware transfer remain unverified.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study')
    args = parser.parse_args(argv)
    result = audit(args.study)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
