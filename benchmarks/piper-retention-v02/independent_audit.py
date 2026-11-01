"""Independent readback of frozen retention evidence. No simulator/runtime imports.

All predicate/streak computations below are independent of probe_evidence.py.
This checks recorded consistency, not physics authenticity or physical sensors.
The raw study and source files are only read and rehashed; no trial is rerun.
"""
from __future__ import annotations

if not __debug__:
    raise RuntimeError("Independent retention audit refuses optimized Python.")

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import math

import numpy as np

MAX_FILE = 32*1024*1024
VARIANTS = ('measured_joint', 'issued_joint', 'cartesian')
FIELDS = {'t_seconds': 1, 'native_tick': 1, 'object_position_m': 3,
          'object_quaternion_wxyz': 4, 'object_linear_velocity_m_s': 3,
          'object_angular_velocity_rad_s': 3, 'tcp_position_m': 3,
          'tcp_quaternion_wxyz': 4, 'object_minus_tcp_local_m': 3,
          'grasp_contact_counts': 2, 'pad_normal_forces_n': 2,
          'table_object_contact_count': 1, 'unexpected_contact_count': 1,
          'gripper_joint_position_m': 2, 'joint_position_rad': 6,
          'arm_actuator_controls_rad': 6, 'requested_jaw_target_m': 1,
          'normal_rate_limited_jaw_target_m': 1, 'applied_jaw_target_m': 1,
          'pulse_active': 1, 'pulse_minimum_opening_m': 1, 'pulse_closing_tail': 1}
ADMISSIONS = ('evidence_qualified', 'timer_admitted')
HOLD_EVENTS = (*ADMISSIONS, 'hold_invalidated', 'abort', 'timed_out_hold')
BOUND_REFUSALS = frozenset({'initial_bias_exhausted', 'native_joint_margin_exhausted',
                            'tcp_position_error_exhausted', 'tcp_orientation_error_exhausted',
                            'joint_correction_exhausted', 'pose_solver_refused',
                            'missing_prior_issued_target'})
CORRECTION_LIMIT_RAD = .03
CORRECTION_RATE_RAD_S = .10
INITIAL_BIAS_LIMIT_RAD = .08
CORRECTION_DT_S = .02
TCP_POSITION_LIMIT_M = .004
TCP_ORIENTATION_LIMIT_RAD = .05


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def bad(token):
    raise ValueError('Nonfinite JSON '+token)


def identity(path):
    size, digest = 0, sha256()
    assert not path.is_symlink() and path.is_file()
    with path.open('rb') as stream:
        while chunk := stream.read(65536):
            size += len(chunk)
            assert size <= MAX_FILE
            digest.update(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def read(path):
    identity(path)
    result = json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=bad)
    canonical(result)
    return result


def episode(path):
    identity(path)
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(MAX_FILE+1)
    assert len(raw) <= MAX_FILE
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=bad)
    canonical(result)
    return result


def row_at(seconds, times):
    index = round(seconds/.001)-1
    assert 0 <= index < len(times) and abs(times[index]-seconds) < 1e-8
    return index


def rotation_local(arrays):
    q = arrays['tcp_quaternion_wxyz']
    assert np.max(np.abs(np.linalg.norm(q, axis=1)-1)) < 1e-8
    # Independently construct the transpose of the quaternion rotation matrix.
    w, x, y, z = q.T
    rotation = np.empty((len(q), 3, 3))
    rotation[:, 0, 0] = 1-2*(y*y+z*z)
    rotation[:, 0, 1] = 2*(x*y-w*z)
    rotation[:, 0, 2] = 2*(x*z+w*y)
    rotation[:, 1, 0] = 2*(x*y+w*z)
    rotation[:, 1, 1] = 1-2*(x*x+z*z)
    rotation[:, 1, 2] = 2*(y*z-w*x)
    rotation[:, 2, 0] = 2*(x*z-w*y)
    rotation[:, 2, 1] = 2*(y*z+w*x)
    rotation[:, 2, 2] = 1-2*(x*x+y*y)
    vector = arrays['object_position_m']-arrays['tcp_position_m']
    local = np.einsum('nji,nj->ni', rotation, vector)
    assert np.max(np.abs(local-arrays['object_minus_tcp_local_m'])) < 1e-9
    return arrays['object_minus_tcp_local_m']


def native_predicate(arrays, probe):
    times = arrays['t_seconds']
    origin = row_at(probe['start_time_s'], times)
    assert np.linalg.norm(arrays['object_position_m'][origin]-probe['object_origin_m']) < 1e-9
    assert np.linalg.norm(arrays['object_minus_tcp_local_m'][origin]-probe['local_offset_origin_m']) < 1e-9
    rise = arrays['object_position_m'][:, 2]-probe['object_origin_m'][2]
    drift = np.linalg.norm(arrays['object_minus_tcp_local_m']-probe['local_offset_origin_m'], axis=1)
    components = {
        'bilateral_contact': np.all(arrays['grasp_contact_counts'] > 0, axis=1),
        'bilateral_force': np.all(arrays['pad_normal_forces_n'] > .01, axis=1),
        'force_ceiling': np.all(arrays['pad_normal_forces_n'] <= 10., axis=1),
        'unsupported': arrays['table_object_contact_count'] == 0,
        'no_unexpected_contact': arrays['unexpected_contact_count'] == 0,
        'minimum_rise': rise+1e-12 >= .006,
        'maximum_local_drift': drift <= .003+1e-12,
    }
    condition = np.arange(len(times)) > origin
    for component in components.values():
        condition &= component
    streak = np.empty(len(times), dtype=np.int64)
    previous = 0
    for index, good in enumerate(condition):
        previous = previous+1 if good else 0
        streak[index] = previous
    return condition, streak, streak >= 100, components, rise, drift


def reasons_at(index, components):
    return [name for name, values in components.items() if not values[index]]


def motion(arrays, seconds):
    start = row_at(seconds, arrays['t_seconds'])
    end = min(start+100, len(arrays['t_seconds'])-1)
    tcp = arrays['tcp_position_m'][start:end+1]
    obj = arrays['object_position_m'][start:end+1]
    return {'requested_interval_s': .1, 'observed_interval_s': (end-start)*.001,
            'complete_100ms': end-start == 100,
            'tcp_endpoint_displacement_m': float(np.linalg.norm(tcp[-1]-tcp[0])),
            'tcp_path_length_m': float(np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1))),
            'tcp_maximum_excursion_m': float(np.max(np.linalg.norm(tcp-tcp[0], axis=1))),
            'object_endpoint_displacement_m': float(np.linalg.norm(obj[-1]-obj[0])),
            'scope': 'Native motion after a position-hold request, not measured stopping distance or physical response.'}


def treatment_boundary(policy):
    """First admission or eligible pre-admission hold-bound refusal."""
    events = policy.get('events') if type(policy) is dict else None
    if type(events) is not list:
        raise ValueError('Policy event tape is missing.')
    admissions = [event for event in events if type(event) is dict and event.get('type') == 'evidence_qualified']
    if len(admissions) > 1:
        raise ValueError('Evidence policy admits more than once.')
    admission_time = admissions[0]['t_seconds'] if admissions else None
    candidates = []
    if admissions:
        candidates.append({'time_s': admission_time, 'reason': 'admission',
                           'detail': admissions[0].get('reason')})
    for event in events:
        if (type(event) is dict and event.get('type') == 'hold_invalidated'
                and event.get('reason') in BOUND_REFUSALS):
            timestamp = event.get('t_seconds')
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                raise ValueError('Bound refusal has a malformed timestamp.')
            if admission_time is None or timestamp <= admission_time+1e-8:
                candidates.append({'time_s': float(timestamp), 'reason': 'pre_admission_bound_refusal',
                                   'detail': event['reason']})
    if not candidates:
        return None
    return min(candidates, key=lambda row: row['time_s'])


def common_cutoff(boundaries):
    """Choose one scene cutoff for all three variants, including refusals."""
    if set(boundaries) != {'measured_joint', 'issued_joint', 'cartesian'}:
        raise ValueError('Common cutoff requires all three variant records.')
    candidates = [{'variant': variant, **row} for variant, row in boundaries.items() if row is not None]
    candidates.sort(key=lambda row: row['variant'])
    timestamp = min((row['time_s'] for row in candidates), default=None)
    reasons = sorted({row['reason'] for row in candidates
                      if timestamp is not None and abs(row['time_s']-timestamp) <= 1e-8})
    return {'time_s': timestamp,
            'reason': '+'.join(reasons) if reasons else 'full_run_no_treatment',
            'treatment_boundaries': candidates}


def prefix_identity(arrays, actions, cutoff):
    """Hash native rows through the cutoff and only prior commands."""
    if len(arrays['t_seconds']) != 20000 or len(actions) != 1000:
        return {'verified': False, 'reason': 'incomplete_native_or_command_prefix'}
    if cutoff['time_s'] is None:
        native_rows = 20000
        command_rows = list(actions)
    else:
        native_rows = row_at(cutoff['time_s'], arrays['t_seconds'])+1
        command_rows = [action for action in actions if action['t_seconds'] < cutoff['time_s']-1e-8]
    prefix_hash = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name][:native_rows])
        prefix_hash.update(json.dumps({'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)},
                                      sort_keys=True, indent=2, allow_nan=False).encode()+b'\n')
        prefix_hash.update(value.tobytes())
    return {'verified': True, 'cutoff_time_s': cutoff['time_s'], 'cutoff_reason': cutoff['reason'],
            'native_rows': native_rows, 'native_sha256': prefix_hash.hexdigest(),
            'commands': len(command_rows), 'commands_sha256': sha256(
                json.dumps(command_rows, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n').hexdigest()}


def stationarity(arrays, start, end, complete, primary_success):
    """Recompute 5s native TCP stationarity from the admission pose."""
    positions = arrays['tcp_position_m'][start:end+1]
    quaternions = arrays['tcp_quaternion_wxyz'][start:end+1]
    displacement = np.linalg.norm(positions-positions[0], axis=1)
    dots = np.abs(quaternions @ quaternions[0])
    angles = 2*np.arccos(np.clip(dots, 0., 1.))
    max_position, max_angle = float(np.max(displacement)), float(np.max(angles))
    within = max_position <= TCP_POSITION_LIMIT_M+1e-12 and max_angle <= TCP_ORIENTATION_LIMIT_RAD+1e-12
    return {'maximum_tcp_position_excursion_m': max_position,
            'maximum_tcp_orientation_excursion_rad': max_angle,
            'terminal_tcp_position_delta_m': (positions[-1]-positions[0]).tolist(),
            'terminal_tcp_orientation_delta_rad': float(angles[-1]),
            'tcp_pose_within_bounds': bool(complete and within),
            'grip_retention_and_stationarity': bool(complete and primary_success and within)}


def paired_prefix(scene, attempts):
    """Return an explicit unknown for incomplete triplets, never an implicit pass."""
    variants = ('measured_joint', 'issued_joint', 'cartesian')
    if set(attempts) != set(variants):
        raise ValueError('Scene must retain all three scheduled variants.')
    if not all(attempts[variant].get('valid') is True for variant in variants):
        entries = [{'variant': variant, 'verified': False,
                    'reason': attempts[variant].get('reason', 'common_triplet_incomplete')
                              if attempts[variant].get('valid') is not True
                              else 'common_triplet_incomplete',
                    'native_rows_available': attempts[variant].get('native_rows_available')}
                   for variant in variants]
        return {'scene': scene, 'verified': False,
                'reason': 'invalid_attempt_prevents_complete_triplet',
                'variants': entries, 'treatment_boundaries': []}
    cutoff = common_cutoff({variant: attempts[variant]['boundary'] for variant in variants})
    prefixes = {variant: prefix_identity(attempts[variant]['arrays'], attempts[variant]['actions'], cutoff)
                for variant in variants}
    if any(record.get('verified') is not True for record in prefixes.values()):
        return {'scene': scene, 'verified': False,
                'reason': 'incomplete_common_prefix_evidence', 'variants': prefixes,
                'treatment_boundaries': cutoff['treatment_boundaries']}
    identities = {(record['cutoff_time_s'], record['cutoff_reason'], record['native_rows'],
                   record['native_sha256'], record['commands'], record['commands_sha256'])
                  for record in prefixes.values()}
    matched = len(identities) == 1
    if not matched:
        raise ValueError('Common acquisition prefixes differ: '+scene)
    expected = next(iter(prefixes.values()))
    return {'scene': scene, 'verified': True, **expected,
            'treatment_boundaries': cutoff['treatment_boundaries'],
            'variants': [{'variant': variant, **prefixes[variant]} for variant in variants]}


def check_stationarity_reported(actual, reported, complete):
    assert reported['complete_5s'] is bool(complete)
    for field in ('maximum_tcp_position_excursion_m', 'maximum_tcp_orientation_excursion_rad'):
        assert abs(reported[field]-actual[field]) <= 1e-10, field
    assert np.allclose(reported['terminal_tcp_position_delta_m'], actual['terminal_tcp_position_delta_m'],
                       atol=1e-10, rtol=0.)
    assert abs(reported['terminal_tcp_orientation_delta_rad']-
               actual['terminal_tcp_orientation_delta_rad']) <= 1e-10
    assert reported['tcp_pose_within_bounds'] is (complete and actual['tcp_pose_within_bounds'])
    assert reported['grip_retention_and_stationarity'] is (
        complete and actual['grip_retention_and_stationarity'])


def corrections_audit(policy, actions, boundary, variant):
    """Independently check bounded hold diagnostics and first issued command."""
    retention = policy.get('retention') or {}
    records = retention.get('hold_records')
    reference = retention.get('hold_reference')
    if type(records) is not list:
        raise ValueError('Hold diagnostics are missing.')
    assert policy.get('variant') == variant
    if reference is not None:
        assert type(reference) is dict
        assert reference['frozen_jaw_target_m'] == policy['frozen_jaw_target_m']
        issued = np.asarray(reference['issued_joint_targets_rad'], dtype=np.float64)
        assert issued.shape == (6,) and np.isfinite(issued).all()
        assert boundary is not None
    if boundary is not None and boundary['reason'] in ('admission', 'pre_admission_bound_refusal'):
        assert reference is not None
        if reference is not None:
            first = records[0]
            assert first['command_issued'] is True
            assert abs(first['t_seconds']-boundary['time_s']) <= 1e-8
            assert abs(reference['t_seconds']-boundary['time_s']) <= 1e-8
            action = next(row for row in actions if abs(row['t_seconds']-boundary['time_s']) <= 1e-8)
            if boundary['reason'] == 'admission':
                measured = np.asarray(reference['measured_joint_position_rad'], dtype=np.float64)
                assert issued.shape == measured.shape == (6,)
                expected_first = measured if variant == 'measured_joint' else issued
                assert np.array_equal(np.asarray(first['joint_targets_rad'], dtype=np.float64), expected_first)
            else:
                # A refused bound emits the terminal measured-joint/frozen-jaw hold.
                assert first['refused_bounds'] == boundary['detail']
                measured = reference['measured_joint_position_rad']
                assert first['joint_targets_rad'] == measured
                assert action['joint_targets_rad'] == measured
            assert first['joint_targets_rad'] == action['joint_targets_rad']
            assert first['frozen_jaw_target_m'] == action['gripper_position_m']
    if reference is not None:
        assert all(row['frozen_jaw_target_m'] == reference['frozen_jaw_target_m'] for row in records)
    for record in records:
        if not record['command_issued']:
            assert record.get('final_observation') is True
            continue
        action = next(row for row in actions if abs(row['t_seconds']-record['t_seconds']) <= 1e-8)
        assert record['joint_targets_rad'] == action['joint_targets_rad']
        assert record['frozen_jaw_target_m'] == action['gripper_position_m']
    records_by_time = {row['t_seconds']: row for row in records if row['command_issued']}
    assert len(records_by_time) == sum(row['command_issued'] for row in records)
    for action in actions:
        if action['t_seconds'] in records_by_time:
            assert records_by_time[action['t_seconds']]['joint_targets_rad'] == action['joint_targets_rad']
            assert records_by_time[action['t_seconds']]['frozen_jaw_target_m'] == action['gripper_position_m']

    if variant == 'cartesian' and reference is not None:
        raw_initial = np.asarray(reference['raw_pose_joint_targets_rad'], dtype=np.float64)
        fixed_bias = np.asarray(reference['fixed_bias_rad'], dtype=np.float64)
        assert raw_initial.shape == fixed_bias.shape == (6,)
        expected_bias = issued-raw_initial
        assert np.allclose(fixed_bias, expected_bias, atol=1e-12, rtol=0.)
        initial_refused = first['refused_bounds'] == 'initial_bias_exhausted'
        if not initial_refused:
            assert np.max(np.abs(fixed_bias)) <= INITIAL_BIAS_LIMIT_RAD+1e-12
        previous = np.zeros(6, dtype=np.float64)
        checked_rate_steps = 0
        for record in records:
            assert record['fixed_bias_rad'] == reference['fixed_bias_rad']
            if record.get('refused_bounds') is not None:
                assert any(event.get('type') == 'hold_invalidated'
                           and event.get('reason') == record['refused_bounds']
                           and abs(event.get('t_seconds', -1)-record['t_seconds']) <= 1e-8
                           for event in policy['events'])
            correction_value = record.get('correction_rad')
            if correction_value is None:
                continue
            correction = np.asarray(correction_value, dtype=np.float64)
            assert correction.shape == (6,) and np.isfinite(correction).all()
            assert np.max(np.abs(correction)) <= CORRECTION_LIMIT_RAD+1e-12
            desired_value = record.get('desired_correction_rad')
            if desired_value is not None:
                desired = np.asarray(desired_value, dtype=np.float64)
                assert desired.shape == (6,) and np.isfinite(desired).all()
                raw_current_value = record.get('raw_pose_joint_targets_rad')
                assert raw_current_value is not None
                raw_current = np.asarray(raw_current_value, dtype=np.float64)
                assert raw_current.shape == (6,) and np.isfinite(raw_current).all()
                assert np.allclose(desired, raw_current-raw_initial, atol=1e-12, rtol=0.)
                failure = record.get('refused_bounds') == 'joint_correction_exhausted'
                if not failure:
                    assert np.max(np.abs(desired)) <= CORRECTION_LIMIT_RAD+1e-12
                    axes = [index for index, (wanted, old) in enumerate(zip(desired, previous))
                            if abs(wanted-old) > CORRECTION_RATE_RAD_S*CORRECTION_DT_S+1e-12]
                    assert record['correction_rate_limited_axes'] == axes
                if record['command_issued']:
                    assert np.max(np.abs(correction-previous)) <= CORRECTION_RATE_RAD_S*CORRECTION_DT_S+1e-12
                    if record.get('refused_bounds') is None:
                        assert np.allclose(record['joint_targets_rad'], issued+correction, atol=1e-12, rtol=0.)
                    previous = correction
                    checked_rate_steps += 1
    elif reference is not None:
        assert reference.get('fixed_bias_rad') is None

    flags, refusals, axis_count, active, desired = {}, {}, 0, [], []
    for record in records:
        for key, value in record.items():
            if type(value) is bool and ('saturat' in key or 'limit' in key or 'refus' in key):
                flags[key] = flags.get(key, 0)+int(value)
        if record.get('refused_bounds') is not None:
            key = record['refused_bounds']; refusals[key] = refusals.get(key, 0)+1
        axis_count += len(record.get('correction_rate_limited_axes') or [])
        for name, target in (('correction_rad', active), ('desired_correction_rad', desired)):
            if record.get(name) is not None:
                values = np.asarray(record[name], dtype=np.float64)
                assert values.shape == (6,) and np.isfinite(values).all()
                target.extend(np.abs(values).tolist())
    computed = {'hold_record_count': len(records), 'flag_counts': flags,
                'bound_refusal_counts': refusals, 'rate_limited_axis_occurrences': axis_count,
                'maximum_abs_active_correction_rad': max(active) if active else None,
                'maximum_abs_desired_correction_rad': max(desired) if desired else None,
                'input_refusals': retention.get('input_refusals', [])}
    return computed


def self_tests():
    count = 240
    arrays = {name: np.zeros((count,) if width == 1 else (count, width), dtype=np.float64)
              for name, width in FIELDS.items()}
    arrays['t_seconds'] = np.arange(1, count+1)*.001
    arrays['object_position_m'][:] = [.3, 0, .08]
    arrays['object_minus_tcp_local_m'][:] = [0, 0, -.1]
    arrays['object_position_m'][20:, 2] += .006
    arrays['grasp_contact_counts'][:] = 1
    arrays['pad_normal_forces_n'][:] = .02
    probe = {'start_time_s': .02, 'object_origin_m': [.3, 0, .08],
             'local_offset_origin_m': [0, 0, -.1]}
    _, streak, qualified, _, _, _ = native_predicate(arrays, probe)
    assert streak[118] == 99 and not qualified[118]
    assert streak[119] == 100 and qualified[119]
    arrays['pad_normal_forces_n'][60, 0] = .01
    _, streak, qualified, _, _, _ = native_predicate(arrays, probe)
    assert streak[60] == 0 and not qualified[119]
    assert qualified[160]
    arrays['object_minus_tcp_local_m'][220, 0] = .0031
    _, _, qualified, components, _, _ = native_predicate(arrays, probe)
    assert not qualified[220] and reasons_at(220, components) == ['maximum_local_drift']
    return {'passed': True, 'checks': ['99_vs100_native_ticks', 'strict_force_threshold_and_gap', '3mm_drift']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study, repo, output = args.study, args.repo, args.output
    if output.exists() or output.is_symlink():
        raise ValueError('Audit output must be new')
    def inventory():
        result = {}
        for p in sorted(study.rglob('*')):
            if p.is_symlink(): raise ValueError('Symlink in frozen study')
            if p.is_file(): result[p.relative_to(study).as_posix()] = identity(p)
        return result
    raw_before = inventory()
    protocol, report, seal = read(study/'protocol.json'), read(study/'report.json'), read(study/'frozen.json')
    assert seal['protocol_sha256'] == raw_before['protocol.json']['sha256']
    assert protocol['schema'] == report['schema'] == 'dvidia.piper-retention-study/v1'
    assert protocol['planned_attempts'] == report['planned_attempts'] == report['attempted'] == 36
    assert protocol['variants'] == ['measured_joint', 'issued_joint', 'cartesian']
    assert protocol['primary_endpoint']['duration_s'] == 5.
    assert protocol['force_ceiling_n'] == 10.
    assert protocol['secondary_endpoint']['duration_s'] == 5.
    assert protocol['secondary_endpoint']['position_ceiling_m'] == TCP_POSITION_LIMIT_M
    assert protocol['secondary_endpoint']['orientation_ceiling_rad'] == TCP_ORIENTATION_LIMIT_RAD
    planned = [(case, variant) for case in protocol['cases'] for variant in case['variant_order']]
    assert len({case['id'] for case, _ in planned}) == 12
    assert [r['id'] for r in report['rows']] == [c['id']+'-'+v for c,v in planned]
    source_before = {name: identity(repo/'src/dvidia_training'/name)['sha256'] for name in protocol['source_sha256']}
    assert source_before == protocol['source_sha256'] == report['source_sha256']
    rows, attempts_by_scene, paired_by_scene = [], {case['id']: {} for case in protocol['cases']}, {}
    def retain_attempt(scene, variant, value):
        attempts_by_scene[scene][variant] = value
        if len(attempts_by_scene[scene]) == 3:
            # The fixed roster is case-major. Release each scene's native arrays
            # immediately after its three common-prefix digests are produced.
            paired_by_scene[scene] = paired_prefix(scene, attempts_by_scene[scene])
            attempts_by_scene[scene] = {}
    totals = {v:Counter(attempts=0, valid=0, admissions=0, false_initial=0,
                        native_5s_holds=0, controller_5s_holds=0, combined_5s_holds=0,
                        final_both_qualified=0, grip_and_stationary_5s=0,
                        native_rows=0, commands=0) for v in protocol['variants']}
    for summary,(case,variant) in zip(report['rows'],planned):
        folder = study/summary['id']
        for name, expected in summary['files'].items():
            assert '/' not in name and name not in ('.','..')
            assert identity(folder/name) == expected
        item = episode(folder/'episode.json.gz')
        assert item['id'] == summary['id'] and item['config'] == case['config'] and item['variant'] == variant
        assert item['valid'] == summary['valid']
        counter = totals[variant]; counter['attempts'] += 1
        if not item['valid'] or item['error'] is not None:
            assert not summary['evaluation']['primary']['success']
            native_summary = summary.get('evaluation', {}).get('native') or {}
            retain_attempt(case['id'], variant, {'valid': False,
                'native_rows_available': native_summary.get('rows'),
                'reason': 'invalid_attempt_retained_as_unknown'})
            rows.append({'id':item['id'],'valid':False,'scope':'Retained unknown; cannot qualify.'})
            continue
        import zipfile
        with zipfile.ZipFile(folder/'native.npz') as bundle:
            entries=bundle.infolist()
            assert len(entries)==22 and len({e.filename for e in entries})==22
            assert sum(e.file_size for e in entries)<=MAX_FILE
            assert all(e.file_size<=2*1024*1024 and '/' not in e.filename for e in entries)
        with np.load(folder/'native.npz',allow_pickle=False) as archive:
            assert set(archive.files)==set(FIELDS)
            arrays={name:archive[name] for name in archive.files}
        for name,width in FIELDS.items():
            assert arrays[name].dtype==np.float64
            assert arrays[name].shape==((20000,) if width==1 else (20000,width))
            assert np.isfinite(arrays[name]).all()
        times=arrays['t_seconds']
        assert np.array_equal(arrays['native_tick'],np.arange(1,20001))
        assert np.max(np.abs(times-np.arange(1,20001)*.001))<1e-8
        assert not np.any(arrays['pulse_active']) and not np.any(arrays['pulse_closing_tail'])
        rotation_local(arrays)
        policy,metrics=item['policy'],item['metrics']
        assert metrics['valid'] and metrics['warnings']==[]
        unexpected_ticks = int(np.count_nonzero(arrays['unexpected_contact_count']))
        assert unexpected_ticks == metrics['unexpected_contact_native_ticks'] == 0
        actions,trajectory=item['actions'],item['trajectory']
        assert len(actions)==1000 and len(trajectory)==1001
        counter.update(valid=1,native_rows=20000,commands=1000)
        control=np.array(trajectory[0]['native_actuator_controls'],dtype=float)
        rates=np.array([case['config']['joint_target_rate_rad_s']*.001]*6+[case['config']['jaw_target_rate_m_s']*.001])
        requested=policy['requested_controls']
        assert len(requested)==1000
        for i,(action,request) in enumerate(zip(actions,requested)):
            assert abs(action['t_seconds']-i*.02)<1e-8 and request['index']==i
            assert all(action[k]==request[k] for k in ('t_seconds','joint_targets_rad','gripper_position_m'))
            target=np.array(action['joint_targets_rad']+[action['gripper_position_m']])
            for tick in range(i*20,(i+1)*20):
                control += np.clip(target-control,-rates,rates)
                actual=np.r_[arrays['arm_actuator_controls_rad'][tick],arrays['applied_jaw_target_m'][tick]]
                assert np.max(np.abs(control-actual))<1e-12
            for field in ('object_position_m','joint_position_rad','tcp_position_m'):
                assert np.max(np.abs(np.array(trajectory[i+1][field])-arrays[field][(i+1)*20-1]))<1e-9
        admissions=[e for e in policy['events'] if e['type']=='evidence_qualified']
        assert len(admissions)<=1
        audited={'id':item['id'],'valid':True,'admitted':bool(admissions),
                 'native_5s_hold':False,'controller_5s_hold':False,'combined_5s_hold':False,
                 'false_initial':False,'final_both_qualified':False,'commands_reconstructed':1000}
        if policy['probes']:
            _,_,last_qualified,_,_,_=native_predicate(arrays,policy['probes'][-1])
            audited['final_both_qualified']=bool(last_qualified[-1] and policy['admitted'])
        boundary = treatment_boundary(policy)
        if admissions:
            event=admissions[0]; counter['admissions']+=1
            probe=next(p for p in policy['probes'] if p['probe_index']==event['probe_index'])
            condition,streak,qualified,components,rise,drift=native_predicate(arrays,probe)
            start=row_at(event['t_seconds'],times)
            target_time=event['t_seconds']+5.
            complete=bool(target_time<=times[-1]+1e-8)
            end=row_at(target_time,times) if complete else len(times)-1
            correct=bool(qualified[start]); ceiling_ok=bool(np.all(arrays['pad_normal_forces_n'][start:]<=10.))
            retained=[q for q in policy['qualification_checks'] if abs(q['t_seconds']-target_time)<1e-8 and q['probe_index']==event['probe_index']]
            controller=bool(complete and len(retained)==1 and retained[0]['qualified'] is True)
            success=bool(correct and complete and np.all(condition[start:end+1]) and ceiling_ok)
            losses=np.flatnonzero(~condition[start:end+1]); crossing=start+int(losses[0]) if len(losses) else None
            audited.update(admission_s=event['t_seconds'],native_admission_streak_ms=int(streak[start]),
                           false_initial=not correct,native_5s_hold=success,controller_5s_hold=controller,
                           combined_5s_hold=success and controller,force_ceiling_respected=ceiling_ok,
                           complete_5s=complete,first_loss_s=float(times[crossing]) if crossing is not None else None,
                           first_loss_reasons=reasons_at(crossing,components) if crossing is not None else [],
                           object_z_delta_at_endpoint_m=float(arrays['object_position_m'][end,2]-arrays['object_position_m'][start,2]),
                           tcp_z_delta_at_endpoint_m=float(arrays['tcp_position_m'][end,2]-arrays['tcp_position_m'][start,2]))
            evaluation=summary['evaluation']
            assert evaluation['first_admission']['native_prefix_qualified']==correct
            assert evaluation['primary']['success']==success
            assert evaluation['primary']['controller_retained_at_5s']==controller
            assert evaluation['primary']['combined_qualified_pause']==(success and controller)
            secondary = stationarity(arrays, start, end, complete, success)
            check_stationarity_reported(secondary, evaluation['stationarity'], complete)
            audited['grip_and_stationary_5s'] = secondary['grip_retention_and_stationarity']
            # Same jaw target after admission: no hidden squeeze escalation.
            held_jaw=policy['frozen_jaw_target_m']
            assert all(a['gripper_position_m']==held_jaw for a in actions if a['t_seconds']>=event['t_seconds']-1e-8)
        else:
            assert summary['evaluation']['first_admission'] is None
            assert not summary['evaluation']['primary']['success']
            assert summary['evaluation']['stationarity']['complete_5s'] is False
            assert summary['evaluation']['stationarity']['tcp_pose_within_bounds'] is False
            assert summary['evaluation']['stationarity']['grip_retention_and_stationarity'] is False
            audited['grip_and_stationary_5s'] = False
        correction_summary = corrections_audit(policy, actions, boundary, variant)
        assert correction_summary == {key: summary['evaluation']['corrections'][key]
                                      for key in correction_summary}
        assert summary['evaluation']['final']['both_qualified']==audited['final_both_qualified']
        counter.update(false_initial=int(audited['false_initial']),native_5s_holds=int(audited['native_5s_hold']),
                       controller_5s_holds=int(audited['controller_5s_hold']),combined_5s_holds=int(audited['combined_5s_hold']),
                       final_both_qualified=int(audited['final_both_qualified']),
                       grip_and_stationary_5s=int(audited['grip_and_stationary_5s']))
        retain_attempt(case['id'], variant, {'valid': True, 'boundary': boundary,
            'arrays': arrays, 'actions': actions})
        rows.append(audited)
    inspection = read(study/'inspection.json')
    assert inspection['schema'] == 'dvidia.piper-retention-inspection/v1'
    assert inspection['attempts'] == 36 and inspection['native_rows'] == 720000
    inspected_rows = {row['id']: row for row in inspection['rows']}
    assert set(inspected_rows) == {row['id'] for row in report['rows']}
    for row in report['rows']:
        checked = inspected_rows[row['id']]
        assert checked['evaluation_recomputed'] is True
        if row['valid']:
            assert checked['native_validated'] is True and checked['controls_recomputed'] is True
        else:
            assert checked['retained_refusal'] is True or checked['native_validated'] is False
    inspected_prefixes = {row['scene_id']: row for row in inspection['acquisition_prefixes']}
    assert set(inspected_prefixes) == {case['id'] for case in protocol['cases']}
    paired=[]
    for scene, pair in paired_by_scene.items():
        if pair['verified'] is False:
            inspection_row = inspected_prefixes[scene]
            assert inspection_row['verified'] is False
            assert inspection_row['reason'] == 'invalid_attempt_prevents_complete_triplet'
            paired.append(pair)
            continue
        inspector = inspected_prefixes[scene]
        assert inspector['verified'] is True
        for field in ('cutoff_time_s', 'cutoff_reason', 'native_rows', 'native_sha256',
                      'commands', 'commands_sha256'):
            value = pair[field]
            assert inspector[field] == value, (scene, field)
        assert inspector['treatment_boundaries'] == pair['treatment_boundaries']
        paired.append(pair)
    source_after={name:identity(repo/'src/dvidia_training'/name)['sha256'] for name in source_before}
    assert source_before==source_after and raw_before==inventory()
    output.parent.mkdir(parents=True,exist_ok=True)
    value={'schema':'dvidia.retention-independent-audit/v1','passed':True,'self_tests':self_tests(),
           'protocol_sha256':raw_before['protocol.json']['sha256'],'source_sha256':source_before,
           'frozen_raw_files':raw_before,'rows':rows,'totals':{v:dict(c) for v,c in totals.items()},
           'paired_acquisition_prefixes':paired,'source_and_raw_inputs_unchanged':True,
           'scope':'Independent recorded-predicate and command readback, not independent physics, authenticated sensors or hardware qualification.'}
    with output.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'passed':True,'totals':value['totals'],'raw_files':len(raw_before)}))


if __name__=='__main__':main()
