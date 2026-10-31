"""Bounded post-run evidence for one simulation-only PiPER probe.

Native sidecars are privileged simulator records, not physical sensor evidence.
Hashes detect corruption and bind context; they do not authenticate an external
claim. The unchanged feasibility v1 gate retains every unknown load/friction/
force/payload assumption. A successful probe never estimates those quantities.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import re

import numpy as np

from . import task_feasibility

SCHEMA = 'dvidia.probe-evidence/v1'
CONTEXT_SCHEMA = 'dvidia.probe-context/v1'
MAX_ROWS = 30000
MAX_JSON_BYTES = 256*1024
DT_S = .001
MIN_FORCE_N = .01
MIN_RISE_M = .006
MAX_DRIFT_M = .003
MIN_DWELL_S = .1
CRITERIA = {'native_dt_s': DT_S, 'minimum_per_finger_force_n_exclusive': MIN_FORCE_N,
            'minimum_rise_m': MIN_RISE_M, 'maximum_tcp_local_drift_m': MAX_DRIFT_M,
            'minimum_continuous_dwell_s': MIN_DWELL_S,
            'distance_comparison_epsilon': 1e-12, 'origin_tolerance_m': 1e-9,
            'initial_sample_tolerance_m': .001, 'time_tolerance_s': 1e-8}
FIELDS = (('t_seconds', 1), ('native_tick', 1), ('object_position_m', 3),
          ('object_quaternion_wxyz', 4), ('object_linear_velocity_m_s', 3),
          ('object_angular_velocity_rad_s', 3), ('tcp_position_m', 3),
          ('tcp_quaternion_wxyz', 4), ('object_minus_tcp_local_m', 3),
          ('grasp_contact_counts', 2), ('pad_normal_forces_n', 2),
          ('table_object_contact_count', 1), ('unexpected_contact_count', 1),
          ('gripper_joint_position_m', 2), ('joint_position_rad', 6),
          ('arm_actuator_controls_rad', 6), ('requested_jaw_target_m', 1),
          ('normal_rate_limited_jaw_target_m', 1), ('applied_jaw_target_m', 1),
          ('pulse_active', 1), ('pulse_minimum_opening_m', 1), ('pulse_closing_tail', 1))
CONTEXT_FIELDS = {'schema', 'run_id', 'object_id', 'grasp_id', 'probe_index', 'robot_profile',
                  'model_sha256', 'source_sha256', 'asset_lock_sha256', 'command_sha256',
                  'pose_binding_sha256', 'feasibility_sha256', 'calibration_id', 'calibration_scope',
                  'initial_object_position_m', 'probe_start_s', 'probe_origin_object_position_m',
                  'probe_origin_tcp_local_offset_m', 'per_finger_force_ceiling_n', 'native_status'}
DIGEST_FIELDS = ('model_sha256', 'source_sha256', 'asset_lock_sha256', 'command_sha256',
                 'pose_binding_sha256', 'feasibility_sha256')
SCOPE = ('Post-run privileged native simulation evidence for one bound probe. Synthetic assumptions '
         'are not hardware calibration. Hashes detect corruption, not authenticity. No physical '
         'actuation, load/friction inference or physical robot qualification.')


class ProbeEvidenceError(ValueError):
    """Malformed, unbounded or internally inconsistent evidence input."""


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ProbeEvidenceError('Evidence must be bounded finite plain JSON.') from exc


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def _number(value, name, low=0., high=30.):
    if type(value) not in (int, float):
        raise ProbeEvidenceError(name+' must be a finite number.')
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ProbeEvidenceError(name+' must be finite.') from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ProbeEvidenceError(name+' is outside its bound.')
    return number


def _vector(value, name):
    if type(value) not in (list, tuple) or len(value) != 3:
        raise ProbeEvidenceError(name+' must have three values.')
    return [_number(item, name, -10., 10.) for item in value]


def _context(value):
    if type(value) is not dict or set(value) != CONTEXT_FIELDS:
        raise ProbeEvidenceError('Probe context fields differ from schema.')
    if (value['schema'] != CONTEXT_SCHEMA or value['robot_profile'] != 'piper'
            or value['calibration_scope'] != 'simulation_assumptions_not_hardware'):
        raise ProbeEvidenceError('Unsupported probe/context scope.')
    result = deepcopy(value)
    for name in ('run_id', 'object_id', 'grasp_id'):
        if (type(value[name]) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,159}', value[name])):
            raise ProbeEvidenceError('Expected a bounded identifier: '+name)
    if type(value['probe_index']) is not int or not 0 <= value['probe_index'] <= 8:
        raise ProbeEvidenceError('probe_index must be an integer in [0,8].')
    for name in DIGEST_FIELDS:
        if type(value[name]) is not str or not re.fullmatch('[0-9a-f]{64}', value[name]):
            raise ProbeEvidenceError('Expected lowercase SHA256: '+name)
    calibration = value['calibration_id']
    if calibration is not None and (type(calibration) is not str
                                   or not re.fullmatch(r'synthetic_[A-Za-z0-9._:-]{1,140}', calibration)):
        raise ProbeEvidenceError('Calibration must be a synthetic_ identifier or unknown.')
    for name in ('initial_object_position_m', 'probe_origin_object_position_m',
                 'probe_origin_tcp_local_offset_m'):
        result[name] = _vector(value[name], name)
    result['probe_start_s'] = _number(value['probe_start_s'], 'probe_start_s', .001, 30.)
    ceiling = value['per_finger_force_ceiling_n']
    result['per_finger_force_ceiling_n'] = None if ceiling is None else _number(ceiling, 'force ceiling', 0., 1e6)
    status = value['native_status']
    if type(status) is not dict or set(status) != {'valid', 'warnings', 'unexpected_contact_native_ticks', 'simulation_time_s'}:
        raise ProbeEvidenceError('Native status envelope fields differ.')
    if (type(status['valid']) is not bool or type(status['warnings']) is not list
            or len(status['warnings']) > 100
            or any(type(item) is not str or len(item) > 1000 for item in status['warnings'])
            or type(status['unexpected_contact_native_ticks']) is not int
            or not 0 <= status['unexpected_contact_native_ticks'] <= MAX_ROWS):
        raise ProbeEvidenceError('Malformed native validity envelope.')
    result['native_status']['simulation_time_s'] = _number(status['simulation_time_s'], 'simulation_time_s')
    if len(_canonical(result)) > MAX_JSON_BYTES:
        raise ProbeEvidenceError('Context exceeds its byte bound.')
    return result


def feasibility_binding(document):
    """Bind all planning assumptions; sensing flags are enforced separately."""
    task_feasibility.assess(document)
    if type(document) is not dict:
        raise ProbeEvidenceError('Feasibility binding requires a plain v1 dictionary.')
    return _digest({name: value for name, value in document.items()
                    if name not in ('test_lift_passed', 'observations_fresh', 'slip_detected')})


def _arrays(arrays, metadata):
    if type(arrays) is not dict or set(arrays) != {name for name, _ in FIELDS}:
        raise ProbeEvidenceError('Native sidecar fields differ from the fixed schema.')
    if type(metadata) is not dict or len(_canonical(metadata)) > MAX_JSON_BYTES:
        raise ProbeEvidenceError('Metadata must be a bounded plain dictionary.')
    if (metadata.get('schema') != 'dvidia.piper-feedback-native/v1'
            or metadata.get('native_timestep_s') != DT_S
            or metadata.get('physical_robot_ready') is not False
            or metadata.get('side_order') != ['left', 'right']
            or metadata.get('fields') != [{'name': name, 'width': width} for name, width in FIELDS]):
        raise ProbeEvidenceError('Unsupported native telemetry contract.')
    first = arrays['t_seconds']
    if not isinstance(first, np.ndarray) or first.ndim != 1 or not 1 <= first.size <= MAX_ROWS:
        raise ProbeEvidenceError('Expected 1..30000 native samples.')
    count = len(first)
    if type(metadata.get('rows')) is not int or metadata['rows'] != count:
        raise ProbeEvidenceError('Metadata row count differs.')
    result = {}
    digest = sha256()
    for name, width in FIELDS:
        value = arrays[name]
        shape = (count,) if width == 1 else (count, width)
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype('float64')
                or value.shape != shape or not np.isfinite(value).all()):
            raise ProbeEvidenceError('Expected finite float64 native array: '+name)
        # Copy once: a caller edit during verification cannot alter later reads.
        result[name] = np.array(value, dtype=np.float64, order='C', copy=True)
        digest.update(_canonical({'name': name, 'shape': list(shape), 'dtype': 'float64'}))
        digest.update(result[name].tobytes())
    ticks = np.arange(1, count+1, dtype=np.float64)
    if (not np.array_equal(result['native_tick'], ticks)
            or np.max(np.abs(result['t_seconds']-ticks*DT_S)) > 1e-8):
        raise ProbeEvidenceError('Native cadence must be contiguous exact 1 ms ticks from reset.')
    for name in ('grasp_contact_counts', 'table_object_contact_count', 'unexpected_contact_count'):
        values = result[name]
        if np.any(values < 0) or np.any(values > 10000) or np.any(values != np.floor(values)):
            raise ProbeEvidenceError('Contact counts must be bounded nonnegative integers.')
    if np.any(result['pad_normal_forces_n'] < 0) or np.any(result['pad_normal_forces_n'] > 1e6):
        raise ProbeEvidenceError('Native pad forces exceed the supported range.')
    for name in ('tcp_quaternion_wxyz', 'object_quaternion_wxyz'):
        if np.max(np.abs(np.linalg.norm(result[name], axis=1)-1.)) > 1e-8:
            raise ProbeEvidenceError('Native quaternion is not normalized.')
    # Check the local-offset field independently against world positions/rotation.
    q = result['tcp_quaternion_wxyz']
    vector = result['object_position_m']-result['tcp_position_m']
    xyz = -q[:, 1:]
    cross = np.cross(xyz, vector)
    local = vector+2*q[:, :1]*cross+2*np.cross(xyz, cross)
    if np.max(np.abs(local-result['object_minus_tcp_local_m'])) > 1e-9:
        raise ProbeEvidenceError('TCP-local offset disagrees with world pose.')
    return result, digest.hexdigest()


def _claims(value, context, times, qualified):
    if type(value) not in (list, tuple) or len(value) > 1500:
        raise ProbeEvidenceError('Controller qualifications must be a bounded sequence.')
    records, mismatches, prior = [], [], -1.
    for row in value:
        if type(row) is not dict or set(row) != {'t_seconds', 'qualified', 'probe_index'}:
            raise ProbeEvidenceError('Controller qualification fields differ.')
        timestamp = _number(row['t_seconds'], 'qualification time')
        if (type(row['qualified']) is not bool or type(row['probe_index']) is not int
                or row['probe_index'] != context['probe_index'] or timestamp <= prior):
            raise ProbeEvidenceError('Qualification order, boolean or probe binding differs.')
        tick = round(timestamp/DT_S)
        if (not 1 <= tick <= len(times) or abs(times[tick-1]-timestamp) > 1e-8
                or timestamp+1e-8 < context['probe_start_s']
                or abs(timestamp/.02-round(timestamp/.02)) > 1e-8):
            raise ProbeEvidenceError('Controller claims must use causal observed 20 ms boundaries.')
        record = {'t_seconds': timestamp, 'qualified': row['qualified'], 'probe_index': row['probe_index']}
        records.append(record)
        if row['qualified'] != bool(qualified[tick-1]):
            mismatches.append({'t_seconds': timestamp, 'controller_qualified': row['qualified'],
                               'native_prefix_qualified': bool(qualified[tick-1]),
                               'classification': ('false_qualification' if row['qualified']
                                                  else 'conservative_abstention')})
        prior = timestamp
    return records, mismatches


def derive_probe_evidence(arrays, metadata, context, *, controller_qualifications=()):
    """Derive one probe receipt from the whole causal native prefix supplied.

    Supply only the final selected probe's claims. Missing probes have no valid
    origins and should be retained as refused attempts without a receipt.
    """
    context = _context(context)
    data, array_hash = _arrays(arrays, metadata)
    times = data['t_seconds']
    start_tick = round(context['probe_start_s']/DT_S)
    if not 1 <= start_tick <= len(times) or abs(times[start_tick-1]-context['probe_start_s']) > 1e-8:
        raise ProbeEvidenceError('Probe origin is not an observed native sample.')
    origin = start_tick-1
    if (np.linalg.norm(data['object_position_m'][origin]-context['probe_origin_object_position_m']) > 1e-9
            or np.linalg.norm(data['object_minus_tcp_local_m'][origin]-context['probe_origin_tcp_local_offset_m']) > 1e-9):
        raise ProbeEvidenceError('Probe origin object/local pose does not match its native row.')
    if np.linalg.norm(data['object_position_m'][0]-context['initial_object_position_m']) > .001:
        raise ProbeEvidenceError('Initial object binding differs from the first post-reset sample.')
    if abs(context['native_status']['simulation_time_s']-times[-1]) > 1e-8:
        raise ProbeEvidenceError('Native status time differs from latest actual sample.')
    rise = data['object_position_m'][:, 2]-context['probe_origin_object_position_m'][2]
    drift = np.linalg.norm(data['object_minus_tcp_local_m']-context['probe_origin_tcp_local_offset_m'], axis=1)
    condition = ((np.arange(len(times)) > origin)
                 & np.all(data['grasp_contact_counts'] > 0, axis=1)
                 & np.all(data['pad_normal_forces_n'] > MIN_FORCE_N, axis=1)
                 & (data['table_object_contact_count'] == 0)
                 & (data['unexpected_contact_count'] == 0)
                 & (rise+1e-12 >= MIN_RISE_M) & (drift <= MAX_DRIFT_M+1e-12))
    streak = np.zeros(len(times), dtype=np.int64)
    count = 0
    for index, good in enumerate(condition):
        count = count+1 if good else 0
        streak[index] = count
    qualified = streak >= round(MIN_DWELL_S/DT_S)
    claims, mismatches = _claims(controller_qualifications, context, times, qualified)
    reasons = []
    status = context['native_status']
    critical_ticks = int(np.count_nonzero(data['unexpected_contact_count']))
    if not status['valid']:
        reasons.append('native_dynamics_invalid')
    if status['warnings']:
        reasons.append('native_solver_warnings')
    if critical_ticks or status['unexpected_contact_native_ticks']:
        reasons.append('unexpected_native_contact')
    if critical_ticks != status['unexpected_contact_native_ticks']:
        reasons.append('native_contact_total_disagrees')
    ceiling = context['per_finger_force_ceiling_n']
    maximum_force = float(np.max(data['pad_normal_forces_n']))
    if ceiling is not None and maximum_force > ceiling:
        reasons.append('declared_force_ceiling_exceeded')
    false_qualifications = [row for row in mismatches if row['controller_qualified']]
    abstentions = [row for row in mismatches if not row['controller_qualified']]
    if false_qualifications:
        reasons.append('controller_native_qualification_disagrees')
    first = np.flatnonzero(qualified)
    positive = [row for row in claims if row['qualified']]
    receipt = {'schema': SCHEMA, 'scope': SCOPE, 'physical_robot_ready': False,
               'context': context, 'context_sha256': _digest(context),
               'native_arrays_sha256': array_hash, 'metadata_sha256': _digest(metadata),
               'criteria': dict(CRITERIA), 'controller_qualifications': claims,
               'audit': {'valid': not reasons, 'reasons': reasons, 'rows': len(times),
                         'controller_claims_checked': len(claims),
                         'controller_positive_claims': len(positive),
                         'controller_claim_mismatches': mismatches,
                         'controller_false_qualification_count': len(false_qualifications),
                         'controller_conservative_abstention_count': len(abstentions),
                         'controller_coverage': 'Only the supplied claims are checked; omitted decisions are not inferred.',
                         'force_ceiling_known': ceiling is not None, 'maximum_observed_per_finger_force_n': maximum_force,
                         'unexpected_contact_native_ticks': critical_ticks},
               'history': {'ever_qualified': bool(first.size),
                           'first_qualified_s': float(times[first[0]]) if first.size else None,
                           'maximum_continuous_qualifying_s': int(np.max(streak))*DT_S},
               'latest': {'time_s': float(times[-1]), 'native_tick': len(times),
                          'condition_valid': bool(condition[-1]), 'qualified': bool(qualified[-1]),
                          'continuous_qualifying_s': int(streak[-1])*DT_S,
                          'rise_m': float(rise[-1]), 'tcp_local_drift_m': float(drift[-1]),
                          'controller_claim': deepcopy(claims[-1]) if claims else None}}
    receipt['receipt_sha256'] = _digest(receipt)
    if len(_canonical(receipt)) > MAX_JSON_BYTES:
        raise ProbeEvidenceError('Receipt exceeds its byte bound.')
    return receipt


def verify_probe_evidence(receipt, arrays, metadata, current_context, *, now_s):
    """Recompute proof and require the latest sample in the unchanged context."""
    verdict = {'schema': 'dvidia.probe-evidence-verification/v1', 'verified': False,
               'admissible': False, 'historical_qualified': False, 'latest_qualified': False,
               'physical_robot_ready': False, 'reasons': [], 'scope': SCOPE}
    try:
        current = _context(current_context)
        now = _number(now_s, 'now_s')
        if type(receipt) is not dict or len(_canonical(receipt)) > MAX_JSON_BYTES:
            raise ProbeEvidenceError('Receipt must be a bounded plain dictionary.')
        expected = derive_probe_evidence(arrays, metadata, receipt.get('context'),
                                         controller_qualifications=receipt.get('controller_qualifications'))
        if _canonical(receipt) != _canonical(expected):
            verdict['reasons'].append('receipt_or_payload_changed')
            return verdict
        verdict.update(verified=True, historical_qualified=expected['history']['ever_qualified'],
                       latest_qualified=expected['latest']['qualified'],
                       receipt_sha256=expected['receipt_sha256'])
        if _canonical(current) != _canonical(expected['context']):
            verdict['reasons'].append('probe_context_changed')
        latest = expected['latest']['time_s']
        if now < latest-1e-8:
            verdict['reasons'].append('future_evidence')
        elif now > latest+1e-8:
            verdict['reasons'].append('stale_evidence')
        verdict['reasons'].extend(expected['audit']['reasons'])
        if not expected['latest']['qualified']:
            verdict['reasons'].append('latest_native_window_not_qualified')
        if not expected['audit']['controller_positive_claims']:
            verdict['reasons'].append('no_controller_qualification_for_this_probe')
        claim = expected['latest']['controller_claim']
        if claim is None or abs(claim['t_seconds']-latest) > 1e-8:
            verdict['reasons'].append('latest_controller_claim_missing_or_stale')
        elif not claim['qualified']:
            # A previous admission cannot undo a later controller invalidation,
            # even if native contact subsequently happens to settle again.
            verdict['reasons'].append('latest_controller_not_qualified')
        verdict['admissible'] = not verdict['reasons']
    except (ProbeEvidenceError, KeyError, TypeError, ValueError, OverflowError) as exc:
        verdict['reasons'].append('malformed_evidence: '+str(exc))
    return verdict


def assess_with_probe(document, receipt, arrays, metadata, current_context, *, now_s):
    """Replace the caller lift flag with recomputed, context-bound native proof.

    All unknown assumptions and caller stale/slip vetoes remain conservative.
    The returned v2 wrapper does not change task_feasibility.py or its v1 API.
    """
    task_feasibility.assess(document)
    if type(document) is not dict:
        raise ProbeEvidenceError('Integration requires a plain feasibility v1 dictionary.')
    verdict = verify_probe_evidence(receipt, arrays, metadata, current_context, now_s=now_s)
    current = current_context if type(current_context) is dict else {}
    if current.get('feasibility_sha256') != feasibility_binding(document):
        verdict['admissible'] = False
        verdict['reasons'].append('planning_assumptions_changed')
    if (document['calibration_id'] != current.get('calibration_id')
            or document['per_finger_normal_force_budget_n'] != current.get('per_finger_force_ceiling_n')):
        verdict['admissible'] = False
        verdict['reasons'].append('calibration_or_force_ceiling_changed')
    checked = deepcopy(document)
    checked['test_lift_passed'] = verdict['admissible']
    gate = task_feasibility.assess(checked)
    return {'schema': 'dvidia.task-feasibility-with-probe/v1', 'status': gate['status'],
            'physical_robot_ready': False, 'probe': verdict, 'planning': gate,
            'observation_origin': 'privileged_native_simulation_plus_caller_planning_assumptions',
            'scope': SCOPE+' Unknown planning values remain unknown. The unchanged approximate vertical-lift '
                     'planning gate describes declared loads; it supplies no transport path, geospatial-goal '
                     'contract or actuator permission. This wrapper never issues controls.'}
