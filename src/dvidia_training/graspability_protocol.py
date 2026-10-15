"""Independent evidence scoring for direct, in-reach object grasp diagnostics.

This module does not import or execute a controller. A successful easy-object
lift is a diagnostic result, never evidence of shoe retrieval or a learned skill.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

SCHEMA = 'dvidia.graspability/v3'
SCORE_SCHEMA = 'dvidia.graspability-score/v3'
FIXTURE = Path(__file__).with_name('graspability_protocol_v03.json')
FIXTURE_SHA256 = 'c33ef6991379fda77d6084931d2b4b3665830ffa448f9ea403a65d2c37b757a3'
SHAPES = ('easy-block', 'simplified-shoe', 'current-shoe')
TOOL_CONDITIONS = ('mounted', 'absent')
POLICY = 'shared-v03'
NATIVE_DT = .002
RETENTION_TICKS = 250
REFERENCE_TICKS = 50
REFERENCE_TIME_S = .4
LIFT_MIN_M = .035
FORCE_MIN_N = .01
_REQUIRED = (
    't_s', 'tool_object_force_n', 'object_robot_force_n',
    'left_finger_force_n', 'right_finger_force_n',
    'object_x_m', 'object_y_m', 'object_z_m', 'world_object_force_n',
)


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Graspability evidence requires finite numeric values')
    return float(value)


def _trace(result):
    columns = result.get('native_trace_columns')
    raw = result.get('native_trace')
    if (not isinstance(columns, list)
            or not all(isinstance(name, str) for name in columns)
            or len(columns) != len(set(columns))
            or not set(_REQUIRED) <= set(columns)
            or not isinstance(raw, list) or not 1 <= len(raw) <= 100000):
        raise ValueError('Missing bounded dense object trace or required columns')
    rows = []
    for sample in raw:
        if not isinstance(sample, (list, tuple)) or len(sample) != len(columns):
            raise ValueError('Malformed native object trace row')
        row = dict(zip(columns, (_number(value) for value in sample)))
        if any(row[key] < 0 for key in _REQUIRED if key.endswith('_force_n')):
            raise ValueError('Negative contact normal force')
        other = (row['object_robot_force_n'] - row['tool_object_force_n']
                 - row['left_finger_force_n'] - row['right_finger_force_n'])
        if other < -1e-6:
            raise ValueError('Inconsistent object contact-force decomposition')
        row['other_robot_force_n'] = max(0., other)
        rows.append(row)
    times = [row['t_s'] for row in rows]
    if abs(times[0]) > 1e-8 and abs(times[0] - NATIVE_DT) > 1e-8:
        raise ValueError('Trace omits its initial native steps')
    if any(abs(b - a - NATIVE_DT) > 1e-8 for a, b in zip(times, times[1:])):
        raise ValueError('Native trace has a missing, duplicated or nonuniform timestep')
    metrics = result.get('metrics', {})
    end = _number(metrics.get('simulation_seconds', metrics.get('simulated_seconds')))
    if abs(times[-1] - end) > 1e-6:
        raise ValueError('Dense trace does not reach the declared episode end')
    steps = metrics.get('native_steps')
    if type(steps) is not int or steps != round(end / NATIVE_DT):
        raise ValueError('Native step count disagrees with trace duration')
    if len(rows) != steps + int(abs(times[0]) <= 1e-8):
        raise ValueError('Native trace coverage is incomplete')
    if type(result.get('valid')) is not bool:
        raise ValueError('Native validity must be explicit')
    return rows


def _reference(result, rows):
    events = result.get('events', [])
    if not isinstance(events, list):
        raise ValueError('Events must be a list')
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get('type'), str):
            raise ValueError('Malformed graspability event')
        if not 0 <= _number(event.get('t')) <= rows[-1]['t_s'] + 1e-6:
            raise ValueError('Event lies outside recorded episode')
    references = [event for event in events if event['type'] == 'settled_reference']
    if len(references) > 1:
        raise ValueError('Reference cannot be reset after an attempted grasp')
    if not references:
        return None
    event = references[0]
    index = next((i for i, row in enumerate(rows)
                  if abs(row['t_s'] - event['t']) < 1e-8), None)
    if index is None:
        raise ValueError('Settled reference must identify a recorded native timestep')
    window = rows[max(0, index - REFERENCE_TICKS + 1):index + 1]
    ranges = [max(row[key] for row in window) - min(row[key] for row in window)
              for key in ('object_x_m', 'object_y_m', 'object_z_m')]
    supported = all(row['world_object_force_n'] > FORCE_MIN_N for row in window)
    clear = all(row['object_robot_force_n'] < FORCE_MIN_N for row in window)
    stable = max(ranges) <= .001
    complete = len(window) == REFERENCE_TICKS
    at_initial_time = abs(rows[index]['t_s'] - REFERENCE_TIME_S) <= 1e-8
    return {'t_s': rows[index]['t_s'], 'object_z_m': rows[index]['object_z_m'],
            'source': 'Recorded native row; event scalar coordinates are not trusted.',
            'window_ticks': len(window), 'window_duration_s': REFERENCE_TICKS * NATIVE_DT,
            'axis_position_ranges_m': ranges, 'world_supported': supported,
            'robot_clear': clear, 'position_stable': stable,
            'at_declared_initial_time': at_initial_time,
            'valid': bool(complete and supported and clear and stable and at_initial_time)}


def _predicate(row, reference_z):
    return bool(
        row['object_z_m'] - reference_z >= LIFT_MIN_M
        and row['left_finger_force_n'] > FORCE_MIN_N
        and row['right_finger_force_n'] > FORCE_MIN_N
        and row['world_object_force_n'] < FORCE_MIN_N
        and row['tool_object_force_n'] < FORCE_MIN_N
        and row['other_robot_force_n'] < FORCE_MIN_N)


def score_result(result):
    """Score every native tick; controller success and reference z are unused.

    Short or unsettled episodes remain retained attempts, with a failed lift.
    Corrupt or incomplete evidence raises ValueError rather than scoring a pass.
    """
    rows = _trace(result)
    reference = _reference(result, rows)
    final = rows[-RETENTION_TICKS:]
    end = rows[-1]['t_s']
    complete = bool(reference and len(final) == RETENTION_TICKS
                    and end - reference['t_s'] >= RETENTION_TICKS * NATIVE_DT - 1e-8)
    reference_z = reference['object_z_m'] if reference else None
    passing = sum(_predicate(row, reference_z) for row in final) if reference else 0
    failures = []
    if not result['valid']:
        failures.append('native_invalid')
    if reference is None:
        failures.append('settled_reference_missing')
    elif not reference['valid']:
        failures.append('initial_reference_unsettled')
    if not complete:
        failures.append('retention_window_incomplete')
    counts = {
        'world_supported_ticks': sum(row['world_object_force_n'] >= FORCE_MIN_N for row in final),
        'tool_supported_ticks': sum(row['tool_object_force_n'] >= FORCE_MIN_N for row in final),
        'other_robot_supported_ticks': sum(row['other_robot_force_n'] >= FORCE_MIN_N for row in final),
        'bilateral_contact_missing_ticks': sum(min(row['left_finger_force_n'], row['right_finger_force_n']) <= FORCE_MIN_N for row in final),
        'relative_lift_failed_ticks': sum(row['object_z_m'] - reference_z < LIFT_MIN_M for row in final) if reference else len(final),
    }
    for key, label in (('world_supported_ticks', 'world_support'),
                       ('tool_supported_ticks', 'tool_support'),
                       ('other_robot_supported_ticks', 'other_robot_support'),
                       ('bilateral_contact_missing_ticks', 'bilateral_contact'),
                       ('relative_lift_failed_ticks', 'relative_lift')):
        if counts[key]:
            failures.append(label)
    strict = bool(reference and reference['valid'] and complete and passing == RETENTION_TICKS)
    after_reference = [row for row in rows if reference and row['t_s'] >= reference['t_s']]
    score = {'schema': SCORE_SCHEMA, 'valid': result['valid'],
             'task_success': bool(result['valid'] and strict),
             'strict_retention_pass': strict, 'reference': reference,
             'final_window_complete': complete, 'final_window_ticks': len(final),
             'required_final_window_s': RETENTION_TICKS * NATIVE_DT,
             'final_window_start_exclusive_s': end - RETENTION_TICKS * NATIVE_DT,
             'final_window_end_s': end, 'strict_passing_ticks': passing,
             'relative_lift_min_m': LIFT_MIN_M,
             'terminal_relative_lift_m': rows[-1]['object_z_m'] - reference_z if reference else None,
             'maximum_relative_lift_m': max(row['object_z_m'] - reference_z for row in after_reference) if after_reference else None,
             'minimum_final_relative_lift_m': min(row['object_z_m'] - reference_z for row in final) if reference else None,
             'failed_criteria': failures,
             'task_scope': 'Direct in-reach pickup; no retrieval, tool acquisition, learned skill or hardware transfer claim.',
             'height_scope': 'Object body-origin displacement from a verified initial settled native reference. Lowest collision surface is a separate diagnostic, not the success threshold.',
             'support_scope': 'Finger normal forces each >0.01 N; world, tool and non-finger robot normal forces each <0.01 N on all final 250 native ticks. Synthetic scoring thresholds, not hardware safety limits.'}
    score.update(counts)
    if 'object_lowest_z_m' in rows[0]:
        score['minimum_final_collision_surface_z_m'] = min(row['object_lowest_z_m'] for row in final)
    return score


def audit_result(result):
    """Flag controller claim disagreement while returning the independent score."""
    score = score_result(result)
    claimed = result.get('success')
    if type(claimed) is not bool:
        raise ValueError('Controller success must be an explicit boolean')
    score['controller_reported_success'] = claimed
    score['controller_claim_agrees'] = claimed == score['task_success']
    return score


def _fixture():
    raw = FIXTURE.read_bytes()
    if sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError('Sealed graspability protocol changed; declare a new protocol revision')
    value = json.loads(raw)
    if value.get('schema') != 'dvidia.graspability-protocol/v3':
        raise ValueError('Unsupported graspability protocol')
    ids = set()
    bounds = {'object_x': (.38, .44), 'object_y': (-.12, -.06), 'friction': (.1, 1.5)}
    loads = {'lightweight-control': .03, 'reference-load': .30}
    for split in ('development', 'heldout'):
        cases = value['cases'][split]
        if not isinstance(cases, list) or len(cases) != (2 if split == 'development' else 4):
            raise ValueError('Unexpected factorial cohort size')
        replicates = {}
        for case in cases:
            identifier = case.get('id')
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError('Case IDs must be nonempty and unique across splits')
            ids.add(identifier)
            if case.get('load_band') not in loads or _number(case.get('mass')) != loads[case['load_band']]:
                raise ValueError('Case load must match a declared explicit load band')
            for key, (low, high) in bounds.items():
                if not low <= _number(case.get(key)) <= high:
                    raise ValueError('Case parameter outside declared study bounds: ' + key)
            replicate = case.get('replicate_id')
            if not isinstance(replicate, str) or not replicate:
                raise ValueError('Case requires a layout replicate ID')
            replicates.setdefault(replicate, []).append(case)
        for paired in replicates.values():
            if len(paired) != 2 or {row['load_band'] for row in paired} != set(loads):
                raise ValueError('Each layout replicate must contain both load bands')
            if any(paired[0][key] != paired[1][key] for key in bounds):
                raise ValueError('Load pair changed layout or friction')
    expected = {'native_dt_s': NATIVE_DT, 'policy_dt_s': .02,
                'initial_settle_s': REFERENCE_TIME_S, 'reference_window_s': .1,
                'reference_axis_range_max_m': .001,
                'final_retention_s': .5, 'final_retention_ticks': RETENTION_TICKS,
                'relative_origin_lift_min_m': LIFT_MIN_M,
                'finger_force_min_n': FORCE_MIN_N, 'support_force_max_n': FORCE_MIN_N}
    if any(value['criteria'].get(key) != number for key, number in expected.items()):
        raise ValueError('Fixture criteria disagree with independent scorer')
    return value


def protocol(split='development'):
    """Expand a sealed factorial; loading a plan never simulates a case."""
    if split not in ('development', 'heldout'):
        raise ValueError('split must be development or heldout')
    value = _fixture()
    attempts = []
    for case in value['cases'][split]:
        for shape in SHAPES:
            for tool_condition in TOOL_CONDITIONS:
                expanded = dict(deepcopy(case), object_shape=shape, tool_condition=tool_condition, split=split)
                attempts.append({'id': case['id'] + '-' + shape + '-' + tool_condition,
                                 'task': 'in-reach-graspability', 'policy': POLICY, 'case': expanded})
    return {'schema': SCHEMA, 'split': split, 'fixture_sha256': FIXTURE_SHA256,
            'cohort': value['cohorts'][split], 'attempts': attempts,
            'criteria': deepcopy(value['criteria']), 'design': value['design'],
            'comparison': value['comparison'], 'cohort_access': value['cohort_access'],
            'scope': value['scope']}
