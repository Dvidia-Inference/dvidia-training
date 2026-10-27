"""Independent fixed-reference scoring for recorded synthetic PiPER episodes.

This validates trace shape and recorded outcomes. It does not authenticate the
episode source or recompute native physics/contact history from raw MuJoCo data.
"""
from __future__ import annotations

import math


CONTROL_DT_S = .02
TIME_TOLERANCE_S = 1e-8


def _finite_vector(value, size, low=-1., high=1.):
    if type(value) not in (list, tuple) or len(value) != size:
        return None
    if any(type(item) not in (int, float) or not math.isfinite(item) or not low <= item <= high
           for item in value):
        return None
    return [float(item) for item in value]


def _finite_number(value, low, high):
    return (type(value) in (int, float) and math.isfinite(value) and low <= value <= high)


def _failed(reason, **details):
    return {'success': False, 'reason': reason, **details}


def score_episode(episode, reference_xy):
    """Score one complete, dense PiPER trace against a fixed destination.

    Malformed and failed trials return a failure result rather than raising, so
    callers can retain every scheduled attempt in its denominator. ``episode``
    is expected to be a simulator-authored run_episode record; its recorded
    native metrics are evidence, not authenticated sensor data.
    """
    reference = _finite_vector(reference_xy, 2, -.5, .5)
    if reference is None:
        return _failed('invalid_reference', reference_xy_m=None)

    if type(episode) is not dict:
        return _failed('malformed_episode', reference_xy_m=reference)
    config = episode.get('config')
    actions = episode.get('actions')
    states = episode.get('trajectory')
    metrics = episode.get('metrics')
    if type(config) is not dict or type(actions) is not list or type(states) is not list or type(metrics) is not dict:
        return _failed('incomplete_episode_record', reference_xy_m=reference)

    control_dt = config.get('control_dt')
    timeout = config.get('timeout_s')
    tolerance = config.get('goal_position_tolerance_m')
    dwell_required = config.get('goal_dwell_s')
    minimum_lift = config.get('min_lift_m')
    lift_dwell_required = config.get('min_lift_dwell_s')
    retreat = config.get('retreat_clearance_m')
    if (not _finite_number(control_dt, .02, .02) or not _finite_number(timeout, .1, 30.)
            or not math.isclose(timeout / CONTROL_DT_S, round(timeout / CONTROL_DT_S), abs_tol=1e-9, rel_tol=0)
            or not _finite_number(tolerance, .01, .025) or not _finite_number(dwell_required, .1, .6)
            or not _finite_number(minimum_lift, .04, .1)
            or not _finite_number(lift_dwell_required, .05, .3) or not _finite_number(retreat, .08, .15)):
        return _failed('invalid_task_config', reference_xy_m=reference)

    expected_actions = round(timeout / CONTROL_DT_S)
    if len(actions) != expected_actions or len(states) != expected_actions + 1:
        return _failed('incomplete_trajectory', reference_xy_m=reference,
                       expected_actions=expected_actions, observed_actions=len(actions),
                       expected_states=expected_actions + 1, observed_states=len(states))

    initial_position = _finite_vector(episode.get('initial_object_position_m'), 3, -.5, .5)
    if initial_position is None:
        return _failed('missing_reset_record', reference_xy_m=reference)

    state_fields = ('t_seconds', 'object_position_m', 'table_object_contact_count',
                    'grasp_contact_counts', 'gripper_joint_position_m',
                    'object_linear_velocity_m_s', 'object_angular_velocity_rad_s', 'tcp_position_m')
    normalized = []
    for index, state in enumerate(states):
        if type(state) is not dict or any(name not in state for name in state_fields):
            return _failed('incomplete_state', reference_xy_m=reference, state_index=index)
        timestamp = state['t_seconds']
        expected_time = index * CONTROL_DT_S
        if (not _finite_number(timestamp, 0., timeout + CONTROL_DT_S)
                or not math.isclose(timestamp, expected_time, abs_tol=TIME_TOLERANCE_S, rel_tol=0)):
            return _failed('invalid_timestamp_cadence', reference_xy_m=reference, state_index=index,
                           expected_t_seconds=expected_time,
                           observed_t_seconds=timestamp if _finite_number(timestamp, 0., timeout + CONTROL_DT_S) else None)
        obj = _finite_vector(state['object_position_m'], 3, -.5, .5)
        linear = _finite_vector(state['object_linear_velocity_m_s'], 3, -100., 100.)
        angular = _finite_vector(state['object_angular_velocity_rad_s'], 3, -100., 100.)
        tcp = _finite_vector(state['tcp_position_m'], 3, -.5, .5)
        jaw = _finite_vector(state['gripper_joint_position_m'], 2, -.04, .045)
        contacts = state['grasp_contact_counts']
        table_count = state['table_object_contact_count']
        if (None in (obj, linear, angular, tcp, jaw)
                or type(contacts) is not dict or set(contacts) != {'left', 'right'}
                or any(type(v) is not int or v < 0 or v > 100 for v in contacts.values())
                or type(table_count) is not int or not 0 <= table_count <= 100):
            return _failed('invalid_state_value', reference_xy_m=reference, state_index=index)

        if index < expected_actions:
            action = actions[index]
            if type(action) is not dict or not _finite_number(action.get('t_seconds'), 0., timeout):
                return _failed('incomplete_action_record', reference_xy_m=reference, action_index=index)
            if not math.isclose(action['t_seconds'], expected_time, abs_tol=TIME_TOLERANCE_S, rel_tol=0):
                return _failed('action_timestamp_mismatch', reference_xy_m=reference, action_index=index)
            targets = _finite_vector(action.get('joint_targets_rad'), 6, -10., 10.)
            jaw_target = action.get('gripper_position_m')
            if targets is None or not _finite_number(jaw_target, 0., .035):
                return _failed('invalid_action_value', reference_xy_m=reference, action_index=index)
            if (state.get('issued_joint_targets_rad') != action['joint_targets_rad']
                    or state.get('issued_gripper_position_m') != jaw_target
                    or state.get('command') != action):
                return _failed('action_state_mismatch', reference_xy_m=reference, action_index=index)
        elif state.get('command') is not None or state.get('issued_joint_targets_rad') is not None or state.get('issued_gripper_position_m') is not None:
            return _failed('invalid_terminal_record', reference_xy_m=reference, state_index=index)

        normalized.append((obj, linear, angular, tcp, jaw, contacts, table_count))

    if any(abs(normalized[0][0][axis] - initial_position[axis]) > 1e-6 for axis in range(3)):
        return _failed('reset_state_mismatch', reference_xy_m=reference)

    native_boolean_fields = ('valid', 'success', 'lift_seen', 'grasp_seen',
                             'carry_complete', 'retention_lost')
    if (any(type(metrics.get(name)) is not bool for name in native_boolean_fields)
            or type(episode.get('valid')) is not bool or type(episode.get('success')) is not bool):
        return _failed('incomplete_native_metrics', reference_xy_m=reference)
    native_valid = metrics['valid'] is True and episode['valid'] is True
    native_success = metrics['success'] is True and episode['success'] is True
    lift_seen = metrics.get('lift_seen') is True
    grasp_seen = metrics.get('grasp_seen') is True
    carry_complete = metrics.get('carry_complete') is True
    retention_lost = metrics.get('retention_lost') is False
    unexpected = metrics.get('unexpected_contact_native_ticks')
    max_lift = metrics.get('max_object_lift_m')
    native_dwell = metrics.get('dwell_seconds')
    live_lift_dwell = metrics.get('lift_dwell_seconds')
    warnings = metrics.get('warnings')
    if (type(unexpected) is not int or unexpected < 0 or not _finite_number(max_lift, 0., 10.)
            or not _finite_number(native_dwell, 0., timeout)
            or not _finite_number(live_lift_dwell, 0., timeout)
            or type(warnings) is not list):
        return _failed('incomplete_native_metrics', reference_xy_m=reference)

    # lift_dwell_seconds is a live streak, so it resets after the block is
    # supported or released. Require the native lift_seen latch and independently
    # reconstruct its historical condition from dense control-boundary samples.
    lift_streak_start = None
    max_trace_lift_dwell = 0.
    for index, (obj, _, _, _, _, contacts, table_count) in enumerate(normalized):
        loaded = (contacts['left'] > 0 and contacts['right'] > 0 and table_count == 0
                  and obj[2] - initial_position[2] + TIME_TOLERANCE_S >= minimum_lift)
        if loaded:
            if lift_streak_start is None:
                lift_streak_start = index * CONTROL_DT_S
            max_trace_lift_dwell = max(max_trace_lift_dwell,
                                       index * CONTROL_DT_S - lift_streak_start)
        else:
            lift_streak_start = None
    trace_lift_seen = max_trace_lift_dwell + TIME_TOLERANCE_S >= lift_dwell_required
    native_gates = (native_valid and native_success and lift_seen and grasp_seen and carry_complete
                    and retention_lost and unexpected == 0 and max_lift >= minimum_lift
                    and native_dwell + TIME_TOLERANCE_S >= dwell_required
                    and trace_lift_seen and not warnings)

    streak_start = None
    for index, (obj, linear, angular, tcp, jaw, contacts, table_count) in enumerate(normalized):
        qualified = (math.dist(obj[:2], reference) <= tolerance and table_count > 0
                     and contacts['left'] == 0 and contacts['right'] == 0
                     and jaw[0] >= .028 and jaw[1] <= -.028
                     and math.sqrt(sum(v*v for v in linear)) <= .035
                     and math.sqrt(sum(v*v for v in angular)) <= .3
                     and tcp[2] - obj[2] >= retreat)
        if not qualified:
            streak_start = None
        elif streak_start is None:
            streak_start = index * CONTROL_DT_S
    terminal_span = (expected_actions * CONTROL_DT_S - streak_start) if streak_start is not None else 0.
    terminal_error = math.dist(normalized[-1][0][:2], reference)
    reference_success = streak_start is not None and terminal_span + TIME_TOLERANCE_S >= dwell_required
    success = bool(native_gates and reference_success)
    reason = ('success' if success else 'native_outcome_failed' if not native_gates else
              'terminal_reference_failed' if streak_start is None else 'terminal_dwell_insufficient')
    return {'success': success, 'reason': reason, 'reference_xy_m': reference,
            'terminal_reference_success': bool(reference_success), 'native_outcome_success': bool(native_gates),
            'native_outcome_valid': bool(native_valid), 'terminal_dwell_s': max(0., terminal_span),
            'required_dwell_s': float(dwell_required), 'terminal_xy_error_m': terminal_error,
            'trace_lift_dwell_s': max_trace_lift_dwell,
            'required_lift_dwell_s': float(lift_dwell_required),
            'sample_interval_s': CONTROL_DT_S, 'trajectory_complete': True,
            'native_metrics_source': 'simulator-authored episode record; not authenticated sensor input'}
