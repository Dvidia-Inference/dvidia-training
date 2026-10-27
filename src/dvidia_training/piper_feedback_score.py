"""Independent numeric scoring of native PiPER grip-feedback evidence.

This module imports no policy, fault injector or environment implementation. It
reconstructs task predicates and contact interruptions from a numeric sidecar;
it does not authenticate the simulator or rerun its physics. Contact restoration
is an observation, not evidence of a learned regrasp or completed task recovery.
"""
from __future__ import annotations

import math

import numpy as np

from .piper_imitation_score import score_episode


SCHEMA = 'dvidia.piper-feedback-score/v1'
NATIVE_SCHEMA = 'dvidia.piper-feedback-native/v1'
DT = .001
ESTABLISHED_S = .020
LOSS_S = .020
RESTORED_S = .100
FIELDS = (
    ('t_seconds', 1), ('native_tick', 1),
    ('object_position_m', 3), ('object_quaternion_wxyz', 4),
    ('object_linear_velocity_m_s', 3), ('object_angular_velocity_rad_s', 3),
    ('tcp_position_m', 3), ('tcp_quaternion_wxyz', 4),
    ('object_minus_tcp_local_m', 3),
    ('grasp_contact_counts', 2), ('pad_normal_forces_n', 2),
    ('table_object_contact_count', 1), ('unexpected_contact_count', 1),
    ('gripper_joint_position_m', 2), ('joint_position_rad', 6),
    ('arm_actuator_controls_rad', 6),
    ('requested_jaw_target_m', 1), ('normal_rate_limited_jaw_target_m', 1),
    ('applied_jaw_target_m', 1), ('pulse_active', 1),
    ('pulse_minimum_opening_m', 1), ('pulse_closing_tail', 1),
)
INTEGER_FIELDS = ('native_tick', 'grasp_contact_counts', 'table_object_contact_count',
                  'unexpected_contact_count', 'pulse_active', 'pulse_closing_tail')


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _failure(reason, strict, **details):
    return dict(schema=SCHEMA, valid=False, success=False, reason=reason,
                strict_v01=strict, **details)


def _ranges(mask):
    """Inclusive index intervals; each row represents one completed native tick."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    return list(zip(np.flatnonzero(np.diff(padded.astype(np.int8)) == 1).tolist(),
                    (np.flatnonzero(np.diff(padded.astype(np.int8)) == -1)-1).tolist()))


def _spans(mask):
    intervals = _ranges(mask)
    return dict(count=len(intervals), total_s=float(np.count_nonzero(mask)*DT),
                maximum_s=max(((last-first+1)*DT for first, last in intervals), default=0.),
                intervals=[dict(start_s=first*DT, end_s=(last+1)*DT,
                                duration_s=(last-first+1)*DT) for first, last in intervals])


def _local_from_pose(object_position, tcp_position, quaternion):
    w, x, y, z = quaternion.T
    rotation = np.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape((-1, 3, 3))
    return np.einsum('nji,nj->ni', rotation, object_position-tcp_position)


def _validated(episode, arrays, metadata, goal):
    if type(arrays) is not dict or set(arrays) != {name for name, _ in FIELDS}:
        raise ValueError('missing_or_unknown_native_field')
    if type(metadata) is not dict or metadata.get('schema') != NATIVE_SCHEMA:
        raise ValueError('invalid_native_metadata')
    if (metadata.get('fields') != [dict(name=name, width=width) for name, width in FIELDS]
            or metadata.get('side_order') != ['left', 'right']
            or not _number(metadata.get('native_timestep_s'), DT, DT)):
        raise ValueError('native_schema_or_cadence_mismatch')
    config = episode.get('config', {})
    timeout = config.get('timeout_s')
    if (not _number(timeout, .1, 30.) or not _number(config.get('timestep', DT), DT, DT)
            or not math.isclose(timeout/DT, round(timeout/DT), abs_tol=1e-8, rel_tol=0)):
        raise ValueError('invalid_native_task_config')
    expected = round(timeout/DT)
    if type(metadata.get('rows')) is not int or metadata['rows'] != expected:
        raise ValueError('incomplete_native_trajectory')
    values = {}
    for name, width in FIELDS:
        value = np.asarray(arrays[name])
        if (value.dtype.kind not in 'fiu' or value.shape != ((expected,) if width == 1 else (expected, width))
                or not np.isfinite(value).all()):
            raise ValueError('invalid_native_array:'+name)
        values[name] = value.astype(np.float64, copy=False)
    if (not np.array_equal(values['native_tick'], np.arange(1, expected+1))
            or not np.allclose(values['t_seconds'], np.arange(1, expected+1)*DT, atol=1e-8, rtol=0)):
        raise ValueError('invalid_native_timestamp_cadence')
    for name in INTEGER_FIELDS:
        value = values[name]
        maximum = expected if name == 'native_tick' else 1 if name.startswith('pulse_') else 1000
        if np.any(value != np.floor(value)) or np.any(value < 0) or np.any(value > maximum):
            raise ValueError('invalid_native_integer:'+name)
    for name in ('pad_normal_forces_n', 'requested_jaw_target_m', 'normal_rate_limited_jaw_target_m',
                 'applied_jaw_target_m', 'pulse_minimum_opening_m'):
        maximum = 1e6 if name == 'pad_normal_forces_n' else .035
        if np.any(values[name] < -1e-12) or np.any(values[name] > maximum+1e-12):
            raise ValueError('native_value_out_of_bounds:'+name)
    for name in ('object_quaternion_wxyz', 'tcp_quaternion_wxyz'):
        if not np.allclose(np.linalg.norm(values[name], axis=1), 1., atol=1e-6, rtol=0):
            raise ValueError('invalid_native_quaternion:'+name)
    local = _local_from_pose(values['object_position_m'], values['tcp_position_m'], values['tcp_quaternion_wxyz'])
    if not np.allclose(local, values['object_minus_tcp_local_m'], atol=2e-7, rtol=0):
        raise ValueError('native_relative_pose_mismatch')
    for name in ('object_position_m', 'tcp_position_m', 'object_minus_tcp_local_m'):
        if np.any(np.abs(values[name]) > 10.):
            raise ValueError('native_value_out_of_bounds:'+name)
    target = np.asarray(config.get('target_position_m', [*goal, .08]), dtype=float)
    initial = np.asarray(episode.get('initial_object_position_m'), dtype=float)
    if (target.shape != (3,) or initial.shape != (3,) or not np.isfinite(target).all()
            or not np.isfinite(initial).all() or not np.allclose(target[:2], goal, atol=1e-9, rtol=0)):
        raise ValueError('native_reference_mismatch')
    states = episode.get('trajectory')
    actions = episode.get('actions')
    if type(states) is not list or type(actions) is not list or len(states) != len(actions)+1 or len(actions)*20 != expected:
        raise ValueError('native_control_trace_mismatch')
    bindings = ('object_position_m', 'object_quaternion_wxyz', 'object_linear_velocity_m_s',
                'object_angular_velocity_rad_s', 'tcp_position_m', 'tcp_quaternion_wxyz',
                'gripper_joint_position_m', 'joint_position_rad')
    boundary_rows = np.arange(19, expected, 20)
    for name in bindings:
        control = np.asarray([state.get(name) for state in states[1:]], dtype=float)
        if control.shape != values[name][boundary_rows].shape or not np.allclose(
                control, values[name][boundary_rows], atol=2e-7, rtol=0):
            raise ValueError('native_control_state_mismatch:'+name)
    for name in ('grasp_contact_counts', 'pad_normal_forces_n'):
        control = np.asarray([[state.get(name, {}).get(side) for side in ('left', 'right')]
                              for state in states[1:]], dtype=float)
        if control.shape != values[name][boundary_rows].shape or not np.allclose(
                control, values[name][boundary_rows], atol=2e-7, rtol=0):
            raise ValueError('native_control_state_mismatch:'+name)
    for name in ('table_object_contact_count', 'unexpected_contact_count'):
        control = np.asarray([state.get(name) for state in states[1:]], dtype=float)
        if not np.array_equal(control, values[name][boundary_rows]):
            raise ValueError('native_control_state_mismatch:'+name)
    requested = np.repeat([action.get('gripper_position_m') for action in actions], 20)
    if not np.allclose(requested, values['requested_jaw_target_m'], atol=1e-12, rtol=0):
        raise ValueError('native_requested_jaw_mismatch')
    return values, target, initial


def _native_task(values, config, target, initial):
    counts, force = values['grasp_contact_counts'], values['pad_normal_forces_n']
    both = np.all(counts > 0, axis=1)
    loaded_bilateral = both & np.all(force > .01, axis=1)
    supported = values['table_object_contact_count'] > 0
    critical = values['unexpected_contact_count'] > 0
    lift = values['object_position_m'][:, 2]-initial[2]
    distance = np.linalg.norm(values['object_position_m']-target, axis=1)
    jaw = values['gripper_joint_position_m']
    released = np.all(counts == 0, axis=1) & (jaw[:, 0] > .028) & (jaw[:, 1] < -.028)
    settled = ((np.linalg.norm(values['object_linear_velocity_m_s'], axis=1) < .035)
               & (np.linalg.norm(values['object_angular_velocity_rad_s'], axis=1) < .3))
    clear = values['tcp_position_m'][:, 2] >= values['object_position_m'][:, 2]+config['retreat_clearance_m']
    lift_seen = grasp_seen = carry_complete = retention_lost = False
    lift_dwell = dwell = loss = maximum_loss = 0.
    critical_ticks = loss_events = 0
    first_success = None
    lift_history, carry_history = [], []
    for index in range(len(both)):
        grasp_seen |= bool(both[index])
        critical_ticks += int(critical[index])
        lift_dwell = lift_dwell+DT if both[index] and not supported[index] and lift[index] >= config['min_lift_m'] else 0.
        lift_seen |= lift_dwell+1e-12 >= config['min_lift_dwell_s']
        if lift_seen and not carry_complete:
            carry_complete |= bool(both[index] and supported[index] and distance[index] <= config['goal_position_tolerance_m'])
            if not carry_complete:
                if both[index]:
                    loss = 0.
                else:
                    before, loss = loss, loss+DT
                    maximum_loss = max(maximum_loss, loss)
                    if before+1e-12 < .04 <= loss+1e-12:
                        loss_events += 1
                        retention_lost = True
        qualified = (lift_seen and grasp_seen and carry_complete and not retention_lost
                     and released[index] and supported[index] and settled[index] and clear[index]
                     and distance[index] <= config['goal_position_tolerance_m'] and critical_ticks == 0)
        dwell = dwell+DT if qualified else 0.
        if dwell+1e-12 >= config['goal_dwell_s'] and first_success is None:
            first_success = float(values['t_seconds'][index])
        lift_history.append(bool(lift_seen))
        carry_history.append(bool(carry_complete))
    terminal = ((distance <= config['goal_position_tolerance_m']) & supported & released & settled & clear)
    terminal_streak = 0
    for value in terminal[::-1]:
        if not value:
            break
        terminal_streak += 1
    # Before intended supported destination contact, any table touchdown or
    # actual open-jaw interval after lift is independently visible.
    prior_lift = np.concatenate(([False], np.asarray(lift_history[:-1])))
    before_carry = ~np.asarray(carry_history)
    premature_open = prior_lift & before_carry & ((jaw[:, 0] > .028) & (jaw[:, 1] < -.028))
    touchdowns = prior_lift & before_carry & supported
    native = dict(success=bool(carry_complete and not retention_lost and dwell+1e-12 >= config['goal_dwell_s']),
                  lift_seen=bool(lift_seen), grasp_seen=bool(grasp_seen), carry_complete=bool(carry_complete),
                  retention_lost=bool(retention_lost), unexpected_contact_native_ticks=critical_ticks,
                  max_object_lift_m=max(0., float(np.max(lift))),
                  max_pad_normal_force_n=float(np.max(force)), dwell_seconds=dwell, lift_dwell_seconds=lift_dwell,
                  max_carry_contact_loss_seconds=maximum_loss, carry_contact_loss_events=loss_events,
                  first_success_time_seconds=first_success)
    terminal_result = dict(success=terminal_streak*DT+1e-12 >= config['goal_dwell_s'],
                           dwell_s=terminal_streak*DT, terminal_xy_error_m=float(np.linalg.norm(values['object_position_m'][-1, :2]-target[:2])),
                           scope='Terminal-only geometry/release/settling/dwell; excludes lift, carry and retention history.')
    diagnostics = dict(premature_open_jaw=_spans(premature_open),
                       support_touchdown_before_destination=_spans(touchdowns))
    return native, terminal_result, diagnostics, loaded_bilateral, supported, ~np.asarray(carry_history)


def _grip_events(values, loaded, supported, monitored):
    """Detect actual contact loss without reading injected faults or decisions."""
    times = values['t_seconds']
    established = False
    loaded_ticks = loss_start = restoration_start = 0
    pending = None
    events = []
    for index, is_loaded in enumerate(loaded):
        if not monitored[index]:
            if pending is not None:
                pending['censored_reason'] = 'supported_destination_reached_before_qualification'
                events.append(pending)
                pending = None
            break
        if is_loaded:
            loaded_ticks += 1
            if loaded_ticks*DT+1e-12 >= ESTABLISHED_S:
                established = True
            if pending is not None:
                if restoration_start == 0:
                    restoration_start = index+1
                if (index+2-restoration_start)*DT+1e-12 >= RESTORED_S:
                    start = restoration_start-1
                    pending.update(restoration_onset_s=float(times[start]),
                                   restoration_qualified_s=float(times[index]),
                                   onset_to_restoration_s=float(times[start]-pending['loss_onset_s']),
                                   onset_to_qualified_restoration_s=float(times[index]-pending['loss_onset_s']),
                                   restored=True, restoration_supported=bool(supported[start]))
                    events.append(pending)
                    pending = None
            loss_start = 0
        else:
            loaded_ticks = 0
            restoration_start = 0
            if established:
                if loss_start == 0:
                    loss_start = index+1
                if pending is None and (index+2-loss_start)*DT+1e-12 >= LOSS_S:
                    start = loss_start-1
                    pending = dict(loss_onset_s=float(times[start]), loss_qualified_s=float(times[index]),
                                   kind='supported_grip_interruption' if supported[start] else 'airborne_contact_loss',
                                   supported_at_onset=bool(supported[start]), restored=False,
                                   restoration_onset_s=None, restoration_qualified_s=None,
                                   onset_to_restoration_s=None, onset_to_qualified_restoration_s=None,
                                   relative_motion_since_loss_m=None)
            if pending is not None:
                origin = int(round(pending['loss_onset_s']/DT))-1
                pending['relative_motion_since_loss_m'] = max(
                    pending['relative_motion_since_loss_m'] or 0.,
                    float(np.linalg.norm(values['object_minus_tcp_local_m'][index]-values['object_minus_tcp_local_m'][origin])))
    if pending is not None:
        events.append(pending)
    return events


def score_feedback_episode(episode, native_arrays, native_metadata, goal_xy):
    """Return failure records for malformed/partial attempts, retaining them.

    ``success`` requires the unchanged strict v0.1 endpoint, a complete consistent
    native sidecar, and independently reconstructed native success. No retention
    latch is cleared. ``terminal_only`` and contact restoration are diagnostics.
    """
    strict = score_episode(episode, goal_xy)
    try:
        goal = np.asarray(goal_xy, dtype=float)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError('invalid_reference')
        values, target, initial = _validated(episode, native_arrays, native_metadata, goal)
        native, terminal, diagnostics, loaded, supported, monitored = _native_task(values, episode['config'], target, initial)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        return _failure(str(exc)[:180], strict)
    events = _grip_events(values, loaded, supported, monitored)
    metrics = episode.get('metrics', {})
    mismatches = []
    for name, expected in native.items():
        actual = metrics.get(name)
        if type(expected) in (bool, int) or expected is None:
            match = type(actual) is type(expected) and actual == expected
        else:
            match = _number(actual, -1e6, 1e6) and math.isclose(actual, expected, abs_tol=1e-8, rel_tol=0)
        if not match:
            mismatches.append(name)
    physics_valid = (episode.get('valid') is True and metrics.get('valid') is True
                     and metrics.get('warnings') == [])
    valid = bool(physics_valid and not mismatches)
    success = bool(valid and strict['success'] and native['success'])
    latencies = [row['onset_to_qualified_restoration_s'] for row in events if row['restored']]
    return dict(schema=SCHEMA, valid=valid, success=success,
                reason='success' if success else 'native_metrics_mismatch' if mismatches else 'invalid_native_physics' if not physics_valid else strict['reason'],
                strict_v01=strict, native_recomputed=native, native_metrics_mismatches=mismatches,
                terminal_only=terminal, grip_events=events,
                grip_summary=dict(loss_events=len(events), restored_events=sum(row['restored'] for row in events),
                                  unresolved_events=sum(not row['restored'] for row in events),
                                  supported_events=sum(row['supported_at_onset'] for row in events),
                                  airborne_events=sum(not row['supported_at_onset'] for row in events),
                                  qualification_latency_s=latencies),
                native_diagnostics=diagnostics,
                false_alarms=dict(status='not_applicable', reason='Continuous residual policy has no declared discrete alarm events.'),
                evidence_scope='Recomputed simulator-authored numeric evidence; no physics rerun, source attestation or hardware sensing.',
                restoration_scope='Loaded contact restored for 100 ms after sustained loss; does not establish task recovery or airborne regrasp.')
