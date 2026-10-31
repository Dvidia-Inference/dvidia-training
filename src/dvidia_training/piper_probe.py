"""Bounded native PiPER acquisition followed by a measured-position hold.

Both variants inherit the authored guarded approach/closure/supported retry.
The evidence variant admits a sampled 100 ms loaded probe; the timer variant
admits after a fixed 600 ms probe clock, without testing probe evidence first.
Neither dispatches the teacher's full lift or transport. Every observation is
privileged simulator state, not calibrated physical telemetry. A held position
target is not an instantaneous physical stop or a force limiter.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math

from . import piper_grip_guard as guard


SCHEMA = 'dvidia.piper-probe-stop/v1'
VARIANTS = ('evidence', 'timer')
ORIGIN = 'privileged_native_simulator_state'
TIMER_ADMISSION_S = .60
DT = .02


def probe_contract():
    return {
        'schema': SCHEMA, 'variants': list(VARIANTS), 'parameters': asdict(guard.GuardConfig()),
        'timer_admission_s': TIMER_ADMISSION_S, 'control_dt_s': DT,
        'observation_origin': ORIGIN, 'physical_robot_ready': False,
        'prefix': 'Original GuardedPiperTaskPolicy approach/close and at most one supported open/recenter/retry; already uses authored native state, not a fully blind baseline.',
        'evidence_admission': 'Current bilateral force >0.01 N, no table support, object rise >=6 mm and TCP-local offset drift <=3 mm continuously across sampled boundaries for elapsed >=100 ms.',
        'timer_admission': 'Same 10 mm/20 mm/s probe target law, admitted at elapsed >=600 ms from that probe start; no evidence predicate or slip alarm before admission. Unexpected contacts and bounded timeouts still stop.',
        'hold': 'Latch measured six arm joints and the last issued jaw target at admission/terminal invalidity. No full lift/transfer, no squeeze escalation, no post-admission retry.',
        'invalidation': 'At each subsequent 20 ms observation and final observation, invalidate immediately on lost contact/force, support, insufficient rise, >3 mm local drift or unexpected contact. Never requalify after terminal invalidity.',
        'sampling_limit': 'Independent 1 ms evidence may contradict sampled admission; that remains a finding. No sensor or material inference is implemented.',
        'stop_scope': 'Holding native position targets is not physical arrest or calibrated force limiting; inertia and existing target rate limits can continue motion.',
        'comparison_scope': 'Closed-loop variants may acquire different trajectories through early hold/retry. This is not an isolated classifier ablation.',
    }


def _number(value, name, low=-1e6, high=1e6):
    if type(value) not in (int, float):
        raise ValueError(name+' must be a finite real scalar')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name+' must be finite') from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(name+' outside bounded range')
    return number


def _vector(value, name, length, low=-1e3, high=1e3):
    if type(value) not in (list, tuple) or len(value) != length:
        raise ValueError(name+' has the wrong shape')
    return [_number(item, name, low, high) for item in value]


def _observation(value):
    if type(value) is not dict:
        raise ValueError('Probe observation must be a dictionary')
    # MuJoCo's repeated 1 ms additions can end at 20.00000000000146.
    # Match the episode-clock tolerance without admitting another native tick.
    _number(value.get('t_seconds'), 't_seconds', 0., 20.+1e-8)
    for name, width in (('joint_position_rad', 6), ('tcp_position_m', 3),
                        ('object_position_m', 3), ('target_position_m', 3),
                        ('gripper_joint_position_m', 2)):
        _vector(value.get(name), name, width)
    quaternion = _vector(value.get('tcp_quaternion_wxyz'), 'tcp_quaternion_wxyz', 4, -1., 1.)
    if not math.isclose(sum(item*item for item in quaternion), 1., rel_tol=0., abs_tol=1e-8):
        raise ValueError('Probe TCP quaternion must be normalized wxyz')
    for name in ('table_object_contact_count', 'unexpected_contact_count'):
        if type(value.get(name)) is not int or not 0 <= value[name] <= 10000:
            raise ValueError(name+' must be a nonnegative integer')
    counts, forces = value.get('grasp_contact_counts'), value.get('pad_normal_forces_n')
    if type(counts) is not dict or type(forces) is not dict or set(counts) != {'left', 'right'} or set(forces) != set(counts):
        raise ValueError('Probe requires explicit left/right contact and force readings')
    for side in counts:
        if type(counts[side]) is not int or not 0 <= counts[side] <= 10000:
            raise ValueError('Contact counts must be nonnegative integers')
        _number(forces[side], 'pad force', 0., 1e6)


class ProbeStopPolicy(guard.GuardedPiperTaskPolicy):
    """Reuse original acquisition while preventing any full-lift dispatch."""

    def __init__(self, environment, variant='evidence'):
        if type(variant) is not str or variant not in VARIANTS:
            raise ValueError('Probe variant must be evidence or timer')
        control_dt = _number(environment.config.control_dt, 'control_dt', DT, DT)
        timeout = _number(environment.config.timeout_s, 'timeout_s', 1., 20.)
        if not math.isclose(timeout/control_dt, round(timeout/control_dt), rel_tol=0., abs_tol=1e-8):
            raise ValueError('Probe timeout must contain complete control boundaries')
        super().__init__(environment)
        self.variant, self.status = variant, 'acquiring'
        self.timeout_s, self.command_budget = timeout, round(timeout/control_dt)
        self.admitted, self.ever_admitted, self.admission_time_s = False, False, None
        self.last_requested, self.frozen_jaw_target = None, None
        self.initial_context, self.probes, self.active_probe_index = None, [], None
        self.requested_controls, self.qualification_checks = [], []
        self.last_observed_s, self.final_observed = None, False

    def _clock(self, observation):
        _observation(observation)
        now = float(observation['t_seconds'])
        expected = 0. if self.last_observed_s is None else self.last_observed_s+DT
        if not math.isclose(now, expected, rel_tol=0., abs_tol=1e-8):
            raise ValueError('Probe requires causal consecutive 20 ms observations starting at zero')
        if now > self.timeout_s+1e-8:
            raise ValueError('Probe observation exceeds its fixed episode budget')
        self.last_observed_s = now
        if self.initial_context is None:
            self.initial_context = {'t_seconds': now, 'object_position_m': list(observation['object_position_m']),
                                    'tcp_position_m': list(observation['tcp_position_m']),
                                    'tcp_quaternion_wxyz': list(observation['tcp_quaternion_wxyz']),
                                    'observation_origin': ORIGIN}

    def _jaw_to_freeze(self, observation):
        if self.last_requested is not None:
            return self.last_requested['gripper_position_m']
        controls = observation.get('native_actuator_controls')
        if controls is not None:
            controls = _vector(controls, 'native_actuator_controls', 7)
            return _number(controls[6], 'jaw control', 0., .035)
        # At the first observation no policy target exists yet. Use the explicit
        # native single-finger coordinate, not a synthesized full jaw gap.
        return _number(observation['gripper_joint_position_m'][0], 'initial jaw coordinate', 0., .035)

    def _hold(self, jaw=None):
        if self.stop_targets is None:
            raise ValueError('Measured stop targets have not been latched')
        if jaw is None:
            jaw = (self.frozen_jaw_target if self.frozen_jaw_target is not None else
                   self.last_requested['gripper_position_m'] if self.last_requested is not None else .035)
        return {'joint_targets_rad': list(self.stop_targets), 'gripper_position_m': jaw}

    def _event(self, kind, observation, reason):
        if kind == 'probe_started':
            self.active_probe_index = len(self.probes)
            self.probes.append({
                'probe_index': self.active_probe_index, 'start_time_s': observation['t_seconds'],
                'object_origin_m': list(self.probe_origin['object']), 'tcp_origin_m': list(self.probe_origin['tcp']),
                'tcp_quaternion_wxyz': list(self.probe_quaternion), 'local_offset_origin_m': list(self.anchor),
                'observation_origin': ORIGIN, 'end_time_s': None, 'disposition': 'probing', 'reason': None,
            })
        if kind == 'alarm' and self.active_probe_index is not None:
            self.probes[self.active_probe_index].update(end_time_s=observation['t_seconds'],
                                                         disposition='interrupted', reason=reason)
        if kind == 'retry_started': self.status = 'acquiring_retry'
        if kind == 'retry_completed': self.status = 'acquiring'
        super()._event(kind, observation, reason)
        self.events[-1].update(variant=self.variant, status=self.status,
                              probe_index=self.active_probe_index, observation_origin=ORIGIN)
        if kind == 'retry_completed': self.active_probe_index = None

    def _terminal_hold(self, observation, reason, *, timeout=False, event='hold_invalidated'):
        self.stop_targets = list(observation['joint_position_rad'])
        self.frozen_jaw_target = self._jaw_to_freeze(observation)
        self.state, self.issued_phase = 'abort', 'probe_stop'
        self.status = 'timed_out_hold' if timeout else 'invalidated_hold'
        self.admitted, self.qualified, self.aborted, self.abort_reason = False, False, True, reason
        self.diagnostic.update(decision='hold_measured_arm_and_freeze_issued_jaw', qualified=False)
        if self.active_probe_index is not None:
            self.probes[self.active_probe_index].update(end_time_s=observation['t_seconds'],
                                                         disposition=self.status, reason=reason)
        self._event(event, observation, reason)
        return self._hold()

    def _abort(self, observation, reason):
        return self._terminal_hold(observation, reason, event='abort')

    def _eligible(self, observation):
        if self.probe_origin is None:
            return False
        return (all(observation['grasp_contact_counts'].values())
                and min(observation['pad_normal_forces_n'].values()) > .01
                and observation['table_object_contact_count'] == 0
                and observation['unexpected_contact_count'] == 0
                and self.diagnostic['object_height_delta_m'] >= self.config.probe_min_object_rise_m
                and self.diagnostic['relative_offset_error_m'] <= self.config.relative_drift_limit_m)

    def _admit(self, observation, kind):
        self.stop_targets = list(observation['joint_position_rad'])
        self.frozen_jaw_target = self._jaw_to_freeze(observation)
        self.state, self.issued_phase = 'probe_hold', 'probe_stop'
        self.status = 'qualified_hold' if self.variant == 'evidence' else 'timer_hold'
        self.admitted, self.ever_admitted, self.qualified = True, True, True
        self.admission_time_s = observation['t_seconds']
        self.qualification_count += 1
        self.diagnostic.update(decision='hold_measured_arm_and_freeze_issued_jaw', qualified=True)
        self.probes[self.active_probe_index].update(disposition=self.status,
                                                    admission_time_s=self.admission_time_s)
        self._event(kind, observation, 'sampled_probe_evidence' if self.variant == 'evidence' else 'fixed_probe_clock_only')
        return self._hold()

    def _probe(self, observation):
        self.status, self.issued_phase = 'probing', 'guard_probe'
        if self.probe_started is None:
            self.probe_started = observation['t_seconds']
            self.probe_origin = {'object': list(observation['object_position_m']), 'tcp': list(observation['tcp_position_m'])}
            self.probe_quaternion = list(observation['tcp_quaternion_wxyz'])
            self.anchor = guard._local_offset(observation)
            self.diagnostic = self._diagnose(observation)
            self._event('probe_started', observation, 'slow_load_bearing_check')
        elapsed = observation['t_seconds']-self.probe_started
        if self.variant == 'evidence':
            stopped = self._monitor(observation, probe=True)
            if stopped is not None: return stopped
            eligible = self._eligible(observation)
            self.qualification_since = ((observation['t_seconds'] if self.qualification_since is None else self.qualification_since)
                                        if eligible else None)
            dwell = 0. if self.qualification_since is None else observation['t_seconds']-self.qualification_since
            self.diagnostic['qualification_dwell_seconds'] = dwell
            if dwell+1e-9 >= self.config.qualification_dwell_s:
                return self._admit(observation, 'evidence_qualified')
        elif elapsed+1e-9 >= TIMER_ADMISSION_S:
            return self._admit(observation, 'timer_admitted')
        if elapsed > self.config.probe_timeout_s:
            return self._alarm(observation, 'probe_timeout')
        goal = list(self.probe_origin['tcp'])
        goal[2] += min(self.config.probe_height_m,
                       self.config.probe_speed_m_s*(elapsed+self.env.config.control_dt))
        return self.env.pose_action(goal, self.probe_quaternion, 0.)

    def _hold_check(self, observation):
        if not self.admitted:
            return self._hold()
        if not self._eligible(observation):
            reason = ('unexpected_contact' if observation['unexpected_contact_count'] else
                      'bilateral_contact_or_force_loss' if not all(observation['grasp_contact_counts'].values()) or min(observation['pad_normal_forces_n'].values()) <= .01 else
                      'support_returned' if observation['table_object_contact_count'] else
                      'relative_slip' if self.diagnostic['relative_offset_error_m'] > self.config.relative_drift_limit_m else
                      'probe_height_not_retained')
            return self._terminal_hold(observation, reason)
        self.issued_phase = 'probe_stop'
        self.diagnostic.update(decision='maintain_latched_arm_and_jaw', qualified=True)
        return self._hold()

    def _claim(self, observation):
        if self.active_probe_index is not None:
            self.qualification_checks.append({'t_seconds': observation['t_seconds'], 'qualified': self.admitted,
                                               'probe_index': self.active_probe_index})
        self.diagnostic.update(variant=self.variant, status=self.status, admitted=self.admitted,
                               probe_index=self.active_probe_index, observation_origin=ORIGIN)

    def __call__(self, observation):
        if self.final_observed or len(self.requested_controls) >= self.command_budget:
            raise ValueError('Probe command budget exhausted or final observation already sealed')
        self._clock(observation)
        self.diagnostic_count += 1
        self.diagnostic = self._diagnose(observation)
        if self.status in ('invalidated_hold', 'timed_out_hold'):
            action = self._hold()
        elif observation['unexpected_contact_count']:
            action = self._terminal_hold(observation, 'unexpected_contact')
        elif self.admitted:
            action = self._hold_check(observation)
        elif observation['t_seconds']+DT >= self.timeout_s-1e-9:
            action = self._terminal_hold(observation, 'episode_acquisition_timeout', timeout=True)
        elif self.state == 'probe':
            action = self._probe(observation)
        else:
            if self.state == 'normal' and self.teacher.phase not in ('approach', 'descend', 'close'):
                action = self._terminal_hold(observation, 'forbidden_full_lift_dispatch')
            else:
                action = super().__call__(observation)
                if self.state == 'probe': self.status = 'probe_pending'
        _vector(action.get('joint_targets_rad'), 'requested joint targets', 6)
        _number(action.get('gripper_position_m'), 'requested jaw target', 0., .035)
        self._claim(observation)
        self.last_requested = deepcopy(action)
        self.requested_controls.append({'index': len(self.requested_controls), 't_seconds': observation['t_seconds'],
                                         'phase': self.issued_phase, 'status': self.status,
                                         'probe_index': self.active_probe_index, **deepcopy(action)})
        return deepcopy(action)

    def observe_final(self, observation):
        """Check the last recorded state without emitting another control vector."""
        if self.final_observed: raise ValueError('Final observation is already sealed')
        self._clock(observation)
        self.diagnostic_count += 1
        self.diagnostic = self._diagnose(observation)
        if self.admitted: self._hold_check(observation)
        elif self.status not in ('invalidated_hold', 'timed_out_hold'):
            self._terminal_hold(observation, 'episode_ended_without_admission', timeout=True)
        self._claim(observation)
        self.final_observed = True
        return self.summary()

    def summary(self):
        return deepcopy({'schema': SCHEMA, 'variant': self.variant, 'status': self.status,
                         'observation_origin': ORIGIN, 'physical_robot_ready': False,
                         'contract': probe_contract(), 'admitted': self.admitted,
                         'ever_admitted': self.ever_admitted, 'admission_time_s': self.admission_time_s,
                         'initial_context': self.initial_context, 'probes': self.probes,
                         'active_probe_index': self.active_probe_index,
                         'events': self.events, 'requested_controls': self.requested_controls,
                         'qualification_checks': self.qualification_checks,
                         'stop_targets_rad': self.stop_targets, 'frozen_jaw_target_m': self.frozen_jaw_target,
                         'retry_count': self.retry_count, 'abort_reason': self.abort_reason,
                         'last_observed_s': self.last_observed_s, 'final_observed': self.final_observed})
