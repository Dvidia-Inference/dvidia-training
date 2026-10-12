"""Causal, privileged-state PiPER grasp checks; unchanged native dynamics.

The guard observes only the current recorded robot/object/contact state. It
does not read object mass, friction, the outcome scorer or future observations.
Stopping means holding measured joint targets, not instant physical arrest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


SCHEMA = 'dvidia.piper-grip-guard/v1'


@dataclass(frozen=True)
class GuardConfig:
    probe_height_m: float = .010
    probe_speed_m_s: float = .020
    probe_min_object_rise_m: float = .006
    qualification_dwell_s: float = .10
    relative_drift_limit_m: float = .003
    monitor_min_probe_motion_m: float = .003
    consecutive_alarm_samples: int = 2
    probe_timeout_s: float = 1.4
    close_timeout_s: float = 1.4
    retry_open_timeout_s: float = 1.2
    retry_recenter_timeout_s: float = 1.5
    retry_clearance_m: float = .012
    max_retries: int = 1

    def __post_init__(self):
        # This first experiment has one fixed, reviewable parameter set.
        for name, value in asdict(self).items():
            try:
                finite = math.isfinite(value)
            except (TypeError, ValueError, OverflowError):
                finite = False
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not finite or value <= 0:
                raise ValueError('Invalid guard parameter: '+name)
        if type(self.consecutive_alarm_samples) is not int or self.consecutive_alarm_samples != 2 or self.max_retries != 1:
            raise ValueError('Guard requires two consecutive observations and at most one retry')
        if self.probe_min_object_rise_m >= self.probe_height_m or self.relative_drift_limit_m >= self.probe_height_m:
            raise ValueError('Guard probe must exceed its qualification/drift thresholds')


def guard_contract():
    return {'schema': SCHEMA, 'parameters': asdict(GuardConfig()),
            'observations': 'grip check uses causal current privileged TCP/object pose and bilateral native contacts at 20 ms; no mass/friction or scorer input; downstream original motion policy remains unchanged and uses the existing lift outcome flag',
            'qualification': 'slow 10 mm probe; object rises at least 6 mm, both pads retain contact, support is absent and TCP-local drift <=3 mm for 100 ms',
            'alarm': 'two consecutive snapshots with relative drift, bilateral-contact loss or supported object failing to follow a moving probe; bounded closure/probe timeouts',
            'recovery': 'hold measured six joint targets first; only supported objects may be opened/recentered/retried, at most once; support loss stops closed',
            'stop': 'requested measured joint targets; native rate limiting and inertia may continue motion, which is recorded',
            'duration': 'original fixed episode timeout including every probe/retry/abort holding command',
            'scope': 'authored simulator controller, not a sensor implementation, hardware safety controller or learned policy'}


def inactive_summary(mode):
    return {'schema': SCHEMA, 'mode': mode, 'active': False, 'contract': guard_contract(), 'events': [],
            'retry_count': 0, 'acquisition_qualified': False, 'acquisition_qualification_count': 0,
            'abort_reason': None, 'aborted': False, 'diagnostic_count': 0}


def _local_offset(observation):
    w, x, y, z = observation['tcp_quaternion_wxyz']
    # R transpose times object minus TCP, using the recorded wxyz orientation.
    rotation = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    difference = [a-b for a, b in zip(observation['object_position_m'], observation['tcp_position_m'])]
    return [sum(rotation[row][column]*difference[row] for row in range(3)) for column in range(3)]


class GuardedPiperTaskPolicy:
    """Wrap the original authored motion policy with a bounded acquisition gate."""

    def __init__(self, environment):
        from .piper_task import PiperTaskPolicy
        self.env, self.teacher, self.config = environment, PiperTaskPolicy(environment), GuardConfig()
        self.state, self.issued_phase = 'normal', 'approach'
        self.events, self.retry_count, self.qualification_count, self.diagnostic_count = [], 0, 0, 0
        self.qualified, self.aborted, self.abort_reason = False, False, None
        self.anchor, self.probe_origin, self.probe_quaternion, self.probe_started = None, None, None, None
        self.qualification_since, self.stage_started, self.stop_targets = None, None, None
        self.drift_streak, self.loss_streak, self.progress_streak = 0, 0, 0
        self.carry_supported, self.diagnostic = False, None

    def summary(self):
        return {'schema': SCHEMA, 'mode': 'guarded', 'active': True, 'contract': guard_contract(),
                'events': self.events, 'retry_count': self.retry_count, 'acquisition_qualified': self.qualified,
                'acquisition_qualification_count': self.qualification_count, 'abort_reason': self.abort_reason,
                'aborted': self.aborted, 'diagnostic_count': self.diagnostic_count}

    def _diagnose(self, observation):
        offset = _local_offset(observation)
        drift = math.dist(offset, self.anchor) if self.anchor is not None else None
        return {'t_seconds': observation['t_seconds'], 'phase': self.state if self.state != 'normal' else self.teacher.phase,
                'retry_index': self.retry_count, 'bilateral_contact': all(observation['grasp_contact_counts'].values()),
                'table_supported': observation['table_object_contact_count'] > 0,
                'relative_offset_error_m': drift,
                'object_height_delta_m': observation['object_position_m'][2]-self.probe_origin['object'][2] if self.probe_origin else None,
                'tcp_probe_height_delta_m': observation['tcp_position_m'][2]-self.probe_origin['tcp'][2] if self.probe_origin else None,
                'qualified': self.qualified, 'qualification_dwell_seconds': 0. if self.qualification_since is None else observation['t_seconds']-self.qualification_since,
                'drift_streak': self.drift_streak, 'contact_loss_streak': self.loss_streak,
                'progress_streak': self.progress_streak, 'decision': 'continue'}

    def _event(self, kind, observation, reason):
        self.events.append({'type': kind, 't_seconds': observation['t_seconds'], 'phase': self.issued_phase,
                            'reason': reason, 'retry_index': self.retry_count, 'diagnostic': dict(self.diagnostic)})

    def _hold(self, jaw=0.):
        return {'joint_targets_rad': list(self.stop_targets), 'gripper_position_m': jaw}

    def _abort(self, observation, reason):
        self.stop_targets = list(observation['joint_position_rad'])
        self.state, self.issued_phase, self.aborted, self.abort_reason = 'abort', 'guard_abort', True, reason
        self.diagnostic['decision'] = 'abort_hold_closed'
        self._event('abort', observation, reason)
        return self._hold()

    def _alarm(self, observation, reason):
        self.stop_targets = list(observation['joint_position_rad'])
        self.issued_phase, self.diagnostic['decision'] = 'guard_stop', 'hold_measured_arm'
        self._event('alarm', observation, reason)
        if self.diagnostic['table_supported'] and observation['unexpected_contact_count'] == 0 and self.retry_count < self.config.max_retries:
            self.retry_count += 1
            self.state, self.stage_started = 'retry_open', observation['t_seconds']
            self.qualified, self.qualification_since = False, None
            self._event('retry_started', observation, 'supported_object_only')
            return self._hold()
        return self._abort(observation, reason if not self.diagnostic['table_supported'] else 'retry_budget_exhausted:'+reason)

    def _monitor(self, observation, *, probe):
        diagnostic = self.diagnostic
        moving = not probe or diagnostic['tcp_probe_height_delta_m'] > self.config.monitor_min_probe_motion_m
        drift = moving and diagnostic['relative_offset_error_m'] > self.config.relative_drift_limit_m
        loss = moving and not diagnostic['bilateral_contact']
        progress = (probe and moving and diagnostic['table_supported']
                    and diagnostic['object_height_delta_m'] < .25*diagnostic['tcp_probe_height_delta_m'])
        self.drift_streak = self.drift_streak+1 if drift else 0
        self.loss_streak = self.loss_streak+1 if loss else 0
        self.progress_streak = self.progress_streak+1 if progress else 0
        diagnostic.update(drift_streak=self.drift_streak, contact_loss_streak=self.loss_streak, progress_streak=self.progress_streak)
        reason = ('object_not_following' if self.progress_streak >= self.config.consecutive_alarm_samples else
                  'relative_slip' if self.drift_streak >= self.config.consecutive_alarm_samples else
                  'bilateral_contact_loss' if self.loss_streak >= self.config.consecutive_alarm_samples else None)
        return self._alarm(observation, reason) if reason else None

    def __call__(self, observation):
        self.diagnostic_count += 1
        self.diagnostic = self._diagnose(observation)
        if self.state == 'abort':
            self.issued_phase, self.diagnostic['decision'] = 'guard_abort', 'abort_hold_closed'
            return self._hold()
        if self.state in ('retry_open', 'retry_recenter'):
            if not self.diagnostic['table_supported'] or observation['unexpected_contact_count']:
                return self._abort(observation, 'support_lost_during_retry')
            if self.state == 'retry_open':
                self.issued_phase, self.diagnostic['decision'] = 'guard_retry_open', 'open_supported_object'
                if observation['t_seconds']-self.stage_started > self.config.retry_open_timeout_s:
                    return self._abort(observation, 'retry_open_timeout')
                if (observation['gripper_joint_position_m'][0] > .028 and observation['gripper_joint_position_m'][1] < -.028
                        and not any(observation['grasp_contact_counts'].values())):
                    self.state, self.stage_started = 'retry_recenter', observation['t_seconds']
                return self._hold(.035)
            self.issued_phase, self.diagnostic['decision'] = 'guard_retry_recenter', 'recenter_with_open_jaw'
            if observation['t_seconds']-self.stage_started > self.config.retry_recenter_timeout_s:
                return self._abort(observation, 'retry_recenter_timeout')
            goal = list(observation['object_position_m'])
            goal[2] += self.config.retry_clearance_m
            action = self.env.pose_action(goal, self.env.config.tool_orientation_wxyz, .035)
            # Keep the correction above the supported object until alignment.
            current = observation['tcp_quaternion_wxyz']
            desired = self.env.config.tool_orientation_wxyz
            aligned = abs(sum(a*b for a, b in zip(current, desired))) > math.cos(.12/2)
            if math.dist(goal, observation['tcp_position_m']) < .003 and aligned:
                self.teacher.phase, self.teacher.ticks = 'descend', 0
                self.teacher.grasp_position[:] = [*observation['object_position_m'][:2], observation['object_position_m'][2]+.002]
                self.teacher.held_offset = None
                self.state, self.anchor, self.probe_origin, self.probe_started = 'normal', None, None, None
                self.drift_streak = self.loss_streak = self.progress_streak = 0
                self.carry_supported = False
                self._event('retry_completed', observation, 'open_recenter_complete')
            return action
        if self.state == 'probe':
            self.issued_phase = 'guard_probe'
            if self.probe_started is None:
                self.probe_started = observation['t_seconds']
                self.probe_origin = {'object': list(observation['object_position_m']), 'tcp': list(observation['tcp_position_m'])}
                self.probe_quaternion = list(observation['tcp_quaternion_wxyz'])
                self.anchor = _local_offset(observation)
                self.diagnostic = self._diagnose(observation)
                self._event('probe_started', observation, 'slow_load_bearing_check')
            guarded = self._monitor(observation, probe=True)
            if guarded is not None:
                return guarded
            eligible = (self.diagnostic['bilateral_contact'] and min(observation['pad_normal_forces_n'].values()) > .01
                        and not self.diagnostic['table_supported']
                        and self.diagnostic['object_height_delta_m'] >= self.config.probe_min_object_rise_m
                        and self.diagnostic['relative_offset_error_m'] <= self.config.relative_drift_limit_m)
            self.qualification_since = (observation['t_seconds'] if self.qualification_since is None else self.qualification_since) if eligible else None
            self.diagnostic['qualification_dwell_seconds'] = 0. if self.qualification_since is None else observation['t_seconds']-self.qualification_since
            if self.diagnostic['qualification_dwell_seconds']+1e-9 >= self.config.qualification_dwell_s:
                self.qualified, self.qualification_count, self.state = True, self.qualification_count+1, 'normal'
                self.teacher.held_offset = _local_offset(observation)
                self.diagnostic['qualified'], self.diagnostic['decision'] = True, 'qualified_resume_full_lift'
                self._event('acquisition_qualified', observation, 'sustained_table_free_probe')
            elif observation['t_seconds']-self.probe_started > self.config.probe_timeout_s:
                return self._alarm(observation, 'probe_timeout')
            goal = list(self.probe_origin['tcp'])
            goal[2] += min(self.config.probe_height_m, self.config.probe_speed_m_s*(observation['t_seconds']-self.probe_started+self.env.config.control_dt))
            return self.env.pose_action(goal, self.probe_quaternion, 0.)
        # Once qualified, the original motion controller remains responsible for
        # lift/transfer/place. Contact scoring remains independent and unchanged.
        if self.qualified and self.teacher.phase in ('lift', 'transfer', 'place') and not self.carry_supported:
            if (self.diagnostic['table_supported'] and self.diagnostic['bilateral_contact']
                    and math.dist(observation['object_position_m'], observation['target_position_m']) <= self.env.config.goal_position_tolerance_m):
                self.carry_supported = True
            else:
                guarded = self._monitor(observation, probe=False)
                if guarded is not None:
                    return guarded
        if self.teacher.phase == 'close' and self.teacher.ticks*self.env.config.control_dt >= self.config.close_timeout_s:
            return self._alarm(observation, 'closure_timeout')
        self.issued_phase = self.teacher.phase
        action = self.teacher(observation)
        self.issued_phase = self.teacher.issued_phase
        if self.issued_phase == 'close' and self.teacher.phase == 'lift':
            self.state = 'probe'
        return action
