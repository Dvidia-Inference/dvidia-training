"""Bounded contact recovery, preserving the v0.3 qualification and hold gates.

This authored controller changes requested joint positions only. The closed-jaw
refusal is evidence that this grasp failed its probe, not a material estimate or
a claim that a different grasp could never work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from . import piper_contact_memory as memory


SCHEMA = 'dvidia.piper-contact-recovery/v0.4'
MODES = ('recovery_carry', 'recovery')


@dataclass(frozen=True)
class RecoveryConfig:
    acquisition_window_s: float = .12
    acquisition_minimum_span_s: float = .10
    acquisition_jaw_excursion_m: float = .0005
    acquisition_normal_excursion_n: float = .05
    restoration_dwell_s: float = .10
    restoration_ramp_s: float = .30

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('Invalid recovery parameter: '+name)
            try:
                valid = math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError('Invalid recovery parameter: '+name)
        if self.acquisition_minimum_span_s > self.acquisition_window_s:
            raise ValueError('Acquisition window must cover the required span')


def recovery_contract():
    return {'schema': SCHEMA, 'parameters': asdict(RecoveryConfig()),
            'prior_contact_contract': memory.contact_memory_contract(),
            'acquisition': memory.contact_memory_contract()['acquisition'],
            'refusal': 'full recovery mode only: original two-observation supported object-not-following alarm plus at least 100 ms of bilateral loaded supported contact, closed requests, jaw excursion <=0.5 mm and each pad normal force excursion <=0.05 N; hold closed and refuse the unchanged retry',
            'refusal_scope': 'an unchanged grasp did not follow its probe; no mass/friction input or conclusion that a different grasp is impossible',
            'carry_only': 'recovery_carry retains exact v0.3 acquisition and supported retry; only carry speed restoration changes',
            'restoration': 'after the unchanged v0.3 settling hold, resume at 0.45 joint-delta scale; the same relative settling evidence must persist for 100 ms before scale increases linearly toward 1 over 300 ms of additional stable observed time; pause the increase while evidence is unstable and reset on a new hold',
            'evidence': 'known relative history, displacement and maximum excursion <=1.5 mm, rate <=8 mm/s, all samples loaded bilateral and unsupported, object rise >=6 mm; unchanged v0.3 trend/contact-loss responses remain active throughout',
            'duration': 'fixed 20-second episode includes refusal holding and every recovery; no force, gain, model, scorer, or instantaneous velocity writes',
            'scope': 'authored controller using privileged simulation observations; no learned model, sensor latency, material inference or hardware qualification'}


def inactive_summary(mode):
    result = memory.inactive_summary(mode)
    result.update(schema=SCHEMA, contract=recovery_contract(), unchanged_retry_refused=False,
                  restoration_started_count=0, full_speed_restored_count=0)
    return result


class ContactRecoveryPiperTaskPolicy(memory.ContactMemoryPiperTaskPolicy):
    def __init__(self, environment, mode='recovery'):
        if mode not in MODES:
            raise ValueError('Recovery mode must be recovery_carry or recovery')
        super().__init__(environment, 'history')
        self.mode, self.recovery_config = mode, RecoveryConfig()
        self.acquisition_history = []
        self.previous_jaw_request = None
        self.unchanged_retry_refused = False
        self.restoration_stable_since = self.restoration_last_time = None
        self.restoration_progress = 0.
        self.restoration_started = self.restoration_complete = False
        self.restoration_started_count = self.full_speed_restored_count = 0

    def summary(self):
        result = super().summary()
        result.update(schema=SCHEMA, contract=recovery_contract(),
                      unchanged_retry_refused=self.unchanged_retry_refused,
                      restoration_started_count=self.restoration_started_count,
                      full_speed_restored_count=self.full_speed_restored_count)
        return result

    def _diagnose(self, observation):
        result = super()._diagnose(observation)
        now, cfg = observation['t_seconds'], self.recovery_config
        sample = {'t': now, 'jaw': list(observation['gripper_joint_position_m']),
                  'forces': dict(observation['pad_normal_forces_n']),
                  'loaded': result['bilateral_contact'] and min(observation['pad_normal_forces_n'].values()) > .01,
                  'supported': result['table_supported'],
                  'closed_requested': self.previous_jaw_request == 0.}
        if self.state == 'probe' or self.teacher.phase == 'close':
            if self.acquisition_history and abs(self.acquisition_history[-1]['t']-now) < 1e-9:
                self.acquisition_history[-1] = sample
            else:
                self.acquisition_history.append(sample)
            self.acquisition_history = [row for row in self.acquisition_history if row['t'] >= now-cfg.acquisition_window_s-1e-9]
        else:
            self.acquisition_history = []
        rows = self.acquisition_history
        span = rows[-1]['t']-rows[0]['t'] if rows else 0.
        known = span+1e-9 >= cfg.acquisition_minimum_span_s
        jaw_excursion = max((max(row['jaw'][i] for row in rows)-min(row['jaw'][i] for row in rows) for i in range(2)), default=0.) if rows else None
        force_excursion = {side: max(row['forces'][side] for row in rows)-min(row['forces'][side] for row in rows) for side in ('left', 'right')} if rows else None
        stable_closed = (known and all(row['loaded'] and row['supported'] and row['closed_requested'] for row in rows)
                         and jaw_excursion <= cfg.acquisition_jaw_excursion_m
                         and max(force_excursion.values()) <= cfg.acquisition_normal_excursion_n)
        result.update(acquisition_window_count=len(rows), acquisition_window_span_s=span,
                      acquisition_window_known=known, acquisition_jaw_excursion_m=jaw_excursion,
                      acquisition_normal_excursion_n=force_excursion,
                      acquisition_stable_closed_supported=stable_closed,
                      restoration_stable_dwell_seconds=0., restoration_progress_seconds=self.restoration_progress,
                      restoration_started=self.restoration_started, restoration_complete=self.restoration_complete,
                      recovery_scale=1.)
        return result

    def _alarm(self, observation, reason):
        if (self.mode == 'recovery' and self.state == 'probe' and reason == 'object_not_following'
                and self.diagnostic['acquisition_stable_closed_supported']
                and observation['unexpected_contact_count'] == 0):
            self.stop_targets = list(observation['joint_position_rad'])
            self.issued_phase = 'guard_stop'
            self.diagnostic['decision'] = 'refuse_unchanged_supported_retry'
            self._event('alarm', observation, reason)
            self.unchanged_retry_refused = True
            self._event('unchanged_retry_refused', observation, 'stable_closed_supported_grasp_did_not_follow')
            return self._abort(observation, 'unchanged_grasp_not_following')
        return super()._alarm(observation, reason)

    def _start_hold(self, observation, reason):
        self.restoration_stable_since = self.restoration_last_time = None
        self.restoration_progress = 0.
        self.restoration_started = self.restoration_complete = False
        return super()._start_hold(observation, reason)

    def _restoration_stable(self):
        d, cfg = self.diagnostic, self.memory_config
        return (d['history_known'] and d['history_displacement_m'] <= cfg.settled_displacement_m
                and d['history_excursion_m'] <= cfg.settled_displacement_m
                and d['history_rate_m_s'] <= cfg.settled_rate_m_s
                and d['history_loaded_bilateral_all'] and d['history_unsupported_all']
                and d['object_height_delta_m'] >= self.config.probe_min_object_rise_m)

    def _scaled_teacher(self, observation, scale):
        restoring = self.resumed and self.memory_stage != 'slowdown'
        event = None
        if restoring:
            now, cfg = observation['t_seconds'], self.recovery_config
            stable = self._restoration_stable()
            elapsed = 0. if self.restoration_last_time is None else max(0., now-self.restoration_last_time)
            self.restoration_last_time = now
            if not self.restoration_started:
                self.restoration_stable_since = (now if self.restoration_stable_since is None else self.restoration_stable_since) if stable else None
                dwell = 0. if self.restoration_stable_since is None else now-self.restoration_stable_since
                self.diagnostic['restoration_stable_dwell_seconds'] = dwell
                if dwell+1e-9 >= cfg.restoration_dwell_s:
                    self.restoration_started = True
                    self.restoration_started_count += 1
                    event = ('speed_restoration_started', 'sustained_loaded_relative_stability_after_resume')
            elif stable and not self.restoration_complete:
                self.restoration_progress = min(cfg.restoration_ramp_s, self.restoration_progress+elapsed)
                if self.restoration_progress+1e-9 >= cfg.restoration_ramp_s:
                    self.restoration_complete = True
                    self.full_speed_restored_count += 1
                    event = ('full_speed_restored', 'bounded_stable_time_ramp_complete')
            scale = self.memory_config.resumed_joint_delta_scale+(1-self.memory_config.resumed_joint_delta_scale)*self.restoration_progress/cfg.restoration_ramp_s
            self.diagnostic.update(restoration_progress_seconds=self.restoration_progress,
                                   restoration_started=self.restoration_started, restoration_complete=self.restoration_complete,
                                   recovery_scale=scale)
        action = super()._scaled_teacher(observation, scale)
        if restoring:
            self.diagnostic['decision'] = 'restored_full_requested_delta' if self.restoration_complete else 'restore_requested_delta_with_stable_evidence'
        if event is not None:
            self._event(event[0], observation, event[1])
        return action

    def __call__(self, observation):
        action = super().__call__(observation)
        self.previous_jaw_request = action['gripper_position_m']
        return action
