"""One bounded second lift after a low but settled contact hold.

This remains an authored, privileged-state simulation controller. It preserves
native physics and the original six-millimetre resume and task success gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from . import piper_contact_recovery as recovery
from . import piper_grip_guard as acquisition


SCHEMA = 'dvidia.piper-reacquisition/v0.5'


@dataclass(frozen=True)
class ReacquisitionConfig:
    low_hold_minimum_rise_m: float = .001
    low_hold_dwell_s: float = .10
    probe_height_m: float = .008
    probe_speed_m_s: float = .010
    probe_minimum_tcp_rise_m: float = .004
    probe_timeout_s: float = 1.2
    maximum_probes: int = 1

    def __post_init__(self):
        for name, value in asdict(self).items():
            try:
                valid = math.isfinite(value) and value > 0
            except (TypeError, ValueError, OverflowError):
                valid = False
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not valid:
                raise ValueError('Invalid reacquisition parameter: '+name)
        if type(self.maximum_probes) is not int or self.maximum_probes != 1:
            raise ValueError('One second probe is allowed per episode')
        if self.probe_height_m/self.probe_speed_m_s >= self.probe_timeout_s:
            raise ValueError('Second probe timeout must include settling time')
        if self.probe_minimum_tcp_rise_m >= self.probe_height_m:
            raise ValueError('Probe travel gate must fit its bounded height')
        if self.low_hold_minimum_rise_m >= acquisition.GuardConfig().probe_min_object_rise_m:
            raise ValueError('Low hold must remain below the unchanged resume floor')


def reacquisition_contract():
    return {'schema': SCHEMA, 'parameters': asdict(ReacquisitionConfig()),
            'prior_recovery_contract': recovery.recovery_contract(),
            'eligibility': 'only an existing postqualification settling hold; unchanged known 100 ms relative history, loaded bilateral unsupported samples, <=1.5 mm displacement/excursion and <=8 mm/s rate; 100 ms low-stable dwell after minimum 120 ms hold; object rise >=1 mm and below original 6 mm resume floor',
            'probe': 'at most one measured-pose vertical second probe, requested rise <=8 mm at 10 mm/s, fixed measured orientation and closed jaw; 1.2 second total timeout including qualification',
            'failure': 'close hold and abort on unknown history, any current unloaded or lost pad, renewed table support, unexpected contact, relative offset drift >3 mm, or persistent original slip trend; never increase force or alter gains',
            'resume': 'second probe measured TCP rise >=4 mm from its origin; unchanged loaded relative settling evidence and original >=6 mm rise sustained 100 ms; refresh measured relative held offset, then original v0.4 reduced delta and restoration; task lift/retention/place gates untouched',
            'scope': 'authored privileged simulator state, not learned skill, sensor pipeline, calibrated robot or hardware safety control'}


def inactive_summary(mode):
    result = recovery.inactive_summary(mode)
    result.update(schema=SCHEMA, contract=reacquisition_contract(), second_probe_count=0, second_probe_resume_count=0)
    return result


class ReacquisitionPiperTaskPolicy(recovery.ContactRecoveryPiperTaskPolicy):
    def __init__(self, environment, mode='reacquire'):
        if mode != 'reacquire':
            raise ValueError('Reacquisition mode must be reacquire')
        super().__init__(environment, 'recovery')
        self.reacquisition_config = ReacquisitionConfig()
        self.second_probe_count = self.second_probe_resume_count = 0
        self.low_settled_since = self.second_probe_started = None
        self.second_probe_origin = self.second_probe_quaternion = self.second_probe_anchor = None
        self.second_probe_qualified_since = None

    def summary(self):
        result = super().summary()
        result.update(schema=SCHEMA, mode='reacquire', contract=reacquisition_contract(),
                      second_probe_count=self.second_probe_count,
                      second_probe_resume_count=self.second_probe_resume_count)
        return result

    def _diagnose(self, observation):
        result = super()._diagnose(observation)
        result.update(second_probe_count=self.second_probe_count,
                      second_probe_started_s=self.second_probe_started,
                      low_settled_dwell_s=0. if self.low_settled_since is None else observation['t_seconds']-self.low_settled_since,
                      second_probe_qualification_dwell_s=0., second_probe_requested_rise_m=0.,
                      second_probe_relative_drift_m=None,
                      second_probe_origin_tcp_m=None if self.second_probe_origin is None else list(self.second_probe_origin),
                      second_probe_orientation_wxyz=None if self.second_probe_quaternion is None else list(self.second_probe_quaternion),
                      second_probe_anchor_local_m=None if self.second_probe_anchor is None else list(self.second_probe_anchor),
                      second_probe_measured_tcp_rise_m=None, second_probe_goal_m=None)
        return result

    def _stable_relative(self):
        d, cfg = self.diagnostic, self.memory_config
        return (d['history_known'] and d['history_displacement_m'] <= cfg.settled_displacement_m
                and d['history_excursion_m'] <= cfg.settled_displacement_m
                and d['history_rate_m_s'] <= cfg.settled_rate_m_s
                and d['history_loaded_bilateral_all'] and d['history_unsupported_all'])

    def _second_probe(self, observation):
        d, cfg, now = self.diagnostic, self.reacquisition_config, observation['t_seconds']
        self.issued_phase = 'memory_second_probe'
        d.update(memory_stage='second_probe', decision='bounded_second_lift')
        drift = math.dist(acquisition._local_offset(observation), self.second_probe_anchor)
        d.update(second_probe_relative_drift_m=drift,
                 second_probe_origin_tcp_m=list(self.second_probe_origin),
                 second_probe_orientation_wxyz=list(self.second_probe_quaternion),
                 second_probe_anchor_local_m=list(self.second_probe_anchor),
                 second_probe_measured_tcp_rise_m=observation['tcp_position_m'][2]-self.second_probe_origin[2])
        unsafe = ('unknown_history' if not d['history_known'] else
                  'unexpected_contact' if observation['unexpected_contact_count'] else
                  'table_support_returned' if d['table_supported'] else
                  'loaded_bilateral_contact_lost' if not d['bilateral_contact'] or min(observation['pad_normal_forces_n'].values()) <= .01 else
                  'relative_slip' if drift > self.config.relative_drift_limit_m else None)
        trend = (d['history_known'] and d['history_rate_m_s'] >= self.memory_config.trend_rate_m_s
                 and d['history_displacement_m'] >= self.memory_config.trend_displacement_m
                 and d['history_straightness'] >= self.memory_config.minimum_straightness)
        self.trend_streak = self.trend_streak+1 if trend else 0
        d['slip_trend_streak'] = self.trend_streak
        if unsafe or self.trend_streak >= self.memory_config.trend_samples:
            return self._abort(observation, 'second_probe_'+(unsafe or 'persistent_relative_trend'))
        eligible = (self._stable_relative() and d['object_height_delta_m'] >= self.config.probe_min_object_rise_m
                    and observation['tcp_position_m'][2]-self.second_probe_origin[2] >= cfg.probe_minimum_tcp_rise_m)
        self.second_probe_qualified_since = (now if self.second_probe_qualified_since is None else self.second_probe_qualified_since) if eligible else None
        dwell = 0. if self.second_probe_qualified_since is None else now-self.second_probe_qualified_since
        d['second_probe_qualification_dwell_s'] = dwell
        if dwell+1e-9 >= self.memory_config.settled_dwell_s:
            self.teacher.held_offset = acquisition._local_offset(observation)
            self.memory_stage, self.resumed = 'observe', True
            self.resume_count += 1
            self.second_probe_resume_count += 1
            self.trend_streak = self.memory_loss_streak = 0
            self.stop_targets = list(observation['joint_position_rad'])
            d.update(memory_stage='observe', decision='second_probe_qualified_resume',
                     resume_local_offset_m=list(self.teacher.held_offset))
            self._event('second_probe_qualified', observation, 'original_height_and_settling_gates_met')
            self._event('carry_resumed', observation, 'second_probe_sustained_loaded_relative_settling')
            return self._hold()
        if now-self.second_probe_started > cfg.probe_timeout_s:
            return self._abort(observation, 'second_probe_timeout')
        rise = min(cfg.probe_height_m, cfg.probe_speed_m_s*(now-self.second_probe_started+self.env.config.control_dt))
        d['second_probe_requested_rise_m'] = rise
        goal = list(self.second_probe_origin)
        goal[2] += rise
        d['second_probe_goal_m'] = list(goal)
        return self.env.pose_action(goal, self.second_probe_quaternion, 0.)

    def _monitor(self, observation, *, probe):
        if probe:
            return super()._monitor(observation, probe=True)
        if self.memory_stage == 'second_probe':
            return self._second_probe(observation)
        d, cfg, now = self.diagnostic, self.reacquisition_config, observation['t_seconds']
        if self.memory_stage == 'hold' and self.second_probe_count < cfg.maximum_probes:
            eligible = (now-self.memory_started+1e-9 >= self.memory_config.hold_minimum_s
                        and self._stable_relative() and observation['unexpected_contact_count'] == 0
                        and cfg.low_hold_minimum_rise_m <= d['object_height_delta_m'] < self.config.probe_min_object_rise_m)
            self.low_settled_since = (now if self.low_settled_since is None else self.low_settled_since) if eligible else None
            d['low_settled_dwell_s'] = 0. if self.low_settled_since is None else now-self.low_settled_since
            if d['low_settled_dwell_s']+1e-9 >= cfg.low_hold_dwell_s:
                self.second_probe_count += 1
                self.second_probe_started = now
                self.second_probe_origin = list(observation['tcp_position_m'])
                self.second_probe_quaternion = list(observation['tcp_quaternion_wxyz'])
                self.second_probe_anchor = acquisition._local_offset(observation)
                self.second_probe_qualified_since = None
                self.memory_stage = 'second_probe'
                self.trend_streak = self.memory_loss_streak = 0
                d.update(second_probe_count=self.second_probe_count, second_probe_started_s=now,
                         memory_stage='second_probe', decision='start_bounded_second_lift')
                self.issued_phase = 'memory_second_probe'
                self._event('second_probe_started', observation, 'sustained_loaded_unsupported_low_hold')
                return self._second_probe(observation)
        else:
            self.low_settled_since = None
        return super()._monitor(observation, probe=False)
