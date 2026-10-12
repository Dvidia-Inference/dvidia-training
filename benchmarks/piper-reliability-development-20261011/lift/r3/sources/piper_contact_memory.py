"""Interpretable causal PiPER contact history, with unchanged native dynamics.

Both modes inherit the exact v0.2 acquisition probe and supported retry. A
hold changes requested joint positions; native motion can continue afterward.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from . import piper_grip_guard as acquisition


SCHEMA = 'dvidia.piper-contact-memory/v0.3'


@dataclass(frozen=True)
class ContactMemoryConfig:
    history_window_s: float = .12
    minimum_history_s: float = .10
    trend_rate_m_s: float = .025
    trend_displacement_m: float = .0025
    minimum_straightness: float = .7
    trend_samples: int = 2
    loss_samples: int = 2
    slowdown_joint_delta_scale: float = .35
    slowdown_duration_s: float = .12
    hold_minimum_s: float = .12
    hold_timeout_s: float = .8
    settled_rate_m_s: float = .008
    settled_displacement_m: float = .0015
    settled_dwell_s: float = .10
    resumed_joint_delta_scale: float = .45
    maximum_reassessments: int = 2

    def __post_init__(self):
        for name, value in asdict(self).items():
            try:
                finite = math.isfinite(value)
            except (OverflowError, TypeError, ValueError):
                finite = False
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not finite or value <= 0:
                raise ValueError('Invalid contact memory parameter: '+name)
        for name in ('trend_samples', 'loss_samples', 'maximum_reassessments'):
            if type(getattr(self, name)) is not int or getattr(self, name) != 2:
                raise ValueError('Contact memory uses two observations and at most two reassessments')
        if (self.minimum_history_s > self.history_window_s or self.hold_minimum_s >= self.hold_timeout_s
                or self.minimum_straightness > 1 or self.slowdown_joint_delta_scale > 1
                or self.resumed_joint_delta_scale > 1 or self.settled_rate_m_s >= self.trend_rate_m_s):
            raise ValueError('Invalid contact memory threshold ordering')


def contact_memory_contract():
    return {'schema': SCHEMA, 'parameters': asdict(ContactMemoryConfig()),
            'acquisition': acquisition.guard_contract(),
            'observations': 'causal recorded privileged TCP/object poses and native contact/normal force snapshots; no material, mass, future observation or outcome input to the contact check',
            'history': 'inclusive past 120 ms of TCP-local object offset; unknown before 100 ms span; displacement is endpoint Euclidean distance, rate is displacement/span, slope is least-squares vector rate, straightness is displacement/summed consecutive path',
            'trend': 'rate >=25 mm/s, displacement >=2.5 mm and straightness >=0.7 for two snapshots; two bilateral-contact-loss snapshots independently request closed hold',
            'slowdown': 'scale original requested joint-position delta from measured joints by 0.35 for 120 ms, then hold measured joints closed; actual native speed is measured separately',
            'reassessment': 'minimum 120 ms hold; known window displacement and maximum pairwise excursion <=1.5 mm and rate <=8 mm/s, loaded bilateral contacts throughout, no table support and object rise >=6 mm sustained for 100 ms; timeout 800 ms, at most two holds',
            'resume': 'causally refresh original teacher held offset to current TCP-local object offset and scale requested joint deltas by 0.45 until supported arrival; subsequent persistent slip may hold again within the budget',
            'probe_only': 'identical acquisition and retry; removes postqualification carry monitoring only',
            'stop': 'measured joint-position requests with jaw closed; inertia and native target rate limits may continue physical motion',
            'duration': 'unchanged fixed episode budget including probe, retry, slowdown, hold and abort; no force/gain/dynamics/scorer change',
            'scope': 'authored simulator controller with privileged observations; no learned memory, material inference, sensor latency, hardware safety or task qualification claim'}


def inactive_summary(mode):
    value = acquisition.inactive_summary(mode)
    value.update(schema=SCHEMA, contract=contact_memory_contract(), slowdown_count=0,
                 reassessment_count=0, resume_count=0)
    return value


contact_contract = contact_memory_contract


class ContactMemoryPiperTaskPolicy(acquisition.GuardedPiperTaskPolicy):
    """Same acquisition, with a causal history controller or probe ablation."""

    def __init__(self, environment, mode='history'):
        if mode not in ('probe_only', 'history'):
            raise ValueError('Contact memory mode must be probe_only or history')
        super().__init__(environment)
        self.mode, self.memory_config = mode, ContactMemoryConfig()
        self.history = []
        self.memory_stage, self.memory_started, self.settled_since = 'observe', None, None
        self.trend_streak, self.memory_loss_streak = 0, 0
        self.slowdown_count = self.reassessment_count = self.resume_count = 0
        self.resumed = False

    def summary(self):
        value = super().summary()
        value.update(schema=SCHEMA, mode=self.mode, contract=contact_memory_contract(),
                     slowdown_count=self.slowdown_count, reassessment_count=self.reassessment_count,
                     resume_count=self.resume_count)
        return value

    def _diagnose(self, observation):
        value = super()._diagnose(observation)
        now = observation['t_seconds']
        if self.state == 'probe' or self.qualified:
            sample = {'t': now, 'offset': acquisition._local_offset(observation),
                      'bilateral': value['bilateral_contact'],
                      'loaded': value['bilateral_contact'] and min(observation['pad_normal_forces_n'].values()) > .01,
                      'supported': value['table_supported']}
            # The parent diagnoses twice when a probe starts; one snapshot is
            # one history sample, never two apparent observations.
            if self.history and abs(self.history[-1]['t']-now) < 1e-9:
                self.history[-1] = sample
            else:
                self.history.append(sample)
            self.history = [row for row in self.history if row['t'] >= now-self.memory_config.history_window_s-1e-9]
        else:
            self.history = []
        span = self.history[-1]['t']-self.history[0]['t'] if self.history else 0.
        known = span+1e-9 >= self.memory_config.minimum_history_s
        displacement = math.dist(self.history[-1]['offset'], self.history[0]['offset']) if self.history else None
        path = sum(math.dist(a['offset'], b['offset']) for a, b in zip(self.history, self.history[1:]))
        excursion = max((math.dist(a['offset'], b['offset']) for a in self.history for b in self.history), default=0.)
        slope = None
        if known:
            mean_t = sum(row['t'] for row in self.history)/len(self.history)
            denominator = sum((row['t']-mean_t)**2 for row in self.history)
            slope = [sum((row['t']-mean_t)*row['offset'][axis] for row in self.history)/denominator for axis in range(3)]
        value.update(memory_stage=self.memory_stage, history_count=len(self.history), history_known=known,
                     history_span_seconds=span, history_displacement_m=displacement,
                     history_rate_m_s=displacement/span if known else None,
                     history_path_m=path, history_excursion_m=excursion,
                     history_slope_m_s=slope,
                     history_straightness=displacement/path if known and path > 1e-12 else (0. if known else None),
                     history_bilateral_fraction=sum(row['bilateral'] for row in self.history)/len(self.history) if self.history else None,
                     history_loaded_bilateral_all=known and all(row['loaded'] for row in self.history),
                     history_unsupported_all=known and all(not row['supported'] for row in self.history),
                     slip_trend_streak=self.trend_streak, persistent_loss_streak=self.memory_loss_streak,
                     settled_dwell_seconds=0. if self.settled_since is None else now-self.settled_since,
                     joint_delta_scale=1., unscaled_joint_targets_rad=None,
                     motion_phase=self.teacher.phase, resume_local_offset_m=None)
        return value

    def _scaled_teacher(self, observation, scale):
        action = self.teacher(observation)
        self.diagnostic['unscaled_joint_targets_rad'] = list(action['joint_targets_rad'])
        self.diagnostic['motion_phase'] = self.teacher.issued_phase
        action['joint_targets_rad'] = [current+scale*(target-current) for current, target
                                       in zip(observation['joint_position_rad'], action['joint_targets_rad'])]
        self.issued_phase = 'memory_slowdown' if self.memory_stage == 'slowdown' else self.teacher.issued_phase
        self.diagnostic['joint_delta_scale'] = scale
        self.diagnostic['decision'] = 'slow_requested_joint_delta' if self.memory_stage == 'slowdown' else 'resume_reduced_requested_delta'
        return action

    def _start_hold(self, observation, reason):
        if self.reassessment_count >= self.memory_config.maximum_reassessments:
            return self._abort(observation, 'reassessment_budget_exhausted:'+reason)
        self.memory_stage, self.memory_started, self.settled_since = 'hold', observation['t_seconds'], None
        self.stop_targets = list(observation['joint_position_rad'])
        self.reassessment_count += 1
        self.issued_phase = 'memory_hold'
        self.diagnostic.update(memory_stage='hold', decision='hold_measured_arm_closed')
        self._event('reassessment_started', observation, reason)
        return self._hold()

    def _monitor(self, observation, *, probe):
        if probe:
            return super()._monitor(observation, probe=True)
        if self.mode == 'probe_only':
            return None
        cfg, diagnostic, now = self.memory_config, self.diagnostic, observation['t_seconds']
        if observation['unexpected_contact_count']:
            return self._abort(observation, 'unexpected_contact_during_carry')
        trend = (diagnostic['history_known'] and diagnostic['history_rate_m_s'] >= cfg.trend_rate_m_s
                 and diagnostic['history_displacement_m'] >= cfg.trend_displacement_m
                 and diagnostic['history_straightness'] >= cfg.minimum_straightness)
        self.trend_streak = self.trend_streak+1 if trend else 0
        self.memory_loss_streak = self.memory_loss_streak+1 if not diagnostic['bilateral_contact'] else 0
        diagnostic.update(slip_trend_streak=self.trend_streak, persistent_loss_streak=self.memory_loss_streak)
        if self.memory_stage == 'hold':
            self.issued_phase = 'memory_hold'
            diagnostic['decision'] = 'hold_measured_arm_closed'
            stable = (now-self.memory_started+1e-9 >= cfg.hold_minimum_s and diagnostic['history_known']
                      and diagnostic['history_displacement_m'] <= cfg.settled_displacement_m
                      and diagnostic['history_excursion_m'] <= cfg.settled_displacement_m
                      and diagnostic['history_rate_m_s'] <= cfg.settled_rate_m_s
                      and diagnostic['history_loaded_bilateral_all'] and diagnostic['history_unsupported_all']
                      and diagnostic['object_height_delta_m'] >= self.config.probe_min_object_rise_m)
            self.settled_since = (now if self.settled_since is None else self.settled_since) if stable else None
            diagnostic['settled_dwell_seconds'] = 0. if self.settled_since is None else now-self.settled_since
            if diagnostic['settled_dwell_seconds']+1e-9 >= cfg.settled_dwell_s:
                self.teacher.held_offset = acquisition._local_offset(observation)
                self.memory_stage, self.resumed = 'observe', True
                self.resume_count += 1
                self.trend_streak = self.memory_loss_streak = 0
                diagnostic.update(memory_stage='observe', decision='settled_resume_reduced_delta',
                                  resume_local_offset_m=list(self.teacher.held_offset))
                self._event('carry_resumed', observation, 'sustained_loaded_relative_settling')
                self.settled_since = None
                # Last held command at the resume boundary; original motion
                # starts on the next observed decision, never retrospectively.
                return self._hold()
            if now-self.memory_started > cfg.hold_timeout_s:
                return self._abort(observation, 'reassessment_timeout_unsettled_or_unknown')
            return self._hold()
        if self.memory_loss_streak >= cfg.loss_samples:
            return self._start_hold(observation, 'persistent_bilateral_contact_loss')
        if self.memory_stage == 'slowdown':
            if now-self.memory_started+1e-9 >= cfg.slowdown_duration_s:
                return self._start_hold(observation, 'bounded_slowdown_complete')
            return self._scaled_teacher(observation, cfg.slowdown_joint_delta_scale)
        if self.trend_streak >= cfg.trend_samples:
            self.memory_stage, self.memory_started = 'slowdown', now
            self.slowdown_count += 1
            self.issued_phase = 'memory_slowdown'
            diagnostic.update(memory_stage='slowdown', decision='slow_requested_joint_delta',
                              joint_delta_scale=cfg.slowdown_joint_delta_scale)
            action = self._scaled_teacher(observation, cfg.slowdown_joint_delta_scale)
            self._event('slowdown_started', observation, 'persistent_local_offset_trend')
            return action
        if self.resumed:
            return self._scaled_teacher(observation, cfg.resumed_joint_delta_scale)
        diagnostic['decision'] = 'history_unknown_continue' if not diagnostic['history_known'] else 'history_observe'
        return None

    def __call__(self, observation):
        action = super().__call__(observation)
        # The inherited arrival gate stops carry monitoring. Clear memory so
        # the original supported release/retreat can run without slow scaling.
        if self.carry_supported:
            self.memory_stage = 'destination_supported'
        return action
