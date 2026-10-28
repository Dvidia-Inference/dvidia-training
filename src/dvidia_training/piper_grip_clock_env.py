"""One external jaw interruption after an independently observable loaded grip.

The intervention is an environment event, never a policy observation or a
controller decision. Native gains, slew limits, physics, scoring and numeric
sidecar fields are inherited unchanged. No pose or contact-force writes occur.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .piper_feedback_env import FeedbackPiperTaskEnv, JawPulse


INTERVENTION_SCHEMA = 'dvidia.piper-post-grip-intervention/v1'
ANALYSIS_SCHEMA = 'dvidia.piper-post-grip-analysis/v1'


def _bounded(value, name, low, high):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be a finite number in [{low},{high}].')
    return float(value)


@dataclass(frozen=True)
class PostGripParameters:
    """Bounded, immutable parameters; all durations use complete 1 ms ticks."""

    interruption: bool = False
    force_threshold_n: float = .10
    lift_threshold_m: float = .065
    eligibility_dwell_s: float = .100
    delay_s: float = .100
    pulse_duration_s: float = .040
    minimum_opening_m: float = .020

    def __post_init__(self):
        if type(self.interruption) is not bool:
            raise ValueError('interruption must be a boolean.')
        bounds = {'force_threshold_n': (.01, 1000.), 'lift_threshold_m': (0., .20),
                  'eligibility_dwell_s': (.001, 1.), 'delay_s': (0., 2.),
                  'pulse_duration_s': (.001, .5), 'minimum_opening_m': (0., .035)}
        for name, (low, high) in bounds.items():
            object.__setattr__(self, name, _bounded(getattr(self, name), name, low, high))
        for name in ('eligibility_dwell_s', 'delay_s', 'pulse_duration_s'):
            ticks = getattr(self, name)/.001
            if not math.isclose(ticks, round(ticks), rel_tol=0., abs_tol=1e-8):
                raise ValueError(f'{name} must contain complete 1 ms native ticks.')


class PostGripPiperTaskEnv(FeedbackPiperTaskEnv):
    """Latch one qualifying grip, then optionally impose one delayed jaw pulse.

    A qualifying interval has bilateral contact, the declared minimum force on
    each pad, no table support and sufficient lift above the settled reset
    height at every native tick. The interval is evidence of eligibility, not a
    claim of hardware grip security. Losing eligibility before the dwell ends
    resets the dwell. After the latch the event cannot be retriggered.
    """

    def __init__(self, assets, config=None, *, interruption=False,
                 force_threshold_n=.10, lift_threshold_m=.065,
                 eligibility_dwell_s=.100, delay_s=.100,
                 pulse_duration_s=.040, minimum_opening_m=.020):
        self._post_grip_parameters = PostGripParameters(
            interruption, force_threshold_n, lift_threshold_m,
            eligibility_dwell_s, delay_s, pulse_duration_s, minimum_opening_m)
        super().__init__(assets, config, pulses=())
        if not math.isclose(self.config.timestep, .001, rel_tol=0., abs_tol=1e-15):
            raise ValueError('Post-grip intervention requires the fixed 1 ms native timestep.')
        self._clear_post_grip()

    @property
    def post_grip_parameters(self):
        return self._post_grip_parameters

    def _clear_post_grip(self):
        self._post_grip_dwell_ticks = 0
        self._post_grip_max_dwell_ticks = 0
        self._post_grip_event = None
        self._post_grip_initial_height = None
        self._post_grip_active_ticks = 0
        self._post_grip_effective_ticks = 0
        self._post_grip_first_effect_tick = None
        self._pulses = ()
        self._pulse_ticks = ()

    def reset(self, seed=0):
        # Clear the dynamic event before the parent's settling path. The parent
        # disables ready and performs no outcome ticks during reset settling.
        self._clear_post_grip()
        try:
            result = super().reset(seed)
        except Exception:
            self._clear_post_grip()
            raise
        self._post_grip_initial_height = float(self.initial_object_position[2])
        return result

    def _outcome_tick(self, **kwargs):
        super()._outcome_tick(**kwargs)
        if not self.ready:
            return
        parameters = self._post_grip_parameters
        self._post_grip_active_ticks += int(self._pulse_active)
        effective = self._applied_jaw > self._normal_jaw+1e-12
        self._post_grip_effective_ticks += int(effective)
        if effective and self._post_grip_first_effect_tick is None:
            self._post_grip_first_effect_tick = self._native_tick
        if self._post_grip_event is not None:
            return
        if self._post_grip_initial_height is None:
            raise RuntimeError('Post-grip eligibility requires a completed reset.')
        counts, forces, supported, _, _ = self._last_contacts
        lift = float(self.data.xpos[self.block][2])-self._post_grip_initial_height
        qualified = (counts['left'] > 0 and counts['right'] > 0
                     and forces['left'] >= parameters.force_threshold_n
                     and forces['right'] >= parameters.force_threshold_n
                     and supported == 0 and lift >= parameters.lift_threshold_m)
        self._post_grip_dwell_ticks = self._post_grip_dwell_ticks+1 if qualified else 0
        self._post_grip_max_dwell_ticks = max(self._post_grip_max_dwell_ticks,
                                              self._post_grip_dwell_ticks)
        if self._post_grip_dwell_ticks < round(parameters.eligibility_dwell_s/.001):
            return
        event = {'eligibility_native_tick': self._native_tick,
                 'eligibility_time_s': self._native_tick*.001,
                 'pulse_start_tick': None, 'pulse_end_tick': None,
                 'reason': 'nominal_condition'}
        if parameters.interruption:
            start = self._native_tick+round(parameters.delay_s/.001)
            end = start+round(parameters.pulse_duration_s/.001)
            if end <= round(self.config.timeout_s/.001):
                pulse = JawPulse(start*.001, end*.001, parameters.minimum_opening_m)
                self._pulses = (pulse,)
                self._pulse_ticks = ((start, end, parameters.minimum_opening_m),)
                event.update(pulse_start_tick=start, pulse_end_tick=end, reason='scheduled')
            else:
                event['reason'] = 'insufficient_episode_budget'
        self._post_grip_event = event

    def native_metadata(self):
        result = super().native_metadata()
        event = None if self._post_grip_event is None else dict(self._post_grip_event)
        if not self._post_grip_parameters.interruption:
            reason = 'nominal_condition'
        elif event is None:
            reason = 'eligibility_not_reached'
        elif event['reason'] != 'scheduled':
            reason = event['reason']
        elif not self._post_grip_active_ticks:
            reason = 'episode_ended_before_pulse'
        else:
            reason = None
        result['post_grip_intervention'] = {
            'schema': INTERVENTION_SCHEMA,
            'scope': 'Environment-only one-shot actuator intervention; not a policy observation or hardware safety check.',
            'parameters': asdict(self._post_grip_parameters),
            'initial_object_height_m': self._post_grip_initial_height,
            'episode_budget_ticks': round(self.config.timeout_s/.001),
            'event': event, 'events': [] if event is None else [event],
            'eligible': event is not None, 'scheduled': bool(self._pulse_ticks),
            'activated': bool(self._post_grip_active_ticks),
            'effective': bool(self._post_grip_effective_ticks),
            'active_native_ticks': self._post_grip_active_ticks,
            'effective_native_ticks': self._post_grip_effective_ticks,
            'first_effect_native_tick': self._post_grip_first_effect_tick,
            'maximum_qualifying_dwell_native_ticks': self._post_grip_max_dwell_ticks,
            'nonactivation_reason': reason,
            'effect_definition': 'applied_jaw_target_m > normal_rate_limited_jaw_target_m + 1e-12, including the closing tail',
        }
        return result


def analyze_post_grip(arrays, metadata):
    """Reconstruct eligibility and pulse exposure from read-only numeric rows.

    This repeats no environment eligibility helper and trusts neither its event
    latch nor pulse counters. It checks the declared metadata against the row
    sequence; it cannot authenticate simulated contacts or physical hardware.
    ``scheduled`` means an actual pulse fits inside the episode budget, while
    ``interruption_condition`` records assignment even if no grip was eligible.
    """
    if type(arrays) is not dict or type(metadata) is not dict:
        raise ValueError('Numeric arrays and native metadata must be dictionaries.')
    declaration = metadata.get('post_grip_intervention')
    if type(declaration) is not dict or declaration.get('schema') != INTERVENTION_SCHEMA:
        raise ValueError('Missing post-grip intervention metadata.')
    declared_parameters = declaration.get('parameters')
    if (type(declared_parameters) is not dict
            or set(declared_parameters) != set(asdict(PostGripParameters()))):
        raise ValueError('Missing or unknown post-grip parameter.')
    parameters = PostGripParameters(**declared_parameters)
    if metadata.get('schema') != 'dvidia.piper-feedback-native/v1':
        raise ValueError('Unexpected native sidecar schema.')
    if metadata.get('side_order') != ['left', 'right']:
        raise ValueError('Unexpected native pad order.')
    if metadata.get('native_timestep_s') != .001:
        raise ValueError('Independent post-grip reconstruction requires 1 ms rows.')
    initial_height = declaration.get('initial_object_height_m')
    if initial_height is None and metadata.get('rows') == 0:
        initial_height = 0.
    initial_height = _bounded(initial_height, 'initial_object_height_m', -1., 1.)
    budget = declaration.get('episode_budget_ticks')
    if type(budget) is not int or not 100 <= budget <= 30000:
        raise ValueError('Invalid native episode budget.')
    rows = metadata.get('rows')
    if type(rows) is not int or not 0 <= rows <= budget:
        raise ValueError('Invalid native row count.')
    widths = {'native_tick': 1, 't_seconds': 1, 'object_position_m': 3,
              'grasp_contact_counts': 2, 'pad_normal_forces_n': 2,
              'table_object_contact_count': 1, 'pulse_active': 1,
              'pulse_minimum_opening_m': 1, 'applied_jaw_target_m': 1,
              'normal_rate_limited_jaw_target_m': 1}
    values = {}
    for name, width in widths.items():
        value = np.asarray(arrays.get(name))
        if (value.dtype.kind not in 'fiu' or not np.isfinite(value).all()
                or value.shape != ((rows,) if width == 1 else (rows, width))):
            raise ValueError('Invalid native array: '+name)
        values[name] = value
    ticks = values['native_tick']
    if (not np.array_equal(ticks, np.arange(1, rows+1))
            or not np.allclose(values['t_seconds'], ticks*.001, atol=1e-8, rtol=0)):
        raise ValueError('Invalid post-physics native tick cadence.')
    for name in ('grasp_contact_counts', 'table_object_contact_count', 'pulse_active'):
        value = values[name]
        if (np.any(value != np.floor(value)) or np.any(value < 0)
                or (name == 'pulse_active' and np.any(value > 1))):
            raise ValueError('Invalid native integer field: '+name)
    if np.any(values['pad_normal_forces_n'] < 0):
        raise ValueError('Negative pad force.')
    for name in ('pulse_minimum_opening_m', 'applied_jaw_target_m', 'normal_rate_limited_jaw_target_m'):
        if np.any(values[name] < -1e-12) or np.any(values[name] > .035+1e-12):
            raise ValueError('Jaw target outside bounds: '+name)
    # Deliberately reconstruct the row predicate and consecutive run locally.
    qualified = (np.all(values['grasp_contact_counts'] > 0, axis=1)
                 & np.all(values['pad_normal_forces_n'] >= parameters.force_threshold_n, axis=1)
                 & (values['table_object_contact_count'] == 0)
                 & (values['object_position_m'][:, 2]-initial_height >= parameters.lift_threshold_m))
    dwell, maximum, eligible_tick = 0, 0, None
    required = round(parameters.eligibility_dwell_s/.001)
    for index, value in enumerate(qualified):
        dwell = dwell+1 if value else 0
        maximum = max(maximum, dwell)
        if dwell >= required and eligible_tick is None:
            eligible_tick = index+1
    start = None if eligible_tick is None else eligible_tick+round(parameters.delay_s/.001)
    end = None if start is None else start+round(parameters.pulse_duration_s/.001)
    scheduled = bool(parameters.interruption and end is not None and end <= budget)
    if not scheduled:
        start = end = None
    # A row numbered T describes controls applied at pre-step tick T-1.
    expected_active = np.zeros(rows, dtype=bool) if not scheduled else ((ticks-1 >= start) & (ticks-1 < end))
    expected_floor = expected_active.astype(float)*parameters.minimum_opening_m
    active = values['pulse_active'] > 0
    effective = values['applied_jaw_target_m'] > values['normal_rate_limited_jaw_target_m']+1e-12
    first_effect = int(ticks[np.flatnonzero(effective)[0]]) if np.any(effective) else None
    reason = ('nominal_condition' if not parameters.interruption else
              'eligibility_not_reached' if eligible_tick is None else
              'insufficient_episode_budget' if not scheduled else
              'episode_ended_before_pulse' if not np.any(expected_active) else None)
    expected_event = None if eligible_tick is None else {
        'eligibility_native_tick': eligible_tick, 'eligibility_time_s': eligible_tick*.001,
        'pulse_start_tick': start, 'pulse_end_tick': end,
        'reason': 'scheduled' if scheduled else 'nominal_condition' if not parameters.interruption else 'insufficient_episode_budget'}
    mismatches = []
    checks = {'event': expected_event, 'events': [] if expected_event is None else [expected_event],
              'eligible': eligible_tick is not None, 'scheduled': scheduled,
              'activated': bool(np.any(active)), 'effective': bool(np.any(effective)),
              'active_native_ticks': int(np.count_nonzero(active)),
              'effective_native_ticks': int(np.count_nonzero(effective)),
              'first_effect_native_tick': first_effect, 'nonactivation_reason': reason}
    for name, expected in checks.items():
        if declaration.get(name) != expected:
            mismatches.append('metadata:'+name)
    # The environment stops updating the dwell once eligibility latches.
    expected_maximum = required if eligible_tick is not None else maximum
    if declaration.get('maximum_qualifying_dwell_native_ticks') != expected_maximum:
        mismatches.append('metadata:maximum_qualifying_dwell_native_ticks')
    expected_schedule = [] if not scheduled else [dict(start_tick=start, end_tick=end,
                                                      minimum_opening_m=parameters.minimum_opening_m)]
    expected_pulses = [] if not scheduled else [dict(start_s=start*.001, end_s=end*.001,
                                                    minimum_opening_m=parameters.minimum_opening_m)]
    if metadata.get('pulse_tick_schedule') != expected_schedule:
        mismatches.append('pulse_tick_schedule')
    if metadata.get('pulses') != expected_pulses:
        mismatches.append('pulses')
    if not np.array_equal(active, expected_active):
        mismatches.append('native:pulse_active')
    if not np.allclose(values['pulse_minimum_opening_m'], expected_floor, atol=1e-12, rtol=0):
        mismatches.append('native:pulse_minimum_opening_m')
    if np.any(effective & (ticks <= (start if scheduled else rows))):
        mismatches.append('native:effect_before_scheduled_pulse')
    return dict(schema=ANALYSIS_SCHEMA, valid=not mismatches, mismatches=mismatches,
                interruption_condition=parameters.interruption,
                eligible=eligible_tick is not None, eligibility_native_tick=eligible_tick,
                scheduled=scheduled, pulse_start_tick=start, pulse_end_tick=end,
                activated=bool(np.any(active)), effective=bool(np.any(effective)),
                active_native_ticks=int(np.count_nonzero(active)),
                expected_active_native_ticks=int(np.count_nonzero(expected_active)),
                effective_native_ticks=int(np.count_nonzero(effective)),
                first_effect_native_tick=first_effect, nonactivation_reason=reason,
                qualifying_native_ticks=int(np.count_nonzero(qualified)),
                maximum_qualifying_dwell_native_ticks=maximum,
                scope='Independent numeric-row reconstruction; not authenticated hardware sensing.')
