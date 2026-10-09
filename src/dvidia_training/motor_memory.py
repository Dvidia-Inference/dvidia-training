"""Compact one-axis motor memory and observation-only control, v0.2 research.

Calibration learns actuator dynamics, not task semantics. Predictive uncertainty
is a development envelope, not a confidence interval or hardware safety bound.
The independent reaction monitor remains outside this module.
"""
from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np


def _finite(value, name, *, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum) or (positive and result <= 0):
        raise ValueError(f'invalid {name}')
    return result


def _validated_numbers(instance, *, nonnegative=(), positive=()):
    for name in instance.__dataclass_fields__:
        if name in nonnegative or name in positive:
            object.__setattr__(instance, name, _finite(getattr(instance, name), name,
                minimum=0., positive=name in positive))


@dataclass(frozen=True)
class MotorObservation:
    acquired_s: float
    delivered_s: float
    position_m: float
    velocity_m_s: float
    position_uncertainty_m: float = 0.

    def __post_init__(self):
        _validated_numbers(self, nonnegative=('acquired_s', 'delivered_s', 'position_uncertainty_m'))
        for name in ('position_m', 'velocity_m_s'):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.delivered_s < self.acquired_s:
            raise ValueError('delivery cannot precede acquisition')


@dataclass(frozen=True)
class AppliedTarget:
    """Acknowledged application, not requested or queued command."""
    applied_s: float
    target_m: float

    def __post_init__(self):
        object.__setattr__(self, 'applied_s', _finite(self.applied_s, 'applied_s', minimum=0.))
        object.__setattr__(self, 'target_m', _finite(self.target_m, 'target_m'))


@dataclass(frozen=True)
class CalibrationTransition:
    position_m: float
    velocity_m_s: float
    applied_target_m: float
    next_velocity_m_s: float
    dt_s: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite(getattr(self, name), name,
                                                  positive=name == 'dt_s'))


@dataclass(frozen=True)
class ExpectedContract:
    """Authored outcomes; calibration does not learn these semantics."""
    goal_tolerance_m: float = .003
    velocity_tolerance_m_s: float = .005
    completion_dwell_s: float = .05
    holding_required: bool = False
    expected_contact: str = 'declared_by_task'

    def __post_init__(self):
        _validated_numbers(self, positive=('goal_tolerance_m', 'velocity_tolerance_m_s', 'completion_dwell_s'))
        if type(self.holding_required) is not bool or not isinstance(self.expected_contact, str) or not self.expected_contact.strip():
            raise ValueError('expected outcome metadata must be explicit')


@dataclass(frozen=True)
class MotorMemory:
    stiffness_s2: float
    damping_s: float
    bias_m_s2: float
    max_acceleration_m_s2: float
    acceleration_rmse_m_s2: float
    acceleration_error_bound_m_s2: float
    sample_count: int
    expected_contract: ExpectedContract = ExpectedContract()
    fit_scope: str = 'privileged_native_calibration_transitions'
    schema_version: str = 'motor-memory-v0.2'
    uncertainty_scope: str = 'development_envelope_not_confidence_interval'

    def __post_init__(self):
        _validated_numbers(self, positive=('stiffness_s2', 'max_acceleration_m_s2'),
            nonnegative=('damping_s', 'acceleration_rmse_m_s2', 'acceleration_error_bound_m_s2'))
        object.__setattr__(self, 'bias_m_s2', _finite(self.bias_m_s2, 'bias_m_s2'))
        if type(self.sample_count) is not int or self.sample_count < 12:
            raise ValueError('memory requires at least 12 calibration transitions')
        if self.acceleration_error_bound_m_s2 < self.acceleration_rmse_m_s2:
            raise ValueError('residual envelope must contain training RMSE')
        if not isinstance(self.expected_contract, ExpectedContract):
            raise ValueError('ExpectedContract required')
        if self.fit_scope != 'privileged_native_calibration_transitions' or self.schema_version != 'motor-memory-v0.2' or self.uncertainty_scope != 'development_envelope_not_confidence_interval':
            raise ValueError('unsupported memory scope or schema')

    def acceleration(self, position_m, velocity_m_s, applied_target_m):
        values = [_finite(x, n) for x, n in zip((position_m, velocity_m_s, applied_target_m),
                                               ('position_m', 'velocity_m_s', 'applied_target_m'))]
        raw = self.stiffness_s2*(values[2]-values[0])-self.damping_s*values[1]+self.bias_m_s2
        if not math.isfinite(raw):
            raise ValueError('actuator prediction overflow')
        return max(-self.max_acceleration_m_s2, min(self.max_acceleration_m_s2, raw))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('complete motor-memory schema required')
        data = dict(value)
        contract = data.get('expected_contract')
        if not isinstance(contract, Mapping) or set(contract) != set(ExpectedContract.__dataclass_fields__):
            raise ValueError('complete expected outcome schema required')
        data['expected_contract'] = ExpectedContract(**contract)
        return cls(**data)


def fit_memory(rows, *, expected_contract=None):
    """Fit saturated second-order dynamics from native calibration only.

Rows have the CalibrationTransition fields. Callers must keep final evaluation
episodes out of this function. The learned acceleration cap slightly exceeds the
largest calibration acceleration; it is not a guaranteed braking deceleration.
"""
    samples = []
    try:
        for row in rows:
            if isinstance(row, Mapping):
                if set(row) != set(CalibrationTransition.__dataclass_fields__):
                    raise ValueError('calibration transition fields required')
                row = CalibrationTransition(**row)
            if not isinstance(row, CalibrationTransition):
                raise ValueError('CalibrationTransition required')
            samples.append(row)
    except TypeError as exc:
        raise ValueError('calibration transitions must be iterable') from exc
    if len(samples) < 12:
        raise ValueError('at least 12 calibration transitions required')
    features = np.array([[r.applied_target_m-r.position_m, -r.velocity_m_s, 1.] for r in samples])
    accelerations = np.array([(r.next_velocity_m_s-r.velocity_m_s)/r.dt_s for r in samples])
    if not np.isfinite(features).all() or not np.isfinite(accelerations).all():
        raise ValueError('calibration arithmetic must remain finite')
    peak = float(np.max(np.abs(accelerations)))
    if peak <= 1e-9:
        raise ValueError('calibration must excite actuator dynamics')
    # Exclude observed saturation when enough unsaturated excitation is available.
    mask = np.abs(accelerations) < .9*peak
    if np.count_nonzero(mask) < 12 or np.linalg.matrix_rank(features[mask]) != 3:
        mask = np.ones(len(samples), dtype=bool)
    coefficients, _, rank, _ = np.linalg.lstsq(features[mask], accelerations[mask], rcond=None)
    if rank != 3 or not np.isfinite(coefficients).all() or coefficients[0] <= 0 or coefficients[1] < -1e-8:
        raise ValueError('calibration must identify stable position and damping terms')
    stiffness, damping, bias = (float(x) for x in coefficients)
    cap = peak*1.02
    prediction = np.clip(features @ np.array([stiffness, max(0., damping), bias]), -cap, cap)
    residual = accelerations-prediction
    rmse = float(np.sqrt(np.mean(residual**2)))
    bound = float(np.max(np.abs(residual)))
    return MotorMemory(stiffness, max(0., damping), bias, cap, rmse, max(bound, rmse),
                       len(samples), ExpectedContract() if expected_contract is None else expected_contract)


@dataclass(frozen=True)
class EstimatedState:
    position_m: float
    velocity_m_s: float
    uncertainty_m: float
    age_s: float


def _history(history, now):
    try:
        events = tuple(history)
    except TypeError as exc:
        raise ValueError('acknowledged target history required') from exc
    if not events or any(not isinstance(e, AppliedTarget) for e in events):
        raise ValueError('acknowledged target history required')
    if any(b.applied_s < a.applied_s for a, b in zip(events, events[1:])) or events[-1].applied_s > now+1e-12:
        raise ValueError('application history must be ordered and causal')
    return events


def _received(now, observation, applied_history, maximum_age):
    if not isinstance(observation, MotorObservation):
        raise ValueError('missing observation')
    if observation.delivered_s > now+1e-12:
        raise ValueError('observation is not delivered')
    age = max(0., now-observation.acquired_s)
    if age > maximum_age+1e-12:
        raise ValueError('stale observation')
    events = _history(applied_history, now)
    if not any(e.applied_s <= observation.acquired_s+1e-12 for e in events):
        raise ValueError('no acknowledged target at acquisition')
    return age, events


def estimate_state(now_s, observation, memory, applied_history, *, max_age_s=.15, step_s=.001):
    """Project a delayed sample using only already-applied command history."""
    now = _finite(now_s, 'now_s', minimum=0.)
    maximum_age = _finite(max_age_s, 'max_age_s', positive=True)
    step = _finite(step_s, 'step_s', positive=True)
    if not isinstance(memory, MotorMemory):
        raise ValueError('observation and learned memory required')
    age, events = _received(now, observation, applied_history, maximum_age)
    if age/step > 10000:
        raise ValueError('projection step budget exceeded')
    anchors = [i for i, e in enumerate(events) if e.applied_s <= observation.acquired_s+1e-12]
    if not anchors:
        raise ValueError('no acknowledged target at acquisition')
    index = anchors[-1]
    position, velocity, time = observation.position_m, observation.velocity_m_s, observation.acquired_s
    while time < now-1e-12:
        while index+1 < len(events) and events[index+1].applied_s <= time+1e-12:
            index += 1
        end = min(now, time+step, events[index+1].applied_s if index+1 < len(events) else now)
        dt = end-time
        if dt <= 0:
            raise ValueError('application history cannot advance projection')
        acceleration = memory.acceleration(position, velocity, events[index].target_m)
        position += velocity*dt+.5*acceleration*dt*dt
        velocity += acceleration*dt
        time = end
        if not math.isfinite(position) or not math.isfinite(velocity):
            raise ValueError('nonfinite projected state')
    uncertainty = (observation.position_uncertainty_m*(1+.5*memory.stiffness_s2*age*age)
        + .5*memory.acceleration_error_bound_m_s2*age*age
        + memory.max_acceleration_m_s2*step*age)
    if not math.isfinite(uncertainty):
        raise ValueError('nonfinite uncertainty')
    return EstimatedState(position, velocity, uncertainty, age)


@dataclass(frozen=True)
class ControllerConfig:
    workspace_min_m: float = -.4
    workspace_max_m: float = .6
    max_speed_m_s: float = .16
    slow_speed_factor: float = .5
    precision_speed_m_s: float = .025
    fixed_slowdown_distance_m: float = .035
    max_target_lead_m: float = .04
    braking_deceleration_m_s2: float = .5
    command_delay_s: float = .02
    max_sensor_age_s: float = .15
    prediction_step_s: float = .001
    uncertainty_scale_m: float = .003
    margin_m: float = .0002
    position_gain_s: float = 8.
    max_update_interval_s: float = .05

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite(getattr(self, name), name,
                minimum=None if name.startswith('workspace_') else 0.,
                positive=name not in ('workspace_min_m', 'workspace_max_m', 'command_delay_s', 'margin_m')))
        if self.workspace_min_m >= self.workspace_max_m:
            raise ValueError('nonempty workspace required')
        if self.slow_speed_factor > 1:
            raise ValueError('slow_speed_factor must not exceed one')


@dataclass(frozen=True)
class MotorCommand:
    target_m: float | None
    desired_speed_m_s: float
    estimated_state: EstimatedState | None
    halt: bool
    reason: str


class MotorController:
    """Bounded position-target primitive with three comparable speed modes.

    Braking deceleration is an explicit measured/configured assumption, separate
    from the learned acceleration cap. A halt has no target: the runner must apply
    its actuator's hold/stop contract rather than treating None as zero position.
    Halts latch until reset with fresh observations and acknowledged target state.
"""
    MODES = ('slow_feedback', 'fast_fixed', 'adaptive_predictive')

    def __init__(self, memory, mode='adaptive_predictive', config=ControllerConfig()):
        if not isinstance(memory, MotorMemory) or mode not in self.MODES or not isinstance(config, ControllerConfig):
            raise ValueError('valid memory, mode and ControllerConfig required')
        self.memory, self.mode, self.config = memory, mode, config
        self.reset()

    def reset(self):
        self._last_now = self._last_acquired = self._target = None
        self._halt_reason = None

    def update(self, now_s, observation, goal_m, applied_history):
        if self._halt_reason is not None:
            return MotorCommand(None, 0., None, True, self._halt_reason)
        try:
            now = _finite(now_s, 'now_s', minimum=0.)
            goal = _finite(goal_m, 'goal_m')
            if self._last_now is not None and now < self._last_now:
                raise ValueError('controller time regressed')
            if not isinstance(observation, MotorObservation):
                raise ValueError('missing observation')
            if self._last_acquired is not None and observation.acquired_s < self._last_acquired:
                raise ValueError('acquisition time regressed')
            c = self.config
            if not c.workspace_min_m <= goal <= c.workspace_max_m:
                raise ValueError('goal outside workspace')
            age, events = _received(now, observation, applied_history, c.max_sensor_age_s)
            state = estimate_state(now, observation, self.memory, events,
                max_age_s=c.max_sensor_age_s, step_s=c.prediction_step_s) if self.mode == 'adaptive_predictive' else EstimatedState(
                observation.position_m, observation.velocity_m_s, observation.position_uncertainty_m, age)
            if not c.workspace_min_m <= state.position_m <= c.workspace_max_m:
                raise ValueError('observed position outside workspace')
            dt = 0. if self._last_now is None else now-self._last_now
            if dt > c.max_update_interval_s+1e-12:
                raise ValueError('controller update deadline missed')
            target = events[-1].target_m if self._target is None else self._target
            if not c.workspace_min_m <= target <= c.workspace_max_m:
                raise ValueError('applied target outside workspace')
            distance = abs(goal-state.position_m)
            speed = min(c.max_speed_m_s, c.position_gain_s*distance)
            if self.mode == 'slow_feedback':
                speed = min(speed, c.max_speed_m_s*c.slow_speed_factor)
            elif self.mode == 'fast_fixed' and distance < c.fixed_slowdown_distance_m:
                speed = min(speed, c.precision_speed_m_s)
            elif self.mode == 'adaptive_predictive':
                available = max(0., distance-state.uncertainty_m-c.margin_m)
                delay = c.command_delay_s
                deceleration = c.braking_deceleration_m_s2
                safe_speed = math.sqrt((deceleration*delay)**2+2*deceleration*available)-deceleration*delay
                speed = min(speed, max(0., safe_speed), c.position_gain_s*distance)
                speed /= 1+state.uncertainty_m/c.uncertainty_scale_m
            direction = 1. if goal > target else -1. if goal < target else 0.
            requested = target+direction*min(abs(goal-target), speed*dt)
            # Stop advancing a reference that has outrun observed motion. Keep
            # the previous target rather than violating the target slew bound.
            if abs(requested-state.position_m) > c.max_target_lead_m and abs(requested-state.position_m) > abs(target-state.position_m):
                requested = target
            if not math.isfinite(requested) or not math.isfinite(speed):
                raise ValueError('nonfinite control output')
            self._last_now, self._last_acquired, self._target = now, observation.acquired_s, requested
            return MotorCommand(requested, direction*speed, state, False, 'tracking')
        except (ValueError, TypeError, OverflowError) as exc:
            # A restart needs a fresh acknowledged reference; no hidden cached
            # movement continues across a missing/invalid observation.
            self._target = None
            self._halt_reason = str(exc)
            return MotorCommand(None, 0., None, True, self._halt_reason)
