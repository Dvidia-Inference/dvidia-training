"""Observation-only reaction primitives, v0.1 (simulation research).

Sphere envelopes and constant velocity prediction are approximations. The braking
envelope uses configured deceleration, not a measured or certified stopping bound.
These primitives are not a hardware safety controller.
"""
from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Mapping


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


def _vector(value, name):
    try:
        result = tuple(value)
        if len(result) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in result):
            raise ValueError(f'{name} must have three finite components')
        result = tuple(float(x) for x in result)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must have three finite components') from exc
    if any(not math.isfinite(x) for x in result):
        raise ValueError(f'{name} must have three finite components')
    return result


@dataclass(frozen=True)
class RobotSphere:
    center_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    radius_m: float

    def __post_init__(self):
        object.__setattr__(self, 'center_m', _vector(self.center_m, 'center_m'))
        object.__setattr__(self, 'velocity_m_s', _vector(self.velocity_m_s, 'velocity_m_s'))
        object.__setattr__(self, 'radius_m', _number(self.radius_m, 'radius_m', positive=True))


@dataclass(frozen=True)
class ObstacleSphere(RobotSphere):
    """An observed obstacle envelope; stationary obstacles have zero velocity."""


@dataclass(frozen=True)
class MotionSample:
    robot_spheres: tuple[RobotSphere, ...]
    obstacle_spheres: tuple[ObstacleSphere, ...]

    def __post_init__(self):
        try:
            robots, obstacles = tuple(self.robot_spheres), tuple(self.obstacle_spheres)
        except TypeError as exc:
            raise ValueError('sphere collections must be iterable') from exc
        if not robots or any(type(x) is not RobotSphere for x in robots):
            raise ValueError('robot_spheres must contain RobotSphere values')
        if any(type(x) is not ObstacleSphere for x in obstacles):
            raise ValueError('obstacle_spheres must contain ObstacleSphere values')
        object.__setattr__(self, 'robot_spheres', robots)
        object.__setattr__(self, 'obstacle_spheres', obstacles)


@dataclass(frozen=True)
class ContactSample:
    unexpected_force_n: float

    def __post_init__(self):
        object.__setattr__(self, 'unexpected_force_n', _number(self.unexpected_force_n, 'unexpected_force_n'))


@dataclass(frozen=True)
class GripSample:
    holding: bool
    slip_speed_m_s: float

    def __post_init__(self):
        if type(self.holding) is not bool:
            raise ValueError('holding must be boolean')
        object.__setattr__(self, 'slip_speed_m_s', _number(self.slip_speed_m_s, 'slip_speed_m_s'))


Payload = MotionSample | ContactSample | GripSample


@dataclass(frozen=True)
class Signal:
    acquired_s: float
    value: Payload

    def __post_init__(self):
        object.__setattr__(self, 'acquired_s', _number(self.acquired_s, 'acquired_s'))
        if not isinstance(self.value, (MotionSample, ContactSample, GripSample)):
            raise ValueError('unsupported signal payload')


@dataclass(frozen=True)
class SensorConfig:
    period_s: float = .02
    delivery_delay_s: float = 0.
    position_noise_m: float = 0.
    velocity_noise_m_s: float = 0.
    force_noise_n: float = 0.
    slip_noise_m_s: float = 0.
    dropout_probability: float = 0.

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=name == 'period_s'))
        if self.dropout_probability > 1:
            raise ValueError('dropout_probability must be between zero and one')


class SensorStream:
    """Sample at actual update times; transport samples with explicit missingness.

    A late update does not fabricate past acquisitions. Random draws occur only
    on acquisition, so extra polls between acquisitions do not change results.
    The caller supplies already-observed values. Noise is uniform and bounded;
    force/slip magnitude noise is clipped at zero. Startup returns None until the
    first delivery. A delivered dropout clears the previous held observation.
    """

    def __init__(self, config: SensorConfig = SensorConfig(), *, seed: int = 0):
        if not isinstance(config, SensorConfig) or type(seed) is not int:
            raise ValueError('valid SensorConfig and integer seed required')
        self.config, self._rng = config, random.Random(seed)
        self._pending = deque()
        self._last_update = self._last_acquired = None
        self.latest: Signal | None = None
        self.sample_count = 0

    def _noise(self, value):
        c, rng = self.config, self._rng
        def jitter(xs, bound):
            return tuple(x + rng.uniform(-bound, bound) for x in xs)
        if isinstance(value, MotionSample):
            def sphere(x):
                return type(x)(jitter(x.center_m, c.position_noise_m),
                               jitter(x.velocity_m_s, c.velocity_noise_m_s), x.radius_m)
            return MotionSample(tuple(sphere(x) for x in value.robot_spheres),
                                tuple(sphere(x) for x in value.obstacle_spheres))
        if isinstance(value, ContactSample):
            return ContactSample(max(0., value.unexpected_force_n + rng.uniform(-c.force_noise_n, c.force_noise_n)))
        return GripSample(value.holding, max(0., value.slip_speed_m_s + rng.uniform(-c.slip_noise_m_s, c.slip_noise_m_s)))

    def update(self, now_s: float, value: Payload | None) -> Signal | None:
        now = _number(now_s, 'now_s')
        if self._last_update is not None and now < self._last_update:
            raise ValueError('sensor update time must be nondecreasing')
        if value is not None and not isinstance(value, (MotionSample, ContactSample, GripSample)):
            raise ValueError('unsupported sensor payload')
        self._last_update = now
        if self._last_acquired is None or now - self._last_acquired >= self.config.period_s - 1e-12:
            self._last_acquired = now
            self.sample_count += 1
            dropped = self._rng.random() < self.config.dropout_probability
            signal = None if dropped or value is None else Signal(now, self._noise(value))
            self._pending.append((now + self.config.delivery_delay_s, signal))
        while self._pending and self._pending[0][0] <= now + 1e-12:
            _, self.latest = self._pending.popleft()
        return self.latest


@dataclass(frozen=True)
class ReactionConfig:
    horizon_s: float = .5
    margin_m: float = .02
    assumed_deceleration_m_s2: float = 2.
    command_delay_s: float = .02
    max_sensor_age_s: float = .1
    contact_force_limit_n: float = 10.
    slip_speed_limit_m_s: float = .01
    require_motion: bool = True
    require_contact: bool = True

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name.startswith('require_'):
                if type(getattr(self, name)) is not bool:
                    raise ValueError(f'{name} must be boolean')
            else:
                object.__setattr__(self, name, _number(getattr(self, name), name,
                                  positive=name in ('horizon_s', 'assumed_deceleration_m_s2')))


@dataclass(frozen=True)
class Decision:
    stop: bool
    reason: str
    detected_s: float | None
    minimum_predicted_clearance_m: float | None = None
    signal_name: str | None = None
    signal_acquired_s: float | None = None


class ReactionMonitor:
    """Latched decisions based only on delivered observations.

    Closest approach uses measured relative velocity over a finite horizon. Its
    surface clearance is compared with a conservative distance envelope:
    margin + robot delay travel + robot braking distance + obstacle travel during
    stopping. This can deliberately trigger early, including for crossing or
    receding trajectories near the envelope. All assumptions require validation.
    """

    def __init__(self, config: ReactionConfig = ReactionConfig()):
        if not isinstance(config, ReactionConfig):
            raise ValueError('valid ReactionConfig required')
        self.config = config
        self.reset()

    def reset(self):
        self._latched = None
        self._last_update = None
        self._holding = False
        self._last_acquired = {}

    def _prediction(self, motion, age):
        c, minimum, hazard = self.config, None, None
        for robot in motion.robot_spheres:
            speed = math.sqrt(sum(x*x for x in robot.velocity_m_s))
            stop_time = c.command_delay_s + speed / c.assumed_deceleration_m_s2
            for obstacle in motion.obstacle_spheres:
                velocity = tuple(o-r for o, r in zip(obstacle.velocity_m_s, robot.velocity_m_s))
                offset = tuple(o-r+v*age for o, r, v in zip(obstacle.center_m, robot.center_m, velocity))
                vv = sum(v*v for v in velocity)
                closest_t = max(0., min(c.horizon_s, -sum(p*v for p, v in zip(offset, velocity))/vv)) if vv else 0.
                clearance = math.sqrt(sum((p+v*closest_t)**2 for p, v in zip(offset, velocity))) - robot.radius_m - obstacle.radius_m
                minimum = clearance if minimum is None else min(minimum, clearance)
                obstacle_speed = math.sqrt(sum(v*v for v in obstacle.velocity_m_s))
                envelope = c.margin_m + speed*c.command_delay_s + speed*speed/(2*c.assumed_deceleration_m_s2) + obstacle_speed*stop_time
                if clearance <= envelope and hazard is None:
                    hazard = 'predicted_collision_moving' if obstacle_speed > 1e-12 else 'predicted_collision_static'
        return minimum, hazard

    def update(self, now_s: float, observations: Mapping[str, Signal | None], *, holding: bool | None = None) -> Decision:
        now = _number(now_s, 'now_s')
        if self._last_update is not None and now < self._last_update:
            raise ValueError('monitor update time must be nondecreasing')
        if holding is not None and type(holding) is not bool:
            raise ValueError('holding must be boolean or None')
        if not isinstance(observations, Mapping):
            raise ValueError('observations must be a mapping')
        expected = {'motion': MotionSample, 'contact': ContactSample, 'grip': GripSample}
        for name, signal in observations.items():
            if name not in expected or (signal is not None and (not isinstance(signal, Signal) or not isinstance(signal.value, expected[name]))):
                raise ValueError(f'invalid observation {name}')
            if signal is not None:
                if signal.acquired_s > now + 1e-12 or signal.acquired_s < self._last_acquired.get(name, 0.):
                    raise ValueError(f'{name} acquisition time must be causal and nondecreasing')
        self._last_update = now
        for name, signal in observations.items():
            if signal is not None:
                self._last_acquired[name] = signal.acquired_s
        if self._latched is not None:
            return self._latched
        grip = observations.get('grip')
        if holding is not None:
            self._holding = holding
        elif grip is not None and now-grip.acquired_s <= self.config.max_sensor_age_s + 1e-12:
            self._holding = grip.value.holding
        motion = observations.get('motion')
        clearance, prediction = (None, None) if motion is None else self._prediction(motion.value, now-motion.acquired_s)
        def stopped(reason, name):
            signal = observations.get(name)
            self._latched = Decision(True, reason, now, clearance, name, None if signal is None else signal.acquired_s)
            return self._latched
        for name, required in (('motion', self.config.require_motion), ('contact', self.config.require_contact), ('grip', self._holding)):
            signal = observations.get(name)
            if required and signal is None:
                return stopped(f'missing_{name}', name)
            if required and now-signal.acquired_s > self.config.max_sensor_age_s + 1e-12:
                return stopped(f'stale_{name}', name)
        contact = observations.get('contact')
        if contact is not None and now-contact.acquired_s <= self.config.max_sensor_age_s + 1e-12 and contact.value.unexpected_force_n > self.config.contact_force_limit_n:
            return stopped('unexpected_contact', 'contact')
        if self._holding and grip is not None and not grip.value.holding:
            return stopped('grip_lost', 'grip')
        if self._holding and grip is not None and grip.value.slip_speed_m_s > self.config.slip_speed_limit_m_s:
            return stopped('grip_slip', 'grip')
        if motion is not None and now-motion.acquired_s <= self.config.max_sensor_age_s + 1e-12 and prediction:
            return stopped(prediction, 'motion')
        return Decision(False, 'clear', None, clearance)
