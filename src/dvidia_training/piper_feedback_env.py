"""Native PiPER telemetry and bounded, external jaw-actuator fault pulses.

This changes only the commanded jaw target during a declared fault and its
rate-limited closing tail. It supplies no recovery decisions or learner labels.
The base task, native scoring, robot gains, contacts and object physics remain
unchanged. Fault schedules and native sidecars are never policy observations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .piper_task import PiperTaskEnv
from . import reference_sim as sim


SCHEMA = 'dvidia.piper-feedback-native/v1'


@dataclass(frozen=True)
class JawPulse:
    """Half-open episode-time interval imposing a minimum jaw target in metres."""

    start_s: float
    end_s: float
    minimum_opening_m: float

    def __post_init__(self):
        for name in ('start_s', 'end_s', 'minimum_opening_m'):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError('Jaw pulse fields must be finite real numbers.')
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError('Jaw pulse must have 0 <= start_s < end_s.')
        if not 0 <= self.minimum_opening_m <= .035:
            raise ValueError('Jaw pulse minimum opening must be in [0,.035] metres.')


# Each row is a numeric native-tick snapshot after native physics and scoring.
# Width-one fields export as (ticks,), vector fields as (ticks,width).
NATIVE_FIELDS = (
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


class FeedbackPiperTaskEnv(PiperTaskEnv):
    """Drop-in task environment with immutable scheduled faults and dense logs.

    ``native_arrays()`` is an NPZ-ready numeric-only sidecar. Pair it with
    ``native_metadata()`` for units/order, the fault schedule and contact names.
    Both are analysis artifacts, never arguments to a learned policy.
    """

    def __init__(self, assets, config=None, *, pulses=()):
        super().__init__(assets, config)
        if type(pulses) not in (tuple, list) or any(type(p) is not JawPulse for p in pulses):
            raise ValueError('pulses must be a sequence of immutable JawPulse values.')
        self._pulses = tuple(sorted(pulses, key=lambda p: (p.start_s, p.end_s, p.minimum_opening_m)))
        self._pulse_ticks = []
        for pulse in self._pulses:
            if pulse.end_s > self.config.timeout_s + 1e-10:
                raise ValueError('Jaw pulse must fit inside the fixed episode budget.')
            ticks = []
            for seconds in (pulse.start_s, pulse.end_s):
                tick = seconds / self.config.timestep
                if not math.isclose(tick, round(tick), rel_tol=0., abs_tol=1e-8):
                    raise ValueError('Jaw pulse boundaries must be integer native ticks.')
                ticks.append(round(tick))
            self._pulse_ticks.append((*ticks, float(pulse.minimum_opening_m)))
        self._pulse_ticks = tuple(self._pulse_ticks)
        self._clear_sidecar()

    @property
    def pulses(self):
        return self._pulses

    def _clear_sidecar(self):
        self._native_rows = []
        self._native_tick = 0
        self._requested_jaw = .035
        self._applied_jaw = .035
        self._normal_jaw = .035
        self._pulse_active = False
        self._pulse_floor = 0.
        self._closing_tail = False
        self._last_contacts = None
        self._contact_names = {}

    def reset(self, seed=0):
        # Parent reset disables ready before settling; no fault or row can be
        # applied to the settling trajectory, including on repeated resets.
        self._clear_sidecar()
        try:
            observation, info = super().reset(seed)
        except Exception:
            self._clear_sidecar()
            raise
        self._applied_jaw = self._normal_jaw = self._requested_jaw = float(self.command_gripper)
        self._refresh_contact_names()
        return observation, info

    def _refresh_contact_names(self):
        mj, _ = sim._native_imports()
        rows = []
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            rows.append({'geom_id': geom,
                         'geom_name': mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom),
                         'body_id': body,
                         'body_name': mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body)})
        self._contact_names = {'geoms': rows, 'pad_geom_ids': {side: list(ids) for side, ids in self.pad_geoms.items()},
                               'block_geom_id': int(self.block_geom), 'table_geom_id': int(self.table_geom),
                               'floor_geom_id': int(self.floor_geom)}

    def _contacts(self):
        result = super()._contacts()
        self._last_contacts = result
        return result

    def step(self, action):
        # Base validation remains authoritative and rejects malformed actions.
        # Stage the requested target before the first native integration; undo
        # that staging when base validation rejects the action.
        previous = self._requested_jaw
        if type(action) is dict and type(action.get('gripper_position_m')) in (int, float):
            self._requested_jaw = float(action['gripper_position_m'])
        try:
            return super().step(action)
        except Exception:
            self._requested_jaw = previous
            raise

    def _controls(self):
        super()._controls()
        if not self.ready:
            # Settling follows the exact original controls.
            self._applied_jaw = self._normal_jaw = float(self.command_gripper)
            return
        normal = float(self.command_gripper)
        floors = [floor for start, end, floor in self._pulse_ticks if start <= self._native_tick < end]
        self._pulse_active, self._pulse_floor = bool(floors), max(floors, default=0.)
        desired = max(normal, self._pulse_floor)
        prior_detached = self._applied_jaw != self._normal_jaw
        self._normal_jaw = normal
        if not floors and not prior_detached:
            # Avoid even rounding differences in the no-fault condition.
            self._applied_jaw = normal
        else:
            maximum = self.config.jaw_target_rate_m_s * self.config.timestep
            distance = desired-self._applied_jaw
            at_limit = math.isclose(abs(distance), maximum, rel_tol=1e-12, abs_tol=0.)
            self._applied_jaw = desired if abs(distance) <= maximum or at_limit else self._applied_jaw+math.copysign(maximum, distance)
        self._closing_tail = not bool(floors) and self._applied_jaw != normal
        self.data.ctrl[6] = self._applied_jaw

    def _outcome_tick(self, **kwargs):
        # Score first, exactly as the base task does. Sidecars cannot alter any
        # success/retention latch, contact result or physics field.
        super()._outcome_tick(**kwargs)
        if not self.ready:
            return
        if self._last_contacts is None:
            raise RuntimeError('Native telemetry requires the current native contact snapshot.')
        counts, forces, supported, critical, _ = self._last_contacts
        mj, _ = sim._native_imports()
        tcp_quat = np.empty(4)
        mj.mju_mat2Quat(tcp_quat, self.data.site_xmat[self.tcp])
        velocity = np.empty(6)
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, self.block, velocity, 0)
        obj = np.asarray(self.data.xpos[self.block])
        tcp = np.asarray(self.data.site_xpos[self.tcp])
        local = np.asarray(self.data.site_xmat[self.tcp]).reshape(3, 3).T @ (obj-tcp)
        self._native_tick += 1
        row = (float(self.elapsed), self._native_tick, *obj, *self.data.xquat[self.block],
               *velocity[3:], *velocity[:3],
               *tcp, *tcp_quat, *local, counts['left'], counts['right'],
               forces['left'], forces['right'], supported, critical,
               *self.data.qpos[6:8], *self.data.qpos[:6], *self.data.ctrl[:6],
               self._requested_jaw, self._normal_jaw, float(self.data.ctrl[6]),
               int(self._pulse_active), self._pulse_floor, int(self._closing_tail))
        self._native_rows.append(row)

    def native_arrays(self):
        """Return copies suitable for ``np.savez_compressed(..., **arrays)``.

        Every full 20 s / 1 ms episode has 20,000 rows; shorter/invalid attempts
        retain their actual row count. No pickle/object arrays are emitted.
        """
        width = sum(size for _, size in NATIVE_FIELDS)
        matrix = np.asarray(self._native_rows, dtype=np.float64).reshape((-1, width))
        arrays, offset = {}, 0
        for name, size in NATIVE_FIELDS:
            values = matrix[:, offset:offset+size]
            arrays[name] = (values[:, 0] if size == 1 else values).copy()
            offset += size
        return arrays

    def native_metadata(self):
        return {'schema': SCHEMA, 'scope': 'Read-only native simulator sidecar; not policy observations or authenticated hardware sensing.',
                'sample_timing': 'Post-physics/post-scoring native ticks; schedule uses the pre-step tick and half-open [start,end) intervals.',
                'native_timestep_s': self.config.timestep, 'rows': len(self._native_rows),
                'fields': [{'name': name, 'width': width} for name, width in NATIVE_FIELDS],
                'side_order': ['left', 'right'], 'joint_order': ['joint'+str(i) for i in range(1, 7)],
                'jaw_order': ['joint7', 'joint8'], 'pulses': [asdict(pulse) for pulse in self._pulses],
                'pulse_tick_schedule': [{'start_tick': start, 'end_tick': end, 'minimum_opening_m': floor}
                                        for start, end, floor in self._pulse_ticks],
                'pulse_actuation': 'Applied jaw target=max(normal rate-limited target,pulse floor), additionally slew limited in both directions at the unchanged jaw target rate. A closing tail can outlast its scheduled pulse.',
                'jaw_target_rate_m_s': self.config.jaw_target_rate_m_s,
                'contact_names': self._contact_names,
                'unchanged': ['six arm actuator targets', 'native gains and force caps', 'physics and contacts',
                              'native success/retention scoring', 'policy observation fields'],
                'recovery_policy': None, 'physical_robot_ready': False}
