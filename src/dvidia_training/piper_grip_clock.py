"""Causal hold/advance timing around unchanged PiPER path and jaw models.

Arm geometry is the frozen primitive evaluated at explicitly logged virtual
progress. Coordinated-reference studies use that progress for the original jaw
reference too. The separately declared legacy architecture keeps the exact v0.2
jaw policy on physical time. The authored teacher is collection-only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np

from . import piper_motor_primitive as primitive
from . import piper_grip_adapter as grip
from . import piper_grip_clock_model as clock_model
from . import piper_imitation as core


SCHEMA = 'dvidia.piper-grip-clock-runtime/v0.3'
REFERENCE_JAW_SCHEMA = 'dvidia.piper-reference-jaw/v1'


def reference_jaw_manifest(base_document):
    return dict(schema=REFERENCE_JAW_SCHEMA,
                base_sha256=core.digest(primitive.validate_model(base_document)),
                mode='coordinated_reference', physical_robot_ready=False)


def validate_jaw_document(base_document, document):
    """Two explicit architectures; never reinterpret the v0.2 jaw tree."""
    value = core._decode(document)
    if isinstance(value, dict) and value.get('schema') == REFERENCE_JAW_SCHEMA:
        expected = reference_jaw_manifest(base_document)
        if value != expected or any(type(value[key]) is not type(item) for key, item in expected.items()):
            raise ValueError('Reference-jaw manifest differs from the exact frozen base binding.')
        return 'coordinated_reference', value
    value = grip.validate_model(value)
    if value['blind'] or value['base_sha256'] != core.digest(primitive.validate_model(base_document)):
        raise ValueError('Every physical-jaw clock variant requires the same full-feedback v0.2 jaw.')
    return 'physical_adapter', value


@dataclass(frozen=True)
class TeacherConfig:
    gate_offset_s: float = .16
    minimum_pad_force_n: float = .20
    secure_dwell_s: float = .10
    maximum_relative_drift_m: float = .002
    acquisition_only: bool = True

    def __post_init__(self):
        if type(self.acquisition_only) is not bool:
            raise ValueError('Acquisition-only mode must be a boolean.')
        for name, low, high in (('gate_offset_s', -.2, .5), ('minimum_pad_force_n', .01, 2.),
                                ('secure_dwell_s', .04, .3), ('maximum_relative_drift_m', .0001, .02)):
            core._number(getattr(self, name), name, low, high)
        for value in (self.gate_offset_s, self.secure_dwell_s):
            if not math.isclose(value/.02, round(value/.02), abs_tol=1e-8):
                raise ValueError('Teacher times must use the 20 ms control grid.')


class FrozenPath:
    """No IK or new path fit: evaluate original six references with native caps."""
    def __init__(self, document):
        self.model = primitive.validate_model(document)
        self.weights = np.asarray(self.model['weights'])
        self.high = self.model['schema'] == primitive.HIGH_RESOLUTION_SCHEMA
        self.xy = self.goal = None

    def latch(self, observation):
        if self.xy is None:
            self.xy = primitive._start(observation['object_position_m']).copy()
            self.goal = primitive._goal(observation['target_position_m'])
            self.coefficient = np.einsum('s,stj->tj', primitive._spatial(self.xy, self.high), self.weights[self.goal])
        elif primitive._goal(observation['target_position_m']) != self.goal:
            raise ValueError('Frozen path goal changed.')

    def raw(self, seconds):
        core._number(seconds, 'virtual reference seconds', 0., 20.)
        if self.xy is None:
            raise ValueError('Frozen path needs its initial observation.')
        # Same contraction order as the original primitive, including rounding.
        return np.einsum('s,t,stj->j', primitive._spatial(self.xy, self.high),
                         primitive._time_basis(seconds, self.high), self.weights[self.goal], optimize=True)

    def targets(self, observation, seconds):
        reference = self.raw(seconds)
        q = core._array(observation['joint_position_rad'], (6,), 'joint position', 10.)
        limits = np.asarray(core.JOINT_LIMITS)
        return np.clip(q+np.clip(reference[:6]*3.-q, -core.JOINT_DELTA_RAD, core.JOINT_DELTA_RAD),
                       limits[:, 0]+.02, limits[:, 1]-.02).tolist()

    def grasp_window(self, offset_s):
        values = [float(core.JAW_MAX_M*np.clip(self.raw(tick*.02)[6], 0., 1.)) for tick in range(1001)]
        closed = next((i for i, value in enumerate(values) if value < .01), None)
        if closed is None:
            raise ValueError('Frozen reference has no grasp-closing interval.')
        release = next((i for i in range(closed+1, 1001) if values[i] > .03), None)
        gate = closed+round(offset_s/.02)
        if release is None or not 0 <= gate < release:
            raise ValueError('Teacher gate must precede the original release interval.')
        return gate, release


class _ClockMotion:
    def __init__(self, base_document, jaw_document, *, blind=False):
        self.path = FrozenPath(base_document)
        self.jaw_mode, self.jaw_document = validate_jaw_document(base_document, jaw_document)
        self.jaw = (grip.GripPolicy(base_document, self.jaw_document, blind=False)
                    if self.jaw_mode == 'physical_adapter' else None)
        self.physical_nominal = primitive.MotorPrimitivePolicy(base_document)
        self.encoder = clock_model.ClockFeatures(blind=blind)
        self.progress_ticks, self.calls = 0, 0
        self.inference_seconds = 0.
        self.last_decision = None

    def __call__(self, observation):
        began = time.perf_counter()
        self.path.latch(observation)
        nominal = self.physical_nominal(observation)
        jaw_action = self.jaw(observation) if self.jaw is not None else nominal
        before = self.progress_ticks*.02
        feature = self.encoder.features(observation, nominal, before)
        advance, details = self._advance(observation, feature)
        if type(advance) is not bool:
            raise ValueError('Clock decisions are binary hold/advance only.')
        # Unit-clock controls reuse the original physical-time arm command
        # exactly. A delayed clock changes reference timing, not geometry.
        targets = (jaw_action['joint_targets_rad'] if self.progress_ticks == self.calls
                   else self.path.targets(observation, before))
        jaw_target = jaw_action['gripper_position_m']
        if self.jaw_mode == 'coordinated_reference' and self.progress_ticks != self.calls:
            jaw_target = float(core.JAW_MAX_M*np.clip(self.path.raw(before)[6], 0., 1.))
        after = min(1000, self.progress_ticks+int(advance))
        self.last_decision = dict(physical_time_s=observation['t_seconds'], progress_before_s=before,
            progress_after_s=after*.02, increment_s=(after-self.progress_ticks)*.02,
            features=feature.tolist(), nominal_physical_jaw_m=nominal['gripper_position_m'],
            frozen_jaw_target_m=jaw_target, jaw_mode=self.jaw_mode, **details)
        self.progress_ticks = after
        self.calls += 1
        self.inference_seconds += time.perf_counter()-began
        return dict(joint_targets_rad=targets, gripper_position_m=jaw_target)

    def diagnostics(self):
        return dict(schema=SCHEMA, progress_seconds=self.progress_ticks*.02,
            physical_action_calls=self.calls, inference_seconds=self.inference_seconds,
            original_arm_path_preserved=True, jaw_mode=self.jaw_mode,
            frozen_jaw_clock='physical_elapsed_time' if self.jaw is not None else 'shared_virtual_progress',
            frozen_jaw_model_sha256=core.digest(self.jaw_document),
            base_model_sha256=core.digest(self.path.model), physical_robot_ready=False)


class FixedClock(_ClockMotion):
    def _advance(self, observation, feature):
        return True, dict(decision_origin='fixed_unit_progress')


class TeacherClock(_ClockMotion):
    """Collection-only gate, using causal supported-grip readiness before lift.

    This signal does not certify a physical grasp. It requires bilateral pad
    contacts and force dwell plus small object/tool relative drift. Table
    support is allowed here; the external airborne disturbance has its own
    eligibility rule. Every decision, including indefinite holds, is logged.
    """
    def __init__(self, base_document, jaw_document, *, config=None):
        super().__init__(base_document, jaw_document)
        self.config = config or TeacherConfig()
        self.history = deque()
        self.gate = self.release = None
        self.acquisition_released = False

    def _advance(self, observation, feature):
        if self.gate is None:
            self.gate, self.release = self.path.grasp_window(self.config.gate_offset_s)
        t = observation['t_seconds']
        offset = np.asarray(observation['object_position_m'])-np.asarray(observation['tcp_position_m'])
        w, x, y, z = core._array(observation['tcp_quaternion_wxyz'], (4,), 'tool quaternion', 1.)
        if not math.isclose(w*w+x*x+y*y+z*z, 1., abs_tol=1e-6, rel_tol=0.):
            raise ValueError('Tool quaternion must be normalized.')
        rotation = np.asarray(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                               (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                               (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))
        relative = rotation.T@offset
        good = all(observation['grasp_contact_counts'][side] > 0 and
                   observation['pad_normal_forces_n'][side] >= self.config.minimum_pad_force_n
                   for side in ('left', 'right'))
        if not good:
            self.history.clear()
        self.history.append((t, relative.copy(), good))
        # Include the current sample and the endpoint at least dwell seconds
        # earlier: six 20ms observations establish a 100ms boundary span.
        while len(self.history) > 1 and self.history[1][0] <= t-self.config.secure_dwell_s+1e-8:
            self.history.popleft()
        duration = t-self.history[0][0]
        drift = max((float(np.linalg.norm(x-relative)) for _, x, _ in self.history), default=0.)
        secure = bool(good and all(row[2] for row in self.history)
                      and duration+1e-8 >= self.config.secure_dwell_s
                      and drift <= self.config.maximum_relative_drift_m)
        in_window = self.gate <= self.progress_ticks < self.release
        if in_window and secure:
            self.acquisition_released = True
        advance = bool(not in_window or secure or self.config.acquisition_only and self.acquisition_released)
        return advance, dict(decision_origin='authored_causal_grip_teacher', secure_grip=secure,
            grip_dwell_s=duration, relative_drift_m=drift, in_grasp_window=in_window,
            acquisition_released=self.acquisition_released,
            gate_progress_s=self.gate*.02, release_progress_s=self.release*.02)


class LearnedClock(_ClockMotion):
    def __init__(self, base_document, jaw_document, document, *, blind=False):
        self.model = clock_model.validate_model(document)
        mode, validated_jaw = validate_jaw_document(base_document, jaw_document)
        expected_schema = ('dvidia.piper-grip-progress-clock/v2' if mode == 'coordinated_reference'
                           else clock_model.SCHEMA)
        if (self.model['blind'] != blind or self.model['base_sha256'] != core.digest(primitive.validate_model(base_document))
                or self.model['jaw_sha256'] != core.digest(validated_jaw) or self.model['schema'] != expected_schema):
            raise ValueError('Clock mode/base/jaw identity differs from its trained model.')
        super().__init__(base_document, jaw_document, blind=blind)

    def _advance(self, observation, feature):
        result = clock_model.predict(self.model, feature)
        return result['advance'], dict(decision_origin='learned_blind_clock' if self.model['blind'] else 'learned_grip_clock',
                                      advance_probability=result['advance_probability'])
