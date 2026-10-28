"""Bounded, authored acquisition supervision around the frozen v0.3 clock.

This is not a newly trained model. The old learned prediction is evaluated and
logged as advisory; the explicit supervisor controls progress. It permits
centering until supported bilateral contact is observed, waits for causal force
and tool-relative stability, then releases the original coordinated reference.
An unsuccessful wait is an explicit abort with capped measured-pose holding,
not a forced advance or a successful recovery. Privileged simulated contact
measurements do not certify a camera or physical-robot safety function.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from . import piper_grip_clock as frozen
from . import piper_grip_clock_model as clock_model
from . import piper_imitation as core


SCHEMA = 'dvidia.piper-bounded-grip-wait/v0.4'
INTERVAL_S = .02


@dataclass(frozen=True)
class GripWaitConfig:
    maximum_wait_s: float = .36
    gate_offset_s: float = 0.
    eligible_minimum_pad_force_n: float = .10
    secure_minimum_pad_force_n: float = .20
    secure_dwell_s: float = .10
    maximum_relative_drift_m: float = .002
    maximum_object_tcp_distance_m: float = .12
    closed_jaw_reference_m: float = .01

    def __post_init__(self):
        for name, low, high in (
                ('maximum_wait_s', .10, 1.), ('gate_offset_s', 0., .5),
                ('eligible_minimum_pad_force_n', .01, 2.),
                ('secure_minimum_pad_force_n', .01, 2.),
                ('secure_dwell_s', .04, .30),
                ('maximum_relative_drift_m', .0001, .02),
                ('maximum_object_tcp_distance_m', .005, .20),
                ('closed_jaw_reference_m', .001, .02)):
            core._number(getattr(self, name), name, low, high)
        for name in ('maximum_wait_s', 'gate_offset_s', 'secure_dwell_s'):
            value = getattr(self, name)
            if not math.isclose(value/INTERVAL_S, round(value/INTERVAL_S), abs_tol=1e-8):
                raise ValueError(f'{name} must use the 20 ms control grid.')
        if self.secure_minimum_pad_force_n < self.eligible_minimum_pad_force_n:
            raise ValueError('Secure force must be at least the eligibility force.')
        if self.maximum_wait_s < self.secure_dwell_s:
            raise ValueError('Maximum wait must accommodate a complete readiness dwell.')


def _tool_relative(observation):
    offset = (core._array(observation['object_position_m'], (3,), 'object position', 10.)
              - core._array(observation['tcp_position_m'], (3,), 'tool position', 10.))
    w, x, y, z = core._array(observation['tcp_quaternion_wxyz'], (4,), 'tool quaternion', 1.)
    if not math.isclose(w*w+x*x+y*y+z*z, 1., abs_tol=1e-6, rel_tol=0.):
        raise ValueError('Tool quaternion must be normalized.')
    rotation = np.asarray(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                           (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                           (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))
    return rotation.T@offset


class GuardedLearnedClock(frozen.LearnedClock):
    """Authored finite waiting policy; frozen learned output is advisory only.

    Progress remains binary 0/20 ms and never catches up. Successful runs reuse
    the original six-joint and jaw references. An abort freezes virtual progress
    and replaces only arm targets by the measured abort pose, respecting the
    same per-action joint and joint-limit caps on every subsequent observation.
    The runner must count aborts as failures even if a later pose looks placed.
    """
    def __init__(self, base_document, jaw_document, document, *, blind=False, config=None):
        if config is not None and type(config) is not GripWaitConfig:
            raise ValueError('Grip waiting requires a validated GripWaitConfig.')
        super().__init__(base_document, jaw_document, document, blind=blind)
        if self.jaw_mode != 'coordinated_reference':
            raise ValueError('Bounded grip waiting requires the frozen coordinated-reference jaw.')
        self.config = config or GripWaitConfig()
        self.gate = self.release = None
        self.history = deque()
        self.acquisition_started = False
        self.acquisition_released = False
        self.acquisition_started_at_s = None
        self.wait_ticks = 0
        self.maximum_wait_ticks = round(self.config.maximum_wait_s/INTERVAL_S)
        self.aborted = False
        self.abort_reason = None
        self.abort_time_s = None
        self.abort_pose_rad = None
        self.override_count = 0
        self.advisory_hold_count = 0
        self.secure_release_time_s = None

    def _readiness(self, observation):
        relative = _tool_relative(observation)
        counts = {side: core._integer(observation['grasp_contact_counts'][side],
                                      f'{side} pad contacts', 0, 10000)
                  for side in ('left', 'right')}
        forces = {side: core._number(observation['pad_normal_forces_n'][side],
                                     f'{side} pad force', 0., 1e6)
                  for side in ('left', 'right')}
        support = core._integer(observation['table_object_contact_count'],
                                'table contacts', 0, 10000) > 0
        bilateral = all(counts[s] > 0 and forces[s] >= self.config.eligible_minimum_pad_force_n
                        for s in counts)
        good = all(counts[s] > 0 and forces[s] >= self.config.secure_minimum_pad_force_n
                   for s in counts)
        t = observation['t_seconds']
        if not good:
            self.history.clear()
        self.history.append((t, relative.copy(), good))
        while len(self.history) > 1 and self.history[1][0] <= t-self.config.secure_dwell_s+1e-8:
            self.history.popleft()
        duration = t-self.history[0][0]
        drift = max(float(np.linalg.norm(row[1]-relative)) for row in self.history)
        near = float(np.linalg.norm(relative)) <= self.config.maximum_object_tcp_distance_m
        secure = bool(near and good and all(row[2] for row in self.history)
                      and duration+1e-8 >= self.config.secure_dwell_s
                      and drift <= self.config.maximum_relative_drift_m)
        return dict(secure_grip=secure, grip_dwell_s=duration, relative_drift_m=drift,
                    bilateral_contact=bilateral, support_observed=support, near_tool=near,
                    object_tool_relative_m=relative.tolist())

    def _abort(self, observation, reason):
        self.aborted = True
        self.abort_reason = reason
        self.abort_time_s = observation['t_seconds']
        self.abort_pose_rad = core._array(observation['joint_position_rad'], (6,),
                                         'abort joint position', 10.).copy()

    def _advance(self, observation, feature):
        if self.gate is None:
            self.gate, self.release = self.path.grasp_window(self.config.gate_offset_s)
        learned = clock_model.predict(self.model, feature)
        learner_advance = learned['advance']
        self.advisory_hold_count += int(not learner_advance)
        readiness = self._readiness(observation)
        in_window = self.gate <= self.progress_ticks < self.release
        raw_jaw = float(core.JAW_MAX_M*np.clip(self.path.raw(self.progress_ticks*INTERVAL_S)[6], 0., 1.))
        closed = raw_jaw < self.config.closed_jaw_reference_m
        eligible = bool(in_window and closed and readiness['bilateral_contact']
                        and readiness['support_observed'] and readiness['near_tool'])
        if self.aborted:
            advance, reason = False, 'aborted_pose_hold'
        elif self.acquisition_released:
            advance, reason = True, 'secure_grip_latched'
        elif in_window and readiness['secure_grip'] and (eligible or self.acquisition_started):
            self.acquisition_released = True
            self.secure_release_time_s = observation['t_seconds']
            advance, reason = True, 'secure_grip_confirmed'
        elif in_window and closed and not readiness['support_observed']:
            # A 20 ms observation can only react after support loss is measured.
            # This conservative abort also catches never-eligible acquisitions;
            # it may reject a benign table-contact bounce and is not predictive.
            self._abort(observation, 'support_lost_before_secure')
            advance, reason = False, self.abort_reason
        else:
            if eligible and not self.acquisition_started:
                self.acquisition_started = True
                self.acquisition_started_at_s = observation['t_seconds']
            if self.acquisition_started:
                if not readiness['support_observed']:
                    self._abort(observation, 'support_lost_before_secure')
                    advance, reason = False, self.abort_reason
                elif self.wait_ticks >= self.maximum_wait_ticks:
                    self._abort(observation, 'wait_budget_exhausted')
                    advance, reason = False, self.abort_reason
                else:
                    self.wait_ticks += 1
                    advance, reason = False, 'supervised_acquisition_wait'
            else:
                advance = True
                reason = 'centering_without_bilateral_support' if in_window else 'outside_acquisition_window'
        overridden = bool(advance != learner_advance)
        self.override_count += int(overridden)
        return advance, dict(
            decision_origin='authored_bounded_grip_supervisor',
            reason=reason, learner_advance=learner_advance,
            advance_probability=learned['advance_probability'], learned_prediction_advisory=True,
            supervisor_override=overridden, in_grasp_window=in_window,
            raw_jaw_closed=closed, eligible_for_wait=eligible,
            gate_progress_s=self.gate*INTERVAL_S, release_progress_s=self.release*INTERVAL_S,
            acquisition_started=self.acquisition_started, acquisition_released=self.acquisition_released,
            acquisition_started_at_s=self.acquisition_started_at_s,
            secure_release_time_s=self.secure_release_time_s,
            wait_ticks=self.wait_ticks, wait_seconds=self.wait_ticks*INTERVAL_S,
            maximum_wait_s=self.config.maximum_wait_s,
            aborted=self.aborted, abort_reason=self.abort_reason, abort_time_s=self.abort_time_s,
            abort_pose_hold=self.aborted, **readiness)

    def __call__(self, observation):
        began = time.perf_counter()
        action = super().__call__(observation)
        # _ClockMotion has measured its work already; add only the abort target
        # override cost so its diagnostic timing remains whole-policy timing.
        override_began = time.perf_counter()
        if self.aborted:
            q = core._array(observation['joint_position_rad'], (6,), 'joint position', 10.)
            limits = np.asarray(core.JOINT_LIMITS)
            action['joint_targets_rad'] = np.clip(
                q+np.clip(self.abort_pose_rad-q, -core.JOINT_DELTA_RAD, core.JOINT_DELTA_RAD),
                limits[:, 0]+.02, limits[:, 1]-.02).tolist()
        self.inference_seconds += time.perf_counter()-override_began
        self.last_decision['whole_policy_seconds'] = time.perf_counter()-began
        return action

    def diagnostics(self):
        return dict(super().diagnostics(), schema=SCHEMA,
                    authored_acquisition_supervisor=True, learned_prediction_advisory=True,
                    newly_trained_model=False, config=asdict(self.config),
                    acquisition_started=self.acquisition_started,
                    acquisition_released=self.acquisition_released,
                    acquisition_started_at_s=self.acquisition_started_at_s,
                    secure_release_time_s=self.secure_release_time_s,
                    wait_ticks=self.wait_ticks, wait_seconds=self.wait_ticks*INTERVAL_S,
                    advisory_hold_count=self.advisory_hold_count,
                    supervisor_override_count=self.override_count,
                    aborted=self.aborted, abort_reason=self.abort_reason, abort_time_s=self.abort_time_s,
                    original_arm_path_preserved=not self.aborted,
                    abort_changes_arm_reference=self.aborted,
                    native_command_caps_preserved=True,
                    physical_robot_ready=False)
