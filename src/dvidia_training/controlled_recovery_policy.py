"""Experiment-only supported lowering before opening; production is unchanged."""
from __future__ import annotations

import numpy as np

from .arm_policy import ArmPickPlacePolicy

LOWER_GOAL_CAP_M = .003
LOWER_TIMEOUT_SECONDS = 4.
SUPPORT_SAMPLES = 5
SUPPORT_HEIGHT_TOLERANCE_M = .008
SUPPORT_LINEAR_SPEED_M_S = .035
SUPPORT_ANGULAR_SPEED_RAD_S = .3


class ControlledRecoveryPolicy(ArmPickPlacePolicy):
    """Same normal controller, trigger and one-retry budget; new recovery only.

The inherited recovery action is computed but intercepted before env.step. No
student IK fallback is added: every lowering goal goes through the same proxy
movement mapping as ordinary motion. A failed support check holds rather than
opening. Native success and elapsed time are never changed.
"""
    def __init__(self, environment):
        super().__init__(environment)
        self._lower_pending = False
        self._lower_active = False
        self._lower_abort = False
        self._lower_xy = None
        self._lower_started = None
        self._lower_ticks = 0
        self._support_ticks = 0
        self._supported_release = None
        self._abort_hold_targets = None
        self._abort_hold_gap = None

    def _recover(self, reason, observation):
        before = self._recovery_count
        super()._recover(reason, observation)
        if self._recovery_count > before and self.failure_reason is None:
            self._lower_pending = True

    def _supported(self, observation):
        support_z = self.env.config.table_height + self.env.config.object_half_size[2]
        return bool(observation['table_contacts'] > 0
                    and abs(observation['object_position'][2] - support_z) < SUPPORT_HEIGHT_TOLERANCE_M
                    and np.linalg.norm(observation['object_velocity']) < SUPPORT_LINEAR_SPEED_M_S
                    and np.linalg.norm(observation['object_angular_velocity']) < SUPPORT_ANGULAR_SPEED_RAD_S)

    def _abort(self, observation, reason):
        self._lower_active = False
        self._lower_abort = True
        self._abort_hold_targets = list(observation['joint_position'])
        self._abort_hold_gap = float(min(observation['gripper_width'], self._grip_gap
                                        if self._grip_gap is not None else observation['gripper_width']))
        self._fail(reason, observation)
        return self._abort_action()

    def _abort_action(self):
        self._action_phase = 'recover_abort_hold'
        return {'joint_targets': list(self._abort_hold_targets), 'gripper_width': self._abort_hold_gap}

    def _lower(self, observation):
        self._action_phase = 'recover_lower_supported'
        self.stage = 'recover_lower_supported'
        self._measured_pad_forces = {side: float(observation['pad_normal_forces'][side]) for side in ('left', 'right')}
        self._support_ticks = self._support_ticks + 1 if self._supported(observation) else 0
        if self._support_ticks >= SUPPORT_SAMPLES:
            try:
                self.env.config._placement(observation['object_position'])
            except ValueError:
                return self._abort(observation, 'controlled_recovery_outside_workspace')
            if not self._axis_aligned(observation['object_quaternion']):
                return self._abort(observation, 'controlled_recovery_unsupported_orientation')
            self._supported_release = {
                'time': float(self.env._elapsed), 'table_contacts': observation['table_contacts'],
                'object_position': list(observation['object_position']),
                'object_velocity': list(observation['object_velocity']),
                'object_angular_velocity': list(observation['object_angular_velocity']),
                'consecutive_control_samples': self._support_ticks,
            }
            self._lower_active = False
            self._grip_gap = None
            self._held_offset = None
            self._recover_goal = np.asarray(observation['object_position']) + [0., 0., .15]
            self._next('recover_up')
            self._action_phase = 'recover_supported_open'
            # Open at the current pose only after support confirmation. The
            # existing recover_up phase resumes on the following control tick.
            self._last_goal = np.asarray(observation['end_effector_position'])
            return {'joint_targets': list(observation['joint_position']), 'gripper_width': .08}
        if self._lower_ticks * self.env.config.control_dt >= LOWER_TIMEOUT_SECONDS:
            return self._abort(observation, 'controlled_recovery_support_timeout')
        tcp = np.asarray(observation['end_effector_position'])
        obj = np.asarray(observation['object_position'])
        support_z = self.env.config.table_height + self.env.config.object_half_size[2]
        goal = np.array([*self._lower_xy, tcp[2] - max(0., obj[2] - support_z)])
        delta = goal - tcp
        distance = float(np.linalg.norm(delta))
        if distance > LOWER_GOAL_CAP_M:
            goal = tcp + delta * (LOWER_GOAL_CAP_M / distance)
        self._last_goal = goal.copy()
        self._lower_ticks += 1
        self._stage_ticks = self._lower_ticks
        return {'joint_targets': self.env.joint_targets_for_pose(goal),
                'gripper_width': self._grip_command(observation)}

    def __call__(self, observation):
        if self._lower_abort:
            return self._abort_action()
        if self._lower_active:
            return self._lower(observation)
        action = super().__call__(observation)
        if self._lower_pending:
            self._lower_pending = False
            self._lower_active = True
            self._lower_xy = list(observation['end_effector_position'][:2])
            self._lower_started = float(self.env._elapsed)
            self._lower_ticks = self._support_ticks = 0
            return self._lower(observation)
        return action

    def diagnostics(self):
        return {**super().diagnostics(),
                'experiment_policy': 'controlled-support-recovery-v1',
                'lower_started_seconds': self._lower_started,
                'lower_elapsed_seconds': self._lower_ticks * self.env.config.control_dt,
                'lower_goal_cap_m': LOWER_GOAL_CAP_M,
                'lower_timeout_seconds': LOWER_TIMEOUT_SECONDS,
                'support_samples': self._support_ticks,
                'supported_open_observation': self._supported_release,
                'controlled_recovery_abort': self._lower_abort}
