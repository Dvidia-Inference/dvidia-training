"""Experiment-only gravity-aligned transfer compliance with fixed native limits."""
from __future__ import annotations

import numpy as np

from .arm_policy import ArmPickPlacePolicy
from .grip_diagnostic import GoalProxy

LOW_HEADROOM_MARGIN = 1.25
VERTICAL_ERROR_GAIN = .5
VERTICAL_INCREMENT_CAP_M = .002
TOTAL_INCREMENT_CAP_M = .005
MAX_DOWNWARD_EXCURSION_M = .025
MIN_OBJECT_CLEARANCE_M = .075
TRANSFER_PLANAR_TOLERANCE_M = .012


class CompliantTransferGoalProxy(GoalProxy):
    """Move the hand toward its original object-relative grasp height.

Only physical target commands change. Observed object state, held-offset guard,
force caps, native success and deadlines remain unchanged. This does not assert
that tracking a slipping object is sufficient retention or task completion.
"""
    def __init__(self, environment, head=None):
        super().__init__(environment, head)
        self.entry_tcp_z = None
        self.last_intervention = None
        self.first_intervention_time = None

    def eligible(self):
        c, obs = self.config, self.observation
        margin = 2 * obs['effective_sliding_friction']['jaw_object'] * c.jaw_force_limit / (c.object_mass * c.gravity)
        return margin < LOW_HEADROOM_MARGIN

    def joint_targets_for_pose(self, goal):
        obs = self.observation
        tcp = np.asarray(obs['end_effector_position'])
        obj = np.asarray(obs['object_position'])
        raw = np.asarray(goal) - tcp
        held = self.policy._held_offset
        transfer = self.policy._action_phase == 'transfer' and self.eligible() and held is not None
        if transfer and self.entry_tcp_z is None:
            self.entry_tcp_z = float(tcp[2])
        elif not transfer:
            self.entry_tcp_z = None
        both = obs['grasp_contacts']['left'] > 0 and obs['grasp_contacts']['right'] > 0
        clearance = float(obj[2] - self.initial_object_position[2])
        active = transfer and both and clearance >= MIN_OBJECT_CLEARANCE_M
        delta = raw.copy()
        error, floor = None, None
        if active:
            error = float((obj - tcp - held)[2])
            floor = max(self.entry_tcp_z - MAX_DOWNWARD_EXCURSION_M,
                        self.initial_object_position[2] + MIN_OBJECT_CLEARANCE_M - held[2])
            delta[2] = max(min(0., VERTICAL_ERROR_GAIN * error), -VERTICAL_INCREMENT_CAP_M)
            delta[2] = float(np.clip(max(delta[2], floor - tcp[2]),
                                     -VERTICAL_INCREMENT_CAP_M, VERTICAL_INCREMENT_CAP_M))
            length = np.linalg.norm(delta)
            if length > TOTAL_INCREMENT_CAP_M:
                delta *= TOTAL_INCREMENT_CAP_M / length
            goal = tcp + delta
            self.intervention_calls += 1
            if self.first_intervention_time is None:
                self.first_intervention_time = float(self._elapsed)
        self.last_intervention = {
            'active': bool(active), 'phase': self.policy._action_phase,
            'raw_increment_m': raw.tolist(), 'commanded_increment_m': delta.tolist(),
            'vertical_offset_error_m': error, 'minimum_tcp_z_m': float(floor) if floor is not None else None,
            'object_clearance_m': clearance, 'entry_tcp_z_m': self.entry_tcp_z,
        }
        return (self.head.predict(obs, goal) if self.head is not None
                else self.environment.joint_targets_for_pose(goal))


class CompliantTransferPolicy(ArmPickPlacePolicy):
    """Use the compliant transit height in phase completion, not in task scoring."""
    def _compliant_transfer_complete(self, observation):
        info = getattr(self.env, 'last_intervention', None) or {}
        if not info.get('active'):
            return False
        planar_error = np.linalg.norm(np.asarray(observation['end_effector_position'][:2]) -
                                      np.asarray(observation['target_position'][:2]))
        relative_error = (np.linalg.norm(np.asarray(observation['object_position']) -
                                         np.asarray(observation['end_effector_position']) - self._held_offset)
                          if self._held_offset is not None else float('inf'))
        return bool(relative_error <= .012
                    and planar_error < TRANSFER_PLANAR_TOLERANCE_M
                    and observation['object_position'][2] - self.env.initial_object_position[2] >= MIN_OBJECT_CLEARANCE_M
                    and observation['grasp_contacts']['left'] > 0
                    and observation['grasp_contacts']['right'] > 0)

    def __call__(self, observation):
        action = super().__call__(observation)
        if self._action_phase == 'transfer' and self.stage == 'transfer' and self._compliant_transfer_complete(observation):
            self._next('place')
            # Match inherited bookkeeping: _next during a command is followed
            # by this command's one tick; no larger phase timeout is granted.
            self._stage_ticks = 1
        return action
