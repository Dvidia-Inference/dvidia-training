"""Experiment-only smoothing of loaded Cartesian increments; no runtime defaults."""
from __future__ import annotations

import numpy as np

from .grip_diagnostic import GoalProxy

SMOOTHING_ALPHA = .2
LOW_HEADROOM_MARGIN = 1.25
LOADED_PHASES = ('lift', 'transfer')


class SmoothedLoadedGoalProxy(GoalProxy):
    """Smooth changes in tracking demand before the same head/IK mapping.

The original controller already caps raw increments at low force headroom. A
convex filter cannot increase that maximum. It changes commands, not physics,
grip force, retention guards, observed state, phase clocks or success scoring.
"""
    def __init__(self, environment, head=None):
        super().__init__(environment, head)
        self._filtered_delta = np.zeros(3)
        self.last_intervention = None
        self.first_intervention_time = None

    def joint_targets_for_pose(self, goal):
        c = self.config
        obs = self.observation
        mu = float(obs['effective_sliding_friction']['jaw_object'])
        margin = 2 * mu * c.jaw_force_limit / (c.object_mass * c.gravity)
        active = self.policy._action_phase in LOADED_PHASES and margin < LOW_HEADROOM_MARGIN
        tcp = np.asarray(obs['end_effector_position'])
        raw = np.asarray(goal) - tcp
        if active:
            self._filtered_delta = SMOOTHING_ALPHA * raw + (1 - SMOOTHING_ALPHA) * self._filtered_delta
            goal = tcp + self._filtered_delta
            self.intervention_calls += 1
            if self.first_intervention_time is None:
                self.first_intervention_time = float(self._elapsed)
        else:
            self._filtered_delta = np.zeros(3)
        self.last_intervention = {
            'active': active, 'phase': self.policy._action_phase,
            'alpha': SMOOTHING_ALPHA, 'static_force_margin': margin,
            'raw_increment_m': raw.tolist(),
            'filtered_increment_m': self._filtered_delta.tolist() if active else raw.tolist(),
        }
        return (self.head.predict(obs, goal) if self.head is not None
                else self.environment.joint_targets_for_pose(goal))
