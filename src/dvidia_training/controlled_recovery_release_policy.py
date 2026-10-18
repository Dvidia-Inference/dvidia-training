"""Experimental v2: wait for measured jaw clearance before withdrawing the hand."""
from __future__ import annotations

from .controlled_recovery_policy import ControlledRecoveryPolicy, LOWER_TIMEOUT_SECONDS

CLEARANCE_GAP_M = .065
CLEARANCE_STABLE_SAMPLES = 5


class SupportedReleaseRecoveryPolicy(ControlledRecoveryPolicy):
    """Hold the same joint pose during opening, within the original recovery budget."""
    def __init__(self, environment):
        super().__init__(environment)
        self._clearance_active = False
        self._clearance_started = None
        self._clearance_ticks = 0
        self._clearance_hold_targets = None
        self._actual_release = None

    def _wait_clearance(self, observation):
        self.stage = self._action_phase = 'recover_wait_clearance'
        elapsed = float(self.env._elapsed - self._lower_started)
        clear = (observation['gripper_width'] > CLEARANCE_GAP_M
                 and observation['grasp_contacts']['left'] == 0
                 and observation['grasp_contacts']['right'] == 0
                 and self._supported(observation))
        self._clearance_ticks = self._clearance_ticks + 1 if clear else 0
        if self._clearance_ticks >= CLEARANCE_STABLE_SAMPLES and elapsed <= LOWER_TIMEOUT_SECONDS + 1e-9:
            self._actual_release = {
                'time': float(self.env._elapsed), 'table_contacts': observation['table_contacts'],
                'object_position': list(observation['object_position']),
                'object_velocity': list(observation['object_velocity']),
                'object_angular_velocity': list(observation['object_angular_velocity']),
                'gripper_width': float(observation['gripper_width']),
                'contacts': dict(observation['grasp_contacts']),
                'tcp_position': list(observation['end_effector_position']),
                'elapsed_recovery_seconds': elapsed,
                'consecutive_control_samples': self._clearance_ticks,
            }
            self._clearance_active = False
            self._next('recover_up')
            self._action_phase = 'recover_clearance_confirmed'
        elif elapsed + 1e-9 >= LOWER_TIMEOUT_SECONDS:
            self._clearance_active = False
            return self._abort(observation, 'controlled_recovery_clearance_timeout')
        # Even on the confirmation tick, hold position. Only the following
        # observed control tick can enter the inherited recovery-up phase.
        return {'joint_targets': list(self._clearance_hold_targets), 'gripper_width': .08}

    def __call__(self, observation):
        if self._clearance_active:
            return self._wait_clearance(observation)
        action = super().__call__(observation)
        if self._action_phase == 'recover_supported_open' and self._clearance_started is None:
            self._clearance_started = float(self.env._elapsed)
            self._clearance_hold_targets = list(observation['joint_position'])
            self._clearance_active = True
            self.stage = 'recover_wait_clearance'
        return action

    def diagnostics(self):
        return {**super().diagnostics(), 'experiment_policy': 'controlled-support-recovery-v2',
                'clearance_started_seconds': self._clearance_started,
                'clearance_stable_samples': self._clearance_ticks,
                'actual_release_observation': self._actual_release,
                'recovery_budget_seconds': LOWER_TIMEOUT_SECONDS}
