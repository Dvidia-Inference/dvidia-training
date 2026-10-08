"""An explicitly authored, contact-gated placement procedure; not learned video."""
from __future__ import annotations
import numpy as np

PROCEDURE = ('approach', 'descend', 'close', 'lift', 'transfer', 'place', 'release', 'retreat', 'verify')
POLICY_REVISION = 'dimension-aware-contact-rates-recovery-round2'
PHASE_TIMEOUTS = {'approach': 8., 'descend': 4., 'close': 3., 'lift': 4.,
                  'transfer': 8., 'place': 4., 'release': 2., 'retreat': 4.,
                  'verify': 2., 'recover_wait': 2., 'recover_up': 3.}


class ArmPickPlacePolicy:
    """One supported-box retry; unchanged native placement success conditions."""

    def __init__(self, environment):
        self.env = environment
        self.stage = 'approach'
        self._action_phase = 'observe'
        self._stage_ticks = 0
        self._stable_ticks = 0
        self._grasp_position = None
        self._lift_position = None
        self._last_goal = None
        self._grip_gap = None
        self._grip_normal_target = 0.
        self._grip_normal_demand = 0.
        self._measured_pad_forces = {'left': 0., 'right': 0.}
        self._contact_loss_ticks = 0
        self._held_offset = None
        self._recovery_count = 0
        self._recovery_reason = None
        self._recover_goal = None
        self._failed_hold_joints = None
        self._saturation_ticks = 0
        self._saturated_joints = []
        self._loaded_tracking_step = .006
        self.failure_reason = None
        self._failure_phase = None
        self._failure_time = None

    def _next(self, stage):
        self.stage = stage
        self._stage_ticks = 0
        self._stable_ticks = 0
        self._contact_loss_ticks = 0

    def diagnostics(self):
        return {'revision': POLICY_REVISION, 'phase': self.stage,
                'action_phase': self._action_phase,
                'phase_elapsed_seconds': self._stage_ticks*self.env.config.control_dt,
                'failure_reason': self.failure_reason, 'failure_phase': self._failure_phase,
                'failure_simulation_time': self._failure_time,
                'grip_normal_demand_n': self._grip_normal_demand,
                'grip_normal_target_n': self._grip_normal_target,
                'grip_force_limited': self._grip_normal_demand > self.env.config.jaw_force_limit,
                'pad_normal_forces_n': dict(self._measured_pad_forces),
                'commanded_grip_gap_m': self._grip_gap,
                'loaded_translation_tracking_step_m': self._loaded_tracking_step,
                'recovery_count': self._recovery_count, 'recovery_reason': self._recovery_reason,
                'saturated_joint_indices': list(self._saturated_joints),
                'actuator_saturation_control_samples': self._saturation_ticks,
                'last_cartesian_goal_m': self._last_goal.tolist() if self._last_goal is not None else None,
                'observations': 'privileged simulator state', 'policy_origin': 'authored controller',
                'recovery_scope': 'at most one resettled upright, axis-aligned box on the same table',
                'command_rate_scope': 'native-tick target ramp before gravity feedforward; actual velocity is measured separately'}

    def _grip_command(self, observation):
        c = self.env.config
        width = float(observation['object_size'][0])
        mu = float(observation['effective_sliding_friction']['jaw_object'])
        self._grip_normal_demand = max(.8, 1.5*c.object_mass*c.gravity/mu)
        self._grip_normal_target = min(self._grip_normal_demand, c.jaw_force_limit)
        # A force-limited grasp retains preload for the load demand, allowing
        # the unchanged native cap to work. Round1's 90%-cap gap regressed DEV-15.
        nominal = width-2*self._grip_normal_demand/250.
        if self._grip_gap is None:
            self._grip_gap = float(np.clip(nominal, 0., .08))
        if observation['gripper_width'] <= width+.002:
            error = self._grip_normal_target-min(self._measured_pad_forces.values())
            correction = float(np.clip(.001*error, -.0002, .0002))
            self._grip_gap = float(np.clip(self._grip_gap-correction,
                                          max(0., nominal-.008), min(.08, nominal+.002)))
        return self._grip_gap

    def _fail(self, reason, observation):
        if self.failure_reason is None:
            self.failure_reason = reason
            self._failure_phase = self.stage
            self._failure_time = float(self.env._elapsed)
            self._failed_hold_joints = list(observation['joint_position'])
            self._next('failed')
            self._action_phase = 'failed'

    def _failure_action(self, observation):
        # Hold the physical arm, open only if the object is already supported.
        # No early completion/termination or object-state override is involved.
        gap = .08 if observation['table_contacts'] > 0 else float(observation['gripper_width'])
        return {'joint_targets': list(self._failed_hold_joints), 'gripper_width': float(np.clip(gap, 0., .08))}

    def _recover(self, reason, observation):
        if self._recovery_count >= 1:
            self._fail('retention_lost_after_recovery', observation)
            return
        self._recovery_count += 1
        self._recovery_reason = reason
        self._recover_goal = np.asarray(observation['end_effector_position']).copy()
        self._next('recover_wait')

    @staticmethod
    def _axis_aligned(quaternion):
        w, x, y, z = quaternion
        # Axis alignment deliberately bounds recovery; this is not an
        # arbitrary-orientation skill.
        diagonal = np.array([1-2*(y*y+z*z), 1-2*(x*x+z*z), 1-2*(x*x+y*y)])
        return bool(np.all(np.abs(diagonal) >= .98))

    def __call__(self, observation):
        c = self.env.config
        object_position = np.asarray(observation['object_position'])
        target = np.asarray(observation['target_position'])
        tcp = np.asarray(observation['end_effector_position'])
        offset = .026-c.object_half_size[2]
        forces = observation['pad_normal_forces']
        self._measured_pad_forces = {side: float(forces[side]) for side in ('left', 'right')}
        self._saturated_joints = np.flatnonzero(np.abs(observation['joint_actuator_torque']) >= .98*c.joint_torque_limit).tolist()
        self._saturation_ticks += bool(self._saturated_joints)
        mu = float(observation['effective_sliding_friction']['jaw_object'])
        if self.failure_reason is None and 2*mu*c.jaw_force_limit < c.object_mass*c.gravity:
            self._fail('insufficient_jaw_force_for_static_load', observation)
        if self.stage == 'failed':
            return self._failure_action(observation)
        if self._stage_ticks*c.control_dt >= PHASE_TIMEOUTS[self.stage]:
            if self.stage in ('close', 'lift', 'place') and observation['table_contacts'] > 0 and self._recovery_count < 1:
                self._recover('phase_timeout:'+self.stage, observation)
            else:
                self._fail('phase_timeout:'+self.stage, observation)
                return self._failure_action(observation)

        holding = self.stage in ('lift', 'transfer')
        both = observation['grasp_contacts']['left'] > 0 and observation['grasp_contacts']['right'] > 0
        if holding:
            slipping = self._held_offset is not None and np.linalg.norm((object_position-tcp)-self._held_offset) > .012
            self._contact_loss_ticks = self._contact_loss_ticks+1 if (not both or slipping) else 0
            if self._contact_loss_ticks >= 5:
                self._recover('lost_bilateral_contact_or_relative_pose', observation)
                if self.stage == 'failed':
                    return self._failure_action(observation)

        width = .08
        action_phase = self.stage
        if self.stage == 'recover_wait':
            goal = self._recover_goal
            settled = observation['table_contacts'] > 0 and np.linalg.norm(observation['object_velocity']) < .035 and np.linalg.norm(observation['object_angular_velocity']) < .3
            self._stable_ticks = self._stable_ticks+1 if settled else 0
            if self._stable_ticks >= 5:
                try:
                    c._placement(object_position)
                except ValueError:
                    self._fail('recovery_outside_authored_workspace', observation)
                if self.failure_reason is None and not self._axis_aligned(observation['object_quaternion']):
                    self._fail('unsupported_recovery_orientation', observation)
                if self.stage == 'failed':
                    return self._failure_action(observation)
                self._grip_gap = None
                self._held_offset = None
                self._recover_goal = object_position+np.array([0., 0., .15])
                self._next('recover_up')
        elif self.stage == 'recover_up':
            goal = self._recover_goal
            if np.linalg.norm(tcp-goal) < .012:
                self._next('approach')
        elif self.stage == 'approach':
            goal = object_position+np.array([0., 0., .15])
            if np.linalg.norm(tcp-goal) < .012:
                self._grasp_position = object_position+np.array([0., 0., offset])
                self._next('descend')
        elif self.stage == 'descend':
            goal = self._grasp_position
            self._stable_ticks = self._stable_ticks+1 if np.linalg.norm(tcp-goal) < .0045 and np.linalg.norm(observation['end_effector_velocity']) < .05 else 0
            if self._stable_ticks >= 5:
                self._next('close')
        elif self.stage == 'close':
            goal = self._grasp_position
            width = self._grip_command(observation)
            force_ready = min(self._measured_pad_forces.values()) >= .75*self._grip_normal_target
            self._stable_ticks = self._stable_ticks+1 if both and force_ready else 0
            if self._stable_ticks >= 10:
                self._held_offset = (object_position-tcp).copy()
                self._lift_position = goal+np.array([0., 0., .13])
                self._next('lift')
        elif self.stage == 'lift':
            goal = self._lift_position
            width = self._grip_command(observation)
            if object_position[2] > self.env.initial_object_position[2]+.06 and np.linalg.norm(tcp-goal) < .015:
                self._next('transfer')
        elif self.stage == 'transfer':
            goal = target+np.array([0., 0., offset+.13])
            width = self._grip_command(observation)
            if np.linalg.norm(tcp-goal) < .012:
                self._next('place')
        elif self.stage == 'place':
            goal = target+np.array([0., 0., offset])
            width = self._grip_command(observation)
            if observation['table_contacts'] > 0 and np.linalg.norm(object_position[:2]-target[:2]) < .018 and np.linalg.norm(tcp-goal) < .008:
                self._next('release')
        elif self.stage == 'release':
            goal = target+np.array([0., 0., offset])
            if observation['gripper_width'] > .065 and self._stage_ticks >= 10:
                self._next('retreat')
        else:  # retreat / verify
            goal = target+np.array([0., 0., offset+.13])
            if self.stage == 'retreat' and np.linalg.norm(tcp-goal) < .02:
                self._next('verify')

        self._last_goal = goal.copy()
        self._action_phase = action_phase
        ik_goal = goal
        # With little force headroom, reduce Cartesian tracking demand during
        # loaded motion. This changes the controller, never object dynamics.
        static_force_margin = 2*mu*c.jaw_force_limit/(c.object_mass*c.gravity)
        self._loaded_tracking_step = .005 if static_force_margin < 1.25 else .006
        if self.stage in ('lift', 'transfer', 'place') and self._loaded_tracking_step < .006:
            delta = goal-tcp
            distance = np.linalg.norm(delta)
            if distance > self._loaded_tracking_step:
                ik_goal = tcp+delta*(self._loaded_tracking_step/distance)
        self._stage_ticks += 1
        return {'joint_targets': self.env.joint_targets_for_pose(ik_goal), 'gripper_width': width}
