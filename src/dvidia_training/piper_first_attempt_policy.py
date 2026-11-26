"""Frozen camera-initialized PiPER policy for an upright, 40-mm rigid cube.

This authored controller uses the initial RGB-D estimate, the known level support
and destination, and encoders. Aperture is a proxy for retention. A jammed finger
can mimic a held object. Independent native physics scores the placement.
"""
import math

import numpy as np


_CLEARANCE = np.array([0., 0., .10])
_OPEN_JAW = .035
_HALF_DIAGONAL = .020 * math.sqrt(2.)


def _corner_orientation(pitch):
    # Fixed twist about the tool axis presents opposite cube corners to the
    # fingers. Their wider separation increases preload in the stock jaw servo.
    c, s = math.cos(math.pi / 8.), math.sin(math.pi / 8.)
    w, y = math.cos(pitch / 2.), math.sin(pitch / 2.)
    return np.array([w*c, y*s, y*c, w*s])


def _blend(start, finish, alpha):
    q = (1. - alpha) * start + alpha * finish
    return q / np.linalg.norm(q)


# Pi is represented exactly as the reference vertical quaternion (0,0,1,0).
_VERTICAL = np.array([0., math.sin(math.pi / 8.), math.cos(math.pi / 8.), 0.])
_CARRY = _corner_orientation(math.radians(160.))
_seating = .85 * _VERTICAL + .15 * _CARRY
_SEATING = _seating / np.linalg.norm(_seating)


def _rotation(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _aligned(measured, desired):
    return abs(np.dot(measured, desired)) > math.cos(.12 / 2.)


def _cube_aperture(fingers):
    left, right = fingers
    return abs(left - _HALF_DIAGONAL) <= .006 and abs(-right - _HALF_DIAGONAL) <= .006


class CornerPlacementPolicy:
    """One initial visual anchor, a test lift, a carry and a supported release."""

    def __init__(self, kinematics, *, check_grip, grasp_pitch_deg=180.):
        if not 170. <= grasp_pitch_deg <= 180.:
            raise ValueError('Grasp pitch must be within ten degrees of vertical')
        self.grasp_orientation = (_VERTICAL if grasp_pitch_deg == 180. else
                                  _corner_orientation(math.radians(grasp_pitch_deg)))
        self.kinematics = kinematics
        self.check_grip = check_grip
        self.phase = self.issued_phase = 'approach'
        self.phase_start_s = 0.
        self.grip_ticks = self.lost_ticks = 0
        self.anchor = self.destination = self.held_offset = None
        self.approach_joints = self.travel_origin = None
        self.refusal_reason = None

    def _advance(self, phase, now):
        self.phase, self.phase_start_s = phase, now
        self.grip_ticks = 0
        self.travel_origin = None

    def _approach_action(self, q, target, orientation, jaw):
        # Static IK for a pregrasp 34 mm above the initially observed cube.
        # Bound each issued target to the stock encoder-relative 0.06-rad step.
        if self.approach_joints is None:
            planned = np.asarray(q)
            for _ in range(120):
                action = self.kinematics.pose_action(planned, target, orientation, jaw)
                updated = np.asarray(action['joint_targets_rad'])
                if np.max(np.abs(updated - planned)) < .00005:
                    break
                planned = updated
            self.approach_joints = planned
        delta = self.approach_joints - q
        maximum = np.max(np.abs(delta))
        if maximum > .03:
            command = np.asarray(q) + min(1., .06 / maximum) * delta
            return {'joint_targets_rad': command.tolist(), 'gripper_position_m': jaw}
        return self.kinematics.pose_action(q, target, orientation, jaw)

    def _travel(self, tcp, destination, elapsed, duration):
        if self.travel_origin is None:
            self.travel_origin = tcp.copy()
        t = np.clip(elapsed / duration, 0., 1.)
        # Quintic time scaling gives zero endpoint velocity and acceleration.
        alpha = t**3 * (10. + t * (-15. + 6. * t))
        return self.travel_origin + (destination - self.travel_origin) * alpha, alpha

    def __call__(self, observation):
        now = observation['t_seconds']
        q = observation['joint_position_rad']
        tcp = np.asarray(observation['tcp_position_world_m'])
        rotation = np.asarray(observation['tcp_quaternion_wxyz'])
        fingers = observation['gripper_joint_position_m'] if self.check_grip else None
        if self.anchor is None:
            self.anchor = np.array(observation['initial_object_position_world_m'], dtype=float)
            self.destination = np.array(observation['goal_position_world_m'], dtype=float)
            if any(v.shape != (3,) or not np.all(np.isfinite(v))
                   for v in (self.anchor, self.destination)):
                raise ValueError('A finite initial camera position and a known destination are required')
            self.anchor[2] = self.destination[2]  # Known shared, level support surface.

        phase = self.issued_phase = self.phase
        elapsed = now - self.phase_start_s
        grasp = self.anchor + [0., 0., .002]
        orientation, jaw = self.grasp_orientation, _OPEN_JAW
        if phase == 'approach':
            target = self.anchor + [0., 0., .034]
            if np.linalg.norm(tcp - target) < .008 and _aligned(rotation, orientation):
                self._advance('descend', now)
            return self._approach_action(q, target, orientation, jaw)
        if phase == 'descend':
            target = grasp
            if np.linalg.norm(tcp - target) < .004 and _aligned(rotation, orientation):
                self._advance('close', now)
        elif phase == 'close':
            target, jaw = grasp, 0.
            if self.check_grip:
                self.grip_ticks = self.grip_ticks + 1 if _cube_aperture(fingers) else 0
            if elapsed >= .72 and (not self.check_grip or self.grip_ticks >= 4):
                # The initial estimate may have become inaccurate during approach.
                self.held_offset = _rotation(rotation).T @ (self.anchor - tcp)
                self._advance('probe', now)
            elif elapsed > 1.5:
                self.refusal_reason = 'aperture_inconsistent_with_cube'
                self._advance('refused', now)
        elif phase == 'probe':
            target, jaw = grasp + [0., 0., min(.008, .025 * elapsed)], 0.
            if elapsed >= .6 and tcp[2] > grasp[2] + .005:
                if self.check_grip and not _cube_aperture(fingers):
                    self.refusal_reason = 'aperture_lost_during_test_lift'
                    self._advance('refused', now)
                else:
                    self._advance('lift', now)
        elif phase == 'lift':
            target, alpha = self._travel(tcp, grasp + _CLEARANCE, elapsed, 1.20)
            orientation, jaw = _blend(self.grasp_orientation, _CARRY, alpha), 0.
            if elapsed >= 1.20 and np.linalg.norm(tcp - target) < .008:
                self._advance('transfer', now)
        elif phase == 'transfer':
            orientation, jaw = _CARRY, 0.
            destination = self.destination - _rotation(orientation) @ self.held_offset + _CLEARANCE
            target, _ = self._travel(tcp, destination, elapsed, 1.20)
            if elapsed >= 1.20 and np.linalg.norm(tcp - target) < .008:
                self._advance('place', now)
        elif phase == 'place':
            destination = self.destination - _rotation(_SEATING) @ self.held_offset
            target, alpha = self._travel(tcp, destination, elapsed, 1.12)
            orientation, jaw = _blend(_CARRY, _SEATING, alpha), 0.
            error = tcp - target
            if elapsed >= 1.12 and np.linalg.norm(error[:2]) < .010 and abs(error[2]) < .0025:
                self._advance('seat', now)
        elif phase in ('seat', 'release'):
            orientation = _SEATING
            target = self.destination - _rotation(orientation) @ self.held_offset
            if phase == 'seat':
                jaw = 0.
                if elapsed >= .36:
                    self._advance('release', now)
            elif elapsed >= .72 and (not self.check_grip or min(fingers[0], -fingers[1]) > .028):
                self._advance('retreat', now)
        elif phase in ('retreat', 'verify'):
            target, orientation = self.destination + [0., 0., .13], _CARRY
            if phase == 'retreat' and np.linalg.norm(tcp - target) < .008:
                self._advance('verify', now)
        elif phase == 'refused':
            target = grasp  # Refused at the initial grasp or during the 8-mm test.
        elif phase == 'lost_grip':
            return {'joint_targets_rad': list(q), 'gripper_position_m': 0.}
        else:
            raise RuntimeError(f'Unknown placement phase {phase}')

        if self.check_grip and phase in ('lift', 'transfer', 'place'):
            self.lost_ticks = 0 if _cube_aperture(fingers) else self.lost_ticks + 1
            if self.lost_ticks >= 4:
                self.refusal_reason = 'aperture_lost_in_motion'
                self._advance('lost_grip', now)
                return {'joint_targets_rad': list(q), 'gripper_position_m': 0.}
        return self.kinematics.pose_action(q, target, orientation, jaw)


def make_policy(controller, kinematics):
    if controller not in ('corner-v8', 'corner-v8-timed'):
        raise ValueError(f'Unknown PiPER first-attempt policy {controller}')
    return CornerPlacementPolicy(kinematics, check_grip=controller == 'corner-v8')
