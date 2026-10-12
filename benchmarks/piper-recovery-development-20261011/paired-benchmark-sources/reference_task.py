"""Authored contact-based rigid-block task on the pinned native six-axis reference.

The policy is a development baseline, not a learned skill. Observations are
privileged simulator state. Only reset initializes the freely moving block;
grasping, transport and release thereafter require native contact dynamics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from . import reference_sim as sim


PROFILE = 'ur5e-2f85-rigid-block-task-v0.1'
SCHEMA = 'dvidia.reference-task-episode/v1'
SCOPE = ('Authored native UR5e/Robotiq rigid-block manipulation, privileged state and synthetic commands. '
         'No shoe model, learned task policy, camera sensing, calibrated contact or physical qualification.')
CARRY_CONTACT_LOSS_TOLERANCE_S = .04
GRIPPER_JOINTS = tuple('gripper/' + side + '_' + link + '_joint'
                       for side in ('right', 'left')
                       for link in ('driver', 'coupler', 'spring_link', 'follower'))


def _number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be finite numeric data')
    try:
        value = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be finite numeric data') from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must lie in [{minimum}, {maximum}]')
    return value


def _vector(value, length, name, minimum=-10., maximum=10.):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{name} requires {length} values')
    return tuple(_number(item, name, minimum, maximum) for item in value)


@dataclass(frozen=True)
class ReferenceTaskConfig:
    timestep: float = .001
    control_dt: float = .02
    timeout_s: float = 20.
    settle_s: float = .5
    object_position_m: tuple = (-.134, .49, .302)
    target_position_m: tuple = (.08, .49, .30)
    object_half_size_m: tuple = (.02, .02, .02)
    mass_kg: float = .04
    friction: float = .8
    table_top_m: float = .28
    table_center_xy_m: tuple = (0., .55)
    table_half_size_xy_m: tuple = (.32, .26)
    table_thickness_m: float = .04
    goal_position_tolerance_m: float = .025
    goal_dwell_s: float = .25
    min_lift_m: float = .06
    min_lift_dwell_s: float = .10
    retreat_clearance_m: float = .10
    joint_target_rate_rad_s: float = 1.5
    jaw_target_rate_0_255_s: float = 500.
    tool_orientation_wxyz: tuple = (0., math.sqrt(.5), math.sqrt(.5), 0.)

    def __post_init__(self):
        bounds = {'timestep': (.0005, .002), 'control_dt': (.02, .02), 'timeout_s': (.1, 30.),
                  'settle_s': (.1, 2.), 'mass_kg': (.01, .15), 'friction': (.02, 2.),
                  'table_top_m': (.24, .32), 'table_thickness_m': (.02, .08),
                  'goal_position_tolerance_m': (.01, .03), 'goal_dwell_s': (.10, .6),
                  'min_lift_m': (.04, .10), 'min_lift_dwell_s': (.05, .3),
                  'retreat_clearance_m': (.08, .15), 'joint_target_rate_rad_s': (.1, 3.),
                  'jaw_target_rate_0_255_s': (50., 1000.)}
        for name, (low, high) in bounds.items():
            object.__setattr__(self, name, _number(getattr(self, name), name, low, high))
        sim._timestep(self.timestep)
        for numerator, denominator in ((self.control_dt, self.timestep), (self.timeout_s, self.control_dt),
                                        (self.settle_s, self.timestep)):
            ratio = numerator / denominator
            if not math.isclose(ratio, round(ratio), abs_tol=1e-9, rel_tol=0):
                raise ValueError('Task durations must contain an integer number of native/control ticks')
        for name, length, low, high in (('object_position_m', 3, -2, 2), ('target_position_m', 3, -2, 2),
                                      ('object_half_size_m', 3, .015, .025), ('table_center_xy_m', 2, -1, 1),
                                      ('table_half_size_xy_m', 2, .1, .5), ('tool_orientation_wxyz', 4, -1, 1)):
            object.__setattr__(self, name, _vector(getattr(self, name), length, name, low, high))
        if not math.isclose(sum(x*x for x in self.tool_orientation_wxyz), 1., abs_tol=1e-8, rel_tol=0):
            raise ValueError('Tool goal quaternion must be normalized wxyz')
        for name in ('object_position_m', 'target_position_m'):
            point = getattr(self, name)
            if not -.25 <= point[0] <= .25 or not .35 <= point[1] <= .70:
                raise ValueError('Task placement lies outside the declared reference reach envelope')
            for axis in range(2):
                if abs(point[axis]-self.table_center_xy_m[axis]) + self.object_half_size_m[axis] > self.table_half_size_xy_m[axis]-.01:
                    raise ValueError('Task placement must be supported within the table edge')
            expected = self.table_top_m + self.object_half_size_m[2] + (.002 if name == 'object_position_m' else 0.)
            if not math.isclose(point[2], expected, abs_tol=1e-8, rel_tol=0):
                raise ValueError('Start/target height must match the declared support plane and block size')
        if math.dist(self.object_position_m[:2], self.target_position_m[:2]) < .075:
            raise ValueError('Task must require at least 75 mm of block transfer')


def task_contract():
    return {'profile': PROFILE, 'scope': SCOPE, 'physical_robot_ready': False,
            'arm_joint_order': list(sim.ARM_JOINTS), 'gripper_linkage_order': list(GRIPPER_JOINTS),
            'actions': {'joint_targets_rad': 'six absolute joint targets, rad; ramped then gravity feedforward',
                        'gripper_0_255': 'one native tendon servo target; 0 open / 255 close, not measured gap or force'},
            'observation_origin': 'privileged_native_simulator_state', 'native_sensor_definitions': 0,
            'observations': {'joint_position_rad': [6], 'joint_velocity_rad_s': [6], 'linkage_position_rad': [8],
                             'tcp_position_m': [3], 'tcp_quaternion_wxyz': [4], 'object_position_m': [3],
                             'object_quaternion_wxyz': [4], 'grasp_contact_counts': 'left/right native block contacts',
                             'pad_normal_forces_n': 'native mj_contactForce, not deployable tactile sensing'},
            'contacts': 'native collision proxies; object priority 1 overrides pad/table friction floors',
            'object': 'one free rigid box; no weld, grasp attachment or runtime object pose writes',
            'success': 'valid physics, retained bilateral table-free lift and supported destination carry completion, then released/settled destination dwell and tool retreat',
            'carry_retention': {'bilateral_contact_required': 'after sustained lift until supported at destination while held',
                                'contact_force_threshold_n': .01,
                                'consecutive_loss_tolerance_seconds': CARRY_CONTACT_LOSS_TOLERANCE_S,
                                'loss_at_or_above_tolerance': 'permanent failure, independent of teacher phase'},
            'fixed_base_collision_proxy': False}


def build_task_model(assets, config=None):
    config = ReferenceTaskConfig() if config is None else config
    if not isinstance(config, ReferenceTaskConfig):
        raise TypeError('config must be ReferenceTaskConfig')
    _, blobs = sim._asset_payload(assets)
    mj, _ = sim._native_imports()
    arm = sim._spec(blobs, 'universal_robots_ur5e', 'ur5e.xml')
    jaw = sim._spec(blobs, 'robotiq_2f85', '2f85.xml')
    arm.option.cone, arm.option.impratio = jaw.option.cone, jaw.option.impratio
    arm.option.timestep = jaw.option.timestep = config.timestep
    arm.option.integrator = jaw.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    arm.attach(jaw, prefix='gripper/', site='attachment_site')
    arm.worldbody.add_geom(name='task_floor', type=mj.mjtGeom.mjGEOM_PLANE, size=[2, 2, .1], rgba=[.9, .9, .88, 1])
    arm.worldbody.add_geom(name='task_table', type=mj.mjtGeom.mjGEOM_BOX,
                          pos=[*config.table_center_xy_m, config.table_top_m-config.table_thickness_m/2],
                          size=[*config.table_half_size_xy_m, config.table_thickness_m/2], rgba=[.73, .70, .64, 1])
    block = arm.worldbody.add_body(name='task_block', pos=config.object_position_m)
    block.add_freejoint(name='task_block_free')
    block.add_geom(name='task_block_geom', type=mj.mjtGeom.mjGEOM_BOX, size=config.object_half_size_m,
                   mass=config.mass_kg, friction=[config.friction, .005, .0001], priority=1,
                   condim=4, rgba=[.75, .30, .16, 1])
    marker = arm.worldbody.add_body(name='task_target_frame',
                                   pos=[*config.target_position_m[:2], config.table_top_m+.0005])
    marker.add_geom(name='task_target', type=mj.mjtGeom.mjGEOM_BOX,
                          size=[config.object_half_size_m[0]+.015, config.object_half_size_m[1]+.015, .0005],
                          rgba=[.17, .47, .35, .3], contype=0, conaffinity=0)
    model = arm.compile()
    if (model.nq, model.nv, model.nu, model.neq) != (21, 20, 7, 3):
        raise ValueError('Task native mechanical contract changed')
    return model


class ReferenceTaskEnv:
    def __init__(self, assets, config=None):
        self.assets = Path(assets)
        self.config = ReferenceTaskConfig() if config is None else config
        if not isinstance(self.config, ReferenceTaskConfig):
            raise TypeError('config must be ReferenceTaskConfig')
        self.ready = False

    def reset(self, seed=0):
        if type(seed) is not int or not 0 <= seed <= 999999:
            raise ValueError('seed must be an integer in [0, 999999]')
        mj, np = sim._native_imports()
        self.ready = False
        self.seed, self.warnings = seed, []
        with sim.native_warnings(self.warnings):
            self.model = build_task_model(self.assets, self.config)
            self.data = mj.MjData(self.model)
            self.arm_ids = np.asarray([mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, x) for x in sim.ARM_JOINTS])
            self.arm_qpos = self.model.jnt_qposadr[self.arm_ids]
            self.arm_dof = self.model.jnt_dofadr[self.arm_ids]
            self.linkage_ids = np.asarray([mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, x) for x in GRIPPER_JOINTS])
            self.linkage_qpos = self.model.jnt_qposadr[self.linkage_ids]
            self.tcp = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, 'gripper/pinch')
            self.block = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'task_block')
            self.block_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'task_block_geom')
            self.table_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'task_table')
            self.floor_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'task_floor')
            self.pad_geoms = {side: [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, f'gripper/{side}_pad{i}')
                                     for i in (1, 2)] for side in ('left', 'right')}
            self.robot_geoms = set(range(self.model.ngeom)) - {
                self.block_geom, self.table_geom,
                self.floor_geom,
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'task_target')}
            arm_bodies = {mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name) for name in sim.ARM_BODIES}
            self.arm_geoms = {i for i in self.robot_geoms if int(self.model.geom_bodyid[i]) in arm_bodies}
            # The upstream keyframe contains only robot qpos; initialize them
            # explicitly and leave the free block's compiled reset coordinates.
            self.data.qpos[self.arm_qpos] = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.]
            self.data.qpos[self.linkage_qpos] = 0.
            self.command_targets = self.data.qpos[self.arm_qpos].copy()
            self.command_gripper = 0.
            mj.mj_forward(self.model, self.data)
            for _ in range(round(self.config.settle_s/self.config.timestep)):
                self._controls()
                mj.mj_step(self.model, self.data)
                mj.mj_forward(self.model, self.data)
                if not self._finite() or self.warnings or any(x.number for x in self.data.warning):
                    raise RuntimeError('Task settling produced invalid native physics')
        self.initial_object_position = self.data.xpos[self.block].copy()
        self.elapsed, self.steps, self.dwell, self.lift_dwell = 0., 0, 0., 0.
        self.max_lift, self.max_pad_force, self.critical_contact_ticks = 0., 0., 0
        self.lift_seen, self.grasp_seen, self.done = False, False, False
        self.carry_complete, self.retention_lost = False, False
        self.carry_contact_loss, self.max_carry_contact_loss = 0., 0.
        self.carry_contact_loss_events = 0
        self.first_success_time = None
        self.reason = 'running'
        self.ready = True
        return self.observation(), self.info()

    def _finite(self):
        _, np = sim._native_imports()
        return all(np.isfinite(value).all() for value in (self.data.qpos, self.data.qvel, self.data.qacc, self.data.xpos))

    def _controls(self):
        _, np = sim._native_imports()
        # Authored gravity feedforward, applied inside the original bounded
        # affine position servos. Actual force/velocity are never overwritten.
        gains = self.model.actuator_gainprm[:6, 0]
        requested = self.command_targets + self.data.qfrc_bias[self.arm_dof]/gains
        self.data.ctrl[:6] = np.clip(requested, self.model.actuator_ctrlrange[:6, 0], self.model.actuator_ctrlrange[:6, 1])
        self.data.ctrl[6] = self.command_gripper

    def _contacts(self):
        mj, np = sim._native_imports()
        counts, forces, table, critical = {'left': 0, 'right': 0}, {'left': 0., 'right': 0.}, 0, 0
        pads = set(sum(self.pad_geoms.values(), []))
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            force = np.zeros(6)
            mj.mj_contactForce(self.model, self.data, index, force)
            if force[0] <= .01:
                continue
            if self.block_geom in pair:
                if self.table_geom in pair:
                    table += 1
                for side in ('left', 'right'):
                    if pair.intersection(self.pad_geoms[side]):
                        counts[side] += 1
                        forces[side] += float(force[0])
                if pair.intersection(self.robot_geoms-pads):
                    critical += 1
            elif pair.intersection({self.table_geom, self.floor_geom}) and pair.intersection(self.robot_geoms):
                critical += 1
            elif pair.intersection(self.arm_geoms) and pair.issubset(self.robot_geoms):
                critical += 1
        return counts, forces, table, critical

    def observation(self):
        mj, np = sim._native_imports()
        if not self._finite():
            raise RuntimeError('Cannot record a nonfinite native state')
        counts, forces, table, critical = self._contacts()
        tcp_quat, velocity = np.empty(4), np.empty(6)
        mj.mju_mat2Quat(tcp_quat, self.data.site_xmat[self.tcp])
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, self.block, velocity, 0)
        separation = float(np.linalg.norm(self.data.geom_xpos[self.pad_geoms['left'][0]]-
                                           self.data.geom_xpos[self.pad_geoms['right'][0]]))
        return {'t_seconds': self.elapsed, 'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist(),
                'joint_position_rad': self.data.qpos[self.arm_qpos].tolist(),
                'joint_velocity_rad_s': self.data.qvel[self.arm_dof].tolist(),
                'linkage_position_rad': self.data.qpos[self.linkage_qpos].tolist(),
                'tcp_position_m': self.data.site_xpos[self.tcp].tolist(), 'tcp_quaternion_wxyz': tcp_quat.tolist(),
                'object_position_m': self.data.xpos[self.block].tolist(), 'object_quaternion_wxyz': self.data.xquat[self.block].tolist(),
                'object_linear_velocity_m_s': velocity[3:].tolist(), 'object_angular_velocity_rad_s': velocity[:3].tolist(),
                'target_position_m': list(self.config.target_position_m), 'pad_center_separation_m': separation,
                'grasp_contact_counts': counts, 'pad_normal_forces_n': forces, 'table_object_contact_count': table,
                'unexpected_contact_count': critical, 'contact_count': int(self.data.ncon),
                'native_actuator_controls': self.data.ctrl.tolist(), 'actuator_force': self.data.actuator_force.tolist(),
                'issued_joint_targets_rad': None, 'issued_gripper_0_255': None, 'command': None,
                'observation_origin': 'privileged_native_simulator_state'}

    def info(self):
        _, np = sim._native_imports()
        valid = self.reason not in ('invalid_state', 'solver_warning') and not self.warnings
        success = valid and self.carry_complete and not self.retention_lost and self.dwell+1e-12 >= self.config.goal_dwell_s
        return {'valid': valid, 'success': success,
                'reason': ('success' if success else self.reason), 'simulation_time': self.elapsed,
                'steps': self.steps, 'lift_seen': self.lift_seen, 'grasp_seen': self.grasp_seen,
                'max_object_lift_m': self.max_lift, 'dwell_seconds': self.dwell, 'lift_dwell_seconds': self.lift_dwell,
                'max_pad_normal_force_n': self.max_pad_force, 'unexpected_contact_native_ticks': self.critical_contact_ticks,
                'carry_complete': self.carry_complete, 'retention_lost': self.retention_lost,
                'max_carry_contact_loss_seconds': self.max_carry_contact_loss,
                'carry_contact_loss_events': self.carry_contact_loss_events,
                'carry_contact_loss_tolerance_seconds': CARRY_CONTACT_LOSS_TOLERANCE_S,
                'first_success_time_seconds': self.first_success_time,
                'target_distance_m': float(np.linalg.norm(self.data.xpos[self.block]-self.config.target_position_m)) if self._finite() else None,
                'warnings': list(self.warnings), 'physical_robot_ready': False}

    def _outcome_tick(self, *, both, supported, lift, distance, released, settled, clear, critical, pad_force):
        """Score native evidence without consulting the authored policy phase."""
        dt = self.config.timestep
        self.max_pad_force = max(self.max_pad_force, pad_force)
        self.critical_contact_ticks += bool(critical)
        self.grasp_seen |= both
        self.max_lift = max(self.max_lift, lift)
        loaded = both and not supported and lift >= self.config.min_lift_m
        self.lift_dwell = self.lift_dwell+dt if loaded else 0.
        self.lift_seen |= self.lift_dwell+1e-12 >= self.config.min_lift_dwell_s
        if self.lift_seen and not self.carry_complete:
            # Release is allowed only once the still-held block actually rests
            # at its destination; a dropped/tossed block cannot acquire this latch.
            self.carry_complete |= bool(both and supported and distance <= self.config.goal_position_tolerance_m)
            if not self.carry_complete:
                if both:
                    self.carry_contact_loss = 0.
                else:
                    before = self.carry_contact_loss
                    self.carry_contact_loss += dt
                    self.max_carry_contact_loss = max(self.max_carry_contact_loss, self.carry_contact_loss)
                    if before+1e-12 < CARRY_CONTACT_LOSS_TOLERANCE_S <= self.carry_contact_loss+1e-12:
                        self.carry_contact_loss_events += 1
                        self.retention_lost = True
        qualifies = (self.lift_seen and self.grasp_seen and self.carry_complete and not self.retention_lost
                     and released and supported and settled and clear
                     and distance <= self.config.goal_position_tolerance_m and self.critical_contact_ticks == 0)
        self.dwell = self.dwell+dt if qualifies else 0.
        if self.dwell+1e-12 >= self.config.goal_dwell_s and self.first_success_time is None:
            self.first_success_time = self.elapsed

    def pose_action(self, position_m, quaternion_wxyz, gripper_0_255):
        if not self.ready or self.done:
            raise RuntimeError('Reset is required before another task action')
        mj, np = sim._native_imports()
        position = np.asarray(_vector(position_m, 3, 'pose position'), dtype=float)
        quaternion = np.asarray(_vector(quaternion_wxyz, 4, 'pose quaternion', -1, 1), dtype=float)
        if not math.isclose(float(quaternion@quaternion), 1., abs_tol=1e-8, rel_tol=0):
            raise ValueError('Pose quaternion must be normalized wxyz')
        jaw = _number(gripper_0_255, 'gripper command', 0, 255)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp)
        current, rotation = np.empty(4), np.empty(3)
        mj.mju_mat2Quat(current, self.data.site_xmat[self.tcp])
        mj.mju_subQuat(rotation, quaternion, current)
        # mju_subQuat returns angular velocity in the current local frame;
        # site rotational Jacobians are in world coordinates.
        rotation = self.data.site_xmat[self.tcp].reshape(3, 3) @ rotation
        translation = position-self.data.site_xpos[self.tcp]
        for error, maximum in ((translation, .018), (rotation, .04)):
            norm = float(np.linalg.norm(error))
            if norm > maximum:
                error *= maximum/norm
        jacobian = np.vstack((jacp[:, self.arm_dof], .25*jacr[:, self.arm_dof]))
        error = np.concatenate((translation, .25*rotation))
        delta = jacobian.T @ np.linalg.solve(jacobian@jacobian.T+.012**2*np.eye(6), error)
        delta = np.clip(delta, -.06, .06)
        targets = np.clip(self.data.qpos[self.arm_qpos]+delta,
                          self.model.actuator_ctrlrange[:6, 0]+.02, self.model.actuator_ctrlrange[:6, 1]-.02)
        return {'joint_targets_rad': targets.tolist(), 'gripper_0_255': jaw}

    def step(self, action):
        if not self.ready or self.done:
            raise RuntimeError('Reset is required before another task step')
        if type(action) is not dict or set(action) != {'joint_targets_rad', 'gripper_0_255'}:
            raise ValueError('Task action requires six joint targets and one native gripper command')
        mj, np = sim._native_imports()
        targets = np.asarray(_vector(action['joint_targets_rad'], 6, 'joint targets'), dtype=float)
        jaw = _number(action['gripper_0_255'], 'gripper command', 0, 255)
        if np.any(targets < self.model.actuator_ctrlrange[:6, 0]) or np.any(targets > self.model.actuator_ctrlrange[:6, 1]):
            raise ValueError('Task joint target exceeds native actuator limits')
        if not self._finite():
            self.reason, self.done = 'invalid_state', True
            return None, -1., True, False, self.info()
        before_distance = float(np.linalg.norm(self.data.xpos[self.block]-self.config.target_position_m))
        messages = []
        with sim.native_warnings(messages):
            for _ in range(round(self.config.control_dt/self.config.timestep)):
                self.command_targets += np.clip(targets-self.command_targets,
                                               -self.config.joint_target_rate_rad_s*self.config.timestep,
                                               self.config.joint_target_rate_rad_s*self.config.timestep)
                self.command_gripper += float(np.clip(jaw-self.command_gripper,
                                                     -self.config.jaw_target_rate_0_255_s*self.config.timestep,
                                                     self.config.jaw_target_rate_0_255_s*self.config.timestep))
                self._controls()
                mj.mj_step(self.model, self.data)
                mj.mj_forward(self.model, self.data)
                self.elapsed += self.config.timestep
                if not self._finite():
                    self.reason, self.done = 'invalid_state', True
                    break
                if messages or any(x.number for x in self.data.warning):
                    self.reason, self.done = 'solver_warning', True
                    break
                counts, forces, supported, critical = self._contacts()
                both = all(counts[side] > 0 for side in ('left', 'right'))
                lift = float(self.data.xpos[self.block, 2]-self.initial_object_position[2])
                velocity = np.empty(6)
                mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, self.block, velocity, 0)
                distance = float(np.linalg.norm(self.data.xpos[self.block]-self.config.target_position_m))
                drivers = self.data.qpos[self.linkage_qpos[[0, 4]]]
                released = not any(counts.values()) and bool(np.max(drivers) < .1)
                settled = np.linalg.norm(velocity[3:]) < .035 and np.linalg.norm(velocity[:3]) < .3
                clear = self.data.site_xpos[self.tcp, 2] >= self.data.xpos[self.block, 2]+self.config.retreat_clearance_m
                self._outcome_tick(both=both, supported=supported, lift=lift, distance=distance,
                                   released=released, settled=settled, clear=clear,
                                   critical=critical, pad_force=max(forces.values()))
        self.warnings.extend(messages)
        self.steps += 1
        if not self.done and self.steps >= round(self.config.timeout_s/self.config.control_dt):
            self.reason, self.done = 'timeout', True
        observation = self.observation() if self._finite() else None
        info = self.info()
        reward = before_distance-info['target_distance_m'] if info['target_distance_m'] is not None else -1.
        return observation, reward, self.done and self.reason != 'timeout', self.done and self.reason == 'timeout', info


class ReferenceTaskPolicy:
    """Authored approach/grasp/place baseline, with no learned model or fallback."""
    def __init__(self, environment):
        self.env, self.phase, self.ticks = environment, 'approach', 0
        self.grasp_position = None
        self.held_offset = None

    def __call__(self, observation):
        _, np = sim._native_imports()
        c = self.env.config
        obj, tcp, target = (np.asarray(observation[name]) for name in
                            ('object_position_m', 'tcp_position_m', 'target_position_m'))
        jaw = 0.
        both = all(observation['grasp_contact_counts'][side] > 0 for side in ('left', 'right'))
        if self.phase == 'approach':
            goal = obj+np.asarray([0., 0., .13])
            if np.linalg.norm(tcp-goal) < .008:
                self.grasp_position = obj+np.asarray([0., 0., .005])
                self.phase, self.ticks = 'descend', 0
        elif self.phase == 'descend':
            goal = self.grasp_position
            if np.linalg.norm(tcp-goal) < .004:
                self.phase, self.ticks = 'close', 0
        elif self.phase == 'close':
            goal, jaw = self.grasp_position, 180.
            if both and min(observation['pad_normal_forces_n'].values()) > .1 and self.ticks >= 25:
                self.held_offset = obj-tcp
                self.phase, self.ticks = 'lift', 0
        elif self.phase == 'lift':
            goal, jaw = self.grasp_position+np.asarray([0., 0., .13]), 180.
            if self.env.lift_seen and np.linalg.norm(tcp-goal) < .008:
                self.phase, self.ticks = 'transfer', 0
        elif self.phase == 'transfer':
            goal, jaw = target-self.held_offset+np.asarray([0., 0., .13]), 180.
            if np.linalg.norm(tcp-goal) < .008:
                self.phase, self.ticks = 'place', 0
        elif self.phase == 'place':
            goal, jaw = target-self.held_offset, 180.
            if observation['table_object_contact_count'] > 0 and np.linalg.norm(obj-target) < .02 and np.linalg.norm(tcp-goal) < .006:
                self.phase, self.ticks = 'release', 0
        elif self.phase == 'release':
            goal = target-self.held_offset
            if not any(observation['grasp_contact_counts'].values()) and self.ticks >= 35:
                self.phase, self.ticks = 'retreat', 0
        else:
            goal = target+np.asarray([0., 0., .135])
            if np.linalg.norm(tcp-goal) < .008:
                self.phase = 'verify'
        self.ticks += 1
        return self.env.pose_action(goal.tolist(), c.tool_orientation_wxyz, jaw)


def run_episode(assets, config=None, *, seed=0, control='placement', arm_tape=None):
    """Retain every issued decision and native state for one attempted episode."""
    config = ReferenceTaskConfig() if config is None else config
    if not isinstance(config, ReferenceTaskConfig):
        raise TypeError('config must be ReferenceTaskConfig')
    if control not in ('placement', 'hold', 'replay_open_jaw'):
        raise ValueError('Unknown reference task control')
    started = time.perf_counter()
    attempt = {'schema': SCHEMA, 'profile': PROFILE, 'scope': SCOPE, 'physical_robot_ready': False,
               'seed': seed, 'control': control, 'config': asdict(config), 'timestep_seconds': config.timestep,
               'simulated_seconds': 0., 'valid': False, 'success': False, 'reason': 'not_started',
               'error': None, 'warnings': [], 'trajectory': [], 'actions': [], 'arm_tape': [],
               'metrics': {}, 'timing': {}, 'contract': task_contract()}
    source_root = Path(__file__).parent
    attempt['implementation_sha256'] = {name: sha256((source_root / name).read_bytes()).hexdigest()
                                        for name in ('reference_task.py', 'reference_sim.py', 'reference_assets.lock.json')}
    environment, phase, latencies = None, 'validation', []
    try:
        if control == 'replay_open_jaw':
            if type(arm_tape) is not list or len(arm_tape) != round(config.timeout_s/config.control_dt):
                raise ValueError('Open-jaw control requires exactly one arm target for every configured control tick')
            arm_tape = [list(_vector(row, 6, 'replay arm target')) for row in arm_tape]
        phase = 'setup'
        attempt['asset_provenance'] = sim.verify_assets(assets)
        environment = ReferenceTaskEnv(assets, config)
        observation, info = environment.reset(seed=seed)
        attempt['compiled_model_sha256'] = sim.model_digest(environment.model)
        attempt['initial_object_position_m'] = environment.initial_object_position.tolist()
        attempt['target_position_m'] = list(config.target_position_m)
        attempt['trajectory'].append(observation)
        attempt['timing']['setup_seconds'] = time.perf_counter()-started
        policy = ReferenceTaskPolicy(environment)
        hold = list(observation['joint_position_rad'])
        phase = 'rollout'
        for index in range(round(config.timeout_s/config.control_dt)):
            began = time.perf_counter()
            if control == 'placement':
                action = policy(observation)
                action_phase = policy.phase
            else:
                action = {'joint_targets_rad': hold if control == 'hold' else arm_tape[index],
                          'gripper_0_255': 0.}
                action_phase = control
            record = {'t_seconds': observation['t_seconds'], **action, 'phase': action_phase}
            attempt['actions'].append(record)
            attempt['arm_tape'].append(list(action['joint_targets_rad']))
            observation['issued_joint_targets_rad'] = list(action['joint_targets_rad'])
            observation['issued_gripper_0_255'] = action['gripper_0_255']
            observation['command'] = record
            observation, _, terminated, truncated, info = environment.step(action)
            latencies.append(time.perf_counter()-began)
            if observation is not None:
                attempt['trajectory'].append(observation)
            if terminated or truncated:
                break
        attempt.update(valid=info['valid'], success=info['success'], reason=info['reason'],
                       simulated_seconds=info['simulation_time'], warnings=info['warnings'], metrics=info)
    except Exception as exc:
        attempt['error'] = {'phase': phase, 'type': type(exc).__name__, 'message': str(exc)[:500]}
        attempt['reason'] = 'exception'
        if environment is not None:
            attempt['warnings'] = list(getattr(environment, 'warnings', []))
            attempt['simulated_seconds'] = getattr(environment, 'elapsed', 0.)
    elapsed = time.perf_counter()-started
    attempt['timing'].update(episode_wall_seconds=elapsed,
                            control_step_seconds=sim._summary(latencies) if latencies else None,
                            missed_control_boundaries=sum(latency > config.control_dt for latency in latencies),
                            control_boundaries=len(latencies),
                            simulated_seconds_per_wall_second=attempt['simulated_seconds']/elapsed,
                            scope='Wall computation, not real-time sensor-to-actuator latency. Includes setup, policy/IK, native physics, contacts, observations and trace construction; excludes process startup, output I/O and rendering')
    return attempt


def run_matched(assets, config=None, *, seed=0):
    placement = run_episode(assets, config, seed=seed)
    controls = [run_episode(assets, config, seed=seed, control='replay_open_jaw', arm_tape=placement['arm_tape'])]
    controls.append(run_episode(assets, config, seed=seed, control='hold'))
    return [placement, *controls]
