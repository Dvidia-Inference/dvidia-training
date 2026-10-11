"""PiPER free-block contact pilot; authored policy and privileged simulator state.

The pinned robot's native servos, force limits and gravity compensation remain
unchanged. The block moves under gravity and contact, with no grasp attachment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from pathlib import Path
import platform
import time

from . import reference_sim as sim
from . import robot_profiles
from .reference_task import ReferenceTaskEnv, _number, _vector


PROFILE = 'piper-rigid-block-task-v0.1'
SCHEMA = 'dvidia.piper-task-episode/v1'
SCOPE = ('Authored PiPER rigid-block contact pilot with privileged native state. '
         'Pinned upstream servos/inertias/collision proxies/gravity compensation, synthetic commands and uncalibrated contact. '
         'No learned skill, camera sensing, shoe manipulation or physical robot qualification.')
ARM_JOINTS = tuple('joint'+str(i) for i in range(1, 7))


@dataclass(frozen=True)
class PiperTaskConfig:
    timestep: float = .001
    control_dt: float = .02
    timeout_s: float = 20.
    settle_s: float = .5
    object_position_m: tuple = (.30, -.03, .082)
    target_position_m: tuple = (.35, .08, .08)
    object_half_size_m: tuple = (.02, .02, .02)
    mass_kg: float = .04
    friction: float = .8
    table_top_m: float = .06
    table_center_xy_m: tuple = (.38, 0.)
    table_half_size_xy_m: tuple = (.17, .22)
    table_thickness_m: float = .04
    goal_position_tolerance_m: float = .025
    goal_dwell_s: float = .25
    min_lift_m: float = .06
    min_lift_dwell_s: float = .10
    retreat_clearance_m: float = .10
    joint_target_rate_rad_s: float = 1.5
    jaw_target_rate_m_s: float = .05
    tool_orientation_wxyz: tuple = (0., 0., 1., 0.)
    carry_orientation_wxyz: tuple = (math.cos(5*math.pi/12), 0., math.sin(5*math.pi/12), 0.)

    def __post_init__(self):
        bounds = {'timestep': (.0005, .002), 'control_dt': (.02, .02), 'timeout_s': (.1, 30.),
                  'settle_s': (.1, 2.), 'mass_kg': (.01, .10), 'friction': (.05, 2.),
                  'table_top_m': (.04, .20), 'table_thickness_m': (.02, .08),
                  'goal_position_tolerance_m': (.01, .025), 'goal_dwell_s': (.1, .6),
                  'min_lift_m': (.04, .1), 'min_lift_dwell_s': (.05, .3),
                  'retreat_clearance_m': (.08, .15), 'joint_target_rate_rad_s': (.1, 3.),
                  'jaw_target_rate_m_s': (.005, .10)}
        for name, (low, high) in bounds.items():
            object.__setattr__(self, name, _number(getattr(self, name), name, low, high))
        sim._timestep(self.timestep)
        for numerator, denominator in ((self.control_dt, self.timestep), (self.timeout_s, self.control_dt), (self.settle_s, self.timestep)):
            ratio = numerator/denominator
            if not math.isclose(ratio, round(ratio), abs_tol=1e-9, rel_tol=0):
                raise ValueError('Durations must contain complete native/control ticks')
        for name, length, low, high in (('object_position_m', 3, -1, 1), ('target_position_m', 3, -1, 1),
                                      ('object_half_size_m', 3, .018, .023), ('table_center_xy_m', 2, -1, 1),
                                      ('table_half_size_xy_m', 2, .1, .3), ('tool_orientation_wxyz', 4, -1, 1),
                                      ('carry_orientation_wxyz', 4, -1, 1)):
            object.__setattr__(self, name, _vector(getattr(self, name), length, name, low, high))
        if any(not math.isclose(sum(x*x for x in quaternion), 1., abs_tol=1e-8, rel_tol=0)
               for quaternion in (self.tool_orientation_wxyz, self.carry_orientation_wxyz)):
            raise ValueError('Tool quaternion must be normalized wxyz')
        for name in ('object_position_m', 'target_position_m'):
            point = getattr(self, name)
            if not .25 <= point[0] <= .48 or not -.15 <= point[1] <= .15:
                raise ValueError('Placement outside the bounded pilot workspace')
            for axis in range(2):
                if abs(point[axis]-self.table_center_xy_m[axis])+self.object_half_size_m[axis] > self.table_half_size_xy_m[axis]-.01:
                    raise ValueError('Block placement must be inside the support table')
            expected = self.table_top_m+self.object_half_size_m[2]+(.002 if name == 'object_position_m' else 0.)
            if not math.isclose(point[2], expected, abs_tol=1e-8, rel_tol=0):
                raise ValueError('Block start/target height differs from support plane')
        if math.dist(self.object_position_m[:2], self.target_position_m[:2]) < .075:
            raise ValueError('Pilot must require at least 75 mm transfer')


def _config(value):
    value = PiperTaskConfig() if value is None else value
    if not isinstance(value, PiperTaskConfig):
        raise TypeError('config must be PiperTaskConfig')
    return value


def _asset_root(assets):
    path = Path(assets)
    return path.parent if path.name == 'agilex_piper' else path


def task_contract():
    return {'profile': PROFILE, 'scope': SCOPE, 'physical_robot_ready': False, 'arm_joint_order': list(ARM_JOINTS),
            'actions': {'joint_targets_rad': 'six native position-servo targets in radians',
                        'gripper_position_m': 'one native single-finger slide target in metres [0,.035]; 0 closed/.035 open; not measured full gap'},
            'gripper_joints': ['joint7', 'joint8'], 'independent_gripper_commands': 1,
            'tcp': 'authored site in link6 at local [0,0,.12003] near the distal finger-box centers; not hardware calibrated',
            'object': 'free rigid block with native gravity, no weld/equality/mocap or post-reset pose writes',
            'contact': 'stock native finger collision boxes; object priority1 authors pair friction explicitly',
            'servo_gains': 'unchanged upstream PiPER position servos and force caps; upstream robot gravcomp=1 retained',
            'collision_support_exemption': 'fixed base capsule may contact floor; moving robot/table/floor or block/nonfinger contacts are critical',
            'observations': 'privileged joint/object/TCP/contact state; no deployable sensor pipeline',
            'success': 'valid native physics, sustained bilateral table-free lift and retained carry to supported destination, then actual released/settled destination dwell and retreat',
            'carry_contact_loss_tolerance_seconds': .04, 'contact_force_threshold_n': .01,
            'simulation_asset_license': 'MIT'}


def build_task_model(assets, config=None):
    config = _config(config)
    item, blobs = robot_profiles._payload('piper', _asset_root(assets))
    mj, _ = sim._native_imports()
    spec = sim._spec(blobs, item['folder'], item['model'])
    spec.option.timestep = config.timestep
    spec.body('link6').add_site(name='piper_task_tcp', pos=[0, 0, .12003], size=[.004, 0, 0], rgba=[.1, .5, .3, 1])
    spec.worldbody.add_geom(name='task_floor', type=mj.mjtGeom.mjGEOM_PLANE, size=[2, 2, .1], rgba=[.9, .9, .88, 1])
    spec.worldbody.add_geom(name='task_table', type=mj.mjtGeom.mjGEOM_BOX,
                            pos=[*config.table_center_xy_m, config.table_top_m-config.table_thickness_m/2],
                            size=[*config.table_half_size_xy_m, config.table_thickness_m/2], rgba=[.73, .70, .64, 1])
    block = spec.worldbody.add_body(name='task_block', pos=config.object_position_m, gravcomp=0)
    block.add_freejoint(name='task_block_free')
    block.add_geom(name='task_block_geom', type=mj.mjtGeom.mjGEOM_BOX, size=config.object_half_size_m,
                   mass=config.mass_kg, friction=[config.friction, .005, .0001], priority=1, condim=4, rgba=[.75, .30, .16, 1])
    target = spec.worldbody.add_body(name='task_target_frame', pos=[*config.target_position_m[:2], config.table_top_m+.0005])
    target.add_geom(name='task_target', type=mj.mjtGeom.mjGEOM_BOX, size=[.035, .035, .0005],
                    contype=0, conaffinity=0, rgba=[.17, .47, .35, .3])
    model = spec.compile()
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != (15, 14, 7, 1, 0):
        raise ValueError('PiPER task mechanical contract changed')
    return model


class PiperTaskEnv:
    _outcome_tick = ReferenceTaskEnv._outcome_tick

    def __init__(self, assets, config=None):
        self.assets, self.config, self.ready = _asset_root(assets), _config(config), False

    def _finite(self):
        _, np = sim._native_imports()
        return (ReferenceTaskEnv._finite(self) and not np.any(self.data.qfrc_applied)
                and not np.any(self.data.xfrc_applied))

    def info(self):
        result = ReferenceTaskEnv.info(self)
        if not self._finite() or any(warning.number for warning in self.data.warning):
            result.update(valid=False, success=False,
                          reason='invalid_state' if not self._finite() else 'solver_warning')
        return result

    def reset(self, seed=0):
        if type(seed) is not int or not 0 <= seed <= 999999:
            raise ValueError('seed must be an integer in [0,999999]')
        mj, np = sim._native_imports()
        self.ready, self.seed, self.warnings = False, seed, []
        with sim.native_warnings(self.warnings):
            self.model = build_task_model(self.assets, self.config)
            self.data = mj.MjData(self.model)
            self.arm_qpos, self.arm_dof = np.arange(6), np.arange(6)
            self.linkage_qpos = np.arange(6, 8)
            self.tcp = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, 'piper_task_tcp')
            self.block = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'task_block')
            self.block_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'task_block_geom')
            self.table_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'task_table')
            self.floor_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'task_floor')
            self.pad_geoms = {}
            for side, name in (('left', 'link7'), ('right', 'link8')):
                body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
                self.pad_geoms[side] = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == body
                                       and self.model.geom_type[i] == mj.mjtGeom.mjGEOM_BOX and self.model.geom_contype[i]]
                if len(self.pad_geoms[side]) != 2:
                    raise ValueError('PiPER stock finger collision boxes changed')
            self.robot_geoms = {i for i in range(self.model.ngeom) if 1 <= self.model.geom_bodyid[i] <= 9}
            self.moving_geoms = {i for i in self.robot_geoms if self.model.geom_bodyid[i] > 1}
            self.arm_geoms = {i for i in self.robot_geoms if 2 <= self.model.geom_bodyid[i] <= 7}
            self.data.qpos[:8] = [0., 1.57, -1.3485, 0., 0., 0., .035, -.035]
            self.command_targets = self.data.qpos[:6].copy()
            self.command_gripper = .035
            mj.mj_forward(self.model, self.data)
            for _ in range(round(self.config.settle_s/self.config.timestep)):
                self._controls()
                mj.mj_step(self.model, self.data)
                mj.mj_forward(self.model, self.data)
                if not self._finite() or self.warnings or any(x.number for x in self.data.warning):
                    raise RuntimeError('PiPER settling produced invalid native physics')
            counts, _, supported, critical, _ = self._contacts()
            velocity = np.empty(6)
            mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, self.block, velocity, 0)
            point = self.data.xpos[self.block]
            expected_z = self.config.table_top_m+self.config.object_half_size_m[2]
            if (not supported or critical or any(counts.values())
                    or np.linalg.norm(point[:2]-self.config.object_position_m[:2]) > .005
                    or abs(float(point[2])-expected_z) > .002
                    or np.linalg.norm(velocity[3:]) > .02 or np.linalg.norm(velocity[:3]) > .2):
                raise RuntimeError('Reset must settle a supported, clear, stationary block near the declared start')
        self.initial_object_position = self.data.xpos[self.block].copy()
        self.elapsed, self.steps, self.dwell, self.lift_dwell = 0., 0, 0., 0.
        self.max_lift, self.max_pad_force, self.critical_contact_ticks = 0., 0., 0
        self.lift_seen, self.grasp_seen, self.done = False, False, False
        self.carry_complete, self.retention_lost = False, False
        self.carry_contact_loss, self.max_carry_contact_loss, self.carry_contact_loss_events = 0., 0., 0
        self.first_success_time, self.reason = None, 'running'
        self.contact_pair_native_ticks, self.ready = {}, True
        return self.observation(), self.info()

    def _controls(self):
        self.data.ctrl[:6] = self.command_targets
        self.data.ctrl[6] = self.command_gripper

    def _contacts(self):
        mj, np = sim._native_imports()
        counts, forces, table, critical, pairs = {'left': 0, 'right': 0}, {'left': 0., 'right': 0.}, 0, 0, []
        pads = set(sum(self.pad_geoms.values(), []))
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            force = np.zeros(6)
            mj.mj_contactForce(self.model, self.data, index, force)
            if force[0] <= .01:
                continue
            pair = {int(contact.geom1), int(contact.geom2)}
            pairs.append({'geom_ids': sorted(pair), 'normal_force_n': float(force[0])})
            if self.block_geom in pair:
                table += int(self.table_geom in pair)
                for side in counts:
                    if pair.intersection(self.pad_geoms[side]):
                        counts[side] += 1
                        forces[side] += float(force[0])
                critical += int(bool(pair.intersection(self.robot_geoms-pads)) or self.floor_geom in pair)
            elif pair.intersection({self.table_geom, self.floor_geom}) and pair.intersection(self.moving_geoms):
                critical += 1
            elif pair.intersection(self.arm_geoms) and pair.issubset(self.robot_geoms):
                critical += 1
        return counts, forces, table, critical, pairs

    def observation(self):
        mj, np = sim._native_imports()
        if not self._finite():
            raise RuntimeError('Cannot record nonfinite PiPER physics')
        counts, forces, table, critical, pairs = self._contacts()
        quat, velocity = np.empty(4), np.empty(6)
        mj.mju_mat2Quat(quat, self.data.site_xmat[self.tcp])
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, self.block, velocity, 0)
        return {'t_seconds': self.elapsed, 'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist(),
                'joint_position_rad': self.data.qpos[:6].tolist(), 'joint_velocity_rad_s': self.data.qvel[:6].tolist(),
                'gripper_joint_position_m': self.data.qpos[6:8].tolist(),
                'tcp_position_m': self.data.site_xpos[self.tcp].tolist(), 'tcp_quaternion_wxyz': quat.tolist(),
                'object_position_m': self.data.xpos[self.block].tolist(), 'object_quaternion_wxyz': self.data.xquat[self.block].tolist(),
                'object_linear_velocity_m_s': velocity[3:].tolist(), 'object_angular_velocity_rad_s': velocity[:3].tolist(),
                'target_position_m': list(self.config.target_position_m), 'grasp_contact_counts': counts,
                'pad_normal_forces_n': forces, 'table_object_contact_count': table, 'unexpected_contact_count': critical,
                'contact_count': int(self.data.ncon), 'active_contact_pairs': pairs,
                'native_actuator_controls': self.data.ctrl.tolist(), 'actuator_force_native': self.data.actuator_force.tolist(),
                'issued_joint_targets_rad': None, 'issued_gripper_position_m': None, 'command': None,
                'observation_origin': 'privileged_native_simulator_state'}

    def pose_action(self, position_m, quaternion_wxyz, gripper_position_m):
        if not self.ready or self.done:
            raise RuntimeError('Reset required before another PiPER action')
        mj, np = sim._native_imports()
        position = np.asarray(_vector(position_m, 3, 'tool position'))
        quaternion = np.asarray(_vector(quaternion_wxyz, 4, 'tool quaternion', -1, 1))
        if not math.isclose(float(quaternion@quaternion), 1., abs_tol=1e-8, rel_tol=0):
            raise ValueError('Tool quaternion must be normalized wxyz')
        jaw = _number(gripper_position_m, 'gripper position', 0, .035)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp)
        current, rotation = np.empty(4), np.empty(3)
        mj.mju_mat2Quat(current, self.data.site_xmat[self.tcp])
        mj.mju_subQuat(rotation, quaternion, current)
        rotation = self.data.site_xmat[self.tcp].reshape(3, 3)@rotation
        translation = position-self.data.site_xpos[self.tcp]
        for error, maximum in ((translation, .015), (rotation, .05)):
            norm = float(np.linalg.norm(error))
            if norm > maximum:
                error *= maximum/norm
        jacobian = np.vstack((jacp[:, :6], .25*jacr[:, :6]))
        error = np.concatenate((translation, .25*rotation))
        delta = jacobian.T@np.linalg.solve(jacobian@jacobian.T+.012**2*np.eye(6), error)
        targets = np.clip(self.data.qpos[:6]+np.clip(delta, -.06, .06),
                          self.model.actuator_ctrlrange[:6, 0]+.02, self.model.actuator_ctrlrange[:6, 1]-.02)
        return {'joint_targets_rad': targets.tolist(), 'gripper_position_m': jaw}

    def step(self, action):
        if not self.ready or self.done:
            raise RuntimeError('Reset required before another PiPER step')
        if type(action) is not dict or set(action) != {'joint_targets_rad', 'gripper_position_m'}:
            raise ValueError('PiPER requires six radian arm targets and one metre jaw target')
        mj, np = sim._native_imports()
        targets = np.asarray(_vector(action['joint_targets_rad'], 6, 'arm targets'))
        jaw = _number(action['gripper_position_m'], 'jaw target', 0, .035)
        if np.any(targets < self.model.actuator_ctrlrange[:6, 0]) or np.any(targets > self.model.actuator_ctrlrange[:6, 1]):
            raise ValueError('Arm target exceeds native limits')
        if not self._finite():
            self.reason, self.done = 'invalid_state', True
            return None, self.info()
        messages = []
        with sim.native_warnings(messages):
            for _ in range(round(self.config.control_dt/self.config.timestep)):
                self.command_targets += np.clip(targets-self.command_targets,
                                               -self.config.joint_target_rate_rad_s*self.config.timestep,
                                               self.config.joint_target_rate_rad_s*self.config.timestep)
                self.command_gripper += float(np.clip(jaw-self.command_gripper,
                                                     -self.config.jaw_target_rate_m_s*self.config.timestep,
                                                     self.config.jaw_target_rate_m_s*self.config.timestep))
                self._controls()
                mj.mj_step(self.model, self.data)
                mj.mj_forward(self.model, self.data)
                self.elapsed += self.config.timestep
                if not self._finite() or messages or any(x.number for x in self.data.warning):
                    self.reason = 'invalid_state' if not self._finite() else 'solver_warning'
                    self.done = True
                    break
                counts, forces, supported, critical, pairs = self._contacts()
                for pair in {tuple(contact['geom_ids']) for contact in pairs}:
                    name = ':'.join(map(str, pair))
                    self.contact_pair_native_ticks[name] = self.contact_pair_native_ticks.get(name, 0)+1
                velocity = np.empty(6)
                mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, self.block, velocity, 0)
                both = all(counts[side] > 0 for side in counts)
                lift = float(self.data.xpos[self.block, 2]-self.initial_object_position[2])
                distance = float(np.linalg.norm(self.data.xpos[self.block]-self.config.target_position_m))
                actual_open = self.data.qpos[6] > .028 and self.data.qpos[7] < -.028
                self._outcome_tick(both=both, supported=supported, lift=lift, distance=distance,
                                   released=not any(counts.values()) and actual_open,
                                   settled=np.linalg.norm(velocity[3:]) < .035 and np.linalg.norm(velocity[:3]) < .3,
                                   clear=self.data.site_xpos[self.tcp, 2] >= self.data.xpos[self.block, 2]+self.config.retreat_clearance_m,
                                   critical=critical, pad_force=max(forces.values()))
        self.warnings.extend(messages)
        self.steps += 1
        if not self.done and self.steps >= round(self.config.timeout_s/self.config.control_dt):
            self.reason, self.done = 'timeout', True
        return self.observation() if self._finite() else None, self.info()


class PiperTaskPolicy:
    def __init__(self, environment):
        self.env, self.phase, self.ticks = environment, 'approach', 0
        self.grasp_position, self.held_offset = None, None
        self.issued_phase = self.phase

    def __call__(self, observation):
        mj, np = sim._native_imports()
        self.issued_phase = self.phase
        obj, tcp, target = (np.asarray(observation[name]) for name in ('object_position_m', 'tcp_position_m', 'target_position_m'))
        both, jaw = all(observation['grasp_contact_counts'][side] for side in ('left', 'right')), .035
        orientation = self.env.config.tool_orientation_wxyz
        if self.phase in ('approach', 'lift', 'transfer', 'retreat', 'verify'):
            orientation = self.env.config.carry_orientation_wxyz
        rotation, angular_error = np.empty(9), np.empty(3)
        mj.mju_quat2Mat(rotation, np.asarray(orientation))
        mj.mju_subQuat(angular_error, np.asarray(orientation), np.asarray(observation['tcp_quaternion_wxyz']))
        aligned = np.linalg.norm(angular_error) < .12
        offset = rotation.reshape(3, 3)@self.held_offset if self.held_offset is not None else None
        if self.phase == 'approach':
            goal = obj+np.asarray([0., 0., .10])
            if np.linalg.norm(tcp-goal) < .008 and aligned:
                self.grasp_position = obj+np.asarray([0., 0., .002])
                self.phase, self.ticks = 'descend', 0
        elif self.phase == 'descend':
            goal = self.grasp_position
            if np.linalg.norm(tcp-goal) < .004 and aligned:
                self.phase, self.ticks = 'close', 0
        elif self.phase == 'close':
            goal, jaw = self.grasp_position, 0.
            if both and min(observation['pad_normal_forces_n'].values()) > .01 and self.ticks >= 35:
                current_rotation = np.empty(9)
                mj.mju_quat2Mat(current_rotation, np.asarray(observation['tcp_quaternion_wxyz']))
                self.held_offset = current_rotation.reshape(3, 3).T@(obj-tcp)
                self.phase, self.ticks = 'lift', 0
        elif self.phase == 'lift':
            goal, jaw = self.grasp_position+np.asarray([0., 0., .10]), 0.
            if self.env.lift_seen and np.linalg.norm(tcp-goal) < .008:
                self.phase, self.ticks = 'transfer', 0
        elif self.phase == 'transfer':
            goal, jaw = target-offset+np.asarray([0., 0., .10]), 0.
            if np.linalg.norm(tcp-goal) < .008:
                self.phase, self.ticks = 'place', 0
        elif self.phase == 'place':
            goal, jaw = target-offset, 0.
            if observation['table_object_contact_count'] and np.linalg.norm(obj-target) < .02 and np.linalg.norm(tcp-goal) < .006:
                self.phase, self.ticks = 'release', 0
        elif self.phase == 'release':
            goal = target-offset
            if not any(observation['grasp_contact_counts'].values()) and self.ticks >= 35:
                self.phase, self.ticks = 'retreat', 0
        else:
            goal = target+np.asarray([0., 0., .13])
            if np.linalg.norm(tcp-goal) < .008:
                self.phase = 'verify'
        self.ticks += 1
        return self.env.pose_action(goal.tolist(), orientation, jaw)


def run_episode(assets, config=None, *, seed=0, control='placement', arm_tape=None):
    config = _config(config)
    if control not in ('placement', 'replay_open_jaw', 'hold'):
        raise ValueError('Unknown PiPER task control')
    started, latencies, environment, phase = time.perf_counter(), [], None, 'validation'
    result = {'schema': SCHEMA, 'profile': PROFILE, 'scope': SCOPE, 'physical_robot_ready': False,
              'seed': seed, 'control': control, 'config': asdict(config), 'timestep_seconds': config.timestep,
              'valid': False, 'success': False, 'reason': 'not_started', 'error': None, 'warnings': [],
              'simulated_seconds': 0., 'trajectory': [], 'actions': [], 'arm_tape': [], 'metrics': {}, 'timing': {},
              'contract': task_contract(), 'platform': {'system': platform.system(), 'machine': platform.machine(), 'python': platform.python_version()}}
    source = Path(__file__).parent
    result['implementation_sha256'] = {name: sha256((source/name).read_bytes()).hexdigest()
                                       for name in ('piper_task.py', 'robot_profiles.py', 'robot_profiles.lock.json', 'reference_task.py', 'reference_sim.py')}
    try:
        if control == 'replay_open_jaw':
            if type(arm_tape) is not list or len(arm_tape) != round(config.timeout_s/config.control_dt):
                raise ValueError('Open-jaw control requires a complete exact-duration arm tape')
            arm_tape = [list(_vector(row, 6, 'arm replay')) for row in arm_tape]
        phase = 'setup'
        result['asset_provenance'] = robot_profiles.verify_assets('piper', _asset_root(assets))
        environment = PiperTaskEnv(assets, config)
        observation, info = environment.reset(seed)
        mj, np = sim._native_imports()
        result['platform'].update(mujoco_version=mj.__version__, numpy_version=np.__version__)
        result['compiled_model_sha256'] = sim.model_digest(environment.model)
        result['initial_object_position_m'] = environment.initial_object_position.tolist()
        result['native_servo_gains'] = environment.model.actuator_gainprm[:, 0].tolist()
        result['native_servo_bias'] = environment.model.actuator_biasprm[:, :3].tolist()
        result['native_force_limits'] = environment.model.actuator_forcerange.tolist()
        result['native_control_limits'] = environment.model.actuator_ctrlrange.tolist()
        result['native_model_settings'] = {'dimensions': {name: int(getattr(environment.model, name)) for name in ('nq', 'nv', 'nu', 'neq', 'nmocap')},
                                          'gravity_m_s2': environment.model.opt.gravity.tolist(),
                                          'body_gravcomp': environment.model.body_gravcomp.tolist()}
        result['contact_geometry'] = {'pad_geom_ids': environment.pad_geoms,
                                      'geom_body_names': [mj.mj_id2name(environment.model, mj.mjtObj.mjOBJ_BODY, int(body)) for body in environment.model.geom_bodyid]}
        result['trajectory'].append(observation)
        result['timing']['setup_seconds'] = time.perf_counter()-started
        policy, hold, phase = PiperTaskPolicy(environment), observation['joint_position_rad'], 'rollout'
        for index in range(round(config.timeout_s/config.control_dt)):
            began = time.perf_counter()
            action = policy(observation) if control == 'placement' else {'joint_targets_rad': hold if control == 'hold' else arm_tape[index], 'gripper_position_m': .035}
            record = {'t_seconds': observation['t_seconds'], **action, 'phase': policy.issued_phase if control == 'placement' else control}
            result['actions'].append(record)
            result['arm_tape'].append(list(action['joint_targets_rad']))
            observation.update(issued_joint_targets_rad=list(action['joint_targets_rad']), issued_gripper_position_m=action['gripper_position_m'], command=record)
            observation, info = environment.step(action)
            latencies.append(time.perf_counter()-began)
            if observation is not None:
                result['trajectory'].append(observation)
            if environment.done:
                break
        result.update(valid=info['valid'], success=info['success'], reason=info['reason'], warnings=info['warnings'],
                      simulated_seconds=info['simulation_time'], metrics={**info, 'contact_pair_native_ticks': environment.contact_pair_native_ticks})
    except Exception as exc:
        result['reason'], result['error'] = 'exception', {'phase': phase, 'type': type(exc).__name__, 'message': str(exc)[:500]}
        if environment is not None:
            result['warnings'] = list(getattr(environment, 'warnings', []))
            result['simulated_seconds'] = getattr(environment, 'elapsed', 0.)
    elapsed = time.perf_counter()-started
    result['timing'].update(episode_wall_seconds=elapsed, control_step_seconds=sim._summary(latencies) if latencies else None,
                            control_boundaries=len(latencies), missed_control_boundaries=sum(x > config.control_dt for x in latencies),
                            simulated_seconds_per_wall_second=result['simulated_seconds']/elapsed,
                            scope='Wall computation including setup/IK/contact/state snapshots; not real-time sensor-to-actuator latency')
    return result


def run_matched(assets, config=None, *, seed=0):
    config = _config(config)
    placement = run_episode(assets, config, seed=seed)
    tape = placement['arm_tape'] if placement['valid'] and len(placement['arm_tape']) == round(config.timeout_s/config.control_dt) else None
    return [placement, run_episode(assets, config, seed=seed, control='replay_open_jaw', arm_tape=tape),
            run_episode(assets, config, seed=seed, control='hold')]
