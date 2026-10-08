"""Original six-axis tabletop arm with a native contact-based parallel-jaw grasp.

This authored simulation profile uses privileged state and simplified rigid links.
It is not calibrated hardware, a learned video policy or a cup/handle model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import json
import math

import mujoco
import numpy as np

from .env import _NATIVE_LOCK

ARM_PROFILE = "dvidia-authored-6dof-parallel-jaw-v0"
ADAPTER_ID = "contact-pick-place-box-v0"
TASK_ID = "ArmPickPlace-v0"
WORKSPACE_MIN = (0.25, -0.20, 0.29)
WORKSPACE_MAX = (0.60, 0.20, 0.55)
JOINT_LIMITS = ((-2.6, 2.6), (-2.2, 1.6), (-2.7, 2.7), (-3.0, 3.0), (-2.8, 2.8), (-3.1, 3.1))
HOME = (-0.60, -1.10, 0.90, 0.0, 0.20, 0.60)
JOINT_KP = np.array((400., 400., 350., 100., 100., 80.))


@dataclass(frozen=True)
class ArmConfig:
    timestep: float = 0.001
    control_dt: float = 0.02
    horizon: float = 18.0
    settle_time: float = 0.4
    object_position: tuple = (0.42, -0.04, 0.31)
    target_position: tuple = (0.54, 0.10, 0.31)
    object_half_size: tuple = (0.02, 0.02, 0.02)
    object_mass: float = 0.04
    object_friction: float = 0.8
    table_height: float = 0.29
    position_tolerance: float = 0.025
    dwell_seconds: float = 0.25
    scene_jitter: float = 0.006
    joint_torque_limit: float = 40.0
    jaw_force_limit: float = 15.0
    joint_target_rate_limit: float = 3.0
    jaw_width_rate_limit: float = 0.16
    gravity: float = 9.81

    def __post_init__(self):
        for name in ("timestep", "control_dt", "horizon", "settle_time", "joint_torque_limit", "jaw_force_limit", "joint_target_rate_limit", "jaw_width_rate_limit", "gravity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for numerator, denominator in ((self.control_dt, self.timestep), (self.horizon, self.control_dt), (self.settle_time, self.timestep)):
            ratio = numerator / denominator
            if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
                raise ValueError("Arm timing ratios must be positive integers")
        for name, low, high in (("object_mass", .02, .08), ("object_friction", .4, 1.5),
                                ("position_tolerance", .01, .035), ("dwell_seconds", .15, .6),
                                ("scene_jitter", 0., .01)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must lie in [{low}, {high}]")
        if self.table_height != .29:
            raise ValueError("This authored arm profile fixes the table height at 0.29 m")
        half_size = np.asarray(self.object_half_size, dtype=float)
        if half_size.shape != (3,) or not np.isfinite(half_size).all() or np.any(half_size < .015) or np.any(half_size > .025):
            raise ValueError("Object full dimensions must lie in [0.03, 0.05] m")
        object.__setattr__(self, "object_half_size", tuple(half_size.tolist()))
        for name in ("object_position", "target_position"):
            values = self._placement(getattr(self, name))
            object.__setattr__(self, name, tuple(values.tolist()))
        if np.linalg.norm(np.asarray(self.object_position)[:2] - np.asarray(self.target_position)[:2]) < .075:
            raise ValueError("A placement task must move the object by at least 0.075 m")

    def _placement(self, value):
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Placements must contain three finite world-frame coordinates")
        if not .25 <= point[0] <= .60 or not -.20 <= point[1] <= .20:
            raise ValueError("Placement lies outside the authored arm reach envelope")
        if np.linalg.norm(point[:2]) > .56:
            raise ValueError("Placement lies outside the raised-tool radial reach envelope (0.56 m)")
        if abs(point[2] - (self.table_height + self.object_half_size[2])) > .01000001:
            raise ValueError("Object and target must lie on the table support plane")
        return point


class ArmEnv:
    def __init__(self, config: ArmConfig | None = None):
        self.config = config or ArmConfig()
        if not isinstance(self.config, ArmConfig):
            raise TypeError("config must be ArmConfig")
        if mujoco.__version__ != "3.15.0":
            raise RuntimeError("This authored arm profile requires native MuJoCo 3.15.0")
        self._ready = False

    @contextmanager
    def _native(self, messages):
        with _NATIVE_LOCK:
            previous = mujoco.get_mju_user_warning()
            mujoco.set_mju_user_warning(lambda message: messages.append(str(message)))
            try:
                yield
            finally:
                mujoco.set_mju_user_warning(previous)

    def _xml(self, object_position):
        c = self.config
        limits = [f'{low} {high}' for low, high in JOINT_LIMITS]
        position = ' '.join(map(str, object_position))
        half_size = ' '.join(map(str, c.object_half_size))
        actuators = ''.join(f'<position name="a{i}" joint="j{i}" kp="{JOINT_KP[i]}" kv="{[25,25,20,4,4,3][i]}" ctrlrange="{limits[i]}" forcerange="{-c.joint_torque_limit} {c.joint_torque_limit}"/>' for i in range(6))
        actuators += ''.join(f'<position name="a_{side}" joint="jaw_{side}" kp="250" kv="5" ctrlrange="0 0.04" forcerange="{-c.jaw_force_limit} {c.jaw_force_limit}"/>' for side in ('left', 'right'))
        return f'''<mujoco model="{TASK_ID}"><compiler angle="radian"/><option timestep="{c.timestep}" gravity="0 0 {-c.gravity}" integrator="implicitfast" solver="Newton" iterations="70" tolerance="1e-10" cone="elliptic"/><size memory="32M"/>
        <default><joint damping="0.06" armature="0.004"/><geom solref="0.008 1" solimp="0.95 0.99 0.001" condim="3" friction="0.8 0 0"/><position ctrllimited="true" forcelimited="true"/></default>
        <worldbody><geom name="table" type="box" pos="0.43 0 {c.table_height/2}" size="0.32 0.28 {c.table_height/2}" rgba="0.8 0.8 0.76 1"/>
        <body name="base" pos="0 0 0.08"><geom type="cylinder" size="0.06 0.08" mass="1" contype="0" conaffinity="0"/>
        <body name="b0" pos="0 0 0.07"><joint name="j0" axis="0 0 1" range="{limits[0]}"/><geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.028" mass="0.15" contype="0" conaffinity="0"/>
        <body name="b1" pos="0 0 0.08"><joint name="j1" axis="0 1 0" range="{limits[1]}"/><geom type="capsule" fromto="0 0 0 0.28 0 0" size="0.024" mass="0.18" contype="0" conaffinity="0"/>
        <body name="b2" pos="0.28 0 0"><joint name="j2" axis="0 1 0" range="{limits[2]}"/><geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.021" mass="0.16" contype="0" conaffinity="0"/>
        <body name="b3" pos="0.24 0 0"><joint name="j3" axis="1 0 0" range="{limits[3]}"/><geom type="capsule" fromto="0 0 0 0.055 0 0" size="0.022" mass="0.07" contype="0" conaffinity="0"/>
        <body name="b4" pos="0.055 0 0"><joint name="j4" axis="0 1 0" range="{limits[4]}"/><geom type="capsule" fromto="0 0 0 0.055 0 0" size="0.020" mass="0.06" contype="0" conaffinity="0"/>
        <body name="b5" pos="0.055 0 0"><joint name="j5" axis="0 0 1" range="{limits[5]}"/><geom name="palm" type="box" pos="0 0 0.006" size="0.048 0.024 0.012" mass="0.06" contype="0" conaffinity="0"/>
        <site name="tcp" pos="0 0 -0.04" size="0.003"/>
        <body name="left_pad" pos="0.006 0 0"><joint name="jaw_left" type="slide" axis="1 0 0" range="0 0.04" damping="0.08" armature="0.002"/><geom name="left_pad_geom" type="box" pos="0 0 -0.032" size="0.006 0.018 0.032" mass="0.018" friction="1 0 0"/></body>
        <body name="right_pad" pos="-0.006 0 0"><joint name="jaw_right" type="slide" axis="-1 0 0" range="0 0.04" damping="0.08" armature="0.002"/><geom name="right_pad_geom" type="box" pos="0 0 -0.032" size="0.006 0.018 0.032" mass="0.018" friction="1 0 0"/></body>
        </body></body></body></body></body></body></body>
        <body name="object" pos="{position}"><freejoint name="object_free"/><geom name="object_geom" type="box" size="{half_size}" mass="{c.object_mass}" friction="{c.object_friction} 0 0"/></body>
        </worldbody><contact><exclude body1="left_pad" body2="right_pad"/></contact><actuator>{actuators}</actuator></mujoco>'''

    def reset(self, seed=0, object_position=None, target_position=None):
        self._ready = False
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        c = self.config
        rng = np.random.default_rng(seed)
        start = np.array(c.object_position if object_position is None else c._placement(object_position), dtype=float)
        target = np.array(c.target_position if target_position is None else c._placement(target_position), dtype=float)
        if object_position is None:
            start[:2] += rng.uniform(-c.scene_jitter, c.scene_jitter, 2)
        if target_position is None:
            target[:2] += rng.uniform(-c.scene_jitter, c.scene_jitter, 2)
        c._placement(start); c._placement(target)
        if np.linalg.norm(start[:2] - target[:2]) < .075:
            raise ValueError("A placement task must require nontrivial movement")
        # Authored reset placement is the only object state initialization.
        start[2] = c.table_height + c.object_half_size[2] + .002
        target[2] = c.table_height + c.object_half_size[2]
        self.seed, self.target = seed, target
        self.scene_xml = self._xml(start)
        warnings = []
        with self._native(warnings):
            self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
            self.data = mujoco.MjData(self.model)
            self.joint_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'j{i}') for i in range(6)])
            self.joint_qpos = self.model.jnt_qposadr[self.joint_ids]
            self.joint_dofs = self.model.jnt_dofadr[self.joint_ids]
            self.jaw_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'jaw_{side}') for side in ('left', 'right')])
            self.jaw_qpos = self.model.jnt_qposadr[self.jaw_ids]
            self.data.qpos[self.joint_qpos] = HOME
            self.data.qpos[self.jaw_qpos] = .04
            self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
            self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'object')
            self.geom_ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ('left_pad_geom', 'right_pad_geom', 'object_geom', 'table')}
            self.arm_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ('base', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5')]
            self._last_targets, self._last_width = np.array(HOME), .08
            self._command_targets, self._command_width = np.array(HOME), .08
            self._command_rate_limited = False
            mujoco.mj_forward(self.model, self.data)
            for _ in range(round(c.settle_time/c.timestep)):
                self._controls(self._last_targets, .08)
                mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
        self.initial_object_position = self.data.xpos[self.object_id].copy()
        self._elapsed = 0.0
        self._steps = 0
        self._dwell = 0.0
        self._reason = 'running'
        self._done = False
        self._grasp_seen = False
        self._lift_seen = False
        self._lift_dwell = 0.0
        self._warnings = list(warnings)
        self._max_lift = 0.0
        self._max_pad_force = 0.0
        self._peak_joint_velocity = np.zeros(6)
        self._peak_jaw_velocity = np.zeros(2)
        self._peak_actuator_force = np.zeros(8)
        self._ready = True
        if warnings or not self._finite() or any(w.number for w in self.data.warning):
            self._ready = False
            raise RuntimeError("Authored arm settling produced invalid physics")
        return self._observation(), self._info()

    def _controls(self, joint_targets, width):
        # Model-based gravity feedforward adjusts native position servo targets;
        # the native actuator's own force cap still bounds every joint torque.
        compensation = self.data.qfrc_bias[self.joint_dofs] / JOINT_KP
        limits = np.asarray(JOINT_LIMITS)
        self.data.ctrl[:6] = np.clip(joint_targets + compensation, limits[:,0], limits[:,1])
        self.data.ctrl[6:] = width / 2

    def _advance_commands(self, targets, width):
        """Ramp authored targets each physics tick, before gravity feedforward.

        This does not clamp qvel or write a kinematic trajectory. Actual motion
        remains the result of bounded native servos, gravity and contacts.
        """
        joint_step = self.config.joint_target_rate_limit*self.config.timestep
        jaw_step = self.config.jaw_width_rate_limit*self.config.timestep
        delta = targets-self._command_targets
        gap_delta = width-self._command_width
        self._command_rate_limited |= bool(np.any(np.abs(delta)>joint_step+1e-12) or abs(gap_delta)>jaw_step+1e-12)
        self._command_targets += np.clip(delta,-joint_step,joint_step)
        self._command_width += float(np.clip(gap_delta,-jaw_step,jaw_step))

    def _finite(self):
        return math.isfinite(float(self.data.time)) and all(np.isfinite(x).all() for x in (self.data.qpos, self.data.qvel, self.data.qacc, self.data.xpos))

    def _contact_state(self):
        counts = {'left':0, 'right':0, 'table':0}
        forces = {'left':0.0, 'right':0.0, 'table':0.0}
        object_geom = self.geom_ids['object_geom']
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            if object_geom not in pair:
                continue
            for side, geom in (('left', self.geom_ids['left_pad_geom']), ('right', self.geom_ids['right_pad_geom']), ('table', self.geom_ids['table'])):
                if geom in pair:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, index, force)
                    if force[0] > .01:
                        counts[side] += 1
                        forces[side] += float(force[0])
        return counts, forces

    def _velocity(self, kind, ident):
        result = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, kind, ident, result, 0)
        return result

    def _observation(self):
        if not self._finite():
            return {'joint_position':[], 'joint_velocity':[], 'joint_actuator_torque':[], 'jaw_actuator_force':[], 'end_effector_position':[], 'end_effector_quaternion':[], 'end_effector_velocity':[], 'object_position':[], 'object_quaternion':[], 'object_velocity':[], 'object_angular_velocity':[], 'target_position':self.target.tolist(), 'arm_points':[], 'gripper_points':[], 'object_size':(2*np.asarray(self.config.object_half_size)).tolist(), 'gripper_width':None, 'grasp_contacts':{'left':0,'right':0}, 'pad_normal_forces':{'left':0.,'right':0.}, 'effective_sliding_friction':{'jaw_object':max(1.,self.config.object_friction),'table_object':max(.8,self.config.object_friction)}, 'table_contacts':0}
        counts, forces = self._contact_state()
        ee_quat = np.empty(4)
        mujoco.mju_mat2Quat(ee_quat, self.data.site_xmat[self.tcp_id])
        object_velocity = self._velocity(mujoco.mjtObj.mjOBJ_BODY, self.object_id)
        ee_velocity = self._velocity(mujoco.mjtObj.mjOBJ_SITE, self.tcp_id)
        return {'joint_position':self.data.qpos[self.joint_qpos].tolist(), 'joint_velocity':self.data.qvel[self.joint_dofs].tolist(),
                'joint_actuator_torque':self.data.actuator_force[:6].tolist(),'jaw_actuator_force':self.data.actuator_force[6:].tolist(),
                'end_effector_position':self.data.site_xpos[self.tcp_id].tolist(), 'end_effector_quaternion':ee_quat.tolist(), 'end_effector_velocity':ee_velocity[3:].tolist(),
                'object_position':self.data.xpos[self.object_id].tolist(), 'object_quaternion':self.data.xquat[self.object_id].tolist(),
                'object_velocity':object_velocity[3:].tolist(), 'object_angular_velocity':object_velocity[:3].tolist(), 'target_position':self.target.tolist(),
                'arm_points':np.vstack((self.data.xpos[self.arm_body_ids], self.data.site_xpos[self.tcp_id])).tolist(),
                'gripper_points':self.data.geom_xpos[[self.geom_ids['left_pad_geom'],self.geom_ids['right_pad_geom']]].tolist(),
                'object_size':(2*np.asarray(self.config.object_half_size)).tolist(), 'gripper_width':float(self.data.qpos[self.jaw_qpos].sum()),
                'grasp_contacts':{'left':counts['left'],'right':counts['right']},
                'pad_normal_forces':{'left':forces['left'],'right':forces['right']},
                'effective_sliding_friction':{'jaw_object':max(1.,self.config.object_friction),'table_object':max(.8,self.config.object_friction)},
                'table_contacts':counts['table']}

    def _info(self):
        counts, forces = self._contact_state() if self._finite() else ({'left':0,'right':0,'table':0},{'left':0.,'right':0.,'table':0.})
        return {'task_id':TASK_ID,'arm_profile':ARM_PROFILE,'adapter_id':ADAPTER_ID,'seed':self.seed,'reason':self._reason,
                'success':self._reason=='success','valid':self._reason not in ('invalid_state','solver_warning','object_dropped'),
                'simulation_time':self._elapsed,'steps':self._steps,'dwell_elapsed':self._dwell,'grasp_seen':self._grasp_seen,'lift_seen':self._lift_seen,
                'max_object_lift':self._max_lift,'grasp_contacts':{'left':counts['left'],'right':counts['right']},'pad_normal_forces':{'left':forces['left'],'right':forces['right']},
                'table_contacts':counts['table'],'max_pad_normal_force':self._max_pad_force,'target_distance':float(np.linalg.norm(self.data.xpos[self.object_id]-self.target)) if self._finite() else None,
                'warnings':list(self._warnings),'lift_dwell_elapsed':self._lift_dwell,
                'effective_sliding_friction':{'jaw_object':max(1.,self.config.object_friction),'table_object':max(.8,self.config.object_friction),'mixing':'maximum at equal geom priority'},
                'initial_object_position':self.initial_object_position.tolist(),'qualification':'simulation-only','observations':'privileged simulator state',
                'actuation':'six bounded joint position servos and two bounded jaw servos','object_profile':'rigid box; unqualified proxy for cup skills',
                'action_contract':{'independent_commands':7,'physical_robot_joints':8,'hinge_target_units':'rad','jaw_gap_units':'m','joint_limits':list(map(list,JOINT_LIMITS)),'jaw_gap_limits':[0.,.08],'joint_torque_limit_nm':self.config.joint_torque_limit,'jaw_actuator_force_limit_n':self.config.jaw_force_limit,'ik_orientation':'fixed identity rotation; XYZ position goal only','force_observation':'summed native per-pad normal contact forces, N; not actuator force'},
                'command_limits':{'joint_target_rate_rad_s':self.config.joint_target_rate_limit,'jaw_gap_rate_m_s':self.config.jaw_width_rate_limit,'scope':'authored target trajectory each native tick, before gravity feedforward; not a physical velocity guarantee'},
                'requested_joint_targets':self._last_targets.tolist(),'requested_gripper_width':self._last_width,
                'commanded_joint_targets':self._command_targets.tolist(),'commanded_gripper_width':self._command_width,
                'command_rate_limited':self._command_rate_limited,'native_position_controls':self.data.ctrl.tolist() if np.isfinite(self.data.ctrl).all() else [],
                'measured_joint_velocity_rad_s':self.data.qvel[self.joint_dofs].tolist() if self._finite() else [],
                'native_tick_peaks':{'cadence_seconds':self.config.timestep,'scope':'absolute peaks after reset, sampled after every native rollout tick','joint_velocity_rad_s':self._peak_joint_velocity.tolist(),'jaw_slide_velocity_m_s':self._peak_jaw_velocity.tolist(),'actuator_force':self._peak_actuator_force.tolist(),'actuator_force_units':['N*m']*6+['N']*2},
                'collision_profile':'jaws/object/table enabled; arm links and palm collision-excluded; no selfcollision qualification',
                'config':asdict(self.config),'engine':mujoco.__version__,'scene_sha256':hashlib.sha256(self.scene_xml.encode()).hexdigest()}

    def joint_targets_for_pose(self, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('IK goal must contain three finite coordinates')
        jacp, jacr = np.zeros((3,self.model.nv)), np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model,self.data,jacp,jacr,self.tcp_id)
        rotation = self.data.site_xmat[self.tcp_id].reshape(3,3)
        orientation_error = .5 * sum((np.cross(rotation[:,i], np.eye(3)[:,i]) for i in range(3)), np.zeros(3))
        translation_error = position - self.data.site_xpos[self.tcp_id]
        distance = np.linalg.norm(translation_error)
        if distance > .006:
            translation_error *= .006 / distance
        angle = np.linalg.norm(orientation_error)
        if angle > .04:
            orientation_error *= .04 / angle
        jacobian = np.vstack((jacp[:,self.joint_dofs], .25*jacr[:,self.joint_dofs]))
        error = np.r_[translation_error, .25*orientation_error]
        delta = jacobian.T @ np.linalg.solve(jacobian@jacobian.T + .015**2*np.eye(6), error)
        delta = np.clip(delta, -.06, .06)
        limits = np.asarray(JOINT_LIMITS)
        return np.clip(self.data.qpos[self.joint_qpos]+delta,limits[:,0]+.02,limits[:,1]-.02).tolist()

    def step(self, action):
        if not self._ready or self._done:
            raise RuntimeError('Reset is required before another arm step')
        if not isinstance(action, dict) or set(action) != {'joint_targets','gripper_width'}:
            raise ValueError('Arm action requires joint_targets and gripper_width only')
        raw_targets = action['joint_targets']
        if not isinstance(raw_targets,(list,tuple,np.ndarray)) or (isinstance(raw_targets,np.ndarray) and raw_targets.ndim!=1) or len(raw_targets)!=6:
            raise ValueError('Joint targets require six finite numeric entries')
        if any(isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(value) for value in raw_targets):
            raise ValueError('Joint targets require six finite numeric entries; boolean and string entries are rejected')
        targets = np.asarray(raw_targets,dtype=float)
        width = action['gripper_width']
        limits = np.asarray(JOINT_LIMITS)
        if targets.shape != (6,) or not np.isfinite(targets).all() or np.any(targets<limits[:,0]) or np.any(targets>limits[:,1]) or isinstance(width,bool) or not isinstance(width,(int,float)) or not math.isfinite(width) or not 0<=width<=.08:
            raise ValueError('Arm targets violate the declared joint/jaw limits')
        # Reject an already corrupted native state before the engine can repair
        # it by internally resetting coordinates. Such a state is not evidence.
        if not self._finite():
            self._reason='invalid_state';self._done=True
            return self._observation(),-1.,True,False,self._info()
        self._last_targets,self._last_width=targets.copy(),float(width)
        self._command_rate_limited=False
        previous_distance=float(np.linalg.norm(self.data.xpos[self.object_id]-self.target))
        warnings = []
        with self._native(warnings):
            for _ in range(round(self.config.control_dt/self.config.timestep)):
                self._advance_commands(targets,width)
                self._controls(self._command_targets,self._command_width)
                before=float(self.data.time)
                mujoco.mj_step(self.model,self.data)
                mujoco.mj_forward(self.model,self.data)
                self._elapsed+=self.config.timestep
                if not self._finite() or not math.isclose(float(self.data.time)-before,self.config.timestep,rel_tol=0,abs_tol=1e-10):
                    self._reason='invalid_state';self._done=True;break
                if warnings or any(w.number for w in self.data.warning):
                    self._reason='solver_warning';self._done=True;break
                self._peak_joint_velocity=np.maximum(self._peak_joint_velocity,np.abs(self.data.qvel[self.joint_dofs]))
                self._peak_jaw_velocity=np.maximum(self._peak_jaw_velocity,np.abs(self.data.qvel[self.model.jnt_dofadr[self.jaw_ids]]))
                self._peak_actuator_force=np.maximum(self._peak_actuator_force,np.abs(self.data.actuator_force))
                counts,forces=self._contact_state()
                self._max_pad_force=max(self._max_pad_force,forces['left'],forces['right'])
                self._grasp_seen |= counts['left']>0 and counts['right']>0
                lift=float(self.data.xpos[self.object_id,2]-self.initial_object_position[2])
                self._max_lift=max(self._max_lift,lift)
                retained_lift = counts['left']>0 and counts['right']>0 and counts['table']==0 and lift>=.055
                self._lift_dwell = self._lift_dwell+self.config.timestep if retained_lift else 0.
                self._lift_seen |= self._lift_dwell+1e-12>=.10
                velocity=self._velocity(mujoco.mjtObj.mjOBJ_BODY,self.object_id)
                distance=float(np.linalg.norm(self.data.xpos[self.object_id]-self.target))
                released=float(self.data.qpos[self.jaw_qpos].sum())>.055 and counts['left']==0 and counts['right']==0
                supported=counts['table']>0 and abs(self.data.xpos[self.object_id,2]-self.target[2])<.008
                settled=np.linalg.norm(velocity[3:])<.035 and np.linalg.norm(velocity[:3])<.3
                hand_clear=self.data.site_xpos[self.tcp_id,2]>=self.data.xpos[self.object_id,2]+.10
                qualifies=self._lift_seen and self._grasp_seen and released and hand_clear and supported and settled and distance<=self.config.position_tolerance
                self._dwell=self._dwell+self.config.timestep if qualifies else 0.
                if self._dwell+1e-12>=self.config.dwell_seconds:
                    self._reason='success';self._done=True;break
                if self.data.xpos[self.object_id,2]<self.config.table_height-.05:
                    self._reason='object_dropped';self._done=True;break
        self._warnings.extend(warnings)
        self._steps+=1
        if not self._done and self._steps>=round(self.config.horizon/self.config.control_dt):
            self._reason='horizon';self._done=True
        obs=self._observation();info=self._info()
        reward=previous_distance-info['target_distance'] if info['target_distance'] is not None else -1.
        if info['success']:reward+=1.
        if not info['valid']:reward=-1.
        return obs,float(reward),self._done and self._reason!='horizon',self._reason=='horizon',info
