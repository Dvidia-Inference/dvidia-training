"""Encoder-only PiPER geometry and one authored differential-IK update.

Each instance owns a private, pinned robot-only model. There is no live model,
MjData, object, contact, camera or teacher argument. Only kinematics, comPos and
site Jacobians are evaluated; no dynamics or actuator execution occurs. Use one
instance per worker: its private scratch data is reused, not thread-safe.

The controlled point is the authored link6-local [0, 0, .12003] TCP in WORLD
coordinates. The older learning feature is the link6 ORIGIN in BASE coordinates;
fk names both separately. Neither point is a hardware-calibrated fingertip.

pose_action matches PiperTaskEnv.pose_action's bounded DLS calculation. It is
one local geometric update, not a reachability test or collision-safe planner.
Its q input is measured encoders, not previous applied servo references. The
existing downstream rate limiter remains necessary and is not executed here.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from . import reference_sim as sim
from . import robot_profiles


ASSET_LOCK_SHA256 = '4b2ee2ca675aceb2f826983cb3082a3e08e9aa0ff419b23078abcff08ebcc6fe'
ARM_JOINTS = tuple(f'joint{i}' for i in range(1, 7))
TCP_OFFSET_LINK6_M = (0., 0., .12003)
ARM_LIMITS_RAD = ((-2.618, 2.618), (0., 3.14), (-2.697, 0.),
                  (-1.832, 1.832), (-1.22, 1.22), (-3.14, 3.14))


def _vector(value, length, name):
    """Reject malformed data rather than coerce strings, booleans or matrices."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f'{name} requires a numeric vector of length {length}')
    if isinstance(value, np.ndarray):
        if value.shape != (length,) or value.dtype.kind not in 'iuf':
            raise ValueError(f'{name} requires a numeric vector of length {length}')
    elif len(value) != length or any(isinstance(x, (bool, np.bool_)) or
            not isinstance(x, (int, float, np.integer, np.floating)) for x in value):
        raise ValueError(f'{name} requires a numeric vector of length {length}')
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} requires finite numbers') from exc
    if not np.isfinite(result).all():
        raise ValueError(f'{name} requires finite numbers')
    return result


class PiperEncoderKinematics:
    """Fixed PiPER geometry; constructor takes only a verified asset directory.

    Measured angles must be inside the exact native joint ranges, inclusively.
    Even a small measured overshoot is refused, never silently clamped. Issued
    targets use a separate .02 rad interior margin. With accepted measurements,
    that margin cannot enlarge the final change beyond the .06 rad DLS bound.
    Jaw commands are one finger's slide target in [0, .035] metres, not full gap.
    """

    def __init__(self, assets):
        if sha256(robot_profiles.LOCK_PATH.read_bytes()).hexdigest() != ASSET_LOCK_SHA256:
            raise ValueError('PiPER asset lock differs from the kinematics contract')
        root = Path(assets)
        root = root.parent if root.name == 'agilex_piper' else root
        item, blobs = robot_profiles._payload('piper', root)
        self._mj, _ = sim._native_imports()
        spec = sim._spec(blobs, item['folder'], item['model'])
        spec.body('link6').add_site(name='encoder_tcp', pos=TCP_OFFSET_LINK6_M,
                                   size=[.004, 0, 0])
        self._model = spec.compile()
        mj, model = self._mj, self._model
        if (model.nq, model.nv, model.nu, model.njnt, model.neq, model.nmocap,
                model.nbody) != (8, 8, 7, 8, 1, 0, 10):
            raise ValueError('Expected the pinned robot only, without a task scene')
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(8)]
        actuators = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(7)]
        if names != [*ARM_JOINTS, 'joint7', 'joint8'] or actuators != [*ARM_JOINTS, 'gripper']:
            raise ValueError('PiPER joint or actuator order changed')
        if (not np.array_equal(model.jnt_qposadr[:6], np.arange(6))
                or not np.array_equal(model.jnt_dofadr[:6], np.arange(6))
                or not np.all(model.jnt_type[:6] == mj.mjtJoint.mjJNT_HINGE)
                or not np.array_equal(model.actuator_trnid[:6, 0], np.arange(6))):
            raise ValueError('PiPER encoder axes changed')
        self._limits = np.array(ARM_LIMITS_RAD, dtype=np.float64)
        if (not np.array_equal(model.jnt_range[:6], self._limits)
                or not np.array_equal(model.actuator_ctrlrange[:6], self._limits)
                or not np.array_equal(model.actuator_ctrlrange[6], [0., .035])):
            raise ValueError('PiPER joint or servo bounds changed')
        self._tcp = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'encoder_tcp')
        self._link = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'link6')
        self._base = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'base_link')
        if (self._tcp < 0 or self._link < 0 or self._base != 1
                or model.site_bodyid[self._tcp] != self._link
                or not np.array_equal(model.site_pos[self._tcp], TCP_OFFSET_LINK6_M)
                or not np.array_equal(model.site_quat[self._tcp], [1., 0., 0., 0.])
                or model.body_parentid[self._base] != 0 or model.body_jntnum[self._base] != 0
                or not np.array_equal(model.body_pos[self._base], [0., 0., 0.])
                or not np.array_equal(model.body_quat[self._base], [1., 0., 0., 0.])):
            raise ValueError('PiPER fixed base or authored TCP changed')
        self._data = mj.MjData(model)
        # Fixed fingers cannot change the upstream link6/TCP pose or Jacobian.
        self._data.qpos[6:8] = [.035, -.035]

    def _update(self, measured_joint_position_rad):
        q = _vector(measured_joint_position_rad, 6, 'measured arm encoders')
        if np.any(q < self._limits[:, 0]) or np.any(q > self._limits[:, 1]):
            raise ValueError('Measured encoders exceed native joint limits')
        self._data.qpos[:6] = q
        self._mj.mj_kinematics(self._model, self._data)
        self._mj.mj_comPos(self._model, self._data)
        return q

    def fk(self, measured_joint_position_rad):
        """Return detached poses with explicit point, frame and wxyz convention."""
        self._update(measured_joint_position_rad)
        d = self._data
        tcp_quat, link_quat = np.empty(4), np.empty(4)
        self._mj.mju_mat2Quat(tcp_quat, d.site_xmat[self._tcp])
        base_rotation = d.xmat[self._base].reshape(3, 3)
        relative_rotation = base_rotation.T @ d.xmat[self._link].reshape(3, 3)
        self._mj.mju_mat2Quat(link_quat, relative_rotation.ravel())
        return {'tcp_position_world_m': d.site_xpos[self._tcp].tolist(),
                'tcp_quaternion_world_wxyz': tcp_quat.tolist(),
                'link6_position_base_m': (base_rotation.T @ (d.xpos[self._link] - d.xpos[self._base])).tolist(),
                'link6_quaternion_base_wxyz': link_quat.tolist()}

    def _jacobian(self):
        position, rotation = np.zeros((3, self._model.nv)), np.zeros((3, self._model.nv))
        self._mj.mj_jacSite(self._model, self._data, position, rotation, self._tcp)
        return np.vstack((position[:, :6], rotation[:, :6]))

    def jacobian(self, measured_joint_position_rad):
        """Return a detached 6x6 TCP WORLD spatial Jacobian: linear then angular."""
        self._update(measured_joint_position_rad)
        return self._jacobian()

    def pose_action(self, measured_joint_position_rad, position_world_m,
                    quaternion_world_wxyz, gripper_position_m):
        """Return one authored DLS target update; do not apply it to an actuator."""
        position = _vector(position_world_m, 3, 'TCP world position')
        if np.any(np.abs(position) > 10.):
            raise ValueError('TCP world position exceeds the numeric input envelope')
        quaternion = _vector(quaternion_world_wxyz, 4, 'TCP world quaternion')
        if np.any(np.abs(quaternion) > 1.) or not np.isclose(quaternion @ quaternion, 1., atol=1e-8, rtol=0):
            raise ValueError('TCP quaternion must be normalized wxyz')
        jaw = _vector([gripper_position_m], 1, 'single-finger target')[0]
        if not 0 <= jaw <= .035:
            raise ValueError('Single-finger target must lie in [0, .035] metres')
        measured = self._update(measured_joint_position_rad)
        current, rotation = np.empty(4), np.empty(3)
        self._mj.mju_mat2Quat(current, self._data.site_xmat[self._tcp])
        self._mj.mju_subQuat(rotation, quaternion, current)
        rotation = self._data.site_xmat[self._tcp].reshape(3, 3) @ rotation
        translation = position - self._data.site_xpos[self._tcp]
        for error, maximum in ((translation, .015), (rotation, .05)):
            norm = float(np.linalg.norm(error))
            if norm > maximum:
                error *= maximum / norm
        jacobian = self._jacobian()
        jacobian[3:] *= .25
        error = np.concatenate((translation, .25 * rotation))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .012**2 * np.eye(6), error)
        targets = np.clip(measured + np.clip(delta, -.06, .06),
                          self._limits[:, 0] + .02, self._limits[:, 1] - .02)
        if not np.isfinite(targets).all() or np.any(np.abs(targets - measured) > .06 + 1e-12):
            raise ValueError('DLS output violates the finite joint-update contract')
        return {'joint_targets_rad': targets.tolist(), 'gripper_position_m': float(jaw)}
