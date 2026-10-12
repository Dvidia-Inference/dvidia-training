"""Experimental fixed alternate-grasp planner; native physics stays unchanged.

This supplies pose targets and bounded differential IK only. It is an authored
development candidate, not a learned grasp selector or a deployable controller.
The world-yaw candidate is fixed before a rollout and never chosen using hidden
mass, friction, scoring, or future observations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from . import reference_sim as sim
from . import piper_reacquisition as reacquisition
from .reference_task import _number, _vector


SCHEMA = 'dvidia.piper-grasp-reacquisition/v0.5'


@dataclass(frozen=True)
class GraspStrategyConfig:
    yaw_degrees: float = -45.
    orientation_error_limit_rad: float = .10
    retain_carry_yaw: bool = False

    def __post_init__(self):
        for name, low, high in (('yaw_degrees', -45., 45.), ('orientation_error_limit_rad', .05, .20)):
            object.__setattr__(self, name, _number(getattr(self, name), name, low, high))
        if type(self.retain_carry_yaw) is not bool:
            raise ValueError('retain_carry_yaw must be a boolean')


def world_yaw(quaternion_wxyz, degrees):
    """Compose a bounded world-z rotation with a normalized wxyz quaternion."""
    w, x, y, z = _vector(quaternion_wxyz, 4, 'tool quaternion', -1, 1)
    if not math.isclose(w*w+x*x+y*y+z*z, 1., abs_tol=1e-8, rel_tol=0):
        raise ValueError('Tool quaternion must be normalized wxyz')
    angle = math.radians(_number(degrees, 'yaw degrees', -45, 45))/2
    c, s = math.cos(angle), math.sin(angle)
    return (c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w)


def grasp_strategy_contract(config=None):
    cfg = GraspStrategyConfig() if config is None else config
    if not isinstance(cfg, GraspStrategyConfig):
        raise TypeError('config must be GraspStrategyConfig')
    return {'schema': 'dvidia.piper-grasp-strategy/development', 'parameters': asdict(cfg),
            'candidate': 'fixed world-z yaw of the downward grasp orientation; original approach orientation; optionally retain that yaw through postapproach carry poses',
            'planner': 'bounded differential IK; translation error <=15 mm, orientation error <= configured limit only when requested quaternion matches the downward grasp target (absolute quaternion dot >=1-1e-12), original 0.05 rad for all other targets; joint delta <=0.06 rad, damping 0.012, original position/jaw target rates and native control limits',
            'unchanged': 'native gains, force limits, geometry, friction, integration, task scoring and 20-second budget',
            'scope': 'authored fixed development candidate, privileged native state, no material/mass/friction/outcome-based grasp selection and no hardware validation'}


class GraspPlanningEnvironment:
    """Read-through policy view with alternate pose targets and bounded IK.

    Pass this view to the policy, and step the original environment. The native
    environment and its physical configuration are never replaced or modified.
    """
    def __init__(self, environment, config=None):
        self.environment = environment
        self.strategy = GraspStrategyConfig() if config is None else config
        if not isinstance(self.strategy, GraspStrategyConfig):
            raise TypeError('config must be GraspStrategyConfig')
        self.motion_phase = 'approach'
        self.grasp_config = replace(environment.config, tool_orientation_wxyz=world_yaw(
            environment.config.tool_orientation_wxyz, self.strategy.yaw_degrees))
        self.carry_config = replace(self.grasp_config, carry_orientation_wxyz=world_yaw(
            environment.config.carry_orientation_wxyz, self.strategy.yaw_degrees))

    @property
    def config(self):
        return self.carry_config if self.strategy.retain_carry_yaw and self.motion_phase != 'approach' else self.grasp_config

    def __getattr__(self, name):
        return getattr(self.environment, name)

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
        grasp_target = abs(float(quaternion@np.asarray(self.config.tool_orientation_wxyz))) >= 1.-1e-12
        orientation_limit = self.strategy.orientation_error_limit_rad if grasp_target else .05
        for error, maximum in ((translation, .015), (rotation, orientation_limit)):
            norm = float(np.linalg.norm(error))
            if norm > maximum:
                error *= maximum/norm
        jacobian = np.vstack((jacp[:, :6], .25*jacr[:, :6]))
        error = np.concatenate((translation, .25*rotation))
        delta = jacobian.T@np.linalg.solve(jacobian@jacobian.T+.012**2*np.eye(6), error)
        targets = np.clip(self.data.qpos[:6]+np.clip(delta, -.06, .06),
                          self.model.actuator_ctrlrange[:6, 0]+.02, self.model.actuator_ctrlrange[:6, 1]-.02)
        return {'joint_targets_rad': targets.tolist(), 'gripper_position_m': jaw}


def combined_contract(config=None):
    return {'schema': SCHEMA, 'reacquisition': reacquisition.reacquisition_contract(),
            'grasp_strategy': grasp_strategy_contract(config),
            'selection': 'fixed before the episode; no hidden mass, friction, outcome, or future observation inputs',
            'scope': 'experimental authored simulator candidate; alternate grasp acquisition alone is not completed placement'}


def inactive_summary(mode):
    result = reacquisition.inactive_summary(mode)
    result.update(schema=SCHEMA, mode=mode, contract=combined_contract())
    return result


class GraspReacquisitionPiperTaskPolicy(reacquisition.ReacquisitionPiperTaskPolicy):
    def __init__(self, environment, mode='reacquire_grasp', strategy=None):
        if mode != 'reacquire_grasp':
            raise ValueError('Grasp reacquisition mode must be reacquire_grasp')
        self.grasp_strategy = GraspStrategyConfig() if strategy is None else strategy
        super().__init__(GraspPlanningEnvironment(environment, self.grasp_strategy), 'reacquire')

    def summary(self):
        result = super().summary()
        result.update(schema=SCHEMA, mode='reacquire_grasp', contract=combined_contract(self.grasp_strategy))
        return result

    def __call__(self, observation):
        self.env.motion_phase = self.teacher.phase
        return super().__call__(observation)
