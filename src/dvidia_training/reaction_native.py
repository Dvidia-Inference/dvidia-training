"""Native mechanical coupons for reaction v0.1, not a physical robot driver.

One bounded horizontal actuator carries an actuated parallel jaw. A free rigid
payload and an externally scripted mocap obstacle remain in native physics.
Only reset initializes dynamic coordinates; a brake uses a bounded hold servo.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import math

import mujoco
import numpy as np

from .env import _NATIVE_LOCK

FIXTURE_ID = "dvidia-linear-actuator-parallel-jaw-reaction-v0.1"


@dataclass(frozen=True)
class FixtureConfig:
    timestep_s: float = .001
    warmup_s: float = .4
    horizon_s: float = 1.4
    speed_m_s: float = .25
    carriage_mass_kg: float = 1.0
    actuator_force_limit_n: float = 15.
    payload_mass_kg: float = .04
    jaw_force_limit_n: float = 8.
    pair_friction: float = .7
    rest_speed_m_s: float = .005
    rest_dwell_s: float = .05
    command_delay_s: float = .02

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
            if value < 0 or (name != "command_delay_s" and value == 0):
                raise ValueError(f"{name} is outside the fixture range")
        if not .0002 <= self.timestep_s <= .002:
            raise ValueError("timestep_s must lie in [.0002, .002]")
        if not .05 <= self.warmup_s <= 1 or not .5 <= self.horizon_s <= 5:
            raise ValueError("fixture durations are bounded")
        if not .02 <= self.speed_m_s <= .6 or self.speed_m_s*self.horizon_s > .6:
            raise ValueError("commanded motion must fit the fixture travel range")
        if not .01 <= self.payload_mass_kg <= .2 or not .1 <= self.pair_friction <= 1.5:
            raise ValueError("payload/material lies outside the coupon range")
        if self.actuator_force_limit_n > 100 or self.jaw_force_limit_n > 30:
            raise ValueError("actuator limits exceed the coupon range")
        for duration in (self.warmup_s, self.horizon_s, self.rest_dwell_s, self.command_delay_s):
            if not math.isclose(duration/self.timestep_s, round(duration/self.timestep_s), abs_tol=1e-9):
                raise ValueError("durations must contain whole native ticks")


class NativeReactionFixture:
    """State truth is for the sensor producer/scorer, never the monitor itself."""

    def __init__(self, config: FixtureConfig, *, holding: bool = False):
        if mujoco.__version__ != "3.15.0":
            raise RuntimeError("Reaction coupons require MuJoCo 3.15.0")
        self.config, self.holding = config, holding
        c = config
        self.scene_xml = f'''<mujoco model="{FIXTURE_ID}">
        <compiler angle="radian"/><option timestep="{c.timestep_s}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" solver="Newton" iterations="70" tolerance="1e-10"/>
        <size memory="8M"/><default><geom solref="0.008 1" solimp="0.95 0.99 0.001" condim="3" friction="{c.pair_friction} 0 0"/></default>
        <worldbody><geom name="floor" type="plane" size="1 1 .1"/>
          <body name="carriage" pos="0 0 .40"><joint name="travel" type="slide" axis="1 0 0" range="-.1 .7" damping=".1"/>
            <geom name="probe" type="sphere" size=".035" mass="{c.carriage_mass_kg}"/>
            <body name="left" pos="0 .045 -.05"><joint name="left_jaw" type="slide" axis="0 -1 0" range="0 .04" damping="1"/>
              <geom name="left_pad" type="box" size=".022 .008 .025" mass=".05"/></body>
            <body name="right" pos="0 -.045 -.05"><joint name="right_jaw" type="slide" axis="0 1 0" range="0 .04" damping="1"/>
              <geom name="right_pad" type="box" size=".022 .008 .025" mass=".05"/></body>
          </body>
          <body name="payload" pos="0 0 {'.35' if holding else '.016'}"><freejoint name="payload_free"/><geom name="payload_geom" type="box" size=".015 .015 .015" mass="{c.payload_mass_kg}"/></body>
          <body name="obstacle" mocap="true" pos=".3 1 .4"><geom name="obstacle_geom" type="sphere" size=".035"/></body>
        </worldbody>
        <contact><exclude body1="left" body2="right"/>
          <pair geom1="left_pad" geom2="payload_geom" condim="3" friction="{c.pair_friction} {c.pair_friction} 0 0 0"/>
          <pair geom1="right_pad" geom2="payload_geom" condim="3" friction="{c.pair_friction} {c.pair_friction} 0 0 0"/>
        </contact>
        <actuator><position name="drive" joint="travel" kp="500" kv="40" ctrllimited="true" ctrlrange="-.1 .7" forcelimited="true" forcerange="-{c.actuator_force_limit_n} {c.actuator_force_limit_n}"/>
          <position name="grip_left" joint="left_jaw" kp="2000" kv="10" ctrllimited="true" ctrlrange="0 .04" forcelimited="true" forcerange="-{c.jaw_force_limit_n} {c.jaw_force_limit_n}"/>
          <position name="grip_right" joint="right_jaw" kp="2000" kv="10" ctrllimited="true" ctrlrange="0 .04" forcelimited="true" forcerange="-{c.jaw_force_limit_n} {c.jaw_force_limit_n}"/>
        </actuator></mujoco>'''
        self.warnings = []
        with self.native():
            self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
            self.data = mujoco.MjData(self.model)
            self.travel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "travel")
            self.travel_qpos = int(self.model.jnt_qposadr[self.travel])
            self.travel_dof = int(self.model.jnt_dofadr[self.travel])
            self.payload_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
            self.carriage_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "carriage")
            self.geom = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("probe", "left_pad", "right_pad", "payload_geom", "obstacle_geom", "floor")}
            for name in ("left_jaw", "right_jaw"):
                ident = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.data.qpos[self.model.jnt_qposadr[ident]] = .0225 if holding else 0
            self.data.ctrl[:] = (0, .024 if holding else 0, .024 if holding else 0)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(round(c.warmup_s/c.timestep_s)):
                self.tick(0, (.3, 1, .4))
        self.initial_payload_z = float(self.data.xpos[self.payload_body, 2])
        self.initial_payload_offset = self.data.xpos[self.payload_body].copy()-self.data.xpos[self.carriage_body]
        if holding and (self.initial_payload_z < .3 or min(self.measure()["pad_contacts"].values()) < 1):
            raise RuntimeError("payload did not establish a bilateral native grip during warmup")

    @contextmanager
    def native(self):
        with _NATIVE_LOCK:
            previous = mujoco.get_mju_user_warning()
            mujoco.set_mju_user_warning(lambda message: self.warnings.append(str(message)))
            try:
                yield
            finally:
                mujoco.set_mju_user_warning(previous)

    def tick(self, target_m, obstacle_position_m, *, carriage_push_n=0., payload_pull_n=0.):
        # A mocap obstacle is an authored external actor; dynamic robot/payload
        # qpos and qvel are never overwritten while moving or braking.
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("nonfinite native state before step")
        self.data.ctrl[0] = float(target_m)
        self.data.mocap_pos[0] = obstacle_position_m
        self.data.xfrc_applied[:] = 0
        self.data.xfrc_applied[self.carriage_body, 0] = carriage_push_n
        self.data.xfrc_applied[self.payload_body, 2] = -payload_pull_n
        before = float(self.data.time)
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if (self.warnings or any(w.number for w in self.data.warning)
                or not all(np.isfinite(value).all() for value in (self.data.qpos, self.data.qvel, self.data.qacc))
                or not math.isclose(float(self.data.time)-before, self.config.timestep_s, abs_tol=1e-10)):
            raise RuntimeError("native warning or invalid state")

    def measure(self):
        position = self.data.xpos[self.carriage_body].copy()
        velocity = float(self.data.qvel[self.travel_dof])
        payload_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.payload_body, payload_velocity, 0)
        slip = float(np.linalg.norm(payload_velocity[[3, 5]]-np.array([velocity, 0.]))) if self.holding else 0.
        unexpected_force = abs(float(self.data.xfrc_applied[self.carriage_body, 0]))
        penetration = 0.
        counts = {"left": 0, "right": 0}
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if self.geom["obstacle_geom"] in pair and pair.intersection({self.geom["probe"], self.geom["left_pad"], self.geom["right_pad"], self.geom["payload_geom"]}):
                unexpected_force += float(np.linalg.norm(force[:3]))
                penetration = max(penetration, max(0., -float(contact.dist)))
            for side in counts:
                if pair == {self.geom[side+"_pad"], self.geom["payload_geom"]} and force[0] > .01:
                    counts[side] += 1
        robot_shapes = []
        gaps = []
        for name in ("probe", "left_pad", "right_pad"):
            geom_id = self.geom[name]
            body_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[geom_id]), body_velocity, 0)
            radius = .035 if name == "probe" else float(np.linalg.norm(self.model.geom_size[geom_id]))
            robot_shapes.append({"center_m": self.data.geom_xpos[geom_id].tolist(), "velocity_m_s": body_velocity[3:].tolist(), "radius_m": radius})
            gaps.append(float(mujoco.mj_geomDistance(self.model, self.data, geom_id, self.geom["obstacle_geom"], 2., None)))
        gap = min(gaps)
        return {"position_m": position.tolist(), "velocity_m_s": [velocity, 0., 0.],
                "unexpected_force_n": unexpected_force, "slip_speed_m_s": slip,
                "pad_contacts": counts, "clearance_m": gap, "penetration_m": penetration,
                "dropped": bool(self.holding and self.data.xpos[self.payload_body, 2] < self.initial_payload_z-.04) if hasattr(self, "initial_payload_z") else False,
                "actuator_force_n": float(self.data.actuator_force[0]),
                "payload_position_m": self.data.xpos[self.payload_body].tolist(), "robot_shapes": robot_shapes}

    def identity(self):
        return {"fixture_id": FIXTURE_ID, "config": asdict(self.config),
                "scene_sha256": hashlib.sha256(self.scene_xml.encode()).hexdigest(),
                "collision_scope": "native probe/pads/payload/obstacle/floor; adjacent carriage-pad and pad-pad interactions excluded; one horizontal robot travel axis",
                "sensor_scope": "synthetic world-frame sphere tracks, native/applied external-force proxy, native payload tangential velocity; no camera/tactile hardware model",
                "brake_scope": "delayed fixed measured-position hold through bounded native servo; no dynamic qpos/qvel overwrite",
                "pair_friction": self.config.pair_friction}
