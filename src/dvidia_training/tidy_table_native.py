"""A single persistent MuJoCo PiPER and three free blocks on a known table.

The planner sees rendered colours projected onto a calibrated horizontal plane.
Joint feedback and an authored corner grasp drive the arm. The independent
physics score is read-only and is never used to position an object.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from . import reference_sim, robot_profiles
from .piper_encoder_kinematics import PiperEncoderKinematics
from .piper_first_attempt_policy import CornerPlacementPolicy

SLOTS = {"left": (.30, -.055), "center": (.40, -.055), "right": (.40, .055), "buffer": (.30, .055)}
COLORS = {"A": (.92, .04, .04, 1.), "B": (.04, .86, .04, 1.), "C": (.04, .04, .92, 1.)}
HOME = np.array([1.1, 1.57, -1.3485, 0., 0., 0.])
CAMERA_POSITION = (.35, 0., .85)
FOV_DEG = 29.
HEIGHT, WIDTH = 240, 320
TOP_Z = .10
CENTER_Z = .08
TOLERANCE_M = .025
DT = .001
OBSERVATION_SOURCE = "320x240 rendered top-down RGB, known primary colours and calibrated flat support"


def _slot(xy):
    nearest = min(SLOTS, key=lambda slot: math.dist(xy, SLOTS[slot]))
    return nearest if math.dist(xy, SLOTS[nearest]) <= TOLERANCE_M else None


def detect(frame):
    """Estimate centers from RGB only. Refuse small, split or heavily occluded blobs."""
    if frame.shape != (HEIGHT, WIDTH, 3) or frame.dtype != np.uint8:
        raise ValueError("Expected a 320x240 uint8 RGB image")
    rgb = frame.astype(np.int16)
    focal = HEIGHT / (2 * math.tan(math.radians(FOV_DEG) / 2))
    pixel_m = (CAMERA_POSITION[2] - TOP_Z) / focal
    expected = .04 / pixel_m
    result = {}
    for channel, obj in enumerate(COLORS):
        primary = rgb[:, :, channel]
        rival = np.maximum(rgb[:, :, (channel + 1) % 3], rgb[:, :, (channel + 2) % 3])
        rows, cols = np.nonzero((primary > 40) & (primary - rival > 35))
        if not (.63 * expected**2 <= len(rows) <= 1.42 * expected**2):
            continue
        if not (.72 * expected <= np.ptp(cols) < 1.6 * expected and
                .72 * expected <= np.ptp(rows) < 1.6 * expected):
            continue
        result[obj] = (CAMERA_POSITION[0] + (float(np.mean(cols)) - (WIDTH-1)/2) * pixel_m,
                       CAMERA_POSITION[1] - (float(np.mean(rows)) - (HEIGHT-1)/2) * pixel_m)
    return result


def _model(assets, initial, offsets_xy, mj):
    root = Path(assets)
    item, blobs = robot_profiles._payload("piper", root.parent if root.name == "agilex_piper" else root)
    spec = reference_sim._spec(blobs, item["folder"], item["model"])
    spec.option.timestep = DT
    world = spec.worldbody
    box = mj.mjtGeom.mjGEOM_BOX
    world.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[2, 2, .1],
                   rgba=[.65, .65, .65, 1])
    world.add_geom(name="table", type=box, pos=[.38, 0., .04], size=[.17, .22, .02],
                   rgba=[.65, .65, .65, 1])
    world.add_camera(name="table_camera", pos=CAMERA_POSITION, quat=[1, 0, 0, 0], fovy=FOV_DEG)
    for obj in COLORS:
        xy = np.asarray(SLOTS[initial[obj]]) + offsets_xy.get(obj, (0., 0.))
        body = world.add_body(name="block_"+obj, pos=[*xy, .082])
        body.add_freejoint(name="free_"+obj)
        body.add_geom(name="cube_"+obj, type=box, size=[.02]*3, mass=.04,
                      friction=[.8, .005, .0001], priority=1, condim=4, rgba=COLORS[obj])
    model = spec.compile()
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != (29, 26, 7, 1, 0):
        raise ValueError("Unexpected model degrees of freedom or attachment constraints")
    return model


class NativeTable:
    """Reset only on construction. Actions never write object poses after initialization."""

    def __init__(self, assets, initial, *, offsets_xy=None):
        if (set(initial) != set(COLORS) or len(set(initial.values())) != 3
                or not set(initial.values()) <= set(SLOTS)):
            raise ValueError("Place each of A, B and C in a different table slot")
        offsets_xy = {} if offsets_xy is None else offsets_xy
        if not set(offsets_xy) <= set(COLORS):
            raise ValueError("Unknown offset object")
        for xy in offsets_xy.values():
            values = np.asarray(xy)
            if (values.shape != (2,) or values.dtype.kind not in "fiu" or
                    not np.isfinite(values).all() or np.max(np.abs(values)) > .01):
                raise ValueError("Offsets must be finite XY vectors within 10 mm")
        self.mj, _ = reference_sim._native_imports()
        self.model = _model(assets, initial, offsets_xy, self.mj)
        self.kinematics = PiperEncoderKinematics(assets)
        self.data = self.mj.MjData(self.model)
        self.renderer = None
        self.data.qpos[:8] = [*HOME, .035, -.035]
        self.data.ctrl[:] = [*HOME, .035]
        self.initial = dict(initial)
        self.block_ids = {obj: self.model.body("block_"+obj).id for obj in COLORS}
        self.cube_ids = {obj: self.model.geom("cube_"+obj).id for obj in COLORS}
        self.floor = self.model.geom("floor").id
        self.table = self.model.geom("table").id
        bodies = self.model.geom_bodyid
        self.pads = {s: set(np.flatnonzero((bodies == self.model.body(b).id) &
                                  (self.model.geom_type == self.mj.mjtGeom.mjGEOM_BOX) &
                                  (self.model.geom_contype != 0))) for s,b in enumerate(("link7","link8"))}
        self.robot = set(np.flatnonzero((bodies >= 1) & (bodies <= self.model.body("link8").id)))
        self.fixed = set(np.flatnonzero(bodies == self.model.body("base_link").id))
        self.poses = {}
        self.control_steps = self.critical_contact_ticks = 0
        self.last_frame = None
        self._scan = None
        self._dwell = {obj: 0 for obj in COLORS}
        self._settled_slot = {obj: None for obj in COLORS}
        self.mj.mj_step(self.model, self.data, nstep=500)
        self.mj.mj_forward(self.model, self.data)
        self._check_valid()
        self.starts = {obj: self.data.xpos[idx].copy() for obj,idx in self.block_ids.items()}
        self.max_travel_m = dict.fromkeys(COLORS, 0.)
        # Independent support, release and motion samples throughout this initial dwell.
        for _ in range(15):
            self.mj.mj_step(self.model, self.data, nstep=20)
            self.mj.mj_forward(self.model, self.data)
            self._update_settled()
        self._check_valid()
        if any(self._dwell[obj] < 13 for obj in COLORS):
            raise RuntimeError("Native initial scene did not settle")
        from mujoco.rendering.classic.renderer import Renderer
        self.renderer = Renderer(self.model, HEIGHT, WIDTH)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _check_valid(self):
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or any(w.number for w in self.data.warning)):
            raise RuntimeError("Invalid native solver state")

    def observe(self):
        self.renderer.update_scene(self.data, camera="table_camera")
        self.last_frame = self.renderer.render()
        self.poses = detect(self.last_frame)
        self._scan = {obj: _slot(self.poses[obj]) if obj in self.poses else None for obj in COLORS}
        return self._scan.copy()

    def save_camera_frame(self, path):
        """Export the last observed RGB pixels without re-rendering the scene."""
        from .piper_action_capture import _png
        if self.last_frame is None:
            raise ValueError("Observe the table before exporting the camera image")
        with Path(path).open("xb") as stream:
            stream.write(_png(self.last_frame))

    def _contacts(self, active):
        """Force-bearing table support, pad contact and unexpected environment collisions."""
        blocks = self.cube_ids
        active_geom = blocks.get(active)
        supported, held, pad_touch = set(), set(), set()
        unsafe = False
        force = np.empty(6)
        for idx, con in enumerate(self.data.contact):
            a, b = int(con.geom1), int(con.geom2)
            self.mj.mj_contactForce(self.model, self.data, idx, force)
            if force[0] <= .01:
                continue
            pair = (a,b)
            if self.table in pair:
                for obj, geom in blocks.items():
                    if geom in pair:
                        supported.add(obj)
            if a in self.robot or b in self.robot:
                robot, other = (a,b) if a in self.robot else (b,a)
                if other in self.robot:
                    # Stock coupled gripper and neighbouring fixed/linkage contacts.
                    if robot not in self.pads[0] | self.pads[1] and other not in self.pads[0] | self.pads[1]:
                        unsafe = True
                elif other in (self.table, self.floor):
                    unsafe |= robot not in self.fixed
                elif other in blocks.values():
                    side = next((s for s,ids in self.pads.items() if robot in ids), None)
                    if side is not None:
                        pad_touch.add(other)
                    if other == active_geom and side is not None:
                        held.add(side)
                    else:
                        unsafe = True
            elif (a in blocks.values() and (b in blocks.values() or b == self.floor)
                  or b in blocks.values() and a == self.floor):
                unsafe = True
        return supported, held, pad_touch, unsafe

    def _update_settled(self):
        supported, _, touched, _ = self._contacts(None)
        for obj, body in self.block_ids.items():
            center = self.data.xpos[body]
            slot = _slot(center[:2])
            velocity = np.empty(6)
            self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
            good = (slot is not None and abs(center[2]-CENTER_Z) < .003 and obj in supported
                    and self.cube_ids[obj] not in touched and np.linalg.norm(velocity[:3]) < .3
                    and np.linalg.norm(velocity[3:]) < .035)
            self._dwell[obj] = (self._dwell[obj] + 1 if slot == self._settled_slot[obj] else 1) if good else 0
            self._settled_slot[obj] = slot if good else None
            self.max_travel_m[obj] = max(self.max_travel_m[obj], float(np.linalg.norm(center-self.starts[obj])))

    def score(self, allowed_zones):
        if (set(allowed_zones) != set(COLORS) or any(not zones or not set(zones) <= set(SLOTS)
                for zones in allowed_zones.values())):
            raise ValueError("A permitted set of slots is required for every block")
        self._check_valid()
        assignments = {obj: self._settled_slot[obj] if self._dwell[obj] >= 13 else None for obj in COLORS}
        correct = {obj: assignments[obj] in zones for obj,zones in allowed_zones.items()}
        return dict(success=all(correct.values()) and not self.critical_contact_ticks,
                    assignments=assignments, satisfied=correct, settled_samples=dict(self._dwell),
                    initially_correct_max_displacement_m={obj: dist for obj,dist in self.max_travel_m.items()
                        if self.initial[obj] in allowed_zones[obj]},
                    critical_contact_native_ticks=self.critical_contact_ticks,
                    issued_control_steps=self.control_steps, physical_robot_ready=False)

    def move(self, object_id, destination):
        start = time.perf_counter()
        obj = object_id
        if obj not in COLORS or destination not in SLOTS:
            raise ValueError("Unknown object or destination")
        if self._scan is None:
            self.observe()
        if len(set(self._scan.values())) != 3 or any(s is None for s in self._scan.values()):
            raise ValueError("A fresh complete table observation is required for a move")
        if destination in self._scan.values():
            raise ValueError("The destination must be visibly vacant")
        anchor = [*self.poses[obj], CENTER_Z]
        self._scan = None  # A subsequent action requires a new physical camera image.
        policy = CornerPlacementPolicy(self.kinematics, check_grip=True,
                                      grasp_pitch_deg=174. if anchor[0] > .35 else 180.)
        target = [*SLOTS[destination], CENTER_Z]
        initial_q = self.data.qpos[:6].copy()
        first_motion = first_motion_sim = None
        peak_lift = 0.
        lifted_ticks = longest_lift = 0
        initial_critical = self.critical_contact_ticks
        begin_steps = self.control_steps
        reason = "timeout"
        phases = []
        parked = 0
        active = self.block_ids[obj]
        for step in range(1100):
            q = self.data.qpos[:6].copy()
            elapsed = step * .02
            if policy.phase == "verify":
                phase = "park"
                action = dict(joint_targets_rad=HOME, gripper_position_m=.035)
                parked = parked + 1 if np.max(np.abs(HOME-q)) < .015 else 0
            else:
                pose = self.kinematics.fk(q)
                obs = dict(t_seconds=elapsed, joint_position_rad=q,
                    tcp_position_world_m=pose["tcp_position_world_m"],
                    tcp_quaternion_wxyz=pose["tcp_quaternion_world_wxyz"],
                    gripper_joint_position_m=self.data.qpos[6:8].copy(),
                    initial_object_position_world_m=anchor, goal_position_world_m=target)
                action = policy(obs)
                phase = policy.issued_phase
            if not phases or phases[-1]["phase"] != phase:
                phases.append(dict(phase=phase,sim_s=elapsed))
            desired = np.array([*action["joint_targets_rad"], action["gripper_position_m"]])
            # Same 1.5 rad/s and .05 m/s servo target ramps as the reference task.
            rate = np.array([.0015]*6 + [.00005])
            self.control_steps += 1
            for tick in range(20):
                self.data.ctrl[:] += np.clip(desired-self.data.ctrl, -rate, rate)
                self.mj.mj_step(self.model, self.data)
                if first_motion is None and np.max(np.abs(self.data.qpos[:6]-initial_q)) >= .001:
                    first_motion = time.perf_counter()
                    first_motion_sim = elapsed + (tick+1)*DT
                # mj_step supplies contacts/derived coordinates at the pre-integration state.
                supported, held, _, critical = self._contacts(obj)
                self.critical_contact_ticks += int(critical)
                lift = float(self.data.xpos[active,2]-CENTER_Z)
                peak_lift = max(peak_lift,lift)
                lifted_ticks = lifted_ticks + 1 if lift >= .06 and len(held) == 2 and obj not in supported else 0
                longest_lift = max(longest_lift,lifted_ticks)
            self.mj.mj_forward(self.model, self.data)
            self._check_valid()
            self._update_settled()
            if self.critical_contact_ticks > initial_critical:
                reason = "unexpected_contact"
                break
            if policy.refusal_reason:
                reason = policy.refusal_reason
                break
            if parked >= 15:
                reason = "parked"
                break
        settled = self._settled_slot[obj] == destination and self._dwell[obj] >= 13
        verified = bool(settled and reason == "parked" and longest_lift >= 100)
        if reason == "parked" and not verified:
            reason = "bilateral_lift_unverified" if longest_lift < 100 else "destination_not_settled"
        return dict(object_id=obj, destination=destination, verified=verified, reason=reason,
            first_motion_ms=None if first_motion is None else (first_motion-start)*1000,
            first_motion_monotonic_s=first_motion, first_motion_sim_s=first_motion_sim,
            phases=phases, peak_lift_m=peak_lift, bilateral_lift_dwell_s=longest_lift*DT,
            critical_contact_native_ticks=self.critical_contact_ticks-initial_critical,
            settled_at_destination=settled, control_steps=self.control_steps-begin_steps,
            wall_ms=(time.perf_counter()-start)*1000,
            final_positions_m={o:self.data.xpos[i].tolist() for o,i in self.block_ids.items()})



def main():
    """Export the unjittered development swap; independent of confirmation fixtures."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Export the PiPER development table camera image")
    parser.add_argument("--assets", required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    with NativeTable(args.assets, {"A": "center", "B": "left", "C": "right"}) as table:
        observation = table.observe()
        table.save_camera_frame(args.png)
        print(json.dumps({"observation": observation, "source": OBSERVATION_SOURCE,
                          "fixture": "development_zero_offset_swap", "png": str(args.png)}))


if __name__ == "__main__":
    main()
