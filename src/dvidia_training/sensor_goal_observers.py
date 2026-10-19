"""Small independent RGB and drive-channel observers for synthetic PiPER scenes.

The RGB path receives pixels and camera calibration only. Green color and an
elongated shape identify a *candidate*, not an authenticated object. Its angle
is axial (modulo pi); a reversed shoe, lookalike, elevation and unseen motion
can remain indistinguishable. Neither observer establishes support, all robot
contact, instantaneous rest or continuous physical completion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

RGB_SCHEMA = "dvidia.rgb-proxy-observation/v1"
ACTUATOR_SCHEMA = "dvidia.piper-drive-observation/v1"
ACTUATOR_UNITS = {
    "arm_position": "rad", "arm_velocity": "rad/s", "arm_command": "rad",
    "arm_effort": "N*m", "finger_position": "m", "finger_velocity": "m/s",
    "gripper_command": "m", "gripper_effort": "N", "time": "s",
}
RGB_SCOPE = ("Green elongated proxy candidate from RGB only; not authenticated identity. "
             "Axial yaw is modulo pi. Position bounds are declared model/calibration "
             "assumptions, not a proven bound under arbitrary lighting or occlusion. "
             "No depth, support, contact or instantaneous velocity evidence.")
DRIVE_SCOPE = ("Simulated actuator output is drive effort, not contact force. "
               "Open low-effort fingers can corroborate jaw release but cannot "
               "prove table support or absence of contact with every robot link.")


def _number(value, low=-1e9, high=1e9):
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


@dataclass(frozen=True)
class CameraCalibration:
    """Trusted fixed projection and declared estimation-error assumptions.

    position_bound_m bounds the proxy origin relative to the visible color
    centroid, as a Euclidean XY radius. pixel_error_bound_px is also a radial
    centroid error, not an independent per-axis bound. Neither bound has been
    calibrated on physical camera data; callers must not reuse these values as
    real-hardware calibration. The calibration ID is a label, not authentication.
    """
    calibration_id: str = "dvidia-overhead-ortho-v04"
    # The frozen v0.3 primary contract uses this exact coordinate-frame ID.
    frame_id: str = "workstation"
    width: int = 640
    height: int = 480
    center_xy_m: tuple = (.44, -.12)
    vertical_span_m: float = .6
    position_bound_m: float = .04
    yaw_bound_rad: float = .12
    pixel_error_bound_px: float = 1.5

    def __post_init__(self):
        if type(self.calibration_id) is not str or not self.calibration_id or len(self.calibration_id) > 100:
            raise ValueError("Expected a bounded calibration ID")
        if type(self.frame_id) is not str or not self.frame_id or len(self.frame_id) > 100:
            raise ValueError("Expected a bounded calibration frame ID")
        if any(type(x) is not int or not 64 <= x <= 2048 for x in (self.width, self.height)):
            raise ValueError("Camera dimensions must be integers from 64 to 2048")
        if type(self.center_xy_m) not in (tuple, list) or len(self.center_xy_m) != 2 or not all(_number(x, -2, 2) for x in self.center_xy_m):
            raise ValueError("Expected a finite two-coordinate camera center")
        object.__setattr__(self, "center_xy_m", tuple(self.center_xy_m))
        for value, low, high in ((self.vertical_span_m, .1, 3.),
                                (self.position_bound_m, .001, .2),
                                (self.yaw_bound_rad, .001, math.pi/2),
                                (self.pixel_error_bound_px, 0., 20.)):
            if not _number(value, low, high):
                raise ValueError("Camera calibration field lies outside declared bounds")


def _components(mask):
    """Eight-connected color regions; no depth, native IDs or model geometry."""
    height, width = mask.shape
    remaining = set(map(int, np.flatnonzero(mask)))
    result = []
    while remaining:
        first = remaining.pop(); stack = [first]; region = []
        while stack:
            index = stack.pop(); region.append(index)
            y, x = divmod(index, width)
            for yy in range(max(0, y-1), min(height, y+2)):
                for xx in range(max(0, x-1), min(width, x+2)):
                    other = yy*width+xx
                    if other in remaining:
                        remaining.remove(other); stack.append(other)
        if len(region) >= 60:
            result.append(np.asarray(region, dtype=int))
    return result


def observe_rgb(rgb, *, calibration=CameraCalibration(), t_s):
    """Estimate a green proxy's XY centroid and axial yaw from a single RGB image.

    The fixed camera has world +X right, +Y up and looks down world -Z.
    Its orthographic pixel centers map directly to XY independent of depth.
    This depth invariance is also why elevated objects cannot be ruled out.
    """
    if not isinstance(calibration, CameraCalibration):
        raise ValueError("Expected trusted CameraCalibration")
    c = calibration
    out = {"schema": RGB_SCHEMA, "status": "unknown", "t_s": t_s if _number(t_s, 0, 1e9) else None,
           "calibration_id": c.calibration_id, "frame_id": c.frame_id,
           "image_dimensions_px": [c.width, c.height], "position_xy_m": None,
           "position_bound_m": c.position_bound_m + c.pixel_error_bound_px*c.vertical_span_m/c.height,
           "yaw_axis_rad": None, "yaw_bound_rad": c.yaw_bound_rad,
           "visible_green_pixels": 0, "candidate_count": 0,
           "reason": None, "scope": RGB_SCOPE}
    if not _number(t_s, 0, 1e9):
        out["reason"] = "invalid_timestamp"; return out
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (c.height, c.width, 3):
        out["reason"] = "invalid_rgb_shape_or_type"; return out
    values = rgb.astype(np.float32)
    red, green, blue = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    mask = (green > 45) & (green > 1.6*red) & (green > 1.3*blue)
    out["visible_green_pixels"] = int(mask.sum())
    scale = c.vertical_span_m/c.height
    candidates = []
    for region in _components(mask):
        ys, xs = np.divmod(region, c.width)
        if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= c.width-2 or ys.max() >= c.height-2:
            continue
        points = np.column_stack(((xs+.5-c.width/2)*scale+c.center_xy_m[0],
                                  (c.height/2-ys-.5)*scale+c.center_xy_m[1]))
        center = points.mean(axis=0)
        covariance = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, 1]
        projected = (points-center) @ eigenvectors
        extents = np.quantile(projected, .99, axis=0)-np.quantile(projected, .01, axis=0)
        width, length = map(float, extents)
        # Declared shape class of the narrow rigid proxy, not a universal shoe detector.
        if not (.14 <= length <= .27 and .025 <= width <= .060 and 3 <= length/width <= 11):
            continue
        angle = (math.atan2(float(axis[1]), float(axis[0]))+math.pi/2) % math.pi-math.pi/2
        candidates.append((center, angle, int(len(region)), [width, length]))
    out["candidate_count"] = len(candidates)
    if not candidates:
        out["reason"] = "no_unambiguous_visible_proxy"; return out
    if len(candidates) != 1:
        out["reason"] = "multiple_proxy_candidates"; return out
    center, angle, pixels, extents = candidates[0]
    out.update(status="observed", position_xy_m=center.tolist(), yaw_axis_rad=angle,
               candidate_pixels=pixels, candidate_extent_m=extents, reason="single_green_shape_candidate")
    return out


def observe_rgb_sequence(frames, *, calibration=CameraCalibration()):
    """Report observed motion over a time interval, never instantaneous rest.

    Bounds are conservative endpoint position-error propagation. Net-average
    speed can be zero during a return trajectory; per-edge displacement is
    exposed so callers do not mistake that for absence of observed motion.
    """
    out = {"schema": "dvidia.rgb-proxy-sequence/v1", "status": "unknown", "reason": None,
           "observations": [], "start_s": None, "end_s": None, "span_s": None,
           "net_displacement_m": None, "net_displacement_interval_m": None,
           "net_average_speed_interval_m_s": None, "edges": [],
           "scope": "Sampled image displacement only; cannot prove instantaneous velocity, motion between frames, contact, or continuous dwell."}
    if type(frames) is not list or not 2 <= len(frames) <= 64:
        out["reason"] = "insufficient_or_excessive_frames"; return out
    if any(type(f) is not dict or set(f) != {"t_s", "rgb"} or not _number(f["t_s"], 0, 1e9) for f in frames):
        out["reason"] = "invalid_frame_fields"; return out
    times = [f["t_s"] for f in frames]
    intervals = [b-a for a, b in zip(times, times[1:])]
    if any(not .001 <= dt <= .12+1e-9 for dt in intervals):
        out["reason"] = "nonmonotonic_or_gapped_frames"; return out
    observations = [observe_rgb(f["rgb"], calibration=calibration, t_s=f["t_s"]) for f in frames]
    out.update(observations=observations, start_s=times[0], end_s=times[-1], span_s=times[-1]-times[0],
               sample_intervals_s=intervals)
    if any(o["status"] != "observed" for o in observations):
        out["reason"] = "unobservable_frame"; return out
    def displacement(a, b):
        distance = math.dist(a["position_xy_m"], b["position_xy_m"])
        error = a["position_bound_m"]+b["position_bound_m"]
        return distance, [max(0., distance-error), distance+error]
    for a, b, dt in zip(observations, observations[1:], intervals):
        distance, bounds = displacement(a, b)
        out["edges"].append({"start_s": a["t_s"], "end_s": b["t_s"], "displacement_m": distance,
                             "displacement_interval_m": bounds,
                             "average_speed_interval_m_s": [v/dt for v in bounds]})
    distance, bounds = displacement(observations[0], observations[-1])
    out.update(status="observed", reason="sampled_motion_available", net_displacement_m=distance,
               net_displacement_interval_m=bounds,
               net_average_speed_interval_m_s=[v/out["span_s"] for v in bounds])
    return out


@dataclass(frozen=True)
class ActuatorCalibration:
    """Per-channel absolute error bounds and the declared native servo model.

    These are asserted calibration assumptions, not authenticated sensor error
    guarantees. Equalities between channels cannot detect consistent forgery.
    """
    calibration_id: str = "dvidia-piper-drive-v04"
    arm_position_bound_rad: float = .001
    arm_velocity_bound_rad_s: float = .01
    arm_effort_bound_nm: float = .02
    finger_position_bound_m: float = .0005
    finger_velocity_bound_m_s: float = .001
    gripper_command_bound_m: float = .0001
    gripper_effort_bound_n: float = .02
    gripper_kp_n_per_m: float = 40.
    gripper_kv_ns_per_m: float = 5.

    def __post_init__(self):
        if type(self.calibration_id) is not str or not self.calibration_id or len(self.calibration_id) > 100:
            raise ValueError("Expected a bounded actuator calibration ID")
        for name, value in vars(self).items():
            if name == "calibration_id": continue
            if not _number(value, 0., 1000.):
                raise ValueError("Invalid actuator calibration field: "+name)
        if not self.gripper_kp_n_per_m > 0:
            raise ValueError("Expected positive gripper stiffness")


def read_native_actuators(model, data):
    """Native acquisition adapter. Reads named joint/drive arrays only.

    Deliberately does not read object positions, contacts, constraint forces,
    segmentation, or the goal. It is not an implemented hardware driver.
    """
    import mujoco as mj
    joints = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 9)]
    actuators = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
                 for name in [*[f"joint{i}" for i in range(1, 7)], "gripper"]]
    if min(joints+actuators) < 0:
        raise ValueError("Expected named PiPER joints and drives")
    q = [float(data.qpos[model.jnt_qposadr[j]]) for j in joints]
    v = [float(data.qvel[model.jnt_dofadr[j]]) for j in joints]
    command = [float(data.ctrl[a]) for a in actuators]
    effort = [float(data.actuator_force[a]) for a in actuators]
    return {"schema": ACTUATOR_SCHEMA, "units": dict(ACTUATOR_UNITS), "t_s": float(data.time),
            "arm_position_rad": q[:6], "arm_velocity_rad_s": v[:6], "arm_command_rad": command[:6],
            "arm_effort_nm": effort[:6], "finger_position_m": q[6:], "finger_velocity_m_s": v[6:],
            "gripper_command_m": command[6], "gripper_effort_n": effort[6]}


def observe_actuators(packet, *, calibration=ActuatorCalibration()):
    """Validate independent channels and expose bounded jaw/servo features.

    PiPER joint7 is positive and joint8 negative: gap=q7-q8; the command
    controls the single positive finger, not the total aperture. Servo residual
    checks cross-channel consistency; a self-consistent forged stream survives.
    """
    if not isinstance(calibration, ActuatorCalibration):
        raise ValueError("Expected trusted ActuatorCalibration")
    out = {"schema": "dvidia.piper-drive-features/v1", "status": "unknown", "reason": None,
           "t_s": None, "calibration_id": calibration.calibration_id, "scope": DRIVE_SCOPE, "jaw_gap_m": None,
           "jaw_gap_interval_m": None, "gripper_effort_interval_n": None,
           "servo_residual_n": None, "servo_residual_bound_n": None}
    arrays = {"arm_position_rad": (6, -7, 7), "arm_velocity_rad_s": (6, -30, 30),
              "arm_command_rad": (6, -7, 7), "arm_effort_nm": (6, -1000, 1000),
              "finger_position_m": (2, -.04, .04), "finger_velocity_m_s": (2, -2, 2)}
    required = {"schema", "units", "t_s", "gripper_command_m", "gripper_effort_n", *arrays}
    if type(packet) is not dict or set(packet) != required or packet.get("schema") != ACTUATOR_SCHEMA or packet.get("units") != ACTUATOR_UNITS:
        out["reason"] = "invalid_packet_schema_or_units"; return out
    if not _number(packet["t_s"], 0, 1e9) or not _number(packet["gripper_command_m"], 0, .03500001) or not _number(packet["gripper_effort_n"], -1000, 1000):
        out["reason"] = "invalid_scalar_measurement"; return out
    for name, (count, low, high) in arrays.items():
        value = packet[name]
        if type(value) is not list or len(value) != count or any(not _number(x, low, high) for x in value):
            out["reason"] = "invalid_channel_"+name; return out
    q7, q8 = packet["finger_position_m"]; v7, v8 = packet["finger_velocity_m_s"]
    if q7 < -.001 or q8 > .001:
        out["reason"] = "finger_sign_convention_mismatch"; return out
    c = calibration
    gap = q7-q8; force = packet["gripper_effort_n"]
    if gap+2*c.finger_position_bound_m < 0:
        out["reason"] = "negative_aperture_outside_uncertainty"; return out
    expected_force = c.gripper_kp_n_per_m*(packet["gripper_command_m"]-q7)-c.gripper_kv_ns_per_m*v7
    residual = force-expected_force
    residual_bound = (c.gripper_effort_bound_n+c.gripper_kp_n_per_m*(c.gripper_command_bound_m+c.finger_position_bound_m)
                      +c.gripper_kv_ns_per_m*c.finger_velocity_bound_m_s)
    out.update(status="observed", reason="drive_channels_available", t_s=packet["t_s"],
               jaw_gap_m=gap, jaw_gap_interval_m=[max(0., gap-2*c.finger_position_bound_m), max(0., gap+2*c.finger_position_bound_m)],
               jaw_gap_rate_m_s=v7-v8,
               jaw_gap_rate_interval_m_s=[v7-v8-2*c.finger_velocity_bound_m_s, v7-v8+2*c.finger_velocity_bound_m_s],
               gripper_command_m=packet["gripper_command_m"],
               gripper_command_interval_m=[max(0., packet["gripper_command_m"]-c.gripper_command_bound_m), packet["gripper_command_m"]+c.gripper_command_bound_m],
               gripper_effort_n=force, gripper_effort_interval_n=[force-c.gripper_effort_bound_n, force+c.gripper_effort_bound_n],
               servo_residual_n=residual, servo_residual_bound_n=residual_bound,
               servo_consistency="compatible" if abs(residual) <= residual_bound else "contradiction",
               finger_symmetry_residual_m=q7+q8, finger_symmetry_bound_m=2*c.finger_position_bound_m,
               arm_position_rad=list(packet["arm_position_rad"]), arm_position_bound_rad=c.arm_position_bound_rad,
               arm_velocity_rad_s=list(packet["arm_velocity_rad_s"]), arm_velocity_bound_rad_s=c.arm_velocity_bound_rad_s,
               arm_command_rad=list(packet["arm_command_rad"]), arm_effort_nm=list(packet["arm_effort_nm"]), arm_effort_bound_nm=c.arm_effort_bound_nm)
    return out


def render_overhead_rgb(model, data, *, calibration=CameraCalibration()):
    """Native RGB-only camera adapter, separate from inference and label paths.

    This is a configured orthographic camera, not native object projection.
    MuJoCo global fovy is world vertical extent for orthographic rendering.
    Model visualization settings are restored even if rendering fails.
    https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
    https://mujoco.readthedocs.io/en/stable/programming/visualization.html
    """
    import mujoco as mj
    if not isinstance(calibration, CameraCalibration):
        raise ValueError("Expected trusted CameraCalibration")
    c = calibration; visual = model.vis.global_
    fields = ("orthographic", "fovy", "ipd", "offwidth", "offheight")
    old = {name: getattr(visual, name) for name in fields}
    renderer = None
    try:
        visual.orthographic = 1; visual.fovy = c.vertical_span_m; visual.ipd = 0
        visual.offwidth = max(visual.offwidth, c.width); visual.offheight = max(visual.offheight, c.height)
        camera = mj.MjvCamera(); camera.lookat[:] = [*c.center_xy_m, 0.]
        camera.distance = 1.; camera.azimuth = 90.; camera.elevation = -90.
        renderer = mj.Renderer(model, c.height, c.width)
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()
    finally:
        if renderer is not None: renderer.close()
        for name, value in old.items(): setattr(visual, name, value)
