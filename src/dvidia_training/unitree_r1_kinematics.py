"""Offline kinematics of the standalone Unitree R1-7A, from base_link to bare Link7."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from xml.etree import ElementTree as ET

import numpy as np

from .unitree_r1_assets import source_info, urdf_bytes


JOINTS = tuple(f"J{i}" for i in range(1, 8))
LINKS = tuple(f"Link{i}" for i in range(1, 8))
FRAME = "base_link"
END_LINK = LINKS[-1]
POSITION_TOLERANCE_M = 1e-9
ROTATION_TOLERANCE_RAD = 1e-9


def _rotation(axis, angle):
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * (cross @ cross)


def _xyz(element, attribute):
    return np.array([float(value) for value in element.attrib[attribute].split()])


@dataclass
class _Joint:
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _chain(raw):
    robot = ET.fromstring(raw)
    result = []
    for name, parent, child in zip(JOINTS, (FRAME, *LINKS[:-1]), LINKS, strict=True):
        joint = robot.find(f"joint[@name='{name}']")
        if (joint is None or joint.attrib["type"] != "revolute"
                or joint.find("parent").attrib["link"] != parent
                or joint.find("child").attrib["link"] != child):
            raise ValueError(f"Unexpected R1-7A kinematic chain at {name}")
        origin = joint.find("origin")
        roll, pitch, yaw = _xyz(origin, "rpy")
        transform = np.eye(4)
        transform[:3, :3] = (_rotation((0, 0, 1), yaw) @ _rotation((0, 1, 0), pitch)
                             @ _rotation((1, 0, 0), roll))
        transform[:3, 3] = _xyz(origin, "xyz")
        axis = _xyz(joint.find("axis"), "xyz")
        limit = joint.find("limit")
        result.append(_Joint(transform, axis / np.linalg.norm(axis),
                             float(limit.attrib["lower"]), float(limit.attrib["upper"])))
    return result


def _checked_positions(values, limits):
    if not isinstance(values, (list, tuple, np.ndarray)) or np.shape(values) != (len(JOINTS),):
        raise ValueError("Expected seven joint positions in J1 through J7 order")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in values):
        raise ValueError("Joint positions must be real numbers in radians")
    try:
        angles = np.asarray(values, dtype=float)
    except OverflowError as exc:
        raise ValueError("Joint positions must be finite") from exc
    if not np.all(np.isfinite(angles)):
        raise ValueError("Joint positions must be finite")
    for name, angle, (lower, upper) in zip(JOINTS, angles, limits, strict=True):
        if not lower <= angle <= upper:
            raise ValueError(f"{name} exceeds its URDF range [{lower}, {upper}] rad")
    return angles


def _transforms(chain, positions):
    current = np.eye(4)
    links = []
    for joint, angle in zip(chain, positions, strict=True):
        turn = np.eye(4)
        turn[:3, :3] = _rotation(joint.axis, angle)
        current = current @ joint.origin @ turn
        links.append(current)
    return links


def pose(side, joint_positions):
    """Return the URDF Link7 frame with the fixed base as the parent frame."""
    chain = _chain(urdf_bytes(side))
    limits = [(joint.lower, joint.upper) for joint in chain]
    q = _checked_positions(joint_positions, limits)
    return {"schema": "dvidia.unitree-r1-7a-pose/v1", "side": side, "source": source_info(side),
            "reference_frame": FRAME, "end_link": END_LINK, "point": "bare Link7 frame origin",
            "position_unit": "m", "joint_names": list(JOINTS),
            "joint_positions_rad": q.tolist(), "joint_limits_rad": [list(pair) for pair in limits],
            "base_from_link7": _transforms(chain, q)[-1].tolist()}


def _cases(limits):
    low, high = limits.T
    center = (low + high) / 2
    width = high - low
    yield "zero", np.zeros(7)
    yield "center", center.copy()
    for index, name in enumerate(JOINTS):
        for label, sign in (("minus", -1), ("plus", 1)):
            angles = center.copy()
            angles[index] += sign * width[index] / 8
            yield f"{name}_{label}", angles
    yield "lower_limits", low.copy()
    yield "upper_limits", high.copy()
    offsets = np.array([1, -2, 3, -3, 2, -1, 3]) / 10
    yield "mixed_a", center + width * offsets
    yield "mixed_b", center - width * offsets


def _native_model(raw, mj):
    # Stable-base URDF import places base_link at world identity. Keep original
    # joints and frame conventions, and remove geometries before native import.
    root = ET.fromstring(raw)
    for link in root.findall("link"):
        for geometry in [*link.findall("visual"), *link.findall("collision")]:
            link.remove(geometry)
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if (model.nq, model.nv, model.nu, model.ngeom) != (7, 7, 0, 0):
        raise ValueError("Expected a mesh-free seven-joint native arm with zero actuators")
    return model


def _rotation_error(reference, measured):
    relative = reference.T @ measured
    skew = relative - relative.T
    sine = np.linalg.norm([skew[2, 1], skew[0, 2], skew[1, 0]]) / 2
    return float(np.arctan2(sine, (relative.trace() - 1) / 2))


def benchmark(side):
    """Cross-check the independent URDF transform chain with native mj_forward."""
    raw = urdf_bytes(side)
    chain = _chain(raw)
    limits = np.array([(joint.lower, joint.upper) for joint in chain])
    try:
        import mujoco as mj
    except ImportError as exc:
        raise ImportError("Install dvidia-training[arm] to run the native reference benchmark") from exc
    model = _native_model(raw, mj)
    addresses = [int(model.joint(name).qposadr[0]) for name in JOINTS]
    native_ranges = np.array([model.joint(name).range.copy() for name in JOINTS])
    if not np.allclose(native_ranges, limits, atol=1e-12, rtol=0):
        raise ValueError("URDF and native joint ranges disagree")
    data = mj.MjData(model)
    rows = []
    for name, q in _cases(limits):
        data.qpos[addresses] = q
        mj.mj_forward(model, data)
        transforms = _transforms(chain, q)
        positions, rotations = [], []
        for link, expected in zip(LINKS, transforms, strict=True):
            body = data.body(link)
            positions.append(float(np.linalg.norm(body.xpos - expected[:3, 3])))
            rotations.append(_rotation_error(expected[:3, :3], body.xmat.reshape(3, 3)))
        rows.append({"case": name, "joint_positions_rad": q.tolist(),
                     "base_from_link7": transforms[-1].tolist(),
                     "max_link_position_error_m": max(positions),
                     "max_link_orientation_error_rad": max(rotations)})
    worst_m = max(row["max_link_position_error_m"] for row in rows)
    worst_rad = max(row["max_link_orientation_error_rad"] for row in rows)
    return {"schema": "dvidia.unitree-r1-7a-fk-benchmark/v1", "side": side, "source": source_info(side),
            "reference_frame": FRAME, "end_link": END_LINK, "point": "bare Link7 frame origin",
            "scope": "Static mesh-free kinematics with an identity mounting transform. Contact, grasping, dynamics, hardware and tool calibration are untested.",
            "mujoco_version": mj.__version__, "joint_names": list(JOINTS), "joint_limits_rad": limits.tolist(),
            "link_names": list(LINKS), "native_actuators": model.nu, "native_geometries": model.ngeom,
            "native_time_s": float(data.time), "cases": rows,
            "summary": {"poses": len(rows), "link_transforms_compared": len(rows) * len(LINKS),
                        "max_link_position_error_m": worst_m, "max_link_orientation_error_rad": worst_rad,
                        "position_tolerance_m": POSITION_TOLERANCE_M, "orientation_tolerance_rad": ROTATION_TOLERANCE_RAD},
            "passed": bool(worst_m <= POSITION_TOLERANCE_M and worst_rad <= ROTATION_TOLERANCE_RAD)}
