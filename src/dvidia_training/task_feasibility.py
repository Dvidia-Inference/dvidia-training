"""Data-only gate for cautious, environment-conditioned manipulation planning.

``per_finger_normal_force_budget_n`` is a calibrated allowable normal-force
ceiling per finger, accounting for both actuator limits and object damage; it
is not a force command. Probe and test-lift values are caller declarations:
there is no live probe, test-lift, or controller integration here. Mass is
never inferred from images, and this module does not qualify physical skills.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math


SCHEMA = "dvidia.task-feasibility-input/v1"
RESULT_SCHEMA = "dvidia.task-feasibility-result/v1"
MAX_INPUT_BYTES = 64 * 1024
GRAVITY_M_S2 = 9.80665
STATUSES = ("blocked", "probe_required", "test_lift_required", "ready_for_simulated_transport")
_FIELDS = {
    "schema", "geometry_known", "reachability", "object_width_interval_m",
    "gripper_max_opening_interval_m", "gripper_min_closure_m",
    "mass_interval_kg", "friction_interval",
    "per_finger_normal_force_budget_n", "pose_specific_payload_limit_kg",
    "planned_upward_accel_m_s2", "safety_factor", "pose_uncertainty_m",
    "allowed_pose_tolerance_m", "observations_fresh", "test_lift_passed",
    "slip_detected", "calibration_id",
}


class TaskFeasibilityError(ValueError):
    """Input is malformed or outside the supported conservative gate."""


def _finite(value, name: str, low: float, high: float) -> float:
    if type(value) not in (int, float):
        raise TaskFeasibilityError(f"{name} must be a finite number in [{low}, {high}].")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise TaskFeasibilityError(f"{name} must be a finite number in [{low}, {high}].") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise TaskFeasibilityError(f"{name} must be a finite number in [{low}, {high}].")
    return number


def _interval(value, name: str, *, nullable: bool, low: float, high: float):
    if value is None and nullable:
        return None
    if type(value) not in (list, tuple) or len(value) != 2:
        raise TaskFeasibilityError(f"{name} must be a two-value interval" + (" or null." if nullable else "."))
    lower = _finite(value[0], name+" lower", low, high)
    upper = _finite(value[1], name+" upper", low, high)
    if lower > upper:
        raise TaskFeasibilityError(f"{name} lower bound must not exceed upper bound.")
    return (lower, upper)


def _optional_number(value, name: str, *, low: float, high: float):
    return None if value is None else _finite(value, name, low, high)


def _validate(document: Mapping) -> dict:
    if not isinstance(document, Mapping) or type(document) is not dict:
        raise TaskFeasibilityError("Input must be a plain object.")
    if set(document) != _FIELDS:
        missing, extra = sorted(_FIELDS-set(document)), sorted(set(document)-_FIELDS)
        raise TaskFeasibilityError(f"Input fields differ from schema (missing={missing}, unknown={extra}).")
    if document["schema"] != SCHEMA:
        raise TaskFeasibilityError("Unsupported task-feasibility schema.")
    if type(document["geometry_known"]) is not bool:
        raise TaskFeasibilityError("geometry_known must be boolean.")
    if document["reachability"] not in ("reachable", "unreachable", "unknown"):
        raise TaskFeasibilityError("reachability must be reachable, unreachable, or unknown.")
    for name in ("observations_fresh", "test_lift_passed", "slip_detected"):
        if type(document[name]) is not bool:
            raise TaskFeasibilityError(name+" must be boolean.")
    calibration = document["calibration_id"]
    if calibration is not None and (type(calibration) is not str or not calibration.strip() or len(calibration) > 160):
        raise TaskFeasibilityError("calibration_id must be a non-empty bounded string or null.")

    result = dict(document)
    result["object_width_interval_m"] = _interval(document["object_width_interval_m"], "object_width_interval_m", nullable=False, low=1e-9, high=10.)
    result["gripper_max_opening_interval_m"] = _interval(document["gripper_max_opening_interval_m"], "gripper_max_opening_interval_m", nullable=False, low=0., high=10.)
    result["gripper_min_closure_m"] = _finite(document["gripper_min_closure_m"], "gripper_min_closure_m", 0., 10.)
    result["mass_interval_kg"] = _interval(document["mass_interval_kg"], "mass_interval_kg", nullable=True, low=1e-9, high=1_000.)
    result["friction_interval"] = _interval(document["friction_interval"], "friction_interval", nullable=True, low=0., high=10.)
    result["per_finger_normal_force_budget_n"] = _optional_number(document["per_finger_normal_force_budget_n"], "per_finger_normal_force_budget_n", low=0., high=1_000_000.)
    result["pose_specific_payload_limit_kg"] = _optional_number(document["pose_specific_payload_limit_kg"], "pose_specific_payload_limit_kg", low=0., high=1_000.)
    result["planned_upward_accel_m_s2"] = _finite(document["planned_upward_accel_m_s2"], "planned_upward_accel_m_s2", 0., 100.)
    result["safety_factor"] = _finite(document["safety_factor"], "safety_factor", 1., 100.)
    result["pose_uncertainty_m"] = _finite(document["pose_uncertainty_m"], "pose_uncertainty_m", 0., 100.)
    result["allowed_pose_tolerance_m"] = _finite(document["allowed_pose_tolerance_m"], "allowed_pose_tolerance_m", 0., 100.)
    return result


def _decode(document: bytes | str) -> dict:
    try:
        raw = document.encode("utf-8") if type(document) is str else document
    except UnicodeError as exc:
        raise TaskFeasibilityError("Input must be valid UTF-8 JSON.") from exc
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise TaskFeasibilityError("JSON input is empty or exceeds 64 KiB.")

    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise TaskFeasibilityError("JSON object contains a duplicate field.")
            value[key] = item
        return value

    def constant(token):
        raise TaskFeasibilityError("JSON numbers must be finite; found " + token + ".")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
        json.dumps(value, allow_nan=False)
        return value
    except TaskFeasibilityError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise TaskFeasibilityError("Input must be finite UTF-8 JSON with unique keys.") from exc


def assess(document: dict | bytes | str) -> dict:
    """Return a conservative gate result; never performs sensing or actuation."""
    if type(document) in (str, bytes):
        document = _decode(document)
    data = _validate(document)
    reasons: list[str] = []
    required_n = friction_n = payload_limit_n = None
    mass = data["mass_interval_kg"]
    if mass is not None:
        required_n = mass[1] * (GRAVITY_M_S2 + data["planned_upward_accel_m_s2"]) * data["safety_factor"]
    friction = data["friction_interval"]
    normal = data["per_finger_normal_force_budget_n"]
    if friction is not None and normal is not None:
        friction_n = 2. * friction[0] * normal
    payload_limit = data["pose_specific_payload_limit_kg"]
    if payload_limit is not None:
        payload_limit_n = payload_limit * GRAVITY_M_S2

    blocked = False
    if data["reachability"] == "unreachable":
        blocked = True
        reasons.append("The requested pose is declared unreachable.")
    if not data["observations_fresh"]:
        blocked = True
        reasons.append("Observations are stale; refresh the scene and robot state before reassessment.")
    if data["slip_detected"]:
        blocked = True
        reasons.append("Slip was detected; stop transport planning and reassess the grasp.")
    object_width = data["object_width_interval_m"]
    grip_width = data["gripper_max_opening_interval_m"]
    min_closure = data["gripper_min_closure_m"]
    if object_width[0] > grip_width[1]:
        blocked = True
        reasons.append("The narrowest declared object width exceeds the widest declared maximum gripper opening.")
    if object_width[1] < min_closure:
        blocked = True
        reasons.append("The widest declared object width is smaller than the gripper's minimum closure gap.")
    if required_n is not None and payload_limit_n is not None and required_n > payload_limit_n:
        blocked = True
        reasons.append("Conservative dynamic load exceeds the calibrated pose-specific payload limit.")
    if required_n is not None and friction_n is not None and required_n > friction_n:
        blocked = True
        reasons.append("Conservative dynamic load exceeds the two-finger friction capacity.")

    unknowns = []
    if not data["geometry_known"]:
        unknowns.append("object/gripper geometry")
    if data["reachability"] == "unknown":
        unknowns.append("reachability")
    if mass is None:
        unknowns.append("object mass interval")
    if friction is None:
        unknowns.append("lower friction bound")
    if normal is None:
        unknowns.append("per-finger normal-force budget")
    if payload_limit is None:
        unknowns.append("pose-specific calibrated payload limit")
    if data["calibration_id"] is None:
        unknowns.append("calibration identity")
    if object_width[1] > grip_width[0]:
        unknowns.append("guaranteed fit within the uncertain maximum gripper opening")
    if object_width[0] < min_closure:
        unknowns.append("guaranteed object fit above the gripper minimum closure gap")
    if data["pose_uncertainty_m"] > data["allowed_pose_tolerance_m"]:
        unknowns.append("pose estimate within allowed tolerance")
    if unknowns:
        reasons.append("Measure or resolve before proceeding: " + ", ".join(unknowns) + ".")

    if blocked:
        status = "blocked"
    elif unknowns:
        status = "probe_required"
    elif not data["test_lift_passed"]:
        status = "test_lift_required"
        reasons.append("Run and record a cautious test lift under the same calibrated pose and load conditions before simulated transport planning.")
    else:
        status = "ready_for_simulated_transport"
        reasons.append("Declared checks pass for simulated transport planning; this is not hardware qualification.")
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "physical_robot_ready": False,
        "observation_origin": "caller_declared_not_verified",
        "reasons": reasons,
        "computed": {
            "conservative_required_load_n": required_n,
            "two_finger_friction_capacity_n": friction_n,
            "pose_specific_payload_limit_force_n": payload_limit_n,
            "gravity_m_s2": GRAVITY_M_S2,
        },
        "scope": ("Planning assessment only. Parallel two-finger vertical-lift approximation; ignores moment, center of mass, "
                  "deformation, transient impacts and unmodeled contact effects. Unknown values remain unknown; mass is never "
                  "inferred from images. Probe/test-lift/controller integration is not implemented. No hardware actuation "
                  "is performed; this is not hardware qualification."),
    }


def template() -> dict:
    """Return a synthetic example with declared values; it still requires a test lift."""
    return {
        "schema": SCHEMA,
        "geometry_known": True,
        "reachability": "reachable",
        "object_width_interval_m": [0.040, 0.044],
        "gripper_max_opening_interval_m": [0.050, 0.060],
        "gripper_min_closure_m": 0.010,
        "mass_interval_kg": [0.180, 0.200],
        "friction_interval": [0.60, 0.70],
        "per_finger_normal_force_budget_n": 4.0,
        "pose_specific_payload_limit_kg": 0.35,
        "planned_upward_accel_m_s2": 0.50,
        "safety_factor": 1.5,
        "pose_uncertainty_m": 0.003,
        "allowed_pose_tolerance_m": 0.010,
        "observations_fresh": True,
        "test_lift_passed": False,
        "slip_detected": False,
        "calibration_id": "synthetic-calibration-example",
    }
