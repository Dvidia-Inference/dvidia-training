"""Independent readback of fixed32 incremental PiPER evidence.

No simulator, controller, benchmark scorer, or robot runtime imports. This
checks the consistency of saved numerical evidence and control requests; it
does not authenticate sensors, rerun trials, or establish physical safety.
"""
from __future__ import annotations

if not __debug__:
    raise RuntimeError("Independent incremental-hold audit refuses optimized Python.")

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import zipfile

import numpy as np

MAX_FILE = 32 * 1024 * 1024
VARIANTS = ("measured_joint", "issued_joint", "cartesian", "incremental")
SCENES = tuple(f"scene-{index}" for index in range(8))
FIELDS = {
    "t_seconds": 1, "native_tick": 1,
    "object_position_m": 3, "object_quaternion_wxyz": 4,
    "object_linear_velocity_m_s": 3, "object_angular_velocity_rad_s": 3,
    "tcp_position_m": 3, "tcp_quaternion_wxyz": 4,
    "object_minus_tcp_local_m": 3, "grasp_contact_counts": 2,
    "pad_normal_forces_n": 2, "table_object_contact_count": 1,
    "unexpected_contact_count": 1, "gripper_joint_position_m": 2,
    "joint_position_rad": 6, "arm_actuator_controls_rad": 6,
    "requested_jaw_target_m": 1, "normal_rate_limited_jaw_target_m": 1,
    "applied_jaw_target_m": 1, "pulse_active": 1,
    "pulse_minimum_opening_m": 1, "pulse_closing_tail": 1,
}
ADMISSIONS = ("evidence_qualified", "timer_admitted")
BOUND_REFUSALS = frozenset({
    "initial_bias_exhausted", "native_joint_margin_exhausted",
    "measured_joint_margin_exhausted", "tcp_position_error_exhausted",
    "tcp_orientation_error_exhausted", "joint_correction_exhausted",
    "pose_solver_refused", "missing_prior_issued_target",
})
CORRECTION_LIMIT_RAD = .03
CORRECTION_STEP_RAD = .002
INITIAL_BIAS_LIMIT_RAD = .08
CONTROL_DT_S = .02
CONTROL_SUBSTEP_S = .001
TCP_POSITION_LIMIT_M = .004
TCP_ORIENTATION_LIMIT_RAD = .05


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def bad(token):
    raise ValueError("Nonfinite JSON " + token)


def identity(path):
    size, digest = 0, sha256()
    assert not path.is_symlink() and path.is_file(), f"Missing or unsafe file: {path}"
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            size += len(chunk)
            assert size <= MAX_FILE, f"File exceeds audit size limit: {path}"
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def read(path):
    identity(path)
    result = json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=bad)
    canonical(result)
    return result


def episode(path):
    identity(path)
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_FILE + 1)
    assert len(raw) <= MAX_FILE
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=bad)
    canonical(result)
    return result


def row_at(seconds, times):
    index = round(seconds / CONTROL_SUBSTEP_S) - 1
    assert 0 <= index < len(times) and abs(times[index] - seconds) < 1e-8
    return index


def _unit_quaternions(quaternions):
    norms = np.linalg.norm(quaternions, axis=1)
    assert np.max(np.abs(norms - 1.)) < 1e-8


def rotation_local(arrays):
    q = arrays["tcp_quaternion_wxyz"]
    _unit_quaternions(q)
    w, x, y, z = q.T
    rotation = np.empty((len(q), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1 - 2 * (y*y + z*z)
    rotation[:, 0, 1] = 2 * (x*y - w*z)
    rotation[:, 0, 2] = 2 * (x*z + w*y)
    rotation[:, 1, 0] = 2 * (x*y + w*z)
    rotation[:, 1, 1] = 1 - 2 * (x*x + z*z)
    rotation[:, 1, 2] = 2 * (y*z - w*x)
    rotation[:, 2, 0] = 2 * (x*z - w*y)
    rotation[:, 2, 1] = 2 * (y*z + w*x)
    rotation[:, 2, 2] = 1 - 2 * (x*x + y*y)
    vector = arrays["object_position_m"] - arrays["tcp_position_m"]
    local = np.einsum("nji,nj->ni", rotation, vector)
    assert np.max(np.abs(local - arrays["object_minus_tcp_local_m"])) < 1e-9
    return arrays["object_minus_tcp_local_m"]


def native_predicate(arrays, probe):
    """Recompute v0.2's acquisition qualification predicate at each 1 ms row."""
    times = arrays["t_seconds"]
    origin = row_at(probe["start_time_s"], times)
    assert np.linalg.norm(arrays["object_position_m"][origin] - probe["object_origin_m"]) < 1e-9
    assert np.linalg.norm(arrays["object_minus_tcp_local_m"][origin] - probe["local_offset_origin_m"]) < 1e-9
    rise = arrays["object_position_m"][:, 2] - probe["object_origin_m"][2]
    drift = np.linalg.norm(arrays["object_minus_tcp_local_m"] - probe["local_offset_origin_m"], axis=1)
    components = {
        "bilateral_contact": np.all(arrays["grasp_contact_counts"] > 0, axis=1),
        "bilateral_force": np.all(arrays["pad_normal_forces_n"] > .01, axis=1),
        "force_ceiling": np.all(arrays["pad_normal_forces_n"] <= 10., axis=1),
        "unsupported": arrays["table_object_contact_count"] == 0,
        "no_unexpected_contact": arrays["unexpected_contact_count"] == 0,
        "minimum_rise": rise + 1e-12 >= .006,
        "maximum_local_drift": drift <= .003 + 1e-12,
    }
    condition = np.arange(len(times)) > origin
    for component in components.values():
        condition &= component
    streak = np.empty(len(times), dtype=np.int64)
    previous = 0
    for index, good in enumerate(condition):
        previous = previous + 1 if good else 0
        streak[index] = previous
    return condition, streak, streak >= 100, components, rise, drift


def reasons_at(index, components):
    return [name for name, values in components.items() if not values[index]]


def stationarity(arrays, start, end, complete, grip_success):
    """Recompute 5 s TCP pose stationarity from the admission observation."""
    positions = arrays["tcp_position_m"][start:end + 1]
    quaternions = arrays["tcp_quaternion_wxyz"][start:end + 1]
    _unit_quaternions(quaternions)
    displacement = np.linalg.norm(positions - positions[0], axis=1)
    dots = np.abs(quaternions @ quaternions[0])
    angles = 2 * np.arccos(np.clip(dots, 0., 1.))
    max_position, max_angle = float(np.max(displacement)), float(np.max(angles))
    within = (max_position <= TCP_POSITION_LIMIT_M + 1e-12
              and max_angle <= TCP_ORIENTATION_LIMIT_RAD + 1e-12)
    return {
        "maximum_tcp_position_excursion_m": max_position,
        "maximum_tcp_orientation_excursion_rad": max_angle,
        "terminal_tcp_position_delta_m": (positions[-1] - positions[0]).tolist(),
        "terminal_tcp_orientation_delta_rad": float(angles[-1]),
        "tcp_pose_within_bounds": bool(complete and within),
        "grip_retention_and_stationarity": bool(complete and grip_success and within),
    }


def treatment_boundary(policy):
    """Earliest admission or pre-admission bound refusal for a scene/variant."""
    events = policy.get("events") if type(policy) is dict else None
    if type(events) is not list:
        raise ValueError("Policy event tape is missing.")
    admissions = [event for event in events
                  if type(event) is dict and event.get("type") in ADMISSIONS]
    if len(admissions) > 1:
        raise ValueError("Policy admits more than once.")
    admission_time = admissions[0]["t_seconds"] if admissions else None
    candidates = []
    if admissions:
        candidates.append({"time_s": admission_time, "reason": "admission",
                           "detail": admissions[0].get("reason", admissions[0].get("type"))})
    for event in events:
        if (type(event) is dict and event.get("type") == "hold_invalidated"
                and event.get("reason") in BOUND_REFUSALS):
            timestamp = event.get("t_seconds")
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                raise ValueError("Bound refusal has malformed timestamp.")
            if admission_time is None or timestamp <= admission_time + 1e-8:
                candidates.append({"time_s": float(timestamp), "reason": "pre_admission_bound_refusal",
                                   "detail": event["reason"]})
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["time_s"])


def common_cutoff(boundaries):
    """One common scene cutoff across all four scheduled variants."""
    if set(boundaries) != set(VARIANTS):
        raise ValueError("Common cutoff requires all four variant records.")
    candidates = [{"variant": variant, **row} for variant, row in boundaries.items() if row is not None]
    candidates.sort(key=lambda row: row["variant"])
    timestamp = min((row["time_s"] for row in candidates), default=None)
    reasons = sorted({row["reason"] for row in candidates
                      if timestamp is not None and abs(row["time_s"] - timestamp) <= 1e-8})
    return {"time_s": timestamp, "reason": "+".join(reasons) if reasons else "full_run_no_treatment",
            "treatment_boundaries": candidates}


def prefix_identity(arrays, actions, cutoff):
    """Digest native rows through cutoff, excluding command at cutoff."""
    if len(arrays["t_seconds"]) != 20000 or len(actions) != 1000:
        return {"verified": False, "reason": "incomplete_native_or_command_prefix"}
    if cutoff["time_s"] is None:
        native_rows, command_rows = 20000, list(actions)
    else:
        native_rows = row_at(cutoff["time_s"], arrays["t_seconds"]) + 1
        command_rows = [action for action in actions
                        if action["t_seconds"] < cutoff["time_s"] - 1e-8]
    prefix_hash = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name][:native_rows])
        prefix_hash.update(json.dumps({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)},
                                      sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        prefix_hash.update(value.tobytes())
    return {"verified": True, "cutoff_time_s": cutoff["time_s"], "cutoff_reason": cutoff["reason"],
            "native_rows": native_rows, "native_sha256": prefix_hash.hexdigest(),
            "commands": len(command_rows),
            "commands_sha256": sha256(json.dumps(command_rows, sort_keys=True, indent=2,
                                                  allow_nan=False).encode() + b"\n").hexdigest()}


def paired_prefix(scene, attempts):
    """Preserve incomplete attempts as unknown; compare all complete four-way prefixes."""
    if set(attempts) != set(VARIANTS):
        raise ValueError("Scene must retain all four scheduled variants.")
    if not all(attempts[variant].get("valid") is True for variant in VARIANTS):
        entries = [{"variant": variant, "verified": False,
                    "reason": attempts[variant].get("reason", "common_quartet_incomplete")
                              if attempts[variant].get("valid") is not True else "common_quartet_incomplete",
                    "native_rows_available": attempts[variant].get("native_rows_available")}
                   for variant in VARIANTS]
        return {"scene": scene, "verified": False,
                "reason": "invalid_attempt_prevents_complete_quartet", "variants": entries,
                "treatment_boundaries": []}
    cutoff = common_cutoff({variant: attempts[variant]["boundary"] for variant in VARIANTS})
    prefixes = {variant: prefix_identity(attempts[variant]["arrays"], attempts[variant]["actions"], cutoff)
                for variant in VARIANTS}
    if any(record.get("verified") is not True for record in prefixes.values()):
        return {"scene": scene, "verified": False, "reason": "incomplete_common_prefix_evidence",
                "variants": prefixes, "treatment_boundaries": cutoff["treatment_boundaries"]}
    identities = {(record["cutoff_time_s"], record["cutoff_reason"], record["native_rows"],
                   record["native_sha256"], record["commands"], record["commands_sha256"])
                  for record in prefixes.values()}
    if len(identities) != 1:
        raise ValueError("Common acquisition prefixes differ: " + scene)
    expected = next(iter(prefixes.values()))
    return {"scene": scene, "verified": True, **expected,
            "treatment_boundaries": cutoff["treatment_boundaries"],
            "variants": [{"variant": variant, **prefixes[variant]} for variant in VARIANTS]}


def _vec(value, label):
    result = np.asarray(value, dtype=np.float64)
    assert result.shape == (6,) and np.isfinite(result).all(), label
    return result


def inside_joint_margin(joints, limits, margin=.02):
    joints, limits = _vec(joints, "measured joints"), np.asarray(limits, dtype=np.float64)
    assert limits.shape == (6, 2) and np.isfinite(limits).all()
    return bool(np.all(joints >= limits[:, 0] + margin - 1e-12)
                and np.all(joints <= limits[:, 1] - margin + 1e-12))


def check_reported_stationarity(actual, reported, complete):
    assert reported["complete_5s"] is bool(complete)
    for field in ("maximum_tcp_position_excursion_m", "maximum_tcp_orientation_excursion_rad"):
        assert abs(reported[field] - actual[field]) <= 1e-10, field
    assert np.allclose(reported["terminal_tcp_position_delta_m"], actual["terminal_tcp_position_delta_m"],
                       atol=1e-10, rtol=0.)
    assert abs(reported["terminal_tcp_orientation_delta_rad"] - actual["terminal_tcp_orientation_delta_rad"]) <= 1e-10
    for field in ("tcp_pose_within_bounds", "grip_retention_and_stationarity"):
        assert reported[field] is actual[field]


def correction_audit(policy, actions, arrays, boundary, variant):
    """Reconstruct candidate law, refusal behavior, margin, and final no-command row."""
    retention = policy.get("retention") or {}
    records = retention.get("hold_records")
    reference = retention.get("hold_reference")
    assert policy.get("variant") == variant
    if type(records) is not list:
        raise ValueError("Hold diagnostics are missing.")
    if variant != "incremental":
        # Legacy controls have their own audit in the historical study; for this
        # fixed comparison we still check bumpless admission, jaw freeze, and final seal.
        return baseline_hold_audit(policy, actions, arrays, boundary, variant)

    assert policy.get("candidate") == "incremental"
    law = retention.get("update_law")
    assert type(law) is str and law
    assert retention.get("update_gain_per_boundary") == 1.
    limits = np.asarray(retention.get("native_joint_control_limits_rad"), dtype=np.float64)
    assert limits.shape == (6, 2) and np.isfinite(limits).all()
    if not records:
        assert reference is None
        assert boundary is None or boundary.get("reason") == "pre_admission_bound_refusal"
        return {"hold_record_count": 0, "bound_refusal_counts": {},
                "rate_limited_axis_occurrences": 0,
                "maximum_abs_pose_error_increment_rad": None,
                "maximum_abs_active_correction_rad": None,
                "maximum_abs_proposed_correction_rad": None,
                "first_admitted_target_continuous": False,
                "final_observation_no_ik_or_command": False}
    refused_counts = Counter()
    rate_limited_axes = 0
    max_increment = max_active = max_proposal = 0.
    previous = np.zeros(6, dtype=np.float64)
    rate_limited_axes = sum(len(row.get("correction_rate_limited_axes") or []) for row in records)
    accepted_rates = []
    inactive_latched_target = None
    command_by_time = {float(row["t_seconds"]): row for row in actions}
    assert len(command_by_time) == len(actions)
    admission = boundary is not None and boundary["reason"] == "admission"

    if reference is not None:
        assert reference.get("frozen_jaw_target_m") == policy["frozen_jaw_target_m"]
        issued = _vec(reference.get("issued_joint_targets_rad"), "admission issued target")
        measured = _vec(reference.get("measured_joint_position_rad"), "admission measured joints")
        first_time = float(reference["t_seconds"])
        if boundary is None or boundary.get("detail") != "measured_joint_margin_exhausted":
            assert inside_joint_margin(measured, limits)
        if admission or boundary is None or boundary.get("detail") != "native_joint_margin_exhausted":
            assert inside_joint_margin(issued, limits)
        if admission:
            first = records[0]
            assert first["command_issued"] is True and abs(first["t_seconds"] - first_time) <= 1e-8
            assert _vec(first["joint_targets_rad"], "first admitted target").tolist() == issued.tolist()
            first_action = command_by_time[first_time]
            assert first_action["joint_targets_rad"] == first["joint_targets_rad"]
            assert first_action["gripper_position_m"] == reference["frozen_jaw_target_m"]
            assert abs(first_time - boundary["time_s"]) <= 1e-8
            previous_actions = [row for row in actions if row["t_seconds"] < first_time - 1e-8]
            if previous_actions:
                assert previous_actions[-1]["joint_targets_rad"] == reference["issued_joint_targets_rad"]

    for record in records:
        time = float(record["t_seconds"])
        assert record.get("candidate") == "incremental"
        assert record.get("update_law") == law
        assert record.get("update_gain_per_boundary") == 1.
        accumulated = _vec(record.get("accumulated_correction_rad"), "stored correction")
        assert np.max(np.abs(accumulated)) <= CORRECTION_LIMIT_RAD + 1e-12
        before = _vec(record.get("correction_before_update_rad"), "prior correction")
        assert np.allclose(before, previous, atol=1e-12, rtol=0.)
        refused = record.get("refused_bounds")
        if refused is not None:
            refused_counts[refused] += 1
            assert any(event.get("type") == "hold_invalidated" and event.get("reason") == refused
                       and abs(event.get("t_seconds", -1.) - time) <= 1e-8
                       for event in policy["events"])

        final = record.get("final_observation") is True
        command_issued = record.get("command_issued") is True
        assert not final or not command_issued
        measured_q = _vec(record.get("measured_joint_position_rad"), "record measured joints")
        native_index = row_at(time, arrays["t_seconds"])
        assert np.allclose(measured_q, arrays["joint_position_rad"][native_index], atol=1e-9, rtol=0.)
        margin_ok = inside_joint_margin(measured_q, limits)
        raw_value = record.get("raw_pose_joint_targets_rad")
        increment_value = record.get("pose_error_increment_rad")
        proposal_value = record.get("proposed_cumulative_correction_rad")
        accepted = record.get("correction_update_accepted") is True

        if not margin_ok:
            assert refused == "measured_joint_margin_exhausted"
            assert raw_value is None and increment_value is None and proposal_value is None
            assert accepted is False and np.allclose(accumulated, before, atol=1e-12, rtol=0.)
        elif refused == "measured_joint_margin_exhausted":
            raise AssertionError("Measured-joint margin refusal has an in-margin observation")

        if final:
            assert raw_value is None and increment_value is None and proposal_value is None
            assert accepted is False and record.get("applied_cumulative_correction_rad") is None
            assert record.get("joint_targets_rad") is None
            assert record.get("preceding_issued_joint_targets_rad") is not None
            assert record["preceding_issued_joint_targets_rad"] == actions[-1]["joint_targets_rad"]
            assert record.get("active_regulation") is (policy.get("admitted") is True)
            assert record.get("correction_rate_limited_axes") == []
            continue

        if record.get("active_regulation") is True and margin_ok and refused is None:
            # Admission has a pose solve to establish the bias but intentionally
            # does not integrate. Later control boundaries must log the full law.
            if time == (reference or {}).get("t_seconds"):
                assert raw_value is not None
                raw = _vec(raw_value, "admission raw pose target")
                bias = _vec(reference.get("fixed_bias_rad"), "initial bias")
                assert np.allclose(issued - raw, bias, atol=1e-12, rtol=0.)
                assert np.max(np.abs(bias)) <= INITIAL_BIAS_LIMIT_RAD + 1e-12
                assert increment_value is None and accepted is False
                assert np.allclose(accumulated, 0., atol=1e-12, rtol=0.)
                previous = accumulated
            else:
                assert raw_value is not None and increment_value is not None and proposal_value is not None
                raw = _vec(raw_value, "raw pose target")
                increment = _vec(increment_value, "pose error increment")
                expected_increment = raw - measured_q
                assert np.allclose(increment, expected_increment, atol=1e-12, rtol=0.)
                max_increment = max(max_increment, float(np.max(np.abs(increment))))
                proposal = _vec(proposal_value, "cumulative proposal")
                expected_proposal = before + increment
                assert np.allclose(proposal, expected_proposal, atol=1e-12, rtol=0.)
                max_proposal = max(max_proposal, float(np.max(np.abs(proposal))))
                assert np.max(np.abs(proposal)) <= CORRECTION_LIMIT_RAD + 1e-12
                applied = before + np.clip(increment, -CORRECTION_STEP_RAD, CORRECTION_STEP_RAD)
                axes = [i for i, (step, old) in enumerate(zip(increment, before))
                        if abs(step) > CORRECTION_STEP_RAD + 1e-12]
                assert record.get("correction_rate_limited_axes") == axes
                assert accepted is True
                assert np.allclose(accumulated, applied, atol=1e-12, rtol=0.)
                assert np.allclose(record.get("applied_cumulative_correction_rad"), applied,
                                   atol=1e-12, rtol=0.)
                target = _vec(record.get("joint_targets_rad"), "incremental joint target")
                expected_target = issued + applied
                assert np.allclose(target, expected_target, atol=1e-12, rtol=0.)
                max_active = max(max_active, float(np.max(np.abs(applied))))
                assert inside_joint_margin(target, limits)
                accepted_rates.append(float(np.max(np.abs(applied - before)) / CONTROL_DT_S))
                previous = accumulated
        elif record.get("active_regulation") is True:
            # The row that refuses measured-joint margin is checked before IK.
            assert not margin_ok and refused == "measured_joint_margin_exhausted"
            assert raw_value is None and increment_value is None and proposal_value is None
            assert accepted is False and np.allclose(accumulated, before, atol=1e-12, rtol=0.)
            assert record.get("applied_cumulative_correction_rad") is None
            previous = accumulated
        elif refused == "joint_correction_exhausted":
            assert raw_value is not None and increment_value is not None and proposal_value is not None
            raw = _vec(raw_value, "refused raw pose target")
            increment = _vec(increment_value, "refused pose error increment")
            assert np.allclose(increment, raw - measured_q, atol=1e-12, rtol=0.)
            proposal = _vec(proposal_value, "refused proposal")
            assert np.allclose(proposal, before + increment, atol=1e-12, rtol=0.)
            assert np.max(np.abs(proposal)) > CORRECTION_LIMIT_RAD + 1e-12
            assert accepted is False and np.allclose(accumulated, before, atol=1e-12, rtol=0.)
            assert record.get("applied_cumulative_correction_rad") is None
            previous = accumulated
        elif refused == "native_joint_margin_exhausted":
            assert raw_value is not None and increment_value is not None and proposal_value is not None
            raw = _vec(raw_value, "joint-margin refusal raw target")
            increment = _vec(increment_value, "joint-margin refusal increment")
            proposal = _vec(proposal_value, "joint-margin refusal proposal")
            assert np.allclose(increment, raw - measured_q, atol=1e-12, rtol=0.)
            assert np.allclose(proposal, before + increment, atol=1e-12, rtol=0.)
            limited = before + np.clip(increment, -CORRECTION_STEP_RAD, CORRECTION_STEP_RAD)
            assert not inside_joint_margin(issued + limited, limits)
            assert accepted is False and np.allclose(accumulated, before, atol=1e-12, rtol=0.)
            assert record.get("applied_cumulative_correction_rad") is None
            previous = accumulated
        elif refused == "initial_bias_exhausted":
            assert reference is not None and reference.get("fixed_bias_rad") is not None
            bias = _vec(reference["fixed_bias_rad"], "refused initial bias")
            assert np.max(np.abs(bias)) > INITIAL_BIAS_LIMIT_RAD + 1e-12
            assert accepted is False and np.allclose(accumulated, before, atol=1e-12, rtol=0.)
            previous = accumulated
        else:
            # Terminal grip/pose/solver refusal may have no fresh IK result and
            # must never integrate a stale proposal.
            assert accepted is False
            assert np.allclose(accumulated, before, atol=1e-12, rtol=0.)
            assert record.get("applied_cumulative_correction_rad") is None
            previous = accumulated

        action = command_by_time.get(time)
        assert action is not None
        assert action["joint_targets_rad"] == record["joint_targets_rad"]
        assert action["gripper_position_m"] == record["frozen_jaw_target_m"]
        if record.get("active_regulation") is False:
            # Terminal fallback samples measured joints once, then holds that
            # latched target. The arm may move slightly after the latch, so
            # comparing every later request with each new measurement would
            # incorrectly reject a valid terminal hold.
            target = _vec(record.get("joint_targets_rad"), "terminal hold target")
            if inactive_latched_target is None:
                assert np.allclose(target, measured_q, atol=1e-12, rtol=0.)
                inactive_latched_target = target.copy()
            else:
                assert np.allclose(target, inactive_latched_target, atol=1e-12, rtol=0.)
        if reference is not None:
            assert record["frozen_jaw_target_m"] == reference["frozen_jaw_target_m"]

    # Every requested hold record must map one-to-one onto an issued action.
    command_records = [row for row in records if row.get("command_issued") is True]
    assert len({float(row["t_seconds"]) for row in command_records}) == len(command_records)
    return {
        "hold_record_count": len(records), "bound_refusal_counts": dict(refused_counts),
        "rate_limited_axis_occurrences": rate_limited_axes,
        "maximum_abs_pose_error_increment_rad": max_increment or None,
        "maximum_abs_active_correction_rad": max_active or None,
        "maximum_abs_proposed_correction_rad": max_proposal or None,
        "maximum_abs_accepted_correction_rate_rad_s": max(accepted_rates) if accepted_rates else None,
        "first_admitted_target_continuous": bool(admission and reference is not None),
        "final_observation_no_ik_or_command": bool(records[-1].get("final_observation")
                                                       and records[-1].get("raw_pose_joint_targets_rad") is None
                                                       and records[-1].get("command_issued") is False),
    }


def baseline_hold_audit(policy, actions, arrays, boundary, variant):
    """Check inherited holds' admission continuity, frozen jaw, and final seal."""
    retention = policy.get("retention") or {}
    records, reference = retention.get("hold_records"), retention.get("hold_reference")
    if type(records) is not list:
        raise ValueError("Hold diagnostics are missing.")
    assert policy.get("variant") == variant
    if reference is not None:
        issued = _vec(reference["issued_joint_targets_rad"], "admission issued target")
        assert reference["frozen_jaw_target_m"] == policy["frozen_jaw_target_m"]
        if boundary is not None and boundary["reason"] == "admission":
            first = records[0]
            assert first["command_issued"] is True
            assert abs(first["t_seconds"] - boundary["time_s"]) <= 1e-8
            expected_first = (_vec(reference["measured_joint_position_rad"], "admission measured joints")
                              if variant == "measured_joint" else issued)
            assert np.array_equal(_vec(first["joint_targets_rad"], "admission target"), expected_first)
            action = next(row for row in actions if abs(row["t_seconds"] - boundary["time_s"]) <= 1e-8)
            assert first["joint_targets_rad"] == action["joint_targets_rad"]
            prior = [row for row in actions if row["t_seconds"] < boundary["time_s"] - 1e-8]
            if prior:
                assert prior[-1]["joint_targets_rad"] == reference["issued_joint_targets_rad"]
    for record in records:
        if record.get("final_observation"):
            assert record.get("command_issued") is False
            assert record.get("raw_pose_joint_targets_rad") is None
            assert record.get("joint_targets_rad") == actions[-1]["joint_targets_rad"]
            continue
        if record.get("command_issued"):
            action = next(row for row in actions if abs(row["t_seconds"] - record["t_seconds"]) <= 1e-8)
            assert action["joint_targets_rad"] == record["joint_targets_rad"]
            assert action["gripper_position_m"] == record["frozen_jaw_target_m"]
    return {"hold_record_count": len(records), "first_admitted_target_continuous": bool(reference),
            "final_observation_no_ik_or_command": bool(records[-1].get("final_observation")
                                                         and not records[-1].get("command_issued")
                                                         and records[-1].get("raw_pose_joint_targets_rad") is None)}


def native_success(arrays, policy, admission):
    probe = next(row for row in policy["probes"] if row["probe_index"] == admission["probe_index"])
    condition, streak, qualified, components, rise, drift = native_predicate(arrays, probe)
    times = arrays["t_seconds"]
    start = row_at(admission["t_seconds"], times)
    target = admission["t_seconds"] + 5.
    # A NumPy scalar comparison yields np.bool_; reports deserialize as bool,
    # so normalize before exact endpoint comparisons and JSON output.
    complete = bool(target <= times[-1] + 1e-8)
    end = row_at(target, times) if complete else len(times) - 1
    admission_ok = bool(qualified[start])
    grip = bool(admission_ok and complete and np.all(condition[start:end + 1])
                and np.all(arrays["pad_normal_forces_n"][start:] <= 10.))
    pose = stationarity(arrays, start, end, complete, grip)
    losses = np.flatnonzero(~condition[start:end + 1])
    loss_index = start + int(losses[0]) if len(losses) else None
    return {
        "admission_ok": admission_ok, "streak_ms": int(streak[start]),
        "grip_only_5s": grip, "grip_and_stationary_5s": pose["grip_retention_and_stationarity"],
        "complete_5s": complete, "start": start, "end": end, "condition": condition,
        "components": components, "stationarity": pose,
        "first_grip_loss_s": float(times[loss_index]) if loss_index is not None else None,
        "first_grip_loss_reasons": reasons_at(loss_index, components) if loss_index is not None else [],
        "first_tcp_pose_failure_s": next((float(times[start + i]) for i, ok in enumerate(
            (np.linalg.norm(arrays["tcp_position_m"][start:end + 1] - arrays["tcp_position_m"][start], axis=1)
             <= TCP_POSITION_LIMIT_M + 1e-12)
            & (2 * np.arccos(np.clip(np.abs(arrays["tcp_quaternion_wxyz"][start:end + 1]
                                                @ arrays["tcp_quaternion_wxyz"][start]), 0., 1.))
               <= TCP_ORIENTATION_LIMIT_RAD + 1e-12)) if not ok), None),
    }


def false_maintained_claims(arrays, policy, admission, admission_probe):
    """Count explicit true controller claims contradicted by native grip or TCP pose."""
    checks = policy.get("qualification_checks")
    if type(checks) is not list:
        raise ValueError("Qualification checks are missing.")
    condition, _, qualified, components, _, _ = native_predicate(arrays, admission_probe)
    start = row_at(admission["t_seconds"], arrays["t_seconds"])
    pose_pos = np.linalg.norm(arrays["tcp_position_m"] - arrays["tcp_position_m"][start], axis=1)
    q = arrays["tcp_quaternion_wxyz"]
    pose_ang = 2 * np.arccos(np.clip(np.abs(q @ q[start]), 0., 1.))
    grip_continuous = np.logical_and.accumulate(condition[start:]) & bool(qualified[start])
    pose_continuous = np.logical_and.accumulate(
        (pose_pos[start:] <= TCP_POSITION_LIMIT_M + 1e-12)
        & (pose_ang[start:] <= TCP_ORIENTATION_LIMIT_RAD + 1e-12))
    claims = [row for row in checks if row.get("qualified") is True
              and row["t_seconds"] > admission["t_seconds"] + 1e-8]
    false_grip, false_pose = [], []
    for claim in claims:
        index = row_at(claim["t_seconds"], arrays["t_seconds"])
        offset = index - start
        if not grip_continuous[offset]:
            false_grip.append({"t_seconds": claim["t_seconds"],
                               "reasons": reasons_at(index, components),
                               "native_prefix_qualified": bool(qualified[index])})
        if not pose_continuous[offset]:
            false_pose.append({"t_seconds": claim["t_seconds"],
                               "position_excursion_m": float(pose_pos[index]),
                               "orientation_excursion_rad": float(pose_ang[index])})
    return {"maintained_checks": len(claims), "false_maintained_grip": false_grip,
            "false_maintained_pose": false_pose}


def reconstruct_controls(arrays, actions, requested, trajectory, initial_controls, case):
    """Replay the fixed requested-to-applied 1 ms per-actuator slew."""
    assert len(actions) == len(requested) == 1000 and len(trajectory) == 1001
    control = np.asarray(initial_controls, dtype=np.float64)
    assert control.shape == (7,) and np.isfinite(control).all()
    max_error = 0.
    rates = np.array([case["config"]["joint_target_rate_rad_s"] * .001] * 6 +
                     [case["config"]["jaw_target_rate_m_s"] * .001], dtype=np.float64)
    for index, (action, request) in enumerate(zip(actions, requested)):
        assert abs(action["t_seconds"] - index * CONTROL_DT_S) < 1e-8
        assert request["index"] == index
        assert all(action[key] == request[key] for key in ("t_seconds", "joint_targets_rad", "gripper_position_m"))
        assert trajectory[index]["command"] == action
        assert trajectory[index]["issued_joint_targets_rad"] == action["joint_targets_rad"]
        assert trajectory[index]["issued_gripper_position_m"] == action["gripper_position_m"]
        target = np.asarray(action["joint_targets_rad"] + [action["gripper_position_m"]], dtype=np.float64)
        assert target.shape == (7,) and np.isfinite(target).all()
        for tick in range(index * 20, (index + 1) * 20):
            control += np.clip(target - control, -rates, rates)
            actual = np.r_[arrays["arm_actuator_controls_rad"][tick], arrays["applied_jaw_target_m"][tick]]
            error = float(np.max(np.abs(control - actual)))
            max_error = max(max_error, error)
            assert error < 1e-12
            assert arrays["requested_jaw_target_m"][tick] == action["gripper_position_m"]
        for field in ("object_position_m", "joint_position_rad", "tcp_position_m"):
            assert np.allclose(trajectory[index + 1][field], arrays[field][(index + 1) * 20 - 1],
                               atol=1e-9, rtol=0.)
    assert all(trajectory[-1].get(key) is None
               for key in ("command", "issued_joint_targets_rad", "issued_gripper_position_m"))
    return {"requested_commands": len(actions), "native_substeps_reconstructed": 20000,
            "maximum_control_reconstruction_error": max_error}


def validate_native(arrays, metadata, *, expected_rows=20000):
    assert set(arrays) == set(FIELDS)
    assert metadata.get("schema") == "dvidia.piper-feedback-native/v1"
    assert metadata.get("native_timestep_s") == CONTROL_SUBSTEP_S
    assert metadata.get("physical_robot_ready") is False
    assert metadata.get("side_order") == ["left", "right"]
    assert metadata.get("fields") == [{"name": name, "width": width} for name, width in FIELDS.items()]
    times = arrays["t_seconds"]
    assert times.ndim == 1 and 0 < len(times) <= 30000
    assert metadata.get("rows") == len(times)
    assert expected_rows is None or len(times) == expected_rows
    for name, width in FIELDS.items():
        expected = (len(times),) if width == 1 else (len(times), width)
        assert arrays[name].dtype == np.dtype("float64") and arrays[name].shape == expected
        assert np.isfinite(arrays[name]).all()
    ticks = np.arange(1, len(times) + 1, dtype=np.float64)
    assert np.array_equal(arrays["native_tick"], ticks)
    assert np.max(np.abs(times - ticks * CONTROL_SUBSTEP_S)) <= 1e-8
    for name in ("grasp_contact_counts", "table_object_contact_count", "unexpected_contact_count"):
        values = arrays[name]
        assert np.all((values >= 0) & (values <= 10000) & (values == np.floor(values)))
    assert np.all((arrays["pad_normal_forces_n"] >= 0) & (arrays["pad_normal_forces_n"] <= 1e6))
    assert not np.any(arrays["pulse_active"]) and not np.any(arrays["pulse_closing_tail"])
    rotation_local(arrays)
    for name in ("tcp_quaternion_wxyz", "object_quaternion_wxyz"):
        _unit_quaternions(arrays[name])
    return len(times)


def check_schedule(protocol, report):
    assert protocol["planned_attempts"] == report["planned_attempts"] == report["attempted"] == 32
    assert protocol["variants"] == list(VARIANTS)
    cases = protocol["cases"]
    assert len(cases) == 8 and len({case["id"] for case in cases}) == 8
    assert [case["seed"] for case in cases] == list(range(311200, 311208))
    planned = [(case, variant) for case in cases for variant in case["variant_order"]]
    assert len(planned) == 32 and all(set(case["variant_order"]) == set(VARIANTS) for case, _ in planned)
    expected_orders = (("measured_joint", "issued_joint", "incremental", "cartesian"),
                       ("issued_joint", "cartesian", "measured_joint", "incremental"),
                       ("cartesian", "incremental", "issued_joint", "measured_joint"),
                       ("incremental", "measured_joint", "cartesian", "issued_joint"))
    assert tuple(tuple(case["variant_order"]) for case in cases) == expected_orders * 2
    for variant in VARIANTS:
        assert [sum(case["variant_order"][position] == variant for case in cases)
                for position in range(4)] == [2, 2, 2, 2]
    assert [case["id"] + "-" + variant for case, variant in planned] == [row["id"] for row in report["rows"]]
    return cases, planned


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in frozen study: " + str(path))
        if path.is_file():
            result[path.relative_to(root).as_posix()] = identity(path)
    return result


def read_npz(path):
    with zipfile.ZipFile(path) as bundle:
        entries = bundle.infolist()
        assert (len(entries) == 22 and len({entry.filename for entry in entries}) == 22
                and sum(entry.file_size for entry in entries) <= MAX_FILE
                and all(entry.file_size <= 2 * 1024 * 1024 and "/" not in entry.filename
                        and entry.filename.endswith(".npy") for entry in entries))
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def compact_policy(policy):
    if policy is None:
        return None
    result = {key: value for key, value in policy.items()
              if key not in ("requested_controls", "qualification_checks")}
    if "retention" in result:
        result["retention"] = {key: value for key, value in result["retention"].items()
                               if key != "hold_records"}
    return result


def correction_summary(policy, variant):
    records = ((policy or {}).get("retention") or {}).get("hold_records", [])
    flags, refusals = {}, Counter()
    active, desired, increments, accepted_rates, axis_count = [], [], [], [], 0
    for record in records:
        for key, value in record.items():
            if type(value) is bool and any(token in key for token in ("saturat", "limit", "refus")):
                flags[key] = flags.get(key, 0) + int(value)
        if record.get("refused_bounds") is not None:
            refusals[record["refused_bounds"]] += 1
        axis_count += len(record.get("correction_rate_limited_axes") or [])
        for key, output in (("correction_rad", active), ("desired_correction_rad", desired)):
            if record.get(key) is not None:
                output.append(_vec(record[key], key))
        increment = record.get("pose_error_increment_rad")
        if increment is not None:
            increments.append(_vec(increment, "pose_error_increment_rad"))
        if record.get("command_issued") and record.get("correction_update_accepted"):
            before = _vec(record["correction_before_update_rad"], "correction_before_update_rad")
            after = _vec(record["accumulated_correction_rad"], "accumulated_correction_rad")
            accepted_rates.append(float(np.max(np.abs(after - before)) / CONTROL_DT_S))
    max_abs = lambda values: float(np.max(np.abs(values))) if values else None
    retention = (policy or {}).get("retention") or {}
    result = {
        "hold_record_count": len(records), "flag_counts": flags,
        "bound_refusal_counts": dict(refusals), "rate_limited_axis_occurrences": axis_count,
        "maximum_abs_active_correction_rad": max_abs(active),
        "maximum_abs_desired_correction_rad": max_abs(desired),
        "maximum_abs_pose_error_increment_rad": None,
        "maximum_abs_accepted_correction_rate_rad_s": None,
        "input_refusals": retention.get("input_refusals", []),
        "scope": "Includes recorded final diagnostic checks; only command_issued rows represent issued controls.",
    }
    if variant == "incremental":
        result.update(maximum_abs_pose_error_increment_rad=max_abs(increments),
                      maximum_abs_accepted_correction_rate_rad_s=max(accepted_rates) if accepted_rates else None)
    return result


def compare_endpoint(summary, item, arrays):
    evaluation, policy = summary["evaluation"], item.get("policy")
    admissions = [event for event in (policy or {}).get("events", [])
                  if event.get("type") == "evidence_qualified"]
    assert len(admissions) <= 1
    if not admissions:
        assert evaluation.get("first_admission") is None
        assert evaluation["primary"]["success"] is False
        assert evaluation["primary"]["native_pause_success"] is False
        assert evaluation["primary"]["combined_qualified_pause"] is False
        assert evaluation["grip_only"]["success"] is False
        assert evaluation["stationarity"]["grip_retention_and_stationarity"] is False
        assert evaluation["claims"]["maintained_checks"] == 0
        return {"admitted": False, "false_initial": False, "grip_only_5s": False,
                "grip_stationary_5s": False, "controller_retained_5s": False,
                "combined_5s": False, "final_both_qualified": bool(evaluation["final"]["both_qualified"]),
                "false_maintained_grip": 0, "false_maintained_pose": 0, "maintained_checks": 0,
                "scope": "No admission; retained in the denominator as failure."}
    event = admissions[0]
    native = native_success(arrays, policy, event)
    first = evaluation["first_admission"]
    assert abs(first["time_s"] - event["t_seconds"]) <= 1e-8
    assert first["probe_index"] == event["probe_index"]
    assert first["native_prefix_qualified"] is native["admission_ok"]
    assert abs(first["native_qualifying_window_s"] - native["streak_ms"] * .001) <= 1e-12
    primary, grip_only = evaluation["primary"], evaluation["grip_only"]
    assert primary["complete_5s"] is native["complete_5s"]
    assert primary["native_pause_success"] is native["grip_only_5s"]
    assert grip_only["success"] is native["grip_only_5s"]
    assert primary["success"] is native["grip_and_stationary_5s"]
    target_time = event["t_seconds"] + 5.
    checks = [row for row in policy["qualification_checks"]
              if row.get("probe_index") == event["probe_index"]
              and abs(row["t_seconds"] - target_time) <= 1e-8]
    controller = bool(native["complete_5s"] and len(checks) == 1 and checks[0]["qualified"] is True)
    assert primary["controller_retained_at_5s"] is controller
    assert primary["combined_qualified_pause"] is (native["grip_and_stationary_5s"] and controller)
    assert primary["first_tcp_pose_failure_s"] == native["first_tcp_pose_failure_s"]
    check_reported_stationarity(native["stationarity"], evaluation["stationarity"], native["complete_5s"])
    probe = next(row for row in policy["probes"] if row["probe_index"] == event["probe_index"])
    false = false_maintained_claims(arrays, policy,
                                    {"t_seconds": event["t_seconds"], "probe_index": event["probe_index"]}, probe)
    claims = evaluation["claims"]
    assert claims["maintained_checks"] == false["maintained_checks"]
    assert claims["false_maintained_grip"] == len(false["false_maintained_grip"])
    assert claims["false_maintained_pose"] == len(false["false_maintained_pose"])
    return {"admitted": True, "false_initial": not native["admission_ok"],
            "grip_only_5s": native["grip_only_5s"],
            "grip_stationary_5s": native["grip_and_stationary_5s"],
            "controller_retained_5s": controller,
            "combined_5s": bool(native["grip_and_stationary_5s"] and controller),
            "final_both_qualified": bool(evaluation["final"]["both_qualified"]),
            "false_maintained_grip": claims["false_maintained_grip"],
            "false_maintained_pose": claims["false_maintained_pose"],
            "maintained_checks": claims["maintained_checks"],
            "native": native, "false_claim_details": false,
            "scope": "Independent native predicate recomputation from saved rows; controller claims checked separately."}


def _report_aggregates(report, variant):
    selected = [row for row in report["rows"] if row["variant"] == variant]
    return {
        "attempts": len(selected), "valid": sum(row["valid"] for row in selected),
        "admissions": sum(row["evaluation"]["first_admission"] is not None for row in selected),
        "false_initial_admissions": sum(bool(row["evaluation"]["first_admission"]
                                              and not row["evaluation"]["first_admission"]["native_prefix_qualified"])
                                         for row in selected),
        "native_5s_pause_success": sum(row["evaluation"]["primary"]["native_pause_success"] for row in selected),
        "native_5s_stationary_success": sum(row["evaluation"]["primary"]["success"] for row in selected),
        "controller_retained_at_5s": sum(row["evaluation"]["primary"]["controller_retained_at_5s"] for row in selected),
        "combined_qualified_5s_pause": sum(row["evaluation"]["primary"]["combined_qualified_pause"] for row in selected),
        "grip_retention_and_stationarity": sum(row["evaluation"]["stationarity"]["grip_retention_and_stationarity"] for row in selected),
        "false_maintained_grip": sum(row["evaluation"]["claims"]["false_maintained_grip"] for row in selected),
        "false_maintained_pose": sum(row["evaluation"]["claims"]["false_maintained_pose"] for row in selected),
        "final_both_qualified": sum(row["evaluation"]["final"]["both_qualified"] for row in selected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study, repo, output = args.study, args.repo, args.output
    if output.exists() or output.is_symlink():
        raise ValueError("Audit output must be new")
    raw_before = inventory(study)
    protocol, report, seal = read(study / "protocol.json"), read(study / "report.json"), read(study / "frozen.json")
    assert seal["protocol_sha256"] == raw_before["protocol.json"]["sha256"]
    assert protocol["schema"] == report["schema"] == "dvidia.piper-incremental-hold-study/v1"
    cases, planned = check_schedule(protocol, report)
    source_before = {}
    for name, expected in protocol["source_sha256"].items():
        current = identity(repo / "src" / "dvidia_training" / name)["sha256"]
        frozen = identity(study / "sources" / name)["sha256"]
        assert current == frozen == expected
        source_before[name] = current
    assert source_before == report["source_sha256"]

    totals = {variant: Counter(attempts=0, valid=0, admissions=0, false_initial=0,
                               grip_only_5s=0, grip_stationary_5s=0, controller_retained_5s=0,
                               combined_5s=0, final_both_qualified=0,
                               false_maintained_grip=0, false_maintained_pose=0,
                               maintained_checks=0, native_rows=0, commands=0)
              for variant in VARIANTS}
    rows, attempts_by_scene, paired = [], {case["id"]: {} for case in cases}, {}

    def retain_attempt(scene, variant, value):
        attempts_by_scene[scene][variant] = value
        if len(attempts_by_scene[scene]) == 4:
            paired[scene] = paired_prefix(scene, attempts_by_scene[scene])
            attempts_by_scene[scene] = {}

    for summary, (case, variant) in zip(report["rows"], planned):
        folder = study / summary["id"]
        for name, expected in summary["files"].items():
            assert "/" not in name and name not in (".", "..")
            assert identity(folder / name) == expected
        item = episode(folder / "episode.json.gz")
        assert item["schema"] == protocol["schema"]
        assert item["id"] == summary["id"] and item["scene_id"] == case["id"]
        assert item["config"] == case["config"] and item["variant"] == variant and item["seed"] == case["seed"]
        assert item["valid"] is summary["valid"] and item["error"] == summary["error"]
        assert item["metrics"] == summary["metrics"]
        assert canonical(compact_policy(item.get("policy"))) == canonical(summary["policy"])
        evaluation = read(folder / "evaluation.json")
        assert canonical(evaluation) == canonical(summary["evaluation"])
        arrays, metadata = read_npz(folder / "native.npz"), read(folder / "native.json")
        counter = totals[variant]
        counter["attempts"] += 1
        counter["native_rows"] += len(arrays.get("t_seconds", []))
        if not item["valid"] or item["error"] is not None:
            assert evaluation["primary"]["success"] is False and evaluation["grip_only"]["success"] is False
            retain_attempt(case["id"], variant, {"valid": False,
                           "reason": "invalid_attempt_retained_as_unknown",
                           "native_rows_available": len(arrays.get("t_seconds", []))})
            rows.append({"id": item["id"], "scene_id": case["id"], "variant": variant, "valid": False,
                         "admitted": False, "grip_only_5s": False, "grip_stationary_5s": False,
                         "controller_retained_5s": False, "combined_5s": False,
                         "scope": "Invalid/incomplete; retained in denominator as unknown failure."})
            continue

        validate_native(arrays, metadata)
        assert item["metrics"].get("valid") is True and item["metrics"].get("warnings") == []
        unexpected = int(np.count_nonzero(arrays["unexpected_contact_count"]))
        assert unexpected == item["metrics"]["unexpected_contact_native_ticks"] == 0
        assert len(item["actions"]) == 1000 and len(item["trajectory"]) == 1001
        controls = reconstruct_controls(arrays, item["actions"], item["policy"]["requested_controls"],
                                        item["trajectory"], item["trajectory"][0]["native_actuator_controls"], case)
        counter.update(valid=1, commands=controls["requested_commands"])
        boundary = treatment_boundary(item["policy"])
        endpoint = compare_endpoint(summary, item, arrays)
        correction = correction_audit(item["policy"], item["actions"], arrays, boundary, variant)
        computed = correction_summary(item["policy"], variant)
        assert canonical(computed) == canonical(evaluation["corrections"])
        if variant == "incremental":
            assert correction["final_observation_no_ik_or_command"]
        counter.update(
            admissions=int(endpoint["admitted"]), false_initial=int(endpoint["false_initial"]),
            grip_only_5s=int(endpoint["grip_only_5s"]), grip_stationary_5s=int(endpoint["grip_stationary_5s"]),
            controller_retained_5s=int(endpoint["controller_retained_5s"]),
            combined_5s=int(endpoint["combined_5s"]),
            final_both_qualified=int(endpoint["final_both_qualified"]),
            false_maintained_grip=int(endpoint["false_maintained_grip"]),
            false_maintained_pose=int(endpoint["false_maintained_pose"]),
            maintained_checks=int(endpoint["maintained_checks"]))
        retain_attempt(case["id"], variant, {"valid": True, "boundary": boundary,
                       "arrays": arrays, "actions": item["actions"]})
        rows.append({"id": item["id"], "scene_id": case["id"], "variant": variant, "valid": True,
                     **{key: value for key, value in endpoint.items()
                        if key not in ("native", "false_claim_details")},
                     "corrections": correction, "control_reconstruction": controls})

    assert len(report["rows"]) == 32 and all(counter["attempts"] == 8 for counter in totals.values())
    paired_rows = [paired[case["id"]] for case in cases]
    inspection = read(study / "inspection.json")
    assert inspection["schema"] == "dvidia.piper-incremental-hold-inspection/v1"
    assert inspection["attempts"] == 32
    assert inspection["native_rows"] == sum(counter["native_rows"] for counter in totals.values())
    inspected = {row["id"]: row for row in inspection["rows"]}
    assert set(inspected) == {row["id"] for row in report["rows"]}
    for row in report["rows"]:
        check = inspected[row["id"]]
        assert check["evaluation_recomputed"] is True
        if row["valid"]:
            assert check["native_validated"] is True and check["controls_recomputed"] is True
        else:
            assert check["retained_refusal"] is True or check["native_validated"] is False
    inspected_prefixes = {row["scene_id"]: row for row in inspection["acquisition_prefixes"]}
    assert set(inspected_prefixes) == {case["id"] for case in cases}
    for scene, prefix in paired.items():
        check = inspected_prefixes[scene]
        assert check["verified"] is prefix["verified"]
        if prefix["verified"]:
            for field in ("cutoff_time_s", "cutoff_reason", "native_rows", "native_sha256", "commands", "commands_sha256"):
                assert check[field] == prefix[field]
            assert check["treatment_boundaries"] == prefix["treatment_boundaries"]
        else:
            assert check["reason"] == "invalid_attempt_prevents_complete_four_variant_comparison"
    assert canonical(inspection["aggregates"]) == canonical(report["aggregates"])
    for variant in VARIANTS:
        assert canonical(report["aggregates"][variant]) == canonical(_report_aggregates(report, variant))

    source_after = {name: identity(repo / "src" / "dvidia_training" / name)["sha256"]
                    for name in source_before}
    assert source_after == source_before and raw_before == inventory(study)
    output.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema": "dvidia.incremental-hold-independent-audit/v1", "passed": True,
             "protocol_sha256": raw_before["protocol.json"]["sha256"],
             "source_sha256": source_before, "frozen_raw_files": raw_before,
             "rows": rows, "totals": {variant: dict(counter) for variant, counter in totals.items()},
             "paired_acquisition_prefixes": paired_rows, "source_and_raw_inputs_unchanged": True,
             "scope": "Independent saved-record predicate, controller-law and control-slew audit. No simulator/runtime/scorer imports or trial reruns; hashes do not authenticate external data or hardware."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "attempts": 32, "aggregates": report["aggregates"],
                      "raw_files": len(raw_before)}))


if __name__ == "__main__":
    main()
