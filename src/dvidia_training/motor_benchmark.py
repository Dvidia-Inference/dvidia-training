"""CPU motor-memory v0.2: native calibration, matched control, auditable evidence.

The learned component is one-axis actuator dynamics. Task phases and completion
criteria are authored. This module never treats video as robot action labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

from .motor_memory import (AppliedTarget, CalibrationTransition, ControllerConfig,
    ExpectedContract, MotorController, MotorMemory, MotorObservation, fit_memory)
from .motor_metrics import render_report, summarize
from .reaction import (ContactSample, GripSample, MotionSample, ReactionConfig,
    ReactionMonitor, RobotSphere, SensorConfig, SensorStream)

VERSION = "0.2"
MODES = ("slow_feedback", "fast_fixed", "adaptive_predictive")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _sources():
    root = Path(__file__).parent
    names = ("motor_memory.py", "motor_benchmark.py", "motor_metrics.py", "reaction.py", "reaction_native.py", "env.py")
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def _new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _manifest(directory):
    _write(directory/"evidence-manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir()) if p.is_file() and p.name != "evidence-manifest.json"})


def _check_files(directory, required):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"evidence-manifest.json").read_text())
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError("incomplete evidence manifest")
    for name, expected in manifest.items():
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("manifest must contain local filenames only")
        path = directory/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 192*1024*1024:
            raise ValueError("unsupported evidence file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed evidence: {name}")
    return directory


def train(directory, *, seed=20261009):
    """Fit separate native calibration transitions, then freeze the candidate."""
    from .reaction_native import FixtureConfig, NativeReactionFixture
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    directory = _new_directory(directory)
    protocol = {"format": "dvidia.motor-calibration", "schema_version": 1,
        "framework_version": VERSION, "seed": seed, "timestep_s": .001,
        "duration_s": 2., "phases_rad": [0., .7, 1.4, 2.1],
        "target_waveform": "min(t/.2,1)*(.025*sin(2*pi*1.1*t+phase)+.012*sin(2*pi*3.7*t))",
        "payload_mass_kg": .04, "pair_friction": .7, "holding": True,
        "braking_speeds_m_s": [.08, .16, .24], "brake_request_s": .6,
        "brake_delay_s": .02, "empirical_deceleration_factor": .5,
        "scope": "privileged native expert transitions, disjoint from task evaluation; authored feedback primitive"}
    _write(directory/"training-protocol.json", protocol)
    sources = _sources()
    _write(directory/"sources.json", sources)
    start = time.perf_counter()
    rows = []
    for phase in protocol["phases_rad"]:
        fixture = NativeReactionFixture(FixtureConfig(horizon_s=2., speed_m_s=.2), holding=True)
        with fixture.native():
            for index in range(2000):
                t = index*.001
                target = min(t/.2, 1.)*(.025*math.sin(2*math.pi*1.1*t+phase)+.012*math.sin(2*math.pi*3.7*t))
                x = float(fixture.data.qpos[fixture.travel_qpos])
                v = float(fixture.data.qvel[fixture.travel_dof])
                fixture.tick(target, (.3, 1., .4))
                rows.append(CalibrationTransition(x, v, target, float(fixture.data.qvel[fixture.travel_dof]), .001))
    with (directory/"training.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), separators=(",", ":"), allow_nan=False)+"\n")
    memory = fit_memory(rows, expected_contract=ExpectedContract(holding_required=True,
        expected_contact="bilateral jaw-payload contact; obstacle contact is unintended"))
    brakes = []
    for speed in protocol["braking_speeds_m_s"]:
        fixture = NativeReactionFixture(FixtureConfig(horizon_s=2., speed_m_s=speed), holding=True)
        hold = initial_v = initial_x = None
        maximum_x = -math.inf
        quiet = 0
        rest = None
        with fixture.native():
            for index in range(2000):
                t = index*.001
                if hold is None and t+1e-12 >= .62:
                    hold = initial_x = float(fixture.data.qpos[fixture.travel_qpos])
                    initial_v = float(fixture.data.qvel[fixture.travel_dof])
                fixture.tick(speed*(t+.001) if hold is None else hold, (.3, 1., .4))
                if hold is not None:
                    maximum_x = max(maximum_x, float(fixture.data.qpos[fixture.travel_qpos]))
                    quiet = quiet+1 if abs(float(fixture.data.qvel[fixture.travel_dof])) < .005 else 0
                    if rest is None and quiet >= 50:
                        rest = (index+1)*.001
        excursion = maximum_x-initial_x
        if rest is None or excursion <= 0 or initial_v <= 0:
            raise RuntimeError("braking calibration failed to measure positive finite stopping response")
        brakes.append({"speed_m_s": speed, "initial_speed_m_s": initial_v,
            "forward_excursion_m": excursion, "brake_requested_s": .6,
            "brake_applied_s": .62, "rest_s": rest,
            "equivalent_deceleration_m_s2": initial_v**2/(2*excursion),
            "dropped": fixture.measure()["dropped"]})
    candidate = {"format": "dvidia.motor-memory-candidate", "schema_version": 1,
        "framework_version": VERSION, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "memory": memory.to_dict(), "training_protocol_sha256": digest(protocol),
        "training_sha256": hashlib.sha256((directory/"training.jsonl").read_bytes()).hexdigest(),
        "source_sha256": sources, "braking_trials": brakes,
        "empirical_deceleration_m_s2": .5*min(r["equivalent_deceleration_m_s2"] for r in brakes),
        "braking_scope": "half minimum equivalent native stopping deceleration observed in three training coupons; assumption, not guaranteed bound",
        "training_wall_seconds": time.perf_counter()-start,
        "physical_robot_ready": False, "task_motion_learned_from_video": False,
        "scope": "fitted one-axis actuator model plus authored feedback movement; synthetic sensing, no whole-arm/hardware qualification"}
    _write(directory/"memory.json", candidate)
    _manifest(directory)
    return candidate


def inspect_memory(directory):
    root = _check_files(directory, {"training-protocol.json", "training.jsonl", "memory.json", "sources.json"})
    candidate = json.loads((root/"memory.json").read_text())
    protocol = json.loads((root/"training-protocol.json").read_text())
    if (candidate.get("format") != "dvidia.motor-memory-candidate" or candidate.get("schema_version") != 1
            or candidate.get("framework_version") != VERSION or candidate.get("physical_robot_ready") is not False
            or candidate.get("task_motion_learned_from_video") is not False):
        raise ValueError("unsupported motor memory scope or readiness")
    memory = MotorMemory.from_dict(candidate["memory"])
    rows = [json.loads(line) for line in (root/"training.jsonl").read_text().splitlines()]
    if len(rows) != len(protocol["phases_rad"])*round(protocol["duration_s"]/protocol["timestep_s"]):
        raise ValueError("incomplete calibration transitions")
    refit = fit_memory(rows, expected_contract=memory.expected_contract)
    import numpy as np
    for key in ("stiffness_s2", "damping_s", "bias_m_s2", "max_acceleration_m_s2",
                "acceleration_rmse_m_s2", "acceleration_error_bound_m_s2", "sample_count"):
        if not np.isclose(getattr(refit, key), getattr(memory, key), rtol=1e-8, atol=1e-10):
            raise ValueError("memory does not match calibration refit")
    brakes = candidate["braking_trials"]
    if ([r["speed_m_s"] for r in brakes] != protocol["braking_speeds_m_s"]
            or any(r["rest_s"] <= r["brake_applied_s"] or r["dropped"] for r in brakes)):
        raise ValueError("invalid braking calibration")
    derived = .5*min(r["initial_speed_m_s"]**2/(2*r["forward_excursion_m"]) for r in brakes)
    if (not math.isclose(derived, candidate["empirical_deceleration_m_s2"], rel_tol=1e-10)
            or candidate["training_protocol_sha256"] != digest(protocol)
            or candidate["training_sha256"] != hashlib.sha256((root/"training.jsonl").read_bytes()).hexdigest()
            or candidate["source_sha256"] != json.loads((root/"sources.json").read_text())):
        raise ValueError("inconsistent training identity")
    return candidate


def make_protocol(candidate, repeats=3, seed=20261010, *, development_only=False):
    if type(repeats) is not int or not 1 <= repeats <= 20 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("bounded integer repeats and 32-bit seed required")
    if type(development_only) is not bool:
        raise ValueError("development_only must be boolean")
    cases = [
        {"id": "dev_short", "split": "development", "goal_m": .08},
        {"id": "dev_mid", "split": "development", "goal_m": .16},
        {"id": "dev_goal_update", "split": "development", "goal_m": .12, "goal_update_s": .45, "updated_goal_m": .18},
        {"id": "dev_load", "split": "development", "goal_m": .18, "payload_mass_kg": .08, "pair_friction": .45},
        {"id": "dev_delay", "split": "development", "goal_m": .16, "sensor_delay_s": .02, "position_noise_m": .0004},
        {"id": "dev_missing", "split": "development", "goal_m": .18, "missing_start_s": .4, "missing_duration_s": .05, "expected_completion": False},
        {"id": "confirm_goal", "split": "confirmation", "goal_m": .215},
        {"id": "confirm_goal_update", "split": "confirmation", "goal_m": .23, "goal_update_s": .7, "updated_goal_m": .105},
        {"id": "confirm_load", "split": "confirmation", "goal_m": .195, "payload_mass_kg": .11, "pair_friction": .32},
        {"id": "confirm_delay", "split": "confirmation", "goal_m": .175, "sensor_delay_s": .04, "position_noise_m": .001},
        {"id": "confirm_blackout", "split": "confirmation", "goal_m": .22, "missing_start_s": .35, "missing_duration_s": .3, "expected_completion": False},
        {"id": "confirm_slip", "split": "confirmation", "goal_m": .20, "pull_start_s": .6, "pull_duration_s": .08, "pull_n": 12., "expected_completion": False},
    ]
    for case in cases:
        case.setdefault("holding", True)
        case.setdefault("expected_completion", True)
    if development_only:
        cases = [c for c in cases if c["split"] == "development"]
    return {"format": "dvidia.motor-protocol", "schema_version": 1, "framework_version": VERSION,
        "seed": seed, "repeats": repeats, "variants": list(MODES), "speed_limits_m_s": [.12, .20, .28],
        "cases": cases, "development_only": development_only, "candidate_sha256": digest(candidate),
        "control_period_s": .01, "sensor_period_s": .01, "score_period_s": .01, "sensor_prime_s": .15,
        "fixture_config": {"timestep_s": .001, "warmup_s": .4, "horizon_s": 4., "speed_m_s": .12,
            "carriage_mass_kg": 1., "actuator_force_limit_n": 15., "payload_mass_kg": .04,
            "jaw_force_limit_n": 8., "pair_friction": .7, "rest_speed_m_s": .005,
            "rest_dwell_s": .05, "command_delay_s": .02},
        "controller_config": asdict(ControllerConfig(workspace_min_m=-.05,
            braking_deceleration_m_s2=candidate["empirical_deceleration_m_s2"])),
        "completion": {"position_tolerance_m": .003, "speed_tolerance_m_s": .005, "dwell_s": .05},
        "contact_diagnostics": {"maximum_penetration_m": .003, "maximum_force_proxy_n": 1000.},
        "reaction_config": asdict(ReactionConfig(max_sensor_age_s=.15, command_delay_s=.02,
            contact_force_limit_n=10., slip_speed_limit_m_s=.03)),
        "scope": "synthetic one-axis approach, settle, retain; learned dynamics with authored primitive; no RGB/whole-arm/hardware qualification",
        "confirmation_scope": "authored combinations excluded from calibration/development; frozen before execution; once inspected these become regression cases",
        "uncertainty_scope": "bounded authored position noise plus calibration residual envelope; not calibrated confidence",
        "completion_scope": "observed position/speed dwell; final native audit and retention separate; missing input and goal changes reset verification",
        "jerk_scope": "finite differences of native velocity at 10 ms scoring cadence, not a native-tick jerk bound",
        "halt_scope": "latched delayed measured-position hold; no automatic resume/regrip recovery"}


def _goal(case, t):
    return case.get("updated_goal_m", case["goal_m"]) if t+1e-12 >= case.get("goal_update_s", math.inf) else case["goal_m"]


def score_trace(trace, protocol):
    """Recompute task outcomes from stored assessment receipts, no native import."""
    if not trace:
        return {"success": False, "completed_s": None, "truth_settled_s": None,
            "completion_false": False, "completion_revoked": False, "completion_claims": [],
            "rest_lost_after_confirmation": False, "final_error_m": None,
            "error_at_completion_m": None, "speed_at_completion_m_s": None,
            "peak_error_after_completion_m": None,
            "final_speed_m_s": None, "overshoot_m": None, "rms_error_m": None,
            "peak_acceleration_m_s2": None, "peak_jerk_m_s3": None, "dropped": None,
            "peak_force_proxy_n": None, "contact_impulse_ns": None, "max_penetration_m": None,
            "sensor_unusable_s": None, "predictor_rmse_m": None}
    dt = protocol["score_period_s"]
    tolerance = protocol["completion"]
    quiet = truth_quiet = 0
    completed = settled = None
    completion_error = completion_speed = peak_completion_error = None
    false = revoked = rest_lost = False
    claims = []
    errors, predictions, accelerations, jerks = [], [], [], []
    previous_goal = trace[0]["goal_m"]
    previous_v = 0.
    previous_a = None
    direction = 1. if previous_goal >= trace[0]["position_m"] else -1.
    overshoot = 0.
    for index, row in enumerate(trace):
        for key in ("time_s", "position_m", "velocity_m_s", "goal_m", "interval_force_peak_n",
                    "interval_impulse_ns", "interval_penetration_m"):
            if isinstance(row.get(key), bool) or not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                raise ValueError("scoring trace must contain finite measurements")
        if type(row.get("sensor_usable")) is not bool or type(row.get("dropped")) is not bool:
            raise ValueError("trace outcome/availability must be boolean")
        for key in ("interval_force_peak_n", "interval_impulse_ns", "interval_penetration_m"):
            if row[key] < 0:
                raise ValueError("contact proxy statistics must be nonnegative")
        for key in ("observed_position_m", "observed_velocity_m_s", "observed_acquired_s",
                    "observed_delivered_s", "estimated_position_m", "estimate_truth_position_m", "estimate_time_s"):
            value = row.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("optional sensing/estimate receipts must be finite or null")
        if (row.get("observed_position_m") is None) != (row.get("observed_velocity_m_s") is None):
            raise ValueError("observed pose and velocity must share availability")
        if (row.get("observed_position_m") is None) != (row.get("observed_acquired_s") is None) or (row.get("observed_acquired_s") is None) != (row.get("observed_delivered_s") is None):
            raise ValueError("observed measurements require acquisition and delivery receipts")
        if row["sensor_usable"] and row.get("observed_position_m") is None:
            raise ValueError("missing motion cannot be marked usable")
        if row.get("observed_acquired_s") is not None:
            if (row.get("observed_delivered_s") is None or row["observed_acquired_s"] > row["observed_delivered_s"]+1e-12
                    or row["observed_delivered_s"] > row["time_s"]+1e-12):
                raise ValueError("sensor timestamp chain must be causal")
        if row.get("estimated_position_m") is not None:
            if row.get("estimate_truth_position_m") is None or row.get("estimate_time_s") is None or row["estimate_time_s"] > row["time_s"]+1e-12:
                raise ValueError("state estimate must have causal same-instant audit truth")
        expected_t = (index+1)*dt
        if not math.isclose(row["time_s"], expected_t, abs_tol=1e-9):
            raise ValueError("incomplete or unordered scoring trace")
        if row["goal_m"] != previous_goal:
            revoked |= completed is not None
            completed = settled = None
            completion_error = completion_speed = peak_completion_error = None
            quiet = truth_quiet = 0
            direction = 1. if row["goal_m"] >= row["position_m"] else -1.
        error = row["goal_m"]-row["position_m"]
        errors.append(error)
        overshoot = max(overshoot, -direction*error)
        good_truth = abs(error) <= tolerance["position_tolerance_m"] and abs(row["velocity_m_s"]) <= tolerance["speed_tolerance_m_s"]
        truth_quiet = truth_quiet+1 if good_truth else 0
        if truth_quiet*dt+1e-12 >= tolerance["dwell_s"] and settled is None:
            settled = row["time_s"]
        if not good_truth and completed is not None:
            revoked = True
        if not good_truth and settled is not None:
            rest_lost = True
        good_observation = (row["observed_position_m"] is not None and row["sensor_usable"]
            and abs(row["goal_m"]-row["observed_position_m"]) <= tolerance["position_tolerance_m"]
            and abs(row["observed_velocity_m_s"]) <= tolerance["speed_tolerance_m_s"])
        quiet = quiet+1 if good_observation else 0
        if quiet*dt+1e-12 >= tolerance["dwell_s"] and completed is None:
            completed = row["time_s"]
            claim_false = truth_quiet*dt+1e-12 < tolerance["dwell_s"]
            false |= claim_false
            completion_error, completion_speed = abs(error), abs(row["velocity_m_s"])
            peak_completion_error = completion_error
            claims.append({"time_s": completed, "goal_m": row["goal_m"], "false": claim_false,
                "truth_error_m": completion_error, "truth_speed_m_s": completion_speed})
        if completed is not None:
            peak_completion_error = max(peak_completion_error, abs(error))
        if row["estimated_position_m"] is not None:
            predictions.append((row["estimated_position_m"]-row["estimate_truth_position_m"])**2)
        acceleration = (row["velocity_m_s"]-previous_v)/dt
        accelerations.append(acceleration)
        if previous_a is not None:
            jerks.append((acceleration-previous_a)/dt)
        previous_v, previous_a, previous_goal = row["velocity_m_s"], acceleration, row["goal_m"]
    dropped = any(row["dropped"] for row in trace)
    force = max(row["interval_force_peak_n"] for row in trace)
    penetration = max(row["interval_penetration_m"] for row in trace)
    contact_pass = force <= protocol["contact_diagnostics"]["maximum_force_proxy_n"] and penetration <= protocol["contact_diagnostics"]["maximum_penetration_m"]
    return {"success": truth_quiet*dt+1e-12 >= tolerance["dwell_s"] and not dropped and contact_pass,
        "completed_s": completed, "truth_settled_s": settled, "completion_false": false,
        "completion_revoked": revoked, "completion_claims": claims,
        "rest_lost_after_confirmation": rest_lost, "final_error_m": abs(errors[-1]),
        "error_at_completion_m": completion_error, "speed_at_completion_m_s": completion_speed,
        "peak_error_after_completion_m": peak_completion_error,
        "final_speed_m_s": abs(trace[-1]["velocity_m_s"]), "overshoot_m": max(0., overshoot),
        "rms_error_m": math.sqrt(statistics.mean(e*e for e in errors)),
        "peak_acceleration_m_s2": max(abs(a) for a in accelerations),
        "peak_jerk_m_s3": max(abs(j) for j in jerks) if jerks else None,
        "dropped": dropped, "peak_force_proxy_n": force,
        "contact_impulse_ns": sum(row["interval_impulse_ns"] for row in trace),
        "max_penetration_m": penetration,
        "sensor_unusable_s": sum(not row["sensor_usable"] for row in trace)*dt,
        "predictor_rmse_m": math.sqrt(statistics.mean(predictions)) if predictions else None}


def run_trial(protocol, candidate, case, variant, speed, seed):
    from .reaction_native import FixtureConfig, NativeReactionFixture
    fc = dict(protocol["fixture_config"])
    for field in ("payload_mass_kg", "pair_friction"):
        if field in case:
            fc[field] = case[field]
    c = FixtureConfig(**fc)
    cc = dict(protocol["controller_config"], max_speed_m_s=speed)
    controller = MotorController(MotorMemory.from_dict(candidate["memory"]), variant, ControllerConfig(**cc))
    monitor = ReactionMonitor(ReactionConfig(**protocol["reaction_config"]))
    delay = case.get("sensor_delay_s", 0.)
    noise = case.get("position_noise_m", 0.)
    streams = {name: SensorStream(SensorConfig(period_s=.01, delivery_delay_s=delay,
        position_noise_m=noise if name == "motion" else 0.), seed=seed+i)
        for i, name in enumerate(("motion", "contact", "grip"))}
    start = time.perf_counter()
    record = {"case_id": case["id"], "split": case["split"], "variant": variant,
        "speed_limit_m_s": speed, "seed": seed, "holding": case["holding"],
        "expected_completion": case["expected_completion"], "valid": False, "error": None,
        "target_final_m": case.get("updated_goal_m", case["goal_m"]), "trace": [],
        "reaction_s": None, "reaction_reason": None, "hold_applied_s": None,
        "command_latency_s": c.command_delay_s, "simulated_seconds": 0., "step_count": 0}
    record["controller_config"] = cc
    try:
        fixture = NativeReactionFixture(c, holding=case["holding"])
        record["fixture"] = fixture.identity()
        observations, last_signals, delivered_at = {}, {}, {}
        prime = protocol["sensor_prime_s"]
        def observe(now, t, truth):
            missing = case.get("missing_start_s", math.inf) <= t < case.get("missing_start_s", math.inf)+case.get("missing_duration_s", 0.)
            samples = {"motion": None if missing else MotionSample(tuple(RobotSphere(**shape) for shape in truth["robot_shapes"]), ()),
                "contact": ContactSample(truth["unexpected_force_n"]),
                "grip": GripSample(min(truth["pad_contacts"].values()) > 0, truth["slip_speed_m_s"])}
            for name, stream in streams.items():
                signal = stream.update(now, samples[name])
                if signal is not last_signals.get(name):
                    delivered_at[name] = now
                last_signals[name] = observations[name] = signal
        with fixture.native():
            for index in range(round(prime/c.timestep_s)+1):
                now = index*c.timestep_s
                observe(now, now-prime, fixture.measure())
                if index < round(prime/c.timestep_s):
                    fixture.tick(0., (.3, 1., .4))
            monitor.reset()
            history = [AppliedTarget(0., 0.)]
            pending = deque()
            applied_target = 0.
            hold_target = None
            hold_requested_at = None
            command = None
            estimate_truth_position = estimate_time = None
            latencies = []
            interval_force = interval_impulse = interval_penetration = 0.
            truth = fixture.measure()
            loop_start = time.perf_counter()
            control_ticks = round(protocol["control_period_s"]/c.timestep_s)
            for index in range(round(c.horizon_s/c.timestep_s)):
                t = index*c.timestep_s
                now = prime+t
                observe(now, t, truth)
                # Acknowledged target history reflects application before any new decision.
                while pending and pending[0][0] <= now+1e-12:
                    at, target = pending.popleft()
                    if hold_target is None:
                        applied_target = target
                        history.append(AppliedTarget(at, target))
                signal = observations.get("motion")
                obs = None if signal is None else MotorObservation(signal.acquired_s,
                    delivered_at["motion"], signal.value.robot_spheres[0].center_m[0],
                    signal.value.robot_spheres[0].velocity_m_s[0], noise)
                decision = monitor.update(now, observations, holding=case["holding"])
                if index % control_ticks == 0 and hold_requested_at is None:
                    before = time.perf_counter()
                    command = controller.update(now, obs, _goal(case, t), history)
                    estimate_truth_position, estimate_time = truth["position_m"][0], t
                    latencies.append(time.perf_counter()-before)
                    if command.halt:
                        hold_requested_at = now
                        record.update(reaction_s=t, reaction_reason="controller: "+command.reason)
                    else:
                        pending.append((now+c.command_delay_s, command.target_m))
                if decision.stop and hold_requested_at is None:
                    hold_requested_at = now
                    record.update(reaction_s=t, reaction_reason=decision.reason)
                if hold_requested_at is not None and hold_target is None and now+1e-12 >= hold_requested_at+c.command_delay_s:
                    hold_target = truth["position_m"][0]
                    applied_target = hold_target
                    pending.clear()
                    history.append(AppliedTarget(now, hold_target))
                    record["hold_applied_s"] = t
                pull = case.get("pull_n", 0.) if case.get("pull_start_s", math.inf) <= t < case.get("pull_start_s", math.inf)+case.get("pull_duration_s", 0.) else 0.
                fixture.tick(applied_target, (.3, 1., .4), payload_pull_n=pull)
                truth = fixture.measure()
                interval_force = max(interval_force, truth["unexpected_force_n"])
                interval_impulse += truth["unexpected_force_n"]*c.timestep_s
                interval_penetration = max(interval_penetration, truth["penetration_m"])
                record["step_count"] += 1
                record["simulated_seconds"] = (index+1)*c.timestep_s
                if (index+1) % control_ticks == 0:
                    assessment_now = prime+(index+1)*c.timestep_s
                    observe(assessment_now, (index+1)*c.timestep_s, truth)
                    signal = observations.get("motion")
                    required = ("motion", "contact", "grip") if case["holding"] else ("motion", "contact")
                    usable = all(observations.get(name) is not None and assessment_now-observations[name].acquired_s <= .15+1e-12 for name in required)
                    # Audit against the same pre-step instant used by the controller.
                    estimate = None if command is None else command.estimated_state
                    predicted = estimate.position_m if estimate is not None and hold_requested_at is None else None
                    record["trace"].append({"time_s": (index+1)*c.timestep_s,
                        "position_m": truth["position_m"][0], "velocity_m_s": truth["velocity_m_s"][0],
                        "goal_m": _goal(case, (index+1)*c.timestep_s),
                        "observed_position_m": None if signal is None else signal.value.robot_spheres[0].center_m[0],
                        "observed_velocity_m_s": None if signal is None else signal.value.robot_spheres[0].velocity_m_s[0],
                        "observed_acquired_s": None if signal is None else signal.acquired_s-prime,
                        "observed_delivered_s": None if signal is None else delivered_at["motion"]-prime,
                        "sensor_usable": usable, "estimated_position_m": predicted,
                        "estimate_time_s": estimate_time if predicted is not None else None,
                        "estimate_truth_position_m": estimate_truth_position if predicted is not None else None,
                        "applied_target_m": applied_target, "dropped": truth["dropped"],
                        "interval_force_peak_n": interval_force, "interval_impulse_ns": interval_impulse,
                        "interval_penetration_m": interval_penetration})
                    interval_force = interval_impulse = interval_penetration = 0.
            record["loop_wall_seconds"] = time.perf_counter()-loop_start
        record["valid"] = True
        record["warnings"] = list(fixture.warnings)
        record.update(score_trace(record["trace"], protocol))
        record["controller_latency_p50_s"] = statistics.median(latencies) if latencies else None
        record["controller_latency_p95_s"] = sorted(latencies)[math.ceil(.95*len(latencies))-1] if latencies else None
        record["controller_latency_max_s"] = max(latencies) if latencies else None
        record["controller_call_latencies_s"] = latencies
        record["controller_call_count"] = len(latencies)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record.update(score_trace(record["trace"], protocol))
        record["success"] = False
    record["wall_seconds"] = time.perf_counter()-start
    return record


def benchmark(memory_directory, directory, *, repeats=3, seed=20261010, development_only=False):
    candidate = inspect_memory(memory_directory)
    protocol = make_protocol(candidate, repeats, seed, development_only=development_only)
    directory = _new_directory(directory)
    sources = _sources()
    _write(directory/"memory.json", candidate)
    _write(directory/"protocol.json", protocol)
    (directory/"protocol.sha256").write_text(digest(protocol)+"\n")
    _write(directory/"sources.json", sources)
    # Candidate and complete attempted matrix are persisted before opening any case.
    freeze = {"candidate_sha256": digest(candidate), "protocol_sha256": digest(protocol),
        "source_sha256": sources, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["confirmation_scope"]}
    _write(directory/"freeze.json", freeze)
    records = []
    start = time.perf_counter()
    with (directory/"trials.jsonl").open("w", encoding="utf-8") as handle:
        for repeat in range(repeats):
            for index, case in enumerate(protocol["cases"]):
                for tier, speed in enumerate(protocol["speed_limits_m_s"]):
                    order = list(MODES)
                    offset = (repeat+index+tier) % len(order)
                    order = order[offset:]+order[:offset]
                    for variant in order:
                        record = run_trial(protocol, candidate, case, variant, speed, seed+repeat*1000+index*10+tier)
                        records.append(record)
                        handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False)+"\n")
                        handle.flush()
    import mujoco
    import numpy as np
    report = {"format": "dvidia.motor-benchmark", "schema_version": 1, "framework_version": VERSION,
        "scope": protocol["scope"], "physical_robot_ready": False, "benchmark_qualification": False,
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "protocol_sha256": digest(protocol), "candidate": candidate, "candidate_sha256": digest(candidate),
        "source_sha256": sources, "freeze": freeze, "records": records, "summary": summarize(records),
        "host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(), "gpu_used": False},
        "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        "benchmark_wall_seconds": time.perf_counter()-start,
        "timing_scope": "per-trial includes fixture/warmup/priming, control, native physics, scoring; benchmark includes JSONL write; excludes imports, calibration and final report serialization",
        "limitations": ["Learned component is fitted actuator dynamics; task motion/completion/contact contract authored.",
            "Synthetic state observations, one travel axis and parallel jaws; no camera, whole-arm geometry or video-to-policy training.",
            "Braking and uncertainty envelopes are empirical development assumptions, not calibrated hardware confidence/safety bounds.",
            "Impact-force convergence was unresolved in v0.1; these contact proxies do not establish physical accuracy.",
            "Repeated authored combinations are correlated; confirmation becomes regression evidence after inspection.",
            "Latched hold does not regrip, resume or recover a dropped load; failures and false completion claims remain visible."]}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["host"]["lifetime_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss*1024
        report["host"]["rss_scope"] = "process lifetime including imports and all trials; not hardware minimum/per-trial requirement"
    except ImportError:
        report["host"]["lifetime_peak_rss_bytes"] = None
    _write(directory/"report.json", report)
    (directory/"report.html").write_text(render_report(report), encoding="utf-8")
    _manifest(directory)
    return report


def inspect(directory):
    root = _check_files(directory, {"protocol.json", "protocol.sha256", "sources.json", "memory.json", "freeze.json", "trials.jsonl", "report.json", "report.html"})
    protocol = json.loads((root/"protocol.json").read_text())
    candidate = json.loads((root/"memory.json").read_text())
    report = json.loads((root/"report.json").read_text())
    records = [json.loads(line) for line in (root/"trials.jsonl").read_text().splitlines()]
    sources = json.loads((root/"sources.json").read_text())
    freeze = json.loads((root/"freeze.json").read_text())
    MotorMemory.from_dict(candidate["memory"])
    if (candidate.get("format") != "dvidia.motor-memory-candidate" or candidate.get("schema_version") != 1
            or candidate.get("framework_version") != VERSION or candidate.get("physical_robot_ready") is not False
            or candidate.get("task_motion_learned_from_video") is not False
            or protocol != make_protocol(candidate, protocol["repeats"], protocol["seed"], development_only=protocol["development_only"])):
        raise ValueError("unsupported frozen candidate/protocol contract")
    if (report.get("format") != "dvidia.motor-benchmark" or report.get("framework_version") != VERSION
            or report.get("schema_version") != 1 or report.get("physical_robot_ready") is not False
            or report.get("benchmark_qualification") is not False):
        raise ValueError("unsupported benchmark scope/readiness")
    expected = Counter((c["id"], mode, speed, protocol["seed"]+repeat*1000+i*10+tier)
        for repeat in range(protocol["repeats"]) for i, c in enumerate(protocol["cases"])
        for tier, speed in enumerate(protocol["speed_limits_m_s"]) for mode in protocol["variants"])
    actual = Counter((r["case_id"], r["variant"], r["speed_limit_m_s"], r["seed"]) for r in records)
    if expected != actual:
        raise ValueError("missing, duplicate or inconsistent planned attempts")
    cases = {c["id"]: c for c in protocol["cases"]}
    for record in records:
        case = cases[record["case_id"]]
        if (record["split"] != case["split"] or record["expected_completion"] is not case["expected_completion"]
                or record["holding"] is not case["holding"] or record["target_final_m"] != case.get("updated_goal_m", case["goal_m"])):
            raise ValueError("trial task contract differs from protocol")
        if any(row["goal_m"] != _goal(case, row["time_s"]) for row in record["trace"]):
            raise ValueError("trace goal differs from public task schedule")
        score = score_trace(record["trace"], protocol)
        if not record["valid"]:
            score["success"] = False
        if any(record.get(key) != value for key, value in score.items()):
            raise ValueError("trial measurements differ from scoring trace")
        if record["valid"] and (len(record["trace"]) != round(protocol["fixture_config"]["horizon_s"]/protocol["score_period_s"])
                or record["step_count"] != round(protocol["fixture_config"]["horizon_s"]/protocol["fixture_config"]["timestep_s"])
                or record["simulated_seconds"] != protocol["fixture_config"]["horizon_s"]):
            raise ValueError("valid trial omitted planned physics/assessment ticks")
        if record["valid"]:
            if record["error"] is not None or record["controller_config"] != dict(protocol["controller_config"], max_speed_m_s=record["speed_limit_m_s"]):
                raise ValueError("valid trial differs from controller/runtime contract")
            latencies = record["controller_call_latencies_s"]
            if (record["controller_call_count"] != len(latencies) or not latencies
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in latencies)
                    or record["controller_latency_p50_s"] != statistics.median(latencies)
                    or record["controller_latency_p95_s"] != sorted(latencies)[math.ceil(.95*len(latencies))-1]
                    or record["controller_latency_max_s"] != max(latencies)):
                raise ValueError("inconsistent controller timing receipts")
    if (report.get("records") != records or report.get("summary") != summarize(records)
            or report.get("protocol") != protocol or report.get("protocol_sha256") != digest(protocol)
            or (root/"protocol.sha256").read_text().strip() != digest(protocol)
            or report.get("candidate") != candidate or report.get("candidate_sha256") != digest(candidate)
            or protocol["candidate_sha256"] != digest(candidate) or report.get("source_sha256") != sources
            or report.get("freeze") != freeze or freeze.get("candidate_sha256") != digest(candidate)
            or freeze.get("protocol_sha256") != digest(protocol) or freeze.get("source_sha256") != sources):
        raise ValueError("inconsistent benchmark identities or recomputed measurements")
    return {"consistent": True, "attempted": len(records), "invalid": report["summary"]["invalid"],
        "physical_robot_ready": False, "identity_scope": "self-consistent receipts, not authenticated provenance/qualification"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Motor memory v0.2: CPU native calibration and speed–precision benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    learn = commands.add_parser("train", help="fit and freeze a compact native actuator memory")
    learn.add_argument("--output", required=True)
    learn.add_argument("--seed", type=int, default=20261009)
    run = commands.add_parser("benchmark", help="compare three controllers and fresh authored combinations")
    run.add_argument("--memory", required=True, help="training evidence directory")
    run.add_argument("--output", required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--seed", type=int, default=20261010)
    run.add_argument("--development-only", action="store_true")
    check = commands.add_parser("inspect", help="recompute evidence without importing native physics")
    check.add_argument("directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            directory = Path(args.directory)
            result = inspect(directory) if (directory/"report.json").exists() else {"consistent": True, "candidate_sha256": digest(inspect_memory(directory)), "physical_robot_ready": False}
        elif args.command == "train":
            candidate = train(args.output, seed=args.seed)
            result = {"memory": str(Path(args.output).resolve()/"memory.json"), "samples": candidate["memory"]["sample_count"], "physical_robot_ready": False}
        else:
            report = benchmark(args.memory, args.output, repeats=args.repeats, seed=args.seed, development_only=args.development_only)
            result = {"report": str(Path(args.output).resolve()/"report.html"), "attempted": report["summary"]["attempted"], "invalid": report["summary"]["invalid"], "physical_robot_ready": False}
            if result["invalid"]:
                print(json.dumps(result, indent=2))
                return 1
    except ImportError as exc:
        parser.exit(2, f"Cannot run native motor training: {exc}\nInstall the optional arm extra.\n")
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, f"Cannot process motor evidence: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
