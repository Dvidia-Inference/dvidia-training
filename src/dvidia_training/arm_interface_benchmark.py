"""Bounded, unloaded native CPU tracking checks for explicit arm interfaces.

This is an unpaced simulator microbenchmark. It records privileged state and
authored setpoints; it does not measure a sensor, bus, physical actuator or task.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time

SCHEMA = "dvidia.arm-interface-benchmark/v1"
PROFILES = ("piper", "so101", "ur5e-2f85")
CASES = ("hold", "arm_perturbation", "gripper_cycle")
PERIODS = (.01, .02, .04)
TIMESTEP = .001
SCOPE = ("Unloaded native CPU tracking of authored setpoints using pinned upstream mechanics. "
         "Privileged simulator state; unpaced offline computation. No object task, learned policy, "
         "hardware latency, calibrated jaw gap/force or physical robot qualification.")


def _interfaces():
    from . import arm_interfaces
    return arm_interfaces


def _settings(profiles, duration, repeats):
    if not isinstance(profiles, (list, tuple)) or not 1 <= len(profiles) <= 3:
        raise ValueError("profiles must contain one to three known unique interfaces")
    if any(not isinstance(p, str) or p not in PROFILES for p in profiles) or len(set(profiles)) != len(profiles):
        raise ValueError("profiles must contain one to three known unique interfaces")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not .4 <= duration <= 2:
        raise ValueError("duration must be finite and within [0.4,2] seconds")
    if not math.isclose(duration/.04, round(duration/.04), rel_tol=0, abs_tol=1e-9):
        raise ValueError("duration must contain complete 40 ms periods")
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("repeats must be an integer within [1,3]")
    return {"profiles": list(profiles), "duration_s": float(duration), "repeats": repeats,
            "cases": list(CASES), "control_periods_s": list(PERIODS), "physics_timestep_s": TIMESTEP,
            "arm_amplitude_limit_rad": .04, "gripper_cycle_fraction_of_range": .2,
            "waveforms": {"arm": "home + signed amplitude*sin(2*pi*t/duration)",
                          "gripper": "home + amplitude*sin(pi*t/duration)^2"},
            "command_sampling": "At interval start; held until the next control boundary. No terminal command after the horizon.",
            "accuracy": "Every post-step native millisecond, against both the applied held target and continuous reference at observation time.",
            "pacing": "Unpaced offline; budget exceedances compare computation with nominal control periods only."}


def _json(value, *, compact=False):
    return (json.dumps(value, allow_nan=False, sort_keys=True,
                       indent=None if compact else 2,
                       separators=(",", ":") if compact else None)+"\n").encode()


def _write(path, value, *, compact=False):
    content = _json(value, compact=compact)
    with Path(path).open("xb") as stream:
        stream.write(content)
    return {"bytes": len(content), "sha256": sha256(content).hexdigest()}


def _source_snapshot(output):
    root = Path(__file__).parent
    names = ("arm_interface_benchmark.py", "arm_interfaces.py", "asset_library.py", "asset_catalog.json",
             "reference_sim.py", "reference_assets.lock.json", "robot_profiles.py", "robot_profiles.lock.json", "env.py")
    source = output/"source"
    source.mkdir()
    inventory = []
    for name in names:
        raw = (root/name).read_bytes()
        with (source/name).open("xb") as stream:
            stream.write(raw)
        inventory.append({"name": name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()})
    return inventory


def _check_sources(inventory):
    root = Path(__file__).parent
    if any(sha256((root/row["name"]).read_bytes()).hexdigest() != row["sha256"] for row in inventory):
        raise RuntimeError("Implementation changed after the benchmark protocol was frozen")


def _percentiles(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    def quantile(p):
        rank = (len(ordered)-1)*p
        low, high = math.floor(rank), math.ceil(rank)
        return ordered[low]+(ordered[high]-ordered[low])*(rank-low)
    return {"count": len(ordered), "min": ordered[0], "p50": quantile(.5),
            "p95": quantile(.95), "p99": quantile(.99), "max": ordered[-1], "mean": sum(ordered)/len(ordered)}


def _amplitudes(home, limits, arm_indices, gripper_indices):
    amplitudes = [0.]*len(home)
    for index in arm_indices:
        low, high = limits[index]
        room = min(home[index]-low, high-home[index])
        if room < 0:
            raise ValueError("Home command exceeds interface limits")
        amplitudes[index] = min(.04, .8*room)*(-1 if index % 2 else 1)
    for index in gripper_indices:
        low, high = limits[index]
        if not low <= home[index] <= high:
            raise ValueError("Home gripper command exceeds interface limits")
        amplitudes[index] = min(.2*(high-low), .8*(high-home[index]))
    return amplitudes


def _command(case, elapsed, duration, home, amplitudes, arm_indices, gripper_indices):
    if case not in CASES:
        raise ValueError("Unknown tracking case")
    result = list(home)
    indices = arm_indices if case == "arm_perturbation" else gripper_indices if case == "gripper_cycle" else ()
    phase = math.sin(2*math.pi*elapsed/duration) if case == "arm_perturbation" else math.sin(math.pi*elapsed/duration)**2
    for index in indices:
        result[index] += amplitudes[index]*phase
    return result


def _servo_equivalent(length, gain, bias0, bias1):
    """Static position-servo coordinate in the native command's own units.

    Affine actuator force = gain*ctrl + bias0 + bias1*length + damping*velocity.
    This is a compiled-model coordinate transformation, not a jaw-gap sensor.
    """
    if not all(math.isfinite(x) for x in (length, gain, bias0, bias1)) or gain <= 0 or bias1 >= 0:
        raise ValueError("Unsupported or nonfinite position-servo mapping")
    return -(bias0+bias1*length)/gain


def _error_summary(values, unit):
    if not values:
        return {"unit": unit, "count": 0, "rms": None, "max_abs": None}
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Cannot summarize nonfinite errors")
    return {"unit": unit, "count": len(values), "rms": math.sqrt(sum(x*x for x in values)/len(values)),
            "max_abs": max(abs(x) for x in values)}


def _metrics(rows, gripper_unit):
    errors = [row["tracking_error"] for row in rows]
    return {"state_rows": len(rows),
        "arm_held_target": _error_summary([v for e in errors for v in e["arm_held_target_rad"]], "rad"),
        "arm_continuous_reference": _error_summary([v for e in errors for v in e["arm_continuous_reference_rad"]], "rad"),
        "gripper_held_target": _error_summary([e["gripper_held_target_native"] for e in errors], gripper_unit),
        "gripper_continuous_reference": _error_summary([e["gripper_continuous_reference_native"] for e in errors], gripper_unit),
        "terminal_position_continuous_reference": _error_summary([e["terminal_position_reference_m"] for e in errors], "m"),
        "terminal_orientation_continuous_reference": _error_summary([e["terminal_orientation_reference_rad"] for e in errors], "rad")}


def _snapshot(mj, np, model, data, reference_data, contract, arm_qpos, held, reference, elapsed, gripper):
    arrays = (data.qpos, data.qvel, data.qacc, data.xpos, data.xquat, data.actuator_length, data.actuator_force)
    if not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("Native state became nonfinite")
    arm_indices = contract["arm_action_indices"]
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, contract["tool_frame_body"])
    if body <= 0:
        raise ValueError("Interface terminal link is missing")
    reference_data.qpos[:] = data.qpos
    for action, qpos in zip(arm_indices, arm_qpos):
        reference_data.qpos[qpos] = reference[action]
    mj.mj_kinematics(model, reference_data)
    position_error = float(np.linalg.norm(data.xpos[body]-reference_data.xpos[body]))
    dot = abs(float(np.dot(data.xquat[body], reference_data.xquat[body])))
    orientation_error = 2*math.acos(min(1., max(0., dot)))
    jaw = gripper["action_index"]
    jaw_equivalent = _servo_equivalent(float(data.actuator_length[jaw]), gripper["gain"], gripper["bias0"], gripper["bias1"])
    arm = [float(data.qpos[q]) for q in arm_qpos]
    return {"elapsed_s": elapsed, "native_time_s": float(data.time),
        "observation_origin": "privileged_native_simulator_state", "applied_command_native": list(held),
        "continuous_reference_command_native": reference, "qpos_native": data.qpos.tolist(),
        "qvel_native_per_s": data.qvel.tolist(), "arm_position_rad": arm,
        "gripper_actuator_length_native": float(data.actuator_length[jaw]),
        "gripper_servo_equivalent_position_native": jaw_equivalent,
        "actuator_force_native": data.actuator_force.tolist(), "contact_count": int(data.ncon),
        "terminal_link_position_m": data.xpos[body].tolist(),
        "terminal_link_quaternion_wxyz": data.xquat[body].tolist(),
        "reference_terminal_link_position_m": reference_data.xpos[body].tolist(),
        "reference_terminal_link_quaternion_wxyz": reference_data.xquat[body].tolist(),
        "tracking_error": {
            "arm_held_target_rad": [value-held[action] for value, action in zip(arm, arm_indices)],
            "arm_continuous_reference_rad": [value-reference[action] for value, action in zip(arm, arm_indices)],
            "gripper_held_target_native": jaw_equivalent-held[jaw],
            "gripper_continuous_reference_native": jaw_equivalent-reference[jaw],
            "terminal_position_reference_m": position_error, "terminal_orientation_reference_rad": orientation_error}}


def _safe_state(data):
    if data is None:
        return None
    def numbers(values):
        return [float(v) if math.isfinite(float(v)) else repr(float(v)) for v in values]
    return {"native_time_s": numbers([data.time])[0], "qpos_native": numbers(data.qpos),
            "qvel_native_per_s": numbers(data.qvel), "last_command_native": numbers(data.ctrl)}


def _attempt(item, assets, folder, settings, inventory):
    from . import reference_sim as sim
    begun = time.perf_counter()
    result = {**item, "valid": False, "error": None, "warnings": [], "warning_counters": [],
              "metrics": {}, "timing": {}, "simulated_seconds": 0., "failure_state": None}
    _write(folder/"started.json", {"schema": SCHEMA, "attempt": item, "state": "started", "unix_ns": time.time_ns()})
    data, model, loop_started = None, None, None
    rows, commands, loop_times, step_times = [], [], [], []
    build_seconds, loop_seconds, stage = None, None, "source_check"
    try:
        _check_sources(inventory)
        stage = "build"
        with sim.native_warnings(result["warnings"]):
            build_started = time.perf_counter()
            model, data, contract = _interfaces().build(item["profile"], assets, timestep=TIMESTEP)
            build_seconds = time.perf_counter()-build_started
            mj, np = sim._native_imports()
            result["contract"] = contract
            arm_indices, jaw_indices = contract["arm_action_indices"], contract["gripper_action_indices"]
            if len(jaw_indices) != 1:
                raise ValueError("Benchmark requires one explicit coupled gripper action")
            if any(contract["action_units"][i] != "rad" for i in arm_indices):
                raise ValueError("Benchmark arm tracking requires angular commands")
            arm_qpos = []
            for action in arm_indices:
                if int(model.actuator_trntype[action]) != int(mj.mjtTrn.mjTRN_JOINT):
                    raise ValueError("Arm actuator must map to a direct scalar joint")
                joint = int(model.actuator_trnid[action, 0])
                if int(model.jnt_type[joint]) != int(mj.mjtJoint.mjJNT_HINGE) or not math.isclose(float(model.actuator_gear[action, 0]), 1., abs_tol=1e-12):
                    raise ValueError("Unsupported arm command coordinate")
                arm_qpos.append(int(model.jnt_qposadr[joint]))
            jaw = jaw_indices[0]
            if int(model.actuator_gaintype[jaw]) != int(mj.mjtGain.mjGAIN_FIXED) or int(model.actuator_biastype[jaw]) != int(mj.mjtBias.mjBIAS_AFFINE) or int(model.actuator_dyntype[jaw]) != int(mj.mjtDyn.mjDYN_NONE):
                raise ValueError("Unsupported gripper servo dynamics")
            gripper = {"action_index": jaw, "unit": contract["action_units"][jaw],
                       "gain": float(model.actuator_gainprm[jaw, 0]), "bias0": float(model.actuator_biasprm[jaw, 0]),
                       "bias1": float(model.actuator_biasprm[jaw, 1]), "damping_bias": float(model.actuator_biasprm[jaw, 2]),
                       "coordinate": "-(bias0+bias1*actuator_length)/gain; static servo-equivalent command coordinate, not jaw gap or calibrated force"}
            _servo_equivalent(0., gripper["gain"], gripper["bias0"], gripper["bias1"])
            home = data.ctrl.tolist()
            amplitudes = _amplitudes(home, model.actuator_ctrlrange.tolist(), arm_indices, jaw_indices)
            result.update(home_command_native=home, amplitude_native=amplitudes, gripper_coordinate=gripper,
                          native_model_sha256=sim.model_digest(model),
                          runtime={"mujoco": mj.__version__, "numpy": np.__version__},
                          model_settings={"gravity_m_s2": model.opt.gravity.tolist(),
                              "gravity_compensation_by_body": model.body_gravcomp.tolist(),
                              "integrator": int(model.opt.integrator), "timestep_s": float(model.opt.timestep)})
            reference_data = mj.MjData(model)
            held = list(home)
            rows.append(_snapshot(mj, np, model, data, reference_data, contract, arm_qpos, held, home, 0., gripper))
            if result["warnings"] or any(w.number for w in data.warning):
                raise RuntimeError("Native initialization emitted a warning")
            intervals, substeps = round(settings["duration_s"]/item["control_period_s"]), round(item["control_period_s"]/TIMESTEP)
            stage, loop_started = "control_loop", time.perf_counter()
            for interval in range(intervals):
                tick = time.perf_counter()
                elapsed = interval*item["control_period_s"]
                held = _interfaces().validate_command(contract, model, _command(item["case"], elapsed, settings["duration_s"], home, amplitudes, arm_indices, jaw_indices))
                commands.append({"elapsed_s": elapsed, "native_time_s": float(data.time), "values_native": held})
                data.ctrl[:] = held
                for offset in range(substeps):
                    step_started = time.perf_counter()
                    mj.mj_step(model, data)
                    mj.mj_forward(model, data)
                    actual_elapsed = (interval*substeps+offset+1)*TIMESTEP
                    reference = _command(item["case"], actual_elapsed, settings["duration_s"], home, amplitudes, arm_indices, jaw_indices)
                    rows.append(_snapshot(mj, np, model, data, reference_data, contract, arm_qpos, held, reference, actual_elapsed, gripper))
                    result["simulated_seconds"] = actual_elapsed
                    if result["warnings"] or any(w.number for w in data.warning):
                        raise RuntimeError("Native physics emitted a warning")
                    step_times.append(time.perf_counter()-step_started)
                loop_times.append(time.perf_counter()-tick)
            loop_seconds = time.perf_counter()-loop_started
            result["metrics"] = _metrics(rows[1:], gripper["unit"])
            result["valid"] = True
    except Exception as exc:
        if loop_started is not None and loop_seconds is None:
            loop_seconds = time.perf_counter()-loop_started
        result["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:1000]}
        result["failure_state"] = _safe_state(data)
    if data is not None:
        result["warning_counters"] = [{"index": i, "number": int(w.number), "lastinfo": int(w.lastinfo)} for i, w in enumerate(data.warning) if w.number]
    trace = {"schema": "dvidia.arm-interface-trace/v1", "attempt": item, "rows": rows, "commands": commands,
             "scope": SCOPE, "initial_row_in_accuracy": False, "terminal_link_scope": "Native link origin; no calibrated tool centre point"}
    trace_identity = _write(folder/"trace.json", trace, compact=True)
    result["trace"] = {"path": "trace.json", **trace_identity, "state_rows": len(rows), "commands": len(commands)}
    result["timing"] = {"compile_and_initialize_wall_s": build_seconds,
        "compile_timing_scope": "arm_interfaces.build, including pinned asset verification, compilation and initialization",
        "loop_wall_s": loop_seconds, "control_loop_compute_s": _percentiles(loop_times),
        "native_step_compute_s": _percentiles(step_times), "control_intervals_s": loop_times,
        "native_steps_s": step_times, "offline_control_budget_exceedances": sum(t > item["control_period_s"] for t in loop_times),
        "simulated_seconds_per_loop_wall_second": result["simulated_seconds"]/loop_seconds if loop_seconds else None,
        "total_before_result_write_wall_s": time.perf_counter()-begun,
        "scope": "Unpaced wall computation; loop includes native stepping, commands, snapshots and reference FK; excludes trace serialization. No hardware or sensor latency."}
    _write(folder/"result.json", result)
    return result


def _summary(attempts):
    groups = []
    for profile in dict.fromkeys(row["profile"] for row in attempts):
        for case in CASES:
            for period in PERIODS:
                selected = [a for a in attempts if (a["profile"], a["case"], a["control_period_s"]) == (profile, case, period)]
                valid = [a for a in selected if a["valid"]]
                metrics = {}
                for key in ("arm_continuous_reference", "gripper_continuous_reference", "terminal_position_continuous_reference", "terminal_orientation_continuous_reference"):
                    rows = [a["metrics"][key] for a in valid]
                    count = sum(r["count"] for r in rows)
                    metrics[key] = {"unit": rows[0]["unit"] if rows else None, "count": count,
                        "rms": math.sqrt(sum(r["rms"]**2*r["count"] for r in rows)/count) if count else None,
                        "max_abs": max((r["max_abs"] for r in rows), default=None)}
                groups.append({"profile": profile, "case": case, "control_period_s": period,
                    "valid_attempts": len(valid), "attempts": len(selected), "metrics": metrics,
                    "control_loop_compute_s": _percentiles([t for a in valid for t in a["timing"]["control_intervals_s"]]),
                    "offline_control_budget_exceedances": sum(a["timing"]["offline_control_budget_exceedances"] for a in valid)})
    return {"attempts": len(attempts), "valid_attempts": sum(a["valid"] for a in attempts),
            "failed_attempts": sum(not a["valid"] for a in attempts), "groups": groups,
            "selection": "All fixed roster attempts retained; no best-profile or period selection, tuning or automatic reruns."}


def _markdown(report):
    lines = ["# Arm interface CPU tracking v0.1", "", SCOPE, "",
        f"{report['summary']['valid_attempts']}/{report['summary']['attempts']} numerically valid attempts. All attempts, warnings, commands and millisecond state traces are retained.", "",
        "| Profile | Case | Update ms | Valid | Arm reference RMS rad | Gripper reference RMS (native unit) | Control compute p95 ms | Offline budget exceeds |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    def fmt(value):
        return "—" if value is None else f"{value:.6g}"
    for row in report["summary"]["groups"]:
        arm, jaw = row["metrics"]["arm_continuous_reference"], row["metrics"]["gripper_continuous_reference"]
        p95 = row["control_loop_compute_s"]["p95"]
        lines.append(f"| {row['profile']} | {row['case']} | {row['control_period_s']*1000:g} | {row['valid_attempts']}/{row['attempts']} | {fmt(arm['rms'])} | {fmt(jaw['rms'])} {jaw['unit'] or ''} | {fmt(p95*1000 if p95 is not None else None)} | {row['offline_control_budget_exceedances']} |")
    lines += ["", "Arm radians and each gripper's native coordinate are reported separately. The Robotiq 0–255 feedback is derived from its pinned affine servo model, not a measured aperture or force.", "",
        "Terminal-link reference errors use forward kinematics at the continuous desired arm configuration. They are native link-origin diagnostics, not calibrated end-effector accuracy.", "",
        "Compile timing includes asset verification and initialization. Loop timing includes commands, native stepping, millisecond snapshots and reference FK; trace serialization is outside the loop. Total timing stops before writing each result receipt.", "",
        "These unloaded authored trajectories do not demonstrate task success, learned skill transfer, physical accuracy or hard real-time control.", ""]
    return "\n".join(lines)


def benchmark(assets, output, *, profiles=PROFILES, duration=.4, repeats=2):
    """Run a fresh fixed roster; assets may be one root or a profile-to-root map."""
    settings = _settings(profiles, duration, repeats)
    if isinstance(assets, Mapping):
        if any(p not in assets for p in profiles) or any(p not in PROFILES for p in assets):
            raise ValueError("assets map must provide every selected known interface")
        roots = {profile: assets[profile] for profile in profiles}
    else:
        roots = dict.fromkeys(profiles, assets)
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Benchmark output must be a fresh directory")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    inventory = _source_snapshot(destination)
    schedule = [{"id": f"attempt-{index:03d}", "profile": profile, "case": case,
                 "control_period_s": period, "repeat": repeat}
                for index, (profile, case, period, repeat) in enumerate(
                    (p, c, dt, r) for p in profiles for c in CASES for dt in PERIODS for r in range(repeats))]
    protocol = {"schema": SCHEMA, "scope": SCOPE, "settings": settings, "source_inventory": inventory,
                "schedule": schedule, "platform": {"system": platform.system(), "machine": platform.machine(),
                                                    "python": platform.python_version()},
                "task_qualified": False, "physical_qualified": False,
                "failure_policy": "Each scheduled attempt gets a started receipt before construction. Errors/warnings are retained; no retries or hidden settling."}
    protocol_identity = _write(destination/"protocol.json", protocol)
    _write(destination/"execution-intent.json", {"schema": SCHEMA, "protocol": protocol_identity,
        "scheduled_attempts": len(schedule), "started_unix_ns": time.time_ns()})
    attempts = []
    for item in schedule:
        folder = destination/item["id"]
        folder.mkdir()
        attempts.append(_attempt(item, roots[item["profile"]], folder, settings, inventory))
    report = {"schema": SCHEMA, "scope": SCOPE, "protocol": protocol_identity, "settings": settings,
        "source_inventory": inventory, "platform": protocol["platform"], "attempts": attempts,
        "summary": _summary(attempts), "task_qualified": False, "physical_qualified": False,
        "total_to_report_write_wall_s": time.perf_counter()-started}
    report_identity = _write(destination/"report.json", report)
    with (destination/"README.md").open("x", encoding="utf-8") as stream:
        stream.write(_markdown(report))
    _write(destination/"completion.json", {"schema": SCHEMA, "report": report_identity,
        "scheduled_attempts": len(schedule), "recorded_attempts": len(attempts),
        "valid_attempts": report["summary"]["valid_attempts"], "state": "complete"})
    return report
