"""Offline, versioned reaction benchmarks with continued native braking physics."""
from __future__ import annotations

import argparse
from collections import Counter
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

from .reaction import (ContactSample, GripSample, MotionSample, ObstacleSphere,
                       ReactionConfig, ReactionMonitor, RobotSphere, SensorConfig, SensorStream)
from .reaction_metrics import render_report, summarize

FRAMEWORK_VERSION = "0.1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def make_protocol(repeats=3, seed=20261009, *, convergence=False):
    if type(convergence) is not bool:
        raise ValueError("convergence must be boolean")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 30:
        raise ValueError("repeats must be an integer in [1,30]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a 32-bit nonnegative integer")
    cases = [
        {"id": "clear", "group": "motion", "kind": "clear", "positive": False},
        {"id": "crossing", "group": "motion", "kind": "crossing", "positive": True, "onset_s": 0.},
        {"id": "crossing_delayed_noisy", "group": "motion", "kind": "crossing", "positive": True, "onset_s": 0., "sensor_delay_s": .04, "position_noise_m": .002},
        {"id": "static_obstacle", "group": "motion", "kind": "static", "positive": True, "onset_s": 0.},
        {"id": "near_miss", "group": "motion", "kind": "near_miss", "positive": False},
        {"id": "receding", "group": "motion", "kind": "receding", "positive": False},
        {"id": "late_occluded_entry", "group": "motion", "kind": "late", "positive": True, "onset_s": .45, "occluded_until_s": .45, "sensor_delay_s": .04},
        {"id": "high_force", "group": "contact", "kind": "force", "positive": True, "onset_s": .4, "push_n": 20.},
        {"id": "low_force", "group": "contact", "kind": "force", "positive": False, "push_n": 2.},
        {"id": "stable_grip", "group": "grip", "kind": "grip", "positive": False, "holding": True},
        {"id": "slip", "group": "grip", "kind": "grip", "positive": True, "holding": True, "onset_s": .4, "pull_n": 12.},
        {"id": "missing_motion", "group": "sensor_fault", "kind": "missing", "positive": True, "onset_s": .4},
        {"id": "stale_motion", "group": "sensor_fault", "kind": "stale", "positive": True, "onset_s": .4},
    ]
    return {"format": "dvidia.reaction-protocol", "schema_version": 1,
            "framework_version": FRAMEWORK_VERSION, "scope": "developmental synthetic mechanical coupons; no RGB perception, learning or hardware qualification",
            "seed": seed, "repeats": repeats, "variants": ["monitor", "disabled_control"],
            "cases": cases, "sensor_period_s": .01, "sensor_prime_s": .15,
            "reaction_period_s": .001, "trace_period_s": .01,
            "convergence_enabled": convergence,
            "convergence_diagnostics": {"event_time_absolute_s": .01, "contact_relative_change": .10, "penetration_absolute_m": .001,
                "scope": "authored developmental timestep diagnostics; no calibrated physical accuracy"},
            "reaction_config": asdict(ReactionConfig(horizon_s=.35, margin_m=.012,
                assumed_deceleration_m_s2=2., command_delay_s=.02, max_sensor_age_s=.10,
                contact_force_limit_n=10., slip_speed_limit_m_s=.015)),
            "fixture_config": {"timestep_s": .001, "warmup_s": .4, "horizon_s": 1.4,
                "speed_m_s": .25, "carriage_mass_kg": 1., "actuator_force_limit_n": 15.,
                "payload_mass_kg": .04, "jaw_force_limit_n": 8., "pair_friction": .7,
                "rest_speed_m_s": .005, "rest_dwell_s": .05, "command_delay_s": .02},
            "contact_diagnostics": {"maximum_penetration_m": .003, "maximum_force_proxy_n": 1000.,
                "scope": "authored engineering diagnostics, not calibrated material accuracy or hardware force limits"},
            "label_scope": "authored case-level hazard/fault label independent of intervention; positives include inevitable collision, slip and missing/stale sensing",
            "baseline_scope": "same detector in shadow, no brake intervention; no claim of absent detection in controls",
            "stop_scope": "bounded fixed-position hold; all trials continue to equal fixed horizons; rest requires measured speed below threshold for sustained dwell",
            "confirmation": False}


def actor(case, elapsed_s):
    """Exogenous trajectory depends only on protocol time, never robot truth."""
    kind = case["kind"]
    if kind == "crossing":
        return ((.18, .20-.28*elapsed_s, .4), (0., -.28, 0.))
    if kind == "static":
        return ((.23, 0., .4), (0., 0., 0.))
    if kind == "near_miss":
        return ((.20, .14, .4), (0., 0., 0.))
    if kind == "receding":
        return ((.18, .15+.20*elapsed_s, .4), (0., .20, 0.))
    if kind == "late":
        dt = max(0., elapsed_s-.45)
        return ((.14, .15-5.*min(dt, .08), .4), (0., -5. if .45 <= elapsed_s < .53 else 0., 0.))
    return ((.3, 1., .4), (0., 0., 0.))


def run_trial(protocol, case, variant, seed):
    # Optional native dependency is imported only when actually benchmarking.
    from .reaction_native import FixtureConfig, NativeReactionFixture
    c = FixtureConfig(**protocol["fixture_config"])
    reaction_ticks = round(protocol["reaction_period_s"]/c.timestep_s)
    if reaction_ticks < 1 or not math.isclose(reaction_ticks*c.timestep_s, protocol["reaction_period_s"], abs_tol=1e-12):
        raise ValueError("reaction period must contain whole native ticks")
    enabled = variant == "monitor"
    holding = case.get("holding", False)
    monitor = ReactionMonitor(ReactionConfig(**protocol["reaction_config"]))
    sensor_config = SensorConfig(period_s=protocol["sensor_period_s"],
        delivery_delay_s=case.get("sensor_delay_s", 0.), position_noise_m=case.get("position_noise_m", 0.))
    streams = {name: SensorStream(sensor_config, seed=seed+index) for index, name in enumerate(("motion", "contact", "grip"))}
    start = time.perf_counter()
    record = {"case_id": case["id"], "group": case["group"], "variant": variant,
        "seed": seed, "expected_detection": case["positive"], "valid": False,
        "hazard_onset_s": case.get("onset_s"), "detected_s": None, "decision_s": None,
        "brake_requested_s": None, "brake_applied_s": None, "stopped_s": None,
        "stopping_distance_m": None, "stopping_displacement_m": None,
        "request_to_rest_path_m": None, "max_forward_excursion_m": None,
        "sensor_sample_s": None, "sensor_delivered_s": None, "detected_reason": None,
        "peak_unexpected_force_n": 0., "unexpected_contact_impulse_ns": 0.,
        "minimum_clearance_m": None, "max_penetration_m": 0., "dropped": None if not holding else False,
        "simulated_seconds": 0., "wall_seconds": 0., "step_count": 0,
        "rest_dwell_s": c.rest_dwell_s, "rest_lost_after_confirmation": False,
        "observation_blackout_seconds": 0., "observation_stale_seconds": 0.,
        "observation_unusable_seconds": 0., "error": None, "intervention_enabled": enabled,
        "holding": holding, "trace": []}
    try:
        fixture = NativeReactionFixture(c, holding=holding)
        record["fixture"] = fixture.identity()
        compile_warmup_wall = time.perf_counter()-start
        prime = protocol["sensor_prime_s"]
        observations = {}
        delivered = {name: None for name in streams}
        delivery_times = {}
        def observe(now_s, elapsed_s, truth):
            obstacle_position, obstacle_velocity = actor(case, max(0., elapsed_s))
            if elapsed_s < 0:
                obstacle_velocity = (0., 0., 0.)
            obstacles = () if elapsed_s < case.get("occluded_until_s", -1.) else (ObstacleSphere(obstacle_position, obstacle_velocity, .035),)
            samples = {"motion": MotionSample(tuple(RobotSphere(**shape) for shape in truth["robot_shapes"]), obstacles),
                "contact": ContactSample(truth["unexpected_force_n"]),
                "grip": GripSample(holding and min(truth["pad_contacts"].values()) > 0, truth["slip_speed_m_s"])}
            for name, stream in streams.items():
                if name == "motion" and case["kind"] == "stale" and elapsed_s >= .4:
                    continue  # Device has stopped producing/delivering updates.
                sample = None if name == "motion" and case["kind"] == "missing" and elapsed_s >= .4 else samples[name]
                signal = stream.update(now_s, sample)
                observations[name] = signal
                if signal is not delivered[name]:
                    delivery_times[name] = now_s
                    delivered[name] = signal
        with fixture.native():
            fixture.tick(0., actor(case, 0.)[0])
            for index in range(round(prime/c.timestep_s)+1):
                now = index*c.timestep_s
                observe(now, -prime+now, fixture.measure())
                if index < round(prime/c.timestep_s):
                    fixture.tick(0., actor(case, 0.)[0])
            monitor.reset()
            hold_target = None
            brake_position = None
            last_brake_position = None
            brake_path = 0.
            request_position = request_last_position = None
            request_path = 0.
            forward_excursion = 0.
            quiet_ticks = 0
            loop_start = time.perf_counter()
            monitor_latencies = []
            truth = fixture.measure()
            decision = None
            for index in range(round(c.horizon_s/c.timestep_s)):
                t = index*c.timestep_s
                now = prime+t
                observe(now, t, truth)
                if index % reaction_ticks == 0:
                    monitor_start = time.perf_counter()
                    decision = monitor.update(now, observations, holding=holding)
                    monitor_latencies.append(time.perf_counter()-monitor_start)
                if decision.stop and record["detected_s"] is None:
                    detected = t  # decision was first made on this native grid tick
                    record.update(detected_s=detected, decision_s=detected, detected_reason=decision.reason)
                    signal_name = decision.signal_name
                    signal = observations.get(signal_name)
                    record["sensor_sample_s"] = signal.acquired_s-prime if signal is not None else None
                    record["sensor_delivered_s"] = delivery_times.get(signal_name, now)-prime if signal is not None else None
                    if enabled:
                        record["brake_requested_s"] = t
                        request_position = request_last_position = truth["position_m"][0]
                if enabled and record["brake_requested_s"] is not None and hold_target is None and t+1e-12 >= record["brake_requested_s"]+c.command_delay_s:
                    hold_target = float(truth["position_m"][0])
                    brake_position = last_brake_position = hold_target
                    record["brake_applied_s"] = t
                target = c.speed_m_s*(t+c.timestep_s) if hold_target is None else hold_target
                push = case.get("push_n", 0.) if .4 <= t < .46 else 0.
                pull = case.get("pull_n", 0.) if .4 <= t < .48 else 0.
                fixture.tick(target, actor(case, t+c.timestep_s)[0], carriage_push_n=push, payload_pull_n=pull)
                truth_after = fixture.measure()
                record["step_count"] += 1
                record["simulated_seconds"] = (index+1)*c.timestep_s
                record["peak_unexpected_force_n"] = max(record["peak_unexpected_force_n"], truth_after["unexpected_force_n"])
                record["unexpected_contact_impulse_ns"] += truth_after["unexpected_force_n"]*c.timestep_s
                gap = truth_after["clearance_m"]
                record["minimum_clearance_m"] = gap if record["minimum_clearance_m"] is None else min(record["minimum_clearance_m"], gap)
                record["max_penetration_m"] = max(record["max_penetration_m"], truth_after["penetration_m"])
                if holding:
                    record["dropped"] |= truth_after["dropped"]
                missing = any(observations.get(name) is None for name in ("motion", "contact")+(("grip",) if holding else ()))
                stale = any(signal is not None and now-signal.acquired_s > monitor.config.max_sensor_age_s+1e-12 for name, signal in observations.items() if name in ("motion", "contact")+(("grip",) if holding else ()))
                if missing:
                    record["observation_blackout_seconds"] += c.timestep_s
                if stale:
                    record["observation_stale_seconds"] += c.timestep_s
                if missing or stale:
                    record["observation_unusable_seconds"] += c.timestep_s
                if hold_target is not None:
                    current = truth_after["position_m"][0]
                    if record["stopped_s"] is None:
                        brake_path += abs(current-last_brake_position)
                    last_brake_position = current
                    quiet = abs(truth_after["velocity_m_s"][0]) < c.rest_speed_m_s
                    quiet_ticks = quiet_ticks+1 if quiet else 0
                    if record["stopped_s"] is None and quiet_ticks*c.timestep_s+1e-12 >= c.rest_dwell_s:
                        record["stopped_s"] = (index+1)*c.timestep_s
                        record["stopping_distance_m"] = brake_path
                        record["stopping_displacement_m"] = abs(current-brake_position)
                        record["request_to_rest_path_m"] = request_path+abs(current-request_last_position)
                        record["max_forward_excursion_m"] = max(forward_excursion, current-request_position)
                    elif record["stopped_s"] is not None and not quiet:
                        record["rest_lost_after_confirmation"] = True
                if request_position is not None and record["stopped_s"] is None:
                    current = truth_after["position_m"][0]
                    request_path += abs(current-request_last_position)
                    forward_excursion = max(forward_excursion, current-request_position)
                    request_last_position = current
                if index % round(protocol["trace_period_s"]/c.timestep_s) == 0 or decision.stop and record["detected_s"] == t or record["brake_applied_s"] == t:
                    record["trace"].append({"time_s": (index+1)*c.timestep_s,
                        "position_m": truth_after["position_m"], "velocity_m_s": truth_after["velocity_m_s"],
                        "command_target_m": target, "braking": hold_target is not None,
                        "unexpected_force_n": truth_after["unexpected_force_n"], "slip_speed_m_s": truth_after["slip_speed_m_s"],
                        "clearance_m": gap, "payload_position_m": truth_after["payload_position_m"],
                        "sensor_age_s": {name: None if signal is None else now-signal.acquired_s for name, signal in observations.items()}})
                truth = truth_after
            record["loop_wall_seconds"] = time.perf_counter()-loop_start
            record["monitor_update_wall_seconds"] = sum(monitor_latencies)
            record["monitor_call_latency_p50_s"] = statistics.median(monitor_latencies)
            record["monitor_call_latency_p95_s"] = sorted(monitor_latencies)[math.ceil(.95*len(monitor_latencies))-1]
            record["monitor_call_latency_max_s"] = max(monitor_latencies)
        record["compile_warmup_wall_seconds"] = compile_warmup_wall
        record["valid"] = True
        record["warnings"] = list(fixture.warnings)
        record["final_position_m"] = fixture.measure()["position_m"]
        record["rest_timestamp_scope"] = "first completion of sustained measured-speed dwell; physics continues afterward"
        record["distance_scope"] = "stopping_distance is path application-to-dwell-completion; request_to_rest_path includes command-delay travel; displacement is net application-to-rest"
        record["observation_available"] = all(observations.get(name) is not None for name in ("motion", "contact"))
        record["required_observations_fresh"] = all(observations.get(name) is not None and prime+c.horizon_s-c.timestep_s-observations[name].acquired_s <= monitor.config.max_sensor_age_s+1e-12 for name in ("motion", "contact")+(("grip",) if holding else ()))
        diagnostics = protocol["contact_diagnostics"]
        record["contact_diagnostic_pass"] = (record["max_penetration_m"] <= diagnostics["maximum_penetration_m"] and record["peak_unexpected_force_n"] <= diagnostics["maximum_force_proxy_n"])
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_seconds"] = time.perf_counter()-start
    return record


def paired_comparisons(records):
    pairs = {}
    for record in records:
        pairs.setdefault((record["case_id"], record["seed"]), {})[record["variant"]] = record
    output = []
    for (case_id, seed), pair in pairs.items():
        active, control = pair.get("monitor"), pair.get("disabled_control")
        row = {"case_id": case_id, "seed": seed, "both_valid": bool(active and control and active["valid"] and control["valid"])}
        for key in ("peak_unexpected_force_n", "unexpected_contact_impulse_ns", "minimum_clearance_m", "max_penetration_m"):
            row[key+"_monitor_minus_control"] = active[key]-control[key] if row["both_valid"] and active.get(key) is not None and control.get(key) is not None else None
        row["monitor_dropped"] = active.get("dropped") if active else None
        row["control_dropped"] = control.get("dropped") if control else None
        output.append(row)
    return output


def benchmark(output, *, repeats=3, seed=20261009, convergence=False):
    from .reaction_native import FIXTURE_ID
    import mujoco
    import numpy
    directory = Path(output).resolve()
    if directory.exists():
        raise ValueError("Use a fresh output directory to preserve evidence")
    protocol = make_protocol(repeats, seed, convergence=convergence)
    directory.mkdir(parents=True)
    protocol_hash = _hash(protocol)
    (directory/"protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False)+"\n")
    (directory/"protocol.sha256").write_text(protocol_hash+"\n")
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("reaction.py", "reaction_native.py", "reaction_benchmark.py", "reaction_metrics.py", "reaction_convergence.py", "env.py")}
    (directory/"sources.json").write_text(json.dumps(sources, indent=2)+"\n")
    records = []
    start = time.perf_counter()
    # Protocol/config/source identities are frozen to disk before any trial.
    with (directory/"trials.jsonl").open("w") as stream:
        for repeat in range(repeats):
            for index, case in enumerate(protocol["cases"]):
                trial_seed = seed+repeat*1000+index*10
                # Counterbalance execution order to reduce warm-cache/order bias.
                variants = protocol["variants"] if (repeat+index)%2 == 0 else list(reversed(protocol["variants"]))
                for variant in variants:
                    record = run_trial(protocol, case, variant, trial_seed)
                    records.append(record)
                    stream.write(json.dumps(record, allow_nan=False)+"\n")
                    stream.flush()
    elapsed = time.perf_counter()-start
    convergence_report = None
    convergence_wall = None
    if convergence:
        from .reaction_convergence import compare_timesteps
        convergence_start = time.perf_counter()
        convergence_report = compare_timesteps(protocol)
        (directory/"convergence.json").write_text(json.dumps(convergence_report, indent=2, allow_nan=False)+"\n")
        convergence_wall = time.perf_counter()-convergence_start
    report = {"format": "dvidia.reaction-benchmark", "schema_version": 1,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "framework_version": FRAMEWORK_VERSION, "scope": protocol["scope"],
        "benchmark_qualification": False, "physical_robot_ready": False,
        "fixture_id": FIXTURE_ID, "protocol": protocol, "protocol_sha256": protocol_hash,
        "source_sha256": sources, "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": numpy.__version__},
        "host": {"system": platform.system(), "release": platform.release(), "architecture": platform.machine(), "logical_cpus": os.cpu_count(), "processor": platform.processor() or None, "gpu_used": False},
        "timing_scope": "Per-trial wall includes native model compilation, physics warmup, sensor priming, rollout, native scoring and trace construction. total_trial_wall_seconds covers primary trials and JSONL writing; convergence_wall_seconds separately covers extra trials and convergence serialization. end_to_end_work_wall_seconds covers both; imports and final report/HTML/manifest writing excluded. Sensor/reaction/stop clocks are simulated time, not hardware deadlines.",
        "total_trial_wall_seconds": elapsed, "convergence_wall_seconds": convergence_wall,
        "end_to_end_work_wall_seconds": time.perf_counter()-start,
        "records": records, "summary": summarize(records),
        "paired_comparisons": paired_comparisons(records),
        "timestep_comparison": convergence_report,
        "limitations": ["Authored developmental cases, not independent confirmation or physical safety evidence.",
            "One horizontal actuator coupon, not full-arm collision clearance or six-dimensional motion.",
            "Synthetic sensed-state geometry and force/slip proxies; no RGB perception or calibrated tactile sensor.",
            "Braking is bounded native servo hold with ideal instantaneous local encoder position at application, not a certified hardware protective stop or encoder-fault model.",
            "Clearance uses native geometry, not sensor sphere approximations; force proxy includes applied external pushes.",
            "Each trial contains at most one authored hazard; scenario labels are not event annotations.",
            "Repeated deterministic cases are correlated; Wilson intervals are descriptive, not generalization guarantees.",
            "No GPU comparison, demonstrated minimum hardware, or contact-material calibration."]}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["host"]["lifetime_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss*1024
        report["host"]["rss_scope"] = "process lifetime including imports, all trials and report assembly; not per-trial/minimum requirement"
    except ImportError:
        report["host"]["lifetime_peak_rss_bytes"] = None
    (directory/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (directory/"report.html").write_text(render_report(report), encoding="utf-8")
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir()) if path.is_file()}
    (directory/"evidence-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return report


def inspect(directory):
    """Verify byte identities and recompute summaries without importing MuJoCo."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"evidence-manifest.json").read_text())
    required = {"protocol.json", "protocol.sha256", "sources.json", "trials.jsonl", "report.json", "report.html"}
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError("incomplete evidence manifest")
    for name, expected in manifest.items():
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("manifest must contain local filenames only")
        path = directory/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024*1024:
            raise ValueError("unsupported evidence file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed evidence: {name}")
    report = json.loads((directory/"report.json").read_text())
    protocol = json.loads((directory/"protocol.json").read_text())
    records = [json.loads(line) for line in (directory/"trials.jsonl").read_text().splitlines()]
    if report.get("format") != "dvidia.reaction-benchmark" or report.get("schema_version") != 1:
        raise ValueError("unsupported report")
    if report.get("framework_version") != FRAMEWORK_VERSION or report.get("physical_robot_ready") is not False or report.get("benchmark_qualification") is not False:
        raise ValueError("unsupported framework scope or readiness claim")
    try:
        expected_attempts = Counter((case["id"], variant, protocol["seed"]+repeat*1000+index*10)
            for repeat in range(protocol["repeats"]) for index, case in enumerate(protocol["cases"]) for variant in protocol["variants"])
        actual_attempts = Counter((record["case_id"], record["variant"], record["seed"]) for record in records)
        cases = {case["id"]: case for case in protocol["cases"]}
        if expected_attempts != actual_attempts or any(record["expected_detection"] is not cases[record["case_id"]]["positive"] or record["group"] != cases[record["case_id"]]["group"] for record in records):
            raise ValueError("missing, duplicate or inconsistent planned attempts")
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid trial/protocol contract") from exc
    if (report.get("protocol") != protocol or report.get("protocol_sha256") != _hash(protocol)
            or (directory/"protocol.sha256").read_text().strip() != _hash(protocol)
            or report.get("source_sha256") != json.loads((directory/"sources.json").read_text())
            or report.get("records") != records or report.get("summary") != summarize(records)
            or report.get("paired_comparisons") != paired_comparisons(records)):
        raise ValueError("inconsistent report identities or measurements")
    if protocol.get("convergence_enabled"):
        if "convergence.json" not in manifest or report.get("timestep_comparison") != json.loads((directory/"convergence.json").read_text()):
            raise ValueError("missing/inconsistent convergence evidence")
        from .reaction_convergence import recompute_comparison
        comparison = report["timestep_comparison"]
        if comparison != recompute_comparison(protocol, comparison["attempts"]):
            raise ValueError("inconsistent convergence verdict or measurements")
    return {"consistent": True, "attempted": len(records), "invalid": report["summary"]["invalid"],
            "scope": report["scope"], "physical_robot_ready": report["physical_robot_ready"],
            "identity_scope": "self-consistent bytes and measurements, not authenticated provenance or qualification"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reaction framework v0.1: offline synthetic native sensing/braking benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("benchmark", help="run CPU native coupons and paired no-brake controls")
    run.add_argument("--output", required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--seed", type=int, default=20261009)
    run.add_argument("--convergence", action="store_true", help="also compare 1 ms and 0.5 ms physics ticks with fixed sensing/reaction cadence")
    check = commands.add_parser("inspect", help="verify evidence bytes, identities and measurements without native physics")
    check.add_argument("directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(inspect(args.directory), indent=2))
            return 0
        report = benchmark(args.output, repeats=args.repeats, seed=args.seed, convergence=args.convergence)
    except ImportError as exc:
        parser.exit(2, f"Cannot run reaction benchmark: {exc}\nInstall the optional arm extra for native physics.\n")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Cannot process reaction evidence: {exc}\n")
    print(json.dumps({"report": str(Path(args.output).resolve()/"report.html"),
        "attempted": report["summary"]["attempted"], "invalid": report["summary"]["invalid"],
        "physical_robot_ready": False}, indent=2))
    return 1 if report["summary"]["invalid"] or (report.get("timestep_comparison") or {}).get("invalid", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
