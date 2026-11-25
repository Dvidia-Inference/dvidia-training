"""One rendered RGB-D anchor and 50 Hz encoder control in a native PiPER scene."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time
import zlib

import numpy as np

from . import action_demonstration as tape
from . import camera_action_capture as capture
from . import piper_rgbd_tracker as tracker
from .piper_encoder_kinematics import PiperEncoderKinematics
from .piper_feedback_score import score_feedback_episode
from .piper_motor_primitive import MotorPrimitivePolicy

DURATION_S = 20.
CONTROL_STEPS = 1000
NEGATIVES = ("missing_initial_frame", "forced_open")
SCHEMA = "dvidia.piper-first-attempt/v1"


def _file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class _MaterialSession(capture._Session):
    def _task_configuration(self, task):
        return task.PiperTaskConfig(timeout_s=DURATION_S,
            object_position_m=self.case["object_position_m"],
            target_position_m=self.case["goal_position_m"],
            mass_kg=self.case["mass_kg"], friction=self.case["friction"])


class _Fixed:
    phase = "learned_time_primitive"

    def __init__(self, model):
        self.primitive = MotorPrimitivePolicy(model)

    def __call__(self, obs):
        return self.primitive(dict(t_seconds=obs["t_seconds"],
            joint_position_rad=obs["joint_position_rad"],
            object_position_m=obs["initial_object_position_world_m"],
            target_position_m=obs["goal_position_world_m"]))


def _observe(elapsed, qpos, qvel, kinematics, initial_position, goal, applied):
    # The fixed-base transform of this asset is validated to be identity by
    # PiperEncoderKinematics. Its independent model contains only the robot.
    arm = qpos[:6].tolist()
    pose = kinematics.fk(arm)
    return dict(t_seconds=elapsed, joint_position_rad=arm,
        joint_velocity_rad_s=qvel[:6].tolist(),
        gripper_joint_position_m=qpos[6:8].tolist(),
        gripper_joint_velocity_m_s=qvel[6:8].tolist(),
        tcp_position_world_m=pose["tcp_position_world_m"],
        tcp_quaternion_wxyz=pose["tcp_quaternion_world_wxyz"],
        initial_object_position_world_m=list(initial_position),
        initial_object_captured_at_seconds=0., initial_object_age_seconds=elapsed,
        goal_position_world_m=list(goal), previous_applied=deepcopy(applied))


def _percentiles(seconds):
    if not seconds:
        return None
    p50, p95, p99 = np.percentile(seconds, [50, 95, 99]) * 1000
    return dict(n=len(seconds), p50_ms=float(p50), p95_ms=float(p95),
                p99_ms=float(p99), max_ms=1000*max(seconds))


def _failure_stage(score):
    if not score["valid"]:
        return "evidence_invalid"
    if score["success"]:
        return None
    native = score["native_recomputed"]
    if native["unexpected_contact_native_ticks"]:
        return "critical_contact"
    for key, phase in (("grasp_seen", "grasp"), ("lift_seen", "lift")):
        if not native[key]:
            return phase
    if native["retention_lost"]:
        return "retention"
    if not native["carry_complete"]:
        return "transport"
    return "release_settle_or_retreat"


def _source_hashes(controller, *, surveyed):
    files = ["piper_first_attempt_eval.py", "piper_action_capture.py",
             "camera_action_capture.py", "piper_rgbd_tracker.py", "piper_encoder_kinematics.py",
             "piper_task.py", "piper_feedback_env.py", "piper_imitation_score.py",
             "piper_feedback_score.py",
             "piper_motor_primitive.py" if controller == "fixed" else "piper_first_attempt_policy.py"]
    if surveyed:
        files.append("piper_workspace_calibration.py")
    return {name: _file_hash(Path(__file__).with_name(name)) for name in files}


def run_attempt(assets, case, controller, *, fixed_model=None, calibration=None,
                native_calibration=False, negative=None, trace=False, _sources=None, action_log=None):
    """Run exactly once, retaining a receipt for refusal, exception or interrupt."""
    case = capture._case(case)
    if any(key not in case for key in ("mass_kg", "friction", "goal_id")):
        raise ValueError("Case requires mass_kg, friction and goal_id.")
    if (calibration is None) != bool(native_calibration):
        raise ValueError("Choose a setup calibration receipt or explicit native diagnostic calibration.")
    if negative is not None and negative not in NEGATIVES:
        raise ValueError("Unknown negative control.")
    config = capture.RolloutConfig(duration_s=DURATION_S, seed=case["seed"])
    session = _MaterialSession(assets, config, case)
    started = time.perf_counter()
    result = dict(schema=SCHEMA, scope="simulation_only", physical_robot_ready=False,
        case=case, controller=controller, negative=negative, status="started",
        success=False, valid=False, expected_control_steps=CONTROL_STEPS,
        control_steps=0, teacher_calls=0, initial_camera_pairs_delivered=0,
        calibration_source="native_oracle_diagnostic" if native_calibration else "surveyed_workspace",
        observation_contract="Single initial 320x240 RGB-D, fixed known goal, 50 Hz encoders and robot-only FK; no later imaging.",
        clock=dict(native_dt_s=.001, control_dt_s=.02, scheduled_s=DURATION_S),
        wall_timing_scope="Offline local execution; initial capture includes RGB/depth rendering and encoding, loop includes physics and dense scoring telemetry. No camera or hardware delivery latency.",
        phases=[], native_ticks=0, peak_pad_force_n=None, final_object_position_m=None,
        first_success_time_s=None, critical_contact_native_ticks=None,
        source_sha256=_sources or _source_hashes(controller, surveyed=calibration is not None), fixed_model_sha256=None, calibration_file_sha256=None)
    phases = result["phases"]
    actions, states, inference, loops = [], [], [], []
    try:
        if calibration is not None:
            from .piper_workspace_calibration import load_workspace_calibration
            result["calibration_file_sha256"] = _file_hash(calibration)
            calibrated = load_workspace_calibration(calibration, duration_s=DURATION_S)
        kinematics = PiperEncoderKinematics(assets)
        if controller == "fixed":
            result["fixed_model_sha256"] = _file_hash(fixed_model)
            policy = _Fixed(json.loads(Path(fixed_model).read_text()))
        else:
            from .piper_first_attempt_policy import make_policy
            policy = make_policy(controller, kinematics)
        header, previous_applied, meta = session.open()
        if not np.allclose(meta["world_from_robot_base"], np.eye(4).ravel(), rtol=0, atol=1e-12):
            raise ValueError("The fixed PiPER base differs from the checked world frame.")
        if native_calibration:
            calibrated = header["calibration"]
        result["tracker_calibration_sha256"] = tape.digest(calibrated)
        result["native_binding"] = header["binding"]
        result["runtime"] = meta["runtime"]
        snapshot, _, diag, timing = session.sample(0)
        rgb, depth = session.latest_rgbd
        result["renderer_warnings"] = list(session.renderer_warnings)
        # macOS classic OpenGL may lack ARB_clip_control. Keep the native depth
        # warning and rely on the tracked survey residual / pose checks.
        depth_warning = "ARB_clip_control unavailable while mjDEPTH_ZEROFAR requested, depth accuracy will be limited"
        if any(message != depth_warning for message in session.renderer_warnings):
            raise RuntimeError(f"Unexpected native renderer warning: {session.renderer_warnings}")
        result["capture_wall_s"] = timing["render_encode_wall_seconds"]
        init_frame = snapshot["camera"]["value"]
        if negative == "missing_initial_frame":
            rgb = depth = None
        else:
            result["initial_camera_pairs_delivered"] = 1
            result["initial_frame_sha256"] = {k: init_frame[k]["sha256"] for k in ("rgb", "depth")}
        began = time.perf_counter()
        est = tracker.estimate_pose(rgb, depth, calibrated, frame_index=0,
            captured_at_seconds=0., available_at_seconds=0., decision_at_seconds=0.)
        result["tracker_wall_s"] = time.perf_counter()-began
        result["initial_tracker"] = dict(status=est["status"], reason=est["reason"],
            object_position_world_m=est.get("object_position_estimate_m"))
        if est["status"] != "available":
            result.update(status="refused", failure_stage="initial_perception",
                          reason=est["reason"])
            return result
        anchor = est["object_position_estimate_m"]
        states.append(diag["policy_observation"])
        result["initial_position_error_m"] = float(np.linalg.norm(
            np.asarray(anchor) - np.asarray(states[0]["object_position_m"])))
        result["initial_position_error_scope"] = "Diagnostic distance to settled native block centre; never passed to the policy."
        trace_rows = []
        for index in range(CONTROL_STEPS):
            tick = time.perf_counter()
            elapsed = float(session.env.elapsed)
            obs = _observe(elapsed, session.env.data.qpos, session.env.data.qvel,
                kinematics, anchor, case["goal_position_m"], previous_applied)
            infer = time.perf_counter()
            requested = policy(obs)
            inference.append(time.perf_counter()-infer)
            tape._action(requested, header, "controller targets")
            issued = dict(requested)
            if negative == "forced_open":
                issued["gripper_position_m"] = .035
            phase = getattr(policy, "issued_phase", getattr(policy, "phase", "unspecified"))
            phase_change = not phases or phase != phases[-1]["phase"]
            if phase_change:
                phases.append(dict(phase=phase, t_seconds=elapsed))
            # Privileged diagnostics never enter _observe or either policy.
            if trace and (index % 25 == 0 or phase_change):
                state = states[-1]
                trace_rows.append(dict(t_seconds=elapsed, phase=phase,
                    tcp_m=obs["tcp_position_world_m"], object_m=state["object_position_m"],
                    jaw_m=obs["gripper_joint_position_m"],
                    contacts=state["grasp_contact_counts"], issued=issued))
            applied = session.advance(issued)
            previous_applied = applied["ramped_targets"]
            action_row = dict(t_seconds=elapsed, **issued, phase=phase)
            states[-1].update(issued_joint_targets_rad=issued["joint_targets_rad"],
                issued_gripper_position_m=issued["gripper_position_m"], command=action_row)
            actions.append(action_row)
            states.append(deepcopy(session.observation))
            loops.append(time.perf_counter()-tick)
        episode = capture._native_episode(session, states, actions)
        arrays, metadata = session.native_evidence()
        score = score_feedback_episode(episode, arrays, metadata, case["goal_position_m"][:2])
        peak = arrays["pad_normal_forces_n"].max(axis=0)
        native = score.get("native_recomputed", {})
        result.update(status="completed", valid=score["valid"], success=score["success"],
            reason=score["reason"], failure_stage=_failure_stage(score), score=score,
            native_ticks=int(len(arrays["native_tick"])), peak_pad_force_n=dict(left=float(peak[0]), right=float(peak[1])),
            first_success_time_s=native.get("first_success_time_seconds"),
            critical_contact_native_ticks=native.get("unexpected_contact_native_ticks"),
            final_object_position_m=arrays["object_position_m"][-1].tolist(),
            final_xy_error_m=score.get("terminal_only", {}).get("terminal_xy_error_m"),
            executed_commands_sha256=tape.digest(actions))
        if trace:
            result["trace"] = trace_rows
        result["policy_refusal_reason"] = getattr(policy, "refusal_reason", None)
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                      failure_stage="execution", reason=f"{type(exc).__name__}: {exc}"[:500])
    finally:
        if action_log is not None and actions:
            # 1000 seven-target vectors are enough to independently rerun the
            # native dynamics. Store exact float64 target values, including
            # overrides, without images or per-tick privileged geometry.
            targets = np.array([[*row["joint_targets_rad"], row["gripper_position_m"]]
                                for row in actions], dtype="<f8")
            compressed = zlib.compress(targets.tobytes(), level=6)
            with Path(action_log).open("xb") as stream:
                stream.write(compressed)
            result["action_log"] = dict(path=Path(action_log).name, codec="zlib",
                dtype="<f8", shape=list(targets.shape),
                columns=[f"joint{i}_target_rad" for i in range(1, 7)]+["gripper_target_m"],
                sha256=sha256(compressed).hexdigest(),
                bytes=len(compressed), includes_forced_open=negative == "forced_open")
        result["control_steps"] = len(actions)
        result["simulated_seconds"] = float(session.env.elapsed) if session.initialized else 0.
        result["inference_wall"] = _percentiles(inference)
        result["control_loop_wall"] = _percentiles(loops)
        session.close()
        result["attempt_wall_s"] = time.perf_counter()-started
    return result


def _write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "assets", "output", "controller"):
        cli.add_argument("--"+name, required=True)
    cli.add_argument("--fixed-model", type=Path)
    setup = cli.add_mutually_exclusive_group(required=True)
    setup.add_argument("--calibration", type=Path)
    setup.add_argument("--native-calibration", action="store_true",
                       help="Exact scene extrinsics for diagnostic use only.")
    cli.add_argument("--negative", choices=NEGATIVES)
    cli.add_argument("--trace", action="store_true", help="Sparse diagnostic samples.")
    args = cli.parse_args()
    if args.controller == "fixed" and not args.fixed_model:
        cli.error("fixed requires --fixed-model")
    cases = json.loads(Path(args.cases).read_text())
    if not isinstance(cases, list) or not cases or len({row["id"] for row in cases}) != len(cases):
        cli.error("A nonempty array of cases with unique IDs is required.")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    sources = _source_hashes(args.controller, surveyed=args.calibration is not None)
    if args.controller != "fixed":
        # Bind the implementation loaded by this process before running cases.
        from .piper_first_attempt_policy import make_policy
    _write(root/"run.json", dict(schema=SCHEMA, source_sha256=sources, controller=args.controller, negative=args.negative,
        registered_cases=cases, roster_sha256=_file_hash(args.cases), retry_count=0))
    statuses = Counter()
    successes = 0
    for index, case in enumerate(cases):
        key = f"attempt-{index:04}"
        _write(root/(key+"-started.json"), dict(case_id=case["id"], group_id=case["group_id"]))
        result = run_attempt(args.assets, case, args.controller, fixed_model=args.fixed_model,
            calibration=args.calibration, native_calibration=args.native_calibration,
            negative=args.negative, trace=args.trace, _sources=sources,
            action_log=root/(key+"-actions.f64.zlib"))
        _write(root/(key+".json"), result)
        statuses[result["status"]] += 1
        successes += result["success"]
        print(f'{case["id"]} {result["status"]} strict={result["success"]} {result["reason"]} '
              f'{result["attempt_wall_s"]:.2f}s', flush=True)
        if result["status"] == "interrupted":
            break
    _write(root/"summary.json", dict(scheduled=len(cases), attempted=sum(statuses.values()),
        statuses=dict(statuses), strict_successes=successes))
    return int(bool(statuses["interrupted"] or statuses["error"]))


if __name__ == "__main__":
    raise SystemExit(main())
