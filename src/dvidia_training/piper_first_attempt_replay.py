"""Replay a completed first-attempt actuator log with no renderer, policy or images."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import zlib

import numpy as np

from . import action_demonstration as tape
from .piper_feedback_env import FeedbackPiperTaskEnv
from .piper_feedback_score import score_feedback_episode
from .piper_first_attempt_eval import CONTROL_STEPS, DURATION_S, SCHEMA
from .piper_task import PiperTaskConfig


def read_actions(root, binding):
    """Read the bounded, exact 1000×7 float64 log from the receipt's directory."""
    name = binding["path"]
    if not isinstance(name, str) or not name.endswith("-actions.f64.zlib") or Path(name).name != name:
        raise ValueError("Expected an action log in the receipt directory.")
    if (binding.get("shape") != [CONTROL_STEPS, 7] or binding.get("dtype") != "<f8"
            or binding.get("codec") != "zlib"):
        raise ValueError("Action log has unknown layout.")
    with (root/name).open("rb") as stream:
        body = stream.read(65_537)
    if len(body) > 65_536 or len(body) != binding["bytes"] or sha256(body).hexdigest() != binding["sha256"]:
        raise ValueError("Action log size or digest mismatch.")
    expected = CONTROL_STEPS * 7 * 8
    decoder = zlib.decompressobj()
    raw = decoder.decompress(body, expected + 1)
    if (len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError("Action log has malformed decoded size or trailing data.")
    values = np.frombuffer(raw, dtype="<f8").reshape(CONTROL_STEPS, 7)
    if not np.isfinite(values).all():
        raise ValueError("Action log contains nonfinite targets.")
    return values


def _grip_diagnostics(arrays, config):
    """Descriptive force/friction proxy for the face-on vertical pinch only.

    This is a bound for tangential force on opposing vertical faces, not a
    material estimate or a general grasp-wrench limit. Exclude table support
    when describing the airborne phase. A transient impact is not preload.
    """
    contact = np.all(arrays["grasp_contact_counts"] > 0, axis=1)
    support = arrays["table_object_contact_count"] > 0
    normal = arrays["pad_normal_forces_n"].sum(axis=1)
    gravity_load = config.mass_kg * 9.81
    ratio = config.friction * normal / gravity_load
    result = dict(weight_n=gravity_load, friction=config.friction,
        minimum_total_normal_n_for_vertical_pinch=gravity_load/config.friction,
        scope="Post-hoc simulator force diagnostic; friction times summed pad normals divided by weight is a vertical-face static pinch proxy. Actual wrench also depends on contact directions and acceleration.")
    for label, mask in (("supported_bilateral", contact & support),
                        ("unsupported_bilateral", contact & ~support)):
        result[label] = dict(duration_s=float(mask.sum())*config.timestep)
        if mask.any():
            result[label].update(median_summed_pad_normal_n=float(np.median(normal[mask])),
                median_vertical_pinch_margin_ratio=float(np.median(ratio[mask])),
                p10_vertical_pinch_margin_ratio=float(np.quantile(ratio[mask], .1)),
                p90_vertical_pinch_margin_ratio=float(np.quantile(ratio[mask], .9)))
    return result


def replay(assets, path):
    path = Path(path)
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "completed":
        raise ValueError("Replay needs a completed first-attempt receipt.")
    if receipt.get("control_steps") != CONTROL_STEPS:
        raise ValueError("Replay needs all 1000 control intervals.")
    targets = read_actions(path.parent, receipt["action_log"])
    phases = receipt["phases"]
    if not phases or phases[0]["t_seconds"] != 0.:
        raise ValueError("Action phase history must begin at reset.")
    case = receipt["case"]
    config = PiperTaskConfig(timeout_s=DURATION_S, object_position_m=case["object_position_m"],
        target_position_m=case["goal_position_m"], mass_kg=case["mass_kg"], friction=case["friction"])
    env = FeedbackPiperTaskEnv(assets, config)
    observation, info = env.reset(case["seed"])
    states, actions = [observation], []
    phase_index = 0
    for row in targets:
        elapsed = float(env.elapsed)
        while phase_index + 1 < len(phases) and phases[phase_index+1]["t_seconds"] <= elapsed:
            phase_index += 1
        action = dict(joint_targets_rad=row[:6].tolist(), gripper_position_m=float(row[6]))
        action_row = dict(t_seconds=elapsed, **action, phase=phases[phase_index]["phase"])
        states[-1].update(issued_joint_targets_rad=action["joint_targets_rad"],
            issued_gripper_position_m=action["gripper_position_m"], command=action_row)
        actions.append(action_row)
        observation, info = env.step(action)
        if observation is None or not info["valid"]:
            raise ValueError("Invalid native replay interval.")
        states.append(observation)
    episode = dict(schema="dvidia.piper-task-episode/v1", config=asdict(config),
        initial_object_position_m=states[0]["object_position_m"], trajectory=states, actions=actions,
        valid=bool(info["valid"]), success=bool(info["success"]), reason=info["reason"],
        metrics={**info, "contact_pair_native_ticks": deepcopy(env.contact_pair_native_ticks)},
        physical_robot_ready=False, simulated_seconds=env.elapsed)
    arrays = env.native_arrays()
    score = score_feedback_episode(episode, arrays, env.native_metadata(), config.target_position_m[:2])
    native = score.get("native_recomputed", {})
    peak = arrays["pad_normal_forces_n"].max(axis=0)
    observed = dict(success=score["success"], valid=score["valid"], reason=score["reason"],
        native_ticks=len(arrays["native_tick"]), critical_contact_native_ticks=native.get("unexpected_contact_native_ticks"),
        first_success_time_s=native.get("first_success_time_seconds"),
        final_object_position_m=arrays["object_position_m"][-1].tolist(),
        peak_pad_force_n=dict(left=float(peak[0]), right=float(peak[1])),
        executed_commands_sha256=tape.digest(actions))
    mismatches = []
    for key, value in observed.items():
        expected = receipt[key]
        if isinstance(value, dict):
            good = (isinstance(expected, dict) and value.keys() == expected.keys() and
                    all(math.isclose(v, expected[k], abs_tol=1e-9, rel_tol=0.) for k, v in value.items()))
        elif isinstance(value, list):
            good = np.allclose(value, expected, atol=1e-9, rtol=0.)
        elif isinstance(value, float) and isinstance(expected, (float, int)):
            good = math.isclose(value, expected, abs_tol=1e-9, rel_tol=0.)
        else:
            good = value == expected and type(value) is type(expected)
        if not good:
            mismatches.append(key)
    return dict(schema="dvidia.piper-first-attempt-command-replay/v1", case_id=case["id"],
                receipt_sha256=sha256(path.read_bytes()).hexdigest(), reproduced=not mismatches,
                mismatches=mismatches, observed=observed, grip_diagnostics=_grip_diagnostics(arrays, config), command_source="recorded_actuator_targets",
                scope="Headless deterministic native dynamics and scoring check, separate from camera and policy inference.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--assets", required=True)
    cli.add_argument("--receipt", required=True)
    args = cli.parse_args()
    outcome = replay(args.assets, args.receipt)
    print(json.dumps(outcome, indent=2, allow_nan=False))
    return int(not outcome["reproduced"])


if __name__ == "__main__":
    raise SystemExit(main())
