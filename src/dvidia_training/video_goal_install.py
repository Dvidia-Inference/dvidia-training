"""Installable data-only bundles for calibrated video goals in PiPER simulation.

This stores a bounded video-goal proposal and its calibration. Execution calls
the existing authored PiPER task controller; it does not learn or install a
motor policy and does not communicate with physical hardware.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import stat
import time

import numpy as np

from . import video_goal_bridge
from .video_goal_bridge import PlanarVideoCalibration, infer_video_goal

SCHEMA = "dvidia.video-goal-install/v1"
RECEIPT_SCHEMA = "dvidia.video-goal-execution/v1"
PROFILE = "piper"
MAX_MANIFEST_BYTES = 128 * 1024
MAX_PROPOSAL_BYTES = 2 * 1024 * 1024
MAX_CALIBRATION_BYTES = 16 * 1024
MAX_EPISODE_BYTES = 64 * 1024 * 1024
FILES = ("manifest.json", "proposal.json", "calibration.json")
RUNTIME_FILES = (
    "video_goal_install.py", "video_goal_bridge.py", "sensor_goal_observers.py",
    "footage_data.py", "piper_task.py", "reference_task.py", "reference_sim.py",
    "robot_profiles.py", "robot_profiles.lock.json", "piper_grip_guard.py",
    "piper_contact_memory.py", "piper_contact_recovery.py", "piper_reacquisition.py",
    "piper_grasp_strategy.py",
)
LIMITATIONS = [
    "The proposal detects one green candidate under a caller-declared fixed orthographic calibration; it does not identify arbitrary objects or infer depth.",
    "Video goal XY is mapped directly into the PiPER task world frame. Camera-to-robot calibration is assumed, not measured by this bundle.",
    "Execution uses privileged live simulator state and the existing authored PiPER controller; no motor policy or action sequence is learned.",
    "The bundle contains a proposal and source-video digest, not the original video. Local hashes are integrity checks, not a signature or proof of proposal provenance.",
    "This simulation-only profile does not qualify a physical robot or establish performance on human footage, shoes, or other arms.",
]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def _write_json(path, value):
    _write_new(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf8") + b"\n")


def _strict_json(path, maximum):
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > maximum:
        raise ValueError(f"{path.name} must be a regular file within its size bound")

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate JSON field: " + key)
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite JSON number")

    try:
        value = json.loads(path.read_text(encoding="utf8"), object_pairs_hook=pairs,
                           parse_constant=constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {path.name}") from exc
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain a JSON object")
    _canonical(value)
    return value


def _reject_symlink_path(path):
    path = Path(path).absolute()
    aliases = {Path("/tmp"): Path("/private/tmp"), Path("/var"): Path("/private/var")}
    for part in (path, *path.parents):
        if part.is_symlink() and not (part in aliases and part.resolve() == aliases[part]):
            raise ValueError("Bundle paths may not contain symlinks")
    return path


def _profile(profile):
    if profile != PROFILE:
        raise ValueError("Only the piper simulation profile is supported")
    return PROFILE


def _runtime():
    root = Path(__file__).parent
    files = {}
    for name in RUNTIME_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Required runtime source is missing or a symlink: " + name)
        files[name] = _sha_file(path)
    import mujoco
    return {"profile": PROFILE, "source_sha256": files,
            "python": platform.python_version(), "numpy": np.__version__,
            "mujoco": mujoco.__version__}


def runtime_contract():
    """Return the exact code and dependency contract pinned by bundles."""
    return _runtime()


def _calibration(value):
    if isinstance(value, PlanarVideoCalibration):
        calibration = value
    elif type(value) is dict:
        calibration = PlanarVideoCalibration(**value)
    elif isinstance(value, (str, Path)):
        path = _reject_symlink_path(value)
        calibration = video_goal_bridge._read_json(path)
    else:
        raise ValueError("calibration must be a PlanarVideoCalibration, object, or JSON path")
    return calibration


def _proposal(value, calibration):
    expected = {"schema", "status", "reason", "goal_xy_m", "frame_count", "observations",
                "timing_ms", "video_sha256", "video_duration_s", "last_sample_time_s",
                "unobserved_tail_s", "calibration", "robot_policy_learned", "scope", "limitations",
                "timing_scope"}
    if type(value) is not dict or set(value) != expected:
        raise ValueError("Video proposal has an unsupported schema")
    if value["schema"] != video_goal_bridge.SCHEMA or value["status"] != "proposed":
        raise ValueError("Only a proposed video destination can be installed")
    if value["reason"] != "stable_visible_terminal_xy":
        raise ValueError("Proposal reason is not a stable visible terminal destination")
    if value["robot_policy_learned"] is not False:
        raise ValueError("Video goal bundles cannot claim a learned policy")
    if _canonical(value["calibration"]) != _canonical(asdict(calibration)):
        raise ValueError("Proposal calibration does not match the supplied calibration")
    if value["scope"] != video_goal_bridge.SCOPE or value["limitations"] != video_goal_bridge.LIMITATIONS:
        raise ValueError("Proposal scope or limitations do not match the bridge contract")
    expected_timing_scope = ("Local stat, probe, decode, observation and source hashing wall time; "
                             "excludes acquisition, terminal dwell before upload, network and actuation.")
    if value["timing_scope"] != expected_timing_scope:
        raise ValueError("Proposal timing scope does not match the bridge contract")
    goal = value["goal_xy_m"]
    if (type(goal) is not list or len(goal) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or not -1. <= v <= 1. for v in goal)):
        raise ValueError("Proposal destination must contain two finite bounded coordinates")
    if type(value["video_sha256"]) is not str or not re.fullmatch(r"[a-f0-9]{64}", value["video_sha256"]):
        raise ValueError("Proposal must bind the source video digest")
    if type(value["video_duration_s"]) not in (int, float) or not math.isfinite(value["video_duration_s"]) \
            or not .6 <= value["video_duration_s"] <= 30.:
        raise ValueError("Proposal video duration is invalid")
    duration = float(value["video_duration_s"])
    last_time, tail = value["last_sample_time_s"], value["unobserved_tail_s"]
    if (type(last_time) not in (int, float) or not math.isfinite(last_time) or not 0. <= last_time <= duration
            or type(tail) not in (int, float) or not math.isfinite(tail) or not 0. <= tail <= .251
            or abs((duration - last_time) - tail) > .002):
        raise ValueError("Proposal sample timing or unobserved tail is invalid")
    if type(value["frame_count"]) is not int or not 3 <= value["frame_count"] <= video_goal_bridge.MAX_FRAMES:
        raise ValueError("Proposal frame count is invalid")
    if type(value["observations"]) is not list or len(value["observations"]) != value["frame_count"]:
        raise ValueError("Proposal observations do not match its frame count")
    previous_time = -1.
    for row in value["observations"]:
        if type(row) is not dict or set(row) != {"t_s", "status", "position_xy_m", "reason"}:
            raise ValueError("Proposal observation schema is invalid")
        t_s = row["t_s"]
        if type(t_s) not in (int, float) or not math.isfinite(t_s) or not previous_time < t_s <= duration:
            raise ValueError("Proposal observation timestamps are invalid")
        previous_time = float(t_s)
        position = row["position_xy_m"]
        if row["status"] == "observed":
            if (row["reason"] != "single_green_candidate" or type(position) is not list or len(position) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not -1. <= v <= 1. for v in position)):
                raise ValueError("Observed proposal position is invalid")
        elif row["status"] == "unknown":
            if position is not None or type(row["reason"]) is not str:
                raise ValueError("Unknown proposal observation is invalid")
        else:
            raise ValueError("Proposal observation status is invalid")
    if abs(float(value["observations"][-1]["t_s"]) - float(last_time)) > .001:
        raise ValueError("Proposal last sample time does not match its observations")
    timings = value["timing_ms"]
    if (type(timings) is not dict or set(timings) != {"decode", "observe", "total"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0. <= v <= 300000.
                   for v in timings.values())):
        raise ValueError("Proposal timing measurements are invalid")
    # Check the destination against the current PiPER task's actual workspace,
    # support surface and transfer-distance validation. Pick the farthest legal
    # corner as a validation-only start so valid goals near the default start
    # are not rejected merely due to that default.
    from .piper_task import PiperTaskConfig
    candidates = ((.25, -.15), (.25, .15), (.48, -.15), (.48, .15))
    start = max(candidates, key=lambda xy: math.dist(xy, goal))
    PiperTaskConfig(object_position_m=(*start, .082), target_position_m=(*goal, .08))
    return [float(goal[0]), float(goal[1])]


def create_candidate(video, calibration, output):
    """Create a fresh data-only goal bundle; raw video is not copied."""
    calibration = _calibration(calibration)
    video_path = _reject_symlink_path(video)
    info = video_path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= video_goal_bridge.MAX_VIDEO_BYTES:
        raise ValueError("Source video must be a bounded regular file")
    output = _reject_symlink_path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a fresh output directory")
    proposal = infer_video_goal(video_path, calibration=calibration)
    goal = _proposal(proposal, calibration)
    video_sha = _sha_file(video_path)
    if proposal["video_sha256"] != video_sha:
        raise ValueError("Inference video digest differs from the source file")
    runtime = runtime_contract()
    proposal_raw = json.dumps(proposal, indent=2, ensure_ascii=False, allow_nan=False).encode("utf8") + b"\n"
    calibration_raw = json.dumps(asdict(calibration), indent=2, ensure_ascii=False, allow_nan=False).encode("utf8") + b"\n"
    if len(proposal_raw) > MAX_PROPOSAL_BYTES or len(calibration_raw) > MAX_CALIBRATION_BYTES:
        raise ValueError("Proposal or calibration exceeds its bundle size bound")
    records = {"proposal.json": {"bytes": len(proposal_raw), "sha256": _sha_bytes(proposal_raw)},
               "calibration.json": {"bytes": len(calibration_raw), "sha256": _sha_bytes(calibration_raw)}}
    binding = {"profile": PROFILE, "video_sha256": video_sha, "video_bytes": info.st_size,
               "goal_xy_m": goal, "files": records, "runtime": runtime}
    manifest = {"schema": SCHEMA, "status": "candidate", "profile": PROFILE,
                "video": {"sha256": video_sha, "bytes": info.st_size, "included": False},
                "goal_xy_m": goal, "proposal_schema": video_goal_bridge.SCHEMA,
                "files": records, "runtime": runtime,
                "runtime_fingerprint": _sha_bytes(_canonical(runtime)),
                "candidate_id": _sha_bytes(_canonical(binding))[:24],
                "scope": "calibrated-video-goal plus authored PiPER simulation controller",
                "policy_learned": False, "physical_robot_ready": False,
                "limitations": list(LIMITATIONS)}
    manifest_raw = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False).encode("utf8") + b"\n"
    if len(manifest_raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Candidate manifest exceeds its size bound")
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "proposal.json", proposal_raw)
    _write_new(output / "calibration.json", calibration_raw)
    _write_new(output / "manifest.json", manifest_raw)
    return manifest


def _check_directory(directory):
    root = _reject_symlink_path(directory)
    if not root.is_dir():
        raise ValueError("Candidate directory must exist")
    found = {item.name for item in root.iterdir()}
    if found != set(FILES):
        raise ValueError("Candidate must contain exactly manifest, proposal and calibration files")
    for name in FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Candidate files must be regular non-symlink files")
    return root


def load_candidate(directory, profile=PROFILE):
    """Verify a local bundle against the current code/runtime before execution."""
    profile = _profile(profile)
    root = _check_directory(directory)
    manifest = _strict_json(root / "manifest.json", MAX_MANIFEST_BYTES)
    expected = {"schema", "status", "profile", "video", "goal_xy_m", "proposal_schema", "files",
                "runtime", "runtime_fingerprint", "candidate_id", "scope", "policy_learned",
                "physical_robot_ready", "limitations"}
    if set(manifest) != expected or manifest["schema"] != SCHEMA or manifest["status"] != "candidate":
        raise ValueError("Unsupported or malformed video-goal candidate manifest")
    if manifest["profile"] != profile or manifest["proposal_schema"] != video_goal_bridge.SCHEMA:
        raise ValueError("Candidate profile or proposal schema mismatch")
    if manifest["scope"] != "calibrated-video-goal plus authored PiPER simulation controller" \
            or manifest["policy_learned"] is not False or manifest["physical_robot_ready"] is not False:
        raise ValueError("Candidate scope/provenance is not supported")
    if type(manifest["limitations"]) is not list or manifest["limitations"] != LIMITATIONS:
        raise ValueError("Candidate limitations do not match this runtime")
    video = manifest["video"]
    if (type(video) is not dict or set(video) != {"sha256", "bytes", "included"}
            or type(video["sha256"]) is not str or not re.fullmatch(r"[a-f0-9]{64}", video["sha256"])
            or type(video["bytes"]) is not int or not 0 < video["bytes"] <= video_goal_bridge.MAX_VIDEO_BYTES
            or video["included"] is not False):
        raise ValueError("Candidate video provenance is invalid")
    if (type(manifest["files"]) is not dict
            or set(manifest["files"]) != {"proposal.json", "calibration.json"}):
        raise ValueError("Candidate file list is invalid")
    for name, maximum in (("proposal.json", MAX_PROPOSAL_BYTES), ("calibration.json", MAX_CALIBRATION_BYTES)):
        record = manifest["files"][name]
        path = root / name
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > maximum:
            raise ValueError("Candidate file exceeds its size bound: " + name)
        raw = path.read_bytes()
        if (type(record) is not dict or set(record) != {"bytes", "sha256"}
                or type(record["bytes"]) is not int or record["bytes"] != len(raw)
                or len(raw) > maximum or record["sha256"] != _sha_bytes(raw)):
            raise ValueError("Candidate file hash or size mismatch: " + name)
    calibration_value = _strict_json(root / "calibration.json", MAX_CALIBRATION_BYTES)
    calibration = PlanarVideoCalibration(**calibration_value)
    if (set(calibration_value) != set(asdict(calibration))
            or _canonical(asdict(calibration)) != _canonical(calibration_value)):
        raise ValueError("Candidate calibration is not canonical for this runtime")
    proposal = _strict_json(root / "proposal.json", MAX_PROPOSAL_BYTES)
    goal = _proposal(proposal, calibration)
    if goal != manifest["goal_xy_m"] or proposal["video_sha256"] != video["sha256"]:
        raise ValueError("Candidate goal or source-video binding mismatch")
    runtime = runtime_contract()
    if manifest["runtime"] != runtime or manifest["runtime_fingerprint"] != _sha_bytes(_canonical(runtime)):
        raise ValueError("Candidate runtime fingerprint is stale")
    binding = {"profile": profile, "video_sha256": video["sha256"], "video_bytes": video["bytes"],
               "goal_xy_m": goal, "files": manifest["files"], "runtime": runtime}
    if manifest["candidate_id"] != _sha_bytes(_canonical(binding))[:24]:
        raise ValueError("Candidate identity does not match its contents")
    return {"manifest": manifest, "proposal": proposal, "calibration": asdict(calibration),
            "directory": str(root)}


def execute_candidate(directory, assets, object_xy, output, seed=0):
    """Run one candidate with the authored PiPER simulation controller only."""
    candidate = load_candidate(directory, profile=PROFILE)
    if type(object_xy) not in (tuple, list) or len(object_xy) != 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in object_xy):
        raise ValueError("object_xy must contain two finite coordinates")
    if type(seed) is not int or not 0 <= seed <= 999999:
        raise ValueError("seed must be an integer in [0, 999999]")
    goal = candidate["manifest"]["goal_xy_m"]
    from .piper_task import PiperTaskConfig, run_episode
    from .robot_profiles import verify_assets
    config = PiperTaskConfig(object_position_m=(float(object_xy[0]), float(object_xy[1]), .082),
                             target_position_m=(goal[0], goal[1], .08))
    provenance = verify_assets(PROFILE, assets)
    output = _reject_symlink_path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a fresh output directory")
    started = time.perf_counter()
    episode = run_episode(assets, config, seed=seed, control="placement")
    run_wall_seconds = time.perf_counter() - started
    episode_raw = json.dumps(episode, separators=(",", ":"), allow_nan=False).encode("utf8")
    if len(episode_raw) > MAX_EPISODE_BYTES:
        raise ValueError("Simulation episode exceeds its receipt size bound")
    episode_gz = gzip.compress(episode_raw, mtime=0)
    if len(episode_gz) > MAX_EPISODE_BYTES:
        raise ValueError("Compressed simulation episode exceeds its receipt size bound")
    output.mkdir(parents=True, exist_ok=False)
    episode_path = output / "episode.json.gz"
    _write_new(episode_path, episode_gz)
    runtime = runtime_contract()
    manifest_sha = _sha_file(Path(candidate["directory"]) / "manifest.json")
    receipt = {"schema": RECEIPT_SCHEMA, "status": "simulation_attempt_recorded",
               "candidate_id": candidate["manifest"]["candidate_id"], "profile": PROFILE,
               "bundle_manifest_sha256": manifest_sha,
               "asset_provenance": provenance, "runtime": runtime,
               "runtime_fingerprint": _sha_bytes(_canonical(runtime)),
               "goal_binding": {"frame": "piper_task_world", "units": "m", "xy_m": list(goal),
                                "target_z_m": .08, "object_xy_m": [float(object_xy[0]), float(object_xy[1])],
                                "object_z_m": .082, "source_video_sha256": candidate["manifest"]["video"]["sha256"]},
               "seed": seed, "episode_file": episode_path.name,
               "episode": {"file": episode_path.name, "bytes": len(episode_gz),
                           "sha256": _sha_bytes(episode_gz)},
               "goal_xy_m": list(goal), "run_wall_seconds": run_wall_seconds,
               "valid": episode.get("valid") is True, "success": episode.get("success") is True,
               "reason": episode.get("reason"), "metrics": episode.get("metrics", {}),
               "controller_origin": "authored PiPER simulation controller; no learned policy",
               "policy_learned": False, "physical_robot_ready": False,
               "network_used": False, "hardware_commands_sent": False,
               "scope": "one simulation attempt; not hardware qualification",
               "limitations": list(LIMITATIONS)}
    _write_json(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--video", required=True, type=Path)
    create.add_argument("--calibration", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--bundle", required=True, type=Path)
    inspect.add_argument("--profile", default=PROFILE)
    run = commands.add_parser("run")
    run.add_argument("--bundle", required=True, type=Path)
    run.add_argument("--assets", required=True, type=Path)
    run.add_argument("--object-x", required=True, type=float)
    run.add_argument("--object-y", required=True, type=float)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "create":
        result = create_candidate(args.video, args.calibration, args.output)
    elif args.command == "inspect":
        result = load_candidate(args.bundle, profile=args.profile)["manifest"]
    else:
        result = execute_candidate(args.bundle, args.assets, (args.object_x, args.object_y), args.output, seed=args.seed)
    summary = result if args.command == "inspect" else (
        {"candidate_id": result["candidate_id"], "status": result["status"], "output": str(args.output)}
        if args.command == "create" else
        {key: result[key] for key in ("schema", "status", "candidate_id", "profile", "goal_xy_m",
                                      "episode_file", "valid", "success", "run_wall_seconds",
                                      "runtime_fingerprint")})
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
