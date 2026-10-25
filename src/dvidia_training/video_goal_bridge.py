"""Propose a terminal planar goal from a calibrated, green-proxy video.

This is a bounded spatial grounding component. It does not recover actions,
identify arbitrary objects, estimate depth or learn a robot policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

import numpy as np

from .footage_data import VIDEO_EXTENSIONS, probe_video
from .sensor_goal_observers import _components

SCHEMA = "dvidia.video-goal-proposal/v1"
MAX_VIDEO_BYTES = 128 * 1024 * 1024
MAX_FRAMES = 180
SELECT_INTERVAL_S = .199
SCOPE = ("Planar terminal XY proposal for one visible green rigid proxy under the supplied "
         "orthographic calibration. This is spatial grounding only; it is not motor learning.")
LIMITATIONS = [
    "The color rule does not establish object identity; it detects a green candidate only.",
    "The projection assumes the supplied fixed orthographic calibration and does not estimate depth.",
    "position_bound_m is a caller-declared bound; this tool does not calibrate or independently validate it.",
    "Approximate 5 Hz sampling can miss changes between frames or brief terminal states; the last selected frame may precede the actual clip end.",
    "No contact, support, grasp, action, force, or causal information is inferred.",
    "A stable terminal image window is not proof of physical rest or successful task completion.",
    "No robot policy is learned or qualified by this proposal.",
]


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


@dataclass(frozen=True)
class PlanarVideoCalibration:
    width: int = 320
    height: int = 240
    center_xy_m: tuple[float, float] = (.36, 0.)
    vertical_span_m: float = .5
    position_bound_m: float = .005
    terminal_dwell_s: float = .5
    maximum_terminal_spread_m: float = .005

    def __post_init__(self):
        if type(self.width) is not int or not 64 <= self.width <= 640:
            raise ValueError("width must be an integer from 64 to 640")
        if type(self.height) is not int or not 64 <= self.height <= 480:
            raise ValueError("height must be an integer from 64 to 480")
        if (type(self.center_xy_m) not in (tuple, list) or len(self.center_xy_m) != 2
                or not all(_number(value, -2., 2.) for value in self.center_xy_m)):
            raise ValueError("center_xy_m must contain two finite coordinates in [-2, 2]")
        object.__setattr__(self, "center_xy_m", tuple(self.center_xy_m))
        for name, value, low, high in (
                ("vertical_span_m", self.vertical_span_m, .1, 2.),
                ("position_bound_m", self.position_bound_m, .0001, .05),
                ("terminal_dwell_s", self.terminal_dwell_s, .5, 5.),
                ("maximum_terminal_spread_m", self.maximum_terminal_spread_m, .0001, .05)):
            if not _number(value, low, high):
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")


def _observe_frame(rgb, t_s, calibration):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (
            calibration.height, calibration.width, 3):
        raise ValueError("Each frame must be uint8 RGB with the calibrated dimensions")
    red, green, blue = (rgb[:, :, index].astype(np.uint16) for index in range(3))
    mask = (green > 45) & (green > 1.6 * red) & (green > 1.3 * blue)
    regions = _components(mask)
    candidates = []
    clipped = False
    for region in regions:
        area = len(region)
        if not 60 <= area <= 5000:
            continue
        ys, xs = np.divmod(region, calibration.width)
        if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= calibration.width - 2 or ys.max() >= calibration.height - 2:
            clipped = True
            continue
        scale = calibration.vertical_span_m / calibration.height
        x_m = (float(xs.mean()) + .5 - calibration.width / 2) * scale + calibration.center_xy_m[0]
        y_m = (calibration.height / 2 - float(ys.mean()) - .5) * scale + calibration.center_xy_m[1]
        candidates.append([x_m, y_m])
    if clipped:
        return {"t_s": t_s, "status": "unknown", "position_xy_m": None, "reason": "candidate_clipped_by_image_edge"}
    if len(candidates) == 0:
        return {"t_s": t_s, "status": "unknown", "position_xy_m": None, "reason": "no_green_candidate"}
    if len(candidates) > 1:
        return {"t_s": t_s, "status": "unknown", "position_xy_m": None, "reason": "ambiguous_green_candidates"}
    return {"t_s": t_s, "status": "observed", "position_xy_m": candidates[0], "reason": "single_green_candidate"}


def _unknown(reason, observations, timings, calibration):
    return {"schema": SCHEMA, "status": "unknown", "reason": reason, "goal_xy_m": None,
            "frame_count": len(observations), "observations": observations,
            "timing_ms": timings, "video_sha256": None,
            "video_duration_s": None,
            "last_sample_time_s": observations[-1]["t_s"] if observations else None,
            "unobserved_tail_s": None,
            "calibration": asdict(calibration), "robot_policy_learned": False,
            "scope": SCOPE, "limitations": list(LIMITATIONS)}


def infer_frames(frames: list[dict], calibration: PlanarVideoCalibration = PlanarVideoCalibration()) -> dict:
    """Infer a terminal XY proposal from actual decoded frames and their PTS."""
    if not isinstance(calibration, PlanarVideoCalibration):
        raise ValueError("calibration must be PlanarVideoCalibration")
    if type(frames) is not list or not 3 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"frames must contain 3 to {MAX_FRAMES} decoded observations")
    for frame in frames:
        if type(frame) is not dict or set(frame) != {"t_s", "rgb"} or not _number(frame["t_s"], 0., 30.):
            raise ValueError("Each frame requires a finite nonnegative t_s and RGB image")
    times = [frame["t_s"] for frame in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Decoded presentation timestamps must be strictly increasing")
    observe_start = time.perf_counter()
    observations = [_observe_frame(frame["rgb"], frame["t_s"], calibration) for frame in frames]
    observe_ms = (time.perf_counter() - observe_start) * 1000
    terminal_end = times[-1]
    boundary = terminal_end - calibration.terminal_dwell_s
    first_in_window = next((index for index, stamp in enumerate(times) if stamp >= boundary), len(times) - 1)
    # Keep the frame immediately before the nominal dwell boundary to demonstrate
    # a sampled interval at least as long as the declared dwell.
    start_index = max(0, first_in_window - 1)
    terminal = observations[start_index:]
    timings = {"decode": 0., "observe": observe_ms, "total": observe_ms}
    if any(b - a > .25 + 1e-9 for a, b in zip(times, times[1:])):
        return _unknown("presentation_timestamp_gap", observations, timings, calibration)
    if terminal[-1]["status"] != "observed":
        return _unknown("final_decoded_frame_unobservable", observations, timings, calibration)
    if times[-1] - times[start_index] + 1e-9 < calibration.terminal_dwell_s:
        return _unknown("insufficient_terminal_dwell", observations, timings, calibration)
    if len(terminal) < 3:
        return _unknown("insufficient_terminal_observations", observations, timings, calibration)
    if any(row["status"] != "observed" for row in terminal):
        return _unknown("terminal_window_unobservable", observations, timings, calibration)
    positions = np.asarray([row["position_xy_m"] for row in terminal], dtype=np.float64)
    median = np.median(positions, axis=0)
    spread = float(np.max(np.linalg.norm(positions - median, axis=1)))
    if spread > calibration.maximum_terminal_spread_m:
        return _unknown("terminal_position_spread_exceeded", observations, timings, calibration)
    return {"schema": SCHEMA, "status": "proposed", "reason": "stable_visible_terminal_xy",
            "goal_xy_m": median.tolist(), "frame_count": len(observations),
            "observations": observations, "timing_ms": timings, "video_sha256": None,
            "video_duration_s": None, "last_sample_time_s": times[-1], "unobserved_tail_s": None,
            "calibration": asdict(calibration), "robot_policy_learned": False,
            "scope": SCOPE, "limitations": list(LIMITATIONS)}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_video(path: Path, calibration: PlanarVideoCalibration, duration_s: float) -> tuple[list[dict], float]:
    width, height = calibration.width, calibration.height
    command = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-nostdin",
               "-threads", "1", "-filter_threads", "1", "-noautorotate", "-copyts",
               "-protocol_whitelist", "file", "-format_whitelist", "mov,matroska,webm,avi",
               "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{SELECT_INTERVAL_S})',showinfo",
               "-fps_mode", "passthrough", "-frames:v", str(MAX_FRAMES + 1), "-f", "rawvideo",
               "-pix_fmt", "rgb24", "pipe:1"]
    start = time.perf_counter()
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=max(15., duration_s * 3), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("FFmpeg could not decode the bounded local video") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000
    frame_bytes = width * height * 3
    if result.returncode != 0 or len(result.stdout) % frame_bytes:
        raise ValueError("FFmpeg decode failed or returned incomplete RGB frames")
    count = len(result.stdout) // frame_bytes
    if not 3 <= count <= MAX_FRAMES or len(result.stdout) > MAX_FRAMES * frame_bytes:
        raise ValueError("FFmpeg selected an unsupported number of frames")
    pts = [float(match.group(1)) for match in re.finditer(rb"pts_time:([-+0-9.eE]+)", result.stderr)]
    if len(pts) != count or any(not math.isfinite(value) or value < 0 or value > 30 for value in pts):
        raise ValueError("FFmpeg did not report one bounded presentation timestamp per decoded frame")
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Video presentation timestamps are not strictly increasing")
    if abs(pts[0]) > .001:
        raise ValueError("The first decoded presentation timestamp must be zero")
    raw = np.frombuffer(result.stdout, dtype=np.uint8).reshape(count, height, width, 3)
    frames = [{"t_s": pts[index], "rgb": raw[index]} for index in range(count)]
    return frames, elapsed_ms


def infer_video_goal(video: Path, calibration: PlanarVideoCalibration = PlanarVideoCalibration()) -> dict:
    """Decode a local video and propose its stable terminal planar XY position."""
    started = time.perf_counter()
    if not isinstance(calibration, PlanarVideoCalibration):
        raise ValueError("calibration must be PlanarVideoCalibration")
    path = Path(video).absolute()
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("unsupported video extension")
    system_aliases = {Path("/tmp"): Path("/private/tmp"), Path("/var"): Path("/private/var")}
    for part in (path, *path.parents):
        if part.is_symlink() and not (part in system_aliases and part.resolve() == system_aliases[part]):
            raise ValueError("video path may not contain symlinks")
    try:
        info = path.stat()
    except OSError as exc:
        raise ValueError("video must be a readable local file") from exc
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_VIDEO_BYTES:
        raise ValueError("video must be a regular file from 1 byte to 128 MiB")
    probe = probe_video(path)
    if probe["width"] != calibration.width or probe["height"] != calibration.height:
        raise ValueError("video dimensions must exactly match calibration")
    if abs(probe["start_seconds"]) > 1e-6:
        raise ValueError("video must have a zero start time")
    duration = probe["duration_seconds"]
    if not .6 <= duration <= 30.:
        raise ValueError("video duration must be from 0.6 to 30 seconds")
    frames, decode_ms = _decode_video(path, calibration, duration)
    result = infer_frames(frames, calibration)
    result["timing_ms"]["decode"] = decode_ms
    result["timing_ms"]["observe"] = max(0., result["timing_ms"]["observe"])
    result["video_sha256"] = _hash_file(path)
    last_sample = frames[-1]["t_s"]
    tail = max(0., duration - last_sample)
    result["video_duration_s"] = duration
    result["last_sample_time_s"] = last_sample
    result["unobserved_tail_s"] = tail
    if tail > .251:
        result.update(status="unknown", reason="unobserved_clip_tail_exceeded", goal_xy_m=None)
    result["timing_ms"]["total"] = (time.perf_counter() - started) * 1000
    result["timing_scope"] = ("Local stat, probe, decode, observation and source hashing wall time; "
                              "excludes acquisition, terminal dwell before upload, network and actuation.")
    return result


def _read_json(path: Path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate calibration JSON field")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite calibration JSON number")
    path = Path(path)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            raise ValueError("calibration JSON must be a regular file of at most 64 KiB")
        with path.open("rb") as stream:
            raw = stream.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ValueError("calibration JSON exceeds 64 KiB")
        value = json.loads(raw.decode("utf8"), object_pairs_hook=pairs, parse_constant=constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("calibration must be a readable JSON object") from exc
    if type(value) is not dict:
        raise ValueError("calibration must be a JSON object")
    try:
        return PlanarVideoCalibration(**value)
    except TypeError as exc:
        raise ValueError("calibration contains unsupported fields or arguments") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("infer", help="propose a planar terminal goal from local video")
    infer.add_argument("video", type=Path)
    infer.add_argument("--calibration", type=Path)
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    calibration = _read_json(args.calibration) if args.calibration else PlanarVideoCalibration()
    result = infer_video_goal(args.video, calibration)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new JSON path; existing files are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "reason": result["reason"], "output": str(output)}))
    return 0 if result["status"] == "proposed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
