"""Fixed-camera PiPER workspace survey from a removable RGB-D fixture.

The surveyed top centres are expressed in the robot base frame. Camera intrinsics
and metric optical-Z depth are known. Only marker observations enter the rigid
fit. Two withheld markers check the resulting transform before it is accepted.
"""
from hashlib import sha256
import math
from pathlib import Path
import time

import numpy as np

from . import action_demonstration as tape

SCHEMA = "dvidia.piper-workspace-survey/v1"
DEFAULT_CAMERA_PROFILE = "nominal"
WIDTH, HEIGHT, FOVY = 320, 240, 45.
# name, RGB channel mask, surveyed top centre (base metres)
MARKERS = (
    ("red", (1, 0, 0), (.19, -.15, .076)),
    ("green", (0, 1, 0), (.49, -.14, .086)),
    ("blue", (0, 0, 1), (.48, .16, .106)),
    ("yellow", (1, 1, 0), (.18, .14, .121)),
    ("magenta", (1, 0, 1), (.29, -.055, .099)),
    ("cyan", (0, 1, 1), (.39, .061, .081)),
)
FIT = ("red", "green", "blue", "yellow")
CHECK = ("magenta", "cyan")
MAX_RESIDUAL_M = .004


class SurveyMismatch(ValueError):
    """Expose residuals for fault diagnosis while withholding an unusable pose."""

    def __init__(self, fit_mm, check_mm):
        super().__init__("Workspace survey exceeded the 4 mm marker bound.")
        self.fit_mm = fit_mm
        self.check_mm = check_mm


def intrinsics():
    f = HEIGHT / (2. * math.tan(math.radians(FOVY) / 2.))
    return dict(width=WIDTH, height=HEIGHT, fx=f, fy=f, cx=WIDTH / 2., cy=HEIGHT / 2.)


def _compact(points, max_diameter_m=.075):
    # The diagonal of an axis-aligned box can exceed the diameter of a
    # rotated marker. Use it only as a cheap upper bound for acceptance.
    if np.linalg.norm(np.ptp(points, axis=0)) <= max_diameter_m:
        return True
    for start in range(0, len(points), 32):
        delta = points[start:start+32, None, :] - points[None, :, :]
        if np.any(np.einsum("ijk,ijk->ij", delta, delta) > max_diameter_m**2):
            return False
    return True


def observations(rgb, depth):
    if (rgb.dtype != np.uint8 or rgb.shape != (HEIGHT, WIDTH, 3)
            or depth.shape != (HEIGHT, WIDTH) or depth.dtype != np.float32):
        raise ValueError("Expected 320 × 240 aligned RGB8 and float32 optical-Z metres.")
    k = intrinsics()
    # Each top face has a distinct saturated colour. Select its interior so
    # anti-aliased boundary pixels cannot sample the depth of its support.
    hi, lo = rgb >= 100, rgb <= 45
    detected = {}
    for name, channels, _ in MARKERS:
        mask = np.logical_and.reduce(
            [hi[:, :, j] if flag else lo[:, :, j] for j, flag in enumerate(channels)]
        )
        interior = np.zeros_like(mask)
        interior[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[1:-1, :-2]
                                & mask[1:-1, 2:] & mask[:-2, 1:-1] & mask[2:, 1:-1])
        rows, cols = np.nonzero(interior)
        if not 6 <= len(rows) <= 1024:
            raise ValueError(f"Missing or ambiguous colour support for {name}.")
        z = depth[rows, cols].astype(np.float64)
        if not (np.isfinite(z).all() and np.all((z >= .05) & (z <= 3.))):
            raise ValueError(f"Unusable metric depth at {name}.")
        points = np.column_stack(((cols + .5 - k["cx"]) * z / k["fx"],
                                  (rows + .5 - k["cy"]) * z / k["fy"], z))
        # Compact physical support also catches widely separated same-colour blobs.
        if not _compact(points):
            raise ValueError(f"Noncompact colour support for {name}.")
        detected[name] = dict(pixels=len(rows), camera_m=points.mean(axis=0).tolist())
    return detected


def _points(observed, names):
    if not isinstance(observed, dict) or set(observed) != {m[0] for m in MARKERS}:
        raise ValueError("Six named optical observations are required.")
    survey = {key: position for key, _, position in MARKERS}
    coordinates = []
    for name in names:
        point = observed[name]
        if not isinstance(point, dict) or type(point.get("pixels")) is not int or not 6 <= point["pixels"] <= 1024:
            raise ValueError("Invalid marker pixel support.")
        xyz = point.get("camera_m")
        if (not isinstance(xyz, (list, tuple)) or len(xyz) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in xyz)
                or not .05 <= xyz[2] <= 3.):
            raise ValueError("Invalid optical marker coordinates.")
        coordinates.append(xyz)
    camera = np.asarray(coordinates, dtype=np.float64)
    return camera, np.asarray([survey[name] for name in names])


def errors_mm(transform, observed, names):
    src, dst = _points(observed, names)
    return (1000. * np.linalg.norm(src @ transform[:3, :3].T + transform[:3, 3] - dst,
                                   axis=1)).tolist()


def solve(observed):
    src, dst = _points(observed, FIT)
    source_mean, target_mean = src.mean(axis=0), dst.mean(axis=0)
    u, singular, vh = np.linalg.svd((src - source_mean).T @ (dst - target_mean))
    if singular[1] < .01:
        raise ValueError("Survey points cannot constrain the camera orientation.")
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vh.T @ u.T))
    rotation = vh.T @ correction @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_mean - rotation @ source_mean
    fit_mm = errors_mm(transform, observed, FIT)
    check_mm = errors_mm(transform, observed, CHECK)
    if max(fit_mm + check_mm) > MAX_RESIDUAL_M * 1000.:
        raise SurveyMismatch(fit_mm, check_mm)
    return transform, fit_mm, check_mm


def make_survey(rgb, depth, *, camera_profile):
    start = time.perf_counter()
    observed = observations(rgb, depth)
    pose, fit, check = solve(observed)
    return dict(schema=SCHEMA, camera_profile=camera_profile,
                survey_top_centres_base_m={name: list(pos) for name, _, pos in MARKERS},
                observations=observed, fit_markers=FIT, check_markers=CHECK,
                fit_error_mm=fit, check_error_mm=check, max_allowed_error_mm=4.,
                base_from_camera=pose.ravel().tolist(), intrinsics=intrinsics(),
                rgb_sha256=sha256(rgb.tobytes()).hexdigest(),
                depth_float32_sha256=sha256(depth.tobytes()).hexdigest(),
                detect_and_fit_seconds=time.perf_counter() - start)


def load_workspace_calibration(path, *, camera_profile=DEFAULT_CAMERA_PROFILE, duration_s=20.,
                               expected_sha256=None):
    """Build a per-episode tracker calibration from a verified prior survey."""
    if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Episode duration must be finite and positive.")
    raw = tape._ordinary_bytes(Path(path), 128 * 1024)
    if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Workspace survey bytes differ from the supplied digest.")
    s = tape._decode(raw)
    if (not isinstance(s, dict) or s.get("schema") != SCHEMA or s.get("camera_profile") != camera_profile
            or s.get("intrinsics") != intrinsics() or s.get("fit_markers") != list(FIT)
            or s.get("check_markers") != list(CHECK)
            or s.get("survey_top_centres_base_m") != {n: list(p) for n, _, p in MARKERS}):
        raise ValueError("Workspace fixture, camera profile or intrinsics differ.")
    transform, fit_mm, check_mm = solve(s.get("observations"))
    for field, expected in (("fit_error_mm", fit_mm), ("check_error_mm", check_mm)):
        recorded = np.asarray(s.get(field), dtype=np.float64)
        if recorded.shape != (len(expected),) or not np.allclose(recorded, expected, atol=1e-7, rtol=0):
            raise ValueError("Stored residuals differ from the surveyed positions.")
    if s.get("max_allowed_error_mm") != 1000.*MAX_RESIDUAL_M:
        raise ValueError("Stored survey acceptance threshold differs.")
    saved = np.asarray(s.get("base_from_camera"), dtype=np.float64)
    if saved.shape != (16,) or not np.allclose(saved, transform.ravel(), atol=1e-10, rtol=0):
        raise ValueError("Stored transform differs from its survey observations.")
    return dict(id="piper-rgbd-survey-v1", base_frame="robot_base", tool_frame="link6",
                camera_frame="rgbd_optical", intrinsics=intrinsics(),
                base_from_camera=saved.tolist(), depth_units="m",
                valid_from_seconds=0., valid_until_seconds=float(duration_s))
