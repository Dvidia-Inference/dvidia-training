"""A bounded, current-frame RGB-D estimator for one declared upright cube.

This is a colour/geometry baseline, not a generic object or learned tracker.
It imports no simulation state and never carries an estimate into another frame.
Depth accuracy and the heuristic uncertainty scales are NOT calibrated.  Yaw is
identifiable only modulo pi/2 for this unmarked square object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np


SCHEMA = 'dvidia.piper-rgbd-estimate/v1'
CLOCK_TOLERANCE_SECONDS = 1e-12
SIGNALS = ('object_position_estimate_m', 'object_quaternion_estimate_wxyz',
           'object_position_uncertainty_m', 'object_orientation_uncertainty_rad')
UNCERTAINTY = ('Heuristic geometric resolution/residual scales, not calibrated '
               'confidence bounds; empirical coverage must be measured separately.')


@dataclass(frozen=True)
class TrackerConfig:
    """Frozen profile for a 40 mm unmarked orange upright rigid cube."""
    side_m: float = .04
    maximum_age_seconds: float = .1
    minimum_depth_m: float = .05
    maximum_depth_m: float = 3.
    minimum_component_pixels: int = 12
    maximum_colour_pixels: int = 20000
    minimum_red_intensity: int = 25
    minimum_green_intensity: int = 8
    minimum_top_pixels: int = 6
    plane_tolerance_m: float = .0007
    maximum_tilt_rad: float = math.radians(6.)
    minimum_span_fraction: float = .60
    minimum_visibility: float = .86
    minimum_mask_iou: float = .70
    maximum_depth_residual_m: float = .008
    yaw_grid_steps: int = 180

    def __post_init__(self):
        # This baseline deliberately exposes no free-form colour/size fitting.
        # A different profile requires a different frozen configuration/study.
        for name, value in asdict(self).items():
            if not _number(value) or value <= 0:
                raise ValueError('Tracker configuration values must be positive finite numbers: '+name)
        for name in ('minimum_component_pixels', 'maximum_colour_pixels', 'minimum_top_pixels',
                     'minimum_red_intensity', 'minimum_green_intensity', 'yaw_grid_steps'):
            if type(getattr(self, name)) is not int:
                raise ValueError('Tracker pixel/grid counts must be integers.')
        if (self.side_m != .04 or self.maximum_age_seconds != .1
                or self.minimum_depth_m >= self.maximum_depth_m
                or not 3 <= self.minimum_top_pixels <= self.minimum_component_pixels
                or not 8 <= self.minimum_component_pixels <= 10000
                or not self.minimum_component_pixels <= self.maximum_colour_pixels <= 20000
                or not 1 <= self.minimum_green_intensity <= self.minimum_red_intensity <= 255
                or not 90 <= self.yaw_grid_steps <= 720
                or not 0 < self.maximum_tilt_rad <= math.radians(10)
                or not 0 < self.minimum_span_fraction <= 1
                or not 0 < self.minimum_visibility <= 1
                or not 0 < self.minimum_mask_iou <= 1):
            raise ValueError('Configuration exceeds the bounded upright 40 mm cube profile.')


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _empty(frame_index, captured, available, decision):
    # Invalid caller input is never echoed into a JSON receipt (NaN included).
    result = dict(schema=SCHEMA, status='refused', reason=None,
        frame_index=frame_index if type(frame_index) is int and frame_index >= 0 else None,
        captured_at_seconds=float(captured) if _number(captured) else None,
        available_at_seconds=float(available) if _number(available) else None,
        decision_at_seconds=float(decision) if _number(decision) else None,
        age_seconds=None, base_frame='robot_base',
        orientation_equivalence='upright yaw modulo pi/2; canonical yaw in [0, pi/2)',
        uncertainty_meaning=UNCERTAINTY, depth_accuracy_verified=False,
        signals={}, diagnostics={})
    for name in SIGNALS:
        result[name] = None
        result['signals'][name] = dict(status='missing', captured_at_seconds=None,
            available_at_seconds=None, age_seconds=None, value=None)
    return result


def _calibration(calibration, shape, captured, decision):
    if not isinstance(calibration, dict):
        raise ValueError('calibration_not_mapping')
    if calibration.get('depth_units') != 'm' or calibration.get('base_frame') != 'robot_base':
        raise ValueError('calibration_units_or_frame')
    intrinsics = calibration.get('intrinsics')
    if not isinstance(intrinsics, dict):
        raise ValueError('intrinsics_missing')
    height, width = shape
    if (type(intrinsics.get('width')) is not int or type(intrinsics.get('height')) is not int
            or (intrinsics['height'], intrinsics['width']) != shape):
        raise ValueError('calibration_image_dimensions')
    values = [intrinsics.get(key) for key in ('fx', 'fy', 'cx', 'cy')]
    if not all(_number(value) for value in values):
        raise ValueError('intrinsics_nonfinite')
    fx, fy, cx, cy = values
    if not (0 < fx < 1e6 and 0 < fy < 1e6 and 0 <= cx <= width and 0 <= cy <= height):
        raise ValueError('intrinsics_invalid')
    raw = calibration.get('base_from_camera')
    if not isinstance(raw, (list, tuple)) or len(raw) != 16 or not all(_number(x) for x in raw):
        raise ValueError('transform_invalid')
    transform = np.array(raw, dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-6, rel_tol=0)
            or np.max(np.abs(transform[:3, 3])) > 100):
        raise ValueError('transform_not_rigid')
    start, end = calibration.get('valid_from_seconds'), calibration.get('valid_until_seconds')
    if not (_number(start) and _number(end) and 0 <= start <= end
            and start <= captured <= decision <= end):
        raise ValueError('calibration_not_valid_at_observation')
    return tuple(float(x) for x in values), transform


def _components(mask):
    """Small bounded 4-connected components; no optional vision dependency."""
    remaining = mask.copy()
    components = []
    height, width = remaining.shape
    for row, column in np.argwhere(mask):
        if not remaining[row, column]:
            continue
        remaining[row, column] = False
        stack, pixels = [(int(row), int(column))], []
        while stack:
            y, x = stack.pop(); pixels.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < height and 0 <= xx < width and remaining[yy, xx]:
                    remaining[yy, xx] = False; stack.append((yy, xx))
        components.append(np.array(pixels, dtype=int))
    return sorted(components, key=len, reverse=True)


def _top_plane(points, config):
    """Deterministic bounded RANSAC, then an ordinary plane fit."""
    candidates = points[points[:, 2] >= np.median(points[:, 2])]
    if len(candidates) < config.minimum_top_pixels:
        return None
    # Fixed indices, no state, no per-frame random search or adaptation.
    rng = np.random.default_rng(701)
    triples = rng.integers(0, len(candidates), (96, 3))
    best = None
    for indices in triples:
        a, b, c = candidates[indices]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if normal[2] < 0:
            normal *= -1
        if normal[2] < math.cos(math.radians(45)):
            continue
        residual = np.abs((points-a) @ normal)
        selected = residual <= config.plane_tolerance_m
        count = int(selected.sum())
        # The high, broad face is the cube's candidate top. A plane through
        # several silhouette samples is insufficient evidence on its own.
        if count >= config.minimum_top_pixels and (best is None or count > best[0]):
            best = count, points[selected]
    if best is None:
        return None
    face = best[1]
    centre = face.mean(axis=0)
    _, singular, vectors = np.linalg.svd(face-centre, full_matrices=False)
    if len(singular) < 3 or singular[1] < .001:
        return None
    normal = vectors[-1]
    if normal[2] < 0:
        normal *= -1
    residual = np.abs((face-centre) @ normal)
    return face, normal, float(np.sqrt(np.mean(residual**2)))


def _rectangle(face, steps):
    xy = face[:, :2]
    angles = np.arange(steps) * (math.pi/2/steps)
    co, si = np.cos(angles), np.sin(angles)
    first = xy[:, 0, None]*co + xy[:, 1, None]*si
    second = -xy[:, 0, None]*si + xy[:, 1, None]*co
    lo_u, hi_u = first.min(axis=0), first.max(axis=0)
    lo_v, hi_v = second.min(axis=0), second.max(axis=0)
    widths = hi_u-lo_u; heights = hi_v-lo_v
    # Minimum rectangle area is invariant to translations and square symmetry.
    index = int(np.argmin(widths*heights))
    u, v = (lo_u[index]+hi_u[index])/2, (lo_v[index]+hi_v[index])/2
    centre = np.array([u*co[index]-v*si[index], u*si[index]+v*co[index]])
    return centre, float(angles[index]), float(widths[index]), float(heights[index])


def _cube_rays(shape, intrinsics, transform, centre, yaw, side):
    """Expected pixel-centre ray hits on the declared opaque geometric cube."""
    height, width = shape; fx, fy, cx, cy = intrinsics
    rows, columns = np.indices(shape)
    optical = np.stack(((columns+.5-cx)/fx, (rows+.5-cy)/fy, np.ones(shape)), axis=-1)
    co, si = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1.]])
    origin = (transform[:3, 3]-centre) @ rotation
    rays = optical @ transform[:3, :3].T @ rotation
    half = side/2
    parallel = np.abs(rays) < 1e-12
    safe = np.where(parallel, 1., rays)
    a, b = (-half-origin)/safe, (half-origin)/safe
    near, far = np.minimum(a, b), np.maximum(a, b)
    inside = np.abs(origin) <= half
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), near)
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), far)
    enter, leave = near.max(axis=-1), far.min(axis=-1)
    valid = (enter > 0) & (leave >= enter)
    return valid, np.where(valid, enter, 0.)


def estimate_pose(rgb, depth, calibration, *, frame_index, captured_at_seconds,
                  available_at_seconds, decision_at_seconds, config=TrackerConfig()):
    """Return current-frame evidence or explicit abstention, never native truth.

    Inputs are aligned RGB uint8 H×W×3 and floating camera-Z metres H×W;
    the camera is right/down/forward, samples at column+.5,row+.5. The caller
    supplies actual availability, not this function's local processing timer.
    Uncertainty scalars are top-level scalars and size-one vectors in signals.
    The 100 ms age boundary permits only 1e-12 s of floating-point tolerance;
    timestamp ordering remains strict. This is not a delivery-latency allowance.
    """
    began = time.perf_counter()
    result = _empty(frame_index, captured_at_seconds, available_at_seconds, decision_at_seconds)

    def finish(status, reason):
        result['status'] = status; result['reason'] = reason
        result['diagnostics']['processing_seconds'] = time.perf_counter()-began
        return result

    if not isinstance(config, TrackerConfig):
        return finish('refused', 'invalid_configuration')
    if type(frame_index) is not int or frame_index < 0:
        return finish('refused', 'invalid_frame_index')
    times = (captured_at_seconds, available_at_seconds, decision_at_seconds)
    if not all(_number(t) and t >= 0 for t in times):
        return finish('refused', 'invalid_timestamps')
    captured, available, decision = map(float, times)
    if not captured <= available <= decision:
        return finish('refused', 'noncausal_timestamps')
    result['age_seconds'] = decision-captured
    if decision-captured > config.maximum_age_seconds+CLOCK_TOLERANCE_SECONDS:
        return finish('stale', 'frame_older_than_100ms')
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[-1] != 3 or not 16 <= rgb.shape[0] <= 480
            or not 16 <= rgb.shape[1] <= 640):
        return finish('refused', 'invalid_rgb_array')
    shape = rgb.shape[:2]
    if (not isinstance(depth, np.ndarray) or depth.dtype.kind != 'f'
            or depth.dtype.itemsize not in (4, 8) or depth.shape != shape):
        return finish('refused', 'invalid_aligned_depth_array')
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        return finish('refused', 'nonfinite_or_nonpositive_depth')
    try:
        intrinsics, transform = _calibration(calibration, shape, captured, decision)
    except (ValueError, TypeError, OverflowError) as error:
        return finish('refused', str(error) if type(error) is ValueError else 'invalid_calibration')
    # Declared orange profile: ratios tolerate uniform shading, not arbitrary
    # lighting/material changes. Convert before multiplication to avoid overflow.
    colour = rgb.astype(float); red, green, blue = np.moveaxis(colour, -1, 0)
    # Keep the chromatic ratios fixed under shading. The positive brightness
    # floors exclude near-black quantization noise; a shadow is not occlusion.
    mask = ((red >= config.minimum_red_intensity) & (green >= config.minimum_green_intensity) & (red >= green*1.35)
            & (red >= blue*1.8) & (green >= blue*1.15))
    result['diagnostics']['orange_pixels'] = int(mask.sum())
    if result['diagnostics']['orange_pixels'] > config.maximum_colour_pixels:
        return finish('missing', 'colour_support_exceeds_bounded_profile')
    components = _components(mask)
    usable = [component for component in components if len(component) >= config.minimum_component_pixels]
    result['diagnostics']['candidate_components'] = len(usable)
    if not usable:
        return finish('missing', 'object_absent_or_too_small')
    if len(usable) != 1:
        return finish('missing', 'ambiguous_colour_components')
    pixels = usable[0]; rows, columns = pixels.T
    if (np.any(rows == 0) or np.any(rows == shape[0]-1)
            or np.any(columns == 0) or np.any(columns == shape[1]-1)):
        return finish('missing', 'object_cropped_at_image_boundary')
    z = depth[rows, columns].astype(float)
    if np.any(z < config.minimum_depth_m) or np.any(z > config.maximum_depth_m):
        return finish('missing', 'object_outside_usable_depth')
    fx, fy, cx, cy = intrinsics
    optical = np.stack(((columns+.5-cx)*z/fx, (rows+.5-cy)*z/fy, z), axis=-1)
    points = optical @ transform[:3, :3].T + transform[:3, 3]
    plane = _top_plane(points, config)
    if plane is None:
        return finish('missing', 'insufficient_top_plane')
    face, normal, residual = plane
    tilt = math.acos(float(np.clip(normal[2], -1, 1)))
    result['diagnostics'].update(top_pixels=len(face), top_tilt_rad=tilt, plane_rms_m=residual)
    if tilt > config.maximum_tilt_rad:
        return finish('missing', 'outside_upright_profile')
    xy, yaw, width, length = _rectangle(face, config.yaw_grid_steps)
    result['diagnostics'].update(top_sample_spans_m=[width, length])
    if (min(width, length) < config.side_m*config.minimum_span_fraction
            or max(width, length) > config.side_m*1.15):
        return finish('missing', 'incomplete_or_wrong_size_top_surface')
    centre = np.array([*xy, float(np.median(face[:, 2]))-config.side_m/2])
    expected, expected_z = _cube_rays(shape, intrinsics, transform, centre, yaw, config.side_m)
    actual = np.zeros(shape, dtype=bool); actual[rows, columns] = True
    intersection = expected & actual
    predicted_count = int(expected.sum()); observed_count = len(pixels)
    visibility = float(intersection.sum()/max(predicted_count, 1))
    iou = float(intersection.sum()/max(int((expected | actual).sum()), 1))
    depth_error = np.abs(depth[intersection]-expected_z[intersection])
    depth_p95 = float(np.quantile(depth_error, .95)) if depth_error.size else None
    result['diagnostics'].update(predicted_pixels=predicted_count, component_pixels=observed_count,
        visibility_fraction=visibility, mask_iou=iou, model_depth_p95_m=depth_p95)
    if (predicted_count < config.minimum_component_pixels or visibility < config.minimum_visibility
            or iou < config.minimum_mask_iou or depth_p95 is None
            or depth_p95 > config.maximum_depth_residual_m):
        return finish('missing', 'occluded_or_inconsistent_cube_geometry')
    pixel_scale = float(np.median(z)/min(fx, fy))
    position_scale = math.sqrt(3)*pixel_scale/2 + residual
    orientation_scale = min(math.pi/4, 2*pixel_scale/config.side_m + math.pi/2/config.yaw_grid_steps)
    values = (centre.tolist(), [math.cos(yaw/2), 0., 0., math.sin(yaw/2)],
              position_scale, orientation_scale)
    for name, value in zip(SIGNALS, values):
        result[name] = value
        result['signals'][name] = dict(status='available', captured_at_seconds=captured,
            available_at_seconds=available, age_seconds=decision-captured,
            value=value if isinstance(value, list) else [value])
    return finish('available', 'current_frame_upright_cube_estimate')
