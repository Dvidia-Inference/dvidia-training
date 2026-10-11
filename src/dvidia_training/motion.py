"""Causal image-anchor motion, without calibration or robot command authority.

Only two actual, contiguous observations enter each velocity estimate. Missing
observations never become interpolated measurements. Forecasts are un-clipped
constant-velocity hypotheses, not object dynamics or calibrated probabilities.
"""
from __future__ import annotations

import math


KIND = 'dvidia.image-motion.v1'
ALGORITHM = 'two-observation-finite-difference'
UNCERTAINTY_SEMANTICS = 'empirical-envelope-not-calibrated-probability'
UNITS = {'position': 'image_pixels', 'velocity': 'image_pixels_per_second',
         'time': 'seconds'}


def _number(value, name, *, minimum=None, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f'{name} must be a finite number, not a boolean.')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f'{name} must be finite.') from None
    if (not math.isfinite(result) or (minimum is not None and result < minimum)
            or (positive and result <= 0)):
        raise ValueError(f'{name} must be finite and within its supported range.')
    return result


def _dimensions(width, height):
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('Image width and height must be positive integers.')
    # The coordinates, not only integer dimensions, must fit finite float space.
    _number(width, 'Image width', positive=True)
    _number(height, 'Image height', positive=True)


def _anchor(value, width, height):
    if type(value) is not list or len(value) != 2:
        raise ValueError('An observed anchor requires a two-number pixel list.')
    center = [_number(v, 'Anchor coordinate', minimum=0) for v in value]
    if center[0] >= width or center[1] >= height:
        raise ValueError('An observed anchor must be inside the image.')
    return center


def _support(value):
    if type(value) is not int or value < 0:
        raise ValueError('Support count must be a nonnegative integer.')
    return value


class MotionEstimator:
    """Causal, unsmoothed finite-difference baseline for one image feature anchor.

    ``update`` takes actual acquisition times in one declared recording clock.
    They must strictly increase, including missing frames. A tracked center must
    be inside the image. ``None`` marks missing evidence; explicit ``lost`` also
    requires ``None``. A loss or a gap greater than ``max_gap_seconds`` clears
    velocity continuity. Two fresh observations are then needed to forecast.

    A provided uncertainty is a caller's empirical anchor radius in pixels. If
    both velocity observations have radii, the forecast radius is current radius
    plus horizon times the sum of both radii divided by their actual time gap.
    This only propagates those supplied radii through the finite difference. It
    excludes acceleration, identity changes, camera motion and timestamp error;
    it is not a calibrated probability or an interaction guarantee. If either
    radius is unknown, forecast uncertainty remains ``None``.

    Prediction records expire for consumer reuse after
    ``max_prediction_age_seconds``. This freshness lifetime does not truncate
    the declared forecast horizons: a fresh record can describe a 500 ms future
    target while itself becoming stale after 150 ms. A consumer must compare its
    current time with ``prediction_use_before_time_seconds`` in the same clock.
    No wall-clock, metric speed, physical centroid or robot control is inferred.
    """

    def __init__(self, width: int, height: int, horizons_seconds=(0.1, 0.25, 0.5),
                 max_gap_seconds=0.15, max_prediction_age_seconds=0.15):
        _dimensions(width, height)
        if type(horizons_seconds) not in (tuple, list) or not horizons_seconds:
            raise ValueError('Provide at least one forecast horizon.')
        horizons = [_number(h, 'Forecast horizon', positive=True)
                    for h in horizons_seconds]
        if any(b <= a for a, b in zip(horizons, horizons[1:])):
            raise ValueError('Forecast horizons must be unique and increasing.')
        self.width, self.height = width, height
        self.horizons_seconds = tuple(horizons)
        self.max_gap_seconds = _number(max_gap_seconds, 'Maximum evidence gap', positive=True)
        self.max_prediction_age_seconds = _number(
            max_prediction_age_seconds, 'Maximum prediction age', positive=True)
        self._last_update = None
        self._previous = None
        self._ever_observed = False

    def update(self, time_seconds: float, center_xy_pixels: list[float] | None,
               *, support_count: int = 0, status: str = 'tracked',
               uncertainty_pixels: float | None = None) -> dict:
        """Return a fresh, strict JSON-compatible observation/estimate/forecast row.

        The observed anchor and current estimate are equal in this unsmoothed
        baseline, but are stored separately to preserve their meanings. Each
        velocity carries the two actual observations used to derive it. Forecast
        coordinates stay un-clipped and carry ``out_of_frame`` explicitly.
        Invalid updates leave estimator state unchanged.
        """
        t = _number(time_seconds, 'Acquisition time', minimum=0)
        if self._last_update is not None and t <= self._last_update:
            raise ValueError('Acquisition times must strictly increase.')
        if type(status) is not str or status not in ('tracked', 'lost'):
            raise ValueError('Tracking status must be tracked or lost.')
        count = _support(support_count)
        radius = (None if uncertainty_pixels is None else
                  _number(uncertainty_pixels, 'Anchor uncertainty radius', minimum=0))
        center = None if center_xy_pixels is None else _anchor(
            center_xy_pixels, self.width, self.height)
        if status == 'lost' and center is not None:
            raise ValueError('Lost tracking cannot supply an observed anchor.')
        if center is None and radius is not None:
            raise ValueError('A missing anchor cannot supply an uncertainty radius.')

        row = {
            'kind': KIND, 'schema_version': 1, 'algorithm': ALGORITHM,
            'acquired_time_seconds': t,
            'status': 'lost' if center is None else 'tracked',
            'loss_reason': ('tracking_lost' if status == 'lost' else 'missing_observation')
                           if center is None else None,
            'continuity': 'lost',
            'support_count': count,
            'image_size_pixels': [self.width, self.height],
            'coordinate_frame': 'image',
            'anchor_semantics': 'tracked-image-feature-anchor-not-physical-centroid',
            'units': dict(UNITS),
            'observed_anchor_xy_pixels': None if center is None else list(center),
            'estimated_center_xy_pixels': None if center is None else list(center),
            'anchor_uncertainty_radius_pixels': radius,
            'velocity_xy_pixels_per_second': None,
            'speed_pixels_per_second': None,
            'velocity_evidence': None,
            'forecasts': [],
            'uncertainty_semantics': UNCERTAINTY_SEMANTICS,
            'uncertainty_basis': 'supplied-anchor-radii-propagated-through-finite-difference',
            'prediction_use_before_time_seconds': None,
            'estimator': {'horizons_seconds': list(self.horizons_seconds),
                          'max_gap_seconds': self.max_gap_seconds,
                          'max_prediction_age_seconds': self.max_prediction_age_seconds},
        }
        if center is not None:
            previous = self._previous
            row['continuity'] = 'loss_reset' if self._ever_observed else 'initial'
            if previous is not None:
                previous_t, previous_center, previous_radius = previous
                dt = t - previous_t
                if dt > self.max_gap_seconds:
                    row['continuity'] = 'gap_reset'
                else:
                    row['continuity'] = 'continued'
                    velocity = [_number((b-a)/dt, 'Derived image velocity')
                                for a, b in zip(previous_center, center)]
                    speed = _number(math.hypot(*velocity), 'Derived image speed', minimum=0)
                    row['velocity_xy_pixels_per_second'] = velocity
                    row['speed_pixels_per_second'] = speed
                    row['velocity_evidence'] = {
                        'start_time_seconds': previous_t, 'end_time_seconds': t,
                        'start_anchor_xy_pixels': list(previous_center),
                        'end_anchor_xy_pixels': list(center), 'delta_time_seconds': dt,
                    }
                    for horizon in self.horizons_seconds:
                        target = _number(t+horizon, 'Forecast target time', minimum=0)
                        if target <= t:
                            raise ValueError('Acquisition clock cannot resolve the forecast horizon.')
                        forecast = [_number(c+v*horizon, 'Forecast coordinate')
                                    for c, v in zip(center, velocity)]
                        envelope = None
                        if radius is not None and previous_radius is not None:
                            envelope = _number(radius+horizon*(radius+previous_radius)/dt,
                                               'Forecast uncertainty radius', minimum=0)
                        row['forecasts'].append({
                            'target_time_seconds': target, 'horizon_seconds': horizon,
                            'center_xy_pixels': forecast,
                            'uncertainty_radius_pixels': envelope,
                            'out_of_frame': not (0 <= forecast[0] < self.width
                                                 and 0 <= forecast[1] < self.height),
                        })
                    expiry = _number(t+self.max_prediction_age_seconds,
                                     'Prediction expiry time', minimum=0)
                    if expiry <= t:
                        raise ValueError('Acquisition clock cannot resolve prediction freshness.')
                    row['prediction_use_before_time_seconds'] = expiry

        # Commit only after all derived numeric values have passed validation.
        self._last_update = t
        if center is None:
            self._previous = None
        else:
            self._previous = (t, list(center), radius)
            self._ever_observed = True
        return row


def _strict_equal(actual, expected):
    """JSON equality with exact types, so True cannot impersonate 1 or 1.0."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _strict_equal(actual[k], value) for k, value in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _strict_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_motion_rows(rows, width, height):
    """Validate strict baseline rows and return the unchanged list.

    This stateless validator causally regenerates every row from its actual
    observed anchors, optional radii and embedded estimator configuration. It
    rejects extra fields, conflicting units/evidence, forged forecasts, changed
    configuration, fabricated estimates through losses and non-monotonic clocks.
    It validates internal contract consistency; it cannot authenticate whether a
    camera really observed an anchor or establish independent prediction accuracy.
    """
    _dimensions(width, height)
    if type(rows) is not list:
        raise ValueError('Motion rows must be a list.')
    estimator = None
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise ValueError(f'Motion row {index} must be a data-only object.')
        config = row.get('estimator')
        if type(config) is not dict or set(config) != {
                'horizons_seconds', 'max_gap_seconds', 'max_prediction_age_seconds'}:
            raise ValueError(f'Motion row {index} has invalid estimator configuration.')
        try:
            if estimator is None:
                estimator = MotionEstimator(width, height, **config)
            reason = row.get('loss_reason')
            input_status = 'lost' if reason == 'tracking_lost' else 'tracked'
            expected = estimator.update(row.get('acquired_time_seconds'),
                row.get('observed_anchor_xy_pixels'), support_count=row.get('support_count'),
                status=input_status, uncertainty_pixels=row.get('anchor_uncertainty_radius_pixels'))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'Motion row {index} is invalid: {error}') from None
        if not _strict_equal(row, expected):
            raise ValueError(f'Motion row {index} does not match its causal evidence or strict schema.')
    return rows
