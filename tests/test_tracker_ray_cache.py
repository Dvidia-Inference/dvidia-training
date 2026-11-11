"""Exact calibration-cache equivalence; synthetic analytical data only."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_rgbd_tracker as tracker
from tests.test_piper_rgbd_tracker import fixture


def uncached_cube_rays(shape, intrinsics, transform, centre, yaw, side):
    """Frozen pre-cache equations, retained as the independent comparison."""
    fx, fy, cx, cy = intrinsics
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


def geometry(shape, intrinsics, transform):
    return tracker._camera_ray_geometry(tuple(shape),
        np.asarray(intrinsics, dtype=np.float64).tobytes(),
        np.asarray(transform[:3, :3], dtype=np.float64).tobytes())


def estimate(rgb, depth, calibration, frame):
    result = tracker.estimate_pose(rgb, depth, calibration, frame_index=frame,
        captured_at_seconds=.2, available_at_seconds=.205, decision_at_seconds=.21)
    # Only local computation duration is nondeterministic; acquisition/age and
    # every numeric estimate, status and diagnostic remain in the comparison.
    result['diagnostics'].pop('processing_seconds')
    return result


class TrackerRayCacheTests(unittest.TestCase):
    def setUp(self):
        tracker._camera_ray_geometry.cache_clear()
        self.addCleanup(tracker._camera_ray_geometry.cache_clear)

    def test_randomized_ray_hits_and_depth_exactly_match_uncached_equations(self):
        rng = np.random.default_rng(32101)
        total_hits = 0
        for index in range(40):
            height, width = (int(x) for x in rng.integers(24, 180, size=2))
            intrinsics = (float(rng.uniform(90, 500)), float(rng.uniform(90, 500)),
                          width/2+float(rng.uniform(-3, 3)), height/2+float(rng.uniform(-3, 3)))
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0: rotation[:, 0] *= -1
            if index % 5 == 0: rotation = np.eye(3)
            transform = np.eye(4); transform[:3, :3] = rotation
            transform[:3, 3] = rng.uniform(-.25, .25, size=3)
            centre = transform[:3, 3]+rotation @ np.array([rng.uniform(-.01, .01),
                rng.uniform(-.01, .01), rng.uniform(.18, .6)])
            yaw = float(rng.uniform(-math.pi, math.pi))
            args = ((height, width), intrinsics, transform, centre, yaw, .04)
            expected = uncached_cube_rays(*args)
            for _ in range(2):
                actual = tracker._cube_rays(*args)
                for observed, reference in zip(actual, expected):
                    np.testing.assert_array_equal(observed, reference)
            total_hits += int(expected[0].sum())
        self.assertGreater(total_hits, 100)

    def test_calibration_changes_miss_but_object_pose_is_never_cached(self):
        transform = np.eye(4); shape = (32, 48); intrinsics = (100., 110., 24., 16.)
        geometry(shape, intrinsics, transform)
        geometry(shape, intrinsics, transform)
        self.assertEqual(tracker._camera_ray_geometry.cache_info().hits, 1)
        for index in range(4):
            changed = list(intrinsics); changed[index] += .001
            misses = tracker._camera_ray_geometry.cache_info().misses
            geometry(shape, changed, transform)
            self.assertEqual(tracker._camera_ray_geometry.cache_info().misses, misses+1)
        for changed_shape in ((33, 48), (32, 49)):
            misses = tracker._camera_ray_geometry.cache_info().misses
            geometry(changed_shape, intrinsics, transform)
            self.assertEqual(tracker._camera_ray_geometry.cache_info().misses, misses+1)
        theta = .31; changed_rotation = np.eye(4)
        changed_rotation[:3, :3] = [[math.cos(theta), 0, math.sin(theta)],
                                   [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]]
        misses = tracker._camera_ray_geometry.cache_info().misses
        geometry(shape, intrinsics, changed_rotation)
        self.assertEqual(tracker._camera_ray_geometry.cache_info().misses, misses+1)
        # Translation changes the per-frame origin, not the direction cache.
        changed_rotation[:3, 3] = [.02, -.01, .10]
        for centre, yaw in ((np.array([.02, 0., .4]), .1), (np.array([-.01, .03, .3]), -.5)):
            args = (shape, intrinsics, changed_rotation, centre, yaw, .04)
            for a, b in zip(tracker._cube_rays(*args), uncached_cube_rays(*args)):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(tracker._camera_ray_geometry.cache_info().misses, misses+1)

    def test_cache_retains_at_most_two_entries_and_is_irreversibly_read_only(self):
        shape=(32, 48); intrinsics=(100., 110., 24., 16.); transform=np.eye(4)
        array = geometry(shape, intrinsics, transform)
        self.assertFalse(array.flags.writeable)
        with self.assertRaises(ValueError): array[0, 0, 0] = 999
        with self.assertRaises(ValueError): array.setflags(write=True)
        with self.assertRaises(ValueError): array.base.setflags(write=True)
        geometry(shape, (101., 110., 24., 16.), transform)
        geometry(shape, (102., 110., 24., 16.), transform)
        info = tracker._camera_ray_geometry.cache_info()
        self.assertEqual((info.maxsize, info.currsize), (2, 2))
        geometry(shape, intrinsics, transform)
        self.assertEqual(tracker._camera_ray_geometry.cache_info().misses, info.misses+1)
        # Every permitted entry's retained array bytes obey the documented cap.
        biggest = geometry((480, 640), (500., 500., 320., 240.), transform)
        self.assertEqual(biggest.nbytes*2, 14745600)
        with self.assertRaises(ValueError): geometry((481, 640), intrinsics, transform)

    def test_current_frame_missing_occluded_and_invalid_calibration_still_match(self):
        rgb, depth, calibration, _, _, mask = fixture(yaw=0)
        full = estimate(rgb, depth, calibration, 4)
        self.assertEqual(full['status'], 'available')
        variants = [(rgb, depth, calibration)]
        blank = np.full_like(rgb, 90)
        variants.append((blank, depth, calibration))
        occluded_rgb, occluded_depth = rgb.copy(), depth.copy()
        split = int(np.median(np.where(mask)[1]))
        occluded_rgb[:, :split] = 90; occluded_depth[:, :split] = .40
        variants.append((occluded_rgb, occluded_depth, calibration))
        expired = deepcopy(calibration); expired['valid_until_seconds'] = .205
        variants.append((rgb, depth, expired))
        for index, values in enumerate(variants):
            cached = estimate(*values, frame=5+index)
            with patch.object(tracker, '_cube_rays', uncached_cube_rays):
                expected = estimate(*values, frame=5+index)
            self.assertEqual(cached, expected)
            if index:
                self.assertNotEqual(cached['status'], 'available')
                for name in tracker.SIGNALS:
                    self.assertEqual(cached['signals'][name]['status'], 'missing')


if __name__ == '__main__': unittest.main()
