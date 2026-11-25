"""The winning inliers determine the fitted plane and subsequent cube pose."""
import math
import unittest

import numpy as np

from dvidia_training import piper_rgbd_tracker as tracker


def reference_face(points, config):
    """Scalar triangle scoring used by the original tracker at 065a10f."""
    candidates = points[points[:, 2] >= np.median(points[:, 2])]
    if len(candidates) < config.minimum_top_pixels:
        return None
    triples = np.random.default_rng(701).integers(0, len(candidates), (96, 3))
    best_count = config.minimum_top_pixels - 1
    best = None
    for triple in triples:
        a, b, c = candidates[triple]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        if normal[2] < 0:
            normal *= -1
        if normal[2] < math.cos(math.radians(45)):
            continue
        selected = np.abs((points - a) @ normal) <= config.plane_tolerance_m
        count = int(selected.sum())
        if count > best_count:
            best_count = count
            best = points[selected]
    if best is None:
        return None
    singular = np.linalg.svd(best - best.mean(axis=0), full_matrices=False)[1]
    if len(singular) < 3 or singular[1] < .001:
        return None
    return best


class TrackerPlaneParityTests(unittest.TestCase):
    def test_exact_selected_points_for_ties_outliers_thresholds_and_degenerate_clouds(self):
        config = tracker.TrackerConfig()
        rng = np.random.default_rng(15783)
        clouds = [np.ones((5, 3)), np.ones((100, 3))]
        for count in (27, 130, 333):
            xy = rng.uniform(-.02, .02, (count, 2))
            heights = np.full(count, .42)
            clouds.append(np.column_stack((xy, heights)))
            heights[:count // 3] -= .002
            heights[count // 3:count // 2] -= config.plane_tolerance_m
            clouds.append(np.column_stack((xy, heights)))
            for tilt in (3., 44.9999, 45.0001):
                slope = math.tan(math.radians(tilt))
                clouds.append(np.column_stack((xy, heights + xy[:, 0] * slope)))
        observed = set()
        for index, cloud in enumerate(clouds):
            with self.subTest(index=index):
                expected = reference_face(cloud, config)
                actual = tracker._top_plane(cloud, config)
                observed.add(expected is not None)
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertIsNotNone(actual)
                    np.testing.assert_array_equal(actual[0], expected)
        self.assertEqual(observed, {False, True})


if __name__ == '__main__':
    unittest.main()
