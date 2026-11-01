"""Analytical camera fixtures, never native ground truth passed to tracker."""
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import math
import unittest

import numpy as np

from dvidia_training import piper_rgbd_tracker as tracker


def fixture(*, width=160, height=120, focal=400., centre=(.012, -.009, .12), yaw=.31):
    """Independent top-plane raster: top-down camera, known square in base XY.

    A camera at z=.6 points down, with optical right along base Y and down
    along base X. Pixel centres intersect the horizontal cube top at z=.14.
    No robot state, native masks or estimator helper constructs these frames.
    """
    camera = np.array([[0., 1., 0., .01], [1., 0., 0., -.02], [0., 0., -1., .6], [0., 0., 0., 1.]])
    calibration = dict(base_frame='robot_base', camera_frame='rgbd_optical', depth_units='m',
        intrinsics=dict(width=width, height=height, fx=focal, fy=focal, cx=width/2, cy=height/2),
        base_from_camera=camera.flatten().tolist(), valid_from_seconds=0., valid_until_seconds=2.)
    top = centre[2]+.02; z = .6-top
    row, column = np.indices((height, width))
    world_x = .01+(row+.5-height/2)*z/focal
    world_y = -.02+(column+.5-width/2)*z/focal
    x, y = world_x-centre[0], world_y-centre[1]
    local_x = x*math.cos(yaw)+y*math.sin(yaw)
    local_y = -x*math.sin(yaw)+y*math.cos(yaw)
    mask = (np.abs(local_x) < .02) & (np.abs(local_y) < .02)
    rgb = np.full((height, width, 3), 90, dtype=np.uint8); rgb[mask] = [191, 77, 41]
    depth = np.full((height, width), 2., dtype=np.float32); depth[mask] = z
    return rgb, depth, calibration, np.array(centre), yaw, mask


def run(rgb, depth, calibration, **kwargs):
    return tracker.estimate_pose(rgb, depth, calibration, frame_index=4,
        captured_at_seconds=.2, available_at_seconds=.205, decision_at_seconds=.21, **kwargs)


class RGBDTrackerTests(unittest.TestCase):
    def test_known_square_pixel_centres_base_transform_and_yaw_symmetry(self):
        for yaw in (0., .31, .74, 1.30, 1.88, 3.12):
            with self.subTest(yaw=yaw):
                rgb, depth, calibration, centre, expected_yaw, _ = fixture(yaw=yaw)
                result = run(rgb, depth, calibration)
                self.assertEqual(result['status'], 'available', result)
                np.testing.assert_allclose(result['object_position_estimate_m'], centre, atol=.0015)
                quaternion = result['object_quaternion_estimate_wxyz']
                self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.)
                estimate = 2*math.atan2(quaternion[3], quaternion[0])
                difference = abs((estimate-expected_yaw+math.pi/4) % (math.pi/2)-math.pi/4)
                self.assertLess(difference, math.radians(3))
                self.assertEqual(result['frame_index'], 4)
                self.assertAlmostEqual(result['age_seconds'], .01)
                self.assertGreater(result['object_position_uncertainty_m'], 0)
                self.assertGreater(result['object_orientation_uncertainty_rad'], 0)
                self.assertFalse(result['depth_accuracy_verified'])
                self.assertIn('not calibrated', result['uncertainty_meaning'])
                for name in tracker.SIGNALS:
                    sample = result['signals'][name]
                    self.assertEqual(sample['captured_at_seconds'], .2)
                    self.assertEqual(sample['available_at_seconds'], .205)
                    self.assertEqual(sample['status'], 'available')
                self.assertIsInstance(result['signals']['object_position_uncertainty_m']['value'], list)
                json.dumps(result, allow_nan=False)

    def assert_abstains(self, result, reason=None):
        self.assertNotEqual(result['status'], 'available', result)
        if reason is not None:
            self.assertEqual(result['reason'], reason, result)
        for name in tracker.SIGNALS:
            self.assertIsNone(result[name])
            self.assertEqual(result['signals'][name], dict(status='missing', value=None,
                captured_at_seconds=None, available_at_seconds=None, age_seconds=None))
        json.dumps(result, allow_nan=False)

    def test_absence_and_full_occlusion_do_not_forward_fill(self):
        rgb, depth, calibration, *_ = fixture()
        self.assertEqual(run(rgb, depth, calibration)['status'], 'available')
        rgb[:] = 90
        result = run(rgb, depth, calibration)
        self.assert_abstains(result, 'object_absent_or_too_small')
        self.assertEqual(result['frame_index'], 4)
        self.assertEqual(result['captured_at_seconds'], .2)

    def test_shading_preserves_colour_ratios_but_grey_or_near_black_abstains(self):
        rgb, depth, calibration, centre, _, mask = fixture()
        for colour in ([41, 16, 9], [50, 20, 11], [31, 12, 7]):
            with self.subTest(colour=colour):
                rgb[mask] = colour
                result = run(rgb, depth, calibration)
                self.assertEqual(result['status'], 'available', result)
                np.testing.assert_allclose(result['object_position_estimate_m'], centre, atol=.0015)
        for colour in ([41, 41, 41], [50, 45, 40], [10, 4, 2]):
            rgb[mask] = colour
            self.assert_abstains(run(rgb, depth, calibration), 'object_absent_or_too_small')

    def test_half_occlusion_abstains_instead_of_inventing_centre(self):
        rgb, depth, calibration, _, _, mask = fixture(yaw=0)
        columns = np.where(mask)[1]; split = int(np.median(columns))
        rgb[:, :split] = 90
        depth[:, :split] = .40
        self.assert_abstains(run(rgb, depth, calibration))

    def test_two_same_colour_objects_are_ambiguous(self):
        rgb, depth, calibration, *_ = fixture(centre=(-.02, -.04, .12))
        rgb2, depth2, _, _, _, mask2 = fixture(centre=(.04, .02, .12))
        rgb[mask2] = rgb2[mask2]; depth[mask2] = depth2[mask2]
        self.assert_abstains(run(rgb, depth, calibration), 'ambiguous_colour_components')

    def test_cropped_and_wrong_sized_colour_regions_abstain(self):
        rgb, depth, calibration, *_ = fixture()
        rgb[:] = 90; rgb[0:30, 20:50] = [191, 77, 41]; depth[0:30, 20:50] = .46
        self.assert_abstains(run(rgb, depth, calibration), 'object_cropped_at_image_boundary')
        rgb[:] = 90; rgb[40:110, 30:100] = [191, 77, 41]; depth[40:110, 30:100] = .46
        self.assert_abstains(run(rgb, depth, calibration), 'incomplete_or_wrong_size_top_surface')

    def test_invalid_far_plane_and_non_metric_depth_never_estimates(self):
        rgb, depth, calibration, _, _, mask = fixture()
        for value, reason in ((float('nan'), 'nonfinite_or_nonpositive_depth'),
                              (float('inf'), 'nonfinite_or_nonpositive_depth'),
                              (0., 'nonfinite_or_nonpositive_depth'),
                              (-1., 'nonfinite_or_nonpositive_depth'),
                              (100., 'object_outside_usable_depth'),
                              (.001, 'object_outside_usable_depth')):
            with self.subTest(value=value):
                bad = depth.copy(); bad[mask] = value
                self.assert_abstains(run(rgb, bad, calibration), reason)

    def test_types_dimensions_alignment_and_calibration_are_fail_closed(self):
        rgb, depth, calibration, *_ = fixture()
        for bad_rgb, bad_depth in ((rgb.astype(float), depth), (rgb[..., 0], depth),
                                   (rgb[..., :2], depth), (rgb, depth.astype(np.uint16)),
                                   (rgb, depth[:-1]), (rgb.tolist(), depth)):
            with self.subTest(rgb_type=type(bad_rgb), depth_shape=getattr(bad_depth, 'shape', None)):
                self.assert_abstains(run(bad_rgb, bad_depth, calibration))
        for mutate in (lambda c: c.update(depth_units='mm'),
                       lambda c: c.update(base_frame='world'),
                       lambda c: c['intrinsics'].update(fx=0),
                       lambda c: c['intrinsics'].update(fy=float('nan')),
                       lambda c: c['intrinsics'].update(width=True),
                       lambda c: c['intrinsics'].update(width=161),
                       lambda c: c['base_from_camera'].__setitem__(0, 1),
                       lambda c: c['base_from_camera'].__setitem__(15, 0),
                       lambda c: c.update(valid_until_seconds=.205),
                       lambda c: c.update(valid_from_seconds=.201)):
            bad = deepcopy(calibration); mutate(bad)
            self.assert_abstains(run(rgb, depth, bad))

    def test_time_order_boundary_source_id_and_json_safe_rejections(self):
        rgb, depth, calibration, *_ = fixture()
        def timed(**changes):
            values = dict(frame_index=4, captured_at_seconds=.2,
                available_at_seconds=.205, decision_at_seconds=.21)
            values.update(changes)
            return tracker.estimate_pose(rgb, depth, calibration, **values)
        self.assertEqual(timed(decision_at_seconds=.3)['status'], 'available')
        self.assertEqual(timed(decision_at_seconds=.3+.5e-12)['status'], 'available')
        self.assert_abstains(timed(decision_at_seconds=.3+2e-12), 'frame_older_than_100ms')
        self.assert_abstains(timed(decision_at_seconds=.300001), 'frame_older_than_100ms')
        for changes in ({'captured_at_seconds':.22}, {'available_at_seconds':.19},
                        {'available_at_seconds':.211}, {'captured_at_seconds':float('nan')},
                        {'captured_at_seconds':True}, {'captured_at_seconds':10**1000}, {'captured_at_seconds':-.1},
                        {'frame_index':True}, {'frame_index':-1}, {'frame_index':4.0}):
            self.assert_abstains(timed(**changes))

    def test_tilted_top_plane_refuses_upright_assumption(self):
        rgb, depth, calibration, centre, _, mask = fixture(yaw=0)
        # Intersect the tilted plane z = top + tan(20deg)*(base_x-centre_x)
        # with the declared top-down perspective rays; the colour footprint
        # stays intentionally generous, so plane orientation must reject it.
        rows, _ = np.indices(depth.shape)
        dx_direction = (rows+.5-60)/400
        slope = math.tan(math.radians(20))
        depth[mask] = ((.6-(centre[2]+.02)-slope*(.01-centre[0])) /
                       (1+slope*dx_direction))[mask]
        self.assert_abstains(run(rgb, depth, calibration), 'outside_upright_profile')

    def test_no_native_truth_inputs_imports_or_stateful_estimate_history(self):
        parameters = inspect.signature(tracker.estimate_pose).parameters
        self.assertEqual(set(parameters), {'rgb', 'depth', 'calibration', 'frame_index',
            'captured_at_seconds', 'available_at_seconds', 'decision_at_seconds', 'config'})
        source = inspect.getsource(tracker)
        self.assertNotIn('import mujoco', source)
        self.assertNotIn('piper_task', source)
        self.assertNotIn('native_info', source)
        rgb, depth, calibration, *_ = fixture()
        with self.assertRaises(TypeError):
            run(rgb, depth, calibration, native_object_position=[0., 0., 0.])
        before_rgb, before_depth, before_calibration = rgb.copy(), depth.copy(), deepcopy(calibration)
        first = run(rgb, depth, calibration); second = run(rgb, depth, calibration)
        first['diagnostics'].pop('processing_seconds'); second['diagnostics'].pop('processing_seconds')
        self.assertEqual(first, second)
        np.testing.assert_array_equal(rgb, before_rgb); np.testing.assert_array_equal(depth, before_depth)
        self.assertEqual(calibration, before_calibration)

    def test_configuration_is_frozen_and_bounded(self):
        for change in ({'side_m':.08}, {'maximum_age_seconds':1.}, {'minimum_top_pixels':True},
                       {'maximum_tilt_rad':float('nan')}, {'minimum_visibility':1.1}):
            with self.assertRaises(ValueError):
                replace(tracker.TrackerConfig(), **change)

    def test_large_colour_field_is_bounded_before_component_walk(self):
        rgb, depth, calibration, *_ = fixture(width=320, height=240)
        rgb[:] = [191, 77, 41]
        self.assert_abstains(run(rgb, depth, calibration), 'colour_support_exceeds_bounded_profile')


if __name__ == '__main__':
    unittest.main()
