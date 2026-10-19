"""Pixel geometry and independent drive-channel tests for sensor observers."""
from copy import deepcopy
from dataclasses import replace
import math
import os
import unittest

import numpy as np

from dvidia_training import sensor_goal_observers as sensors


def ellipse_image(centers=((.44, -.12, 0.),), calibration=sensors.CameraCalibration()):
    c = calibration
    ys, xs = np.mgrid[:c.height, :c.width]
    x = (xs+.5-c.width/2)*c.vertical_span_m/c.height+c.center_xy_m[0]
    y = (c.height/2-ys-.5)*c.vertical_span_m/c.height+c.center_xy_m[1]
    rgb = np.full((c.height, c.width, 3), 180, dtype=np.uint8)
    for cx, cy, yaw in centers:
        along = (x-cx)*math.cos(yaw)+(y-cy)*math.sin(yaw)
        across = -(x-cx)*math.sin(yaw)+(y-cy)*math.cos(yaw)
        rgb[(along/.115)**2+(across/.0235)**2 <= 1] = [30, 130, 60]
    return rgb


def packet():
    return {"schema": sensors.ACTUATOR_SCHEMA, "units": dict(sensors.ACTUATOR_UNITS), "t_s": .6,
            "arm_position_rad": [0.]*6, "arm_velocity_rad_s": [0.]*6,
            "arm_command_rad": [0.]*6, "arm_effort_nm": [0.]*6,
            "finger_position_m": [.035, -.035], "finger_velocity_m_s": [0., 0.],
            "gripper_command_m": .035, "gripper_effort_n": 0.}


class RGBObserverTests(unittest.TestCase):
    def test_calibrated_centroid_y_sign_scale_and_axial_angle(self):
        for x, y, yaw in ((.44, -.12, .0), (.50, -.19, .4), (.31, -.02, -1.2)):
            o = sensors.observe_rgb(ellipse_image(((x, y, yaw),)), t_s=.6)
            self.assertEqual(o['status'], 'observed')
            self.assertLess(math.dist(o['position_xy_m'], [x, y]), .002)
            error = abs((o['yaw_axis_rad']-yaw+math.pi/2) % math.pi-math.pi/2)
            self.assertLess(error, .01)
            self.assertEqual(o['image_dimensions_px'], [640, 480])
            self.assertNotIn('complete', o)
            self.assertNotIn('presence', o)

    def test_half_turn_is_a_declared_alias(self):
        a = sensors.observe_rgb(ellipse_image(((.44, -.12, .1),)), t_s=1.)
        b = sensors.observe_rgb(ellipse_image(((.44, -.12, .1+math.pi),)), t_s=1.)
        self.assertEqual(a['position_xy_m'], b['position_xy_m'])
        self.assertAlmostEqual(a['yaw_axis_rad'], b['yaw_axis_rad'])

    def test_empty_occluded_ambiguous_and_clipped_images_are_unknown(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        multi = ellipse_image(((.44, -.23, 0.), (.44, .04, 0.)))
        clipped = ellipse_image(((.09, -.12, 0.),))
        for image in (blank, multi, clipped):
            o = sensors.observe_rgb(image, t_s=1.)
            self.assertEqual(o['status'], 'unknown')
            self.assertIsNone(o['position_xy_m'])
            self.assertNotIn('absent', o.values())
        self.assertEqual(sensors.observe_rgb(multi, t_s=1.)['reason'], 'multiple_proxy_candidates')

    def test_bad_image_timestamp_and_calibration_do_not_become_measurements(self):
        for image in (None, np.zeros((480, 640, 3)), np.zeros((100, 100, 3), dtype=np.uint8)):
            self.assertEqual(sensors.observe_rgb(image, t_s=.6)['status'], 'unknown')
        for time in (True, float('nan'), float('inf'), 10**400, -.1):
            self.assertEqual(sensors.observe_rgb(ellipse_image(), t_s=time)['status'], 'unknown')
        for kwargs in ({'width': True}, {'center_xy_m': (float('nan'), 0)}, {'vertical_span_m': 10**400}, {'calibration_id': ''}):
            with self.assertRaises(ValueError): sensors.CameraCalibration(**kwargs)

    def test_temporal_displacement_bound_detects_large_motion_not_rest(self):
        moving = [{'t_s': i*.1, 'rgb': ellipse_image(((.35+i*.07, -.12, 0.),))} for i in range(3)]
        result = sensors.observe_rgb_sequence(moving)
        self.assertEqual(result['status'], 'observed')
        self.assertGreater(result['net_average_speed_interval_m_s'][0], .01)
        still = [{'t_s': i*.1, 'rgb': ellipse_image()} for i in range(3)]
        result = sensors.observe_rgb_sequence(still)
        self.assertEqual(result['net_average_speed_interval_m_s'][0], 0.)
        self.assertGreater(result['net_average_speed_interval_m_s'][1], .01)
        self.assertNotIn('stationary', result)

    def test_sequence_rejects_insufficient_invalid_extra_and_unobservable_frames(self):
        good = [{'t_s': i*.1, 'rgb': ellipse_image()} for i in range(3)]
        cases = [good[:1], [{'t_s': 0., 'rgb': ellipse_image()}, {'t_s': 0., 'rgb': ellipse_image()}],
                 [{'t_s': 0., 'rgb': ellipse_image()}, {'t_s': .2, 'rgb': ellipse_image()}],
                 [dict(good[0], goal=[0, 0]), good[1]],
                 [good[0], {'t_s': .1, 'rgb': np.zeros((480, 640, 3), dtype=np.uint8)}]]
        for frames in cases:
            self.assertEqual(sensors.observe_rgb_sequence(frames)['status'], 'unknown')

    def test_calibration_drift_changes_estimate_without_detectable_self_authentication(self):
        image = ellipse_image()
        a = sensors.observe_rgb(image, t_s=1.)
        b = sensors.observe_rgb(image, t_s=1., calibration=replace(sensors.CameraCalibration(), center_xy_m=(.54, -.12)))
        self.assertAlmostEqual(b['position_xy_m'][0]-a['position_xy_m'][0], .1)
        self.assertEqual(b['status'], 'observed')


class ActuatorObserverTests(unittest.TestCase):
    def test_signed_finger_positions_and_single_finger_command(self):
        o = sensors.observe_actuators(packet())
        self.assertEqual(o['status'], 'observed')
        self.assertAlmostEqual(o['jaw_gap_m'], .07)
        self.assertAlmostEqual(o['gripper_command_m'], .035)
        self.assertEqual(o['servo_consistency'], 'compatible')
        self.assertEqual(o['servo_residual_n'], 0.)
        self.assertNotIn('released', o)
        self.assertNotIn('supported', o)

    def test_effort_spoof_disagrees_with_encoder_and_command_but_consistent_forgery_survives(self):
        held = packet(); held.update(finger_position_m=[.02, -.02], gripper_command_m=0., gripper_effort_n=-.8)
        h = sensors.observe_actuators(held)
        self.assertEqual(h['servo_consistency'], 'compatible')
        self.assertAlmostEqual(h['jaw_gap_m'], .04)
        held['gripper_effort_n'] = 0.
        self.assertEqual(sensors.observe_actuators(held)['servo_consistency'], 'contradiction')
        # Internally consistent observations cannot authenticate the real scene.
        self.assertEqual(sensors.observe_actuators(packet())['servo_consistency'], 'compatible')

    def test_channel_faults_units_labels_and_nonfinite_values_are_not_silently_used(self):
        cases = []
        p = packet(); p['units']['arm_effort'] = 'N'; cases.append(p)
        p = packet(); p['physical_truth'] = True; cases.append(p)
        p = packet(); p['finger_position_m'] = [.035, .035]; cases.append(p)
        for field in ('gripper_effort_n', 't_s', 'gripper_command_m'):
            for value in (True, 10**400, float('nan'), float('inf')):
                p = packet(); p[field] = value; cases.append(p)
        p = packet(); p['arm_position_rad'][2] = True; cases.append(p)
        p = packet(); del p['arm_effort_nm']; cases.append(p)
        for p in cases: self.assertEqual(sensors.observe_actuators(p)['status'], 'unknown')

    def test_negative_aperture_bounds_never_invert(self):
        p = packet(); p.update(finger_position_m=[-.0009, .0009], gripper_command_m=0., gripper_effort_n=.036)
        self.assertEqual(sensors.observe_actuators(p)['reason'], 'negative_aperture_outside_uncertainty')
        p.update(finger_position_m=[-.0001, .0001], gripper_effort_n=.004)
        o = sensors.observe_actuators(p)
        self.assertEqual(o['status'], 'observed')
        self.assertEqual(o['jaw_gap_interval_m'][0], 0.)
        self.assertGreaterEqual(o['jaw_gap_interval_m'][1], o['jaw_gap_interval_m'][0])


@unittest.skipUnless(os.environ.get('DVIDIA_ROBOT_PROFILE_ASSETS'), 'Set pinned native PiPER assets')
class NativeSensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dvidia_training import local_goal_scenes as s
        cls.s = s; cls.assets = os.environ['DVIDIA_ROBOT_PROFILE_ASSETS']
        cls.layout = s.make_layouts(2026101904, 1)[0]
        cls.arm = s.frozen._initial_arm(cls.assets)

    def test_native_projection_without_segmentation_or_object_projection(self):
        mj, _ = self.s.sim._native_imports()
        # Physical marker positions are test oracles, never observer inputs.
        for x, y in ((.52, -.20), (.31, -.02)):
            model = mj.MjModel.from_xml_string(f'''<mujoco><visual><global offwidth="640" offheight="480"/><headlight ambient=".8 .8 .8" diffuse=".2 .2 .2"/></visual><worldbody><geom type="ellipsoid" pos="{x} {y} .05" size=".115 .0235 .03" rgba=".12 .5 .24 1"/></worldbody></mujoco>''')
            data = mj.MjData(model); mj.mj_forward(model, data)
            original = (model.vis.global_.orthographic, model.vis.global_.fovy, model.vis.global_.ipd)
            image = sensors.render_overhead_rgb(model, data)
            self.assertEqual(original, (model.vis.global_.orthographic, model.vis.global_.fovy, model.vis.global_.ipd))
            observed = sensors.observe_rgb(image, t_s=0.)
            self.assertEqual(observed['status'], 'observed')
            self.assertLess(math.dist(observed['position_xy_m'], [x, y]), .002)

    def test_native_open_held_occluded_and_velocity_alias_read_separate_channels(self):
        complete = self.s._run(self.assets, self.layout, 'complete', self.arm)
        held = self.s._run(self.assets, self.layout, 'unreleased', self.arm)
        moving = self.s._run(self.assets, self.layout, 'moving_alias', self.arm)
        covered = self.s._run(self.assets, self.layout, 'occluded', self.arm)
        open_drive = sensors.read_native_actuators(complete[0], complete[1])
        moving_drive = sensors.read_native_actuators(moving[0], moving[1])
        self.assertEqual(open_drive, moving_drive)
        open_features = sensors.observe_actuators(open_drive)
        held_features = sensors.observe_actuators(sensors.read_native_actuators(held[0], held[1]))
        self.assertGreater(open_features['jaw_gap_interval_m'][0], .06)
        self.assertLess(held_features['jaw_gap_interval_m'][1], .06)
        self.assertGreater(abs(held_features['gripper_effort_n']), .1)
        self.assertEqual(open_features['servo_consistency'], 'compatible')
        self.assertEqual(held_features['servo_consistency'], 'compatible')
        rgb = sensors.render_overhead_rgb(complete[0], complete[1])
        alias = sensors.render_overhead_rgb(moving[0], moving[1])
        np.testing.assert_array_equal(rgb, alias)
        c = sensors.observe_rgb(rgb, t_s=complete[1].time)
        self.assertEqual(c['status'], 'observed')
        self.assertLess(math.dist(c['position_xy_m'], complete[3]['target_position_m'][:2]), c['position_bound_m'])
        hidden = sensors.observe_rgb(sensors.render_overhead_rgb(covered[0], covered[1]), t_s=covered[1].time)
        self.assertEqual(hidden['status'], 'unknown')


if __name__ == '__main__': unittest.main()
