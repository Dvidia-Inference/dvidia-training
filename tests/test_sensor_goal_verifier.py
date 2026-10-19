"""Conservative fusion behavior for separately processed synthetic channels."""
import math
import unittest

import numpy as np

from dvidia_training.local_goal_verifier import GoalContract
from dvidia_training.sensor_goal_observers import ACTUATOR_SCHEMA, ACTUATOR_UNITS, CameraCalibration
from dvidia_training.sensor_goal_verifier import CAMERA, DRIVE, camera_window, actuator_window, verify_inputs


def green_proxy(x=.44, y=-.12, yaw=0.):
    c = CAMERA
    yy, xx = np.mgrid[:c.height, :c.width]
    scale = c.vertical_span_m/c.height
    px = (xx+.5-c.width/2)*scale+c.center_xy_m[0]
    py = (c.height/2-yy-.5)*scale+c.center_xy_m[1]
    along=(px-x)*math.cos(yaw)+(py-y)*math.sin(yaw)
    across=-(px-x)*math.sin(yaw)+(py-y)*math.cos(yaw)
    rgb=np.full((c.height,c.width,3),180,dtype=np.uint8)
    rgb[(along/.115)**2+(across/.0235)**2<=1]=[30,130,60]
    return rgb


def primary(complete=True):
    rows=[]
    for i in range(100):
        t=.402+i*.002
        rows.append(dict(t_s=t,object_id='green-shoe',frame_id='workstation_world',presence='present',
            position_m=[.44,-.12,0.],yaw_rad=0.,linear_speed_m_s=0.,angular_speed_rad_s=0.,
            table_support_force_n=.2943,robot_contact_force_n=0.))
    if not complete: rows[-1]['robot_contact_force_n']=.2
    return dict(schema='dvidia.local-goal-observations/v1',units=dict(position='m',angle='rad',time='s',
        linear_speed='m/s',angular_speed='rad/s',force='N'),sample_period_s=.002,
        uncertainty_bounds={k:0. for k in ('position_m','yaw_rad','linear_speed_m_s','angular_speed_rad_s','table_support_force_n','robot_contact_force_n')},
        history_start_s=.4,samples=rows)


def camera_packet():
    return dict(schema='dvidia.sensor-camera-window/v1',source_id='overhead-rgb',
        calibration_id=CAMERA.calibration_id,frames=[dict(t_s=.4+i*.02,image=f'{i}.png') for i in range(11)])


def actuator_sample(t, *, open_jaw=True, effort=None):
    q=.035 if open_jaw else .02
    command=.035 if open_jaw else 0.
    expected=40*(command-q)
    return dict(schema=ACTUATOR_SCHEMA,units=dict(ACTUATOR_UNITS),t_s=t,
        arm_position_rad=[0.]*6,arm_velocity_rad_s=[0.]*6,arm_command_rad=[0.]*6,
        arm_effort_nm=[0.]*6,finger_position_m=[q,-q],finger_velocity_m_s=[0.,0.],
        gripper_command_m=command,gripper_effort_n=expected if effort is None else effort)


def drive_packet(*, open_jaw=True, bad_effort=False):
    return dict(schema='dvidia.sensor-drive-window/v1',source_id='piper-drive-bus-v1',
        calibration_id=DRIVE.calibration_id,history_start_s=.4,
        samples=[actuator_sample(.402+i*.002,open_jaw=open_jaw,effort=.5 if bad_effort else None) for i in range(100)])


class SensorGoalVerifierTests(unittest.TestCase):
    def setUp(self):
        self.contract=GoalContract('green-shoe','workstation_world',(.44,-.12),0.)
        self.rgb=[green_proxy() for _ in range(11)]

    def run_inputs(self, p=None, c=None, a=None, images=None):
        return verify_inputs(p or primary(),c or camera_packet(),a or drive_packet(),
            images if images is not None else self.rgb,contract=self.contract,now_s=.6)

    def test_three_compatible_channels_preserve_complete(self):
        result=self.run_inputs()
        self.assertEqual(result['baseline']['verdict'],'complete')
        self.assertEqual(result['camera']['status'],'observed')
        self.assertTrue(result['actuator']['release_corroborated'])
        self.assertEqual(result['fused']['verdict'],'complete')

    def test_camera_motion_conflict_abstains_without_rewriting_primary(self):
        moving=[green_proxy(x=.44+.025*i,y=-.12) for i in range(11)]
        result=self.run_inputs(images=moving)
        self.assertEqual(result['baseline']['verdict'],'complete')
        self.assertEqual(result['fused']['verdict'],'unknown')
        self.assertIn('camera_motion_contradiction',result['fused']['reasons'])

    def test_closed_gripper_and_inconsistent_effort_abstain(self):
        for packet in (drive_packet(open_jaw=False),drive_packet(bad_effort=True)):
            result=self.run_inputs(a=packet)
            self.assertEqual(result['baseline']['verdict'],'complete')
            self.assertEqual(result['fused']['verdict'],'unknown')
            self.assertTrue(result['actuator']['reasons'])

    def test_missing_occluded_stale_and_misaligned_evidence_fail_closed(self):
        cases=[]
        c=camera_packet();c['frames']=[];cases.append((c,drive_packet(),self.rgb))
        cases.append((camera_packet(),drive_packet(),[np.zeros((480,640,3),dtype=np.uint8) for _ in range(11)]))
        c=camera_packet();c['frames']=[dict(f,t_s=f['t_s']-.5) for f in c['frames']];cases.append((c,drive_packet(),self.rgb))
        c=camera_packet();c['frames']=[dict(f,t_s=f['t_s']+.01) for f in c['frames']];cases.append((c,drive_packet(),self.rgb))
        for c,a,images in cases:
            with self.subTest(camera=c):
                out=self.run_inputs(c=c,a=a,images=images)
                self.assertEqual(out['baseline']['verdict'],'complete')
                self.assertEqual(out['fused']['verdict'],'unknown')

    def test_noncomplete_primary_is_never_upgraded_or_changed(self):
        out=self.run_inputs(p=primary(complete=False))
        self.assertEqual(out['baseline']['verdict'],'incomplete')
        self.assertEqual(out['fused']['verdict'],'incomplete')

    def test_calibration_label_is_not_authentication(self):
        c=camera_packet();c['calibration_id']='fabricated-but-correct-label'
        self.assertEqual(camera_window(c,self.rgb,now_s=.6)['status'],'unknown')
        # This implementation validates declared IDs only; limitation stays public.
        self.assertTrue(any('authenticat' in text for text in self.run_inputs()['fused']['limitations']))


if __name__=='__main__': unittest.main()
