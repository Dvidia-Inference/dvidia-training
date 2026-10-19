"""Corroborate a primary placement packet using separately processed RGB and drives.

These synthetic channels share one simulator. A pass remains conditional on the
primary support/contact and identity assertions; this is not sensor authentication.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from .local_goal_verifier import GoalContract, verify_goal
from .sensor_goal_observers import (CameraCalibration, ActuatorCalibration,
    observe_rgb_sequence, observe_actuators)

CAMERA = CameraCalibration()
DRIVE = ActuatorCalibration()
TOL = 1e-6
RELEASE = dict(jaw_gap_min_m=.060, command_min_m=.030,
               effort_abs_max_n=.10, jaw_rate_abs_max_m_s=.01)
LIMITATIONS = [
    'Separate RGB/drive processing shares the same simulator; no physical sensor independence is established.',
    'A green elongated candidate is not authenticated target identity. Axial yaw cannot distinguish a 180 degree flip.',
    'Orthographic RGB does not measure height or table support. Drive effort is not target contact force.',
    'Compatible auxiliary observations do not prove that the primary support/contact assertions are true.',
    'Freshly timestamped consistent replay/common-mode forgery and motion between camera samples may pass.',
    'Uncertainty bounds and calibration IDs are declared synthetic assumptions, not authenticated or hardware calibrated.',
]


def specification():
    return dict(camera=asdict(CAMERA), actuator=asdict(DRIVE), release=RELEASE.copy(),
        camera_source_id='overhead-rgb', drive_source_id='piper-drive-bus-v1',
        camera_frames=11, camera_period_s=.020, drive_samples=100, drive_period_s=.002,
        history_s=.200, maximum_age_s=.100, time_tolerance_s=TOL,
        policy='Primary incomplete/unknown is retained. Primary complete requires valid aligned RGB and drive windows, compatible pose/motion, and release corroboration; any refusal returns unknown.',
        limitations=LIMITATIONS)


def _number(x):
    return type(x) in (int,float) and 0 <= x <= 1e9 and math.isfinite(x)


def _times(rows, count, period, start, now_s):
    if type(rows) is not list or len(rows) != count or not _number(now_s) or not _number(start):
        return False
    for i,row in enumerate(rows):
        if type(row) is not dict or not _number(row.get('t_s')) or abs(row['t_s']-(start+i*period)) > TOL:
            return False
    return -TOL <= now_s-rows[-1]['t_s'] <= .100+TOL


def camera_window(packet, decoded_frames, *, now_s):
    invalid = dict(status='unknown', reason='invalid_camera_window', observations=[])
    if (type(packet) is not dict or set(packet) != {'schema','source_id','calibration_id','frames'}
        or packet['schema'] != 'dvidia.sensor-camera-window/v1'
        or packet['source_id'] != 'overhead-rgb' or packet['calibration_id'] != CAMERA.calibration_id):
        return invalid
    frames = packet['frames']
    if (type(frames) is not list or not frames or any(type(f) is not dict or set(f)!={'t_s','image'} for f in frames)
        or not _times(frames,11,.020,frames[0].get('t_s'),now_s)):
        return invalid
    if type(decoded_frames) is not list or len(decoded_frames) != len(frames):
        return dict(invalid, reason='missing_or_invalid_camera_pixels')
    return observe_rgb_sequence([dict(t_s=f['t_s'],rgb=rgb) for f,rgb in zip(frames,decoded_frames)], calibration=CAMERA)


def actuator_window(packet, *, now_s):
    invalid = dict(status='unknown',reason='invalid_actuator_window',observations=[])
    if (type(packet) is not dict or set(packet) != {'schema','source_id','calibration_id','history_start_s','samples'}
        or packet['schema'] != 'dvidia.sensor-drive-window/v1'
        or packet['source_id'] != 'piper-drive-bus-v1' or packet['calibration_id'] != DRIVE.calibration_id
        or not _number(packet['history_start_s'])
        or not _times(packet['samples'],100,.002,packet['history_start_s']+.002,now_s)):
        return invalid
    observations = [observe_actuators(s,calibration=DRIVE) for s in packet['samples']]
    if any(o['status'] != 'observed' for o in observations):
        return dict(invalid,reason='invalid_actuator_sample',observations=observations)
    reasons = set()
    for o in observations:
        if o['servo_consistency'] != 'compatible': reasons.add('drive_servo_contradiction')
        if abs(o['finger_symmetry_residual_m']) > o['finger_symmetry_bound_m']: reasons.add('finger_symmetry_contradiction')
        if o['jaw_gap_interval_m'][0] < RELEASE['jaw_gap_min_m']: reasons.add('jaw_opening_not_corroborated')
        if o['gripper_command_interval_m'][0] < RELEASE['command_min_m']: reasons.add('release_command_not_corroborated')
        if max(abs(x) for x in o['gripper_effort_interval_n']) > RELEASE['effort_abs_max_n']: reasons.add('low_drive_effort_not_corroborated')
        if max(abs(x) for x in o['jaw_gap_rate_interval_m_s']) > RELEASE['jaw_rate_abs_max_m_s']: reasons.add('settled_gripper_not_corroborated')
    return dict(schema='dvidia.drive-window-features/v1',status='observed',reason='valid_drive_window',
        observations=observations,start_s=packet['history_start_s'],end_s=packet['samples'][-1]['t_s'],
        release_corroborated=not reasons,reasons=sorted(reasons))


def fuse(primary, baseline, camera, actuator, *, contract):
    """Fuse already processed evidence; no physical-truth or case-label inputs."""
    reasons=[]
    if camera.get('status') != 'observed': reasons.append(camera.get('reason','camera_unavailable'))
    if actuator.get('status') != 'observed': reasons.append(actuator.get('reason','actuator_unavailable'))
    # Only validated primary complete packets have the required numeric fields.
    if baseline['verdict'] == 'complete':
        samples = primary['samples'][-100:]
        end = samples[-1]['t_s']; start = end-.200
        if camera.get('status') == 'observed':
            if (contract.frame_id != CAMERA.frame_id or abs(camera['start_s']-start)>TOL
                    or abs(camera['end_s']-end)>TOL):
                reasons.append('camera_primary_alignment_mismatch')
            else:
                # First RGB is the interval start. Subsequent frames align with primary samples9,19,...99.
                for o,p in zip(camera['observations'][1:],samples[9::10]):
                    radius=o['position_bound_m']+primary['uncertainty_bounds']['position_m']
                    if math.dist(o['position_xy_m'],p['position_m'][:2]) > radius:
                        reasons.append('camera_primary_position_contradiction')
                    delta=o['yaw_axis_rad']-p['yaw_rad']
                    axial=abs(math.atan2(math.sin(2*delta),math.cos(2*delta)))/2
                    if axial > o['yaw_bound_rad']+primary['uncertainty_bounds']['yaw_rad']:
                        reasons.append('camera_primary_axial_yaw_contradiction')
                if (camera['net_average_speed_interval_m_s'][0] > .01
                        or any(edge['average_speed_interval_m_s'][0] > .01 for edge in camera['edges'])):
                    reasons.append('camera_motion_contradiction')
        if actuator.get('status') == 'observed':
            if abs(actuator['start_s']-start)>TOL or abs(actuator['end_s']-end)>TOL:
                reasons.append('actuator_primary_alignment_mismatch')
            reasons.extend(actuator['reasons'])
    verdict = baseline['verdict']
    if verdict == 'complete' and reasons: verdict='unknown'
    return dict(schema='dvidia.sensor-goal-verdict/v1',verdict=verdict,
        primary_verdict=baseline['verdict'], reasons=sorted(set(reasons)),
        primary_reasons=baseline['reasons'],limitations=LIMITATIONS)


def verify_inputs(primary, camera_packet, actuator_packet, decoded_frames, *, contract:GoalContract, now_s):
    baseline=verify_goal(primary,contract=contract,now_s=now_s)
    camera=camera_window(camera_packet,decoded_frames,now_s=now_s)
    actuator=actuator_window(actuator_packet,now_s=now_s)
    return dict(baseline=baseline,camera=camera,actuator=actuator,
        fused=fuse(primary,baseline,camera,actuator,contract=contract))
