"""Matched destination scoring must not trust a controller's own success flag."""
from copy import deepcopy
import unittest

import numpy as np

from dvidia_training.video_goal_benchmark import make_protocol, score_reference, _fault_frames


def good_attempt(x=.34, y=.08):
    states = [dict(t_seconds=i*.02, object_position_m=[x, y, .08],
        table_object_contact_count=1, grasp_contact_counts={'left': 0, 'right': 0},
        gripper_joint_position_m=[.035, -.035], object_linear_velocity_m_s=[0., 0., 0.],
        object_angular_velocity_rad_s=[0., 0., 0.], tcp_position_m=[x, y, .22]) for i in range(30)]
    return dict(valid=True, success=True, trajectory=states,
        metrics=dict(lift_seen=True, carry_complete=True, retention_lost=False,
                     max_object_lift_m=.08, unexpected_contact_native_ticks=0))


class VideoGoalBenchmarkTests(unittest.TestCase):
    def test_wrong_destination_self_success_cannot_pass_common_reference(self):
        good = score_reference(good_attempt(), [.34, .08])
        wrong = score_reference(good_attempt(.40, -.08), [.34, .08])
        self.assertTrue(good['success'])
        self.assertTrue(wrong['controller_self_success'])
        self.assertFalse(wrong['success'])
        self.assertGreater(wrong['terminal_xy_error_m'], .1)

    def test_terminal_pose_without_retained_lift_release_or_dwell_fails(self):
        variants = []
        a = good_attempt(); a['metrics']['lift_seen'] = False; variants.append(a)
        a = good_attempt(); a['metrics']['retention_lost'] = True; variants.append(a)
        a = good_attempt(); a['trajectory'][-1]['grasp_contact_counts']['left'] = 1; variants.append(a)
        a = good_attempt(); a['trajectory'] = a['trajectory'][:3]; variants.append(a)
        a = good_attempt(); a['valid'] = False; variants.append(a)
        for a in variants:
            self.assertFalse(score_reference(a, [.34, .08])['success'])

    def test_future_correct_pose_cannot_preserve_dwell_after_terminal_motion(self):
        a = good_attempt()
        a['trajectory'][-1]['object_linear_velocity_m_s'] = [.2, 0., 0.]
        self.assertEqual(score_reference(a, [.34, .08])['terminal_dwell_s'], 0)

    def test_frozen_schedule_contains_all_paired_attempts_and_separated_destinations(self):
        p = make_protocol(seed=35123, count=4)
        self.assertEqual(p, make_protocol(seed=35123, count=4))
        self.assertEqual(p['planned_controller_attempts'], 20)
        self.assertEqual(p['planned_source_episodes'], 4)
        for case in p['cases']:
            other = next(c for c in p['cases'] if c['id'] == case['shuffled_source_id'])
            self.assertGreater(np.linalg.norm(np.array(case['task_goal_xy_m'])-other['task_goal_xy_m']), .05)

    def test_pixel_faults_leave_original_frames_intact(self):
        frames = [np.full((240, 320, 3), [30, 130, 60], dtype=np.uint8) for _ in range(12)]
        originals = deepcopy(frames)
        self.assertTrue(np.all(_fault_frames(frames, 'terminal_occlusion')[-1] == 20))
        self.assertTrue(np.all(_fault_frames(frames, 'extra_green_candidate')[0][25, 280] == [20, 180, 30]))
        for a, b in zip(frames, originals): np.testing.assert_array_equal(a, b)


if __name__ == '__main__': unittest.main()
