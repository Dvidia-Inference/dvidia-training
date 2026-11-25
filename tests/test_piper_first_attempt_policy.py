import unittest

import numpy as np

from dvidia_training.piper_first_attempt_policy import make_policy


class RecordTargets:
    def pose_action(self, joints, target, orientation, jaw):
        return tuple(target), tuple(orientation), jaw


def observation():
    return {
        't_seconds': 0., 'joint_position_rad': [0.] * 6,
        'tcp_position_world_m': [.3, -.03, .18],
        'tcp_quaternion_wxyz': [float(np.cos(5*np.pi/12)), 0., float(np.sin(5*np.pi/12)), 0.],
        'gripper_joint_position_m': [.035, -.035],
        'initial_object_position_world_m': [.3, -.03, .0805],
        'goal_position_world_m': [.35, .08, .08],
    }


class FirstAttemptPolicyTests(unittest.TestCase):
    def close(self, policy):
        obs = observation()
        policy.phase = 'descend'
        obs.update(t_seconds=.02, tcp_position_world_m=[.3, -.03, .082],
                   tcp_quaternion_wxyz=[0., float(np.sin(np.pi/8)), float(np.cos(np.pi/8)), 0.])
        policy(obs)
        self.assertEqual(policy.phase, 'close')
        return obs

    def test_signed_bilateral_closure_gate_and_timed_ablation(self):
        for fingers, expected in (([.028, -.028], 'probe'), ([.035, -.035], 'refused'),
                                  ([.040, 0.], 'refused'), ([.020, .020], 'refused')):
            for controller, final in (('corner-v8', expected), ('corner-v8-timed', 'probe')):
                with self.subTest(fingers=fingers, controller=controller):
                    policy = make_policy(controller, RecordTargets())
                    obs = self.close(policy)
                    for i in range(2, 90):
                        obs.update(t_seconds=i * .02, gripper_joint_position_m=fingers)
                        policy(obs)
                        if policy.phase != 'close':
                            break
                    self.assertEqual(policy.phase, final)

    def test_initial_acquisition_is_historical_and_inputs_are_detached(self):
        first = observation()
        latest = make_policy('corner-v8', RecordTargets())
        latest.phase = 'descend'
        latest(first)
        first['initial_object_position_world_m'][0] = 10.
        first['goal_position_world_m'][1] = -100.
        first.update(object_position_m=[2., 3., 4.], grasp_contact_counts={'left': 1, 'right': 1},
                     teacher_command={'joint_targets_rad': [9.] * 6})
        target, _, _ = latest(first)
        np.testing.assert_allclose(target, [.3, -.03, .082])
        np.testing.assert_allclose(latest.destination, [.35, .08, .08])

    def test_corner_test_lift_refuses_open_jaws_before_transport(self):
        policy = make_policy('corner-v8', RecordTargets())
        policy.phase = 'probe'
        obs = observation()
        obs.update(t_seconds=.64, tcp_position_world_m=[.3, -.03, .090],
                   gripper_joint_position_m=[.035, -.035])
        policy(obs)
        self.assertEqual(policy.phase, 'refused')
        self.assertEqual(policy.refusal_reason, 'aperture_lost_during_test_lift')

    def test_motion_requires_four_consecutive_bad_aperture_samples_to_stop(self):
        policy = make_policy('corner-v8', RecordTargets())
        policy.phase = 'lift'
        obs = observation()
        obs['gripper_joint_position_m'] = [.001, -.001]
        for tick in (1, 2):
            obs['t_seconds'] = .02*tick
            policy(obs)
        obs.update(t_seconds=.06, gripper_joint_position_m=[.028, -.028])
        policy(obs)
        obs['gripper_joint_position_m'] = [.001, -.001]
        for tick in (4, 5, 6, 7):
            obs['t_seconds'] = .02*tick
            action = policy(obs)
            self.assertEqual(policy.phase, 'lost_grip' if tick == 7 else 'lift')
        self.assertEqual(action, {'joint_targets_rad': [0.] * 6, 'gripper_position_m': 0.})

    def test_matched_ablation_can_run_without_a_finger_channel(self):
        policy = make_policy('corner-v8-timed', RecordTargets())
        policy.phase = 'close'
        obs = observation()
        del obs['gripper_joint_position_m']
        obs.update(t_seconds=.74, tcp_position_world_m=[.3, -.03, .082])
        policy(obs)
        self.assertEqual(policy.phase, 'probe')
        obs.update(t_seconds=1.4, tcp_position_world_m=[.3, -.03, .090])
        policy(obs)
        self.assertEqual(policy.phase, 'lift')

    def test_absent_or_non_finite_acquisition_is_refused(self):
        for input_value in (None, [0., 0.], [0., np.nan, .1]):
            obs = observation()
            obs['initial_object_position_world_m'] = input_value
            with self.assertRaisesRegex(ValueError, 'initial camera'):
                make_policy('corner-v8', RecordTargets())(obs)


if __name__ == '__main__':
    unittest.main()
