"""Fixed-reference scoring rejects incomplete traces and policy-side claims."""
import unittest

from dvidia_training.piper_imitation_score import score_episode


def valid_episode():
    timeout, dt = .60, .02
    start, goal = [.30, -.03, .08], [.34, .08, .08]
    trajectory, actions = [], []
    for index in range(31):
        t = index * dt
        lifted = 1 <= index <= 7
        position = ([start[0], start[1], .15] if lifted else
                    start if index < 15 else goal)
        contacts = {'left': 1, 'right': 1} if lifted else {'left': 0, 'right': 0}
        state = {
            't_seconds': t,
            'object_position_m': list(position),
            'table_object_contact_count': 0 if lifted else 1,
            'grasp_contact_counts': contacts,
            'gripper_joint_position_m': [.035, -.035],
            'object_linear_velocity_m_s': [0., 0., 0.],
            'object_angular_velocity_rad_s': [0., 0., 0.],
            'tcp_position_m': [goal[0], goal[1], .20],
            'issued_joint_targets_rad': None,
            'issued_gripper_position_m': None,
            'command': None,
        }
        if index < 30:
            action = {'t_seconds': t, 'joint_targets_rad': [0.] * 6,
                      'gripper_position_m': .035, 'phase': 'test-fixture'}
            state.update(issued_joint_targets_rad=action['joint_targets_rad'],
                         issued_gripper_position_m=action['gripper_position_m'], command=action)
            actions.append(action)
        trajectory.append(state)
    metrics = {'valid': True, 'success': True, 'lift_seen': True, 'grasp_seen': True,
               'max_object_lift_m': .08, 'carry_complete': True, 'retention_lost': False,
               'unexpected_contact_native_ticks': 0, 'dwell_seconds': .26,
               # This live streak counter resets after the block is placed.
               'lift_dwell_seconds': 0., 'warnings': []}
    return {'config': {'control_dt': dt, 'timeout_s': timeout,
                       'goal_position_tolerance_m': .025, 'goal_dwell_s': .25,
                       'min_lift_m': .06, 'min_lift_dwell_s': .10,
                       'retreat_clearance_m': .10},
            'initial_object_position_m': start, 'trajectory': trajectory,
            'actions': actions, 'metrics': metrics, 'valid': True, 'success': True}


class PiperImitationScoreTests(unittest.TestCase):
    def test_released_success_with_reset_live_lift_counter_passes_from_trace_history(self):
        result = score_episode(valid_episode(), [.34, .08])
        self.assertTrue(result['success'], result)
        self.assertEqual(result['reason'], 'success')
        self.assertTrue(result['terminal_reference_success'])
        self.assertTrue(result['native_outcome_success'])
        self.assertGreaterEqual(result['terminal_dwell_s'], .25)
        self.assertGreaterEqual(result['trace_lift_dwell_s'], .10)

    def test_native_lift_latch_without_recorded_lift_streak_does_not_pass(self):
        episode = valid_episode()
        for state in episode['trajectory']:
            state['object_position_m'] = list(episode['initial_object_position_m'])
            state['grasp_contact_counts'] = {'left': 0, 'right': 0}
            state['table_object_contact_count'] = 1
        result = score_episode(episode, [.34, .08])
        self.assertFalse(result['success'])
        self.assertFalse(result['native_outcome_success'])
        self.assertEqual(result['trace_lift_dwell_s'], 0.)

    def test_gap_and_reordered_trace_cannot_create_terminal_dwell(self):
        for mutate in (
            lambda episode: episode['trajectory'][8].update(t_seconds=.17),
            lambda episode: episode['trajectory'][8].update(t_seconds=.30),
        ):
            episode = valid_episode()
            mutate(episode)
            result = score_episode(episode, [.34, .08])
            self.assertFalse(result['success'])
            self.assertEqual(result['reason'], 'invalid_timestamp_cadence')

    def test_wrong_reference_open_jaw_and_noop_fail_independent_predicate(self):
        wrong_goal = score_episode(valid_episode(), [.40, -.08])
        self.assertFalse(wrong_goal['success'])
        self.assertFalse(wrong_goal['terminal_reference_success'])

        open_jaw = valid_episode()
        for state in open_jaw['trajectory']:
            state['gripper_joint_position_m'] = [.01, -.01]
        result = score_episode(open_jaw, [.34, .08])
        self.assertFalse(result['success'])
        self.assertFalse(result['terminal_reference_success'])

        no_op = valid_episode()
        for state in no_op['trajectory']:
            state['object_position_m'] = list(no_op['initial_object_position_m'])
        no_op['metrics'].update(success=False, lift_seen=False, grasp_seen=False,
                                max_object_lift_m=0., carry_complete=False)
        result = score_episode(no_op, [.34, .08])
        self.assertFalse(result['success'])
        self.assertFalse(result['native_outcome_success'])

    def test_native_metrics_and_complete_record_are_required(self):
        episode = valid_episode()
        episode['metrics']['retention_lost'] = True
        self.assertEqual(score_episode(episode, [.34, .08])['reason'], 'native_outcome_failed')

        incomplete = valid_episode()
        incomplete['trajectory'].pop()
        result = score_episode(incomplete, [.34, .08])
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'incomplete_trajectory')

    def test_policy_success_flag_without_native_metrics_does_not_pass(self):
        episode = valid_episode()
        episode['metrics'].pop('success')
        result = score_episode(episode, [.34, .08])
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'incomplete_native_metrics')


if __name__ == '__main__':
    unittest.main()
