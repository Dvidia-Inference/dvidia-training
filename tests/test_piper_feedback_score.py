"""Independent evidence rejects gaps/forgeries and separates contact recovery.

These numeric fixtures exercise scoring predicates, not native physics. Real
environment evidence is separately bound to simulator/runtime/asset receipts.
"""
from copy import deepcopy
import unittest

import numpy as np

from dvidia_training import piper_feedback_score as score


GOAL = [.34, .08]


def fixture(*, loss=None):
    """A supported close, native lift/carry, supported release and final dwell."""
    count = 800
    times = np.arange(1, count+1)*.001
    arrays = {name: np.zeros((count,) if width == 1 else (count, width)) for name, width in score.FIELDS}
    arrays['t_seconds'] = times
    arrays['native_tick'] = np.arange(1, count+1, dtype=float)
    arrays['object_position_m'][:] = [.30, -.03, .08]
    arrays['object_position_m'][199:399, 2] = .15
    arrays['object_position_m'][399:] = [*GOAL, .08]
    arrays['object_quaternion_wxyz'][:, 0] = 1.
    arrays['tcp_quaternion_wxyz'][:, 0] = 1.
    arrays['tcp_position_m'][:] = [*GOAL, .25]
    arrays['object_minus_tcp_local_m'][:] = arrays['object_position_m']-arrays['tcp_position_m']
    arrays['grasp_contact_counts'][19:480] = [1., 1.]
    arrays['pad_normal_forces_n'][19:480] = [.2, .3]
    arrays['table_object_contact_count'][:] = 1.
    arrays['table_object_contact_count'][199:399] = 0.
    arrays['gripper_joint_position_m'][:] = [.035, -.035]
    arrays['gripper_joint_position_m'][19:480] = [.01, -.01]
    for name in ('requested_jaw_target_m', 'normal_rate_limited_jaw_target_m', 'applied_jaw_target_m'):
        arrays[name][:] = .035
    if loss is not None:
        first, last = loss
        arrays['grasp_contact_counts'][first-1:last] = 0.
        arrays['pad_normal_forces_n'][first-1:last] = 0.
    config = dict(timestep=.001, control_dt=.02, timeout_s=.8, goal_position_tolerance_m=.025,
                  goal_dwell_s=.25, min_lift_m=.06, min_lift_dwell_s=.1,
                  retreat_clearance_m=.1, target_position_m=[*GOAL, .08])
    metrics = dict(valid=True, success=True, lift_seen=True, grasp_seen=True,
                   max_object_lift_m=.07, max_pad_normal_force_n=.3, carry_complete=True,
                   retention_lost=False, unexpected_contact_native_ticks=0,
                   dwell_seconds=.320, lift_dwell_seconds=0., max_carry_contact_loss_seconds=0.,
                   carry_contact_loss_events=0, first_success_time_seconds=.730, warnings=[])
    episode = dict(config=config, initial_object_position_m=[.30, -.03, .08],
                   trajectory=[], actions=[], metrics=metrics, valid=True, success=True)
    names = ('object_position_m', 'object_quaternion_wxyz', 'object_linear_velocity_m_s',
             'object_angular_velocity_rad_s', 'tcp_position_m', 'tcp_quaternion_wxyz',
             'gripper_joint_position_m', 'joint_position_rad')
    for control in range(41):
        index = control*20-1
        state = dict(t_seconds=control*.02)
        for name in names:
            state[name] = (arrays[name][index] if control else arrays[name][0]).tolist()
        if not control:
            state['gripper_joint_position_m'] = [.035, -.035]
        for name in ('grasp_contact_counts', 'pad_normal_forces_n'):
            values = arrays[name][index] if control else [0., 0.]
            state[name] = {side: int(value) if name == 'grasp_contact_counts' else float(value)
                           for side, value in zip(('left', 'right'), values)}
        for name in ('table_object_contact_count', 'unexpected_contact_count'):
            state[name] = int(arrays[name][index]) if control else int(name == 'table_object_contact_count')
        state.update(issued_joint_targets_rad=None, issued_gripper_position_m=None, command=None)
        if control < 40:
            action = dict(t_seconds=control*.02, joint_targets_rad=[0.]*6, gripper_position_m=.035)
            episode['actions'].append(action)
            state.update(issued_joint_targets_rad=[0.]*6, issued_gripper_position_m=.035, command=action)
        episode['trajectory'].append(state)
    metadata = dict(schema=score.NATIVE_SCHEMA, native_timestep_s=.001, rows=count,
                    fields=[dict(name=name, width=width) for name, width in score.FIELDS],
                    side_order=['left', 'right'])
    return episode, arrays, metadata


def evaluate(parts):
    return score.score_feedback_episode(*parts, GOAL)


class FeedbackScoreTests(unittest.TestCase):
    def test_native_reconstruction_and_old_endpoint_agree_without_final_release_alarm(self):
        result = evaluate(fixture())
        self.assertTrue(result['valid'], result)
        self.assertTrue(result['success'], result)
        self.assertTrue(result['strict_v01']['success'])
        self.assertEqual(result['native_metrics_mismatches'], [])
        self.assertTrue(result['native_recomputed']['carry_complete'])
        self.assertAlmostEqual(result['native_recomputed']['dwell_seconds'], .320)
        self.assertAlmostEqual(result['native_recomputed']['first_success_time_seconds'], .730)
        self.assertEqual(result['grip_events'], [])
        self.assertEqual(result['false_alarms']['status'], 'not_applicable')

    def test_supported_interruption_has_native_loss_onset_and_100ms_restoration(self):
        result = evaluate(fixture(loss=(80, 120)))
        self.assertTrue(result['success'], result)
        event, = result['grip_events']
        self.assertEqual(event['kind'], 'supported_grip_interruption')
        self.assertAlmostEqual(event['loss_onset_s'], .080)
        self.assertAlmostEqual(event['loss_qualified_s'], .099)
        self.assertAlmostEqual(event['restoration_onset_s'], .121)
        self.assertAlmostEqual(event['restoration_qualified_s'], .220)
        self.assertAlmostEqual(event['onset_to_qualified_restoration_s'], .140)
        self.assertEqual(result['grip_summary']['restored_events'], 1)

    def test_initial_acquisition_and_ten_ms_contact_flicker_are_not_sustained_losses(self):
        # The whole first 19 ms has no contact; it is not loss of an acquired grip.
        self.assertEqual(evaluate(fixture())['grip_events'], [])
        self.assertEqual(evaluate(fixture(loss=(80, 89)))['grip_events'], [])

    def test_short_restoration_flicker_does_not_finish_recovery(self):
        parts = fixture(loss=(80, 190))
        arrays = parts[1]
        # 50 ms of bilateral contact inside a longer interruption, then loss.
        arrays['grasp_contact_counts'][100:150] = [1., 1.]
        arrays['pad_normal_forces_n'][100:150] = [.2, .3]
        # Synchronize control-boundary evidence, while retaining fixed native gates.
        for k, state in enumerate(parts[0]['trajectory'][1:], 1):
            for name in ('grasp_contact_counts', 'pad_normal_forces_n'):
                state[name] = dict(zip(('left', 'right'), arrays[name][k*20-1].tolist()))
                if name == 'grasp_contact_counts':
                    state[name] = {key: int(value) for key, value in state[name].items()}
        result = evaluate(parts)
        self.assertTrue(result['valid'], result)
        event, = result['grip_events']
        self.assertAlmostEqual(event['restoration_onset_s'], .191)
        self.assertAlmostEqual(event['restoration_qualified_s'], .290)

    def test_native_retention_history_cannot_be_erased_by_restoration_or_terminal_pose(self):
        parts = fixture(loss=(330, 380))
        # Already completed 100 ms lift before loss; 51 ms contact loss latches
        # retention failure, even though contact returns and placement is near.
        metrics = parts[0]['metrics']
        metrics.update(success=False, retention_lost=True, dwell_seconds=0.,
                       max_carry_contact_loss_seconds=.051, carry_contact_loss_events=1,
                       first_success_time_seconds=None)
        parts[0]['success'] = False
        result = evaluate(parts)
        self.assertTrue(result['valid'], result)
        self.assertFalse(result['success'])
        self.assertTrue(result['terminal_only']['success'])
        self.assertTrue(result['native_recomputed']['retention_lost'])
        event, = result['grip_events']
        self.assertEqual(event['kind'], 'airborne_contact_loss')
        # Destination arrived before another 100 ms could qualify restoration.
        self.assertFalse(event['restored'])
        self.assertIn('censored_reason', event)

    def test_forged_native_success_latch_disagrees_with_numeric_evidence(self):
        parts = fixture(loss=(330, 380))
        result = evaluate(parts)
        self.assertFalse(result['valid'])
        self.assertFalse(result['success'])
        self.assertIn('retention_lost', result['native_metrics_mismatches'])
        self.assertIn('success', result['native_metrics_mismatches'])

    def test_missing_nonfinite_wrong_shape_and_boolean_arrays_are_invalid_not_dropped(self):
        mutations = [lambda arrays: arrays.pop('pad_normal_forces_n'),
                     lambda arrays: arrays['t_seconds'].__setitem__(10, float('nan')),
                     lambda arrays: arrays.update(tcp_position_m=np.zeros((800, 2))),
                     lambda arrays: arrays.update(pulse_active=np.zeros(800, dtype=bool))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                parts = fixture()
                mutate(parts[1])
                result = evaluate(parts)
                self.assertFalse(result['valid'])
                self.assertFalse(result['success'])
                self.assertIn('strict_v01', result)

    def test_native_gap_reordered_tick_and_truncated_sidecar_fail(self):
        for change in ('gap', 'tick', 'truncated'):
            with self.subTest(change=change):
                parts = fixture()
                if change == 'gap':
                    parts[1]['t_seconds'][50] += .001
                elif change == 'tick':
                    parts[1]['native_tick'][50] -= 1
                else:
                    parts[2]['rows'] -= 1
                result = evaluate(parts)
                self.assertFalse(result['valid'])
                self.assertFalse(result['success'])

    def test_changed_relative_pose_or_control_state_does_not_authenticate_itself(self):
        parts = fixture()
        parts[1]['object_minus_tcp_local_m'][30, 0] += .001
        self.assertEqual(evaluate(parts)['reason'], 'native_relative_pose_mismatch')
        parts = fixture()
        parts[0]['trajectory'][2]['tcp_position_m'][0] += .001
        self.assertEqual(evaluate(parts)['reason'], 'native_control_state_mismatch:tcp_position_m')

    def test_premature_open_and_supported_touchdown_are_separate_from_terminal_success(self):
        parts = fixture()
        # A native touch down away from goal after qualified lift. The trace
        # remains shape-consistent but the recorded outcome flags are untrusted.
        arrays = parts[1]
        arrays['table_object_contact_count'][350:370] = 1.
        arrays['gripper_joint_position_m'][350:370] = [.035, -.035]
        for k, state in enumerate(parts[0]['trajectory'][1:], 1):
            state['table_object_contact_count'] = int(arrays['table_object_contact_count'][k*20-1])
            state['gripper_joint_position_m'] = arrays['gripper_joint_position_m'][k*20-1].tolist()
        result = evaluate(parts)
        diagnostics = result['native_diagnostics']
        self.assertAlmostEqual(diagnostics['premature_open_jaw']['total_s'], .020)
        self.assertAlmostEqual(diagnostics['support_touchdown_before_destination']['total_s'], .020)
        self.assertEqual(diagnostics['support_touchdown_before_destination']['count'], 1)


if __name__ == '__main__':
    unittest.main()
