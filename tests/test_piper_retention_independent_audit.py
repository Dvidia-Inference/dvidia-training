"""Pure evidence audits; no simulation/runtime imports or native runs."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks' / 'piper-retention-v02'))
import independent_audit as audit


class TreatmentBoundaryTests(unittest.TestCase):
    def test_earliest_pre_admission_bound_refusal_is_cutoff_but_later_refusal_is_not(self):
        policy = {'events': [
            {'type': 'hold_invalidated', 'reason': 'tcp_position_error_exhausted', 't_seconds': 1.02},
            {'type': 'evidence_qualified', 'reason': 'sampled', 't_seconds': 2.},
            {'type': 'hold_invalidated', 'reason': 'tcp_orientation_error_exhausted', 't_seconds': 3.},
            {'type': 'hold_invalidated', 'reason': 'relative_slip', 't_seconds': .5},
        ]}
        self.assertEqual(audit.treatment_boundary(policy), {
            'time_s': 1.02, 'reason': 'pre_admission_bound_refusal',
            'detail': 'tcp_position_error_exhausted'})

    def test_only_declared_bound_reasons_and_admissions_create_cutoffs(self):
        self.assertIsNone(audit.treatment_boundary({'events': [
            {'type': 'hold_invalidated', 'reason': 'relative_slip', 't_seconds': 1.},
            {'type': 'abort', 'reason': 'pose_solver_refused', 't_seconds': 2.},
        ]}))
        self.assertEqual(audit.treatment_boundary({'events': [
            {'type': 'evidence_qualified', 'reason': 'sampled', 't_seconds': 2.},
            {'type': 'hold_invalidated', 'reason': 'joint_correction_exhausted', 't_seconds': 2.1},
        ]})['reason'], 'admission')

    def test_common_cutoff_combines_tied_reasons_deterministically(self):
        result = audit.common_cutoff({
            'measured_joint': {'time_s': 1., 'reason': 'admission', 'detail': 'evidence_qualified'},
            'issued_joint': {'time_s': 1., 'reason': 'pre_admission_bound_refusal', 'detail': 'pose_solver_refused'},
            'cartesian': None,
        })
        self.assertEqual(result['time_s'], 1.)
        self.assertEqual(result['reason'], 'admission+pre_admission_bound_refusal')
        self.assertEqual([row['variant'] for row in result['treatment_boundaries']],
                         ['issued_joint', 'measured_joint'])

    def test_invalid_triplet_remains_unknown_instead_of_asserting_three_way_match(self):
        attempts = {variant: {'valid': True} for variant in audit.VARIANTS}
        attempts['cartesian'] = {'valid': False, 'reason': 'invalid native sidecar',
                                 'native_rows_available': 100}
        result = audit.paired_prefix('scene-1', attempts)
        self.assertFalse(result['verified'])
        self.assertEqual(result['reason'], 'invalid_attempt_prevents_complete_triplet')
        self.assertEqual(result['variants'][2]['reason'], 'invalid native sidecar')


class PrefixTests(unittest.TestCase):
    def test_common_prefix_includes_cutoff_native_row_and_excludes_cutoff_command(self):
        times = np.arange(1, 20001, dtype=np.float64)*.001
        arrays = {'t_seconds': times, 'signal': np.arange(20000, dtype=np.float64)}
        actions = [{'t_seconds': i*.02, 'command': i} for i in range(1000)]
        cutoff = {'time_s': .04, 'reason': 'admission', 'treatment_boundaries': []}
        result = audit.prefix_identity(arrays, actions, cutoff)
        self.assertEqual(result['native_rows'], 40)
        self.assertEqual(result['commands'], 2)
        self.assertEqual(result['cutoff_time_s'], .04)
        self.assertEqual(result['cutoff_reason'], 'admission')


class StationarityTests(unittest.TestCase):
    def arrays(self):
        return {'tcp_position_m': np.zeros((4, 3), dtype=np.float64),
                'tcp_quaternion_wxyz': np.tile([1., 0., 0., 0.], (4, 1))}

    def test_stationary_window_and_quaternion_sign_invariance(self):
        arrays = self.arrays()
        arrays['tcp_quaternion_wxyz'][2:] *= -1.
        actual = audit.stationarity(arrays, 0, 3, True, True)
        self.assertTrue(actual['tcp_pose_within_bounds'])
        self.assertTrue(actual['grip_retention_and_stationarity'])
        report = {'complete_5s': True, **actual}
        audit.check_stationarity_reported(actual, report, True)

    def test_position_orientation_and_incomplete_windows_fail_stationarity(self):
        arrays = self.arrays(); arrays['tcp_position_m'][2, 0] = .004001
        actual = audit.stationarity(arrays, 0, 3, True, True)
        self.assertFalse(actual['tcp_pose_within_bounds'])
        self.assertFalse(actual['grip_retention_and_stationarity'])
        arrays = self.arrays(); theta = .050001
        arrays['tcp_quaternion_wxyz'][2:] = [np.cos(theta/2), 0., 0., np.sin(theta/2)]
        actual = audit.stationarity(arrays, 0, 3, True, True)
        self.assertFalse(actual['tcp_pose_within_bounds'])
        incomplete = audit.stationarity(self.arrays(), 0, 2, False, True)
        self.assertFalse(incomplete['tcp_pose_within_bounds'])
        self.assertFalse(incomplete['grip_retention_and_stationarity'])


def correction_fixture():
    issued = [.1]*6
    raw0 = [.1]*6
    jaw = .02
    records, actions = [], []
    for index, (time, delta, desired_delta, limited) in enumerate(
            ((1., 0., 0., []), (1.02, .001, .001, []), (1.04, .003, .01, [0]))):
        correction = [delta]+[0.]*5
        desired = [desired_delta]+[0.]*5
        raw = [raw0[0]+desired_delta]+raw0[1:]
        targets = [base+offset for base, offset in zip(issued, correction)]
        record = {'t_seconds': time, 'command_issued': True, 'fixed_bias_rad': [0.]*6,
                  'joint_targets_rad': targets, 'frozen_jaw_target_m': jaw,
                  'raw_pose_joint_targets_rad': raw, 'desired_correction_rad': desired,
                  'correction_rad': correction, 'correction_rate_limited_axes': limited,
                  'correction_rate_limited': bool(limited), 'refused_bounds': None}
        records.append(record)
        actions.append({'t_seconds': time, 'joint_targets_rad': targets,
                        'gripper_position_m': jaw})
    reference = {'t_seconds': 1., 'issued_joint_targets_rad': issued,
                 'measured_joint_position_rad': issued, 'raw_pose_joint_targets_rad': raw0,
                 'fixed_bias_rad': [0.]*6, 'frozen_jaw_target_m': jaw}
    policy = {'variant': 'cartesian', 'frozen_jaw_target_m': jaw,
              'events': [{'type': 'evidence_qualified', 't_seconds': 1., 'reason': 'sampled'}],
              'retention': {'hold_reference': reference, 'hold_records': records, 'input_refusals': []}}
    return policy, actions


class CorrectionAuditTests(unittest.TestCase):
    def test_cartesian_bias_first_command_correction_envelope_and_rate_are_checked(self):
        policy, actions = correction_fixture()
        result = audit.corrections_audit(policy, actions, audit.treatment_boundary(policy), 'cartesian')
        self.assertEqual(result['hold_record_count'], 3)
        self.assertEqual(result['rate_limited_axis_occurrences'], 1)
        self.assertEqual(result['maximum_abs_active_correction_rad'], .003)
        self.assertEqual(result['maximum_abs_desired_correction_rad'], .01)

    def test_bad_first_target_bias_or_rate_exceedance_is_rejected(self):
        policy, actions = correction_fixture()
        policy['retention']['hold_records'][0]['joint_targets_rad'][0] += .001
        actions[0]['joint_targets_rad'][0] += .001
        with self.assertRaises(AssertionError):
            audit.corrections_audit(policy, actions, audit.treatment_boundary(policy), 'cartesian')
        policy, actions = correction_fixture()
        policy['retention']['hold_reference']['fixed_bias_rad'][0] = .081
        with self.assertRaises(AssertionError):
            audit.corrections_audit(policy, actions, audit.treatment_boundary(policy), 'cartesian')
        policy, actions = correction_fixture()
        policy['retention']['hold_records'][2]['correction_rad'][0] = .004
        policy['retention']['hold_records'][2]['joint_targets_rad'][0] = .104
        actions[2]['joint_targets_rad'][0] = .104
        with self.assertRaises(AssertionError):
            audit.corrections_audit(policy, actions, audit.treatment_boundary(policy), 'cartesian')


if __name__ == '__main__':
    unittest.main()
