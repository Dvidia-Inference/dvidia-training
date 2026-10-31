"""Native-shaped falsification fixtures; no native simulator or hardware runs."""
from copy import deepcopy
import unittest

import numpy as np

from dvidia_training import probe_evidence as proof
from dvidia_training import task_feasibility


def fixture(rows=300):
    arrays = {name: np.zeros((rows,) if width == 1 else (rows, width), dtype=np.float64)
              for name, width in proof.FIELDS}
    arrays['native_tick'][:] = np.arange(1, rows+1)
    arrays['t_seconds'][:] = arrays['native_tick']*.001
    arrays['object_position_m'][:] = [.3, -.03, .08]
    arrays['tcp_position_m'][:] = [.3, -.03, .18]
    lifted = arrays['t_seconds'] >= .04
    arrays['object_position_m'][lifted, 2] += .01
    arrays['tcp_position_m'][lifted, 2] += .01
    arrays['object_minus_tcp_local_m'][:] = arrays['object_position_m']-arrays['tcp_position_m']
    arrays['object_quaternion_wxyz'][:, 0] = 1.
    arrays['tcp_quaternion_wxyz'][:, 0] = 1.
    arrays['grasp_contact_counts'][:] = [1., 1.]
    arrays['pad_normal_forces_n'][:] = [.2, .3]
    arrays['table_object_contact_count'][:] = ~lifted
    metadata = {'schema': 'dvidia.piper-feedback-native/v1', 'native_timestep_s': .001,
                'rows': rows, 'fields': [{'name': name, 'width': width} for name, width in proof.FIELDS],
                'side_order': ['left', 'right'], 'physical_robot_ready': False}
    document = task_feasibility.template()
    document['calibration_id'] = 'synthetic_probe_v01'
    context = {'schema': proof.CONTEXT_SCHEMA, 'run_id': 'run-00', 'object_id': 'task_block',
               'grasp_id': 'run-00-probe-0', 'probe_index': 0, 'robot_profile': 'piper',
               **{name: 'a'*64 for name in proof.DIGEST_FIELDS},
               'calibration_id': document['calibration_id'],
               'calibration_scope': 'simulation_assumptions_not_hardware',
               'initial_object_position_m': [.3, -.03, .08], 'probe_start_s': .02,
               'probe_origin_object_position_m': arrays['object_position_m'][19].tolist(),
               'probe_origin_tcp_local_offset_m': arrays['object_minus_tcp_local_m'][19].tolist(),
               'per_finger_force_ceiling_n': document['per_finger_normal_force_budget_n'],
               'native_status': {'valid': True, 'warnings': [], 'unexpected_contact_native_ticks': 0,
                                 'simulation_time_s': float(arrays['t_seconds'][-1])}}
    context['feasibility_sha256'] = proof.feasibility_binding(document)
    claims = [{'t_seconds': .14, 'qualified': True, 'probe_index': 0}] if rows >= 140 else []
    if rows > 140 and rows % 20 == 0:
        claims.append({'t_seconds': rows*.001, 'qualified': True, 'probe_index': 0})
    return arrays, metadata, context, document, claims


def derive(arrays, metadata, context, claims):
    return proof.derive_probe_evidence(arrays, metadata, context, controller_qualifications=claims)


class ProbeEvidenceTests(unittest.TestCase):
    def test_fresh_bound_native_proof_replaces_caller_lift_flag_without_mutation(self):
        arrays, metadata, context, document, claims = fixture()
        originals = deepcopy((arrays, metadata, context, document, claims))
        receipt = derive(arrays, metadata, context, claims)
        self.assertTrue(receipt['history']['ever_qualified'])
        self.assertTrue(receipt['latest']['qualified'])
        result = proof.assess_with_probe(document, receipt, arrays, metadata, context, now_s=.3)
        self.assertEqual(result['status'], 'ready_for_simulated_transport')
        self.assertTrue(result['probe']['admissible'])
        self.assertFalse(result['physical_robot_ready'])
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], originals[0][name])
        self.assertEqual((metadata, context, document, claims), originals[1:])

    def test_exact_100_native_tick_window_not_99_ticks(self):
        for rows, qualified in ((138, False), (139, True)):
            arrays, metadata, context, _, claims = fixture(rows)
            receipt = derive(arrays, metadata, context, claims)
            self.assertEqual(receipt['latest']['qualified'], qualified)
        self.assertEqual(receipt['latest']['continuous_qualifying_s'], .1)

    def test_single_hidden_force_gap_invalidates_20ms_controller_claim(self):
        arrays, metadata, context, _, claims = fixture()
        arrays['pad_normal_forces_n'][79, 0] = .01  # Exactly threshold is insufficient.
        receipt = derive(arrays, metadata, context, claims)
        self.assertFalse(receipt['audit']['valid'])
        self.assertEqual(receipt['audit']['controller_claim_mismatches'][0]['t_seconds'], .14)
        self.assertTrue(receipt['latest']['qualified'])  # Later recovery cannot fix an earlier false claim.
        verdict = proof.verify_probe_evidence(receipt, arrays, metadata, context, now_s=.3)
        self.assertFalse(verdict['admissible'])

    def test_prior_qualification_does_not_substitute_for_latest_live_window(self):
        for field in ('pad_normal_forces_n', 'grasp_contact_counts', 'table_object_contact_count'):
            arrays, metadata, context, _, claims = fixture()
            if field == 'table_object_contact_count':
                arrays[field][249] = 1.
            else:
                arrays[field][249, 0] = 0.
            receipt = derive(arrays, metadata, context, claims)
            verdict = proof.verify_probe_evidence(receipt, arrays, metadata, context, now_s=.3)
            self.assertTrue(verdict['historical_qualified'])
            self.assertFalse(verdict['latest_qualified'])
            self.assertFalse(verdict['admissible'])

    def test_terminal_controller_invalidation_cannot_be_resurrected_by_native_recovery(self):
        arrays, metadata, context, document, claims = fixture()
        claims[-1]['qualified'] = False
        receipt = derive(arrays, metadata, context, claims)
        self.assertTrue(receipt['history']['ever_qualified'])
        self.assertTrue(receipt['latest']['qualified'])
        self.assertTrue(receipt['audit']['valid'])  # Abstaining is not a false positive.
        self.assertEqual(receipt['audit']['controller_false_qualification_count'], 0)
        self.assertEqual(receipt['audit']['controller_conservative_abstention_count'], 1)
        self.assertEqual(receipt['audit']['controller_claim_mismatches'][0]['classification'], 'conservative_abstention')
        result = proof.assess_with_probe(document, receipt, arrays, metadata, context, now_s=.3)
        self.assertFalse(result['probe']['admissible'])
        self.assertIn('latest_controller_not_qualified', result['probe']['reasons'])
        self.assertEqual(result['status'], 'test_lift_required')

    def test_latest_controller_claim_must_match_latest_native_sample(self):
        arrays, metadata, context, _, claims = fixture()
        claims.pop()
        receipt = derive(arrays, metadata, context, claims)
        verdict = proof.verify_probe_evidence(receipt, arrays, metadata, context, now_s=.3)
        self.assertTrue(verdict['historical_qualified'])
        self.assertTrue(verdict['latest_qualified'])
        self.assertFalse(verdict['admissible'])
        self.assertIn('latest_controller_claim_missing_or_stale', verdict['reasons'])

    def test_world_pose_redundancy_and_tcp_local_drift(self):
        arrays, metadata, context, _, claims = fixture()
        arrays['object_minus_tcp_local_m'][249, 0] = .004
        with self.assertRaisesRegex(proof.ProbeEvidenceError, 'world pose'):
            derive(arrays, metadata, context, claims)
        arrays['tcp_position_m'][249, 0] -= .004
        receipt = derive(arrays, metadata, context, claims)
        self.assertFalse(receipt['latest']['qualified'])
        self.assertTrue(receipt['history']['ever_qualified'])

    def test_less_than_six_mm_rise_fails_without_inventing_success(self):
        arrays, metadata, context, _, _ = fixture()
        arrays['object_position_m'][39:, 2] = .0859
        arrays['tcp_position_m'][39:, 2] = .1859
        arrays['object_minus_tcp_local_m'][:] = arrays['object_position_m']-arrays['tcp_position_m']
        claims = [{'t_seconds': .14, 'qualified': False, 'probe_index': 0}]
        receipt = derive(arrays, metadata, context, claims)
        self.assertFalse(receipt['history']['ever_qualified'])
        self.assertFalse(proof.verify_probe_evidence(receipt, arrays, metadata, context, now_s=.3)['admissible'])

    def test_stale_and_future_evidence_fail_even_after_historical_success(self):
        arrays, metadata, context, _, claims = fixture()
        receipt = derive(arrays, metadata, context, claims)
        for now, reason in ((.32, 'stale_evidence'), (.28, 'future_evidence')):
            verdict = proof.verify_probe_evidence(receipt, arrays, metadata, context, now_s=now)
            self.assertFalse(verdict['admissible'])
            self.assertIn(reason, verdict['reasons'])

    def test_changed_run_grasp_object_pose_command_calibration_or_model_invalidates(self):
        arrays, metadata, context, _, claims = fixture()
        receipt = derive(arrays, metadata, context, claims)
        for field, value in (('run_id', 'run-other'), ('object_id', 'other'), ('grasp_id', 'another-grasp'),
                             ('pose_binding_sha256', 'b'*64), ('command_sha256', 'b'*64),
                             ('calibration_id', 'synthetic_other'), ('model_sha256', 'b'*64)):
            current = deepcopy(context)
            current[field] = value
            verdict = proof.verify_probe_evidence(receipt, arrays, metadata, current, now_s=.3)
            self.assertFalse(verdict['admissible'], field)
            self.assertIn('probe_context_changed', verdict['reasons'])

    def test_receipt_metadata_or_array_mutations_fail_recomputation(self):
        arrays, metadata, context, _, claims = fixture()
        receipt = derive(arrays, metadata, context, claims)
        changed = deepcopy(receipt)
        changed['latest']['qualified'] = False
        self.assertFalse(proof.verify_probe_evidence(changed, arrays, metadata, context, now_s=.3)['admissible'])
        changed_meta = deepcopy(metadata)
        changed_meta['new_note'] = 'changed'
        self.assertFalse(proof.verify_probe_evidence(receipt, arrays, changed_meta, context, now_s=.3)['admissible'])
        arrays['joint_position_rad'][149, 0] += .001
        self.assertFalse(proof.verify_probe_evidence(receipt, arrays, metadata, context, now_s=.3)['admissible'])

    def test_invalid_dynamics_warning_unexpected_contact_and_force_ceiling_refuse(self):
        for mutation in ('invalid', 'warning', 'contact', 'ceiling'):
            arrays, metadata, context, _, claims = fixture()
            if mutation == 'invalid':
                context['native_status']['valid'] = False
            elif mutation == 'warning':
                context['native_status']['warnings'] = ['solver warning']
            elif mutation == 'contact':
                arrays['unexpected_contact_count'][249] = 1.
                context['native_status']['unexpected_contact_native_ticks'] = 1
            else:
                context['per_finger_force_ceiling_n'] = .25
            receipt = derive(arrays, metadata, context, claims)
            self.assertFalse(receipt['audit']['valid'], mutation)
            self.assertFalse(proof.verify_probe_evidence(receipt, arrays, metadata, context, now_s=.3)['admissible'])

    def test_unknown_mass_friction_budget_payload_and_calibration_remain_unknown(self):
        for field in ('mass_interval_kg', 'friction_interval', 'per_finger_normal_force_budget_n',
                      'pose_specific_payload_limit_kg', 'calibration_id'):
            arrays, metadata, context, document, claims = fixture()
            document[field] = None
            if field == 'calibration_id':
                context['calibration_id'] = None
            if field == 'per_finger_normal_force_budget_n':
                context['per_finger_force_ceiling_n'] = None
            context['feasibility_sha256'] = proof.feasibility_binding(document)
            receipt = derive(arrays, metadata, context, claims)
            result = proof.assess_with_probe(document, receipt, arrays, metadata, context, now_s=.3)
            self.assertTrue(result['probe']['admissible'])
            self.assertEqual(result['status'], 'probe_required', field)
            self.assertIsNone(document[field])
        arrays, metadata, context, document, claims = fixture()
        context['per_finger_force_ceiling_n'] = None
        receipt = derive(arrays, metadata, context, claims)
        self.assertFalse(proof.assess_with_probe(document, receipt, arrays, metadata, context, now_s=.3)['probe']['admissible'])

    def test_native_critical_total_counts_ticks_not_sum_of_contact_counts(self):
        arrays, metadata, context, _, claims = fixture()
        arrays['unexpected_contact_count'][249] = 5.
        context['native_status']['unexpected_contact_native_ticks'] = 1
        receipt = derive(arrays, metadata, context, claims)
        self.assertEqual(receipt['audit']['unexpected_contact_native_ticks'], 1)
        self.assertIn('unexpected_native_contact', receipt['audit']['reasons'])
        self.assertNotIn('native_contact_total_disagrees', receipt['audit']['reasons'])
        context['native_status']['unexpected_contact_native_ticks'] = 5
        receipt = derive(arrays, metadata, context, claims)
        self.assertIn('native_contact_total_disagrees', receipt['audit']['reasons'])

    def test_planning_change_and_caller_stale_slip_vetoes_preserved(self):
        arrays, metadata, context, document, claims = fixture()
        receipt = derive(arrays, metadata, context, claims)
        document['planned_upward_accel_m_s2'] = .6
        result = proof.assess_with_probe(document, receipt, arrays, metadata, context, now_s=.3)
        self.assertFalse(result['probe']['admissible'])
        self.assertIn('planning_assumptions_changed', result['probe']['reasons'])
        self.assertEqual(result['status'], 'test_lift_required')
        document['planned_upward_accel_m_s2'] = .5
        for field, value in (('observations_fresh', False), ('slip_detected', True)):
            changed = deepcopy(document)
            changed[field] = value
            self.assertEqual(proof.assess_with_probe(changed, receipt, arrays, metadata, context, now_s=.3)['status'], 'blocked')

    def test_caller_test_lift_boolean_cannot_bypass_missing_receipt_or_claim(self):
        arrays, metadata, context, document, _ = fixture()
        document['test_lift_passed'] = True
        self.assertEqual(proof.assess_with_probe(document, None, arrays, metadata, context, now_s=.3)['status'], 'test_lift_required')
        receipt = derive(arrays, metadata, context, [])
        self.assertTrue(receipt['latest']['qualified'])
        self.assertFalse(proof.verify_probe_evidence(receipt, arrays, metadata, context, now_s=.3)['admissible'])
        self.assertEqual(proof.assess_with_probe(document, receipt, arrays, metadata, None, now_s=.3)['status'], 'test_lift_required')

    def test_inclusive_rise_and_drift_boundaries_and_old_probe_claim_refusal(self):
        arrays, metadata, context, _, claims = fixture()
        lifted = arrays['t_seconds'] >= .04
        arrays['object_position_m'][lifted, 2] = .086
        arrays['tcp_position_m'][lifted, 2] = .186
        arrays['tcp_position_m'][lifted, 0] -= .003
        arrays['object_minus_tcp_local_m'][:] = arrays['object_position_m']-arrays['tcp_position_m']
        receipt = derive(arrays, metadata, context, claims)
        self.assertTrue(proof.verify_probe_evidence(receipt, arrays, metadata, context, now_s=.3)['admissible'])
        new_context = deepcopy(context)
        new_context['probe_index'] = 1
        new_context['grasp_id'] = 'run-00-probe-1'
        self.assertFalse(proof.verify_probe_evidence(receipt, arrays, metadata, new_context, now_s=.3)['admissible'])

    def test_malformed_cadence_shape_nan_origin_and_controller_binding_reject(self):
        for mutation in ('tick', 'nan', 'dtype', 'origin', 'future_claim', 'grasp_claim'):
            arrays, metadata, context, _, claims = fixture()
            if mutation == 'tick':
                arrays['native_tick'][49] += 1
            elif mutation == 'nan':
                arrays['qpos' if 'qpos' in arrays else 'joint_position_rad'][49, 0] = np.nan
            elif mutation == 'dtype':
                arrays['joint_position_rad'] = arrays['joint_position_rad'].astype(np.float32)
            elif mutation == 'origin':
                context['probe_origin_object_position_m'][0] += .001
            elif mutation == 'future_claim':
                claims[0]['t_seconds'] = .32
            else:
                claims[0]['probe_index'] = 1
            with self.assertRaises(proof.ProbeEvidenceError, msg=mutation):
                derive(arrays, metadata, context, claims)


if __name__ == '__main__':
    unittest.main()
