"""Fixed-protocol and native-shaped falsification tests; no physics runs."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_retention_benchmark as study
from dvidia_training import probe_evidence as proof


def fixture(admission_s=2.):
    count = 20000
    arrays = {name: np.zeros((count,) if width == 1 else (count, width), dtype=np.float64)
              for name, width in proof.FIELDS}
    arrays['native_tick'][:] = np.arange(1, count+1)
    arrays['t_seconds'][:] = arrays['native_tick']*.001
    arrays['object_position_m'][:] = [.303, -.026, .082]
    arrays['tcp_position_m'][:] = [.303, -.026, .182]
    lifted = arrays['t_seconds'] >= 1.1
    arrays['object_position_m'][lifted, 2] += .01
    arrays['tcp_position_m'][lifted, 2] += .01
    arrays['object_minus_tcp_local_m'][:] = arrays['object_position_m']-arrays['tcp_position_m']
    arrays['object_quaternion_wxyz'][:, 0] = 1.
    arrays['tcp_quaternion_wxyz'][:, 0] = 1.
    arrays['grasp_contact_counts'][:] = [1., 1.]
    arrays['pad_normal_forces_n'][:] = [.2, .3]
    arrays['table_object_contact_count'][:] = ~lifted
    metadata = {'schema': 'dvidia.piper-feedback-native/v1', 'native_timestep_s': .001,
                'rows': count, 'fields': [{'name': name, 'width': width} for name, width in proof.FIELDS],
                'side_order': ['left', 'right'], 'physical_robot_ready': False}
    case = study.case_definitions()[0]
    probe = {'probe_index': 0, 'start_time_s': 1.,
             'object_origin_m': arrays['object_position_m'][999].tolist(),
             'tcp_origin_m': arrays['tcp_position_m'][999].tolist(),
             'tcp_quaternion_wxyz': [1., 0., 0., 0.],
             'local_offset_origin_m': arrays['object_minus_tcp_local_m'][999].tolist(),
             'end_time_s': 20., 'disposition': 'admitted', 'reason': 'qualified'}
    checks = [{'t_seconds': t, 'qualified': True, 'probe_index': 0}
              for t in sorted({admission_s, min(20., admission_s+5.), 20.})]
    result = {'id': case['id']+'-measured_joint', 'scene_id': case['id'], 'seed': case['seed'],
              'variant': 'measured_joint', 'config': case['config'], 'valid': True, 'error': None,
              'initial_object_position_m': arrays['object_position_m'][0].tolist(),
              'compiled_model_sha256': 'a'*64,
              'actions': [{} for _ in range(1000)], 'trajectory': [{} for _ in range(1001)],
              'metrics': {'valid': True, 'warnings': [], 'unexpected_contact_native_ticks': 0,
                          'simulation_time': 20.},
              'policy': {'probes': [probe], 'admitted': True,
                         'events': [{'type': 'evidence_qualified', 't_seconds': admission_s, 'probe_index': 0}],
                         'qualification_checks': checks, 'requested_controls': [],
                         'retention': {'hold_records': [], 'input_refusals': []}}}
    protocol = {'force_ceiling_n': 10., 'source_sha256': {}, 'asset_lock_sha256': 'a'*64}
    return result, arrays, metadata, protocol


def endpoint(result, arrays, metadata, protocol):
    # The bound receipt has its own integration test below. Endpoint mutations
    # stay isolated from that existing module's terminal qualification semantics.
    with patch.object(study.prior, '_evaluate', return_value={'verification': {'admissible': False}}):
        return study.evaluate(result, arrays, metadata, protocol)


def controls_fixture():
    result, arrays, _, _ = fixture()
    result['actions'], result['trajectory'] = [], []
    result['policy']['requested_controls'] = []
    control = np.zeros(7)
    config = result['config']
    maximum = np.array([config['joint_target_rate_rad_s']*.001]*6+[config['jaw_target_rate_m_s']*.001])
    for i in range(1001):
        sample = max(0, i*20-1)
        obs = {field: arrays[field][sample].tolist()
               for field in ('object_position_m', 'joint_position_rad', 'tcp_position_m')}
        obs.update(native_actuator_controls=control.tolist(), command=None,
                   issued_joint_targets_rad=None, issued_gripper_position_m=None)
        result['trajectory'].append(obs)
        if i == 1000:
            break
        action = {'t_seconds': i*.02, 'joint_targets_rad': [(.01 if i % 2 else -.01), 0., 0., 0., 0., 0.],
                  'gripper_position_m': .005}
        result['actions'].append(action)
        result['policy']['requested_controls'].append({'index': i, **deepcopy(action)})
        obs.update(command=deepcopy(action), issued_joint_targets_rad=list(action['joint_targets_rad']),
                   issued_gripper_position_m=action['gripper_position_m'])
        target = np.array(action['joint_targets_rad']+[action['gripper_position_m']])
        for tick in range(i*20, (i+1)*20):
            control += np.clip(target-control, -maximum, maximum)
            arrays['arm_actuator_controls_rad'][tick] = control[:6]
            arrays['applied_jaw_target_m'][tick] = control[-1]
            arrays['requested_jaw_target_m'][tick] = target[-1]
    return result, arrays


class RetentionStudyTests(unittest.TestCase):
    def test_fresh_fixed_cells_balanced_order_and_every_attempt_once(self):
        cases = study.case_definitions()
        self.assertEqual(len(cases), 12)
        self.assertEqual([case['seed'] for case in cases], list(range(311100, 311112)))
        self.assertEqual({case['config']['mass_kg'] for case in cases}, {.045, .055})
        self.assertEqual({case['config']['friction'] for case in cases}, {.75, .50, .20})
        self.assertEqual({tuple(case['config']['object_position_m'][:2]) for case in cases},
                         {(.303, -.026), (.307, -.037)})
        roster = study.roster({'cases': cases})
        self.assertEqual(len({(case['id'], variant) for case, variant in roster}), 36)
        for variant in study.VARIANTS:
            self.assertEqual([sum(case['variant_order'][position] == variant for case in cases)
                              for position in range(3)], [4, 4, 4])

    def test_correct_five_second_retention_and_bound_receipt_integration(self):
        result, arrays, metadata, protocol = fixture()
        evaluation = study.evaluate(result, arrays, metadata, protocol)
        self.assertTrue(evaluation['primary']['success'])
        self.assertTrue(evaluation['stationarity']['grip_retention_and_stationarity'])
        self.assertTrue(evaluation['receipt']['verification']['admissible'])
        self.assertFalse(evaluation['physical_robot_ready'])

    def test_one_ms_force_gap_and_ceiling_breach_reset_admission_prefix(self):
        for force in (.01, 10.000001):
            result, arrays, metadata, protocol = fixture()
            arrays['pad_normal_forces_n'][1949, 0] = force
            evaluation = endpoint(result, arrays, metadata, protocol)
            self.assertFalse(evaluation['first_admission']['native_prefix_qualified'])
            self.assertEqual(evaluation['primary']['reason'], 'false_initial_admission')
            self.assertFalse(evaluation['primary']['success'])

    def test_99_native_prefix_samples_do_not_equal_100(self):
        result, arrays, _, _ = fixture()
        probe = result['policy']['probes'][0]
        _, streak, qualified, _, _ = study.probe_series(arrays, probe)
        self.assertEqual(streak[1197], 99)
        self.assertFalse(qualified[1197])
        self.assertEqual(streak[1198], 100)
        self.assertTrue(qualified[1198])

    def test_five_second_endpoint_includes_all_5000_followup_ticks(self):
        for failure_time, success in ((6.999, False), (7., False), (7.001, True)):
            result, arrays, metadata, protocol = fixture()
            arrays['pad_normal_forces_n'][round(failure_time/.001)-1, 0] = 0.
            evaluation = endpoint(result, arrays, metadata, protocol)
            self.assertEqual(evaluation['primary']['success'], success, failure_time)
            self.assertEqual(evaluation['primary']['complete_5s'], True)

    def test_less_than_five_seconds_followup_is_never_padded(self):
        for admission, complete in ((15., True), (15.02, False)):
            result, arrays, metadata, protocol = fixture(admission)
            evaluation = endpoint(result, arrays, metadata, protocol)
            self.assertEqual(evaluation['primary']['success'], complete)
            self.assertEqual(evaluation['primary']['complete_5s'], complete)
            if not complete:
                self.assertEqual(evaluation['primary']['reason'], 'insufficient_followup')

    def test_native_retention_does_not_resurrect_controller_abstention(self):
        result, arrays, metadata, protocol = fixture()
        result['policy']['admitted'] = False
        for claim in result['policy']['qualification_checks'][1:]:
            claim['qualified'] = False
        evaluation = study.evaluate(result, arrays, metadata, protocol)
        self.assertTrue(evaluation['primary']['success'])
        self.assertFalse(evaluation['primary']['controller_retained_at_5s'])
        self.assertFalse(evaluation['primary']['combined_qualified_pause'])
        self.assertFalse(evaluation['final']['both_qualified'])
        self.assertFalse(evaluation['final']['evidence_admissible'])

    def test_no_admission_and_invalid_native_attempts_are_retained_as_failures(self):
        result, arrays, metadata, protocol = fixture()
        result['policy']['events'] = []
        result['policy']['admitted'] = False
        evaluation = endpoint(result, arrays, metadata, protocol)
        self.assertEqual(evaluation['primary']['reason'], 'no_admission')
        self.assertIsNone(evaluation['first_admission'])
        self.assertFalse(evaluation['stationarity']['grip_retention_and_stationarity'])
        for field in ('warning', 'incomplete', 'critical_count_mismatch', 'critical_before_probe', 'malformed_sidecar'):
            changed, data, meta, protocol = fixture()
            if field == 'warning': changed['metrics']['warnings'] = ['native warning']
            elif field == 'incomplete': changed['actions'].pop()
            elif field == 'critical_count_mismatch': data['unexpected_contact_count'][100] = 1.
            elif field == 'critical_before_probe':
                data['unexpected_contact_count'][100] = 1.
                changed['metrics']['unexpected_contact_native_ticks'] = 1
            else: data['native_tick'][100] += 1.
            evaluation = endpoint(changed, data, meta, protocol)
            self.assertFalse(evaluation['primary']['success'], field)
            self.assertEqual(evaluation['status'], 'refused')

    def test_broader_post_endpoint_force_ceiling_veto_is_preserved(self):
        result, arrays, metadata, protocol = fixture()
        arrays['pad_normal_forces_n'][15000, 0] = 10.0001
        evaluation = endpoint(result, arrays, metadata, protocol)
        self.assertFalse(evaluation['primary']['success'])
        self.assertEqual(evaluation['primary']['reason'], 'force_ceiling_exceeded')
        self.assertFalse(evaluation['primary']['all_native_hold_forces_within_ceiling'])

    def test_common_tcp_position_and_orientation_endpoint_for_all_variants(self):
        for variant in study.VARIANTS:
            result, arrays, metadata, protocol = fixture()
            result['variant'] = variant
            # The tool and object move together: retained grip alone cannot
            # qualify a stationary pause.
            arrays['tcp_position_m'][2999:, 0] += .004001
            arrays['object_position_m'][2999:, 0] += .004001
            evaluation = endpoint(result, arrays, metadata, protocol)
            self.assertTrue(evaluation['primary']['success'])
            self.assertFalse(evaluation['stationarity']['grip_retention_and_stationarity'])
            self.assertAlmostEqual(evaluation['stationarity']['maximum_tcp_position_excursion_m'], .004001)
        result, arrays, metadata, protocol = fixture()
        angle = .050001
        arrays['tcp_quaternion_wxyz'][2999:] = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
        evaluation = endpoint(result, arrays, metadata, protocol)
        self.assertTrue(evaluation['primary']['success'])
        self.assertFalse(evaluation['stationarity']['grip_retention_and_stationarity'])
        self.assertAlmostEqual(evaluation['stationarity']['maximum_tcp_orientation_excursion_rad'], angle)
        arrays['tcp_quaternion_wxyz'][2999:] *= -1  # Quaternion sign changes no pose.
        opposite = endpoint(result, arrays, metadata, protocol)
        self.assertEqual(evaluation['stationarity'], opposite['stationarity'])

    def test_exact_stationarity_bounds_pass_and_terminal_recovery_cannot_hide_excursion(self):
        result, arrays, metadata, protocol = fixture()
        arrays['tcp_position_m'][2999:3999, 0] += .004
        arrays['object_position_m'][2999:3999, 0] += .004
        angle = .05
        arrays['tcp_quaternion_wxyz'][2999:3999] = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
        # With a z-only local offset, yaw preserves the redundant local field.
        evaluation = endpoint(result, arrays, metadata, protocol)
        self.assertTrue(evaluation['stationarity']['grip_retention_and_stationarity'])
        self.assertEqual(evaluation['stationarity']['terminal_tcp_position_delta_m'], [0., 0., 0.])
        arrays['tcp_position_m'][3499, 0] += .000001
        arrays['object_position_m'][3499, 0] += .000001
        changed = endpoint(result, arrays, metadata, protocol)
        self.assertFalse(changed['stationarity']['grip_retention_and_stationarity'])
        self.assertEqual(changed['stationarity']['terminal_tcp_position_delta_m'], [0., 0., 0.])

    def test_corrected_control_tape_requires_every_native_slew_sample_and_boundary(self):
        result, arrays = controls_fixture()
        study._controls(result, arrays)
        arrays['arm_actuator_controls_rad'][54, 0] += 1e-6
        with self.assertRaisesRegex(ValueError, 'slew law'):
            study._controls(result, arrays)
        result, arrays = controls_fixture()
        result['trajectory'][4]['issued_gripper_position_m'] += .001
        with self.assertRaisesRegex(ValueError, 'disagree'):
            study._controls(result, arrays)

    def test_prefix_excludes_first_hold_command_but_includes_admission_native_row(self):
        result, arrays = controls_fixture()
        original = study._prefix(result, arrays)
        result['actions'][100]['joint_targets_rad'][0] += .01  # Command at admission=2s.
        arrays['joint_position_rad'][2000, 0] += .01  # First post-admission native tick.
        self.assertEqual(study._prefix(result, arrays), original)
        arrays['joint_position_rad'][1999, 0] += .01
        self.assertNotEqual(study._prefix(result, arrays)['native_sha256'], original['native_sha256'])

    def test_cartesian_pre_admission_bound_refusal_keeps_common_acquisition_proof(self):
        measured, arrays = controls_fixture()
        issued = deepcopy(measured); issued['variant'] = 'issued_joint'
        cartesian = deepcopy(measured); cartesian['variant'] = 'cartesian'
        cartesian['policy']['events'] = [{'type': 'hold_invalidated', 't_seconds': 2.,
                                          'reason': 'initial_bias_exhausted', 'probe_index': 0}]
        cutoff = study._common_cutoff([measured, issued, cartesian])
        self.assertEqual(cutoff['time_s'], 2.)
        self.assertIn('pre_admission_bound_refusal', cutoff['reason'])
        self.assertEqual(len(cutoff['treatment_boundaries']), 3)
        # Different treatment command at the common boundary is expected.
        cartesian['actions'][100]['joint_targets_rad'][0] += .01
        prefixes = [study._prefix(row, arrays, cutoff=cutoff) for row in (measured, issued, cartesian)]
        self.assertEqual(prefixes[0], prefixes[1]); self.assertEqual(prefixes[0], prefixes[2])
        self.assertEqual(prefixes[0]['native_rows'], 2000)
        self.assertEqual(prefixes[0]['commands'], 100)
        cartesian['actions'][99]['joint_targets_rad'][0] += .01
        self.assertNotEqual(study._prefix(cartesian, arrays, cutoff=cutoff)['commands_sha256'],
                            prefixes[0]['commands_sha256'])

    def test_acquisition_alarm_is_not_a_treatment_cutoff_and_no_treatment_uses_full_run(self):
        result, arrays = controls_fixture()
        result['policy']['events'].insert(0, {'type': 'alarm', 't_seconds': 1.5, 'reason': 'relative_slip'})
        self.assertEqual(study._common_cutoff([result])['time_s'], 2.)
        result['policy']['events'] = [{'type': 'hold_invalidated', 't_seconds': 1.5,
                                       'reason': 'bilateral_contact_or_force_loss'}]
        cutoff = study._common_cutoff([result])
        self.assertIsNone(cutoff['time_s'])
        prefix = study._prefix(result, arrays, cutoff=cutoff)
        self.assertEqual(prefix['native_rows'], 20000)
        self.assertEqual(prefix['commands'], 1000)

    def test_correction_summary_does_not_mix_terminal_checks_with_issued_controls(self):
        record = {'command_issued': True, 'correction_rate_limited': True,
                  'correction_rate_limited_axes': [0, 2], 'correction_rad': [.01]*6,
                  'desired_correction_rad': [.02]*6, 'refused_bounds': None}
        final = {**record, 'command_issued': False, 'refused_bounds': 'tcp_position_error_exhausted'}
        summary = study._corrections({'retention': {'hold_records': [record, final], 'input_refusals': []}})
        self.assertEqual(summary['hold_record_count'], 2)
        self.assertEqual(summary['bound_refusal_counts'], {'tcp_position_error_exhausted': 1})
        self.assertEqual(summary['rate_limited_axis_occurrences'], 4)
        self.assertEqual(summary['maximum_abs_active_correction_rad'], .01)
        self.assertEqual(summary['maximum_abs_desired_correction_rad'], .02)

    def test_existing_output_and_started_study_are_not_overwritten(self):
        with TemporaryDirectory() as folder, patch.object(study, 'make_protocol') as make:
            with self.assertRaises(ValueError): study.prepare(folder, 'unused-assets')
            make.assert_not_called()
        protocol = {'schema': study.SCHEMA, 'cases': []}
        with TemporaryDirectory() as folder, patch.object(study, 'make_protocol', return_value=protocol), \
                patch.object(study, '_attempt') as attempt:
            study.write(Path(folder)/'protocol.json', protocol)
            study.write(Path(folder)/'started.json', {'original': 'retained'})
            with self.assertRaises(FileExistsError): study.run(folder, 'unused-assets')
            attempt.assert_not_called()
            self.assertEqual(study.read(Path(folder)/'started.json'), {'original': 'retained'})

    def test_changed_protocol_refuses_before_native_attempt(self):
        original = {'schema': study.SCHEMA, 'planned_attempts': 36, 'cases': []}
        with TemporaryDirectory() as folder, patch.object(study, 'make_protocol', return_value=original), \
                patch.object(study, '_attempt') as attempt:
            study.write(Path(folder)/'protocol.json', {**original, 'planned_attempts': 35})
            with self.assertRaises(ValueError): study.run(folder, 'unused-assets')
            attempt.assert_not_called()
            self.assertFalse((Path(folder)/'started.json').exists())

    def test_inspection_checks_frozen_protocol_bytes_before_any_report_or_source_read(self):
        with TemporaryDirectory() as folder, patch.object(study.prior, 'sources') as sources:
            path = Path(folder)/'protocol.json'
            study.write(path, {'planned_attempts': 36})
            study.write(Path(folder)/'frozen.json', {'protocol_sha256': sha256(path.read_bytes()).hexdigest()})
            path.write_bytes(path.read_bytes().replace(b'36', b'35'))
            with self.assertRaisesRegex(ValueError, 'Frozen protocol digest changed'):
                study.inspect(folder)
            sources.assert_not_called()


if __name__ == '__main__':
    unittest.main()
