"""Pure fixed-roster, endpoint and evidence-lifecycle tests; no physics."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import io
import gzip
import json
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_incremental_benchmark as study
from tests.test_piper_retention_benchmark import fixture, controls_fixture


def endpoint(result, arrays, metadata, protocol):
    with patch.object(study.prior, '_evaluate', return_value={'verification': {'admissible': False}}):
        return study.evaluate(result, arrays, metadata, protocol)


def seal(folder, protocol):
    path = Path(folder)/'protocol.json'
    study.write(path, protocol)
    study.write(Path(folder)/'frozen.json', {'protocol_sha256': sha256(path.read_bytes()).hexdigest()})


class IncrementalBenchmarkTests(unittest.TestCase):
    def test_exact_fresh_factorial_and_williams_order(self):
        cases = study.case_definitions()
        self.assertEqual(len(cases), 8)
        self.assertEqual([row['seed'] for row in cases], list(range(311200, 311208)))
        self.assertEqual({row['config']['mass_kg'] for row in cases}, {.0475, .0525})
        self.assertEqual({row['config']['friction'] for row in cases}, {.78, .52})
        self.assertEqual({tuple(row['config']['object_position_m']) for row in cases},
                         {(.304, -.024, .082), (.308, -.035, .082)})
        self.assertEqual(len({row['id'] for row in cases}), 8)
        self.assertEqual(len(study.roster({'cases': cases})), 32)
        self.assertEqual(len({(row['id'], variant) for row, variant in study.roster({'cases': cases})}), 32)
        self.assertEqual(study.ORDERS, ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1)))
        for variant in study.VARIANTS:
            self.assertEqual([sum(row['variant_order'][i] == variant for row in cases) for i in range(4)],
                             [2, 2, 2, 2])
        transitions = Counter((a, b) for row in cases
                              for a, b in zip(row['variant_order'], row['variant_order'][1:]))
        self.assertEqual(len(transitions), 12)
        self.assertEqual(set(transitions.values()), {2})

    def test_protocol_promotes_stationarity_and_keeps_both_contracts(self):
        previous = {'primary_endpoint': {'definition': 'native grip', 'predicate': 'all samples'},
                    'secondary_endpoint': {'name': 'grip_retention_and_tcp_stationarity'},
                    'controller_contract': {'baseline': True}}
        with patch.object(study.legacy, 'make_protocol', return_value=deepcopy(previous)), \
                patch.object(study, '_source_commit', return_value='a'*40):
            protocol = study.make_protocol('unused-assets')
        self.assertEqual(protocol['schema'], study.SCHEMA)
        self.assertEqual(protocol['planned_attempts'], 32)
        self.assertEqual(protocol['variants'], list(study.VARIANTS))
        self.assertEqual(protocol['controller_contract'], previous['controller_contract'])
        self.assertEqual(protocol['candidate_controller_contract']['candidate'], 'incremental')
        self.assertEqual(protocol['primary_endpoint']['name'], 'grip_retention_and_tcp_stationarity')
        self.assertEqual(protocol['grip_only_endpoint'], previous['primary_endpoint'])
        self.assertEqual(protocol['source_commit'], 'a'*40)

    def test_stationary_grip_and_controller_conjunction_succeed(self):
        result, arrays, metadata, protocol = fixture()
        evaluation = study.evaluate(result, arrays, metadata, protocol)
        self.assertTrue(evaluation['primary']['success'])
        self.assertTrue(evaluation['primary']['native_pause_success'])
        self.assertTrue(evaluation['primary']['combined_qualified_pause'])
        self.assertTrue(evaluation['grip_only']['success'])
        self.assertTrue(evaluation['receipt']['verification']['admissible'])
        self.assertIsNone(evaluation['primary']['first_tcp_pose_failure_s'])

    def test_one_native_pose_excursion_fails_primary_for_every_variant(self):
        for variant in study.VARIANTS:
            with self.subTest(variant=variant):
                result, arrays, metadata, protocol = fixture()
                result['variant'] = variant
                arrays['tcp_position_m'][3499, 0] += .004001
                arrays['object_position_m'][3499, 0] += .004001
                evaluation = endpoint(result, arrays, metadata, protocol)
                self.assertFalse(evaluation['primary']['success'])
                self.assertFalse(evaluation['primary']['combined_qualified_pause'])
                self.assertTrue(evaluation['primary']['native_pause_success'])
                self.assertTrue(evaluation['grip_only']['success'])
                self.assertEqual(evaluation['primary']['reason'], 'tcp_stationarity_lost')
                self.assertEqual(evaluation['primary']['first_tcp_pose_failure_s'], 3.5)
                self.assertEqual(evaluation['claims']['false_maintained_pose'], 2)
                self.assertEqual(evaluation['claims']['false_maintained_grip'], 0)
                self.assertEqual(evaluation['stationarity']['terminal_tcp_position_delta_m'], [0., 0., 0.])

    def test_orientation_sign_and_exact_bound_remain_common(self):
        result, arrays, metadata, protocol = fixture()
        angle = .05
        arrays['tcp_quaternion_wxyz'][3499] = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
        self.assertTrue(endpoint(result, arrays, metadata, protocol)['primary']['success'])
        angle += .000001
        arrays['tcp_quaternion_wxyz'][3499] = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
        first = endpoint(result, arrays, metadata, protocol)
        self.assertFalse(first['primary']['success'])
        arrays['tcp_quaternion_wxyz'][3499] *= -1
        second = endpoint(result, arrays, metadata, protocol)
        self.assertEqual(first['stationarity'], second['stationarity'])

    def test_five_second_pose_endpoint_and_full_run_force_veto(self):
        for time_s, expected in ((7., False), (7.001, True)):
            result, arrays, metadata, protocol = fixture()
            index = round(time_s/.001)-1
            arrays['tcp_position_m'][index, 0] += .004001
            arrays['object_position_m'][index, 0] += .004001
            self.assertEqual(endpoint(result, arrays, metadata, protocol)['primary']['success'], expected)
        result, arrays, metadata, protocol = fixture()
        arrays['pad_normal_forces_n'][15000, 0] = 10.0001
        evaluation = endpoint(result, arrays, metadata, protocol)
        self.assertFalse(evaluation['primary']['success'])
        self.assertEqual(evaluation['primary']['reason'], 'force_ceiling_exceeded')

    def test_terminal_fallback_cannot_become_active_control_success(self):
        result, arrays, metadata, protocol = fixture()
        result['policy']['admitted'] = False
        for claim in result['policy']['qualification_checks'][1:]:
            claim['qualified'] = False
        evaluation = endpoint(result, arrays, metadata, protocol)
        self.assertTrue(evaluation['primary']['success'])
        self.assertTrue(evaluation['grip_only']['success'])
        self.assertFalse(evaluation['primary']['controller_retained_at_5s'])
        self.assertFalse(evaluation['primary']['combined_qualified_pause'])

    def test_false_maintained_claims_include_between_boundary_loss(self):
        result, arrays, metadata, protocol = fixture()
        arrays['pad_normal_forces_n'][2004, 0] = 0.
        evaluation = endpoint(result, arrays, metadata, protocol)
        self.assertFalse(evaluation['primary']['success'])
        self.assertEqual(evaluation['claims']['false_maintained_grip'], 2)
        self.assertEqual(evaluation['claims']['false_maintained_pose'], 0)

    def test_no_admission_incomplete_or_invalid_attempts_stay_failed(self):
        for failure in ('no_admission', 'short_followup', 'incomplete', 'warning', 'malformed'):
            with self.subTest(failure=failure):
                result, arrays, metadata, protocol = fixture(15.02 if failure == 'short_followup' else 2.)
                if failure == 'no_admission':
                    result['policy']['events'] = []
                    result['policy']['admitted'] = False
                elif failure == 'incomplete': result['actions'].pop()
                elif failure == 'warning': result['metrics']['warnings'] = ['native warning']
                elif failure == 'malformed': arrays['native_tick'][100] += 1
                evaluation = endpoint(result, arrays, metadata, protocol)
                self.assertFalse(evaluation['primary']['success'])
                self.assertFalse(evaluation['primary']['combined_qualified_pause'])
                self.assertFalse(evaluation['grip_only']['success'])

    def test_aggregates_keep_grip_and_stationary_denominators_distinct(self):
        result, arrays, metadata, protocol = fixture()
        arrays['tcp_position_m'][3499, 0] += .004001
        arrays['object_position_m'][3499, 0] += .004001
        row = {'variant': 'incremental', 'valid': True,
               'evaluation': endpoint(result, arrays, metadata, protocol)}
        failed = {'variant': 'incremental', 'valid': False,
                  'evaluation': study._failed_evaluation('retained failure')}
        totals = study.aggregates([row, failed])['incremental']
        self.assertEqual(totals['attempts'], 2)
        self.assertEqual(totals['native_5s_pause_success'], 1)
        self.assertEqual(totals['native_5s_stationary_success'], 0)
        self.assertEqual(totals['combined_qualified_5s_pause'], 0)

    def test_four_way_prefix_includes_measured_margin_refusal(self):
        result, arrays = controls_fixture()
        rows = [{**deepcopy(result), 'variant': variant} for variant in study.VARIANTS]
        candidate = rows[-1]
        candidate['policy']['events'] = [{'type': 'hold_invalidated', 't_seconds': 1.98,
                                          'reason': 'measured_joint_margin_exhausted', 'probe_index': 0}]
        cutoff = study._common_cutoff(rows)
        self.assertEqual(cutoff['time_s'], 1.98)
        self.assertEqual(cutoff['reason'], 'pre_admission_bound_refusal')
        self.assertEqual(len(cutoff['treatment_boundaries']), 4)
        candidate['actions'][99]['joint_targets_rad'][0] += .01
        prefixes = [study._prefix(row, arrays, cutoff=cutoff) for row in rows]
        self.assertTrue(all(prefix == prefixes[0] for prefix in prefixes))
        self.assertEqual(prefixes[0]['native_rows'], 1980)
        self.assertEqual(prefixes[0]['commands'], 99)
        candidate['actions'][98]['joint_targets_rad'][0] += .01
        self.assertNotEqual(study._prefix(candidate, arrays, cutoff=cutoff), prefixes[0])
        self.assertEqual(study._prefix(candidate, arrays)['cutoff_time_s'], 1.98)

    def test_control_readback_still_checks_native_slew(self):
        result, arrays = controls_fixture()
        study._controls(result, arrays)
        arrays['arm_actuator_controls_rad'][54, 0] += 1e-6
        with self.assertRaisesRegex(ValueError, 'slew law'):
            study._controls(result, arrays)

    def test_prepare_does_not_replace_existing_evidence(self):
        with TemporaryDirectory() as folder, patch.object(study, 'make_protocol') as make:
            with self.assertRaises(ValueError): study.prepare(folder, 'unused-assets')
            make.assert_not_called()

    def test_started_study_cannot_resume(self):
        protocol = {'schema': study.SCHEMA, 'cases': []}
        with TemporaryDirectory() as folder, patch.object(study, 'make_protocol', return_value=protocol), \
                patch.object(study, '_attempt') as attempt:
            seal(folder, protocol)
            study.write(Path(folder)/'started.json', {'original': 'retained'})
            with self.assertRaises(FileExistsError): study.run(folder, 'unused-assets')
            attempt.assert_not_called()
            self.assertEqual(study.read(Path(folder)/'started.json'), {'original': 'retained'})

    def test_changed_protocol_or_freeze_rejected_before_native(self):
        for change_seal in (False, True):
            protocol = {'schema': study.SCHEMA, 'planned_attempts': 32, 'cases': []}
            with TemporaryDirectory() as folder, patch.object(study, 'make_protocol', return_value=protocol), \
                    patch.object(study, '_attempt') as attempt:
                seal(folder, {**protocol, 'planned_attempts': 31})
                if change_seal:
                    path = Path(folder)/'protocol.json'
                    path.write_bytes(path.read_bytes().replace(b'31', b'32'))
                with self.assertRaises(ValueError): study.run(folder, 'unused-assets')
                attempt.assert_not_called()
                self.assertFalse((Path(folder)/'started.json').exists())

    def test_inspect_checks_freeze_before_report_and_source_access(self):
        with TemporaryDirectory() as folder, patch.object(study.prior, 'sources') as sources:
            seal(folder, {'planned_attempts': 32})
            path = Path(folder)/'protocol.json'
            path.write_bytes(path.read_bytes().replace(b'32', b'31'))
            with self.assertRaisesRegex(ValueError, 'Frozen protocol digest changed'):
                study.inspect(folder)
            sources.assert_not_called()

    def test_all_32_invalid_attempts_are_written_and_read_back_without_resume(self):
        _, arrays, metadata, _ = fixture()
        empty = {name: value[:0].copy() for name, value in arrays.items()}
        metadata['rows'] = 0
        protocol = {'schema': study.SCHEMA, 'planned_attempts': 32, 'cases': study.case_definitions(),
                    'variants': list(study.VARIANTS), 'source_sha256': {}, 'runtime': {},
                    'source_commit': 'a'*40, 'force_ceiling_n': 10., 'asset_lock_sha256': 'b'*64}

        def refused_attempt(assets, case, variant):
            return ({'schema': study.SCHEMA, 'id': case['id']+'-'+variant, 'scene_id': case['id'],
                     'variant': variant, 'seed': case['seed'], 'config': case['config'],
                     'valid': False, 'error': {'type': 'RuntimeError', 'message': 'retained mock failure'},
                     'metrics': {'valid': False, 'warnings': [], 'simulation_time': 0.},
                     'policy': None, 'actions': [],
                     'trajectory': [{'tcp_position_m': [float('nan'), float('inf'), -float('inf')]}],
                     'timing': {'summaries': {}}},
                    empty, metadata)

        with TemporaryDirectory() as folder, patch.object(study, 'make_protocol', return_value=protocol), \
                patch.object(study.prior, 'sources', return_value={}), \
                patch.object(study.prior, 'runtime', return_value={}), \
                patch.object(study, '_attempt', side_effect=refused_attempt) as attempt, \
                patch('sys.stdout', new_callable=io.StringIO):
            seal(folder, protocol)
            report = study.run(folder, 'unused-assets')
            self.assertEqual(attempt.call_count, 32)
            self.assertEqual(report['attempted'], 32)
            self.assertEqual(report['valid_attempts'], 0)
            for aggregate in report['aggregates'].values():
                self.assertEqual(aggregate['attempts'], 8)
                self.assertEqual(aggregate['native_5s_stationary_success'], 0)
            # A later tooling commit does not change source-byte identity.
            with patch.object(study, '_source_commit', side_effect=AssertionError('inspect must not read HEAD')):
                inspection = study.inspect(folder)
            self.assertEqual(inspection['attempts'], 32)
            self.assertEqual(len(inspection['acquisition_prefixes']), 8)
            self.assertTrue(all(row['retained_refusal'] for row in inspection['rows']))
            self.assertTrue(all(not row['verified'] for row in inspection['acquisition_prefixes']))
            first = Path(folder)/report['rows'][0]['id']/'episode.json.gz'
            recorded = json.loads(gzip.decompress(first.read_bytes()))
            self.assertEqual(recorded['trajectory'][0]['tcp_position_m'][0],
                             {'invalid_nonfinite_number': 'nan'})
            self.assertEqual(len(recorded['json_invalid_numbers']), 3)
            self.assertFalse(recorded['valid'])
            with self.assertRaises(FileExistsError): study.run(folder, 'unused-assets')
            self.assertEqual(attempt.call_count, 32)

    def test_environment_construction_failure_retains_empty_unknown_native_evidence(self):
        with patch('dvidia_training.piper_feedback_env.FeedbackPiperTaskEnv',
                   side_effect=RuntimeError('construction failed')):
            result, arrays, metadata = study._attempt('unused-assets', study.case_definitions()[0], 'incremental')
        self.assertFalse(result['valid'])
        self.assertEqual(result['error']['message'], 'construction failed')
        self.assertEqual(len(arrays), 22)
        self.assertTrue(all(len(values) == 0 for values in arrays.values()))
        self.assertEqual(metadata['rows'], 0)
        self.assertIn('unavailable', metadata['recording_status'])
        self.assertFalse(study.evaluate(result, arrays, metadata, {'force_ceiling_n': 10.})['primary']['success'])

    def test_native_export_failure_retains_available_arrays_and_refuses(self):
        _, arrays, _, _ = fixture()
        arrays['joint_position_rad'][0, 0] = float('nan')
        with patch('dvidia_training.piper_feedback_env.FeedbackPiperTaskEnv') as constructor:
            env = constructor.return_value
            env.ready = False
            env.reset.side_effect = RuntimeError('reset failed')
            env.native_arrays.return_value = arrays
            env.native_metadata.side_effect = RuntimeError('metadata failed')
            result, saved, metadata = study._attempt('unused-assets', study.case_definitions()[0], 'incremental')
        self.assertFalse(result['valid'])
        self.assertEqual(result['error']['message'], 'reset failed')
        self.assertEqual(result['native_recording_error']['message'], 'metadata failed')
        self.assertEqual(metadata['rows'], 20000)
        self.assertTrue(np.isnan(saved['joint_position_rad'][0, 0]))
        self.assertIs(saved, arrays)

    def test_native_array_export_exception_retains_zero_rows(self):
        with patch('dvidia_training.piper_feedback_env.FeedbackPiperTaskEnv') as constructor:
            env = constructor.return_value
            env.ready = False
            env.reset.side_effect = RuntimeError('reset failed')
            env.native_arrays.side_effect = RuntimeError('array export failed')
            result, arrays, metadata = study._attempt('unused-assets', study.case_definitions()[0], 'incremental')
        self.assertFalse(result['valid'])
        self.assertEqual(result['native_recording_error']['message'], 'array export failed')
        self.assertEqual(len(arrays), 22)
        self.assertEqual(metadata['rows'], 0)


if __name__ == '__main__':
    unittest.main()
