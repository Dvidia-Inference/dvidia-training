"""Complete motor-memory trial denominators, frontier scope and offline receipts."""
from copy import deepcopy
import builtins
import json
import math
from unittest.mock import patch
import unittest

from dvidia_training.motor_metrics import render_report, summarize


def trial(case='move', **changes):
    record = {'case_id': case, 'split': 'development', 'variant': 'slow_feedback',
              'speed_limit_m_s': .2, 'seed': 101, 'holding': False,
              'expected_completion': True, 'valid': True, 'error': None,
              'target_final_m': .1, 'completed_s': .5, 'truth_settled_s': .48,
              'completion_false': False, 'success': True, 'final_error_m': .001,
              'final_speed_m_s': .001, 'overshoot_m': .002, 'rms_error_m': .03,
              'error_at_completion_m': .002, 'speed_at_completion_m_s': .004,
              'peak_error_after_completion_m': .0025,
              'peak_acceleration_m_s2': 1., 'peak_jerk_m_s3': 10., 'dropped': None,
              'peak_force_proxy_n': 0., 'contact_impulse_ns': 0., 'max_penetration_m': 0.,
              'reaction_s': None, 'reaction_reason': None, 'sensor_unusable_s': 0.,
              'predictor_rmse_m': None, 'command_latency_s': .02,
              'controller_latency_p50_s': .0001, 'controller_latency_p95_s': .0002,
              'controller_latency_max_s': .0003, 'simulated_seconds': 1.,
              'wall_seconds': .1, 'step_count': 1000, 'trace': []}
    record.update(changes)
    return record


class MotorMetricsTests(unittest.TestCase):
    def test_all_attempt_and_conditional_success_expose_invalid_and_unknowns(self):
        records = [trial(), trial('failure', seed=102, success=False, completed_s=None),
                   trial('invalid', seed=103, valid=False, error='native warning'),
                   trial('unknown', seed=104, success=None)]
        report = summarize(records)
        s = report['by_split']['development']
        self.assertEqual((report['attempted'], report['valid'], report['invalid']), (4, 3, 1))
        self.assertEqual(s['success']['conditional_valid']['estimate'], .5)
        self.assertEqual(s['success']['all_attempts']['estimate'], .25)
        self.assertEqual(s['success']['unknown_valid'], 1)
        self.assertEqual(s['expected_completion']['invalid_positive'], 1)
        self.assertEqual(s['completion']['expected_censored'], 1)

    def test_development_and_confirmation_are_not_pooled_for_a_success_claim(self):
        records = [trial(), trial('unseen', split='confirmation', success=False, completed_s=None)]
        report = summarize(records)
        self.assertEqual(report['by_split']['development']['success']['all_attempts']['estimate'], 1.)
        self.assertEqual(report['by_split']['confirmation']['success']['all_attempts']['estimate'], 0.)
        self.assertNotIn('overall', report)

    def test_appropriate_halt_is_separate_from_task_success_and_expectation_strata(self):
        records = [trial(), trial('fault', expected_completion=False, success=False,
                                  completed_s=None, reaction_s=.2, reaction_reason='stale')]
        s = summarize(records)['by_split']['development']
        self.assertEqual(s['success']['all_attempts']['estimate'], .5)
        self.assertEqual(s['by_expected_completion']['expected']['attempted'], 1)
        self.assertEqual(s['by_expected_completion']['not_expected']['attempted'], 1)
        self.assertEqual(s['completion']['expected_censored'], 0)
        self.assertEqual(s['measurements']['reaction_s']['count'], 1)

    def test_false_claims_include_superseded_claims_without_final_completion(self):
        records = [trial(), trial('early', seed=102, completion_false=True, completed_s=None,
                                  completion_claims=[{'at_s': .3, 'truth_verified': False}], success=False),
                   trial('unknown-claim', seed=103, completion_false=None)]
        completion = summarize(records)['by_split']['development']['completion']
        self.assertEqual(completion['valid_claims'], 3)
        self.assertEqual(completion['assessed_claims'], 2)
        self.assertEqual(completion['claim_truth_unknown'], 1)
        self.assertEqual(completion['false_claims'], 1)
        self.assertEqual(completion['false_claims_without_final_completion'], 1)
        self.assertEqual(completion['false_given_assessed_claim']['estimate'], .5)
        self.assertAlmostEqual(completion['false_claim_per_attempt']['estimate'], 1 / 3)

    def test_censored_completion_never_becomes_zero_time(self):
        records = [trial(), trial('censored', seed=102, success=False, completed_s=None,
                                  truth_settled_s=None)]
        s = summarize(records)['by_split']['development']
        self.assertEqual(s['completion']['expected_censored'], 1)
        self.assertEqual(s['completion']['truth_unsettled'], 1)
        self.assertEqual(s['measurements']['claimed_completion_s']['median'], .5)
        self.assertEqual(s['measurements']['claimed_completion_s']['missing'], 1)

    def test_revoked_claims_and_later_rest_loss_remain_visible(self):
        r = trial(completion_revoked=True, rest_lost_after_confirmation=True,
                  completed_s=None, truth_settled_s=None, success=False)
        summary = summarize([r])
        completion = summary['by_split']['development']['completion']
        self.assertEqual(completion['revoked'], 1)
        self.assertEqual(completion['rest_lost_after_confirmation'], 1)
        html = render_report({'records': [r], 'summary': summary})
        self.assertIn('revoked: 1', html)
        self.assertIn('rest lost after confirmation: 1', html)

    def test_frontier_uses_same_verified_success_cohort_and_prints_missing_failures(self):
        records = [trial(), trial('bad-fast', seed=102, variant='fast_fixed', success=False,
                                  completed_s=None, final_error_m=.02),
                   trial('false-fast', seed=103, variant='fast_fixed', success=True,
                         completion_false=True, completed_s=.1, final_error_m=.004),
                   trial('clean-adaptive', seed=104, variant='adaptive_predictive',
                         speed_limit_m_s=.28, completed_s=.3, final_error_m=.0005)]
        rows = summarize(records)['frontier']['development']
        slow = next(row for row in rows if row['variant'] == 'slow_feedback')
        fast = next(row for row in rows if row['variant'] == 'fast_fixed')
        adaptive = next(row for row in rows if row['variant'] == 'adaptive_predictive')
        self.assertEqual(slow['completion_time_s'], .5)
        self.assertEqual(slow['error_at_completion_m'], .002)
        self.assertIsNone(fast['completion_time_s'])
        self.assertEqual(fast['summary']['attempted'], 2)
        self.assertEqual(fast['summary']['completion']['false_claims'], 1)
        self.assertEqual(adaptive['speed_limit_m_s'], .28)
        self.assertEqual(adaptive['plotted_verified_success_count'], 1)

    def test_missing_claim_error_does_not_substitute_final_settled_error(self):
        r = trial(error_at_completion_m=None, final_error_m=0.)
        row = summarize([r])['frontier']['development'][0]
        self.assertIsNone(row['completion_time_s'])
        self.assertEqual(row['plotted_verified_success_count'], 0)
        self.assertEqual(row['summary']['measurements']['final_error_m']['median'], 0.)

    def test_matching_requires_identical_split_case_seed_and_speed(self):
        records = [trial(), trial(variant='adaptive_predictive', completed_s=.3, final_error_m=.0005),
                   trial(variant='fast_fixed', speed_limit_m_s=.28)]
        matched = summarize(records)['matched_comparisons']
        pair = next(row for row in matched if row['speed_limit_m_s'] == .2)
        adaptive = next(c for c in pair['comparisons'] if c['variant'] == 'adaptive_predictive')
        self.assertTrue(adaptive['both_valid'])
        self.assertAlmostEqual(adaptive['completed_s_candidate_minus_baseline'], -.2)
        unmatched = next(row for row in matched if row['speed_limit_m_s'] == .28)
        self.assertFalse(unmatched['comparisons'][0]['unique_pair'])

    def test_duplicate_matched_variant_is_visible_and_cannot_create_a_pair(self):
        records = [trial(), trial(), trial(variant='adaptive_predictive')]
        matched = summarize(records)['matched_comparisons'][0]
        self.assertEqual(matched['variant_attempt_counts']['slow_feedback'], 2)
        self.assertFalse(matched['comparisons'][1]['unique_pair'])
        self.assertIsNone(matched['comparisons'][1]['final_error_m_candidate_minus_baseline'])

    def test_invalid_comparison_retains_status_and_has_no_measured_delta(self):
        records = [trial(), trial(variant='adaptive_predictive', valid=False, error='bad engine state')]
        comparison = summarize(records)['matched_comparisons'][0]['comparisons'][1]
        self.assertTrue(comparison['unique_pair'])
        self.assertFalse(comparison['both_valid'])
        self.assertIsNone(comparison['completed_s_candidate_minus_baseline'])

    def test_false_completion_cannot_supply_an_apparent_completion_speedup(self):
        records = [trial(), trial(variant='adaptive_predictive', completed_s=.1, completion_false=True)]
        comparison = summarize(records)['matched_comparisons'][0]['comparisons'][1]
        self.assertTrue(comparison['candidate_completion_false'])
        self.assertIsNone(comparison['completed_s_candidate_minus_baseline'])

    def test_load_and_predictor_missingness_are_not_invented_zero_measurements(self):
        records = [trial(), trial('held', seed=102, holding=True, dropped=True),
                   trial('load-unknown', seed=103, holding=True, dropped=None)]
        s = summarize(records)['by_split']['development']
        self.assertEqual(s['load']['holding_valid'], 2)
        self.assertEqual(s['load']['outcome_unknown'], 1)
        self.assertEqual(s['load']['dropped'], 1)
        self.assertEqual(s['measurements']['predictor_rmse_m']['missing'], 3)
        self.assertIsNone(s['measurements']['predictor_rmse_m']['median'])

    def test_throughput_includes_invalid_work_but_missing_time_prevents_ratio(self):
        records = [trial(), trial('invalid', seed=102, valid=False, wall_seconds=.2,
                                  simulated_seconds=.5, step_count=500)]
        t = summarize(records)['by_split']['development']['throughput']
        self.assertAlmostEqual(t['steps_per_wall_second'], 5000.)
        records[1]['wall_seconds'] = None
        t = summarize(records)['by_split']['development']['throughput']
        self.assertIsNone(t['steps_per_wall_second'])
        self.assertEqual(t['wall_time_missing'], 1)

    def test_single_success_wilson_interval_keeps_uncertainty(self):
        rate = summarize([trial()])['by_split']['development']['success']['all_attempts']
        self.assertEqual(rate['estimate'], 1.)
        self.assertAlmostEqual(rate['ci95_wilson']['lower'], .2065493144)

    def test_empty_split_and_empty_report_have_json_safe_unknowns(self):
        s = summarize([])
        self.assertEqual(s['attempted'], 0)
        self.assertIsNone(s['by_split']['confirmation']['success']['all_attempts']['estimate'])
        self.assertEqual(s['frontier']['development'], [])
        json.dumps(s, allow_nan=False)
        html = render_report({'records': [], 'summary': s})
        self.assertIn('No verified-success', html)

    def test_nonfinite_nonnumeric_invalid_flags_and_unphysical_times_are_refused(self):
        for key, value in [('final_error_m', math.nan), ('final_error_m', math.inf),
                           ('peak_jerk_m_s3', -1), ('speed_limit_m_s', 0),
                           ('step_count', .5), ('step_count', True), ('success', 1),
                           ('valid', 1), ('split', 'mixed'), ('completed_s', 2.),
                           ('truth_settled_s', 2.), ('seed', True), ('trace', 'missing')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize([trial(**{key: value})])

    def test_html_escapes_receipts_titles_and_metadata_and_is_offline(self):
        records = [trial('<script>alert(1)</script>', variant='<img src=x onerror=bad>'),
                   trial('bad', seed=102, valid=False, error='<script>bad()</script>')]
        report = {'records': records, 'summary': summarize(records),
                  'memory': {'description': '<img src=x onerror=bad>'},
                  'protocol': {'scope': 'native only'}, 'physical_robot_ready': False}
        html = render_report(report)
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', html)
        self.assertIn('&lt;img src=x onerror=bad&gt;', html)
        self.assertIn('<svg', html)
        self.assertIn('default-src', html)
        self.assertIn('invalid', html)
        self.assertIn('Confirmation', html)

    def test_no_mutation_or_native_dependencies_for_reporting(self):
        records = [trial()]
        original = deepcopy(records)
        importer = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'mujoco'):
                raise AssertionError('optional/native dependency imported')
            return importer(name, *args, **kwargs)
        with patch('builtins.__import__', guarded):
            summary = summarize(records)
            render_report({'records': records, 'summary': summary})
        self.assertEqual(records, original)

    def test_html_trace_excerpt_keeps_receipts_without_mutating_full_evidence(self):
        r = trial(trace=[{'at_s': 0., 'state': 'first'},
                         {'at_s': .5, 'state': 'unique-middle-row'},
                         {'at_s': 1., 'state': 'last'}])
        before = deepcopy(r)
        html = render_report({'records': [r]})
        self.assertIn('trace_excerpt', html)
        self.assertIn('first', html)
        self.assertIn('last', html)
        self.assertNotIn('unique-middle-row', html)
        self.assertIn('href="trials.jsonl"', html)
        self.assertEqual(r, before)


if __name__ == '__main__':
    unittest.main()
