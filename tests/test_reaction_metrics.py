"""Denominators, censored reaction timings and self-contained evidence reports."""
from copy import deepcopy
import builtins
import json
import math
from unittest.mock import patch
import unittest

from dvidia_training.reaction_metrics import render_report, summarize


def record(case='positive', **updates):
    result = {'case_id': case, 'variant': 'monitor', 'group': 'motion',
              'expected_detection': True, 'valid': True, 'hazard_onset_s': .1,
              'sensor_sample_s': .11, 'sensor_delivered_s': .12, 'decision_s': .13,
              'detected_s': .13, 'brake_requested_s': .14, 'brake_applied_s': .15,
              'stopped_s': .25, 'stopping_distance_m': .01,
              'stopping_displacement_m': .009, 'peak_unexpected_force_n': 0.,
              'unexpected_contact_impulse_ns': 0., 'minimum_clearance_m': .02,
              'max_penetration_m': 0., 'rest_dwell_s': .05, 'dropped': None,
              'simulated_seconds': 1., 'wall_seconds': .1, 'step_count': 1000,
              'error': None}
    result.update(updates)
    return result


class ReactionMetricTests(unittest.TestCase):
    def test_confusion_denominators_keep_invalid_and_unknown_attempts_visible(self):
        records = [record('tp'), record('fn', detected_s=None, brake_requested_s=None,
                                       brake_applied_s=None, stopped_s=None),
                   record('fp', expected_detection=False, hazard_onset_s=None),
                   record('tn', expected_detection=False, hazard_onset_s=None, detected_s=None,
                          brake_requested_s=None, brake_applied_s=None, stopped_s=None),
                   record('invalid-positive', valid=False, error='native warning'),
                   record('unknown', expected_detection=None),
                   record('invalid-unknown', valid=False, expected_detection=None, error='setup error')]
        summary = summarize(records)
        self.assertEqual((summary['attempted'], summary['valid'], summary['invalid']), (7, 5, 2))
        detection = summary['overall']['detection']
        self.assertEqual({key: detection[key] for key in ('tp', 'fn', 'fp', 'tn')},
                         {'tp': 1, 'fn': 1, 'fp': 1, 'tn': 1})
        self.assertEqual(detection['evaluated'], 4)
        self.assertEqual(detection['unknown_label'], 1)
        self.assertEqual(detection['attempted_positive'], 3)
        self.assertEqual(detection['invalid_positive'], 1)
        self.assertEqual(detection['attempted_unknown_label'], 2)
        self.assertEqual(detection['invalid_unknown_label'], 1)
        self.assertEqual(detection['accuracy']['estimate'], .5)
        self.assertEqual(detection['valid_correct_per_attempted_labeled']['denominator'], 5)
        self.assertEqual(detection['valid_correct_per_attempted_labeled']['estimate'], .4)
        self.assertEqual(detection['recall']['denominator'], 2)
        self.assertEqual(detection['false_positive_rate']['denominator'], 2)

    def test_no_positive_or_detected_examples_produce_null_rates(self):
        d = summarize([record(expected_detection=False, detected_s=None)])['overall']['detection']
        for name in ('precision', 'recall'):
            self.assertIsNone(d[name]['estimate'])
            self.assertIsNone(d[name]['ci95_wilson'])
        self.assertEqual(d['false_positive_rate']['estimate'], 0.)

    def test_wilson_interval_does_not_treat_one_success_as_certainty(self):
        rate = summarize([record()])['overall']['detection']['recall']
        self.assertEqual(rate['estimate'], 1.)
        self.assertAlmostEqual(rate['ci95_wilson']['lower'], .2065493144)
        self.assertAlmostEqual(rate['ci95_wilson']['upper'], 1.)

    def test_empty_summary_is_json_safe_and_has_no_fake_observations(self):
        s = summarize([])
        self.assertEqual(s['attempted'], 0)
        self.assertEqual(s['by_variant'], {})
        self.assertIsNone(s['overall']['throughput']['steps_per_wall_second'])
        self.assertIsNone(s['overall']['detection']['accuracy']['estimate'])
        json.dumps(s, allow_nan=False)

    def test_censored_stop_is_missing_in_distribution_and_counts_unstopped(self):
        s = summarize([record(), record('censored', stopped_s=None,
                                        stopping_distance_m=None, stopping_displacement_m=None)])['overall']
        self.assertEqual(s['stop']['brake_requested'], 2)
        self.assertEqual(s['stop']['unstopped_at_horizon'], 1)
        stop = s['measurements']['application_to_stop_s']
        self.assertEqual((stop['count'], stop['missing']), (1, 1))
        self.assertAlmostEqual(stop['median'], .1)
        self.assertEqual(s['measurements']['stopping_distance_m']['median'], .01)

    def test_shadow_detection_is_separate_from_command_and_stopping_metrics(self):
        s = summarize([record(), record('control', variant='disabled_control',
                                        brake_requested_s=None, brake_applied_s=None, stopped_s=None,
                                        stopping_distance_m=None, stopping_displacement_m=None)])
        control = s['by_variant']['disabled_control']
        self.assertEqual(control['detection']['tp'], 1)
        self.assertEqual(control['stop']['brake_requested'], 0)
        self.assertIsNone(control['measurements']['application_to_stop_s']['median'])

    def test_timing_stages_are_separate_and_onset_does_not_force_detection_order(self):
        r = record(hazard_onset_s=.2)
        s = summarize([r])['overall']['measurements']
        self.assertAlmostEqual(s['detection_latency_s']['median'], -.07)
        self.assertAlmostEqual(s['sample_to_delivery_s']['median'], .01)
        self.assertAlmostEqual(s['delivery_to_decision_s']['median'], .01)
        self.assertAlmostEqual(s['decision_to_brake_request_s']['median'], .01)
        self.assertAlmostEqual(s['request_to_application_s']['median'], .01)

    def test_incoherent_command_and_sensor_chains_are_refused(self):
        for key, value in [('stopped_s', .145), ('brake_applied_s', .135),
                           ('sensor_delivered_s', .105), ('decision_s', .115), ('detected_s', .2)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize([record(**{key: value})])

    def test_pre_roll_samples_and_float_clock_conversion_are_valid(self):
        r = record(sensor_sample_s=-.02, sensor_delivered_s=-.01,
                   decision_s=.13 + 1e-16, detected_s=.13 + 1e-16,
                   brake_requested_s=.13)
        self.assertEqual(summarize([r])['valid'], 1)
        self.assertEqual(summarize([r])['overall']['measurements']['decision_to_brake_request_s']['median'], 0.)

    def test_rest_loss_and_request_path_are_explicit(self):
        r = record(rest_lost_after_confirmation=True, request_to_rest_path_m=.03,
                   max_forward_excursion_m=.028, observation_stale_seconds=.5,
                   observation_unusable_seconds=.6)
        s = summarize([r])['overall']
        self.assertEqual(s['stop']['rest_lost_after_confirmation'], 1)
        self.assertEqual(s['measurements']['request_to_rest_path_m']['median'], .03)
        self.assertEqual(s['measurements']['observation_stale_seconds']['median'], .5)

    def test_nearest_rank_tail_and_worst_clearance_direction(self):
        records = [record(str(i), stopping_distance_m=i / 1000., minimum_clearance_m=-i / 1000.)
                   for i in range(1, 21)]
        metrics = summarize(records)['overall']['measurements']
        self.assertEqual(metrics['stopping_distance_m']['p95'], .019)
        self.assertAlmostEqual(metrics['stopping_distance_m']['median'], .0105)
        self.assertEqual(metrics['minimum_clearance_m']['worst'], -.02)

    def test_wall_throughput_counts_invalid_work_and_missing_durations_block_ratios(self):
        records = [record(), record('invalid', valid=False, wall_seconds=.2, simulated_seconds=.5,
                                    step_count=500, error='native warning')]
        throughput = summarize(records)['overall']['throughput']
        self.assertAlmostEqual(throughput['steps_per_wall_second'], 5000.)
        self.assertAlmostEqual(throughput['simulated_seconds_per_wall_second'], 5.)
        records[1]['wall_seconds'] = None
        throughput = summarize(records)['overall']['throughput']
        self.assertIsNone(throughput['steps_per_wall_second'])
        self.assertEqual(throughput['wall_time_missing'], 1)

    def test_load_retention_is_measured_only_when_supplied(self):
        s = summarize([record(dropped=True), record('held', dropped=False), record('no-load')])['overall']['load']
        self.assertEqual((s['known'], s['unknown'], s['dropped']), (2, 1, 1))
        self.assertEqual(s['drop_rate']['estimate'], .5)

    def test_group_and_variant_partitions_keep_every_attempt(self):
        records = [record(), record('fault', group='sensor_fault', valid=False),
                   record('control', group='sensor_fault', variant='disabled_control')]
        s = summarize(records)
        self.assertEqual(sum(v['attempted'] for v in s['by_variant'].values()), 3)
        self.assertEqual(sum(v['attempted'] for v in s['by_group'].values()), 3)
        self.assertEqual(s['by_variant_group']['monitor']['sensor_fault']['invalid'], 1)

    def test_nonnumeric_and_nonfinite_values_and_boolean_labels_are_refused(self):
        for key, value in [('wall_seconds', math.nan), ('wall_seconds', math.inf),
                           ('wall_seconds', -1), ('step_count', .5), ('step_count', True),
                           ('minimum_clearance_m', '0'), ('valid', 1), ('expected_detection', 1),
                           ('stopped_s', 10 ** 1000)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize([record(**{key: value})])

    def test_report_escapes_untrusted_text_and_preserves_invalid_evidence(self):
        r = record('<script>alert(1)</script>', valid=False,
                   error='<img src=x onerror=alert(2)>')
        report = {'protocol': {'name': '<script>bad</script>'}, 'scope': 'simulation-only',
                  'versions': {}, 'host': {}, 'records': [r], 'summary': summarize([r])}
        html = render_report(report)
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', html)
        self.assertIn('&lt;img src=x onerror=alert(2)&gt;', html)
        self.assertIn('invalid', html)
        self.assertIn('Whole-arm geometry protection is a future adapter', html)
        self.assertNotIn('<script src=', html)

    def test_summary_and_renderer_do_not_mutate_source_or_import_native_libraries(self):
        records = [record()]
        original = deepcopy(records)
        importer = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('mujoco', 'numpy'):
                raise AssertionError('optional dependency imported')
            return importer(name, *args, **kwargs)
        with patch('builtins.__import__', guarded):
            summary = summarize(records)
            render_report({'records': records, 'summary': summary})
        self.assertEqual(records, original)


if __name__ == '__main__':
    unittest.main()
