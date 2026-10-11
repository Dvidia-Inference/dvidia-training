"""Causal image-motion contract, including variable frame times and loss resets."""
from copy import deepcopy
import json
import math
import unittest

from dvidia_training.motion import MotionEstimator, validate_motion_rows


class MotionEstimatorTest(unittest.TestCase):
    def test_two_actual_observations_use_variable_acquisition_times(self):
        estimator = MotionEstimator(640, 480)
        first = estimator.update(0, [100, 80], support_count=12, uncertainty_pixels=1)
        second = estimator.update(.04, [104, 82], support_count=11, uncertainty_pixels=2)
        third = estimator.update(.13, [113, 86.5], support_count=10, uncertainty_pixels=2)
        self.assertIsNone(first['velocity_xy_pixels_per_second'])
        self.assertEqual(first['forecasts'], [])
        for row in (second, third):
            self.assertAlmostEqual(row['velocity_xy_pixels_per_second'][0], 100)
            self.assertAlmostEqual(row['velocity_xy_pixels_per_second'][1], 50)
            self.assertAlmostEqual(row['speed_pixels_per_second'], math.hypot(100, 50))
            for forecast in row['forecasts']:
                h = forecast['horizon_seconds']
                self.assertAlmostEqual(forecast['target_time_seconds'], row['acquired_time_seconds']+h)
                self.assertAlmostEqual(forecast['center_xy_pixels'][0], row['observed_anchor_xy_pixels'][0]+100*h)
                self.assertAlmostEqual(forecast['center_xy_pixels'][1], row['observed_anchor_xy_pixels'][1]+50*h)
        self.assertEqual(third['velocity_evidence']['start_time_seconds'], .04)
        self.assertEqual(third['velocity_evidence']['end_time_seconds'], .13)
        self.assertEqual(first['observed_anchor_xy_pixels'], first['estimated_center_xy_pixels'])
        self.assertEqual(second['units']['velocity'], 'image_pixels_per_second')
        self.assertIn('not-physical-centroid', second['anchor_semantics'])

    def test_missing_and_explicit_loss_clear_continuity(self):
        estimator = MotionEstimator(100, 100)
        rows = [estimator.update(0, [20, 20]), estimator.update(.05, [21, 20]),
                estimator.update(.07, None), estimator.update(.1, [24, 20]),
                estimator.update(.12, [25, 20]), estimator.update(.13, None, status='lost'),
                estimator.update(.15, [27, 20])]
        for index in (2, 5):
            self.assertEqual(rows[index]['status'], 'lost')
            for key in ('observed_anchor_xy_pixels', 'estimated_center_xy_pixels',
                        'velocity_xy_pixels_per_second', 'speed_pixels_per_second',
                        'prediction_use_before_time_seconds'):
                self.assertIsNone(rows[index][key])
            self.assertEqual(rows[index]['forecasts'], [])
        for index in (3, 6):
            self.assertEqual(rows[index]['continuity'], 'loss_reset')
            self.assertIsNone(rows[index]['velocity_xy_pixels_per_second'])
            self.assertEqual(rows[index]['forecasts'], [])
        self.assertIsNotNone(rows[4]['velocity_xy_pixels_per_second'])
        self.assertEqual(rows[2]['loss_reason'], 'missing_observation')
        self.assertEqual(rows[5]['loss_reason'], 'tracking_lost')
        self.assertIs(validate_motion_rows(rows, 100, 100), rows)

    def test_gap_reset_needs_two_fresh_observations(self):
        estimator = MotionEstimator(100, 100, max_gap_seconds=.15)
        estimator.update(0, [10, 10])
        second = estimator.update(.15, [11, 10])
        self.assertEqual(second['continuity'], 'continued')
        third = estimator.update(.301, [50, 30])
        self.assertEqual(third['continuity'], 'gap_reset')
        self.assertIsNone(third['velocity_xy_pixels_per_second'])
        fourth = estimator.update(.35, [51, 30])
        self.assertEqual(fourth['continuity'], 'continued')
        self.assertEqual(fourth['velocity_evidence']['start_time_seconds'], .301)

    def test_future_observations_cannot_change_any_prefix(self):
        samples = [(0, [10, 20]), (.03, [12, 22]), (.08, [13, 24]),
                   (.1, None), (.12, [15, 25]), (.14, [16, 26])]
        def replay(sequence):
            estimator = MotionEstimator(100, 100)
            return [estimator.update(t, p) for t, p in sequence]
        whole = replay(samples)
        for length in range(1, len(samples)+1):
            self.assertEqual(replay(samples[:length]), whole[:length])
        modified = replay(samples[:-1]+[(.14, [90, 90])])
        self.assertEqual(modified[:-1], whole[:-1])

    def test_forecasts_are_unclipped_and_mark_out_of_frame(self):
        estimator = MotionEstimator(100, 100)
        estimator.update(0, [89, 10])
        row = estimator.update(.05, [99, 10])
        self.assertEqual(row['velocity_xy_pixels_per_second'], [200.0, 0.0])
        self.assertEqual(row['forecasts'][0]['center_xy_pixels'], [119.0, 10.0])
        self.assertTrue(all(f['out_of_frame'] for f in row['forecasts']))
        estimator = MotionEstimator(100, 100)
        estimator.update(0, [11, 10])
        row = estimator.update(.05, [1, 10])
        self.assertEqual(row['forecasts'][0]['center_xy_pixels'], [-19.0, 10.0])

    def test_radius_propagation_is_declared_not_probability(self):
        estimator = MotionEstimator(100, 100)
        estimator.update(0, [20, 20], uncertainty_pixels=1)
        row = estimator.update(.1, [21, 20], uncertainty_pixels=2)
        self.assertAlmostEqual(row['forecasts'][0]['uncertainty_radius_pixels'], 5)
        self.assertEqual(row['uncertainty_semantics'], 'empirical-envelope-not-calibrated-probability')
        estimator = MotionEstimator(100, 100)
        estimator.update(0, [20, 20])
        row = estimator.update(.1, [21, 20], uncertainty_pixels=2)
        self.assertTrue(all(f['uncertainty_radius_pixels'] is None for f in row['forecasts']))

    def test_freshness_lifetime_does_not_truncate_forecast_horizon(self):
        estimator = MotionEstimator(100, 100, max_prediction_age_seconds=.02)
        estimator.update(0, [10, 10])
        row = estimator.update(.1, [11, 10])
        self.assertAlmostEqual(row['prediction_use_before_time_seconds'], .12)
        self.assertAlmostEqual(row['forecasts'][-1]['target_time_seconds'], .6)

    def test_output_and_input_do_not_alias_estimator_state(self):
        estimator = MotionEstimator(100, 100)
        point = [10, 10]
        row = estimator.update(0, point)
        point[0] = 99
        row['observed_anchor_xy_pixels'][0] = 99
        row['estimated_center_xy_pixels'][0] = 99
        row['estimator']['horizons_seconds'][0] = 99
        next_row = estimator.update(.1, [11, 10])
        self.assertEqual(next_row['velocity_evidence']['start_anchor_xy_pixels'], [10.0, 10.0])
        self.assertEqual(next_row['velocity_xy_pixels_per_second'], [10.0, 0.0])
        self.assertEqual(next_row['forecasts'][0]['horizon_seconds'], .1)
        json.dumps(next_row, allow_nan=False)

    def test_numeric_inputs_reject_booleans_nan_infinity_and_overflow(self):
        invalid = [True, False, float('nan'), float('inf'), -float('inf'), '1', 10**1000]
        for value in invalid:
            with self.subTest(value=str(value)[:30]):
                with self.assertRaises(ValueError):
                    MotionEstimator(100, 100).update(value, [10, 10])
                with self.assertRaises(ValueError):
                    MotionEstimator(100, 100).update(0, [value, 10])
                with self.assertRaises(ValueError):
                    MotionEstimator(100, 100).update(0, [10, 10], uncertainty_pixels=value)
                with self.assertRaises(ValueError):
                    MotionEstimator(100, 100, horizons_seconds=[value])
        for time in (-1,):
            with self.assertRaises(ValueError):
                MotionEstimator(100, 100).update(time, [10, 10])
        for point in ([100, 0], [0, 100], [-1, 0], (1, 1), [1], [1, 2, 3]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                MotionEstimator(100, 100).update(0, point)

    def test_constructor_and_tracking_metadata_are_strict(self):
        for dimensions in [(True, 100), (100, False), (0, 100), (100, 1.5), (-1, 100)]:
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                MotionEstimator(*dimensions)
        for horizons in ([], (), None, [.1, .1], [.5, .1], [0], [-1]):
            with self.subTest(horizons=horizons), self.assertRaises(ValueError):
                MotionEstimator(100, 100, horizons_seconds=horizons)
        for keyword in ('max_gap_seconds', 'max_prediction_age_seconds'):
            for value in (0, -1, True, float('nan')):
                with self.subTest(keyword=keyword, value=value), self.assertRaises(ValueError):
                    MotionEstimator(100, 100, **{keyword: value})
        for status in ('pending', True, None):
            with self.assertRaises(ValueError):
                MotionEstimator(100, 100).update(0, [10, 10], status=status)
        for count in (True, -1, 1.0, float('nan')):
            with self.assertRaises(ValueError):
                MotionEstimator(100, 100).update(0, [10, 10], support_count=count)
        with self.assertRaises(ValueError):
            MotionEstimator(100, 100).update(0, [10, 10], status='lost')
        with self.assertRaises(ValueError):
            MotionEstimator(100, 100).update(0, None, uncertainty_pixels=1)

    def test_invalid_update_is_atomic_and_time_must_increase(self):
        estimator = MotionEstimator(100, 100)
        estimator.update(.1, [10, 10])
        for time in (.1, .05):
            with self.assertRaises(ValueError):
                estimator.update(time, [11, 10])
        with self.assertRaises(ValueError):
            estimator.update(.2, [100, 10])
        row = estimator.update(.2, [11, 10])
        self.assertEqual(row['velocity_evidence']['start_time_seconds'], .1)
        self.assertEqual(row['velocity_xy_pixels_per_second'], [10.0, 0.0])

    def test_derived_overflow_rejected_without_mutation(self):
        estimator = MotionEstimator(100, 100)
        estimator.update(0, [0, 0])
        with self.assertRaises(ValueError):
            estimator.update(1e-320, [1, 0])
        row = estimator.update(.1, [1, 0])
        self.assertEqual(row['velocity_xy_pixels_per_second'], [10.0, 0.0])


class MotionValidationTest(unittest.TestCase):
    def fixture(self):
        estimator = MotionEstimator(100, 100)
        return [estimator.update(0, [10, 10], uncertainty_pixels=1),
                estimator.update(.1, [11, 10], uncertainty_pixels=1),
                estimator.update(.2, None), estimator.update(.3, [12, 10])]

    def test_json_round_trip_and_empty_stream_are_valid(self):
        rows = json.loads(json.dumps(self.fixture(), allow_nan=False))
        self.assertIs(validate_motion_rows(rows, 100, 100), rows)
        self.assertEqual(validate_motion_rows([], 100, 100), [])

    def test_rejects_extra_fields_bad_units_and_forged_forecasts(self):
        mutations = [
            lambda r: r[1].update(extra='hidden-command'),
            lambda r: r[1]['units'].update(velocity='metres_per_second'),
            lambda r: r[1]['forecasts'][0]['center_xy_pixels'].__setitem__(0, 0),
            lambda r: r[1]['velocity_evidence'].update(start_time_seconds=0.05),
            lambda r: r[1].update(velocity_xy_pixels_per_second=[999.0, 0.0]),
            lambda r: r[2].update(estimated_center_xy_pixels=[12.0, 10.0]),
            lambda r: r[3].update(velocity_xy_pixels_per_second=[10.0, 0.0]),
            lambda r: r[1]['estimator'].update(max_gap_seconds=.2),
            lambda r: r[1].update(schema_version=True),
            lambda r: r[1].update(acquired_time_seconds=True),
            lambda r: r[1]['forecasts'][0].update(out_of_frame=0),
            lambda r: r[1].update(speed_pixels_per_second=float('nan')),
            lambda r: r[1].update(acquired_time_seconds=0),
        ]
        for mutate in mutations:
            rows = deepcopy(self.fixture()); mutate(rows)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_motion_rows(rows, 100, 100)

    def test_rejects_wrong_geometry_and_non_objects(self):
        with self.assertRaises(ValueError):
            validate_motion_rows(self.fixture(), 101, 100)
        for rows in (None, {}, (), [None], [{}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_motion_rows(rows, 100, 100)


if __name__ == '__main__':
    unittest.main()
