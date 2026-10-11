"""Scoring denominators, authored references and causal pipeline separation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training.motion import MotionEstimator
from dvidia_training import motion_benchmark as benchmark


def reference_rows(count=11, fps=10):
    return [{'frame_index': index, 'source_pts_seconds': index/fps,
        'anchor_xy_pixels': [10+10*index, 100], 'visible': True,
        'velocity_scoreable': index > 0,
        'velocity_xy_pixels_per_second': [10*fps, 0] if index else None}
        for index in range(count)]


def estimated_rows(references):
    estimator = MotionEstimator(640, 360)
    return [estimator.update(ref['source_pts_seconds'], ref['anchor_xy_pixels'],
                             support_count=15, uncertainty_pixels=1)
            for ref in references]


class MotionScoresTest(unittest.TestCase):
    def test_perfect_constant_velocity_pairs_baseline_and_counts_terminal(self):
        refs = reference_rows()
        scores = benchmark.score_motion(estimated_rows(refs), refs, fps=10)
        first = scores['forecasts']['0.1']
        self.assertEqual(first['attempted_rows'], 11)
        self.assertEqual(first['future_reference_available'], 10)
        self.assertEqual(first['terminal_censored'], 1)
        self.assertEqual(first['velocity_warmup_abstentions'], 1)
        self.assertEqual(first['constant_velocity']['count'], 9)
        self.assertEqual(first['hold_current_paired']['count'], 9)
        self.assertAlmostEqual(first['constant_velocity']['mean'], 0)
        self.assertAlmostEqual(first['hold_current_paired']['mean'], 10)
        self.assertAlmostEqual(first['forecast_answer_coverage'], .9)
        self.assertEqual(scores['tracking']['track_coverage_visible'], 1)

    def test_untracked_frames_are_abstentions_not_zero_errors(self):
        refs = reference_rows()
        estimator = MotionEstimator(640, 360)
        rows = [estimator.update(r['source_pts_seconds'], None, status='lost') for r in refs]
        scores = benchmark.score_motion(rows, refs, fps=10)
        self.assertEqual(scores['tracking']['lost_rows'], 11)
        self.assertEqual(scores['tracking']['track_coverage_visible'], 0)
        first = scores['forecasts']['0.1']
        self.assertEqual(first['lost_abstentions'], 10)
        self.assertEqual(first['constant_velocity']['count'], 0)
        self.assertIsNone(first['constant_velocity']['p50'])
        self.assertEqual(first['forecast_answer_coverage'], 0)

    def test_unsampled_frames_remain_coverage_gaps(self):
        refs = reference_rows()
        rows = estimated_rows(refs)
        del rows[5]
        scores = benchmark.score_motion(rows, refs, fps=10)
        self.assertEqual(scores['tracking']['unsampled_reference_frames'], 1)
        self.assertAlmostEqual(scores['tracking']['track_coverage_visible'], 10/11)

    def test_current_off_grid_rows_cannot_borrow_nearest_frame_truth(self):
        refs = reference_rows()
        rows = estimated_rows(refs)
        for row in rows:
            row['acquired_time_seconds'] += .001
            for forecast in row['forecasts']:
                forecast['target_time_seconds'] += .001
        scores = benchmark.score_motion(rows, refs, fps=10)
        tracking = scores['tracking']
        self.assertEqual(tracking['unmatched_observation_rows'], 11)
        self.assertEqual(tracking['unsampled_reference_frames'], 11)
        self.assertEqual(tracking['anchor_error_pixels']['count'], 0)
        self.assertEqual(scores['forecasts']['0.1']['constant_velocity']['count'], 0)

    def test_future_reference_gap_is_explicit_and_does_not_round_across_gap(self):
        refs = reference_rows()
        rows = estimated_rows(refs)
        del refs[5]
        scores = benchmark.score_motion(rows, refs, fps=10)
        first = scores['forecasts']['0.1']
        self.assertEqual(first['future_reference_gap'], 1)
        self.assertEqual(first['current_reference_unmatched'], 1)
        self.assertEqual(scores['tracking']['unmatched_observation_rows'], 1)
        self.assertEqual(first['constant_velocity']['count'], 7)

    def test_half_frame_matching_uses_earlier_identity_and_discloses_offset(self):
        refs = reference_rows()
        scores = benchmark.score_motion(estimated_rows(refs), refs, fps=10)
        quarter = scores['forecasts']['0.25']
        self.assertAlmostEqual(quarter['constant_velocity']['mean'], 5)
        self.assertAlmostEqual(quarter['reference_max_absolute_time_offset_seconds'], .05)
        self.assertAlmostEqual(quarter['reference_time_offset_seconds']['mean'], -.05)
        self.assertEqual(quarter['terminal_censored'], 3)

    def test_scoring_does_not_change_any_estimator_output_or_reference(self):
        refs = reference_rows()
        rows = estimated_rows(refs)
        originals = deepcopy((rows, refs))
        benchmark.score_motion(rows, refs, fps=10)
        self.assertEqual((rows, refs), originals)
        changed = deepcopy(refs)
        changed[-1]['anchor_xy_pixels'] = [500, 200]
        before = benchmark.score_motion(rows, refs, fps=10)
        after = benchmark.score_motion(rows, changed, fps=10)
        self.assertNotEqual(before['forecasts'], after['forecasts'])
        self.assertEqual(rows, originals[0])

    def test_velocity_discontinuity_neighborhood_is_excluded(self):
        refs = reference_rows()
        rows = estimated_rows(refs)
        refs[5]['velocity_scoreable'] = False
        refs[5]['velocity_xy_pixels_per_second'] = [999999, 0]
        scores = benchmark.score_motion(rows, refs, fps=10)
        velocity = scores['tracking']['velocity_error_pixels_per_second']
        self.assertEqual(velocity['count'], 9)
        self.assertAlmostEqual(velocity['mean'], 0)

    def test_tracked_occlusion_not_counted_as_visible_tracking_accuracy(self):
        refs = reference_rows()
        rows = estimated_rows(refs)
        refs[5]['visible'] = False
        scores = benchmark.score_motion(rows, refs, fps=10)
        tracking = scores['tracking']
        self.assertEqual(tracking['reference_visible_frames'], 10)
        self.assertEqual(tracking['anchor_error_pixels']['count'], 10)
        self.assertEqual(tracking['tracked_during_reference_occlusion_rows'], 1)

    def test_uncertainty_is_evaluated_without_claiming_calibration(self):
        refs = reference_rows()
        scores = benchmark.score_motion(estimated_rows(refs), refs, fps=10)
        uncertainty = scores['forecasts']['0.1']['uncertainty']
        self.assertFalse(uncertainty['calibrated'])
        self.assertEqual(uncertainty['covered'], 9)
        self.assertEqual(uncertainty['evaluated'], 9)

    def test_invalid_time_identity_forecast_and_nonfinite_data_rejected(self):
        refs = reference_rows()
        rows = estimated_rows(refs)
        invalid_refs = deepcopy(refs)
        invalid_refs[5]['frame_index'] = invalid_refs[4]['frame_index']
        with self.assertRaises(ValueError):
            benchmark.score_motion(rows, invalid_refs, fps=10)
        invalid_rows = deepcopy(rows)
        invalid_rows[2]['acquired_time_seconds'] = invalid_rows[1]['acquired_time_seconds']
        with self.assertRaises(ValueError):
            benchmark.score_motion(invalid_rows, refs, fps=10)
        for field, value in [('target_time_seconds', 10), ('center_xy_pixels', [float('nan'), 1])]:
            invalid_rows = deepcopy(rows)
            invalid_rows[1]['forecasts'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                benchmark.score_motion(invalid_rows, refs, fps=10)

    def test_distribution_documents_interpolation_and_no_sample_nulls(self):
        self.assertEqual(benchmark.distribution([])['count'], 0)
        self.assertIsNone(benchmark.distribution([])['mean'])
        self.assertEqual(benchmark.distribution([0, 10])['p50'], 5)
        self.assertEqual(benchmark.distribution([0, 10])['p95'], 9.5)


def fake_fixtures(directory):
    directory = Path(directory)
    directory.mkdir()
    benchmark._write(directory/'source.json', {'kind': 'synthetic-manifest-test', 'episodes': []})
    result = []
    for case in benchmark.CASES:
        reference = directory/(case+'.reference.json')
        benchmark._write(reference, {'rows': reference_rows()})
        result.append({'id': case, 'reference_path': reference.name,
            'reference_sha256': benchmark._hash(reference),
            'seed_box_xyxy_normalized': [.1, .1, .2, .2]})
    return result


class BenchmarkAttemptTest(unittest.TestCase):
    def test_failed_attempts_preserve_receipts_and_cannot_appear_fast(self):
        calls = []
        def failure(manifest, episode_id, output, **kwargs):
            calls.append((episode_id, kwargs))
            self.assertEqual(set(kwargs), {'seed_box', 'label', 'sample_hz'})
            self.assertNotIn('reference', Path(manifest).read_text())
            output.mkdir(parents=True)
            receipt = {'kind': 'dvidia.motion-run.v1', 'status': 'failed',
                       'errors': [{'stage': 'decode', 'message': 'controlled failure'}]}
            benchmark._write(output/'run.json', receipt)
            return receipt
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(benchmark, 'generate_fixtures', fake_fixtures), \
                patch('dvidia_training.motion_video.analyze_video', failure):
            output = Path(temp)/'new-run'
            report = benchmark.run_benchmark(output)
            self.assertEqual(report['attempted'], 6)
            self.assertEqual(report['failed'], 6)
            self.assertEqual(len(calls), 6)
            self.assertEqual(report['timing']['completed_source_duration_seconds'], 0)
            self.assertIsNone(report['timing']['batch_processing_time_source_duration_ratio'])
            for row in report['cases']:
                self.assertTrue((output/row['run_path']).exists())
                self.assertEqual(row['errors'][0]['stage'], 'decode')

    def test_changed_evaluator_reference_invalidates_case_without_rewriting_run(self):
        def complete(manifest, episode_id, output, **kwargs):
            output.mkdir(parents=True)
            if episode_id == benchmark.CASES[0]:
                reference = Path(manifest).parent/(episode_id+'.reference.json')
                reference.write_text('{}')
            receipt = {'kind': 'dvidia.motion-run.v1', 'status': 'completed',
                       'rows': estimated_rows(reference_rows())}
            benchmark._write(output/'run.json', receipt)
            return receipt
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(benchmark, 'generate_fixtures', fake_fixtures), \
                patch('dvidia_training.motion_video.analyze_video', complete):
            output = Path(temp)/'new-run'
            report = benchmark.run_benchmark(output)
            self.assertEqual(report['completed'], 5)
            self.assertEqual(report['failed'], 1)
            first = report['cases'][0]
            self.assertIn('reference differs', first['error'])
            self.assertEqual(json.loads((output/first['run_path']).read_bytes())['status'], 'completed')
            self.assertIsNone(report['timing']['batch_processing_time_source_duration_ratio'])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class AuthoredFixtureTest(unittest.TestCase):
    def test_generated_media_references_and_manual_seeds_are_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'inputs'
            fixtures = benchmark.generate_fixtures(root)
            self.assertEqual(len(fixtures), 6)
            manifest = json.loads((root/'source.json').read_bytes())
            for fixture, episode in zip(fixtures, manifest['episodes'], strict=True):
                self.assertEqual(fixture['video_sha256'], benchmark._hash(root/fixture['video_path']))
                self.assertEqual(episode['license'], 'MIT')
                self.assertNotIn('reference', json.dumps(episode))
                self.assertNotIn('seed', json.dumps(episode))
                ref = json.loads((root/fixture['reference_path']).read_bytes())
                self.assertEqual(ref['role'], 'evaluator_only_never_supplied_to_tracker')
                self.assertEqual(len(ref['rows']), 120)
                self.assertEqual(ref['rows'][0]['source_pts_seconds'], 0)
                x1, y1, x2, y2 = fixture['seed_box_xyxy_normalized']
                self.assertAlmostEqual((x1+x2)*320, ref['rows'][0]['anchor_xy_pixels'][0])
                self.assertAlmostEqual((y1+y2)*180, ref['rows'][0]['anchor_xy_pixels'][1])
            self.assertEqual(list(root.glob('*.rgb')), [])

    def test_existing_benchmark_evidence_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp)/'report.json'
            marker.write_text('preserved')
            with self.assertRaises(ValueError):
                benchmark.run_benchmark(temp)
            self.assertEqual(marker.read_text(), 'preserved')


if __name__ == '__main__':
    unittest.main()
