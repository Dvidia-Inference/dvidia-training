import base64
import json
from pathlib import Path
import tempfile
import unittest

from dvidia_training.image_goal_report import build_report


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfZkAAAAASUVORK5CYII=')


class ImageGoalReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / 'run'
        self.run.mkdir()
        (self.run / 'current.png').write_bytes(PNG)
        (self.run / 'goal.png').write_bytes(PNG)
        self.cases = [
            {'id': 'missing', 'category': 'missing-object', 'current_image': 'current.png', 'goal_image': 'goal.png', 'ground_truth': {'present': False, 'complete': False}},
            {'id': 'held', 'category': 'false-completion', 'current_image': 'current.png', 'goal_image': 'goal.png', 'ground_truth': {'present': True, 'complete': False}},
            {'id': 'finished', 'category': 'complete', 'current_image': 'current.png', 'goal_image': 'goal.png', 'ground_truth': {'present': True, 'complete': True}},
            {'id': 'unfinished', 'category': 'incomplete', 'current_image': 'current.png', 'goal_image': 'goal.png', 'ground_truth': {'present': True, 'complete': False}},
        ]
        self.write_cases()

    def tearDown(self):
        self.temp.cleanup()

    def write_cases(self):
        (self.run / 'cases.json').write_text(json.dumps({'cases': self.cases, 'conditions': ['rgb', 'rgb_telemetry']}))

    def result(self, rows, **extra):
        (self.run / 'results.json').write_text(json.dumps({'status': 'complete', 'model': 'jev-1.14.0', 'planned_conditions': ['rgb', 'rgb_telemetry'], 'results': rows, **extra}))

    def build(self):
        manifest = build_report(self.run, self.root / 'report')
        return json.loads((self.root / 'report/report.json').read_text()), manifest

    def test_missing_provider_is_unmeasured_not_zero_accuracy(self):
        data, manifest = self.build()
        self.assertEqual(data['status'], 'not_run')
        self.assertEqual(len(data['cases']), 4)
        for summary in data['summary']:
            self.assertEqual(summary['attempted'], 0)
            self.assertEqual(summary['not_run'], 4)
            self.assertEqual(summary['completion_accuracy']['answered'], 0)
            self.assertIsNone(summary['successful_response_latency_ms']['p95'])
        self.assertEqual(sum(name.startswith('images/') for name in manifest['files']), 1)
        self.assertIn('Jev has not been run.', (self.root / 'report/index.html').read_text())

    def test_denominators_keep_errors_abstentions_and_not_run_cases(self):
        self.result([
            {'case_id': 'missing', 'condition': 'rgb', 'status': 'ok', 'decision': {'present': True, 'complete': True}, 'p_complete': .9, 'timing_ms': {'total_ms': 100}},
            {'case_id': 'held', 'condition': 'rgb', 'status': 'error', 'timing_ms': {'total_ms': 5000}},
            {'case_id': 'finished', 'condition': 'rgb', 'status': 'ok', 'decision': {'present': True, 'complete': None}, 'timing_ms': {'total_ms': 300}},
        ])
        data, _ = self.build()
        summary = data['summary'][0]
        self.assertEqual(summary['planned'], 4)
        self.assertEqual(summary['attempted'], 3)
        self.assertEqual(summary['responses'], 2)
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['not_run'], 1)
        self.assertEqual(summary['completion_abstentions'], 1)
        self.assertEqual(summary['completion_accuracy'], {'correct': 0, 'answered': 1, 'planned': 4})
        self.assertEqual(summary['false_completion'], {'count': 1, 'answered_incomplete': 1, 'planned_incomplete': 3, 'valid_incomplete': 1})
        self.assertEqual(summary['false_presence'], {'count': 1, 'answered_absent': 1, 'planned_absent': 1, 'valid_absent': 1})
        self.assertEqual(summary['all_attempt_latency_ms']['max'], 5000)
        self.assertEqual(summary['successful_response_latency_ms']['max'], 300)
        self.assertEqual(summary['successful_response_latency_ms']['p50'], 200)

    def test_probabilities_never_invent_a_decision_and_private_fields_are_omitted(self):
        secret = 'private-credential-must-not-be-exported'
        self.cases[0]['raw_response'] = secret
        self.write_cases()
        self.result([{'case_id': 'missing', 'condition': 'rgb', 'status': 'ok', 'p_present': .99,
            'p_complete': .99, 'raw_response': secret, 'headers': {'Authorization': secret},
            'reason': '</script><script>alert(1)</script>', 'timing_ms': {'total_ms': 42, 'debug': secret}}], token=secret)
        data, _ = self.build()
        self.assertIsNone(data['results'][0]['decision']['complete'])
        self.assertEqual(data['summary'][0]['completion_accuracy']['answered'], 0)
        for path in (self.root / 'report').rglob('*'):
            if path.is_file():
                self.assertNotIn(secret.encode(), path.read_bytes())
                self.assertNotIn(b'alert(1)', path.read_bytes())

    def test_positive_recall_keeps_incomplete_abstained_error_and_unrun_cases(self):
        positive = self.cases[2]
        self.cases = [{**positive, 'id': f'positive-{i}'} for i in range(5)] + [self.cases[1]]
        self.write_cases()
        self.result([
            {'case_id': 'positive-0', 'condition': 'rgb', 'status': 'ok', 'verdict': 'complete'},
            {'case_id': 'positive-1', 'condition': 'rgb', 'status': 'ok', 'verdict': 'incomplete'},
            {'case_id': 'positive-2', 'condition': 'rgb', 'status': 'ok', 'verdict': 'unknown'},
            {'case_id': 'positive-3', 'condition': 'rgb', 'status': 'error', 'verdict': 'complete'},
            {'case_id': 'positive-4', 'condition': 'rgb', 'status': 'not_run'},
            {'case_id': 'held', 'condition': 'rgb', 'status': 'ok', 'verdict': 'incomplete'},
        ])
        data, _ = self.build()
        self.assertEqual(data['summary'][0]['true_completion'],
            {'count': 1, 'planned_complete': 5, 'answered_complete': 2, 'valid_complete': 3})
        self.assertEqual(data['summary'][0]['false_completion']['count'], 0)
        self.assertEqual(data['summary'][1]['true_completion']['planned_complete'], 5)
        self.assertEqual(data['summary'][1]['true_completion']['valid_complete'], 0)
        html = (self.root / 'report/index.html').read_text()
        self.assertIn('<option value="complete">Physically complete</option>', html)
        self.assertIn('Physically complete cases recognized in this input condition:', html)

    def test_explicit_provider_vocabulary_and_missing_credentials(self):
        self.result([
            {'case_id': 'missing', 'condition': 'rgb', 'status': 'ok', 'presence': 'absent', 'verdict': 'incomplete', 'timing_ms': {'total_ms': 10}},
            {'case_id': 'held', 'condition': 'rgb', 'status': 'missing_credentials'},
        ], status='missing_credentials')
        data, _ = self.build()
        self.assertEqual(data['results'][0]['decision'], {'present': False, 'complete': False})
        self.assertEqual(data['summary'][0]['attempted'], 1)
        self.assertEqual(data['summary'][0]['errors'], 0)
        self.assertEqual(data['summary'][0]['not_run'], 3)
        self.assertIsNone(data['results'][1]['verdict'])
        self.assertEqual(data['results'][1]['decision'], {'present': None, 'complete': None})

    def test_observability_labels_distinguish_correct_abstention_from_completion(self):
        for case in self.cases:
            case['physical_truth'] = case.pop('ground_truth')
            expected = 'complete' if case['physical_truth']['complete'] else 'incomplete'
            case['expected_verdict'] = {'rgb': expected, 'rgb_telemetry': expected}
        self.cases[2]['expected_verdict']['rgb'] = 'unknown'
        self.write_cases()
        self.result([
            {'case_id': 'finished', 'condition': 'rgb', 'status': 'ok', 'presence': 'present', 'verdict': 'unknown'},
            {'case_id': 'missing', 'condition': 'rgb', 'status': 'ok', 'presence': 'absent', 'verdict': 'incomplete'},
        ])
        data, _ = self.build()
        summary = data['summary'][0]
        self.assertEqual(summary['observability_aware_accuracy'], {'correct': 2, 'planned': 4})
        self.assertEqual(summary['known_verdict_accuracy'], {'correct': 1, 'planned': 3})
        self.assertEqual(summary['ambiguous_abstention'], {'correct': 1, 'planned': 1})
        self.assertEqual(summary['completion_accuracy']['answered'], 1)

    def test_rejects_duplicate_or_unplanned_results_instead_of_double_counting(self):
        row = {'case_id': 'missing', 'condition': 'rgb', 'status': 'ok'}
        self.result([row, row])
        with self.assertRaisesRegex(ValueError, 'unique planned'):
            self.build()
        self.assertFalse((self.root / 'report').exists())
        self.result([{**row, 'case_id': 'unplanned'}])
        with self.assertRaisesRegex(ValueError, 'unique planned'):
            self.build()

    def test_rejects_escaping_and_symlink_images_before_writing(self):
        (self.root / 'private.png').write_bytes(PNG)
        self.cases[0]['current_image'] = '../private.png'
        self.write_cases()
        with self.assertRaisesRegex(ValueError, 'inside'):
            self.build()
        self.assertFalse((self.root / 'report').exists())
        (self.run / 'linked.png').symlink_to(self.root / 'private.png')
        self.cases[0]['current_image'] = 'linked.png'
        self.write_cases()
        with self.assertRaisesRegex(ValueError, 'plain local'):
            self.build()
        self.assertFalse((self.root / 'report').exists())

    def test_refuses_existing_output_and_html_in_label(self):
        self.cases[0]['id'] = '</script>'
        self.write_cases()
        with self.assertRaisesRegex(ValueError, 'public case id'):
            self.build()
        self.cases[0]['id'] = 'missing'
        self.write_cases()
        (self.root / 'report').mkdir()
        (self.root / 'report/keep.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'fresh directory'):
            self.build()
        self.assertEqual((self.root / 'report/keep.txt').read_text(), 'keep')

    def test_inflight_request_cannot_be_hidden_as_missing(self):
        self.result([], status='running', active_trial={'case_id': 'missing', 'condition': 'rgb'})
        with self.assertRaisesRegex(ValueError, 'active or interrupted'):
            self.build()
        self.result([], status='not_run', active_trial={'case_id': 'missing', 'condition': 'rgb'})
        with self.assertRaisesRegex(ValueError, 'active or interrupted'):
            self.build()

    def test_external_results_keep_fixture_directory_immutable_and_count_unknown_cost(self):
        external = self.root / 'external-results.json'
        external.write_text(json.dumps({'status': 'completed', 'results': [
            {'case_id': 'missing', 'condition': 'rgb', 'status': 'ok', 'presence': 'absent',
             'verdict': 'incomplete', 'estimated_input_cost_usd': .001},
            {'case_id': 'held', 'condition': 'rgb', 'status': 'error'},
        ]}))
        build_report(self.run, self.root / 'report', results_path=external)
        data = json.loads((self.root / 'report/report.json').read_text())
        self.assertFalse((self.run / 'results.json').exists())
        self.assertEqual(data['summary'][0]['known_usage_estimated_input_cost_usd'], .001)
        self.assertEqual(data['summary'][0]['unknown_cost_attempts'], 1)


if __name__ == '__main__':
    unittest.main()
