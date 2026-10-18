"""Diagnosis preserves benchmark failures, uncertainty and evidence bindings."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from dvidia_training.improvement import (brief, build_dashboard, capture_plan,
                                        diagnose, load_benchmark)

FIXTURE = Path(__file__).resolve().parents[1] / 'benchmarks/sample-count-19047'


class ImprovementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        for name in ('report.json', 'protocol.json', 'protocol.sha256', 'models-frozen.json'):
            shutil.copyfile(FIXTURE / name, self.root / name)
        self.bundle = load_benchmark(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def change_report(self, mutate):
        report = json.loads((self.root / 'report.json').read_text())
        mutate(report)
        (self.root / 'report.json').write_text(json.dumps(report))

    def test_known_candidate_and_shared_failures_get_different_actions(self):
        original = deepcopy(self.bundle)
        items = diagnose(self.bundle, 'groups-004')
        self.assertEqual(self.bundle, original)
        self.assertEqual([i['category'] for i in items], ['candidate', 'shared'])
        self.assertEqual(items[0]['evidence'][0]['case_id'], 'scene-04')
        self.assertEqual(items[1]['evidence'][0]['case_id'], 'scene-11')
        self.assertEqual(items[0]['request']['capture_mode'], 'aligned_simulation')
        self.assertEqual(items[1]['request']['capture_mode'], 'diagnostic')
        self.assertIsNone(items[1]['request']['required_groups'])
        self.assertTrue(all(i['status'] == 'provisional' for i in items))

    def test_empty_collection_plan_does_not_claim_existing_data_or_success(self):
        dashboard = build_dashboard(self.bundle)
        plan = capture_plan(dashboard)
        self.assertEqual(plan['evidence'], [])
        self.assertEqual(len(plan['cases']), 1)
        self.assertIn('simulation', plan['benchmark']['environment'])
        self.assertIn('unmeasured', brief(dashboard, 'candidate-investigation'))
        self.assertEqual(dashboard['comparisons'], [])

    def test_shared_failure_alone_does_not_request_more_training_videos(self):
        view = build_dashboard(self.bundle, 'groups-002')
        self.assertEqual(view['hypotheses'][0]['category'], 'shared')
        with self.assertRaises(ValueError):
            capture_plan(view)

    def test_unknown_baseline_and_brief_fail(self):
        with self.assertRaises(ValueError):
            build_dashboard(self.bundle, 'unknown')
        with self.assertRaises(ValueError):
            brief(build_dashboard(self.bundle), '../../private')

    def test_incomplete_trials_cannot_silently_disappear(self):
        self.change_report(lambda r: r['comparisons'][0]['outcomes'].pop())
        with self.assertRaisesRegex(ValueError, 'Every planned scene'):
            load_benchmark(self.root)

    def test_duplicate_outcome_does_not_count_twice(self):
        self.change_report(lambda r: r['comparisons'][0]['outcomes'].__setitem__(1, r['comparisons'][0]['outcomes'][0]))
        with self.assertRaisesRegex(ValueError, 'repeated'):
            load_benchmark(self.root)

    def test_negative_control_success_cannot_qualify(self):
        self.change_report(lambda r: r['comparisons'][0]['outcomes'][0].update(control_success=True))
        with self.assertRaisesRegex(ValueError, 'negative control'):
            load_benchmark(self.root)

    def test_execution_and_control_problems_precede_capture_requests(self):
        row = self.bundle['report']['comparisons'][1]['outcomes'][0]
        row.update(qualified=False, student_success=False, control_success=False, status='error')
        self.assertEqual(diagnose(self.bundle, 'groups-004')[0]['category'], 'execution')
        row.update(status='completed', student_success=True, control_success=True)
        self.assertEqual(diagnose(self.bundle, 'groups-004')[0]['category'], 'control')

    def test_report_cannot_rebind_protocol_or_models(self):
        self.change_report(lambda r: r.update(protocol_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'hashes'):
            load_benchmark(self.root)

    def test_frozen_candidate_fields_cannot_be_rewritten(self):
        self.change_report(lambda r: r['comparisons'][1].update(training_group_count=1))
        with self.assertRaisesRegex(ValueError, 'frozen model'):
            load_benchmark(self.root)

    def test_scene_binding_and_finite_clock(self):
        self.change_report(lambda r: r['comparisons'][0]['outcomes'][0].update(scene_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'frozen scene'):
            load_benchmark(self.root)
        shutil.copyfile(FIXTURE / 'report.json', self.root / 'report.json')
        self.change_report(lambda r: r.update(benchmark_wall_seconds=float('inf')))
        with self.assertRaises(ValueError):
            load_benchmark(self.root)

    def test_followup_cannot_reuse_inspected_development_layouts(self):
        with self.assertRaises(ValueError):
            build_dashboard(self.bundle, evaluation=deepcopy(self.bundle))

    def test_successful_but_unqualified_trial_is_not_a_shared_task_failure(self):
        row = self.bundle['report']['comparisons'][1]['outcomes'][0]
        row.update(qualified=False, student_success=True, control_success=False)
        item = diagnose(self.bundle, 'groups-004')[0]
        self.assertEqual(item['category'], 'qualification')
        self.assertEqual(item['request']['capture_mode'], 'diagnostic')

    def test_no_failures_does_not_invent_a_capture_request(self):
        comparison = self.bundle['report']['comparisons'][1]
        for row in comparison['outcomes']:
            row.update(student_success=True, qualified=True, control_success=False)
        self.assertEqual(diagnose(self.bundle, 'groups-004'), [])

    def test_archived_executed_pilot_remains_reviewable_without_raw_media(self):
        evaluation = load_benchmark(FIXTURE.parent / 'improvement-20261015' / 'benchmark')
        view = build_dashboard(self.bundle, evaluation=evaluation)
        self.assertEqual(view['conclusion']['status'], 'no_measured_gain')
        self.assertEqual([(c['nominal']['qualified'], c['stress']['qualified'])
                          for c in view['comparisons']], [(12, 3), (9, 2), (12, 3)])
        self.assertEqual(view['next_hypothesis_id'], 'shared-investigation')
        self.assertEqual(view['hypotheses'][0]['request']['capture_mode'], 'diagnostic')
        self.assertEqual(len(view['previous_hypotheses']), 2)


if __name__ == '__main__':
    unittest.main()
