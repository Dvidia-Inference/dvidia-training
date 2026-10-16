"""Read actual diagnostic receipts; mutations are test-only, never measured data."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from dvidia_training.improvement import brief, build_dashboard, load_benchmark
from dvidia_training.grip_diagnostic_view import attach_diagnostics, load_diagnostic

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'benchmarks/grip-recovery-20261015'


class DiagnosticViewTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = load_benchmark(ROOT / 'benchmarks/improvement-20261015/benchmark')
        self.view = build_dashboard(load_benchmark(ROOT / 'benchmarks/sample-count-19047'),
                                    evaluation=self.evaluation)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name).resolve()
        for name in ('report.json', 'protocol.json'):
            shutil.copyfile(ARCHIVE / 'phase-one' / name, self.path / name)

    def tearDown(self):
        self.temp.cleanup()

    def mutate_report(self, fn):
        path = self.path / 'report.json'
        value = json.loads(path.read_text())
        fn(value)
        path.write_text(json.dumps(value))

    def mutate_protocol(self, fn):
        path = self.path / 'protocol.json'
        value = json.loads(path.read_text())
        fn(value)
        path.write_text(json.dumps(value))
        self.mutate_report(lambda r: r.update(protocol_sha256=sha256(path.read_bytes()).hexdigest()))

    def test_archived_two_phase_findings_preserve_original_qualification(self):
        studies = [load_diagnostic(ARCHIVE / phase, self.evaluation)
                   for phase in ('phase-one', 'phase-two')]
        original = deepcopy(self.view)
        result = attach_diagnostics(self.view, studies)
        self.assertEqual(self.view, original)
        self.assertEqual(result['summary'], original['summary'])
        self.assertEqual(result['comparisons'], original['comparisons'])
        diagnostic = result['diagnostic']
        self.assertEqual(diagnostic['completed_native_episodes'], 48)
        self.assertAlmostEqual(diagnostic['elapsed_seconds'], 53.405319, places=5)
        successes = [r['id'] for r in diagnostic['rows'] if r['student']['success']]
        self.assertEqual(successes, ['jaw060-horizon24'])
        self.assertIn('changed conditions', diagnostic['headline'])
        self.assertIn(studies[-1]['report_sha256'], brief(result, result['next_hypothesis_id']))

    def test_followup_requires_its_measured_parent(self):
        study = load_diagnostic(ARCHIVE / 'phase-two', self.evaluation)
        with self.assertRaisesRegex(ValueError, 'preceding'):
            attach_diagnostics(self.view, [study])

    def test_duplicate_study_cannot_inflate_run_count(self):
        study = load_diagnostic(ARCHIVE / 'phase-one', self.evaluation)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            attach_diagnostics(self.view, [study, deepcopy(study)])

    def test_claimed_success_requires_native_lift_dwell_and_success_reason(self):
        for fields in ({'success': True},
                       {'success': True, 'reason': 'success', 'lift_seen': True,
                        'placement_dwell_seconds': .01, 'final_target_error_m': .001}):
            shutil.copyfile(ARCHIVE / 'phase-one/report.json', self.path / 'report.json')
            self.mutate_report(lambda r: r['results'][0]['measurement'].update(fields))
            with self.assertRaises(ValueError):
                load_diagnostic(self.path, self.evaluation)

    def test_followup_original_conditions_require_trajectory_parity_too(self):
        for name in ('protocol.json', 'report.json'):
            shutil.copyfile(ARCHIVE / 'phase-two' / name, self.path / name)
        self.mutate_report(lambda r: r['results'][0].update(exact_original_trajectory_reproduced=False))
        with self.assertRaisesRegex(ValueError, 'reproduce'):
            load_diagnostic(self.path, self.evaluation)

    def test_source_benchmark_runtime_and_candidate_must_match(self):
        for field in ('benchmark_report_sha256', 'benchmark_protocol_sha256',
                      'capsule_file_sha256', 'runtime'):
            with self.subTest(field=field):
                shutil.copyfile(ARCHIVE / 'phase-one/protocol.json', self.path / 'protocol.json')
                shutil.copyfile(ARCHIVE / 'phase-one/report.json', self.path / 'report.json')
                self.mutate_protocol(lambda p: p.update({field: None}))
                with self.assertRaises(ValueError):
                    load_diagnostic(self.path, self.evaluation)

    def test_missing_duplicate_or_changed_outcomes_fail(self):
        mutations = [lambda r: r['results'].pop(),
                     lambda r: r['results'].__setitem__(1, deepcopy(r['results'][0])),
                     lambda r: r['results'][0].update(horizon=30),
                     lambda r: r['results'][0].update(exact_original_trajectory_reproduced=False),
                     lambda r: r.update(completed_native_episodes=27)]
        for fn in mutations:
            shutil.copyfile(ARCHIVE / 'phase-one/report.json', self.path / 'report.json')
            self.mutate_report(fn)
            with self.assertRaises(ValueError):
                load_diagnostic(self.path, self.evaluation)

    def test_invalid_metrics_and_extra_time_cannot_claim_success(self):
        for field, value in (('success', 'yes'), ('episode_wall_seconds', -1),
                             ('simulated_seconds', 100), ('final_target_error_m', float('inf'))):
            with self.subTest(field=field):
                shutil.copyfile(ARCHIVE / 'phase-one/report.json', self.path / 'report.json')
                self.mutate_report(lambda r: r['results'][0]['measurement'].update({field: value}))
                with self.assertRaises(ValueError):
                    load_diagnostic(self.path, self.evaluation)

    def test_incomplete_or_successful_controls_block_positive_interpretation(self):
        for bad in ('error', 'success'):
            studies = [load_diagnostic(ARCHIVE / phase, self.evaluation)
                       for phase in ('phase-one', 'phase-two')]
            if bad == 'error':
                studies[0]['rows'][0]['errors'] = 1
            else:
                studies[0]['rows'][0]['control_successes'] = 1
            result = attach_diagnostics(self.view, studies)
            self.assertIn('Resolve diagnostic', result['diagnostic']['headline'])


if __name__ == '__main__':
    unittest.main()
