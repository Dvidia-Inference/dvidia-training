"""Retention UI must preserve measured failures, lineage and original limits."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from dvidia_training.improvement import build_dashboard, load_benchmark
from dvidia_training.retention_view import attach_retention, load_retention

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'benchmarks/retention-20261016'


class RetentionViewTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = load_benchmark(ROOT / 'benchmarks/improvement-20261015/benchmark')
        self.dashboard = build_dashboard(load_benchmark(ROOT / 'benchmarks/sample-count-19047'), evaluation=self.evaluation)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name).resolve()
        self.reset()

    def tearDown(self):
        self.temp.cleanup()

    def reset(self, phase='phase-one'):
        for name in ('protocol.json', 'report.json'):
            shutil.copyfile(ARCHIVE / phase / name, self.path / name)

    def report(self, fn):
        path = self.path / 'report.json'
        data = json.loads(path.read_text())
        fn(data)
        path.write_text(json.dumps(data))

    def protocol(self, fn):
        path = self.path / 'protocol.json'
        data = json.loads(path.read_text())
        fn(data)
        path.write_text(json.dumps(data))
        self.report(lambda r: r.update(protocol_sha256=sha256(path.read_bytes()).hexdigest()))

    def test_measured_runs_preserve_negative_outcome_and_native_lift_gain(self):
        studies = [load_retention(ARCHIVE / phase, self.evaluation) for phase in ('phase-one', 'phase-two')]
        result = attach_retention(self.dashboard, studies)['retention']
        self.assertEqual(result['completed_native_episodes'], 68)
        self.assertEqual(result['total_completed_native_episodes'], 136)
        self.assertEqual(result['regression']['scenes'], 16)
        self.assertEqual(result['episode_errors'], 0)
        self.assertEqual(result['control_successes'], 0)
        self.assertEqual(result['promotion'], 'not-promoted')
        self.assertEqual(result['headline'], 'Lift improved. Placement still fails.')
        original, candidate = result['target']['rows'][:2]
        self.assertIs(original['lift_seen'], False)
        self.assertIs(candidate['lift_seen'], True)
        self.assertIs(candidate['success'], False)
        self.assertEqual(result['regression']['actuator_stress']['candidate_successes'], 3)
        self.assertEqual(result['previous_studies'], [studies[0]])
        self.assertEqual(self.dashboard['task']['skillspace_binding'], 'unlinked')
        self.assertEqual(self.dashboard['task']['qualification']['physical_validation'], 'not_tested')

    def test_complete_attempts_controls_and_original_settings_are_required(self):
        for change in (
            lambda p: p['cases'][15]['scene'].update(jaw_force_limit=.6),
            lambda p: p['settings'].update(horizon_seconds=20),
            lambda p: p['settings'].update(smoothing_alpha=.4),
            lambda p: p['settings'].update(phases=['transfer']),
            lambda p: p['attempts'].pop(),
            lambda p: p.update(capsule_file_sha256='0'*64),
            lambda p: p.update(source_sha256={}),
        ):
            self.reset(); self.protocol(change)
            with self.assertRaises(ValueError):
                load_retention(self.path, self.evaluation)
        for change in (
            lambda r: r['results'].pop(),
            lambda r: r['results'].__setitem__(1, deepcopy(r['results'][0])),
            lambda r: r.update(completed_native_episodes=67),
            lambda r: r['results'][0].update(exact_original_trajectory_reproduced=False),
            lambda r: r['results'][2].update(identical_before_intervention=False),
            lambda r: r['results'][2]['measurement'].update(success=True),
        ):
            self.reset(); self.report(change)
            with self.assertRaises(ValueError):
                load_retention(self.path, self.evaluation)

    def test_events_cannot_invent_lift_or_change_thresholds_and_timing(self):
        def event(r, identity):
            return next(e for e in r['results'][2]['measurement']['retention']['events'] if e['id'] == identity)
        for change in (
            lambda r: event(r, 'offset-12mm').update(relative_offset_error_m=.001),
            lambda r: event(r, 'offset-12mm').update(time=19),
            lambda r: event(r, 'offset-12mm').update(contacts={'left': True, 'right': 1}),
            lambda r: event(r, 'offset-12mm').update(id='invented-event'),
            lambda r: r['results'][2]['measurement'].update(lift_seen=True),
            lambda r: r['results'][2]['measurement']['retention'].update(first_intervention_time=None),
            lambda r: r['results'][2]['measurement'].update(intervention_calls=9999),
            lambda r: r['results'][2]['measurement']['retention']['events'].append(deepcopy(event(r, 'offset-12mm'))),
            lambda r: r['results'][2]['measurement']['retention']['curve'][0].update(time=19),
            lambda r: r['results'][2]['measurement']['retention']['curve'][0].update(relative_offset_error_m=float('nan')),
        ):
            self.reset(); self.report(change)
            with self.assertRaises(ValueError):
                load_retention(self.path, self.evaluation)

    def test_followup_requires_parent_and_no_duplicate_study(self):
        a, b = [load_retention(ARCHIVE / p, self.evaluation) for p in ('phase-one', 'phase-two')]
        for studies in ([b], [a, a], [a, b, b], [a, {**b, 'parent_protocol_sha256': '0'*64}]):
            with self.assertRaises(ValueError):
                attach_retention(self.dashboard, studies)
        self.reset('phase-two')
        self.protocol(lambda p: p['settings'].update(maximum_downward_excursion_m=.05))
        with self.assertRaises(ValueError):
            load_retention(self.path, self.evaluation)

    def test_failed_execution_and_successful_controls_prevent_positive_claims(self):
        for mode in ('error', 'control'):
            self.reset()
            def change(r):
                if mode == 'error':
                    r['results'][2]['status'] = 'error'
                    r['completed_native_episodes'] -= 1
                    r['episode_errors'] += 1
                else:
                    passed = deepcopy(next(row['measurement'] for row in r['results'] if row['measurement']['success']))
                    passed['retention']['events'] = []
                    passed['retention']['curve'] = []
                    passed['retention']['first_intervention_time'] = None
                    passed['intervention_calls'] = 0
                    r['results'][2]['control'] = passed
            self.report(change)
            study = load_retention(self.path, self.evaluation)
            result = attach_retention(self.dashboard, [study])['retention']
            self.assertIn('needs review', result['headline'])
            self.assertEqual(result['regression']['actuator_stress']['incomplete'], 1)
            self.assertEqual(result['promotion'], 'not-promoted')

    def test_untrusted_summary_cannot_overwrite_measured_trials(self):
        self.report(lambda r: r.update(summary={'all_passed': True, 'promotion': 'promoted'}))
        study = load_retention(self.path, self.evaluation)
        self.assertEqual(study['regression']['actuator_stress']['candidate_successes'], 3)
        self.assertEqual(study['promotion'], 'not-promoted')

    def test_errors_in_earlier_studies_remain_visible(self):
        a, b = [load_retention(ARCHIVE / p, self.evaluation) for p in ('phase-one', 'phase-two')]
        a['episode_errors'] = 1
        result = attach_retention(self.dashboard, [a, b])['retention']
        self.assertEqual(result['episode_errors'], 1)
        self.assertIn('needs review', result['headline'])


if __name__ == '__main__':
    unittest.main()
