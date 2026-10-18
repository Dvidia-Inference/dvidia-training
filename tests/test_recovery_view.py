"""Bounded recovery receipts preserve failures and distinguish supported opening."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from dvidia_training.improvement import brief, build_dashboard, load_benchmark
from dvidia_training.recovery_view import attach_recovery, load_recovery

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'benchmarks/controlled-recovery-20261015'


class RecoveryViewTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = load_benchmark(ROOT / 'benchmarks/improvement-20261015/benchmark')
        self.view = build_dashboard(load_benchmark(ROOT / 'benchmarks/sample-count-19047'), evaluation=self.evaluation)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name).resolve()
        self.reset()

    def tearDown(self):
        self.temp.cleanup()

    def reset(self):
        for name in ('protocol.json', 'report.json'):
            shutil.copyfile(ARCHIVE / 'phase-one' / name, self.path / name)

    def report(self, fn):
        path = self.path / 'report.json'
        value = json.loads(path.read_text())
        fn(value)
        path.write_text(json.dumps(value))

    def protocol(self, fn):
        path = self.path / 'protocol.json'
        value = json.loads(path.read_text())
        fn(value)
        path.write_text(json.dumps(value))
        self.report(lambda r: r.update(protocol_sha256=sha256(path.read_bytes()).hexdigest()))

    def test_actual_receipts_distinguish_support_from_task_completion(self):
        result = load_recovery(self.path, self.evaluation)
        self.assertEqual(result['completed_native_episodes'], 68)
        self.assertEqual(result['episode_errors'], 0)
        self.assertEqual(result['control_successes'], 0)
        rows = result['target']['rows']
        self.assertTrue(all(r['success'] is False for r in rows))
        self.assertTrue(all(r['support_confirmed_before_open'] for r in rows if r['variant'] == 'controlled_lowering'))
        # A later unsupported opening command does not erase the first
        # confirmed command, nor can first support establish full release.
        teacher = next(r for r in rows if r['variant'] == 'controlled_lowering' and r['policy'] == 'teacher')
        self.assertEqual(teacher['unsupported_release_actions'], 1)
        original = deepcopy(self.view)
        attached = attach_recovery(self.view, result)
        self.assertEqual(self.view, original)
        self.assertEqual(attached['summary'], original['summary'])
        self.assertEqual(attached['comparisons'], original['comparisons'])
        self.assertIn('First opening', attached['recovery']['headline'])
        self.assertIn(result['report_sha256'], brief(attached, attached['next_hypothesis_id']))

    def test_cases_capability_runtime_weights_and_deadline_cannot_change(self):
        for change in (
            lambda p: p['cases'][15]['scene'].update(jaw_force_limit=.6),
            lambda p: p['settings'].update(horizon_seconds=24),
            lambda p: p['settings'].update(lower_goal_cap_m=.006),
            lambda p: p.update(capsule_file_sha256='0'*64),
            lambda p: p.update(runtime={}),
            lambda p: p.update(benchmark_report_sha256='0'*64),
            lambda p: p.update(source_sha256={}),
        ):
            self.reset()
            self.protocol(change)
            with self.assertRaises(ValueError):
                load_recovery(self.path, self.evaluation)

    def test_missing_duplicate_and_changed_attempts_rejected(self):
        for change in (
            lambda r: r['results'].pop(),
            lambda r: r['results'].__setitem__(1, deepcopy(r['results'][0])),
            lambda r: r['results'][0].update(horizon=24),
            lambda r: r['results'][0].update(scene_sha256='0'*64),
            lambda r: r['results'][0].update(exact_original_trajectory_reproduced=False),
            lambda r: r['results'][2].update(identical_before_recovery=False),
            lambda r: r.update(completed_native_episodes=67),
        ):
            self.reset()
            self.report(change)
            with self.assertRaises(ValueError):
                load_recovery(self.path, self.evaluation)

    def test_support_confirmation_must_match_settled_first_opening(self):
        for change in (
            lambda s: s.update(table_contacts=0),
            lambda s: s.update(consecutive_control_samples=1),
            lambda s: s.update(object_velocity=[0., 0., .1]),
            lambda s: s.update(object_angular_velocity=[0., 0., 1.]),
            lambda s: s.update(object_position=[.4, 0., .5]),
            lambda s: s.update(time=19.),
        ):
            self.reset()
            self.report(lambda r: change(r['results'][2]['measurement']['recovery']['support_confirmation']))
            with self.assertRaises(ValueError):
                load_recovery(self.path, self.evaluation)

    def test_counts_and_opening_cannot_outlive_episode(self):
        for change in (
            lambda m: m['recovery'].update(lower_control_samples=901),
            lambda m: m['recovery'].update(unsupported_open_command_samples=901),
            lambda m: m['recovery'].update(contact_loss_during_lower_samples=500),
            lambda m: m['recovery']['first_open_command'].update(time=18.),
            lambda m: m.update(control_steps=899),
        ):
            self.reset()
            self.report(lambda r: change(r['results'][2]['measurement']))
            with self.assertRaises(ValueError):
                load_recovery(self.path, self.evaluation)

    def test_false_task_success_never_follows_from_supported_opening(self):
        self.report(lambda r: r['results'][2]['measurement'].update(success=True))
        with self.assertRaises(ValueError):
            load_recovery(self.path, self.evaluation)

    def test_errors_and_control_success_block_improvement(self):
        for mode in ('error', 'control'):
            self.reset()
            def mutate(r):
                if mode == 'error':
                    r['results'][2].update(status='error')
                    r['completed_native_episodes'] -= 1
                    r['episode_errors'] += 1
                else:
                    passed = next(row['measurement'] for row in r['results'] if row['measurement']['success'])
                    r['results'][2]['control'] = deepcopy(passed)
            self.report(mutate)
            result = load_recovery(self.path, self.evaluation)
            attached = attach_recovery(self.view, result)
            self.assertIn('Resolve', attached['recovery']['headline'])
            self.assertEqual(result['regression']['actuator_stress']['incomplete'], 1)

    def test_successful_summary_cannot_override_underlying_failed_trials(self):
        self.report(lambda r: r.update(summary={'all_passed': True}))
        result = load_recovery(self.path, self.evaluation)
        self.assertEqual(result['regression']['actuator_stress']['candidate_successes'], 3)
        self.assertEqual(result['regression']['actuator_stress']['improved'], 0)

    def test_followup_keeps_first_study_and_distinguishes_actual_clearance(self):
        studies = [load_recovery(ARCHIVE / phase, self.evaluation) for phase in ('phase-one', 'phase-two')]
        result = attach_recovery(self.view, studies)['recovery']
        self.assertEqual(result['completed_native_episodes'], 136)
        self.assertEqual(result['previous_studies'], [studies[0]])
        self.assertEqual(result['regression']['scenes'], 16)
        self.assertIn('Release clearance', result['headline'])
        self.assertTrue(all(r['release_clearance_confirmed'] for r in result['target']['rows'] if r['variant'] == 'controlled_lowering'))
        self.assertTrue(all(r['success'] is False for r in result['target']['rows']))

    def test_followup_requires_both_parent_bindings_and_no_duplicate_studies(self):
        a, b = [load_recovery(ARCHIVE / phase, self.evaluation) for phase in ('phase-one', 'phase-two')]
        for studies in ([b], [a, a], [a, b, b], [a, {**b, 'parent_protocol_sha256': '0'*64}]):
            with self.subTest(count=len(studies)), self.assertRaises(ValueError):
                attach_recovery(self.view, studies)

    def test_timeline_contains_measured_events_without_inventing_baseline_clearance(self):
        study = load_recovery(ARCHIVE / 'phase-two', self.evaluation)
        original, candidate = study['target']['rows'][:2]
        self.assertEqual([e['kind'] for e in original['events']],
                         ['recovery_start', 'first_open_command', 'episode_end'])
        self.assertIn('without table contact', original['events'][1]['label'])
        self.assertEqual([e['kind'] for e in candidate['events']],
                         ['recovery_start', 'first_open_command', 'jaw_clearance', 'episode_end'])
        for event, expected in zip(candidate['events'], (13.2, 15.6, 15.98, 18.0)):
            self.assertAlmostEqual(event['time_seconds'], expected)
        self.assertEqual(candidate['events'][-1]['state'], 'failed')
        self.assertIn('settled support', candidate['events'][1]['label'])

    def test_timeline_rejects_out_of_episode_or_reversed_recovery_start(self):
        for value in (-1, True, 19, float('inf')):
            self.reset()
            self.report(lambda r: r['results'][2]['measurement']['recovery']['first_recovery_observation'].update(time=value))
            with self.assertRaises(ValueError):
                load_recovery(self.path, self.evaluation)

    def test_actual_clearance_requires_open_jaws_no_contacts_and_shared_budget(self):
        for change in (
            lambda c: c.update(gripper_width=.04),
            lambda c: c.update(contacts={'left': 1, 'right': 0}),
            lambda c: c.update(table_contacts=0),
            lambda c: c.update(consecutive_control_samples=1),
            lambda c: c.update(elapsed_recovery_seconds=4.5),
            lambda c: c.update(time=19.),
            lambda c: c.update(object_velocity=[0., 0., .2]),
        ):
            for name in ('protocol.json', 'report.json'):
                shutil.copyfile(ARCHIVE / 'phase-two' / name, self.path / name)
            self.report(lambda r: change(r['results'][2]['measurement']['recovery']['actual_release_confirmation']))
            with self.assertRaises(ValueError):
                load_recovery(self.path, self.evaluation)

    def test_prior_error_cannot_disappear_from_aggregate_followup(self):
        studies = [load_recovery(ARCHIVE / phase, self.evaluation) for phase in ('phase-one', 'phase-two')]
        studies[0]['episode_errors'] = 1
        result = attach_recovery(self.view, studies)['recovery']
        self.assertIn('Resolve', result['headline'])
        self.assertEqual(result['episode_errors'], 1)


if __name__ == '__main__':
    unittest.main()
