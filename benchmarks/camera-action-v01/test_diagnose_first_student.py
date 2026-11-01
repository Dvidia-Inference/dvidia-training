"""Pure counterexamples for local diagnosis wording and stage membership."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('first_student_diagnosis', Path(__file__).with_name('diagnose_first_student.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def result(*, success=False, contacts=0, limited=977, missing=1.56):
    return dict(strict_success=success, bilateral_object_contact_ticks=contacts,
        complete_intervals=1000, rate_limited_intervals=limited,
        native_score=dict(unexpected_contact_native_ticks=1427 if not success else 0),
        first_missing_estimate_seconds=missing,
        closure_events=dict(jaw_target_first_below_20mm=dict(time_seconds=1.2, tcp_object_xy_distance_m=.133)))


class InterpretationTests(unittest.TestCase):
    def test_control_clock_rejects_wrong_event_alignment(self):
        steps = [dict(before=dict(timestamp_seconds=i * .02)) for i in range(1000)]
        self.assertEqual(module.control_clock(steps)[60], 1.2)
        for index, value in ((60, 1.201), (0, .02), (999, float('nan'))):
            changed = deepcopy(steps); changed[index]['before']['timestamp_seconds'] = value
            with self.assertRaisesRegex(ValueError, '50 Hz clock'): module.control_clock(changed)
        with self.assertRaisesRegex(ValueError, '50 Hz clock'): module.control_clock(steps[:-1])

    def test_measured_failure_matches_pattern(self):
        actual = module.interpretation(result(), result(success=True, contacts=100, limited=75),
            dict(first_more_than_10mm_seconds=.409))
        self.assertTrue(all(actual['hypothesis_predicates'].values()))
        self.assertIn('consistent with covariate shift', actual['hypothesis'])
        self.assertTrue(any('commanded' in text for text in actual['observed']))

    def test_success_and_late_divergence_cannot_repeat_failure_conclusions(self):
        actual = module.interpretation(result(success=True, contacts=300, limited=0, missing=.1), result(),
            dict(first_more_than_10mm_seconds=2.))
        text = ' '.join(actual['observed'])
        self.assertNotIn('No bilateral', text)
        self.assertNotIn('before any missing', text)
        self.assertIn('acquisition failure alone cannot explain', text)
        self.assertFalse(actual['hypothesis_predicates']['student_failed'])
        self.assertFalse(actual['hypothesis_predicates']['divergence_precedes_missing'])
        self.assertIn('no mechanism is inferred', actual['hypothesis'])

    def test_no_closure_and_no_divergence_are_reported(self):
        student = result(); student['closure_events']['jaw_target_first_below_20mm'] = None
        actual = module.interpretation(student, result(), dict(first_more_than_10mm_seconds=None))
        text = ' '.join(actual['observed'])
        self.assertIn('never commanded', text)
        self.assertIn('never separated', text)
        self.assertIn('no mechanism is inferred', actual['hypothesis'])


class StageBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); (self.root / 'stages').mkdir()
        self.name = 'development-initial'; self.ph = 'a' * 64; self.mh = 'b' * 64
        self.attempt = dict(id='initial-development-00-goal0-student', controller='student',
            case=dict(id='development-00-goal0', split='development'), negative_control=None)
        self.relative = 'episodes/' + self.name + '/' + self.attempt['id']
        self.episode = self.root / self.relative; self.episode.mkdir(parents=True)
        self.rollout = dict(case={**self.attempt['case'], 'policy_artifact_sha256': self.mh},
            score=dict(success=False), negative_control=None)
        self.write(self.episode / 'rollout.json', self.rollout)
        self.write(self.episode / 'integrity.json', dict(files={}))
        self.row = dict(**self.attempt, path=self.relative, status='complete', complete=True,
            recorded_complete_intervals=1000, policy_artifact_sha256=self.mh,
            integrity_sha256=module.identity((self.episode / 'integrity.json').read_bytes())['sha256'],
            receipt_sha256=module.identity((self.episode / 'rollout.json').read_bytes())['sha256'],
            strict_success=False, score=self.rollout['score'])
        self.protocol = dict(source_sha256={'source.py': 'c' * 64}, rosters={self.name: [self.attempt]})
        self.receipt = dict(schema='dvidia.camera-action-study/v1/stage', stage=self.name, status='complete',
            protocol_sha256=self.ph, model_bindings={'student': self.mh}, rows=[self.row], scheduled=1)
        self.started = dict(stage=self.name, protocol_sha256=self.ph,
            source_sha256=self.protocol['source_sha256'], model_bindings=self.receipt['model_bindings'])
        self.seal()

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def seal(self):
        path = self.root / 'stages' / (self.name + '.json')
        self.write(path, self.receipt)
        self.write(path.with_name(self.name + '.integrity.json'),
            dict(sha256=module.identity(path.read_bytes())['sha256'], protocol_sha256=self.ph))
        self.write(path.with_name(self.name + '.started.json'), self.started)

    def check(self):
        return module.stage_binding(self.root, self.name, self.attempt, self.protocol, self.ph)

    def test_binds_stage_start_seal_row_and_episode(self):
        row, sources = self.check()
        self.assertEqual(row, self.row); self.assertEqual(len(sources), 3)

    def test_changed_episode_manifest_refused(self):
        self.write(self.episode / 'integrity.json', dict(files={'changed': {}}))
        with self.assertRaisesRegex(ValueError, 'sealed stage row'): self.check()

    def test_changed_episode_receipt_refused(self):
        self.write(self.episode / 'rollout.json', dict(self.rollout, extra=True))
        with self.assertRaisesRegex(ValueError, 'sealed stage row'): self.check()

    def test_wrong_row_membership_duplicate_and_unsealed_stage_refused(self):
        original = deepcopy(self.receipt)
        for mutation in ('case', 'duplicate', 'model', 'status'):
            with self.subTest(mutation=mutation):
                self.receipt = deepcopy(original)
                if mutation == 'case': self.receipt['rows'][0]['case']['id'] = 'other'
                if mutation == 'duplicate': self.receipt['rows'].append(deepcopy(self.receipt['rows'][0]))
                if mutation == 'model': self.receipt['rows'][0]['policy_artifact_sha256'] = 'd' * 64
                if mutation == 'status': self.receipt['status'] = 'running'
                self.seal()
                with self.assertRaises(ValueError): self.check()

    def test_stage_score_cannot_disagree_with_bound_rollout(self):
        self.receipt['rows'][0]['score'] = dict(success=True)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'outcome differs'): self.check()

    def test_broken_stage_seal_and_source_freeze_refused(self):
        path = self.root / 'stages' / (self.name + '.json')
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'not sealed'): self.check()
        self.started['source_sha256'] = {}
        self.seal()
        with self.assertRaisesRegex(ValueError, 'started bindings'): self.check()


if __name__ == '__main__':
    unittest.main()
