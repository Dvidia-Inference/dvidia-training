"""Pure orchestration checks; these tests never initialize native physics."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dvidia_training import camera_action_study as study


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'study'
        self.fixed = self.base / 'fixed.json'
        study.write(self.fixed, {'fixture': 'fixed comparator'})
        self.runtime = patch.object(study, 'runtime_hashes', return_value={'fixture': '0' * 64})
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        with patch('dvidia_training.piper_motor_primitive.validate_model', side_effect=lambda value: value):
            study.prepare(self.root, fixed_model=self.fixed)

    def protocol(self):
        return study._protocol(self.root)

    def row(self, attempt, *, success=True, anchor=True, contacts=0):
        return dict(**deepcopy(attempt), status='complete', complete=True,
            strict_success=success, initial_anchor_available=anchor,
            initial_anchor_binding_valid=anchor, initial_anchor_error_m=.002, critical_contact_ticks=contacts,
            provenance_ok=attempt['controller'] != 'teacher', recorded_complete_intervals=1000,
            score=dict(valid=True, success=success, strict_v01=dict(trajectory_complete=True)))

    def receipt(self, stage, *, success=True):
        return dict(schema=study.SCHEMA + '/stage', stage=stage, status='complete',
            protocol_sha256=study.file_hash(self.root / 'protocol.json'),
            rows=[self.row(attempt, success=success) for attempt in self.protocol()['rosters'][stage]])

    def test_roster_counts_and_complete_split_isolation(self):
        p = self.protocol()
        self.assertEqual({name: len(rows) for name, rows in p['cases'].items()},
                         dict(train=64, development=16, confirmation=64))
        self.assertEqual({name: len(rows) for name, rows in p['rosters'].items()},
            dict(diagnostic=4, demonstrations=80, corrections=32,
                 **{'development-initial': 32, 'development-corrected': 32}, confirmation=256, negative=16))
        groups, points = {}, {}
        for split, cases in p['cases'].items():
            for case in cases:
                point = tuple(case['object_position_m'][:2])
                self.assertTrue(.287 <= point[0] <= .311 and -.042 <= point[1] <= -.018)
                self.assertNotIn(point, study._historical_starts())
                self.assertEqual(groups.setdefault(case['group_id'], split), split)
                self.assertEqual(points.setdefault(point, split), split)
        self.assertEqual(len(points), 72)
        self.assertEqual(len(groups), 72)
        correction = p['rosters']['corrections']
        self.assertEqual(len({a['case']['group_id'] for a in correction}), 32)
        self.assertTrue(all(a['case']['split'] == 'train' for a in correction))
        self.assertEqual([a['case']['goal_id'] for a in correction], [i % 2 for i in range(32)])
        self.assertEqual(p['cases'], study.registered_cases())

    def test_registration_is_exclusive_and_detects_runtime_or_protocol_changes(self):
        with self.assertRaises(ValueError):
            study.prepare(self.root, fixed_model=self.fixed)
        with patch.object(study, 'runtime_hashes', return_value={'fixture': '1' * 64}):
            with self.assertRaisesRegex(ValueError, 'Runtime changed'):
                study._protocol(self.root, runtime=True)
        protocol = self.root / 'protocol.json'
        protocol.write_bytes(protocol.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'protocol bytes changed'):
            study.inspect(self.root)

    def test_group_and_per_goal_quality_gates_are_both_required(self):
        rows = self.receipt('demonstrations')['rows']
        self.assertTrue(study.data_quality(rows, self.protocol())['passed'])
        # Sixty-two successes are insufficient if a whole starting group is absent.
        rows[0]['strict_success'] = rows[1]['strict_success'] = False
        result = study.data_quality(rows, self.protocol())
        self.assertFalse(result['passed'])
        self.assertEqual(result['missing_training_groups'], ['train-00'])
        rows = self.receipt('demonstrations')['rows']
        for row in rows[:10:2]:
            row['strict_success'] = False
        result = study.data_quality(rows, self.protocol())
        self.assertEqual(result['covered_training_groups'], 32)
        self.assertEqual(result['successful_episodes_per_goal']['0'], 27)
        self.assertFalse(result['passed'])

    def test_missing_initial_anchor_is_not_usable_training_data(self):
        rows = self.receipt('demonstrations')['rows']
        rows[0]['initial_anchor_available'] = False
        rows[1]['initial_anchor_binding_valid'] = False
        self.assertFalse(study.data_quality(rows, self.protocol())['passed'])

    def test_diagnostic_rejects_available_but_inaccurate_or_missing_camera_error(self):
        receipt = self.receipt('diagnostic')
        receipt['rows'][0]['initial_anchor_error_m'] = .010001
        with patch.object(study, '_run_stage', return_value=receipt):
            result = study.collect(self.root, '/unused/assets')
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('10 mm', result['reason'])

    def test_diagnostic_requires_finite_native_error_measurement(self):
        receipt = self.receipt('diagnostic')
        receipt['rows'][0]['initial_anchor_error_m'] = None
        with patch.object(study, '_run_stage', return_value=receipt):
            result = study.collect(self.root, '/unused/assets')
        self.assertEqual(result['status'], 'blocked')

    def test_blocked_diagnostic_stops_training_and_exposes_ui_reason(self):
        receipt = self.receipt('diagnostic')
        receipt['rows'][0]['strict_success'] = False
        with patch.object(study, '_run_stage', return_value=receipt):
            result = study.collect(self.root, '/unused/assets')
        self.assertEqual(result['status'], 'blocked')
        state = study.inspect(self.root)
        self.assertEqual(state['status'], 'blocked')
        self.assertEqual(state['next_actions'], [])
        self.assertFalse(state['qualified'])
        with self.assertRaisesRegex(ValueError, 'diagnostic must be complete'):
            study.collect(self.root, '/unused/assets', stage='demonstrations')
        with self.assertRaisesRegex(ValueError, 'demonstrations must be complete'):
            study.train(self.root)

    def test_inspection_is_read_only_and_has_one_action(self):
        before = {str(p.relative_to(self.root)): study.file_hash(p)
                  for p in self.root.rglob('*') if p.is_file()}
        state = study.inspect(self.root)
        self.assertEqual(state['next_actions'], [dict(command='collect', stage='diagnostic', requires_assets=True)])
        self.assertFalse(state['physics_executed'])
        after = {str(p.relative_to(self.root)): study.file_hash(p)
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_stage_receipt_tamper_is_detected(self):
        study._seal_stage(self.root, 'diagnostic', self.receipt('diagnostic'))
        target = self.root / 'stages/diagnostic.json'
        target.write_bytes(target.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'receipt identity changed'):
            study.inspect(self.root)

    def test_disk_reserve_and_study_budget_refuse_before_next_attempt(self):
        with patch.object(study.shutil, 'disk_usage', return_value=SimpleNamespace(free=study.DISK_RESERVE_BYTES)):
            with self.assertRaisesRegex(ValueError, '30 GiB'):
                study.storage_preflight(self.root)

    def test_second_worker_is_refused_and_never_clears_existing_lock(self):
        lock = self.root / 'worker.lock'
        study.write(lock, {'pid': 123, 'operation': 'another worker'})
        with self.assertRaisesRegex(ValueError, 'active or interrupted worker'):
            study.collect(self.root, '/unused/assets')
        self.assertTrue(lock.exists())
        with patch.object(study, 'STUDY_MAX_BYTES', 1):
            with self.assertRaisesRegex(ValueError, '120 GiB'):
                study.storage_preflight(self.root)

    def test_prior_unsealed_attempt_is_unknown_and_never_retried(self):
        p = self.protocol()
        for attempt in p['rosters']['diagnostic']:
            directory = self.root / 'episodes/diagnostic' / attempt['id']
            directory.mkdir(parents=True)
            study.write(directory / 'started.json', {'status': 'interrupted before receipt'})
        with patch('dvidia_training.camera_action_capture.run_rollout') as run:
            result = study.collect(self.root, '/unused/assets')
        run.assert_not_called()
        self.assertEqual(result['status'], 'blocked')
        self.assertTrue(all(r['status'] == 'unknown' for r in result['rows']))

    def test_paired_bootstrap_keeps_both_goals_in_one_group(self):
        rows = self.receipt('confirmation')['rows']
        for row in rows:
            row['strict_success'] = row['controller'] == 'student'
        interval = study.paired_bootstrap(rows, 'fixed')
        self.assertEqual(interval['lower'], 1.)
        self.assertEqual(interval['upper'], 1.)
        self.assertTrue(interval['positive'])
        for row in rows:
            row['strict_success'] = row['case']['goal_id'] == 0
        tied = study.paired_bootstrap(rows, 'fixed')
        self.assertEqual(tied['lower'], 0.)
        self.assertFalse(tied['positive'])
        with self.assertRaisesRegex(ValueError, '32 complete'):
            study.paired_bootstrap(rows[4:], 'fixed')

    def test_qualification_requires_improvement_not_just_58_successes(self):
        confirmation = self.receipt('confirmation')
        negatives = self.receipt('negative', success=False)
        for row in negatives['rows']:
            if row['negative_control'] != 'open_gripper':
                row.update(status='refused', complete=False, recorded_complete_intervals=0)
        study.write(self.root / 'freeze.json', dict(schema=study.SCHEMA + '/freeze', selected_stage='initial',
            source_sha256=self.protocol()['source_sha256'], protocol_sha256=study.file_hash(self.root/'protocol.json'),
            fixed_model_sha256=study.file_hash(self.root/'fixed-model.json'), artifacts={}))
        for receipt in (confirmation, negatives):
            receipt['freeze_sha256'] = study.file_hash(self.root / 'freeze.json')
            study._seal_stage(self.root, receipt['stage'], receipt)
        report = study.report_study(self.root)
        self.assertEqual(report['controllers']['student']['strict_successes'], 64)
        self.assertTrue(report['gates']['at_least_58_of_64'])
        self.assertFalse(report['gates']['positive_paired_comparisons'])
        self.assertFalse(report['qualified'])

    def test_negative_controls_must_actually_refuse_or_run_and_fail(self):
        a = self.protocol()['rosters']['negative']
        missing = self.row(a[0], success=False)
        missing.update(status='refused', complete=False, recorded_complete_intervals=0)
        self.assertTrue(study._negative_correct(missing))
        missing['recorded_complete_intervals'] = 1
        self.assertFalse(study._negative_correct(missing))
        jaw = self.row(a[3], success=False)
        self.assertTrue(study._negative_correct(jaw))
        jaw['score']['valid'] = False
        self.assertFalse(study._negative_correct(jaw))
        jaw['score']['valid'] = True
        jaw['score']['strict_v01']['trajectory_complete'] = False
        self.assertFalse(study._negative_correct(jaw))
        jaw['score']['strict_v01']['trajectory_complete'] = True
        jaw.update(status='refused', complete=False, recorded_complete_intervals=0)
        self.assertFalse(study._negative_correct(jaw))

    def test_development_loss_selection_never_falls_back_to_training_loss(self):
        self.assertEqual(study._loss({'training': {'validation': {'action_scale_mse': .2}, 'final_loss': .001}}), .2)
        with self.assertRaisesRegex(ValueError, 'development prediction loss'):
            study._loss({'training': {'final_loss': .001}})

    def test_rollout_identity_cannot_be_swapped_between_groups(self):
        attempt = self.protocol()['rosters']['diagnostic'][0]
        summary = dict(case=deepcopy(attempt['case']), mode='teacher', negative_control=None,
                       status='complete', recorded_complete_intervals=1000, score={'success': True})
        summary['case']['id'] = 'other-scene'
        with self.assertRaisesRegex(ValueError, 'registered case'):
            study._row(self.root, attempt, summary, self.root / 'nonexistent')

    def test_package_does_not_claim_an_unrun_skill_is_qualified(self):
        destination = self.base / 'package'
        result = study.package(self.root, destination)
        self.assertFalse(result['qualified'])
        manifest = study.read(destination / 'manifest.json')
        self.assertEqual(manifest['qualification'], 'research_candidate')
        self.assertFalse(manifest['model_runtime_installation_included'])
        with self.assertRaises(ValueError):
            study.package(self.root, destination)

    def test_registered_selection_and_confirmation_lifecycle_without_native_execution(self):
        def fake_run(root, assets, stage, model_stage=None):
            receipt = self.receipt(stage)
            receipt['model_bindings'] = study._model_bindings(root, stage, model_stage)
            for row in receipt['rows']:
                if stage.startswith('development-'):
                    row['strict_success'] = stage.endswith('corrected') or row['case']['goal_id'] == 0
                elif stage == 'confirmation':
                    row['strict_success'] = (row['controller'] in ('student', 'teacher')
                        or int(row['case']['group_id'].split('-')[-1]) < 12)
                elif stage == 'negative':
                    row['strict_success'] = False
                    if row['negative_control'] != 'open_gripper':
                        row.update(status='refused', complete=False, recorded_complete_intervals=0)
            return receipt

        def fake_export(root, rows, blind, label_source):
            return dict(features=[], targets=[], groups=[], provenance={'blind': blind,
                'source_identity_sha256': 'same'}), [r['id'] for r in rows]

        def fake_fit(dataset, **kwargs):
            self.assertEqual(kwargs['epochs'], 50)
            self.assertEqual(kwargs['batch_size'], 512)
            self.assertEqual(kwargs['learning_rate'], .003)
            self.assertEqual(kwargs['seed'], 17)
            self.assertIsNotNone(kwargs['validation_dataset'])
            return dict(training={'validation': {'action_scale_mse': .1}, 'final_loss': .001})

        with patch.object(study, '_run_stage', side_effect=fake_run), \
                patch.object(study, '_dataset', side_effect=fake_export), \
                patch('dvidia_training.camera_action_export.blind_dataset', side_effect=lambda value: value), \
                patch('dvidia_training.camera_action_export.combine_datasets', side_effect=lambda values: values[0]), \
                patch('dvidia_training.camera_action_model.train', side_effect=fake_fit):
            study.collect(self.root, '/unused/assets')
            study.collect(self.root, '/unused/assets', stage='demonstrations')
            study.train(self.root)
            study.evaluate(self.root, '/unused/assets')
            study.collect(self.root, '/unused/assets', stage='corrections')
            study.train(self.root, stage='corrected')
            study.evaluate(self.root, '/unused/assets', stage='development-corrected')
            study.evaluate(self.root, '/unused/assets', stage='confirmation')
            frozen = study.read(self.root / 'freeze.json')
            self.assertEqual(frozen['selected_stage'], 'corrected')
            study.evaluate(self.root, '/unused/assets', stage='negative')
            report = study.read(self.root / 'report.json')
            self.assertTrue(report['qualified'])
            self.assertTrue(all(report['gates'].values()))
            self.assertEqual(study.inspect(self.root)['next_actions'], [])
            self.assertEqual(len(study._stage(self.root, 'confirmation')['rows']), 256)
            model_path = study._model_path(self.root, 'corrected', 'student')
            model_path.write_bytes(model_path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                study.report_study(self.root)

    def test_refusal_alone_cannot_win_common_admitted_comparison(self):
        rows = self.receipt('confirmation')['rows']
        for row in rows:
            if row['controller'] == 'fixed':
                row.update(status='refused', complete=False, strict_success=False,
                           recorded_complete_intervals=0)
        self.assertTrue(study.paired_bootstrap(rows, 'fixed')['positive'])
        admitted = study.paired_bootstrap(rows, 'fixed', admitted_only=True)
        self.assertEqual(admitted['groups'], 0)
        self.assertIsNone(admitted['lower'])
        self.assertFalse(admitted['positive'])

    def test_policy_load_rejects_model_replaced_after_training(self):
        path = study._model_path(self.root, 'initial', 'student')
        study.write(path, {'fixture': 'trained'})
        receipt = dict(status='complete', models={'full': {'model_sha256': study.file_hash(path)}})
        study._seal_stage(self.root, 'train-initial', receipt)
        self.assertEqual(study._model_document(self.root, 'initial', 'student'), {'fixture': 'trained'})
        path.write_bytes(b'{"fixture":"replacement"}\n')
        with self.assertRaisesRegex(ValueError, 'sealed training receipt'):
            study._model_document(self.root, 'initial', 'student')


if __name__ == '__main__':
    unittest.main()
