"""Lifecycle tests use tiny finite data and mocked attempts, never native physics."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dvidia_training import piper_grip_clock_cli as cli
from dvidia_training import piper_grip_clock_model as model
from dvidia_training import piper_grip_adapter as grip
from dvidia_training import piper_imitation as core
from dvidia_training.piper_imitation_benchmark import read, write, sha, save_episode
from tests.test_piper_feedback_model import base_model


class GripClockLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.root = self.path/'study'
        self.base = base_model()
        self.jaw = grip.train(dict(features=np.zeros((8, grip.FEATURE_COUNT)),
                                  targets=np.zeros(8), groups=['a']*4+['b']*4),
                              base_sha256=core.digest(self.base), max_depth=4, min_leaf=2)
        write(self.path/'base.json', self.base)
        write(self.path/'jaw.json', self.jaw)
        for target, value in (('BASE_SHA', sha(self.path/'base.json')), ('JAW_SHA', sha(self.path/'jaw.json'))):
            context = patch.object(cli.runner, target, value)
            context.start()
            self.addCleanup(context.stop)
        source = {Path(cli.__file__).name: sha(cli.__file__)}
        context = patch.object(cli, 'source_hashes', return_value=source)
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(cli, '_attempt', side_effect=lambda root, case, variant, folder, model_path=None, labels=False: read(folder/'score.json'))
        context.start()
        self.addCleanup(context.stop)
        self.execute = patch.object(cli.runner, 'execute', side_effect=self.fake_execute)
        self.mock_execute = self.execute.start()
        self.addCleanup(self.execute.stop)

    def prepared(self):
        return cli.prepare(self.root, self.path/'base.json', self.path/'jaw.json')

    def fake_execute(self, root, assets, case, variant, folder, *, model_path=None, collect_labels=False):
        folder = Path(folder)
        score = dict(valid=True, success=False,
                     native_recomputed=dict(grasp_seen=False, lift_seen=False,
                                            unexpected_contact_native_ticks=0, retention_lost=False),
                     grip_summary={}, terminal_only={'success': False},
                     post_grip_analysis=dict(valid=True, eligible=False, activated=False, effective=False,
                                             nonactivation_reason='eligibility_not_reached'))
        if folder.exists():
            return read(folder/'score.json')
        write(folder/'score.json', score)
        write(folder/'receipt.json', dict(case_id=case['id'], variant=variant))
        save_episode(folder/'episode.json.gz', dict(
            timing=dict(inference_samples_s=[.001]*1000, control_step_samples_s=[.002]*1000,
                        wall_seconds=2.), platform=dict(system='synthetic-test', machine='no-physics')))
        if collect_labels:
            x = np.zeros((4, model.FEATURE_COUNT))
            cli.runner._save_npz(folder/'samples.npz', dict(
                features=x, blind_features=x.copy(), targets=np.zeros(4),
                weights=np.full(4, 20.), groups=np.asarray([case['id']]*4)))
        return score

    def fitted(self):
        self.prepared()
        cli.collect(self.root, 'unused')
        for name, (blind, depth) in cli.CANDIDATES.items():
            cli.train(self.root, name, blind=blind, max_depth=depth)

    def developed(self):
        self.fitted()
        for name in cli.CANDIDATES:
            cli.development(self.root, 'unused', name)

    def test_roster_has_exact_split_sizes_paired_seeds_and_minimum_distance(self):
        cases = cli.make_cases()
        self.assertEqual(cases, cli.make_cases())
        self.assertEqual({key: len(rows) for key, rows in cases.items()},
                         {'train': 64, 'development': 32, 'confirmation': 128})
        points = []
        for split, rows in cases.items():
            grouped = {}
            for case in rows:
                self.assertEqual(set(case), {'id', 'start_xy', 'goal_xy', 'seed', 'condition'})
                point = tuple(case['start_xy'])
                grouped.setdefault(point, []).append(case)
                self.assertTrue(case['id'].startswith(split+'-xy'))
            for point, group in grouped.items():
                self.assertEqual(len(group), 4)
                for goal in range(2):
                    selected = [c for c in group if c['goal_xy'] == list(cli.primitive.GOALS[goal])]
                    self.assertEqual(len(selected), 2)
                    self.assertEqual(len({c['seed'] for c in selected}), 1)
                points.append(point)
        self.assertEqual(len(set(points)), 56)
        for index, point in enumerate(points):
            for other in [*cli.historical_starts(), *points[:index]]:
                self.assertGreaterEqual(np.linalg.norm(np.asarray(point)-other), .00075-1e-12)

    def test_prepare_copies_exact_bytes_and_refuses_existing_or_wrong_raw_model(self):
        protocol = self.prepared()
        self.assertEqual((self.root/'base-model.json').read_bytes(), (self.path/'base.json').read_bytes())
        self.assertEqual(protocol['analysis']['scheduled_confirmation_attempts'], 512)
        self.assertFalse(protocol['physical_robot_ready'])
        self.assertEqual(protocol['teacher_config']['gate_offset_s'], .32)
        self.assertTrue(protocol['teacher_config']['acquisition_only'])
        self.assertEqual(cli._study(self.root), protocol)
        with self.assertRaises(ValueError):
            self.prepared()
        with patch.object(cli.runner, 'JAW_SHA', '0'*64), self.assertRaises(ValueError):
            cli.prepare(self.path/'wrong', self.path/'base.json', self.path/'jaw.json')

    def test_coordinated_reference_manifest_is_bound_as_separate_architecture(self):
        from dvidia_training.piper_grip_clock import reference_jaw_manifest
        write(self.path/'reference-jaw.json', reference_jaw_manifest(self.base))
        protocol = cli.prepare(self.root, self.path/'base.json', self.path/'reference-jaw.json')
        self.assertEqual(protocol['jaw_mode'], 'coordinated_reference')
        self.assertEqual(protocol['jaw_model_raw_sha256'], sha(self.path/'reference-jaw.json'))
        self.assertEqual(protocol['teacher_config']['gate_offset_s'], .16)
        self.assertIn('no v0.2 jaw adapter', protocol['limitations'][-1])
        self.assertEqual(cli._study(self.root), protocol)
        (self.root/'jaw-model.json').write_text((self.root/'jaw-model.json').read_text()+'\n')
        with self.assertRaises(ValueError):
            cli._study(self.root)

    def test_coordinated_training_uses_the_separate_v2_contract(self):
        from dvidia_training.piper_grip_clock import reference_jaw_manifest
        write(self.path/'reference-jaw.json', reference_jaw_manifest(self.base))
        cli.prepare(self.root, self.path/'base.json', self.path/'reference-jaw.json')
        cli.collect(self.root, 'unused')
        cli.train(self.root, 'full4')
        document = read(self.root/'candidates'/'full4'/'model.json')
        self.assertEqual(document['schema'], 'dvidia.piper-grip-progress-clock/v2')
        self.assertEqual(document['jaw_sha256'], core.digest(reference_jaw_manifest(self.base)))
        self.assertEqual(document['training']['scene_count'], 64)

    def test_source_protocol_and_snapshot_changes_close_the_study(self):
        self.prepared()
        with patch.object(cli, 'source_hashes', return_value={'changed.py': 'a'*64}), self.assertRaises(ValueError):
            cli.collect(self.root, 'unused')
        snapshot = self.root/'sources'/Path(cli.__file__).name
        snapshot.write_bytes(snapshot.read_bytes()+b'\n')
        with self.assertRaises(ValueError):
            cli.collect(self.root, 'unused')
        self.assertEqual(self.mock_execute.call_count, 0)

    def test_every_failed_teacher_prefix_is_fitted_and_training_never_opens_other_splits(self):
        self.prepared()
        manifest = cli.collect(self.root, 'unused')
        self.assertEqual(manifest['scheduled'], 64)
        self.assertEqual(manifest['successes'], 0)
        self.assertEqual(manifest['samples'], 256)
        self.assertEqual(self.mock_execute.call_count, 64)
        for call in self.mock_execute.call_args_list:
            self.assertTrue(call.args[2]['id'].startswith('train-'))
            self.assertEqual(call.args[3], 'teacher')
            self.assertTrue(call.kwargs['collect_labels'])
        receipt = cli.train(self.root, 'full4', max_depth=4)
        document = read(self.root/'candidates'/'full4'/'model.json')
        self.assertEqual(document['training']['sample_count'], 256)
        self.assertEqual(document['training']['scene_count'], 64)
        self.assertEqual(document['training']['weight_sum'], 5120.)
        self.assertEqual(receipt['teacher_manifest_sha256'], sha(self.root/'teacher-manifest.json'))
        with self.assertRaises(ValueError):
            cli.train(self.root, 'full4', max_depth=4)
        self.assertFalse((self.root/'confirmation').exists())

    def test_missing_teacher_or_unplanned_fit_is_rejected(self):
        protocol = self.prepared()
        with self.assertRaises(ValueError):
            cli.train(self.root, 'full4')
        cli.collect(self.root, 'unused')
        for kwargs in (dict(name='another'), dict(name='full4', blind=True),
                       dict(name='full4', max_depth=6), dict(name='full4', min_leaf=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cli.train(self.root, **kwargs)
        sample = self.root/'demonstrations'/protocol['cases']['train'][0]['id']/'samples.npz'
        sample.unlink()
        with self.assertRaises(FileNotFoundError):
            cli.train(self.root, 'full4')

    def test_development_keeps_anchors_and_each_candidate_without_automatic_selection(self):
        self.fitted()
        cli.development(self.root, 'unused', 'full4')
        self.assertEqual(read(self.root/'development'/'fixed.json')['scheduled'], 32)
        self.assertEqual(read(self.root/'development'/'teacher.json')['scheduled'], 32)
        self.assertEqual(read(self.root/'candidates'/'full4'/'development.json')['scheduled'], 32)
        self.assertFalse((self.root/'selected.json').exists())
        self.assertFalse((self.root/'frozen.json').exists())
        with self.assertRaises((FileNotFoundError, ValueError)):
            cli.freeze(self.root, 'full4', 'blind4')

    def test_freeze_requires_all_four_candidates_and_binds_every_preconfirmation_file(self):
        self.developed()
        with self.assertRaises(ValueError):
            cli.freeze(self.root, 'full4', 'blind6')
        frozen = cli.freeze(self.root, 'full4', 'blind4')
        self.assertEqual(frozen['selected_sha256'], sha(self.root/'selected.json'))
        self.assertEqual(frozen['blind_sha256'], sha(self.root/'blind.json'))
        self.assertIn('candidates/full6/development.json', frozen['files'])
        self.assertEqual(cli._frozen(self.root)[1], frozen)
        with self.assertRaises(ValueError):
            cli.collect(self.root, 'unused')
        with self.assertRaises(ValueError):
            cli.train(self.root, 'full6', max_depth=6)
        with self.assertRaises(ValueError):
            cli.development(self.root, 'unused', 'full4')
        write(self.root/'unexpected.json', {'changed': True})
        with self.assertRaises(ValueError):
            cli.confirm(self.root, 'unused')

    def test_confirm_interleaves_all_512_attempts_and_report_is_immutable_and_denominator_explicit(self):
        self.developed()
        cli.freeze(self.root, 'full4', 'blind4')
        self.mock_execute.reset_mock()
        report = cli.confirm(self.root, 'unused')
        self.assertEqual(self.mock_execute.call_count, 512)
        self.assertEqual([call.args[3] for call in self.mock_execute.call_args_list[:8]],
                         list(cli.CONFIRM_VARIANTS)*2)
        self.assertEqual(len(report['rows']), 512)
        self.assertEqual(len(report['start_groups']), 32)
        self.assertFalse(report['advance_to_followup_simulation'])
        self.assertFalse(report['qualified_robot_skill'])
        self.assertEqual(report['variants']['clock']['scheduled'], 128)
        self.assertEqual(report['variants']['clock']['strict_successes'], 0)
        self.assertEqual(report['variants']['clock']['post_grip_effective_fraction'], 0.)
        self.assertEqual(report['variants']['clock']['timing']['full_step']['count'], 128000)
        self.assertEqual(cli.report(self.root), report)
        altered = deepcopy(report)
        altered['qualified_robot_skill'] = True
        (self.root/'report.json').write_text(__import__('json').dumps(altered))
        with self.assertRaises(ValueError):
            cli.report(self.root)

    def test_frozen_selection_names_and_architecture_cannot_drift_from_bound_copies(self):
        self.developed()
        original = cli.freeze(self.root, 'full4', 'blind4')
        for changes in (dict(selected='full6'), dict(blind='blind6'),
                        dict(selected='full6', blind='blind6'), dict(jaw_mode='coordinated_reference'),
                        dict(jaw_model_raw_sha256='0'*64), dict(physical_robot_ready=True),
                        dict(undeclared_field='not allowed')):
            with self.subTest(changes=changes):
                (self.root/'frozen.json').write_text(__import__('json').dumps({**original, **changes}))
                with self.assertRaises(ValueError):
                    cli._frozen(self.root)
        (self.root/'frozen.json').write_text(__import__('json').dumps(original))
        self.assertEqual(cli._frozen(self.root)[1], original)

    def test_paired_statistics_use_32_groups_and_simultaneous_not_independent_case_intervals(self):
        groups = [dict(clock=[1, 1, 1, 1], fixed=[0, 0, 0, 0]) for _ in range(16)]
        groups += [dict(clock=[0, 0, 0, 0], fixed=[0, 0, 0, 0]) for _ in range(16)]
        result = cli._paired(groups, 'clock', 'fixed')
        self.assertEqual(result['mean_difference'], .5)
        self.assertEqual(result['independent_groups'], 32)
        expected_half = cli.T975_DF31*np.std([1.]*16+[0.]*16, ddof=1)/np.sqrt(32)
        np.testing.assert_allclose(result['interval'], [.5-expected_half, .5+expected_half])
        self.assertEqual(result['interval_coverage'], .975)
        self.assertEqual(result['simultaneous_coverage'], .95)
        with self.assertRaises(ValueError):
            cli._paired(groups[:16], 'clock', 'fixed')

    def test_terminal_commands_require_concrete_inputs_and_never_silently_select(self):
        for args in (['prepare', str(self.root)], ['collect', str(self.root)],
                     ['train', str(self.root)], ['freeze', str(self.root)]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                cli.main(args)


if __name__ == '__main__':
    unittest.main()
