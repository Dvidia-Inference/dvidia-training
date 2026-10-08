"""Independent group selection, frozen experiment plans and unseen scene bounds."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import footage_benchmark as benchmark


def receipt_group_id(episode_ids):
    """The prepared receipt names a connected group by its sorted episode IDs."""
    encoded = json.dumps(sorted(episode_ids), separators=(',', ':')).encode()
    return 'group-' + sha256(encoded).hexdigest()[:24]


def grouped_dataset():
    """Metadata fixture: several recordings have more than one video episode."""
    episodes, groups = [], []
    for split, sizes in (('train', [2, 1, 3, 1, 2, 1, 2]),
                         ('dev', [2, 1]), ('test', [2, 1])):
        for index, size in enumerate(sizes):
            group_id = f'{split}-recording-{index}'
            ids = []
            for clip in range(size):
                identifier = f'{group_id}-clip-{clip}'
                ids.append(identifier)
                episodes.append({
                    'id': identifier, 'group_id': group_id, 'split': split,
                    'media_sha256': sha256(identifier.encode()).hexdigest(),
                    'source_recording_id': group_id,
                    'session_id': f'{group_id}-session',
                    'shoe_pair_id': f'{group_id}-pair',
                })
            group_id = receipt_group_id(ids)
            for episode in episodes[-size:]:
                episode['group_id'] = group_id
            groups.append({'id': group_id, 'episode_ids': ids, 'split': split})
    return {'episodes': episodes, 'groups': groups,
            'splits': {split: [row['id'] for row in episodes if row['split'] == split]
                       for split in ('train', 'dev', 'test')}}


class CountTests(unittest.TestCase):
    def test_counts_are_sorted_and_defaults_fit_available_training_groups(self):
        self.assertEqual(benchmark.normalize_counts([7, 2, 4], 7), (2, 4, 7))
        self.assertEqual(benchmark.normalize_counts([1], 7), (1,))
        self.assertEqual(benchmark.normalize_counts(None, 7), (2, 4, 7))
        self.assertEqual(benchmark.normalize_counts(None, 3), (2,))
        self.assertEqual(benchmark.normalize_counts(None, 1), (1,))

    def test_invalid_counts_cannot_silently_change_the_experiment(self):
        for counts in ([], [2, 2], [True], [False, 2], [1.5], ['2'],
                       [0], [-1], [8]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                benchmark.normalize_counts(counts, 7)
        for available in (0, True, 65):
            with self.subTest(available=available), self.assertRaises(ValueError):
                benchmark.normalize_counts(None, available)
        with self.assertRaises(ValueError):
            benchmark.normalize_counts(list(range(1, 10)), 9)


class GroupSelectionTests(unittest.TestCase):
    def test_nested_subsets_keep_recordings_whole_and_evaluation_fixed(self):
        dataset = grouped_dataset()
        original = deepcopy(dataset)
        subsets = benchmark.nested_subsets(dataset, [2, 4, 7], seed=391)
        self.assertEqual(dataset, original)
        self.assertEqual(subsets, benchmark.nested_subsets(dataset, [2, 4, 7], seed=391))
        self.assertEqual([row['training_group_count'] for row in subsets], [2, 4, 7])
        dev = {e['id'] for e in dataset['episodes'] if e['split'] == 'dev'}
        test = {e['id'] for e in dataset['episodes'] if e['split'] == 'test'}
        prior = set()
        for row in subsets:
            selected = set(row['train_group_ids'])
            self.assertEqual(len(selected), row['training_group_count'])
            self.assertTrue(prior <= selected)
            expected = {e['id'] for e in dataset['episodes']
                        if e['split'] == 'train' and e['group_id'] in selected}
            self.assertEqual(set(row['train_episode_ids']), expected)
            self.assertEqual(set(row['dev_episode_ids']), dev)
            self.assertEqual(set(row['test_episode_ids']), test)
            self.assertTrue(expected.isdisjoint(dev | test))
            self.assertTrue(dev.isdisjoint(test))
            prior = selected
        self.assertEqual(set(subsets[-1]['train_episode_ids']),
                         {e['id'] for e in dataset['episodes'] if e['split'] == 'train'})

    def test_cross_split_source_identity_is_rejected_before_subsetting(self):
        for key in ('media_sha256', 'source_recording_id', 'session_id', 'shoe_pair_id'):
            dataset = grouped_dataset()
            heldout = next(e for e in dataset['episodes'] if e['split'] == 'test')
            affected = [e for e in dataset['episodes']
                        if e['group_id'] in (dataset['episodes'][0]['group_id'], heldout['group_id'])]
            dataset['episodes'][0][key] = heldout[key]
            merged = receipt_group_id([e['id'] for e in affected])
            for episode in affected:
                episode['group_id'] = merged
            with self.subTest(identity=key), self.assertRaisesRegex(ValueError, 'cross.*splits'):
                benchmark.nested_subsets(dataset, [2], seed=391)

    def test_declared_groups_cannot_split_connected_recordings(self):
        dataset = grouped_dataset()
        dataset['episodes'][0]['group_id'] = 'pretend-independent-recording'
        with self.assertRaisesRegex(ValueError, 'connected recording lineage'):
            benchmark.nested_subsets(dataset, [2], seed=391)

    def test_absent_development_or_test_cannot_become_a_training_curve(self):
        for missing in ('dev', 'test'):
            dataset = grouped_dataset()
            dataset['episodes'] = [e for e in dataset['episodes'] if e['split'] != missing]
            dataset['groups'] = [g for g in dataset['groups'] if g['split'] != missing]
            with self.subTest(split=missing), self.assertRaises(ValueError):
                benchmark.nested_subsets(dataset, [2], seed=391)

    def test_split_receipt_cannot_disagree_with_episode_assignments(self):
        dataset = grouped_dataset()
        dataset['splits']['test'] = dataset['splits']['dev']
        with self.assertRaisesRegex(ValueError, 'split metadata'):
            benchmark.nested_subsets(dataset, [2], seed=391)


class QualificationSummaryTests(unittest.TestCase):
    def test_control_success_and_errors_remain_visible_without_qualification(self):
        rows = [
            {'group': 'nominal', 'status': 'completed', 'student_success': True,
             'control_success': False, 'qualified': True,
             'student_episode_wall_seconds': 0.2, 'student_simulated_seconds': 10.,
             'student_final_target_error_m': 0.002,
             'control_episode_wall_seconds': 0.3, 'control_simulated_seconds': 18.,
             'control_final_target_error_m': 0.12},
            {'group': 'nominal', 'status': 'completed', 'student_success': True,
             'control_success': True, 'qualified': False},
            {'group': 'actuator_stress', 'status': 'error', 'qualified': False},
            {'group': 'actuator_stress', 'status': 'completed', 'student_success': False,
             'control_success': False, 'qualified': False},
        ]
        summary = benchmark._aggregate(rows)
        self.assertEqual(summary['planned_scenes'], 4)
        self.assertEqual(summary['student_successes'], 2)
        self.assertEqual(summary['qualified_scenes'], 1)
        self.assertEqual(summary['control_successes'], 1)
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['groups']['nominal']['planned_scenes'], 2)
        self.assertEqual(summary['groups']['nominal']['qualified_scenes'], 1)
        self.assertEqual(summary['groups']['actuator_stress']['planned_scenes'], 2)
        self.assertEqual(summary['groups']['actuator_stress']['qualified_scenes'], 0)
        self.assertEqual(summary['groups']['actuator_stress']['errors'], 1)
        self.assertAlmostEqual(summary['student_episode_wall_seconds'], 0.2)
        self.assertAlmostEqual(summary['control_episode_wall_seconds'], 0.3)
        self.assertEqual(summary['student_measurements']['measured_scenes'], 1)
        self.assertEqual(summary['groups']['actuator_stress']['student_measurements']['measured_scenes'], 0)

    def test_throughput_uses_total_simulated_time_over_total_wall_time(self):
        rows = [
            {'student_episode_wall_seconds': 2., 'student_simulated_seconds': 10.,
             'student_final_target_error_m': 0.01,
             'control_episode_wall_seconds': 1., 'control_simulated_seconds': 100.,
             'control_final_target_error_m': 0.05},
            {'student_episode_wall_seconds': 8., 'student_simulated_seconds': 8.,
             'student_final_target_error_m': 0.03},
            {'status': 'error'},
        ]
        summary = benchmark._measurement_summary(rows, 'student')
        self.assertEqual(summary['measured_scenes'], 2)
        self.assertEqual(summary['total_simulated_seconds'], 18.)
        self.assertEqual(summary['total_episode_wall_seconds'], 10.)
        self.assertAlmostEqual(summary['simulated_seconds_per_wall_second'], 1.8)
        self.assertNotAlmostEqual(summary['simulated_seconds_per_wall_second'], (5. + 1.) / 2)
        self.assertAlmostEqual(summary['final_target_error_mean_m'], 0.02)
        self.assertAlmostEqual(summary['final_target_error_max_m'], 0.03)
        self.assertEqual(benchmark._measurement_summary(rows, 'control')['measured_scenes'], 1)
        unmeasured = benchmark._measurement_summary([{'status': 'error'}], 'student')
        self.assertEqual(unmeasured['measured_scenes'], 0)
        self.assertIsNone(unmeasured['simulated_seconds_per_wall_second'])
        self.assertIsNone(unmeasured['final_target_error_mean_m'])

    def test_missing_or_nonfinite_target_errors_do_not_discard_timed_failures(self):
        distances = [0.01, None, True, False, float('nan'), float('inf'), 0.03, 0]
        rows = [{
            'group': 'nominal', 'status': 'error' if index in {1, 2, 3, 4, 5} else 'completed',
            'qualified': False, 'student_episode_wall_seconds': float(index + 1),
            'student_simulated_seconds': float(10 * (index + 1)),
            'student_final_target_error_m': distance,
        } for index, distance in enumerate(distances)]
        measurements = benchmark._measurement_summary(rows, 'student')
        self.assertEqual(measurements['timed_scenes'], 8)
        self.assertEqual(measurements['measured_scenes'], 8)
        self.assertEqual(measurements['measured_error_scenes'], 3)
        self.assertEqual(measurements['total_episode_wall_seconds'], 36.)
        self.assertEqual(measurements['total_simulated_seconds'], 360.)
        self.assertEqual(measurements['simulated_seconds_per_wall_second'], 10.)
        self.assertAlmostEqual(measurements['final_target_error_mean_m'], 0.04 / 3)
        self.assertEqual(measurements['final_target_error_max_m'], 0.03)
        summary = benchmark._aggregate(rows)
        self.assertEqual(summary['planned_scenes'], 8)
        self.assertEqual(summary['errors'], 5)
        self.assertEqual(summary['groups']['nominal']['planned_scenes'], 8)
        self.assertEqual(summary['groups']['nominal']['errors'], 5)
        self.assertEqual(summary['student_measurements'], measurements)


class FrozenProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / 'new-experiment'
        self.protocol = {'schema_version': 1, 'seed': 391,
                         'candidate_sha256': 'a' * 64,
                         'scenes': [{'id': 'unseen-scene', 'mass': 0.04}]}

    def test_frozen_plan_verifies_and_refuses_replacement(self):
        benchmark.freeze_protocol(self.output, self.protocol)
        original = {name: (self.output / name).read_bytes()
                    for name in ('protocol.json', 'protocol.sha256')}
        self.assertEqual(json.loads(original['protocol.json']), self.protocol)
        benchmark.verify_protocol(self.output, self.protocol)
        with self.assertRaises((ValueError, FileExistsError)):
            benchmark.freeze_protocol(self.output, {**self.protocol, 'seed': 999})
        self.assertEqual({name: (self.output / name).read_bytes() for name in original}, original)

    def test_changed_plan_json_hash_or_expected_protocol_is_rejected(self):
        benchmark.freeze_protocol(self.output, self.protocol)
        expected = deepcopy(self.protocol)
        expected['scenes'][0]['mass'] = 0.08
        with self.assertRaises(ValueError):
            benchmark.verify_protocol(self.output, expected)
        for name in ('protocol.json', 'protocol.sha256'):
            path = self.output / name
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            with self.subTest(file=name), self.assertRaises(ValueError):
                benchmark.verify_protocol(self.output, self.protocol)
            path.write_bytes(original)
        benchmark.verify_protocol(self.output, self.protocol)

    def test_matching_replacement_json_and_hash_do_not_replace_the_expected_plan(self):
        benchmark.freeze_protocol(self.output, self.protocol)
        replacement = {**self.protocol, 'seed': 999}
        raw = json.dumps(replacement, sort_keys=True, separators=(',', ':')).encode() + b'\n'
        (self.output / 'protocol.json').write_bytes(raw)
        (self.output / 'protocol.sha256').write_text(sha256(raw).hexdigest() + '\n')
        with self.assertRaises(ValueError):
            benchmark.verify_protocol(self.output, self.protocol)


@unittest.skipUnless(importlib.util.find_spec('mujoco') is not None,
                     'Scene generation needs the optional native arm engine.')
class SceneProtocolTests(unittest.TestCase):
    def test_source_evidence_total_limit_uses_retained_bytes_after_small_stat(self):
        payloads = [json.dumps({'recording': index, 'padding': 'x' * 48}).encode()
                    for index in range(2)]
        file_limit = max(map(len, payloads)) + 1
        total_limit = sum(map(len, payloads)) - 1
        self.assertLess(file_limit, total_limit)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            evidence_paths = {directory / f'recording-{index}.json'
                              for index in range(len(payloads))}
            for path, raw in zip(sorted(evidence_paths), payloads):
                path.write_bytes(raw)
            original_stat = Path.stat

            def stale_small_stat(path, *args, **kwargs):
                result = original_stat(path, *args, **kwargs)
                if path in evidence_paths:
                    fields = list(result)
                    fields[6] = 1  # Preserve real file mode, but simulate pre-read growth.
                    return os.stat_result(fields)
                return result

            with patch.object(Path, 'stat', stale_small_stat), \
                    patch.object(benchmark, 'EVIDENCE_FILE_MAX_BYTES', file_limit), \
                    patch.object(benchmark, 'EVIDENCE_TOTAL_MAX_BYTES', total_limit):
                self.assertLess(sum(path.stat().st_size for path in evidence_paths), total_limit)
                with self.assertRaisesRegex(ValueError, 'Native evidence exceeds'):
                    benchmark._source_scenes({'episodes': []}, {}, directory)

    def test_scene_protocol_is_deterministic_varied_and_within_supported_bounds(self):
        from dvidia_training.arm_runner import validate_scene
        rows = benchmark.make_scenes(804213, nominal=8, stress=4)
        self.assertEqual(rows, benchmark.make_scenes(804213, nominal=8, stress=4))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r['id'] for r in rows}), 12)
        self.assertEqual(len({r['scene_sha256'] for r in rows}), 12)
        self.assertEqual(len({r['layout_sha256'] for r in rows}), 12)
        self.assertEqual(sum(r['group'] == 'nominal' for r in rows), 8)
        self.assertEqual(sum(r['group'] == 'actuator_stress' for r in rows), 4)
        for row in rows:
            scene = validate_scene(row['scene'])
            self.assertEqual(scene, row['scene'])
            for key in ('scene_sha256', 'layout_sha256'):
                self.assertRegex(row[key], r'^[a-f0-9]{64}$')
        for field in ('object_position', 'target_position', 'object_size',
                      'object_mass', 'object_friction'):
            values = {json.dumps(r['scene'][field]) for r in rows}
            self.assertGreater(len(values), 1, field)
        self.assertNotEqual(rows, benchmark.make_scenes(804214, nominal=8, stress=4))

    def test_source_layout_is_excluded_even_when_materials_or_seed_change(self):
        original = benchmark.make_scenes(804213, nominal=8, stress=4)
        source = deepcopy(original[0]['scene'])
        source['seed'] += 10000
        source['object_mass'] = 0.08 if source['object_mass'] != 0.08 else 0.02
        fresh = benchmark.make_scenes(804213, nominal=8, stress=4, source_scenes=[source])
        self.assertNotIn(original[0]['layout_sha256'], {r['layout_sha256'] for r in fresh})
        self.assertEqual(len(fresh), 12)
        self.assertEqual(len({r['layout_sha256'] for r in fresh}), 12)

    def test_source_scene_uses_hashed_bytes_when_the_file_changes_before_parsing(self):
        from dvidia_training import footage_train
        expected = {'object_position': [.42, -.04, .31], 'target_position': [.54, .10, .31],
                    'object_size': [.04, .04, .04], 'object_mass': .04, 'object_friction': .8,
                    'jaw_force_limit': 15., 'joint_torque_limit': 40., 'seed': 17}
        evidence = {'kind': 'dvidia.native-telemetry-evidence', 'schema_version': 1,
                    'scope': 'simulation-only', 'seed': 17,
                    'final_info': {'config': {
                        'object_position': expected['object_position'],
                        'target_position': expected['target_position'],
                        'object_half_size': [.02, .02, .02], 'object_mass': .04,
                        'object_friction': .8, 'jaw_force_limit': 15.,
                        'joint_torque_limit': 40., 'scene_jitter': 0., 'table_height': .29}}}
        original = json.dumps(evidence, sort_keys=True).encode()
        digest = sha256(original).hexdigest()
        replacement = deepcopy(evidence)
        replacement['final_info']['config']['object_mass'] = .08
        replacement['final_info']['config']['object_position'] = [.35, .08, .31]
        changed = json.dumps(replacement, sort_keys=True).encode()
        dataset = {'episodes': [{'id': 'original-source', 'split': 'train'}]}
        sidecars = {'original-source': {'provenance': {
            'source_kind': 'native-simulation-telemetry',
            'validation': {'evidence_sha256': digest}}}}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            path = directory / 'original-source.json'
            path.write_bytes(original)
            parser = footage_train._json

            def mutate_then_parse(raw, **kwargs):
                path.write_bytes(changed)
                return parser(raw, **kwargs)

            with patch.object(footage_train, '_json', side_effect=mutate_then_parse) as parsing:
                scenes, receipt = benchmark._source_scenes(dataset, sidecars, directory)
            self.assertEqual(parsing.call_count, 1)
            self.assertEqual(path.read_bytes(), changed)
        self.assertEqual(scenes, [expected])
        self.assertEqual(receipt['bindings'][0]['evidence_sha256'], digest)
        self.assertEqual(receipt['bindings'][0]['scene'], expected)


if __name__ == '__main__':
    unittest.main()
