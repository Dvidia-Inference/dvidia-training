"""Preregistered data-selection logic, without fitting or opening new outcomes."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import footage_benchmark as benchmark
from dvidia_training import improvement_experiment as experiment
from test_footage_benchmark import grouped_dataset


def fixture():
    dataset = grouped_dataset()
    subset = benchmark.nested_subsets(dataset, [4], 19047)[0]
    scene = {'object_position': [.3, 0., .31], 'target_position': [.4, 0., .31]}
    case = {'id': 'scene-04', 'group': 'nominal', 'scene_sha256': 'c' * 64, 'scene': scene}
    protocol = {'kind': 'dvidia.footage-sample-count-benchmark', 'scope': 'simulation-only',
                'seed': 19047, 'source_dataset_sha256': 'd' * 64, 'subsets': [subset], 'scenes': [case]}
    report = {'kind': 'dvidia.footage-sample-count-result', 'scope': 'simulation-only',
              'protocol_sha256': 'a' * 64,
              'comparisons': [{'training_group_count': 4, 'outcomes': [{
                  'case_id': 'scene-04', 'scene_sha256': 'c' * 64, 'status': 'completed',
                  'student_success': False, 'student_reason': 'horizon'}]}],
              'teacher': {'outcomes': [{'case_id': 'scene-04', 'scene_sha256': 'c' * 64, 'success': True}]}}
    bindings = []
    for index, episode in enumerate(dataset['episodes']):
        position = [.3 + index * .001, 0., .31] if episode['split'] == 'train' else [.3, 0., .31]
        bindings.append({'episode_id': episode['id'], 'split': episode['split'],
                         'layout_sha256': sha256(episode['id'].encode()).hexdigest(),
                         'scene': {'object_position': position, 'target_position': [.4, 0., .31]}})
    return dataset, protocol, report, bindings


def plan(values, **kwargs):
    return experiment.build_plan(*values, prior_protocol_sha256='a' * 64,
                                 prior_report_sha256='b' * 64, **kwargs)


class ImprovementSelectionTests(unittest.TestCase):
    def test_explicit_baseline_and_equal_group_additions_preserve_source_splits(self):
        values = fixture()
        original = deepcopy(values)
        result = plan(values)
        self.assertEqual(values, original)
        self.assertEqual(result, plan(values))
        self.assertEqual([row['cohort_id'] for row in result['memberships']], ['baseline4', 'generic5', 'targeted5'])
        self.assertEqual([row['training_group_count'] for row in result['memberships']], [4, 5, 5])
        base, generic, targeted = [set(row['train_group_ids']) for row in result['memberships']]
        self.assertEqual(generic & targeted, base)
        for row in result['memberships']:
            self.assertEqual(row['dev_episode_ids'], values[0]['splits']['dev'])
            self.assertEqual(row['test_episode_ids'], values[0]['splits']['test'])
        self.assertFalse(result['selection']['row_or_duration_matched'])
        self.assertFalse(result['physical_robot_ready'])

    def test_targeted_group_is_nearest_reserve_train_layout_and_generic_is_distinct_seeded(self):
        result = plan(fixture())
        selection = result['selection']
        ranks = selection['reserve_ranking']
        self.assertEqual(selection['targeted_group_id'], min(ranks, key=lambda r: (r['distance_m'], r['group_id']))['group_id'])
        eligible = [row for row in ranks if row['group_id'] != selection['targeted_group_id']]
        expected = min(eligible, key=lambda r: (r['seeded_rank_sha256'], r['group_id']))
        self.assertEqual(selection['generic_group_id'], expected['group_id'])
        self.assertNotEqual(selection['targeted_group_id'], selection['generic_group_id'])

    def test_nearest_test_or_dev_source_cannot_become_a_training_addition(self):
        values = fixture()
        result = plan(values)
        training = {row['group_id'] for row in values[0]['episodes'] if row['split'] == 'train'}
        self.assertTrue({row['group_id'] for row in result['selection']['reserve_ranking']} <= training)

    def test_historical_membership_or_reference_binding_cannot_change_silently(self):
        for mutation in ('membership', 'hash', 'scene'):
            values = fixture()
            if mutation == 'membership':
                values[1]['subsets'][0]['train_group_ids'].reverse()
            elif mutation == 'hash':
                values[2]['protocol_sha256'] = 'f' * 64
            else:
                values[2]['comparisons'][0]['outcomes'][0]['scene_sha256'] = 'f' * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan(values)

    def test_shared_teacher_failure_is_not_the_selected_data_shortage_hypothesis(self):
        values = fixture()
        values[2]['teacher']['outcomes'][0]['success'] = False
        with self.assertRaisesRegex(ValueError, 'paired teacher success'):
            plan(values)

    def test_failed_execution_is_not_reclassified_as_completed_task_failure(self):
        values = fixture()
        values[2]['comparisons'][0]['outcomes'][0]['status'] = 'error'
        with self.assertRaises(ValueError):
            plan(values)

    def test_evidence_must_cover_every_source_once_and_keep_its_split(self):
        for mutation in ('missing', 'duplicate', 'split'):
            values = fixture()
            if mutation == 'missing':
                values[3].pop()
            elif mutation == 'duplicate':
                values[3].append(values[3][0])
            else:
                values[3][0]['split'] = 'test'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plan(values)

    def test_known_evaluation_seed_cannot_be_presented_as_new(self):
        with self.assertRaisesRegex(ValueError, 'new evaluation seed'):
            plan(fixture(), seed=19047)

    def test_history_remains_explicit_development_and_all_old_layouts_are_excluded(self):
        values = fixture()
        result = plan(values)
        self.assertEqual(result['development_diagnosis']['status'], 'known_development_failure')
        self.assertEqual(result['excluded_known_development_scenes'], values[1]['scenes'])
        self.assertEqual(result['training_seed'], 19047)
        self.assertEqual(result['evaluation_seed'], 381527)


class ExperimentDispatchTests(unittest.TestCase):
    def test_plan_is_on_disk_before_dispatch_and_same_seed_exclusions_cohorts_are_forwarded(self):
        frozen = plan(fixture())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / 'pilot'
            def run_benchmark(dataset, destination, **options):
                saved = json.loads((output / 'experiment-plan.json').read_text())
                self.assertEqual(saved['cohorts'], frozen['cohorts'])
                self.assertEqual(options['experiment']['plan'], saved)
                self.assertEqual(options['training_seed'], 19047)
                self.assertEqual(options['seed'], 381527)
                self.assertEqual(options['cohorts'], frozen['cohorts'])
                self.assertEqual(options['excluded_scenes'], [row['scene'] for row in frozen['excluded_known_development_scenes']])
                destination.mkdir()
                result = {'comparisons': [{'cohort_id': 'baseline4', 'training_group_count': 4,
                    'training_action_samples': 100, 'summary': {'qualified_scenes': 0}, 'paired_with_teacher': {}}]}
                benchmark._write(destination / 'report.json', result)
                return result
            with patch.object(experiment, 'prepare_plan', return_value=frozen), \
                    patch.object(benchmark, 'benchmark', side_effect=run_benchmark):
                result = experiment.run_experiment('dataset.json', output,
                    prior_protocol_path='protocol.json', prior_report_path='report.json', source_evidence='evidence')
            self.assertFalse(result['automatic_promotion'])
            self.assertIsNone(result['training_benefit_generalized'])
            self.assertIsNone(result['cost']['local_compute_usd'])
            self.assertTrue((output / 'report.json').is_file())
            self.assertEqual(result['benchmark_report_sha256'], sha256((output / 'benchmark/report.json').read_bytes()).hexdigest())

    def test_invalid_budget_is_rejected_before_planning_or_creating_output(self):
        with patch.object(experiment, 'prepare_plan') as planning:
            with self.assertRaises(ValueError):
                experiment.run_experiment('dataset.json', 'unused', prior_protocol_path='protocol.json',
                    prior_report_path='report.json', source_evidence='evidence', nominal=24, stress=4)
            planning.assert_not_called()


if __name__ == '__main__':
    unittest.main()
