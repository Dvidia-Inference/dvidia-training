"""Adversarial follow-up contracts cannot turn leakage into apparent benefit.

The follow-up fixture below is an authored contract, never an executed result.
It exists only to exercise the coordinator's comparison acceptance rules.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import unittest

from dvidia_training.footage_benchmark import _layout_digest
from dvidia_training.footage_pipeline import _canonical
from dvidia_training.improvement import brief, build_dashboard, capture_plan, load_benchmark
from dvidia_training.skill_capsule import scene_sha256


FIXTURE = Path(__file__).resolve().parents[1] / 'benchmarks/sample-count-19047'


def authored_followup(development):
    followup = deepcopy(development)
    protocol = followup['protocol']
    baseline = deepcopy(protocol['subsets'][1])
    original = deepcopy(followup['report']['comparisons'][1])
    protocol['training']['seed'] = development['protocol']['seed']
    protocol['seed'] = 381527
    protocol['subsets'] = []
    followup['report']['comparisons'] = []
    for name, extra in (('baseline4', None), ('generic5', 'general'), ('targeted5', 'targeted')):
        spec = deepcopy(baseline)
        spec['cohort_id'] = name
        if extra:
            spec['train_group_ids'].append('authored-group-' + extra)
            spec['train_episode_ids'].append('authored-episode-' + extra)
        spec['training_group_count'] = len(spec['train_group_ids'])
        protocol['subsets'].append(spec)
        result = deepcopy(original)
        result['cohort_id'] = name
        result['training_group_count'] = spec['training_group_count']
        followup['report']['comparisons'].append(result)
    for scene in protocol['scenes']:
        scene['scene']['object_position'][0] += .0001
        scene['scene_sha256'] = scene_sha256(scene['scene'])
        scene['layout_sha256'] = _layout_digest(scene['scene'])
    hashes = {s['id']: s['scene_sha256'] for s in protocol['scenes']}
    for result in [followup['report']['teacher'], *followup['report']['comparisons']]:
        for row in result['outcomes']:
            row['scene_sha256'] = hashes[row['case_id']]
    plan = {'kind': 'dvidia.improvement-experiment-plan.v1',
            'scope': 'simulation-only', 'physical_robot_ready': False,
            'test_fixture_only': True,
            'historical_protocol_sha256': development['protocol_sha256'],
            'historical_report_sha256': development['report_sha256'],
            'source_dataset_sha256': protocol['source_dataset_sha256'],
            'training_seed': protocol['training']['seed'], 'evaluation_seed': protocol['seed'],
            'evaluation_budget': {'nominal_scenes': 8, 'actuator_stress_scenes': 4},
            'cohorts': [{key: deepcopy(spec[key]) for key in ('cohort_id', 'train_group_ids')}
                        for spec in protocol['subsets']],
            'memberships': [{key: deepcopy(spec[key]) for key in ('cohort_id', 'training_group_count',
                            'train_group_ids', 'train_episode_ids', 'dev_episode_ids', 'test_episode_ids')}
                            for spec in protocol['subsets']]}
    experiment = {'plan': plan, 'plan_sha256': sha256(_canonical(plan) + b'\n').hexdigest()}
    protocol['experiment'] = deepcopy(experiment)
    followup['report']['experiment'] = deepcopy(experiment)
    return followup


class ImprovementBindingTests(unittest.TestCase):
    def setUp(self):
        self.development = load_benchmark(FIXTURE)
        self.followup = authored_followup(self.development)

    def test_synthetic_matched_contract_has_no_invented_gain(self):
        view = build_dashboard(self.development, evaluation=self.followup)
        self.assertEqual(view['conclusion']['status'], 'no_measured_gain')
        self.assertEqual(len(view['comparisons']), 3)
        self.assertEqual(view['comparisons'][2]['paired']['improved'], 0)

    def test_next_investigation_uses_latest_retained_model_evidence(self):
        self.followup['report_sha256'] = 'e' * 64
        baseline = self.followup['report']['comparisons'][0]
        for row in baseline['outcomes']:
            if row['group'] == 'nominal':
                row.update(student_success=True, qualified=True, control_success=False)
        view = build_dashboard(self.development, evaluation=self.followup)
        self.assertEqual(view['baseline']['id'], 'baseline4')
        self.assertEqual([h['category'] for h in view['hypotheses']], ['shared'])
        self.assertEqual([h['category'] for h in view['previous_hypotheses']], ['candidate', 'shared'])
        self.assertIn('Diagnosis report: ' + 'e' * 64, brief(view, 'shared-investigation'))
        with self.assertRaises(ValueError):
            capture_plan(view)

    def test_nominal_gain_cannot_hide_stress_regression(self):
        targeted = self.followup['report']['comparisons'][2]
        failed = next(r for r in targeted['outcomes'] if r['group'] == 'nominal' and not r['qualified'])
        failed.update(student_success=True, qualified=True)
        stress = next(r for r in targeted['outcomes'] if r['group'] == 'actuator_stress' and r['qualified'])
        stress.update(student_success=False, qualified=False)
        view = build_dashboard(self.development, evaluation=self.followup)
        self.assertEqual(view['comparisons'][2]['paired']['improved'], 1)
        self.assertEqual(view['comparisons'][2]['paired_stress']['regressed'], 1)
        self.assertEqual(view['conclusion']['status'], 'regression_observed')

    def test_latest_success_does_not_repeat_an_old_collection_request(self):
        baseline = self.followup['report']['comparisons'][0]
        for row in baseline['outcomes']:
            row.update(student_success=True, qualified=True, control_success=False)
        view = build_dashboard(self.development, evaluation=self.followup)
        self.assertEqual(view['hypotheses'], [])
        self.assertIsNone(view['next_hypothesis_id'])
        self.assertTrue(view['previous_hypotheses'])

    def test_wrong_source_task_recipe_or_training_seed_is_rejected(self):
        for field in ('source_dataset_sha256', 'task', 'resolution', 'seed'):
            with self.subTest(field=field):
                followup = deepcopy(self.followup)
                protocol = followup['protocol']
                if field == 'source_dataset_sha256':
                    protocol[field] = 'f' * 64
                elif field == 'task':
                    protocol['input_files_sha256']['task.skill.json'] = 'f' * 64
                else:
                    protocol['training'][field] += 1
                with self.assertRaises(ValueError):
                    build_dashboard(self.development, evaluation=followup)

    def test_experiment_must_bind_to_both_historical_receipts(self):
        for field in ('historical_report_sha256', 'historical_protocol_sha256'):
            with self.subTest(field=field):
                followup = deepcopy(self.followup)
                experiment = followup['protocol']['experiment']
                experiment['plan'][field] = 'a' * 64
                experiment['plan_sha256'] = sha256(_canonical(experiment['plan']) + b'\n').hexdigest()
                followup['report']['experiment'] = deepcopy(experiment)
                with self.assertRaises(ValueError):
                    build_dashboard(self.development, evaluation=followup)

    def test_changed_baseline_or_shared_addition_is_rejected(self):
        followup = deepcopy(self.followup)
        followup['protocol']['subsets'][0]['train_episode_ids'].pop()
        with self.assertRaises(ValueError):
            build_dashboard(self.development, evaluation=followup)
        followup = deepcopy(self.followup)
        followup['protocol']['subsets'][2]['train_group_ids'] = deepcopy(followup['protocol']['subsets'][1]['train_group_ids'])
        with self.assertRaises(ValueError):
            build_dashboard(self.development, evaluation=followup)

    def test_known_source_or_development_layout_cannot_be_confirmation(self):
        for source in (self.development['protocol']['scenes'][0],
                       self.followup['protocol']['source_scene_exclusion']['bindings'][0]):
            with self.subTest(source=source.get('id', source.get('episode_id'))):
                followup = deepcopy(self.followup)
                followup['protocol']['scenes'][0]['layout_sha256'] = source['layout_sha256']
                with self.assertRaises(ValueError):
                    build_dashboard(self.development, evaluation=followup)

    def test_held_out_episode_cannot_be_added_to_training(self):
        for split in ('dev_episode_ids', 'test_episode_ids'):
            with self.subTest(split=split):
                followup = deepcopy(self.followup)
                targeted = followup['protocol']['subsets'][2]
                targeted['train_episode_ids'][-1] = targeted[split][0]
                with self.assertRaises(ValueError):
                    build_dashboard(self.development, evaluation=followup)

    def test_changed_simulation_runtime_cannot_be_the_same_comparison(self):
        self.followup['protocol']['runtime']['files_sha256']['arm_env.py'] = '0' * 64
        with self.assertRaises(ValueError):
            build_dashboard(self.development, evaluation=self.followup)

    def test_executed_additions_cannot_swap_their_frozen_selection_labels(self):
        for rows in (self.followup['protocol']['subsets'], self.followup['report']['comparisons']):
            for row in rows:
                if row['cohort_id'] == 'generic5':
                    row['cohort_id'] = 'targeted5'
                elif row['cohort_id'] == 'targeted5':
                    row['cohort_id'] = 'generic5'
        with self.assertRaisesRegex(ValueError, 'frozen selection'):
            build_dashboard(self.development, evaluation=self.followup)

    def test_prepared_source_must_match_original_source_binding(self):
        self.followup['protocol']['input_files_sha256']['source/dataset.json'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'prepared source bindings'):
            build_dashboard(self.development, evaluation=self.followup)

    def test_seed_or_scene_budget_cannot_differ_from_hashed_plan(self):
        for field in ('training_seed', 'evaluation_seed', 'evaluation_budget'):
            with self.subTest(field=field):
                followup = deepcopy(self.followup)
                experiment = followup['protocol']['experiment']
                if field == 'evaluation_budget':
                    experiment['plan'][field]['nominal_scenes'] += 1
                else:
                    experiment['plan'][field] += 1
                experiment['plan_sha256'] = sha256(_canonical(experiment['plan']) + b'\n').hexdigest()
                followup['report']['experiment'] = deepcopy(experiment)
                with self.assertRaises(ValueError):
                    build_dashboard(self.development, evaluation=followup)

    def test_frozen_memberships_cannot_describe_different_episodes(self):
        experiment = self.followup['protocol']['experiment']
        experiment['plan']['memberships'][2]['train_episode_ids'][-1] = 'wrong-episode'
        experiment['plan_sha256'] = sha256(_canonical(experiment['plan']) + b'\n').hexdigest()
        self.followup['report']['experiment'] = deepcopy(experiment)
        with self.assertRaisesRegex(ValueError, 'episode memberships'):
            build_dashboard(self.development, evaluation=self.followup)


if __name__ == '__main__':
    unittest.main()
