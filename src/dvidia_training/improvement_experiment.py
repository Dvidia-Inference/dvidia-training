"""A frozen, local data-selection pilot for the simplified native placement head.

The historical failure is development evidence. Extra source groups contain
aligned simulator actions, not recovered human-video actions. Selecting a nearby
layout is a hypothesis; this module never promises an improvement.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from . import footage_benchmark as benchmark_module
from .footage_pipeline import _canonical, _fresh

HISTORICAL_SEED = 19047
DEFAULT_SEED = 381527
DIAGNOSTIC_CASE = 'scene-04'
KIND = 'dvidia.improvement-experiment-plan.v1'


def _hash(value):
    return sha256(_canonical(value)).hexdigest()


def _layout_distance(scene, reference):
    """Euclidean distance over object+target XYZ; not a dynamics distance."""
    return math.sqrt(sum((scene[key][i] - reference[key][i]) ** 2
                         for key in ('object_position', 'target_position') for i in range(3)))


def build_plan(dataset, prior_protocol, prior_report, source_bindings, *, seed=DEFAULT_SEED,
               prior_protocol_sha256=None, prior_report_sha256=None):
    """Choose memberships using known development evidence and TRAIN metadata.

    No candidate results or new evaluation scenes enter this selection. All
    source bindings must be produced by the verified native-evidence loader.
    """
    benchmark_module._integer(seed, 0, 999900, 'New evaluation seed')
    if any(type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
           for value in (prior_protocol_sha256, prior_report_sha256)):
        raise ValueError('Historical artifact SHA256 bindings are required.')
    if seed == HISTORICAL_SEED:
        raise ValueError('Use a new evaluation seed; historical scenes are development evidence.')
    if (prior_protocol.get('kind') != 'dvidia.footage-sample-count-benchmark'
            or prior_protocol.get('seed') != HISTORICAL_SEED
            or prior_protocol.get('scope') != 'simulation-only'):
        raise ValueError('Expected the historical 19047 simulation protocol.')
    if (prior_report.get('kind') != 'dvidia.footage-sample-count-result'
            or prior_report.get('scope') != 'simulation-only'
            or prior_report.get('protocol_sha256') != prior_protocol_sha256):
        raise ValueError('Historical report must bind the supplied protocol bytes.')
    expected = benchmark_module.nested_subsets(dataset, [4], HISTORICAL_SEED)[0]
    historical = [row for row in prior_protocol['subsets'] if row['training_group_count'] == 4]
    if len(historical) != 1 or any(historical[0][key] != expected[key] for key in (
            'train_group_ids', 'train_episode_ids', 'dev_episode_ids', 'test_episode_ids')):
        raise ValueError('Historical baseline membership differs from the prepared dataset.')
    diagnostic_cases = [case for case in prior_protocol['scenes'] if case['id'] == DIAGNOSTIC_CASE]
    students = [row for row in prior_report['comparisons'] if row['training_group_count'] == 4]
    if len(diagnostic_cases) != 1 or diagnostic_cases[0]['group'] != 'nominal' or len(students) != 1:
        raise ValueError('Expected one known nominal development case and four-group candidate.')
    case = diagnostic_cases[0]
    failed = [row for row in students[0]['outcomes'] if row['case_id'] == DIAGNOSTIC_CASE]
    teachers = [row for row in prior_report['teacher']['outcomes'] if row['case_id'] == DIAGNOSTIC_CASE]
    if (len(failed) != 1 or len(teachers) != 1 or failed[0].get('status') != 'completed'
            or failed[0].get('student_success') is not False or teachers[0].get('success') is not True
            or failed[0].get('scene_sha256') != case['scene_sha256']
            or teachers[0].get('scene_sha256') != case['scene_sha256']):
        raise ValueError('The diagnosis needs a completed student failure and paired teacher success.')
    bindings = {row['episode_id']: row for row in source_bindings}
    if len(bindings) != len(source_bindings) or set(bindings) != {row['id'] for row in dataset['episodes']}:
        raise ValueError('Native scene evidence must bind every unique source episode.')
    groups = {}
    for episode in dataset['episodes']:
        if bindings[episode['id']]['split'] != episode['split']:
            raise ValueError('Scene evidence and prepared splits disagree.')
        if episode['split'] == 'train':
            groups.setdefault(episode['group_id'], []).append(episode['id'])
    baseline = expected['train_group_ids']
    reserve = sorted(set(groups) - set(baseline))
    if len(reserve) < 2:
        raise ValueError('Need two distinct unused TRAIN groups for the matched additions.')
    ranking = []
    for group in reserve:
        distances = [{'episode_id': eid, 'layout_distance_m': _layout_distance(bindings[eid]['scene'], case['scene']),
                      'layout_sha256': bindings[eid]['layout_sha256']} for eid in groups[group]]
        ranking.append({'group_id': group, 'distance_m': min(row['layout_distance_m'] for row in distances),
                        'episodes': distances, 'seeded_rank_sha256': _hash([seed, 'generic', group])})
    ranking.sort(key=lambda row: (row['distance_m'], row['group_id']))
    targeted = ranking[0]['group_id']
    generic = min((row for row in ranking if row['group_id'] != targeted),
                  key=lambda row: (row['seeded_rank_sha256'], row['group_id']))['group_id']
    cohorts = [{'cohort_id': 'baseline4', 'train_group_ids': list(baseline)},
               {'cohort_id': 'generic5', 'train_group_ids': [*baseline, generic]},
               {'cohort_id': 'targeted5', 'train_group_ids': [*baseline, targeted]}]
    specs = benchmark_module.explicit_subsets(dataset, cohorts)
    return {'kind': KIND, 'scope': 'simulation-only', 'physical_robot_ready': False,
            'historical_protocol_sha256': prior_protocol_sha256,
            'historical_report_sha256': prior_report_sha256,
            'source_dataset_sha256': prior_protocol['source_dataset_sha256'],
            'development_diagnosis': {'case_id': DIAGNOSTIC_CASE, 'scene': deepcopy(case['scene']),
                'scene_sha256': case['scene_sha256'], 'status': 'known_development_failure',
                'student_reason': failed[0].get('student_reason'), 'teacher_success': True,
                'hypothesis': 'A nearby unused training layout may improve the learned movement head; distance is a selection heuristic, not evidence of cause.'},
            'selection': {'rule': 'Minimum object+target XYZ Euclidean distance among unused TRAIN groups; ties by group ID.',
                'generic_rule': 'Seeded SHA256 order among the remaining unused TRAIN groups, excluding the targeted choice.',
                'seed': seed, 'reserve_ranking': ranking, 'targeted_group_id': targeted, 'generic_group_id': generic,
                'group_budget_per_addition': 1, 'row_or_duration_matched': False,
                'reason': 'Whole-source group counts are matched; actual action rows and video duration are reported separately.'},
            'cohorts': cohorts, 'memberships': specs,
            'training_seed': HISTORICAL_SEED, 'evaluation_seed': seed,
            'excluded_known_development_scenes': deepcopy(prior_protocol['scenes']),
            'recipe': {'movement_centers': 8, 'selection': 'Existing fixed development sources choose ridge regularization.',
                'control': 'Same recorded arm targets with jaws open.',
                'freeze': 'Membership and selection before fitting; every candidate before evaluation.',
                'evaluation': 'Same fresh nominal/stress scenes for every candidate; all source and known development layouts excluded.'},
            'limitations': ['One selected composition per condition cannot establish a general learning curve.',
                'Historical test recordings already inspected remain regression diagnostics, not fresh confirmation.',
                'A fresh seed excludes declared layouts but does not independently attest prior non-exposure.',
                'Privileged simulator state and authored phase, gripper and recovery logic remain.',
                'No human-video action recovery, new human footage, physical skill, automatic promotion or reward.']}


def prepare_plan(dataset_path, prior_protocol_path, prior_report_path, source_evidence, *, seed=DEFAULT_SEED):
    from .footage_train import _load_dataset
    dataset, dataset_hash, _, sidecars = _load_dataset(Path(dataset_path))
    prior_protocol_path, prior_report_path = Path(prior_protocol_path), Path(prior_report_path)
    # Parse the exact bytes whose hashes enter the selection receipt.
    from .footage_train import _json as parse_json, _read
    protocol_raw = _read(prior_protocol_path, 8 * 1024 * 1024)
    report_raw = _read(prior_report_path, 8 * 1024 * 1024)
    protocol = parse_json(protocol_raw, maximum=8 * 1024 * 1024)
    report = parse_json(report_raw, maximum=8 * 1024 * 1024)
    if dataset_hash != protocol.get('source_dataset_sha256'):
        raise ValueError('Prepared dataset does not match the historical source bytes.')
    _, evidence = benchmark_module._source_scenes(dataset, sidecars, source_evidence)
    if evidence['status'] != 'excluded_hash_bound_native_declarations':
        raise ValueError('Verified native scene evidence is required for this experiment.')
    plan = build_plan(dataset, protocol, report, evidence['bindings'], seed=seed,
                      prior_protocol_sha256=sha256(protocol_raw).hexdigest(),
                      prior_report_sha256=sha256(report_raw).hexdigest())
    for spec in plan['memberships']:
        ids = set(spec['train_episode_ids'])
        spec['training_action_samples'] = sum(len(sidecars[eid]['samples']) for eid in ids)
        spec['training_video_seconds'] = sum(row['duration_seconds'] for row in dataset['episodes'] if row['id'] in ids)
    return plan


def run_experiment(dataset_path, output, *, prior_protocol_path, prior_report_path,
                   source_evidence, task_source=None, seed=DEFAULT_SEED, nominal=12, stress=4):
    started = time.perf_counter()
    benchmark_module._integer(nominal, 0, 24, 'Nominal scene count')
    benchmark_module._integer(stress, 0, 24, 'Stress scene count')
    if not 1 <= nominal + stress <= 24:
        raise ValueError('Use one to 24 total evaluation scenes.')
    sources = benchmark_module._source_hashes()
    plan = prepare_plan(dataset_path, prior_protocol_path, prior_report_path, source_evidence, seed=seed)
    plan['evaluation_budget'] = {'nominal_scenes': nominal, 'actuator_stress_scenes': stress}
    plan['source_files_sha256'] = sources
    output = _fresh(output)
    benchmark_module._write(output / 'experiment-plan.json', plan)
    plan_hash = sha256((output / 'experiment-plan.json').read_bytes()).hexdigest()
    if sources != benchmark_module._source_hashes():
        raise ValueError('Experiment source changed during planning.')
    result = benchmark_module.benchmark(dataset_path, output / 'benchmark',
        cohorts=plan['cohorts'], seed=seed, training_seed=HISTORICAL_SEED, nominal=nominal, stress=stress,
        task_source=task_source, source_evidence=source_evidence,
        excluded_scenes=[row['scene'] for row in plan['excluded_known_development_scenes']],
        experiment={'plan_sha256': plan_hash, 'plan': plan})
    if sha256((output / 'experiment-plan.json').read_bytes()).hexdigest() != plan_hash:
        raise ValueError('Frozen experiment plan changed during execution.')
    comparisons = result['comparisons']
    report = {'kind': 'dvidia.improvement-experiment-result.v1', 'scope': 'simulation-only',
        'experiment_plan_sha256': plan_hash,
        'benchmark_report_sha256': sha256((output / 'benchmark' / 'report.json').read_bytes()).hexdigest(),
        'benchmark_report': 'benchmark/report.json',
        'cohorts': [{'cohort_id': row['cohort_id'], 'training_group_count': row['training_group_count'],
                    'training_action_samples': row['training_action_samples'], 'summary': row['summary'],
                    'paired_with_teacher': row['paired_with_teacher']} for row in comparisons],
        'wall_seconds': time.perf_counter() - started,
        'timing_scope': 'Planning, source checks, fitting, evaluation and artifact output before this summary; excludes process startup and final summary write.',
        'cost': {'external_api_usd': 0.0, 'local_compute_usd': None,
                 'note': 'Local CPU experiment; hardware, electricity and operator costs are not priced.'},
        'automatic_promotion': False, 'physical_robot_ready': False,
        'training_benefit_generalized': None, 'limitations': plan['limitations']}
    benchmark_module._write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--prior-protocol', type=Path, required=True)
    parser.add_argument('--prior-report', type=Path, required=True)
    parser.add_argument('--source-evidence', type=Path, required=True)
    parser.add_argument('--task-source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--nominal', type=int, default=12)
    parser.add_argument('--stress', type=int, default=4)
    args = parser.parse_args()
    options = {'prior_protocol_path': args.prior_protocol, 'prior_report_path': args.prior_report,
               'source_evidence': args.source_evidence, 'seed': args.seed}
    if args.command == 'plan':
        result = prepare_plan(args.dataset, **options)
        root = _fresh(args.output)
        benchmark_module._write(root / 'experiment-plan.json', result)
    else:
        result = run_experiment(args.dataset, args.output, task_source=args.task_source,
                                nominal=args.nominal, stress=args.stress, **options)
    print(json.dumps({'output': str(args.output), 'kind': result['kind'], 'scope': result['scope']}))


if __name__ == '__main__':
    main()
