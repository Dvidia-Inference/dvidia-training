"""Turn bounded benchmark evidence into provisional next experiments.

No model call, training, account mutation, or physical qualification occurs here.
The diagnosis rules propose investigations; they do not establish causal truth.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re

FORMAT = 'dvidia.skillspace-improvement.v1'
MAX_BYTES = 8 * 1024 * 1024
MAX_PREPARED_EPISODES = 1024


def _read(path):
    path = Path(path).absolute()
    if (any(part.is_symlink() for part in (path, *path.parents)) or
            not path.is_file() or path.stat().st_size > MAX_BYTES):
        raise ValueError('Expected a bounded regular benchmark JSON file.')
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError('Benchmark JSON exceeds its size limit.')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate benchmark JSON key.')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('Non-finite benchmark JSON number.')

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if type(value) is not dict:
        raise ValueError('Expected a benchmark JSON object.')
    return value, sha256(raw).hexdigest()


def _prepared_episodes(dataset):
    """Bound and type-check metadata before connected-group reconstruction."""
    episodes = dataset.get('episodes')
    if (type(episodes) is not list or not 1 <= len(episodes) <= MAX_PREPARED_EPISODES or
            any(type(row) is not dict for row in episodes)):
        raise ValueError('Expected one to 1024 prepared episode objects.')
    identities = set()
    for row in episodes:
        identity = _identity(row.get('id'))
        if identity in identities:
            raise ValueError('Prepared episode identities must be unique.')
        identities.add(identity)
        if (type(row.get('media_sha256')) is not str or
                not re.fullmatch(r'[a-f0-9]{64}', row['media_sha256']) or
                row.get('split') not in ('train', 'dev', 'test')):
            raise ValueError('Prepared episode media or split is invalid.')
        _identity(row.get('group_id'))
        for key in ('source_recording_id', 'session_id', 'shoe_pair_id'):
            if key not in row or (row[key] is not None and
                    (type(row[key]) is not str or not 1 <= len(row[key]) <= 200)):
                raise ValueError('Prepared episode lineage must be bounded text or null.')


def _validate_experiment_plan(protocol, report):
    """Require executed memberships and budgets to match the frozen selection."""
    experiment = protocol.get('experiment')
    if type(experiment) is not dict or experiment != report.get('experiment'):
        raise ValueError('Follow-up selection provenance is missing or inconsistent.')
    plan = experiment.get('plan')
    from .footage_pipeline import _canonical
    if (type(plan) is not dict or
            sha256(_canonical(plan) + b'\n').hexdigest() != experiment.get('plan_sha256') or
            plan.get('kind') != 'dvidia.improvement-experiment-plan.v1' or
            plan.get('scope') != 'simulation-only' or plan.get('physical_robot_ready') is not False):
        raise ValueError('Expected the supported frozen improvement experiment plan.')
    inputs = protocol.get('input_files_sha256')
    source = protocol.get('source_dataset_sha256')
    if (type(inputs) is not dict or type(source) is not str or
            not re.fullmatch(r'[a-f0-9]{64}', source) or
            inputs.get('source/dataset.json') != source or plan.get('source_dataset_sha256') != source):
        raise ValueError('Follow-up original and prepared source bindings differ.')
    training = protocol.get('training')
    if type(training) is not dict:
        raise ValueError('Follow-up requires a frozen training recipe.')
    for key, actual in (('training_seed', training.get('seed')), ('evaluation_seed', protocol.get('seed'))):
        expected = plan.get(key)
        if type(expected) is not int or not 0 <= expected <= 999999 or type(actual) is not int or actual != expected:
            raise ValueError('Follow-up seeds differ from the frozen plan.')
    budget = plan.get('evaluation_budget')
    if (type(budget) is not dict or set(budget) != {'nominal_scenes', 'actuator_stress_scenes'} or
            any(type(value) is not int or not 0 <= value <= 24 for value in budget.values()) or
            not 1 <= sum(budget.values()) <= 24):
        raise ValueError('Follow-up requires a bounded frozen scene budget.')
    for group in ('nominal', 'actuator_stress'):
        if sum(scene.get('group') == group for scene in protocol['scenes']) != budget[group + '_scenes']:
            raise ValueError('Follow-up scene counts differ from the frozen budget.')

    expected_names = {'baseline4', 'generic5', 'targeted5'}
    indexes = []
    for rows in (protocol.get('subsets'), plan.get('cohorts'), plan.get('memberships')):
        if (type(rows) is not list or len(rows) != 3 or any(type(row) is not dict for row in rows) or
                any(type(row.get('cohort_id')) is not str for row in rows)):
            raise ValueError('Frozen plan requires three distinct cohort memberships.')
        index = {row['cohort_id']: row for row in rows}
        if set(index) != expected_names:
            raise ValueError('Frozen plan cohort identities differ from the executed comparison.')
        indexes.append(index)
    executed, cohorts, memberships = indexes
    for name, spec in executed.items():
        if (set(cohorts[name]) != {'cohort_id', 'train_group_ids'} or
                cohorts[name]['train_group_ids'] != spec.get('train_group_ids')):
            raise ValueError('Executed cohort groups differ from the frozen selection.')
        for key in ('training_group_count', 'train_group_ids', 'train_episode_ids',
                    'dev_episode_ids', 'test_episode_ids'):
            if key not in memberships[name] or memberships[name][key] != spec.get(key):
                raise ValueError('Executed episode memberships differ from the frozen plan.')
    return plan


def _number(value, label):
    if type(value) not in (float, int) or not 0 <= value <= 1e12 or not math.isfinite(value):
        raise ValueError(f'Invalid {label}.')
    return value


def _identity(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('Invalid benchmark identifier.')
    return value


def _rows(rows, scenes, *, teacher=False):
    if type(rows) is not list or len(rows) != len(scenes):
        raise ValueError('Every planned scene must retain an outcome, including errors.')
    seen = set()
    for row in rows:
        if type(row) is not dict or row.get('case_id') not in scenes or row['case_id'] in seen:
            raise ValueError('Unknown or repeated benchmark scene outcome.')
        seen.add(row['case_id'])
        scene = scenes[row['case_id']]
        if row.get('scene_sha256') != scene['scene_sha256'] or row.get('group') != scene['group']:
            raise ValueError('Outcome no longer matches its frozen scene.')
        if row.get('status') not in ('completed', 'error'):
            raise ValueError('Unknown benchmark outcome state.')
        fields = ('success',) if teacher else ('student_success', 'control_success', 'qualified')
        if row['status'] == 'completed' and any(type(row.get(key)) is not bool for key in fields):
            raise ValueError('Completed outcomes require explicit boolean results.')
        if row['status'] == 'error' and any(row.get(key) is True for key in fields):
            raise ValueError('An execution error cannot qualify as success.')
        if not teacher and row.get('qualified') is True and not (
                row.get('student_success') is True and row.get('control_success') is False):
            raise ValueError('Qualification requires student success and failed negative control.')
        for key, value in row.items():
            if key.endswith(('_seconds', '_error_m')) and value is not None:
                _number(value, key)


def load_benchmark(directory):
    """Check receipt bindings and complete per-case accounting, without executing it."""
    root = Path(directory).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Benchmark directories must not traverse symbolic links.')
    report, report_hash = _read(root / 'report.json')
    protocol, protocol_hash = _read(root / 'protocol.json')
    models, models_hash = _read(root / 'models-frozen.json')
    checksum = root / 'protocol.sha256'
    if checksum.is_symlink() or not checksum.is_file() or checksum.stat().st_size > 128:
        raise ValueError('Missing bounded protocol checksum.')
    if (report.get('protocol_sha256') != protocol_hash or
            checksum.read_text().strip() != protocol_hash or
            models.get('protocol_sha256') != protocol_hash or
            report.get('models_frozen_sha256') != models_hash):
        raise ValueError('Benchmark receipt hashes do not match.')
    if (report.get('kind') != 'dvidia.footage-sample-count-result' or
            protocol.get('kind') != 'dvidia.footage-sample-count-benchmark' or
            any(d.get('schema_version') != 1 or d.get('scope') != 'simulation-only' or
                d.get('physical_robot_ready') is not False for d in (report, protocol))):
        raise ValueError('Only the supported simulation benchmark is accepted.')
    cases = protocol.get('scenes')
    if type(cases) is not list or not 1 <= len(cases) <= 24:
        raise ValueError('Expected one to 24 planned scenes.')
    scenes = {}
    from .skill_capsule import scene_sha256
    from .footage_benchmark import _layout_digest
    for scene in cases:
        if type(scene) is not dict or type(scene.get('scene')) is not dict:
            raise ValueError('Expected a frozen scene object.')
        identity = _identity(scene.get('id'))
        if identity in scenes or scene.get('group') not in ('nominal', 'actuator_stress'):
            raise ValueError('Duplicate or unsupported planned scene.')
        if (scene_sha256(scene['scene']) != scene.get('scene_sha256') or
                _layout_digest(scene['scene']) != scene.get('layout_sha256')):
            raise ValueError('Frozen scene contents do not match their hashes.')
        scenes[identity] = scene
    if type(report.get('teacher')) is not dict:
        raise ValueError('Expected an authored-controller result object.')
    _rows(report['teacher'].get('outcomes'), scenes, teacher=True)
    comparisons = report.get('comparisons')
    candidates = models.get('candidates')
    if type(comparisons) is not list or not 1 <= len(comparisons) <= 8 or type(candidates) is not list:
        raise ValueError('Expected bounded frozen benchmark candidates.')
    if len(comparisons) != len(candidates):
        raise ValueError('Every frozen candidate must retain a result.')
    identities = set()
    for row, frozen in zip(comparisons, candidates):
        if type(row) is not dict or type(frozen) is not dict:
            raise ValueError('Expected candidate result objects.')
        if any(row.get(key) != value for key, value in frozen.items()):
            raise ValueError('Candidate results differ from the frozen model record.')
        count = row.get('training_group_count')
        if type(count) is not int or not 1 <= count <= 64:
            raise ValueError('Invalid training group count.')
        identity = _identity(row.get('cohort_id', f'groups-{count:03}'))
        if identity in identities:
            raise ValueError('Repeated candidate identity.')
        identities.add(identity)
        _number(row.get('training_action_samples'), 'training sample count')
        _number(row.get('training_pipeline_seconds'), 'training time')
        _rows(row.get('outcomes'), scenes)
    _number(report.get('benchmark_wall_seconds'), 'benchmark time')
    if protocol.get('experiment') is not None:
        # Explicit cohorts must bind the prepared source's real connected groups.
        _validate_experiment_plan(protocol, report)
        dataset, dataset_hash = _read(root / 'source' / 'dataset.json')
        if dataset_hash != protocol.get('input_files_sha256', {}).get('source/dataset.json'):
            raise ValueError('Follow-up prepared source hash changed.')
        _prepared_episodes(dataset)
        from .footage_benchmark import explicit_subsets
        specs = protocol.get('subsets')
        if type(specs) is not list or any(type(s) is not dict for s in specs):
            raise ValueError('Expected explicit frozen source memberships.')
        derived = explicit_subsets(dataset, [{'cohort_id': s.get('cohort_id'),
                                              'train_group_ids': s.get('train_group_ids')} for s in specs])
        for spec, truth in zip(specs, derived):
            if any(spec.get(key) != value for key, value in truth.items()):
                raise ValueError('Follow-up membership differs from the prepared source groups.')
    return {'report': report, 'protocol': protocol,
            'report_sha256': report_hash, 'protocol_sha256': protocol_hash}


def candidate(bundle, identity):
    for row in bundle['report']['comparisons']:
        key = row.get('cohort_id', f"groups-{row['training_group_count']:03}")
        if key == identity:
            return row
    raise ValueError('Requested baseline is absent from this benchmark.')


def _counts(rows, group):
    rows = [r for r in rows if r['group'] == group]
    return {'qualified': sum(r.get('qualified') is True for r in rows),
            'total': len(rows), 'successes': sum(r.get('student_success') is True for r in rows),
            'errors': sum(r['status'] == 'error' for r in rows)}


def _evidence(row, teacher):
    return {key: row.get(key) for key in ('case_id', 'group', 'student_success', 'control_success')} | {
        'teacher_success': teacher.get('success'),
        'reason': row.get('student_reason', 'Execution did not complete')}


def diagnose(bundle, baseline_id):
    baseline = candidate(bundle, baseline_id)
    teachers = {r['case_id']: r for r in bundle['report']['teacher']['outcomes']}
    cases = {r['id']: r for r in bundle['protocol']['scenes']}
    buckets = {k: [] for k in ('execution', 'control', 'qualification', 'candidate', 'shared')}
    for row in baseline['outcomes']:
        teacher = teachers[row['case_id']]
        if row['status'] == 'error' or teacher['status'] == 'error':
            buckets['execution'].append(row)
        elif row['control_success']:
            buckets['control'].append(row)
        elif row['qualified']:
            continue
        elif row['student_success']:
            buckets['qualification'].append(row)
        elif teacher['success'] and not row['student_success']:
            buckets['candidate'].append(row)
        else:
            buckets['shared'].append(row)
    definitions = {
        'execution': ('Resolve incomplete trials',
                      'Execution errors prevent a reliable comparison. Restore the runner and repeat the frozen cases.',
                      'Inspect runtime errors and preserve every failed attempt before requesting new demonstrations.'),
        'control': ('Check the success test',
                    'The open-jaw control also completed. The current trial does not isolate useful learned behavior.',
                    'Inspect control traces and the outcome test. Any changed benchmark must receive a new version.'),
        'qualification': ('Investigate the rejected qualification',
                          'The candidate completed the task but its qualification did not pass. Inspect the qualification receipt before making a training-data diagnosis.',
                          'Inspect the qualification gate and bound artifacts; preserve successful and rejected outcomes separately.'),
        'candidate': ('Test examples near the missed placement',
                      'The authored controller succeeded where this candidate failed. Training coverage or the learner may explain the difference; the cause is unproven.',
                      'Compare targeted additions with the current dataset and an equal-group alternative on fresh scenes.'),
        'shared': ('Investigate the shared failure',
                   'Both the authored controller and candidate failed. Controller behavior, actuator limits or simulation assumptions may be responsible.',
                   'Inspect the grasp and actuator response before deciding whether additional recordings are useful.'),
    }
    result = []
    for category, rows in buckets.items():
        if not rows:
            continue
        title, explanation, action = definitions[category]
        capture = category == 'candidate'
        scene = cases[rows[0]['case_id']]['scene']
        position = ', '.join(f'{x:.3f}' for x in scene['object_position'])
        target = ', '.join(f'{x:.3f}' for x in scene['target_position'])
        instruction = (f'Collect three new, independent simulated rigid-block placement episodes in the supported envelope near start [{position}] m and target [{target}] m. '
                       'Vary nearby layouts rather than copying the development trial. Preserve synchronized video, measured state, issued robot commands, and the visible result. '
                       'Retain unsuccessful attempts and recovery with their actual outcome.' if capture else action)
        result.append({'id': f'{category}-investigation', 'category': category,
                       'title': title, 'explanation': explanation, 'status': 'provisional',
                       'next_action': action,
                       'request': {'title': title, 'instruction': instruction,
                                   'why': f'{len(rows)} development case(s) support this investigation; expected training benefit is unmeasured.',
                                   'capture_mode': 'aligned_simulation' if capture else 'diagnostic',
                                   'required_groups': 3 if capture else None,
                                   'acceptance': ['Keep source, session, object and derivative identities connected.',
                                                  'Keep evaluation-only sources outside training.',
                                                  'Retain before-state, the full attempt and visible after-state; mark hidden outcomes unknown.',
                                                  'Training requires validated aligned commands; ordinary phone video is observation evidence.',
                                                  'Three groups is a bounded collection budget, not a sufficient demonstration count.'] if capture else [
                                                      'Retain complete traces, failures and matched control evidence.',
                                                      'Keep changed conditions and benchmark versions explicit.']},
                       'evidence': [_evidence(row, teachers[row['case_id']]) for row in rows]})
    return result


def _paired(baseline, rows, group='nominal'):
    prior = {r['case_id']: r for r in baseline}
    values = {'improved': 0, 'regressed': 0, 'unchanged': 0, 'errors': 0}
    for row in rows:
        if row['group'] != group:
            continue
        old = prior[row['case_id']]
        if row['status'] == 'error' or old['status'] == 'error':
            values['errors'] += 1
        else:
            key = 'unchanged' if row['qualified'] == old['qualified'] else 'improved' if row['qualified'] else 'regressed'
            values[key] += 1
    return values


def _validate_followup(development, evaluation, baseline_id):
    """Bind the follow-up to the diagnosed source and fixed comparison recipe."""
    old, new = development['protocol'], evaluation['protocol']
    plan = _validate_experiment_plan(new, evaluation['report'])
    if new.get('source_dataset_sha256') != old.get('source_dataset_sha256'):
        raise ValueError('Follow-up source dataset differs from the diagnosed dataset.')
    if new.get('runtime') != old.get('runtime'):
        raise ValueError('Follow-up runtime or simulator implementation changed.')
    for name in ('footage_pipeline.py', 'footage_data.py', 'footage_train.py'):
        if new.get('benchmark_source_sha256', {}).get(name) != old.get('benchmark_source_sha256', {}).get(name):
            raise ValueError('Follow-up learner or input implementation changed.')
    if new.get('input_files_sha256', {}).get('task.skill.json') != old.get('input_files_sha256', {}).get('task.skill.json'):
        raise ValueError('Follow-up task definition changed.')
    for key, value in old.get('training', {}).items():
        if new.get('training', {}).get(key) != value:
            raise ValueError('Follow-up training recipe changed.')
    if new.get('training', {}).get('seed') != old.get('training', {}).get('seed', old.get('seed')):
        raise ValueError('Follow-up training seed changed.')
    if (plan.get('historical_protocol_sha256') != development['protocol_sha256'] or
            plan.get('historical_report_sha256') != development['report_sha256']):
        raise ValueError('Follow-up experiment is not bound to the diagnosed evidence.')
    prior = next((s for s in old['subsets'] if s.get('cohort_id', f"groups-{s['training_group_count']:03}") == baseline_id), None)
    if type(new.get('subsets')) is not list or len(new['subsets']) != 3 or any(type(s) is not dict for s in new['subsets']):
        raise ValueError('Follow-up requires three distinct cohort memberships.')
    specs = {s.get('cohort_id'): s for s in new['subsets']}
    if prior is None or set(specs) != {'baseline4', 'generic5', 'targeted5'}:
        raise ValueError('Follow-up cohort memberships are incomplete.')
    for spec in specs.values():
        partition = []
        for key in ('train_episode_ids', 'dev_episode_ids', 'test_episode_ids'):
            ids = spec.get(key)
            if type(ids) is not list or not ids or any(type(i) is not str for i in ids):
                raise ValueError('Follow-up partitions require nonempty episode identities.')
            if len(ids) != len(set(ids)):
                raise ValueError('Follow-up partitions cannot duplicate episodes.')
            partition.append(set(ids))
        if any(partition[a] & partition[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Follow-up episodes cross partitions.')
    for key in ('train_group_ids', 'train_episode_ids', 'dev_episode_ids', 'test_episode_ids'):
        if specs['baseline4'].get(key) != prior.get(key):
            raise ValueError('Follow-up baseline membership changed.')
    baseline_groups = set(prior['train_group_ids'])
    additions = []
    for name in ('generic5', 'targeted5'):
        spec = specs[name]
        groups = spec.get('train_group_ids', [])
        if (len(set(groups)) != len(groups) or not baseline_groups < set(groups) or
                len(groups) != len(baseline_groups) + 1 or
                not set(prior['train_episode_ids']) < set(spec.get('train_episode_ids', []))):
            raise ValueError('Follow-up additions must retain the baseline and add one distinct source group.')
        for key in ('dev_episode_ids', 'test_episode_ids'):
            if spec.get(key) != prior.get(key):
                raise ValueError('Follow-up development/test partition changed.')
        train_ids = spec['train_episode_ids']
        heldout_ids = set(prior['dev_episode_ids']) | set(prior['test_episode_ids'])
        if len(set(train_ids)) != len(train_ids) or set(train_ids) & heldout_ids:
            raise ValueError('Follow-up training episodes duplicate or cross held-out partitions.')
        additions.append(set(groups) - baseline_groups)
    if additions[0] == additions[1]:
        raise ValueError('Targeted and general additions must be distinct.')
    for comparison in evaluation['report']['comparisons']:
        spec = specs.get(comparison.get('cohort_id'), {})
        if comparison['training_group_count'] != len(spec.get('train_group_ids', [])):
            raise ValueError('Follow-up candidate count differs from its frozen membership.')
    known = {r['layout_sha256'] for r in old['scenes']}
    known.update(r['layout_sha256'] for r in new.get('source_scene_exclusion', {}).get('bindings', []))
    if known & {r['layout_sha256'] for r in new['scenes']}:
        raise ValueError('Follow-up scenes reuse known development or source layouts.')


def build_dashboard(development, baseline_id='groups-004', evaluation=None):
    """A read model of measured results and clearly provisional capture hypotheses."""
    base = candidate(development, baseline_id)
    hypotheses = diagnose(development, baseline_id)
    previous_hypotheses = []
    diagnosis_source = development
    shown_id = baseline_id
    comparisons = []
    conclusion = {'status': 'planned', 'headline': 'Test the next useful example.',
                  'detail': 'These hypotheses come from development failures. Their training benefit has not been measured.',
                  'next_step': 'Freeze a baseline, a targeted addition and an equal-group alternative before evaluating them.'}
    shown = base
    wall = development['report']['benchmark_wall_seconds']
    if evaluation is not None:
        _validate_followup(development, evaluation, baseline_id)
        # This first adapter deliberately accepts one named experimental recipe.
        labels = {'baseline4': 'Current dataset', 'generic5': 'General addition', 'targeted5': 'Targeted addition'}
        cohorts = {r.get('cohort_id'): r for r in evaluation['report']['comparisons']}
        if set(cohorts) != set(labels):
            raise ValueError('Follow-up requires baseline4, generic5 and targeted5 cohorts.')
        shown = cohorts['baseline4']
        shown_id = 'baseline4'
        # Once inspected, the evaluation is development evidence for the next
        # investigation. Never keep issuing a stale collection request merely
        # because the prior hypothesis was plausible.
        previous_hypotheses = hypotheses
        hypotheses = diagnose(evaluation, shown_id)
        diagnosis_source = evaluation
        if cohorts['generic5']['training_group_count'] != cohorts['targeted5']['training_group_count']:
            raise ValueError('Addition arms must use an equal group budget.')
        for key, label in labels.items():
            row = cohorts[key]
            comparisons.append({'id': key, 'label': label,
                                'train_groups': row['training_group_count'],
                                'train_samples': row['training_action_samples'],
                                'nominal': _counts(row['outcomes'], 'nominal'),
                                'stress': _counts(row['outcomes'], 'actuator_stress'),
                                'paired': _paired(shown['outcomes'], row['outcomes']),
                                'paired_stress': _paired(shown['outcomes'], row['outcomes'], 'actuator_stress'),
                                'training_seconds': row['training_pipeline_seconds']})
        targeted = comparisons[2]
        paired = targeted['paired']
        stress = targeted['paired_stress']
        total_errors = sum(c['nominal']['errors'] + c['stress']['errors'] for c in comparisons)
        if total_errors:
            state, headline = 'incomplete', 'Resolve incomplete trials before judging benefit.'
        elif paired['regressed'] or stress['regressed']:
            state, headline = 'regression_observed', 'The targeted addition introduced regressions.'
        elif paired['improved']:
            state, headline = 'observed_gain', 'The targeted addition improved this nominal suite.'
        else:
            state, headline = 'no_measured_gain', 'No nominal completion gain demonstrated.'
        conclusion = {'status': state, 'headline': headline,
                      'detail': f"Across paired nominal cases: {paired['improved']} improved, {paired['regressed']} regressed, {paired['unchanged']} unchanged and {paired['errors']} incomplete. Stress cases: {stress['improved']} improved, {stress['regressed']} regressed, {stress['unchanged']} unchanged and {stress['errors']} incomplete. These are shared cases, not independent repeated studies. Compare the general addition before crediting the selection rule.",
                      'next_step': 'Keep the current model until a predeclared improvement passes regression checks. Use these inspected cases for development; a further claim needs fresh confirmation.'}
        wall = evaluation['report']['benchmark_wall_seconds']
    nominal, stress = (_counts(shown['outcomes'], group) for group in ('nominal', 'actuator_stress'))
    task_hash = diagnosis_source['protocol']['input_files_sha256'].get('task.skill.json')
    return {'format': FORMAT, 'title': 'Place a block', 'scope': 'Simulation pilot',
            'task': {'id': 'task-' + task_hash[:16], 'name': 'Place a block',
                     'definition_sha256': task_hash, 'scope': 'simulation-only',
                     'skillspace_binding': 'unlinked',
                     'qualification': {'passed': nominal['qualified'] + stress['qualified'],
                                       'total': nominal['total'] + stress['total'],
                                       'status': 'development', 'physical_validation': 'not_tested'},
                     'binding_note': 'This synthetic research task is not linked to an account Skillspace. Evidence must match the exact Skillspace, brief, task and dataset revision before it can appear there.'},
            'stage': 'Experiment complete' if evaluation else 'Hypotheses ready',
            'baseline': {'id': shown_id, 'label': 'Current dataset'},
            'summary': {'training_groups': shown['training_group_count'],
                        'nominal': _counts(shown['outcomes'], 'nominal'),
                        'stress': _counts(shown['outcomes'], 'actuator_stress'), 'wall_seconds': wall},
            'hypotheses': hypotheses, 'next_hypothesis_id': hypotheses[0]['id'] if hypotheses else None,
            'previous_hypotheses': previous_hypotheses,
            'comparisons': comparisons, 'conclusion': conclusion,
            'evidence': {'development_report_sha256': development['report_sha256'],
                         'development_protocol_sha256': development['protocol_sha256'],
                         'diagnosis_report_sha256': diagnosis_source['report_sha256'],
                         'diagnosis_protocol_sha256': diagnosis_source['protocol_sha256'],
                         'evaluation_report_sha256': evaluation['report_sha256'] if evaluation else None,
                         'evaluation_protocol_sha256': evaluation['protocol_sha256'] if evaluation else None},
            'limitations': ['Local research pilot; no production account or automatic training integration.',
                            'Rule-based diagnoses are hypotheses, not causal findings or calibrated probabilities.',
                            'Movement is learned from aligned numerical commands. The video predictor does not control this arm.',
                            'Equal source-group budgets can contain different numbers of frames and action samples.',
                            'Source hashes bind reported evidence; they do not independently attest the capture history.',
                            'Nominal and actuator-stress results are separate. Simulation qualification does not establish physical robot readiness.']}


def capture_plan(dashboard):
    """Generate a real empty collection plan; never fabricate accepted evidence."""
    from .coverage import example_plan, validate_plan
    requests = [h for h in dashboard['hypotheses'] if h['request']['capture_mode'] == 'aligned_simulation']
    if not requests:
        raise ValueError('This diagnosis does not support a recording request yet.')
    plan = deepcopy(example_plan())
    plan['plan_id'] = 'improvement-' + dashboard['evidence']['diagnosis_report_sha256'][:16]
    plan['skill']['name'] = 'Rigid block placement · targeted collection'
    plan['evidence'] = []
    plan['cases'] = [{'id': h['id'], 'label': h['title'], 'brief': h['request']['instruction'],
                      'outcome': 'success', 'required_groups': h['request']['required_groups'], 'priority': 10}
                     for h in requests]
    plan['benchmark']['environment'] = 'Existing authored rigid-box placement simulation adapter'
    plan['benchmark']['reason'] = 'Development hypothesis; collection targets do not establish training benefit. Preserve failures separately with their actual outcome.'
    return validate_plan(plan)


def brief(dashboard, identity):
    item = next((h for h in dashboard['hypotheses'] if h['id'] == identity), None)
    if item is None:
        raise ValueError('Unknown investigation.')
    request = item['request']
    return '\n'.join(['DVIDIA · ' + dashboard['title'], request['title'], 'Provisional investigation', '',
                      item['explanation'], '', request['instruction'], '', request['why'], '',
                      *['- ' + s for s in request['acceptance']], '',
                      'Diagnosis report: ' + dashboard['evidence']['diagnosis_report_sha256'],
                      *(['Diagnostic report: ' + item['diagnostic_report_sha256']]
                        if item.get('diagnostic_report_sha256') else []),
                      'Evidence cases: ' + ', '.join(e['case_id'] for e in item['evidence']),
                      'No reward, model improvement or physical robot qualification is promised.', ''])
