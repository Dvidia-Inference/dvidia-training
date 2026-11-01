"""Descriptive comparison of all sealed initial/corrected development attempts.

Reads no confirmation outcomes; executes no physics, policy or training. Saved
numeric outcomes are reconstructed, while media/perception hashes remain bound
by episode manifests rather than being re-audited in full.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

import numpy as np


def sibling(name):
    spec = importlib.util.spec_from_file_location('camera_development_' + name,
        Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


audit = sibling('study_audit')
first = sibling('diagnose_first_student')
SCHEMA = 'dvidia.camera-action-development-failures/v1'
DEVELOPMENT = ('development-initial', 'development-corrected')


def load(root, name, sources):
    path = audit.relative(root, name)
    value = audit.read(path); sources[name] = audit.identity(path)
    return value


def stage(root, name, protocol, protocol_hash, sources):
    value = load(root, 'stages/' + name + '.json', sources)
    seal = load(root, 'stages/' + name + '.integrity.json', sources)
    start = load(root, 'stages/' + name + '.started.json', sources)
    audit.require(seal == dict(sha256=sources['stages/' + name + '.json']['sha256'],
        protocol_sha256=protocol_hash), 'Stage seal differs: ' + name)
    expected_schema='dvidia.camera-action-study/v1/'+('training' if name.startswith('train-') else 'stage')
    audit.require(value['schema'] == expected_schema
        and start['schema'] == 'dvidia.camera-action-study/v1/stage-start'
        and value['stage'] == start['stage'] == name
        and value['protocol_sha256'] == start['protocol_sha256'] == protocol_hash
        and start['source_sha256'] == protocol['source_sha256']
        and value.get('status') in ('complete', 'blocked')
        and start.get('model_bindings') == value.get('model_bindings'), 'Stage registration differs: ' + name)
    if not name.startswith('train-'):
        roster = protocol['rosters'][name]; rows = value['rows']
        audit.require(value['scheduled'] == len(roster)
            and [r['id'] for r in rows] == [r['id'] for r in roster]
            and len({r['id'] for r in rows}) == len(rows), 'Sealed stage roster differs: ' + name)
        for row, registered in zip(rows, roster):
            audit.require(all(row.get(key) == registered.get(key) for key in
                ('id', 'case', 'controller', 'negative_control')), 'Registered attempt identity differs.')
    return value


def measure(steps, arrays, native_score):
    """Numeric measurements for a complete, validated trajectory only."""
    times = first.control_clock(steps)
    for index, step in enumerate(steps):
        for actual, expected in ((step['command']['issued_at_seconds'], index * .02),
                (step['after']['timestamp_seconds'], (index + 1) * .02),
                (step['applied']['interval_start_seconds'], index * .02),
                (step['applied']['interval_end_seconds'], (index + 1) * .02)):
            audit._close(actual, expected, 'Command/transition clock differs.')
        audit.require(type(step['applied']['rate_limited']) is bool
            and step['applied']['rate_limited'] == (step['applied']['ramped_targets'] != step['command']['issued']),
            'Recorded rate-limited flag differs from actual applied targets.')
        tick = (index + 1) * 20 - 1
        np.testing.assert_allclose(step['applied']['ramped_targets']['joint_targets_rad'],
            arrays['arm_actuator_controls_rad'][tick], atol=1e-10, rtol=0)
        audit._close(step['applied']['ramped_targets']['gripper_position_m'],
            float(arrays['normal_rate_limited_jaw_target_m'][tick]), 'Applied jaw reference differs.', tolerance=1e-10)
    jaw = np.asarray([s['command']['issued']['gripper_position_m'] for s in steps])
    closure = None; indices = np.flatnonzero(jaw < .020)
    if len(indices):
        index = int(indices[0]); tick = index * 20 - 1
        closure = dict(time_seconds=float(times[index]), issued_single_finger_target_m=float(jaw[index]),
            geometry_available=tick >= 0, tcp_object_xy_distance_m=None,
            tcp_minus_object_z_m=None, actual_finger_positions_m=None)
        if tick >= 0:
            delta = arrays['tcp_position_m'][tick] - arrays['object_position_m'][tick]
            closure.update(tcp_object_xy_distance_m=float(np.linalg.norm(delta[:2])),
                tcp_minus_object_z_m=float(delta[2]),
                actual_finger_positions_m=arrays['gripper_joint_position_m'][tick].tolist())
    critical = int(np.count_nonzero(arrays['unexpected_contact_count'] > 0))
    audit.require(critical == native_score['unexpected_contact_native_ticks'], 'Native critical count differs.')
    return dict(control_intervals=1000, rate_limited_intervals=sum(s['applied']['rate_limited'] for s in steps),
        critical_contact_ticks=critical, critical_contact_episode=critical > 0,
        bilateral_contact_ticks=int(np.count_nonzero(np.all(arrays['grasp_contact_counts'] > 0, axis=1))),
        strict_grasp_seen=native_score['grasp_seen'], lift_seen=native_score['lift_seen'],
        maximum_object_lift_m=native_score['max_object_lift_m'],
        no_closure_command=closure is None, first_closure_command=closure,
        closure_command_beyond_20mm_xy=(False if closure is None else
            closure['tcp_object_xy_distance_m'] > .020 if closure['geometry_available'] else None))


def episode(root, stage_name, row, model_binding, tracker_hash, sources):
    audit.require(type(row.get('complete')) is bool and type(row.get('strict_success')) is bool
        and row['status'] in ('complete','interrupted','refused','unknown','unattempted'),
        'Unknown attempt status or nonboolean outcome.')
    result = dict(id=row['id'], stage=stage_name, controller=row['controller'], case=row['case'],
        policy_artifact_sha256=model_binding,
        status=row['status'], complete=row.get('complete') is True,
        strict_success=row.get('strict_success') is True, metrics=None, missing_reason=None)
    relative = 'episodes/' + stage_name + '/' + row['id']
    if row['status'] in ('unknown', 'unattempted'):
        audit.require(not result['complete'] and not result['strict_success']
            and (root / relative).exists() is (row['status'] == 'unknown'),
            'Unknown/unattempted outcome or directory presence differs.')
        # Do not inspect an unknown prefix, which may not have sealed byte evidence.
        result['missing_reason'] = 'No sealed episode for this scheduled attempt.'
        return result
    audit.require(row['path'] == relative and row['policy_artifact_sha256'] == model_binding,
        'Episode path or model binding differs.')
    directory = audit.relative(root, relative)
    manifest = load(root, relative + '/integrity.json', sources)
    audit.require(sources[relative + '/integrity.json']['sha256'] == row['integrity_sha256']
        and manifest['schema'] == 'dvidia.camera-action-rollout/v2/integrity', 'Episode integrity differs from sealed stage.')
    files = manifest['files']; local = {}
    def bound(name):
        raw = first.bound(directory, name, files, local)
        sources[relative + '/' + name] = local[name]
        return json.loads(raw)
    receipt = bound('rollout.json'); started = bound('started.json')
    mode = 'student' if row['controller'] == 'blind' else row['controller']
    case = {**row['case'], 'policy_artifact_sha256': model_binding}
    audit.require(sources[relative + '/rollout.json']['sha256'] == row['receipt_sha256']
        and receipt['case'] == started['case'] == case and receipt['mode'] == started['mode'] == mode
        and receipt['negative_control'] == started['negative_control'] == row['negative_control'] is None
        and receipt['status'] == row['status'] and receipt['score'] == row['score']
        and receipt['strict_success'] is result['strict_success']
        and receipt['schema'] == started['schema'] == 'dvidia.camera-action-rollout/v2',
        'Episode receipt differs from stage or registration.')
    audit.require(receipt['tracker_sha256'] == tracker_hash, 'Frozen tracker binding differs.')
    config = started['config']
    audit.require(config['duration_s'] == 20. and config['seed'] == case['seed']
        and config['width'] == 320 and config['height'] == 240 and config['camera_every_controls'] == 2,
        'Started capture budget differs.')
    teacher_files = sorted(name for name in files if name.startswith('teacher/') and name.endswith('.json'))
    audit.require(receipt['teacher_queries'] == len(teacher_files)
        and teacher_files == [f'teacher/{i:04}.json' for i in range(len(teacher_files))],
        'Recorded teacher query inventory differs.')
    if mode == 'student':
        audit.require(not teacher_files and receipt['no_privileged_features_or_teacher_intervention'] is True,
            'Learner contains teacher intervention.')
    for name in ('native_episode.json', 'native_metadata.json', 'native.npz'):
        if name in files:
            first.bound(directory, name, files, local)
            sources[relative + '/' + name] = local[name]
    native, arrays, metadata, score = audit._native(directory, files, case, receipt)
    steps = []
    if 'recording.json' in files:
        recording = bound('recording.json'); steps = recording['steps']
        audit.require(recording['schema'] == 'dvidia.camera-action-rollout/v2' and recording['control_mode'] == mode
            and recording['provenance']['case_id'] == row['case']['id']
            and recording['provenance']['source_group'] == row['case']['group_id']
            and recording['goal_position_m'] == row['case']['goal_position_m']
            and recording['tracker_sha256'] == receipt['tracker_sha256'], 'Recording identity differs.')
        for index, step in enumerate(steps):
            audit._close(step['before']['timestamp_seconds'], index * .02, 'Recorded decision clock differs.')
            audit.require(step['index'] == index and step['command']['authority'] == mode
                and step['command']['operator_correction'] is False
                and step['command']['requested'] == step['command']['issued'], 'Unexpected intervention or control authority.')
            audit.require(native is not None and len(native['actions']) > index,
                'Recorded command lacks native action evidence.')
            audit.require({key:native['actions'][index][key] for key in ('joint_targets_rad','gripper_position_m')}
                == step['command']['issued'], 'Issued command differs from native action evidence.')
            if mode == 'student':audit.require(step['teacher_query'] is None, 'Learner contains a teacher query.')
    if mode == 'teacher':
        audit.require(len(steps) <= receipt['teacher_queries'] <= len(steps) + 1 and receipt['student_calls'] == 0,
            'Teacher query or student-call count differs.')
    else:
        audit.require(len(steps) <= receipt['student_calls'] <= len(steps) + 1,
            'Student-call count differs from completed prefix.')
    audit.require(len(steps) == receipt['recorded_complete_intervals'] == row['recorded_complete_intervals']
        and result['complete'] == (receipt['status'] == 'complete' and len(steps) == 1000), 'Episode completeness differs.')
    audit.require(result['strict_success'] == (result['complete'] and score.get('success') is True),
        'Strict success differs from complete independently scored evidence.')
    if result['complete']:
        audit.require(metadata is not None and metadata['rows'] == 20000, 'Complete trajectory lacks 20,000 native ticks.')
    result.update(recorded_intervals=len(steps), native_ticks=metadata['rows'] if metadata else None,
        native_score_valid=score['valid'], teacher_queries=receipt['teacher_queries'])
    if result['complete'] and score['valid']:
        result['metrics'] = measure(steps, arrays, score['native_recomputed'])
    else:
        result['missing_reason'] = 'No complete valid trajectory; prefix/refusal is retained without full-episode metric imputation.'
    return result


def compare(row, teacher):
    value = dict(teacher_id=teacher['id'], teacher_strict_success=teacher['strict_success'],
        both_complete_metrics=row['metrics'] is not None and teacher['metrics'] is not None,
        closure_time_difference_seconds=None, closure_xy_difference_m=None,
        closes_before_teacher=None, rate_limited_interval_difference=None,
        critical_contact_tick_difference=None)
    if not value['both_complete_metrics']:return value
    a, b = row['metrics'], teacher['metrics']
    value.update(rate_limited_interval_difference=a['rate_limited_intervals']-b['rate_limited_intervals'],
        critical_contact_tick_difference=a['critical_contact_ticks']-b['critical_contact_ticks'])
    x, y = a['first_closure_command'], b['first_closure_command']
    if x and y:
        value['closure_time_difference_seconds'] = x['time_seconds']-y['time_seconds']
        value['closes_before_teacher'] = x['time_seconds'] < y['time_seconds']
        if x['geometry_available'] and y['geometry_available']:
            value['closure_xy_difference_m'] = x['tcp_object_xy_distance_m']-y['tcp_object_xy_distance_m']
    return value


def distribution(values):
    values = [float(x) for x in values if x is not None]
    return dict(known=len(values), median=float(np.median(values)) if values else None,
        p95=float(np.quantile(values, .95)) if values else None,
        minimum=min(values) if values else None, maximum=max(values) if values else None)


def aggregate(rows):
    metrics = [r['metrics'] for r in rows if r['metrics'] is not None]
    def count(predicate):
        values = [predicate(m) for m in metrics]; known = [v for v in values if v is not None]
        return dict(events=sum(bool(v) for v in known), known=len(known), missing=len(rows)-len(known))
    result = dict(scheduled=len(rows), strict_successes=sum(r['strict_success'] for r in rows),
        statuses=dict(sorted(Counter(r['status'] for r in rows).items())), complete_valid_metrics=len(metrics),
        unavailable_metrics=len(rows)-len(metrics),
        no_bilateral_contact=count(lambda m:m['bilateral_contact_ticks']==0),
        no_strict_grasp=count(lambda m:not m['strict_grasp_seen']),
        no_closure_command=count(lambda m:m['no_closure_command']),
        closure_command_beyond_20mm_xy=count(lambda m:m['closure_command_beyond_20mm_xy']),
        grasp_without_lift=dict(events=sum(m['strict_grasp_seen'] and not m['lift_seen'] for m in metrics),
            known=sum(m['strict_grasp_seen'] for m in metrics),
            not_applicable=sum(not m['strict_grasp_seen'] for m in metrics), missing=len(rows)-len(metrics)),
        critical_contact_episodes=count(lambda m:m['critical_contact_episode']),
        critical_contact_ticks_observed=sum(m['critical_contact_ticks'] for m in metrics),
        rate_limited_intervals_observed=sum(m['rate_limited_intervals'] for m in metrics),
        control_intervals_observed=sum(m['control_intervals'] for m in metrics),
        closure_time_seconds=distribution([m['first_closure_command']['time_seconds'] if m['first_closure_command'] else None for m in metrics]),
        closure_xy_distance_m=distribution([m['first_closure_command']['tcp_object_xy_distance_m'] if m['first_closure_command'] else None for m in metrics]))
    comparisons = [r['teacher_comparison'] for r in rows if 'teacher_comparison' in r]
    if comparisons:
        earlier = [c['closes_before_teacher'] for c in comparisons if c['closes_before_teacher'] is not None]
        result['matched_teacher'] = dict(paired_complete_metrics=sum(c['both_complete_metrics'] for c in comparisons),
            closure_earlier=dict(events=sum(earlier), known=len(earlier), missing=len(rows)-len(earlier)),
            closure_time_difference_seconds=distribution([c['closure_time_difference_seconds'] for c in comparisons]),
            closure_xy_difference_m=distribution([c['closure_xy_difference_m'] for c in comparisons]),
            rate_limited_interval_difference=distribution([c['rate_limited_interval_difference'] for c in comparisons]))
    return result


def development_cases(protocol, stages):
    expected_cases = {c['id']:c for c in protocol['cases']['development']}
    audit.require(len(expected_cases)==len(protocol['cases']['development'])==16,
        'Expected all 16 unique registered development cases.')
    for name in DEVELOPMENT:
        rows = stages[name]['rows']
        audit.require(len(rows)==32 and {(r['case']['id'],r['controller']) for r in rows}
            == {(key,controller) for key in expected_cases for controller in ('student','blind')}
            and all(r['case']==expected_cases[r['case']['id']] for r in rows), 'Development coverage differs.')
    return expected_cases


def summarize(study):
    root = Path(study).absolute(); sources = {}
    # No episode or outcome is read until BOTH development stage seals exist.
    for name in DEVELOPMENT:
        audit.require((root / 'stages' / (name+'.integrity.json')).is_file(), 'Both development stages must be sealed.')
    protocol = load(root, 'protocol.json', sources); registration = load(root, 'registration.json', sources)
    ph = sources['protocol.json']['sha256']
    audit.require(registration['protocol_sha256'] == ph and registration['source_sha256'] == protocol['source_sha256'], 'Registration differs.')
    stages = {name:stage(root, name, protocol, ph, sources) for name in
        ('demonstrations', 'train-initial', 'development-initial', 'train-corrected', 'development-corrected')}
    expected_cases = development_cases(protocol, stages)
    runtime = {}
    for name, digest in protocol['source_sha256'].items():
        path = Path(audit.scoring.__file__).with_name(name)
        actual = audit.identity(path); audit.require(actual['sha256']==digest, 'Frozen runtime file differs: '+name)
        runtime[name]=actual
    for candidate in ('initial','corrected'):
        fit = stages['train-'+candidate]
        audit.require(fit['status']=='complete', 'A referenced model fitting stage is incomplete.')
        for variant, controller in (('full','student'),('blind','blind')):
            path = 'models/'+candidate+'/'+variant+'/model.json'
            model = load(root,path,sources)
            audit.require(fit['models'][variant]['path']==path
                and fit['models'][variant]['model_sha256']==sources[path]['sha256']
                and stages['development-'+candidate]['model_bindings'][controller]==sources[path]['sha256']
                and model['schema'] == 'dvidia.camera-action-mlp/v1'
                and model['feature_contract'] == 'camera-encoder-absolute-causal160/v1'
                and model['training']['stage'] == candidate
                and model['training'] == fit['models'][variant]['training']
                and model['blind'] is (variant=='blind')
                and model['training']['model_source_sha256']==protocol['source_sha256']['camera_action_model.py'], 'Model provenance differs.')
    audit.require(stages['demonstrations']['model_bindings'] == {'teacher':audit.digest(protocol['source_sha256'])},
        'Teacher source identity differs.')
    tracker_hash=protocol['source_sha256']['piper_rgbd_tracker.py']
    teachers = {}
    for row in stages['demonstrations']['rows']:
        if row['case']['id'] in expected_cases:
            audit.require(row['controller']=='teacher' and row['case']==expected_cases[row['case']['id']]
                and row['case']['id'] not in teachers, 'Matched teacher is duplicated or differs.')
            teachers[row['case']['id']] = episode(root,'demonstrations',row,
                stages['demonstrations']['model_bindings']['teacher'],tracker_hash,sources)
    audit.require(set(teachers)==set(expected_cases), 'A matching teacher is missing.')
    rows=[]
    for name in DEVELOPMENT:
        for row in stages[name]['rows']:
            result=episode(root,name,row,stages[name]['model_bindings'][row['controller']],tracker_hash,sources)
            result['teacher_comparison']=compare(result,teachers[row['case']['id']]);rows.append(result)
    cohorts={name+'/'+controller:aggregate([r for r in rows if r['stage']==name and r['controller']==controller])
        for name in DEVELOPMENT for controller in ('student','blind')}
    helpers={name:audit.identity(Path(__file__).with_name(name)) for name in
        ('development_failure_summary.py','diagnose_first_student.py','study_audit.py','independent_audit.py')}
    return dict(schema=SCHEMA,posthoc=True,scope='All 64 registered learned development attempts and 16 exact matched teacher demonstrations. Descriptive numeric reconstruction; not causal inference, qualification or a full media/perception audit.',
        audit_status='sealed_local_development_summary_pending_full_study_audit',
        physics_executed=False,model_executed=False,training_executed=False,teacher_queried=False,
        confirmation_read=False,selection_changed=False,qualification_changed=False,
        definitions=dict(closure='First issued single-finger jaw target below 0.020 m, not measured closure or total aperture.',
            far_closure='XY distal TCP-object separation above 0.020 m at that command; descriptive posthoc threshold, not a preregistered task gate.',
            geometry='Saved native tick exactly at the decision boundary, before applying the new command; no t=0 native sample means geometry unavailable.',
            earlier='Earlier than the matched teacher in episode time; a different valid path or timing is allowed.',
            denominators='All scheduled rows retained. Full-episode metrics use only complete valid trajectories; prefix/refusal metrics are unavailable, never zero-imputed.',
            hidden_inputs='No phase, teacher, native object or future observation is inferred as a learner input here; source/model/control-authority bindings do not attest arbitrary external code.',
            depth='Synthetic camera transforms do not validate depth accuracy; renderer warnings remain a separate limitation, not a demonstrated cause.'),
        cohorts=cohorts,teacher_cohort=aggregate(list(teachers.values())),attempts=rows,teachers=list(teachers.values()),
        source_files=sources,source_inventory_sha256=audit.digest(sources),registered_runtime=runtime,helper_sources=helpers)


def markdown(value):
    def event(item):
        text=f"{item['events']}/{item['known']} known; {item['missing']} unavailable"
        return text+(f"; {item['not_applicable']} without a grasp" if 'not_applicable' in item else '')
    def number(x,scale=1):return 'Unavailable' if x is None else f'{x*scale:.2f}'
    lines=['# Development grip-acquisition summary','',value['scope'],'',
        '**Posthoc; pending full-study audit. No confirmation outcomes were read.**','',
        '| Cohort | Strict success / scheduled | No strict grasp | Closure >20 mm XY | Rate-limited / observed intervals | Critical-contact attempts |',
        '| --- | ---: | --- | --- | ---: | --- |']
    for name,c in [('Matched teachers',value['teacher_cohort']),*value['cohorts'].items()]:
        lines.append(f"| {name} | {c['strict_successes']}/{c['scheduled']} | {event(c['no_strict_grasp'])} | {event(c['closure_command_beyond_20mm_xy'])} | {c['rate_limited_intervals_observed']}/{c['control_intervals_observed']} | {event(c['critical_contact_episodes'])} |")
    lines+=['','| Learner cohort | First closure median (s) | XY distance median (mm) | Earlier than teacher | Grasp without lift |', '| --- | ---: | ---: | --- | --- |']
    for name,c in value['cohorts'].items():
        lines.append(f"| {name} | {number(c['closure_time_seconds']['median'])} | {number(c['closure_xy_distance_m']['median'],1000)} | {event(c['matched_teacher']['closure_earlier'])} | {event(c['grasp_without_lift'])} |")
    lines+=['','A closure command is not an attained jaw position. The 20 mm thresholds are descriptive choices, not new acceptance gates. Native geometry uses the distal tool site, not the legacy encoder link6 feature. Missing metrics and no-command cases remain explicit in the JSON.',
        '', 'These comparisons describe whether the first-case pattern recurs. They do not identify its cause, establish a population failure rate, tune a model, or replace registered development selection and frozen confirmation. Teacher phase labels are not used. Depth accuracy remains unvalidated.',
        '', 'Source inventory SHA-256: `'+value['source_inventory_sha256']+'`. All exact case/model/stage/source bindings and conditional denominators are in `development-failure-summary.json`.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output).absolute()
    audit.require(not output.exists() and not output.resolve().is_relative_to(Path(args.study).resolve()), 'Use a fresh output folder outside the study.')
    result=summarize(args.study);output.mkdir(parents=True)
    target=output/'development-failure-summary.json';target.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (output/'README.md').write_text(markdown(result))
    print(json.dumps(dict(output=str(output),attempts=len(result['attempts']),sha256=audit.identity(target)['sha256'],physics_executed=False)))


if __name__=='__main__':main()
