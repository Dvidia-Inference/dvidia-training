"""Read-only evidence for the bounded controlled-lowering experiment."""
from copy import deepcopy
import math
from pathlib import Path
import re

from .grip_diagnostic_view import _measurement
from .improvement import _number, _read, candidate


def _count(value):
    if type(value) is not int or not 0 <= value <= 10000:
        raise ValueError('Expected a bounded recovery count.')
    return value


def _vector(value):
    if (type(value) is not list or len(value) != 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000 for v in value)):
        raise ValueError('Expected finite recovery measurements.')
    return value


def _hashes(value, keys):
    if (type(value) is not dict or set(value) != set(keys) or
            any(type(v) is not str or re.fullmatch('[a-f0-9]{64}', v) is None for v in value.values())):
        raise ValueError('Expected complete frozen source bindings.')


def _recovery(value, measured, case, *, clearance_expected=False):
    if type(value) is not dict or type(value.get('controlled_abort')) is not bool:
        raise ValueError('Recovery measurements are missing.')
    counts = {key: _count(value.get(key)) for key in (
        'unsupported_open_command_samples', 'supported_open_command_samples',
        'lower_control_samples', 'contact_loss_during_lower_samples')}
    if counts['contact_loss_during_lower_samples'] > counts['lower_control_samples']:
        raise ValueError('Contact-loss count exceeds lowering observations.')
    if (counts['lower_control_samples'] > measured['control_steps'] or
            counts['unsupported_open_command_samples'] + counts['supported_open_command_samples'] > measured['control_steps']):
        raise ValueError('Recovery counts exceed the executed control steps.')
    peak = value.get('sampled_peak_descent_speed_before_support_m_s')
    if peak is not None:
        _number(peak, 'sampled descent speed')
    first = value.get('first_open_command')
    opens = counts['supported_open_command_samples'] + counts['unsupported_open_command_samples']
    if (first is None) != (opens == 0):
        raise ValueError('Opening counts and evidence disagree.')
    if first is not None:
        if type(first) is not dict or not 0 <= _number(first.get('time'), 'opening time') <= measured['simulated_seconds'] + 1e-8:
            raise ValueError('Invalid opening evidence.')
        contact = _count(first.get('table_contacts')) > 0
        if first.get('supported_by_contact_sample') is not contact:
            raise ValueError('Opening contact evidence contradicts its label.')
        if not .065 < _number(first.get('requested_gripper_width'), 'opening command') <= .08:
            raise ValueError('Opening evidence does not command an open jaw.')
        if counts['supported_open_command_samples' if contact else 'unsupported_open_command_samples'] == 0:
            raise ValueError('Opening evidence is missing from the counts.')
    confirmation = value.get('support_confirmation')
    if confirmation is not None:
        if (type(confirmation) is not dict or first is None or
                _count(confirmation.get('consecutive_control_samples')) < 5 or
                _count(confirmation.get('table_contacts')) == 0 or
                confirmation.get('time') != first.get('time') or
                not first['supported_by_contact_sample']):
            raise ValueError('Support confirmation does not match the opening.')
        position = _vector(confirmation.get('object_position'))
        velocity = _vector(confirmation.get('object_velocity'))
        angular = _vector(confirmation.get('object_angular_velocity'))
        if (abs(position[2] - case['scene']['object_position'][2]) >= .008 or
                sum(v*v for v in velocity) >= .035**2 or sum(v*v for v in angular) >= .3**2 or
                any(confirmation.get(k) != first.get(k) for k in
                    ('object_position', 'object_velocity', 'object_angular_velocity', 'table_contacts'))):
            raise ValueError('Confirmed support violates the frozen settled-state conditions.')
    if measured['recovery_count'] == 0 and (opens or confirmation is not None or counts['lower_control_samples']):
        raise ValueError('Recovery observations exist without a recovery.')
    start = value.get('first_recovery_observation')
    if start is not None:
        if (type(start) is not dict or measured['recovery_count'] == 0 or
                not 0 <= _number(start.get('time'), 'recovery start') <= measured['simulated_seconds'] + 1e-8 or
                (first is not None and start['time'] > first['time'])):
            raise ValueError('Recovery start contradicts the measured episode.')
    release = value.get('actual_release_confirmation')
    if release is not None:
        start = value.get('first_recovery_observation')
        if (type(release) is not dict or confirmation is None or type(start) is not dict or
                _count(release.get('consecutive_control_samples')) < 5 or
                _count(release.get('table_contacts')) == 0 or
                release.get('contacts') != {'left': 0, 'right': 0} or
                not .065 < _number(release.get('gripper_width'), 'actual jaw width') <= .081):
            raise ValueError('Release clearance evidence is incomplete.')
        elapsed = _number(release.get('elapsed_recovery_seconds'), 'recovery elapsed time')
        released_at = _number(release.get('time'), 'clearance time')
        began = _number(start.get('time'), 'recovery start')
        if (not confirmation['time'] + .08 - 1e-8 <= released_at <= measured['simulated_seconds'] + 1e-8 or
                elapsed > 4.00000001 or abs(released_at - began - elapsed) > 1e-8):
            raise ValueError('Release clearance exceeded the shared recovery budget.')
        pos = _vector(release.get('object_position'))
        velocity = _vector(release.get('object_velocity'))
        angular = _vector(release.get('object_angular_velocity'))
        _vector(release.get('tcp_position'))
        if (abs(pos[2] - case['scene']['object_position'][2]) >= .008 or
                sum(v*v for v in velocity) >= .035**2 or sum(v*v for v in angular) >= .3**2):
            raise ValueError('Release clearance lacks settled support.')
    events = []
    if start is not None:
        events.append({'kind': 'recovery_start', 'label': 'Recovery triggered',
                       'time_seconds': start['time'], 'state': 'observed'})
    if first is not None:
        events.append({'kind': 'first_open_command',
                       'label': 'Open commanded after settled support' if confirmation is not None else
                                'Open commanded without table contact' if not first['supported_by_contact_sample'] else
                                'Open commanded with table contact',
                       'time_seconds': first['time'], 'state': 'observed'})
    if release is not None:
        events.append({'kind': 'jaw_clearance', 'label': 'Clearance confirmed',
                       'time_seconds': release['time'], 'state': 'observed'})
    events.append({'kind': 'episode_end', 'label': 'Task completed' if measured['success'] else 'Task incomplete',
                   'time_seconds': measured['simulated_seconds'],
                   'state': 'observed' if measured['success'] else 'failed'})
    return {'events': events,
            'support_confirmed_before_open': None if measured['recovery_count'] == 0 else
                confirmation is not None,
            'release_clearance_confirmed': release is not None if clearance_expected and measured['recovery_count'] else None,
            'unsupported_release_actions': counts['unsupported_open_command_samples'],
            'contact_loss_during_lower_samples': counts['contact_loss_during_lower_samples'],
            'lower_control_samples': counts['lower_control_samples'],
            'sampled_peak_descent_speed_before_support_m_s': peak,
            'controlled_abort': value['controlled_abort']}


def load_recovery(directory, evaluation):
    """Validate the original 16-case paired experiment; never import a simulator."""
    root = Path(directory).absolute()
    report, report_hash = _read(root / 'report.json')
    protocol, protocol_hash = _read(root / 'protocol.json')
    if (protocol.get('kind') != 'dvidia.controlled-recovery-experiment' or
            report.get('kind') != 'dvidia.controlled-recovery-report' or
            report.get('protocol_sha256') != protocol_hash or
            any(p.get('schema_version') != 1 or p.get('scope') != 'simulation-only' or
                p.get('physical_robot_ready') is not False for p in (report, protocol))):
        raise ValueError('Unsupported controlled-recovery receipts.')
    baseline = candidate(evaluation, 'baseline4')
    if (protocol.get('benchmark_report_sha256') != evaluation['report_sha256'] or
            protocol.get('benchmark_protocol_sha256') != evaluation['protocol_sha256'] or
            protocol.get('runtime') != evaluation['protocol']['runtime'] or
            protocol.get('capsule_file_sha256') != baseline['capsule_file_sha256'] or
            protocol.get('cases') != evaluation['protocol']['scenes']):
        raise ValueError('Recovery changed its source cases, model or runtime.')
    cases = {case['id']: case for case in protocol['cases']}
    if len(cases) != 16 or 'scene-15' not in cases:
        raise ValueError('Expected the original 16-case recovery suite.')
    settings = protocol.get('settings')
    required = {'lower_goal_cap_m': .003, 'lower_timeout_seconds': 4., 'support_samples': 5,
                'support_height_tolerance_m': .008, 'support_linear_speed_m_s': .035,
                'support_angular_speed_rad_s': .3, 'horizon_seconds': 18.,
                'force_caps': 'each original case unchanged'}
    if type(settings) is not dict or any(settings.get(k) != v for k, v in required.items()):
        raise ValueError('Recovery intervention or original limits changed.')
    revision = protocol.get('recovery_revision')
    if revision not in (None, 'supported-release-v2') or report.get('recovery_revision') != revision:
        raise ValueError('Unknown recovery revision.')
    files = ['controlled_recovery_policy.py', 'grip_diagnostic.py']
    if revision == 'supported-release-v2':
        if (settings.get('clearance_gap_m') != .065 or settings.get('clearance_stable_samples') != 5 or
                settings.get('clearance_budget') != 'shared four-second recovery budget from initial trigger; no new time allowance'):
            raise ValueError('Release clearance changed its frozen conditions.')
        files += ['controlled_recovery_release_policy.py', 'controlled_recovery_release_experiment.py']
        _hashes({key: protocol.get(key) for key in ('parent_report_sha256', 'parent_protocol_sha256')},
                ('parent_report_sha256', 'parent_protocol_sha256'))
    else:
        files += ['controlled_recovery_experiment.py']
    runtime_sources = protocol['runtime']['files_sha256']
    _hashes(protocol.get('source_sha256'), [*runtime_sources, *files])
    if any(protocol['source_sha256'][k] != v for k, v in runtime_sources.items()):
        raise ValueError('Recovery runtime source hashes changed.')
    _hashes(protocol.get('original_run_sha256'), [*['student-' + key for key in cases], 'teacher-scene-15'])
    expected = {(key, 'student', variant) for key in cases for variant in ('baseline', 'controlled_lowering')}
    expected |= {('scene-15', 'teacher', variant) for variant in ('baseline', 'controlled_lowering')}
    attempts = protocol.get('attempts')
    if (type(attempts) is not list or len(attempts) != 34 or
            any(type(a) is not dict or set(a) != {'case_id', 'policy', 'variant'} for a in attempts) or
            any(type(a[k]) is not str for a in attempts for k in a) or
            {(a['case_id'], a['policy'], a['variant']) for a in attempts} != expected or
            protocol.get('planned_native_episodes') != 68 or
            protocol.get('variants') != ['baseline', 'controlled_lowering']):
        raise ValueError('Recovery comparison is incomplete or changed.')
    rows = report.get('results')
    if type(rows) is not list or len(rows) != len(expected):
        raise ValueError('Every planned recovery attempt must remain present.')
    outcomes, completed, errors, controls = {}, 0, 0, 0
    originals = {('student', r['case_id']): r for r in baseline['outcomes']}
    originals.update({('teacher', r['case_id']): r for r in evaluation['report']['teacher']['outcomes']})
    for row in rows:
        if type(row) is not dict or any(type(row.get(k)) is not str for k in ('case_id', 'policy', 'variant')):
            raise ValueError('Invalid recovery outcome identity.')
        key = (row['case_id'], row['policy'], row['variant'])
        if key not in expected or key in outcomes:
            raise ValueError('Unexpected or duplicate recovery outcome.')
        case = cases[key[0]]
        if (row.get('group') != case['group'] or row.get('scene_sha256') != case['scene_sha256'] or
                row.get('horizon') != 18):
            raise ValueError('Recovery outcome changed its scene or deadline.')
        parsed = {}
        for status, metric in (('status', 'measurement'), ('control_status', 'control')):
            if row.get(status) == 'completed':
                parsed[metric] = _measurement(row.get(metric))
                _count(parsed[metric]['recovery_count'])
                steps = _count(row[metric].get('control_steps'))
                if steps != math.ceil(parsed[metric]['simulated_seconds'] / .02 - 1e-7):
                    raise ValueError('Recovery control steps differ from the measured duration.')
                parsed[metric]['control_steps'] = steps
                if parsed[metric]['simulated_seconds'] > 18.000001:
                    raise ValueError('Recovery ran beyond the original 18-second deadline.')
                parsed[metric].update(_recovery(row[metric].get('recovery'), parsed[metric], case,
                    clearance_expected=revision == 'supported-release-v2' and key[2] == 'controlled_lowering' and metric == 'measurement'))
                completed += 1
            elif row.get(status) in ('error', 'skipped'):
                if type(row.get(metric)) is dict and row[metric].get('success') is True:
                    raise ValueError('Incomplete recovery cannot count as success.')
                parsed[metric] = None
                errors += 1
            else:
                raise ValueError('Unknown recovery execution status.')
        measured = parsed['measurement']
        if measured is not None:
            if key[2] == 'baseline':
                original = originals[(key[1], key[0])]
                field = 'student_' if key[1] == 'student' else 'teacher_'
                if (row.get('exact_original_trajectory_reproduced') is not True or
                        measured['success'] != original[field + 'success' if key[1] == 'student' else 'success'] or
                        measured['simulated_seconds'] != original[field + 'simulated_seconds'] or
                        measured['final_target_error_m'] != original[field + 'final_target_error_m']):
                    raise ValueError('Recovery baseline does not reproduce its original result.')
            elif row.get('identical_before_recovery') is not True:
                raise ValueError('Candidate changed motion before the recovery intervention.')
        controls += int(parsed['control'] is not None and parsed['control']['success'])
        outcomes[key] = parsed
    if (report.get('completed_native_episodes') != completed or report.get('episode_errors') != errors):
        raise ValueError('Recovery totals disagree with retained attempts.')
    regression = {'scenes': len(cases)}
    for group in ('nominal', 'actuator_stress'):
        counts = dict(cases=0, baseline_successes=0, candidate_successes=0, improved=0, regressed=0, unchanged=0, incomplete=0)
        for key, case in cases.items():
            if case['group'] != group:
                continue
            a, b = (outcomes[(key, 'student', variant)] for variant in ('baseline', 'controlled_lowering'))
            counts['cases'] += 1
            counts['baseline_successes'] += int(a['measurement'] is not None and a['measurement']['success'])
            counts['candidate_successes'] += int(b['measurement'] is not None and b['measurement']['success'])
            if any(p[m] is None or (m == 'control' and p[m]['success']) for p in (a, b) for m in ('measurement', 'control')):
                counts['incomplete'] += 1
            else:
                sa, sb = a['measurement']['success'], b['measurement']['success']
                counts['improved' if sb and not sa else 'regressed' if sa and not sb else 'unchanged'] += 1
        regression[group] = counts
    target_rows = []
    for policy in ('student', 'teacher'):
        for variant in ('baseline', 'controlled_lowering'):
            measured = outcomes[('scene-15', policy, variant)]['measurement']
            target_rows.append({'policy': policy, 'variant': variant,
                'label': ('Learned' if policy == 'student' else 'Reference') + (' · original' if variant == 'baseline' else ' · lowering'),
                **(measured or {k: None for k in ('success', 'simulated_seconds', 'recovery_count', 'support_confirmed_before_open', 'unsupported_release_actions')})})
    scene = cases['scene-15']['scene']
    return {'report_sha256': report_hash, 'protocol_sha256': protocol_hash,
            'recovery_revision': revision, 'parent_report_sha256': protocol.get('parent_report_sha256'),
            'parent_protocol_sha256': protocol.get('parent_protocol_sha256'),
            'completed_native_episodes': completed, 'episode_errors': errors, 'control_successes': controls,
            'elapsed_seconds': _number(report.get('elapsed_seconds'), 'recovery elapsed time'),
            'deadline_seconds': 18, 'regression': regression,
            'target': {'case_id': 'scene-15', 'condition': f"{scene['jaw_force_limit']:g} N jaws · {scene['joint_torque_limit']:g} N·m joints · 18 s", 'rows': target_rows},
            'limits': ['Original force limits per case and the 18-second deadline are unchanged.',
                       'Inspected simulation scenes measure regressions, not fresh generalization.',
                       'Support before opening does not prove continuous retention or low impact.',
                       'Release commands and contact observations are sampled at 50 Hz.',
                       'Historical qualification and physical readiness remain unchanged.']}


def attach_recovery(dashboard, studies):
    """Show the latest intervention without rewriting historical qualification."""
    studies = [studies] if type(studies) is dict else studies
    if not studies:
        return dashboard
    if (len({s['report_sha256'] for s in studies}) != len(studies) or
            len({s['protocol_sha256'] for s in studies}) != len(studies)):
        raise ValueError('Duplicate recovery studies cannot inflate the evidence.')
    for index, study in enumerate(studies):
        if study.get('recovery_revision') == 'supported-release-v2':
            parents = [s for s in studies[:index] if s['report_sha256'] == study['parent_report_sha256']]
            if len(parents) != 1 or parents[0]['protocol_sha256'] != study['parent_protocol_sha256']:
                raise ValueError('Recovery follow-up needs its preceding measured parent.')
    view = deepcopy(dashboard)
    latest = deepcopy(studies[-1])
    latest.update(previous_studies=deepcopy(studies[:-1]),
                  elapsed_seconds=sum(s['elapsed_seconds'] for s in studies),
                  completed_native_episodes=sum(s['completed_native_episodes'] for s in studies),
                  episode_errors=sum(s['episode_errors'] for s in studies),
                  control_successes=sum(s['control_successes'] for s in studies))
    regression = latest['regression']
    regressions = sum(regression[group]['regressed'] for group in ('nominal', 'actuator_stress'))
    incomplete = sum(regression[group]['incomplete'] for group in ('nominal', 'actuator_stress'))
    improved = sum(regression[group]['improved'] for group in ('nominal', 'actuator_stress'))
    candidate = next(row for row in latest['target']['rows']
                     if row['policy'] == 'student' and row['variant'] == 'controlled_lowering')
    bad = latest['episode_errors'] or latest['control_successes'] or incomplete
    if bad:
        headline = 'Resolve incomplete runs or control failures.'
        detail = 'The recovery experiment cannot support an improvement claim until every attempt and control validates.'
        title = 'Check the experiment evidence'
        instruction = 'Inspect errors and paired controls before changing the controller or collecting more demonstrations.'
    elif regressions:
        headline = 'The recovery change regressed existing cases.'
        detail = f'{regressions} previously completed cases now fail. Keep the original controller.'
        title = 'Investigate the regressions'
        instruction = 'Compare the first divergent observations and commands in each regression. Keep the original force limits, deadline and completion checks.'
    elif improved:
        headline = 'More inspected cases complete; confirmation is still required.'
        detail = f'{improved} paired cases improved with no completion regressions in this inspected suite. Historical qualification is unchanged.'
        title = 'Confirm on unseen conditions'
        instruction = 'Freeze a fresh paired evaluation before making a general improvement claim. Include recovery behavior, negative controls and all failed attempts.'
    elif candidate.get('release_clearance_confirmed') is True:
        headline = 'Release clearance improved; the task still misses its deadline.'
        detail = 'The learned controller confirms settled support and open, contact-free jaws before withdrawing. The inspected suite shows no task-completion gain or regression; physical performance remains untested.'
        title = 'Improve retention before recovery'
        instruction = 'Measure where loaded motion accumulates object drift and spends the remaining time. Freeze one earlier retention intervention before comparing; preserve original force limits, the 18-second deadline and sustained-lift checks.'
    elif candidate['support_confirmed_before_open'] is True:
        headline = 'First opening waited for support; the task still misses its deadline.'
        detail = 'Support was confirmed before the learned controller first opened its jaws. This does not prove continuous retention, a completed physical release or full placement success within 18 seconds.'
        title = 'Improve retention before recovery'
        instruction = 'Measure where loaded motion accumulates object drift and spends the remaining time. Freeze one earlier retention intervention before comparing; preserve the original force limits, 18-second deadline and sustained-lift checks.'
    else:
        headline = 'Controlled lowering has not resolved the failure.'
        detail = 'Keep the original task result. Inspect whether the grasp survives lowering and whether support is reached before the deadline.'
        title = 'Inspect the lowering failure'
        instruction = 'Separate loss of grip during lowering, support detection and recovery time. Freeze a mechanism-specific change before another paired comparison at the same limits.'
    latest.update(headline=headline, detail=detail, next_step=instruction)
    view['recovery'] = latest
    view['previous_hypotheses'].extend(view['hypotheses'])
    view['hypotheses'] = [{
        'id': 'recovery-next', 'category': 'diagnostic', 'title': title,
        'status': 'provisional', 'explanation': detail, 'next_action': instruction,
        'diagnostic_report_sha256': latest['report_sha256'],
        'request': {'title': title, 'instruction': instruction, 'why': detail,
                    'capture_mode': 'diagnostic', 'required_groups': None,
                    'acceptance': ['Freeze the intervention and comparison before execution.',
                                   'Retain original actuator limits, the 18-second deadline and all native success checks.',
                                   'Report supported release, task completion and regressions separately.']},
        'evidence': []}]
    view['next_hypothesis_id'] = 'recovery-next'
    return view
