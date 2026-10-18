"""Validate and project bounded retention experiments without running simulation."""
from copy import deepcopy
import math
from pathlib import Path

from .grip_diagnostic_view import _measurement
from .improvement import _number, _read, candidate
from .recovery_view import _count, _hashes, _recovery


def _text(value, limit=200):
    if type(value) is not str or not value or len(value) > limit:
        raise ValueError('Expected bounded retention evidence text.')
    return value


def _time(value, duration):
    if not 0 <= _number(value, 'observation time') <= duration + 1e-8:
        raise ValueError('Observation is outside the measured episode.')
    return value


def _retention(value, duration, *, lift_seen, instrumented):
    if type(value) is not dict or value.get('curve_cadence_seconds') != .1:
        raise ValueError('Expected sampled retention observations.')
    first = value.get('first_intervention_time')
    if first is not None:
        _time(first, duration)
    maximum = _number(value.get('maximum_pre_recovery_loaded_offset_m'), 'grip offset')
    if maximum > 10:
        raise ValueError('Retention offset exceeds the bounded workspace.')
    events, curve = value.get('events'), value.get('curve')
    if type(events) is not list or len(events) > 100 or type(curve) is not list or len(curve) > 181:
        raise ValueError('Retention observations exceed the episode budget.')
    previous, seen = -1., set()
    phases = {'approach', 'descend', 'close', 'lift', 'transfer', 'place', 'release', 'retreat', 'verify',
              'recover_wait', 'recover_up', 'failed'}
    for event in events:
        if type(event) is not dict:
            raise ValueError('Invalid retention event.')
        for key in ('id', 'basis', 'stage'):
            _text(event.get(key))
        when = _time(event.get('time'), duration)
        frame = _time(event.get('frame_time'), duration)
        if when < previous or not 0 <= frame - when <= .02000001:
            raise ValueError('Retention event order or time basis is inconsistent.')
        previous = when
        identity = event['id']
        if event['stage'] not in phases:
            raise ValueError('Unknown retention event phase.')
        if identity.startswith('phase:'):
            if identity != 'phase:' + event['stage'] or event['basis'] != 'pre-action observation at first command of phase':
                raise ValueError('Phase event is mislabeled.')
        elif identity in {'offset-2mm', 'offset-6mm', 'offset-12mm'}:
            threshold = int(identity[7:-2]) / 1000
            error = _number(event.get('relative_offset_error_m'), 'threshold offset')
            contacts = event.get('contacts')
            if (identity in seen or event['stage'] not in ('lift', 'transfer') or
                    not threshold < error <= maximum + 1e-10 or
                    event['basis'] != 'first sampled pre-action offset above threshold' or
                    type(contacts) is not dict or set(contacts) != {'left', 'right'}):
                raise ValueError('Threshold event contradicts its measurements.')
            for count in contacts.values():
                _count(count)
            seen.add(identity)
        elif identity == 'native-lift-seen':
            if (identity in seen or not lift_seen or when != frame or
                    event['basis'] != 'first post-control observation with native lift predicate true'):
                raise ValueError('Native lift event contradicts the episode outcome.')
            seen.add(identity)
        else:
            raise ValueError('Unknown retention event kind.')
    if instrumented and ('native-lift-seen' in seen) is not lift_seen:
        raise ValueError('Native lift event is missing from instrumented observations.')
    if not instrumented and (events or curve or first is not None):
        raise ValueError('Replay control cannot invent controller observations.')
    previous = -1.
    for sample in curve:
        if type(sample) is not dict or sample.get('stage') not in ('lift', 'transfer', 'place'):
            raise ValueError('Unexpected retention curve phase.')
        when = _time(sample.get('time'), duration)
        if when <= previous or abs(when / .1 - round(when / .1)) > 1e-6:
            raise ValueError('Retention curve is not a monotonic 10 Hz sample.')
        previous = when
        offset = sample.get('relative_offset_error_m')
        if offset is not None and not 0 <= _number(offset, 'sampled offset') <= 10:
            raise ValueError('Invalid sampled grip offset.')
        contacts = sample.get('contacts')
        if type(contacts) is not dict or set(contacts) != {'left', 'right'}:
            raise ValueError('Missing bilateral contact counts.')
        for count in (*contacts.values(), sample.get('table_contacts')):
            _count(count)
        for key in ('object_z_m', 'tcp_z_m'):
            if type(sample.get(key)) not in (int, float) or not math.isfinite(sample[key]) or abs(sample[key]) > 10:
                raise ValueError('Invalid measured height.')
    extra = {}
    for key in ('minimum_sampled_transfer_object_clearance_m', 'maximum_sampled_transfer_tcp_descent_m'):
        measured = value.get(key)
        if measured is not None and not 0 <= _number(measured, key) <= 10:
            raise ValueError('Invalid sampled transfer path measurement.')
        extra[key] = measured
    return {**extra, 'first_intervention_time': first,
            'maximum_pre_recovery_loaded_offset_m': maximum,
            'events': deepcopy(events), 'curve': deepcopy(curve), 'curve_cadence_seconds': .1,
            'curve_scope': '10 Hz subset of 50 Hz observations during loaded phases, including any retry. Object-to-hand offset is measured in the world frame.'}


def load_retention(directory, evaluation):
    root = Path(directory).absolute()
    protocol, protocol_hash = _read(root / 'protocol.json')
    report, report_hash = _read(root / 'report.json')
    if (protocol.get('kind') != 'dvidia.retention-experiment' or report.get('kind') != 'dvidia.retention-report' or
            report.get('protocol_sha256') != protocol_hash or
            any(p.get('schema_version') != 1 or p.get('scope') != 'simulation-only' or
                p.get('physical_robot_ready') is not False for p in (protocol, report))):
        raise ValueError('Unsupported retention experiment receipts.')
    revision = protocol.get('retention_revision')
    if revision not in ('increment-smoothing-v1', 'vertical-compliance-v1') or report.get('retention_revision') != revision:
        raise ValueError('Unknown frozen retention intervention.')
    compliance = revision == 'vertical-compliance-v1'
    variant = 'vertical_compliance' if compliance else 'loaded_goal_smoothing'
    base = candidate(evaluation, 'baseline4')
    if (protocol.get('benchmark_report_sha256') != evaluation['report_sha256'] or
            protocol.get('benchmark_protocol_sha256') != evaluation['protocol_sha256'] or
            protocol.get('runtime') != evaluation['protocol']['runtime'] or
            protocol.get('capsule_file_sha256') != base['capsule_file_sha256'] or
            protocol.get('cases') != evaluation['protocol']['scenes']):
        raise ValueError('Retention changed its source, force limits, model or runtime.')
    cases = {row['id']: row for row in protocol['cases']}
    if set(cases) != {f'scene-{i:02}' for i in range(16)}:
        raise ValueError('Expected all original 16 regression cases.')
    settings = protocol.get('settings')
    expected = {'low_headroom_margin': 1.25,
                'force_caps': 'each original case unchanged', 'horizon_seconds': 18.,
                'phase_budgets': 'all original phase budgets unchanged',
                'movement_mapping': 'same frozen student head / authored teacher IK; no fallback',
                'recovery': 'original recovery unchanged to isolate the pre-recovery intervention'}
    expected.update({'phases': ['transfer'], 'vertical_error_gain': .5,
                     'vertical_increment_cap_m': .002, 'total_increment_cap_m': .005,
                     'maximum_downward_excursion_m': .025, 'minimum_object_clearance_m': .075,
                     'transfer_planar_tolerance_m': .012,
                     'offset_reference': 'original held offset is never reset or relaxed'} if compliance else
                    {'smoothing_alpha': .2, 'phases': ['lift', 'transfer'], 'control_dt_seconds': .02})
    if type(settings) is not dict or any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen retention settings or limits changed.')
    runtime = protocol['runtime']['files_sha256']
    sources = (['retention_compliance_policy.py', 'retention_compliance_experiment.py'] if compliance else
               ['retention_policy.py', 'retention_experiment.py']) + ['grip_diagnostic.py',
               'controlled_recovery_release_experiment.py', 'controlled_recovery_policy.py',
               'controlled_recovery_release_policy.py']
    _hashes(protocol.get('source_sha256'), [*runtime, *sources])
    if compliance:
        _hashes({key: protocol.get(key) for key in ('parent_report_sha256', 'parent_protocol_sha256')},
                ('parent_report_sha256', 'parent_protocol_sha256'))
    if any(protocol['source_sha256'][key] != value for key, value in runtime.items()):
        raise ValueError('Retention runtime source bindings changed.')
    _hashes(protocol.get('original_run_sha256'), [*['student-' + key for key in cases], 'teacher-scene-15'])
    expected_attempts = {(key, 'student', v) for key in cases for v in ('baseline', variant)}
    expected_attempts |= {('scene-15', 'teacher', v) for v in ('baseline', variant)}
    attempts = protocol.get('attempts')
    if (type(attempts) is not list or len(attempts) != 34 or
            any(type(a) is not dict or set(a) != {'case_id', 'policy', 'variant'} or
                any(type(v) is not str for v in a.values()) for a in attempts) or
            {(a['case_id'], a['policy'], a['variant']) for a in attempts} != expected_attempts or
            protocol.get('variants') != ['baseline', variant] or protocol.get('planned_native_episodes') != 68):
        raise ValueError('Retention attempts differ from the complete paired suite.')
    rows = report.get('results')
    if type(rows) is not list or len(rows) != 34:
        raise ValueError('Every retention attempt must remain present.')
    originals = {('student', r['case_id']): r for r in base['outcomes']}
    originals.update({('teacher', r['case_id']): r for r in evaluation['report']['teacher']['outcomes']})
    outcomes, completed, errors, controls = {}, 0, 0, 0
    for row in rows:
        if type(row) is not dict or any(type(row.get(k)) is not str for k in ('case_id', 'policy', 'variant')):
            raise ValueError('Invalid retention attempt identity.')
        key = row['case_id'], row['policy'], row['variant']
        if key not in expected_attempts or key in outcomes:
            raise ValueError('Unknown or duplicate retention attempt.')
        case = cases[key[0]]
        if row.get('group') != case['group'] or row.get('scene_sha256') != case['scene_sha256'] or row.get('horizon') != 18:
            raise ValueError('Retention outcome changed its case or deadline.')
        parsed = {}
        for status, metric in (('status', 'measurement'), ('control_status', 'control')):
            if row.get(status) == 'completed':
                raw = row.get(metric)
                measured = _measurement(raw)
                _count(measured['recovery_count'])
                steps = _count(raw.get('control_steps'))
                if (measured['simulated_seconds'] > 18.000001 or
                        steps != math.ceil(measured['simulated_seconds'] / .02 - 1e-7) or type(raw.get('lift_seen')) is not bool):
                    raise ValueError('Retention timing or native lift observation changed.')
                measured.update(control_steps=steps, lift_seen=raw['lift_seen'])
                measured.update(_recovery(raw.get('recovery'), measured, case))
                measured['retention'] = _retention(raw.get('retention'), measured['simulated_seconds'],
                                                  lift_seen=measured['lift_seen'], instrumented=metric == 'measurement')
                calls = _count(raw.get('intervention_calls'))
                if (calls > steps or (calls > 0) != (measured['retention']['first_intervention_time'] is not None) or
                        ((key[2] == 'baseline' or metric == 'control') and calls != 0)):
                    raise ValueError('Intervention timing contradicts its command count.')
                measured['intervention_calls'] = calls
                parsed[metric] = measured
                completed += 1
            elif row.get(status) in ('error', 'skipped'):
                if type(row.get(metric)) is dict and row[metric].get('success') is True:
                    raise ValueError('Incomplete attempt cannot count as success.')
                parsed[metric] = None
                errors += 1
            else:
                raise ValueError('Unknown retention execution status.')
        measured = parsed['measurement']
        if measured is not None:
            if key[2] == 'baseline':
                original = originals[(key[1], key[0])]
                prefix = 'student_' if key[1] == 'student' else 'teacher_'
                if (row.get('exact_original_trajectory_reproduced') is not True or
                        measured['success'] != original['student_success' if key[1] == 'student' else 'success'] or
                        measured['simulated_seconds'] != original[prefix + 'simulated_seconds'] or
                        measured['final_target_error_m'] != original[prefix + 'final_target_error_m'] or
                        measured['retention']['first_intervention_time'] is not None):
                    raise ValueError('Retention baseline does not reproduce the original episode.')
            elif row.get('identical_before_intervention') is not True:
                raise ValueError('Candidate changed commands before the declared intervention.')
        controls += int(parsed['control'] is not None and parsed['control']['success'])
        outcomes[key] = parsed
    if report.get('completed_native_episodes') != completed or report.get('episode_errors') != errors:
        raise ValueError('Retention totals disagree with retained attempts.')
    regression = {'scenes': len(cases)}
    for group in ('nominal', 'actuator_stress'):
        counts = dict(cases=0, baseline_successes=0, candidate_successes=0, improved=0, regressed=0, unchanged=0, incomplete=0)
        for key, case in cases.items():
            if case['group'] != group:
                continue
            a, b = (outcomes[(key, 'student', v)] for v in ('baseline', variant))
            counts['cases'] += 1
            for label, result in (('baseline_successes', a), ('candidate_successes', b)):
                counts[label] += int(result['measurement'] is not None and result['measurement']['success'])
            if any(p[m] is None or (m == 'control' and p[m]['success']) for p in (a, b) for m in ('measurement', 'control')):
                counts['incomplete'] += 1
            else:
                sa, sb = a['measurement']['success'], b['measurement']['success']
                counts['improved' if sb and not sa else 'regressed' if sa and not sb else 'unchanged'] += 1
        regression[group] = counts
    target = []
    for policy in ('student', 'teacher'):
        for v in ('baseline', variant):
            target.append({'policy': policy, 'variant': v,
                           'label': ('Learned' if policy == 'student' else 'Reference') + (' · original' if v == 'baseline' else ' · compliance' if compliance else ' · smoothing'),
                           **(outcomes[('scene-15', policy, v)]['measurement'] or {'success': None, 'retention': None})})
    scene = cases['scene-15']['scene']
    return {'revision': revision, 'candidate_variant': variant,
            'retention_guard': {'offset_m': .012, 'consecutive_control_samples': 5},
            'report_sha256': report_hash, 'protocol_sha256': protocol_hash,
            'parent_report_sha256': protocol.get('parent_report_sha256'),
            'parent_protocol_sha256': protocol.get('parent_protocol_sha256'),
            'deadline_seconds': 18, 'completed_native_episodes': completed, 'episode_errors': errors,
            'control_successes': controls, 'elapsed_seconds': _number(report.get('elapsed_seconds'), 'retention wall time'),
            'regression': regression, 'promotion': 'not-promoted',
            'target': {'case_id': 'scene-15', 'condition': f"{scene['jaw_force_limit']:g} N jaws · {scene['joint_torque_limit']:g} N·m joints · 18 s", 'rows': target},
            'limits': ['16 inspected simulation scenes; repeated episodes are not independent conditions.',
                       'Original force limits, 18-second deadline and native success checks retained.',
                       'The candidate retains original recovery to isolate retention; supported release is a separate experiment.',
                       'No physical robot test or automatic promotion.']}


def attach_retention(dashboard, studies):
    if not studies:
        return dashboard
    if len({s['report_sha256'] for s in studies}) != len(studies) or len({s['protocol_sha256'] for s in studies}) != len(studies):
        raise ValueError('Duplicate retention study cannot inflate evidence.')
    for index, study in enumerate(studies):
        if study['revision'] == 'vertical-compliance-v1':
            parents = [s for s in studies[:index] if s['report_sha256'] == study['parent_report_sha256']]
            if len(parents) != 1 or parents[0]['protocol_sha256'] != study['parent_protocol_sha256']:
                raise ValueError('Retention follow-up requires its measured parent.')
    view, latest = deepcopy(dashboard), deepcopy(studies[-1])
    latest['previous_studies'] = deepcopy(studies[:-1])
    latest['study_count'] = len(studies)
    latest['total_completed_native_episodes'] = sum(s['completed_native_episodes'] for s in studies)
    latest['total_elapsed_seconds'] = sum(s['elapsed_seconds'] for s in studies)
    groups = [latest['regression'][g] for g in ('nominal', 'actuator_stress')]
    gains, regressions, incomplete = (sum(g[k] for g in groups) for k in ('improved', 'regressed', 'incomplete'))
    latest['episode_errors'] = sum(s['episode_errors'] for s in studies)
    latest['control_successes'] = sum(s['control_successes'] for s in studies)
    bad = latest['episode_errors'] or latest['control_successes'] or incomplete
    if bad:
        headline, detail = 'Retention evidence needs review.', 'Resolve incomplete runs or successful negative controls before interpreting this experiment.'
        title = 'Inspect the experiment evidence'
    elif regressions:
        headline, detail = 'The retention change regressed existing cases.', f'{regressions} previously completed cases now fail. Keep the original controller.'
        title = 'Investigate the regressions'
    elif gains:
        headline, detail = 'More inspected cases complete. Confirmation comes next.', f'{gains} cases improved without completion regressions. New conditions and composition with supported release still need testing.'
        title = 'Confirm retention on new conditions'
    else:
        headline = 'Compliant transfer did not complete the task.' if latest['revision'] == 'vertical-compliance-v1' else 'Smoother motion did not resolve retention.'
        detail = 'Task completion is unchanged. Smaller object-to-hand drift or a later recovery is not a successful placement or proof of retained grip.'
        title = 'Investigate steady loaded drift'
        before = next(row for row in latest['target']['rows'] if row['policy'] == 'student' and row['variant'] == 'baseline')
        after = next(row for row in latest['target']['rows'] if row['policy'] == 'student' and row['variant'] == latest['candidate_variant'])
        if before.get('lift_seen') is False and after.get('lift_seen') is True and after.get('success') is False:
            headline = 'Lift improved. Placement still fails.'
            detail = 'The learned controller now meets the native sustained-lift check, but still misses the original 18-second placement deadline. The compliant hand movement does not establish a retained grasp throughout transfer.'
            title = 'Investigate contact during loaded transfer'
    latest.update(headline=headline, detail=detail, next_step=title)
    view['retention'] = latest
    view['previous_hypotheses'].extend(view['hypotheses'])
    view['hypotheses'] = [{'id': 'retention-next', 'category': 'diagnostic', 'title': title,
                          'diagnostic_report_sha256': latest['report_sha256'],
                          'status': 'provisional', 'explanation': detail, 'next_action': title,
                          'request': {'title': title, 'instruction': detail, 'why': detail, 'capture_mode': 'diagnostic',
                                      'required_groups': None, 'acceptance': ['Preserve original forces, deadline and native completion checks.',
                                          'Freeze the next intervention before comparing all original cases and controls.',
                                          'Confirm new conditions and supported-release composition before any promotion.']}, 'evidence': []}]
    view['next_hypothesis_id'] = 'retention-next'
    return view
