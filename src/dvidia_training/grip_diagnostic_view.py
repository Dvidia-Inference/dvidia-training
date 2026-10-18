"""Read measured grip diagnostics without importing or running the simulator."""
from copy import deepcopy
from pathlib import Path

from .improvement import _identity, _number, _read, candidate


def _measurement(value):
    if type(value) is not dict or type(value.get('success')) is not bool:
        raise ValueError('Completed diagnostic episodes need explicit outcomes.')
    for key in ('simulated_seconds', 'episode_wall_seconds', 'recovery_count', 'final_target_error_m'):
        _number(value.get(key), key)
    if type(value.get('reason')) is not str or len(value['reason']) > 200:
        raise ValueError('Diagnostic reason must be bounded text.')
    if value.get('valid_final_state') is not True and value['success']:
        raise ValueError('Invalid final state cannot be a success.')
    if value['success']:
        # This bounded adapter retains the source placement task's 25 mm,
        # 250 ms completion rule and required native sustained-lift check.
        dwell = _number(value.get('placement_dwell_seconds'), 'placement dwell')
        if (value['reason'] != 'success' or value.get('lift_seen') is not True or
                dwell < .25 - 1e-9 or value['final_target_error_m'] > .025):
            raise ValueError('Diagnostic success contradicts the retained placement checks.')
    return {key: value[key] for key in ('success', 'reason', 'simulated_seconds',
                                      'recovery_count', 'final_target_error_m')}


def load_diagnostic(directory, evaluation):
    """Bind complete diagnostic rows to one frozen experiment and source case."""
    root = Path(directory).absolute()
    report, report_hash = _read(root / 'report.json')
    protocol, protocol_hash = _read(root / 'protocol.json')
    if (report.get('kind') != 'dvidia.grip-recovery-diagnostic-result' or
            protocol.get('kind') != 'dvidia.grip-recovery-diagnostic' or
            report.get('protocol_sha256') != protocol_hash or
            any(p.get('schema_version') != 1 or p.get('scope') != 'simulation-only' or
                p.get('physical_robot_ready') is not False for p in (report, protocol))):
        raise ValueError('Unsupported or mismatched diagnostic receipts.')
    if (protocol.get('benchmark_report_sha256') != evaluation['report_sha256'] or
            protocol.get('benchmark_protocol_sha256') != evaluation['protocol_sha256']):
        raise ValueError('Diagnostic is not bound to the displayed benchmark.')
    if protocol.get('runtime') != evaluation['protocol'].get('runtime'):
        raise ValueError('Diagnostic runtime differs from the source benchmark.')
    if ('parent_report_sha256' in report and
            report['parent_report_sha256'] != protocol.get('parent_report_sha256')):
        raise ValueError('Diagnostic follow-up parent bindings differ.')
    baseline = candidate(evaluation, 'baseline4')
    if protocol.get('capsule_file_sha256') != baseline.get('capsule_file_sha256'):
        raise ValueError('Diagnostic changed the frozen movement candidate.')
    case = protocol.get('case')
    if type(case) is not dict or case not in evaluation['protocol']['scenes']:
        raise ValueError('Diagnostic source case differs from the evaluated case.')
    arms, policies = protocol.get('arms'), protocol.get('policies')
    if (type(arms) is not list or not 1 <= len(arms) <= 12 or
            policies != ['student', 'teacher']):
        raise ValueError('Expected bounded diagnostic arms and paired policies.')
    specs = {}
    for arm in arms:
        if type(arm) is not dict:
            raise ValueError('Expected diagnostic arm metadata.')
        identity = _identity(arm.get('id'))
        if identity in specs:
            raise ValueError('Duplicate diagnostic arm.')
        overrides = arm.get('scene_overrides')
        if type(overrides) is not dict or set(overrides) - {'jaw_force_limit', 'joint_torque_limit'}:
            raise ValueError('Unsupported diagnostic scene intervention.')
        for key, value in overrides.items():
            _number(value, key)
            low, high = (.01, 15.) if key == 'jaw_force_limit' else (.1, 40.)
            if not low <= value <= high:
                raise ValueError('Diagnostic actuator limit is outside the supported range.')
        horizon = _number(arm.get('horizon'), 'horizon')
        if not 1 <= horizon <= 60:
            raise ValueError('Diagnostic horizon exceeds the supported budget.')
        cap = arm.get('loaded_goal_cap_m')
        if cap is not None and not .0005 <= _number(cap, 'loaded goal cap') <= .006:
            raise ValueError('Invalid loaded goal cap.')
        specs[identity] = arm
    if protocol.get('planned_native_episodes') != len(specs) * 4:
        raise ValueError('Diagnostic episode budget is inconsistent.')
    rows = report.get('results')
    if type(rows) is not list or len(rows) != len(specs) * 2:
        raise ValueError('Every planned policy and arm must retain an outcome.')
    outcomes, completed, errors = {}, 0, 0
    for row in rows:
        if type(row) is not dict:
            raise ValueError('Expected diagnostic outcome.')
        arm_id, policy = row.get('arm_id'), row.get('policy')
        if type(arm_id) is not str or type(policy) is not str:
            raise ValueError('Invalid diagnostic outcome identity.')
        key = (arm_id, policy)
        if arm_id not in specs or policy not in policies or key in outcomes:
            raise ValueError('Unknown or repeated diagnostic outcome.')
        arm = specs[arm_id]
        if (row.get('qualification') != 'diagnostic-only' or
                row.get('scene') != {**case['scene'], **arm['scene_overrides']} or
                any(row.get(k) != arm[k] for k in ('horizon', 'loaded_goal_cap_m'))):
            raise ValueError('Diagnostic outcome changed its frozen conditions.')
        parsed = {}
        for status_key, measure_key in (('status', 'measurement'), ('control_status', 'control')):
            status = row.get(status_key)
            if status == 'completed':
                parsed[measure_key] = _measurement(row.get(measure_key))
                if parsed[measure_key]['simulated_seconds'] > arm['horizon'] + .05:
                    raise ValueError('Diagnostic episode exceeds its frozen deadline.')
                completed += 1
            elif status in ('error', 'skipped'):
                if type(row.get(measure_key)) is dict and row[measure_key].get('success') is True:
                    raise ValueError('An incomplete diagnostic cannot count as success.')
                parsed[measure_key] = None
                errors += 1
            else:
                raise ValueError('Unknown diagnostic execution status.')
        original_conditions = (row['scene'] == case['scene'] and arm['horizon'] == 18 and
                               arm['loaded_goal_cap_m'] is None)
        if original_conditions and row.get('exact_original_trajectory_reproduced') is not True:
            raise ValueError('Original diagnostic must reproduce the prior trajectory.')
        outcomes[key] = parsed
    if report.get('completed_native_episodes') != completed or report.get('episode_errors') != errors:
        raise ValueError('Diagnostic totals differ from the retained attempts.')
    _number(report.get('elapsed_seconds'), 'diagnostic elapsed time')
    view_rows = []
    original = case['scene']
    for identity, arm in specs.items():
        changes = []
        for key, unit, label in (('jaw_force_limit', 'N', 'Grip limit'),
                                 ('joint_torque_limit', 'N·m', 'Arm torque')):
            if key in arm['scene_overrides']:
                changes.append(f"{label} {arm['scene_overrides'][key]:g} {unit}")
        if arm['horizon'] != 18:
            changes.append(f"{arm['horizon']:g} s deadline")
        if arm['loaded_goal_cap_m'] is not None:
            changes.append(f"{arm['loaded_goal_cap_m'] * 1000:g} mm loaded goal cap")
        student, teacher = outcomes[(identity, 'student')], outcomes[(identity, 'teacher')]
        pair = (student, teacher)
        view_rows.append({'id': identity, 'label': ' · '.join(changes) or 'Original conditions',
                          'change': f"{arm['scene_overrides'].get('jaw_force_limit', original['jaw_force_limit']):g} N grip · {arm['horizon']:g} s · unchanged model",
                          'student': student['measurement'], 'teacher': teacher['measurement'],
                          'control_successes': sum(p['control'] is not None and p['control']['success'] for p in pair),
                          'errors': sum(p[k] is None for p in pair for k in ('measurement', 'control'))})
    return {'case_id': case['id'], 'rows': view_rows, 'elapsed_seconds': report['elapsed_seconds'],
            'completed_native_episodes': completed, 'episode_errors': errors,
            'report_sha256': report_hash, 'protocol_sha256': protocol_hash,
            'parent_report_sha256': protocol.get('parent_report_sha256'),
            'limits': ['Previously inspected simulation case; these results are diagnosis, not fresh qualification.',
                       'Higher grip limits change simulated capability and the controller response to it.',
                       'Longer deadlines do not repair the original 18-second benchmark failure.',
                       'Success criteria and model weights are unchanged. Physical performance remains untested.']}


def attach_diagnostics(dashboard, studies):
    """Advance the investigation without rewriting old qualification counts."""
    if not studies:
        return dashboard
    if (len({s['report_sha256'] for s in studies}) != len(studies) or
            len({s['protocol_sha256'] for s in studies}) != len(studies)):
        raise ValueError('Duplicate diagnostic studies cannot count twice.')
    for index, study in enumerate(studies):
        parent = study['parent_report_sha256']
        if parent is not None and parent not in {s['report_sha256'] for s in studies[:index]}:
            raise ValueError('Follow-up diagnostic needs its preceding measured study.')
    view = deepcopy(dashboard)
    latest = deepcopy(studies[-1])
    bad = any(row['errors'] or row['control_successes'] for s in studies for row in s['rows'])
    successful = [r for r in latest['rows'] if all(r[p] is not None and r[p]['success'] for p in ('student', 'teacher'))]
    if bad:
        headline = 'Resolve diagnostic errors or control failures first.'
        detail = 'At least one incomplete trial or successful open-jaw control prevents a clean interpretation.'
        title, instruction = 'Validate the diagnostic evidence', 'Inspect failed attempts and controls before changing the controller or requesting recordings.'
    elif successful:
        headline = 'The case completes under changed conditions.'
        detail = 'Both controllers completed with ' + successful[0]['label'] + '. The original benchmark remains unchanged.'
        title, instruction = 'Test a controlled recovery', 'Freeze a recovery that lowers the object to support before opening the jaws. Compare it at the original grip limit and deadline on regression and fresh scenes, preserving lift and placement checks.'
    else:
        headline = 'The tested changes have not resolved the failure.'
        detail = 'No tested condition completed for both controllers. Keep the original result and investigate retention and phase timing together.'
        title, instruction = 'Test retention and recovery together', 'Inspect pre-action object movement, contact and phase timeouts. Freeze a controlled set-down recovery before testing it at original capability; preserve all failure checks.'
    latest.update(headline=headline, detail=detail, next_step=instruction,
                  elapsed_seconds=sum(s['elapsed_seconds'] for s in studies),
                  completed_native_episodes=sum(s['completed_native_episodes'] for s in studies),
                  episode_errors=sum(s['episode_errors'] for s in studies),
                  control_successes=sum(r['control_successes'] for s in studies for r in s['rows']),
                  previous_studies=deepcopy(studies[:-1]))
    view['diagnostic'] = latest
    view['previous_hypotheses'].extend(view['hypotheses'])
    view['hypotheses'] = [{'id': 'recovery-followup', 'category': 'diagnostic', 'title': title,
                           'status': 'provisional', 'explanation': detail, 'next_action': instruction,
                           'diagnostic_report_sha256': latest['report_sha256'],
                           'request': {'title': title, 'instruction': instruction, 'why': detail,
                                       'capture_mode': 'diagnostic', 'required_groups': None,
                                       'acceptance': ['Freeze the controller and test suite before comparing.',
                                                      'Keep actuator limits, success checks and deadline explicit.',
                                                      'Retain every attempt; require regression checks and fresh confirmation.']},
                           'evidence': []}]
    view['next_hypothesis_id'] = 'recovery-followup'
    return view
