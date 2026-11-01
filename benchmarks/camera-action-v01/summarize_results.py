#!/usr/bin/env python3
"""Read-only posthoc tables from sealed camera/action study receipts.

This tool never imports a policy, simulator, trainer or the frozen study runtime.
It verifies evidence receipts and model bytes, then summarizes recorded native
scores and reconstructs terminal components numerically. It never reruns physics,
changes qualification gates or rehashes image archives. Missing evidence stays unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

SCHEMA = 'dvidia.camera-action-posthoc-summary/v1'
MAX_JSON_BYTES = 32 * 1024 * 1024
TIMING_FIELDS = ('estimator_wall_seconds', 'policy_inference_wall_seconds',
                 'full_boundary_wall_seconds', 'render_encode_wall_seconds')
COMPONENTS = (
    ('grasp', ('native_recomputed', 'grasp_seen'), True, None),
    ('lift', ('native_recomputed', 'lift_seen'), True, ('native_recomputed', 'grasp_seen')),
    ('carry', ('native_recomputed', 'carry_complete'), True, ('native_recomputed', 'lift_seen')),
    ('retention', ('native_recomputed', 'retention_lost'), False, ('native_recomputed', 'lift_seen')),
    ('terminal_composite', ('terminal_only', 'success'), True, ('native_recomputed', 'carry_complete')),
    ('release', None, True, ('native_recomputed', 'carry_complete')),
    ('settling', None, True, ('native_recomputed', 'carry_complete')),
    ('retreat', None, True, ('native_recomputed', 'carry_complete')),
)
NOTES = [
    'Posthoc description only: no tuning, model inference, new episodes or qualification changes.',
    'All-attempt success denominators include scheduled refusals, interrupted attempts and missing receipts.',
    'Component counts use valid independent native score receipts. Missing or invalid evidence remains unassessed.',
    'Conditional component denominators require the preceding grasp, lift or supported carry predicate; they are not all-attempt success rates.',
    'Release, settling and retreat are separately reconstructed posthoc from sealed native arrays using the frozen scorer predicates. Final predicate and consecutive dwell are diagnostic localization, not new qualification gates.',
    'Latencies pool recorded boundary samples, never averages of episode percentiles. The simulator pauses for computation on a shared host; these are not physical sensor-to-actuator latencies.',
    'Camera acquisition and availability timestamps are synthetic. Depth and physical hardware accuracy remain unqualified.',
    'Peak RSS is a process high-water mark including dataset export and any preceding fit, not an isolated model memory measurement.',
    'Evidence verification covers protocol/stage seals, referenced rollout/context receipts, native numeric evidence and model bytes. Image archives and full policy/scoring correctness remain the scope of the separately published independent audit.',
]


def digest(raw):
    return sha256(raw).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key: '+key)
        result[key] = value
    return result


def _read(path, maximum=MAX_JSON_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('Expected a bounded regular evidence file: '+path.name)
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
    return value, raw


def _relative(root, value):
    if not isinstance(value, str):
        raise ValueError('Evidence path must be relative.')
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Evidence path escapes the study.')
    path = root/relative
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError('Evidence cannot traverse symlinks.')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Evidence path escapes the study.')
    return path


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0.


def _integer(value):
    return type(value) is int and value >= 0


def _at(value, path):
    for key in path or ():
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if path else None


def _quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    point = (len(ordered)-1)*q
    lower = int(point); upper = min(lower+1, len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(point-lower)


def distribution(values):
    return dict(samples=len(values), p50=_quantile(values, .5), p95=_quantile(values, .95),
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def _verify_identity(path, expected):
    value, raw = _read(path)
    if expected != dict(bytes=len(raw), sha256=digest(raw)):
        raise ValueError('Bound evidence identity differs: '+path.name)
    return value


def native_terminal_components(arrays, metadata, episode):
    """Exact terminal predicates from the frozen scorer, recomputed numerically.

    No grasp attachment, action, teacher phase or scored success flag is used.
    Full 1 ms evidence is required; missing arrays produce an explicit refusal.
    """
    config = episode.get('config', {})
    dt, timeout = config.get('timestep'), config.get('timeout_s')
    required = config.get('goal_dwell_s')
    retreat = config.get('retreat_clearance_m')
    tolerance = config.get('goal_position_tolerance_m')
    if (not _number(dt) or dt != .001 or not _number(timeout) or not .1 <= timeout <= 30.
            or not math.isclose(timeout/dt, round(timeout/dt), abs_tol=1e-8, rel_tol=0)
            or not _number(required) or not .1 <= required <= .6
            or not _number(retreat) or not .08 <= retreat <= .15
            or not _number(tolerance) or not .01 <= tolerance <= .025):
        raise ValueError('Invalid native component task config.')
    count = round(timeout/dt)
    if (metadata.get('schema') != 'dvidia.piper-feedback-native/v1'
            or metadata.get('native_timestep_s') != dt or metadata.get('rows') != count
            or metadata.get('side_order') != ['left', 'right']):
        raise ValueError('Native component metadata/cadence differs.')
    widths = dict(t_seconds=1, native_tick=1, grasp_contact_counts=2,
        gripper_joint_position_m=2, object_position_m=3, tcp_position_m=3,
        object_linear_velocity_m_s=3, object_angular_velocity_rad_s=3,
        table_object_contact_count=1)
    values = {}
    for name, width in widths.items():
        value = np.asarray(arrays.get(name))
        if (value.dtype.kind not in 'fiu' or value.shape != ((count,) if width == 1 else (count, width))
                or not np.isfinite(value).all()):
            raise ValueError('Missing or malformed native component field: '+name)
        values[name] = value
    if (not np.array_equal(values['native_tick'], np.arange(1, count+1))
            or not np.allclose(values['t_seconds'], np.arange(1, count+1)*dt, atol=1e-8, rtol=0)):
        raise ValueError('Native component time/tick evidence differs.')
    for name in ('grasp_contact_counts', 'table_object_contact_count'):
        if np.any(values[name] < 0) or np.any(values[name] != np.floor(values[name])):
            raise ValueError('Native component contact count is not a nonnegative integer.')
    goal = np.asarray(config.get('target_position_m'))
    if goal.shape != (3,) or goal.dtype.kind not in 'fiu' or not np.isfinite(goal).all():
        raise ValueError('Native component destination is malformed.')
    jaw = values['gripper_joint_position_m']; obj = values['object_position_m']
    released = (np.all(values['grasp_contact_counts'] == 0, axis=1)
                & (jaw[:, 0] > .028) & (jaw[:, 1] < -.028))
    settled = ((np.linalg.norm(values['object_linear_velocity_m_s'], axis=1) < .035)
               & (np.linalg.norm(values['object_angular_velocity_rad_s'], axis=1) < .3))
    clear = values['tcp_position_m'][:, 2] >= obj[:, 2]+retreat
    supported = values['table_object_contact_count'] > 0
    at_destination = np.linalg.norm(obj-goal, axis=1) <= tolerance
    predicates = dict(release=released, settling=settled, retreat=clear,
        supported=supported, at_destination=at_destination,
        joint_terminal=released & settled & clear & supported & at_destination)
    result = {}
    for name, mask in predicates.items():
        failures = np.flatnonzero(~mask)
        terminal_ticks = count-int(failures[-1])-1 if len(failures) else count
        dwell = terminal_ticks*dt
        result[name] = dict(final_predicate=bool(mask[-1]), final_consecutive_dwell_seconds=dwell,
            required_dwell_seconds=required, dwell_met=bool(dwell+1e-12 >= required))
    return dict(status='available', native_ticks=count, predicates=result,
        scope='Posthoc native terminal-component diagnosis; does not replace the frozen task qualification gates.')


def _native_components(episode_root, files, receipt):
    if receipt.get('score', {}).get('valid') is not True:
        return dict(status='unassessed', reason='No valid independent native score receipt.')
    required = ('native.npz', 'native_metadata.json', 'native_episode.json')
    if any(name not in files for name in required):
        return dict(status='unassessed', reason='Required native component evidence is missing.')
    metadata = _verify_identity(episode_root/'native_metadata.json', files['native_metadata.json'])
    episode = _verify_identity(episode_root/'native_episode.json', files['native_episode.json'])
    path = episode_root/'native.npz'
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('Native numeric artifact must be a bounded regular file.')
    raw = path.read_bytes()
    if files['native.npz'] != dict(bytes=len(raw), sha256=digest(raw)):
        raise ValueError('Native numeric evidence differs from its bound inventory.')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) > 64 or sum(entry.file_size for entry in entries) > 64*1024*1024:
            raise ValueError('Native numeric expansion exceeds the bounded study artifact.')
    names = ('t_seconds', 'native_tick', 'grasp_contact_counts', 'gripper_joint_position_m',
             'object_position_m', 'tcp_position_m', 'object_linear_velocity_m_s',
             'object_angular_velocity_rad_s', 'table_object_contact_count')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in names if name in archive.files}
    result = native_terminal_components(arrays, metadata, episode)
    terminal = receipt['score'].get('terminal_only', {})
    reconstructed = result['predicates']['joint_terminal']
    if (type(terminal.get('success')) is not bool or not _number(terminal.get('dwell_s'))
            or terminal['success'] != reconstructed['dwell_met']
            or not math.isclose(terminal['dwell_s'], reconstructed['final_consecutive_dwell_seconds'], abs_tol=1e-8, rel_tol=0)):
        raise ValueError('Posthoc joint terminal predicate differs from the sealed independent scorer.')
    result['source_sha256'] = {name: files[name]['sha256'] for name in required}
    return result


def _episode(root, row):
    """Only read files already bound by the sealed stage and episode inventory."""
    if row.get('path') is None or row.get('receipt_sha256') is None:
        if row.get('strict_success') is True or row.get('complete') is True or row.get('score', {}).get('valid') is True:
            raise ValueError('Successful or valid outcomes require a bound rollout receipt.')
        return None, None
    episode_root = _relative(root, row['path'])
    receipt, raw = _read(episode_root/'rollout.json')
    if digest(raw) != row['receipt_sha256']:
        raise ValueError('Rollout receipt differs from sealed stage.')
    inventory, inventory_raw = _read(episode_root/'integrity.json')
    if digest(inventory_raw) != row.get('integrity_sha256'):
        raise ValueError('Episode inventory differs from sealed stage.')
    files = inventory.get('files')
    if not isinstance(files, dict):
        raise ValueError('Episode inventory is missing.')
    if files.get('rollout.json') != dict(bytes=len(raw), sha256=digest(raw)):
        raise ValueError('Rollout receipt differs from episode inventory.')
    if receipt.get('score') != row.get('score') or receipt.get('perception') != row.get('perception'):
        raise ValueError('Stage score/perception differs from bound rollout.')
    if receipt.get('status') != row.get('status'):
        raise ValueError('Stage completion differs from bound rollout.')
    complete = receipt.get('status') == 'complete' and receipt.get('recorded_complete_intervals') == 1000
    strict = bool(complete and receipt.get('score', {}).get('success') is True)
    if row.get('complete') is not complete or row.get('strict_success') is not strict:
        raise ValueError('Stage strict outcome differs from recorded completion and score.')
    context = None
    if 'context.json' in files:
        context = _verify_identity(episode_root/'context.json', files['context.json'])
    receipt['_posthoc_components'] = _native_components(episode_root, files, receipt)
    return receipt, context


def component_counts(rows, expected_count, component):
    name, path, success_value, prerequisite = component
    known = passed = failed = eligible = conditional_known = conditional_failed = unknown_prerequisite = 0
    for row in rows:
        score = row.get('score') or {}
        if score.get('valid') is not True:
            unknown_prerequisite += prerequisite is not None
            continue
        value = (_at(row.get('_posthoc_components', {}), ('predicates', name, 'dwell_met'))
                 if path is None else _at(score, path))
        admitted = True if prerequisite is None else _at(score, prerequisite)
        if type(admitted) is not bool:
            unknown_prerequisite += 1
        if admitted is True:
            eligible += 1
        if type(value) is bool:
            known += 1
            passes = value is success_value
            passed += passes; failed += not passes
            if admitted is True:
                conditional_known += 1; conditional_failed += not passes
    terminal_values = [_at(row.get('_posthoc_components', {}), ('predicates', name)) for row in rows] if path is None else []
    terminal_values = [value for value in terminal_values if isinstance(value, dict)]
    final = [value['final_predicate'] for value in terminal_values if type(value.get('final_predicate')) is bool]
    dwell = [value['final_consecutive_dwell_seconds'] for value in terminal_values if _number(value.get('final_consecutive_dwell_seconds'))]
    return dict(metric=name, all_attempts=expected_count, assessed=known, passed=passed, failed=failed,
        unassessed=expected_count-known, conditional_eligible=eligible,
        conditional_assessed=conditional_known, conditional_failed=conditional_failed,
        conditional_unassessed=eligible-conditional_known,
        prerequisite_unknown=unknown_prerequisite+(expected_count-len(rows) if prerequisite else 0),
        condition='all valid native score receipts' if prerequisite is None else '.'.join(prerequisite)+' == true',
        availability='posthoc_reconstructed_native_predicate' if path is None else 'recorded_independent_native_predicate',
        final_predicate=dict(assessed=len(final), true=sum(final), false=sum(not value for value in final)) if path is None else None,
        final_consecutive_dwell_seconds=distribution(dwell) if path is None else None,
        required_dwell_seconds=sorted({value['required_dwell_seconds'] for value in terminal_values
            if _number(value.get('required_dwell_seconds'))}) if path is None else None)


def _controller(stage_name, controller, roster, rows, episodes):
    expected = len(roster)
    values = [{**row, '_posthoc_components': (episodes.get(row['id'], (None, None))[0] or {}).get('_posthoc_components', {})}
              for row in rows if row['controller'] == controller]
    actual = [episodes.get(row['id'], (None, None))[0] for row in values]
    bound = [value for value in actual if value is not None]
    success = sum(row.get('strict_success') is True for row in values)
    complete = sum(row.get('complete') is True for row in values)
    native_valid = sum(row.get('score', {}).get('valid') is True for row in values)
    perception_rows = [row['perception'] for row in values if isinstance(row.get('perception'), dict)
        and all(_integer(row['perception'].get(name)) for name in ('decisions', 'available', 'missing'))]
    if any(row['available']+row['missing'] != row['decisions'] for row in perception_rows):
        raise ValueError('Perception counts disagree.')
    decisions = sum(row['decisions'] for row in perception_rows)
    available = sum(row['available'] for row in perception_rows)
    missing_spans = [row['longest_missing_seconds'] for row in perception_rows if _number(row.get('longest_missing_seconds'))]
    latencies = {}
    for metric in TIMING_FIELDS:
        samples = [boundary[metric] for value in bound for boundary in value.get('timing', {}).get('boundaries', [])
                   if isinstance(boundary, dict) and _number(boundary.get(metric))]
        applicable = not (metric == 'policy_inference_wall_seconds' and controller == 'teacher')
        result = distribution(samples if applicable else [])
        result.update(applicable=applicable, episode_receipts=len(bound), unit='seconds')
        if metric == 'full_boundary_wall_seconds':
            result['wall_samples_over_20ms'] = sum(sample > .02 for sample in samples)
            result['deadline_claim'] = False
        latencies[metric] = result
    durations = [value['timing']['wall_seconds'] for value in bound
                 if _number(value.get('timing', {}).get('wall_seconds'))]
    critical_counts = [_at(row.get('score'), ('native_recomputed', 'unexpected_contact_native_ticks'))
                       for row in values if row.get('score', {}).get('valid') is True]
    critical_counts = [value for value in critical_counts if _integer(value)]
    return dict(stage=stage_name, controller=controller, scheduled=expected,
        retained_rows=len(values), missing_rows=expected-len(values), complete=complete,
        native_assessable=native_valid, strict_successes=success,
        strict_success_rate_all_attempts=success/expected if expected else None,
        strict_success_rate_complete=success/complete if complete else None,
        not_confirmed_successful=expected-success,
        critical_contacts=dict(assessed_attempts=len(critical_counts),
            unassessed_attempts=expected-len(critical_counts),
            attempts_with_contacts=sum(value > 0 for value in critical_counts),
            affected_fraction_assessed=sum(value > 0 for value in critical_counts)/len(critical_counts) if critical_counts else None,
            native_ticks=sum(critical_counts), native_timestep_seconds=.001,
            scope='Independent native unexpected-contact ticks; repeated contact is counted at each 1 ms physics tick.'),
        statuses=dict(Counter(row.get('status', 'unknown') for row in values)),
        components={item[0]: component_counts(values, expected, item) for item in COMPONENTS},
        terminal_components_by_attempt=[dict(id=row['id'], **row['_posthoc_components']) for row in values],
        perception=dict(episodes_reported=len(perception_rows), episodes_unknown=expected-len(perception_rows),
            available_decisions=available, assessed_decisions=decisions,
            available_fraction=available/decisions if decisions else None,
            maximum_missing_seconds=max(missing_spans) if missing_spans else None),
        timing=latencies, episode_wall_seconds=distribution(durations))


def summarize(study):
    root = Path(study).resolve()
    registration, _ = _read(root/'registration.json')
    protocol, protocol_raw = _read(root/'protocol.json')
    protocol_hash = digest(protocol_raw)
    if registration.get('protocol_sha256') != protocol_hash:
        raise ValueError('Protocol differs from registration.')
    rosters = protocol.get('rosters')
    if not isinstance(rosters, dict):
        raise ValueError('Registered attempt rosters are required.')
    result = dict(schema=SCHEMA, protocol_sha256=protocol_hash, posthoc=True,
        physical_robot_ready=False, evidence_scope=NOTES[-1], notes=list(NOTES),
        controllers=[], stages=[], training=[], hardware=[], source_receipts={}, qualification=None)
    hardware = {}
    stage_names = list(dict.fromkeys([*rosters, 'train-initial', 'train-corrected']))
    for name in stage_names:
        path = root/'stages'/(name+'.json')
        if not path.exists():
            result['stages'].append(dict(stage=name, status='pending_receipt', scheduled=len(rosters.get(name, []))))
            continue
        stage, raw = _read(path)
        seal, _ = _read(path.with_name(name+'.integrity.json'))
        if seal != dict(sha256=digest(raw), protocol_sha256=protocol_hash):
            raise ValueError('Stage seal differs: '+name)
        if stage.get('stage') != name or stage.get('protocol_sha256') != protocol_hash:
            raise ValueError('Stage protocol/name binding differs.')
        result['source_receipts'][str(path.relative_to(root))] = digest(raw)
        result['stages'].append(dict(stage=name, status=stage.get('status'), reason=stage.get('reason'),
            scheduled=stage.get('scheduled'), strict_successes=stage.get('strict_successes')))
        if name.startswith('train-'):
            for variant, record in stage.get('models', {}).items():
                model_path = _relative(root, record['path'])
                _, model_raw = _read(model_path)
                if digest(model_raw) != record['model_sha256'] or len(model_raw) != record['model_bytes']:
                    raise ValueError('Training model bytes differ from its receipt.')
                result['training'].append(dict(stage=name, variant=variant, model_bytes=len(model_raw),
                    model_sha256=digest(model_raw), training_wall_seconds=record.get('training_wall_seconds'),
                    shared_export_wall_seconds=record.get('shared_export_wall_seconds'),
                    process_peak_rss_bytes=record.get('process_peak_rss_bytes'), memory_scope=record.get('memory_scope'),
                    training_episodes=len(record.get('training_episodes', []))))
            continue
        roster = rosters.get(name, [])
        expected = {row['id']: row for row in roster}
        if len(expected) != len(roster) or stage.get('scheduled') != len(roster):
            raise ValueError('Registered or sealed attempt counts disagree.')
        rows = stage.get('rows', [])
        if not isinstance(rows, list) or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Stage attempt IDs are duplicated or invalid.')
        if {row['id'] for row in rows} != set(expected):
            raise ValueError('Stage attempt ID set differs from the registered roster; unattempted rows must be retained.')
        episodes = {}
        for row in rows:
            registered = expected.get(row['id'])
            if registered is None or any(row.get(key) != registered.get(key)
                                         for key in ('controller', 'case', 'negative_control')):
                raise ValueError('Attempt differs from the registered roster.')
            receipt, context = _episode(root, row)
            episodes[row['id']] = receipt, context
            if context is not None:
                runtime = context.get('metadata', {}).get('runtime')
                if isinstance(runtime, dict):
                    key = json.dumps(runtime, sort_keys=True, allow_nan=False)
                    hardware.setdefault(key, dict(recorded_runtime=runtime, episode_receipts=0,
                        cpu_model=None, gpu_model=None, physical_sensor_model=None,
                        unreported_fields='CPU/GPU device identity and physical sensor identity were not captured in the runtime receipt.'))['episode_receipts'] += 1
        if stage.get('strict_successes') != sum(row.get('strict_success') is True for row in rows):
            raise ValueError('Stage strict-success total disagrees with its rows.')
        for controller in dict.fromkeys(row['controller'] for row in roster):
            registered = [row for row in roster if row['controller'] == controller]
            result['controllers'].append(_controller(name, controller, registered, rows, episodes))
    result['hardware'] = list(hardware.values())
    report_path = root/'report.json'
    if report_path.exists():
        report, raw = _read(report_path)
        if report.get('protocol_sha256') != protocol_hash:
            raise ValueError('Study report protocol binding differs.')
        result['qualification'] = dict(status=report.get('status'), qualified=report.get('qualified'),
            qualification=report.get('qualification'), report_sha256=digest(raw),
            source='existing study report; not newly awarded by this posthoc tool')
    return result


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ') if value is not None else 'Not reported'


def _percent(value):
    return 'Not reported' if value is None else f'{value*100:.1f}%'


def markdown(summary):
    lines = ['# Camera-to-action study: posthoc evidence tables', '',
        'These tables summarize sealed evidence without running or tuning policies.', '',
        '| Stage / controller | Strict success / scheduled | Complete | Native assessable | All-attempt success |',
        '|---|---:|---:|---:|---:|']
    for row in summary['controllers']:
        lines.append(f"| {_cell(row['stage'])} / {_cell(row['controller'])} | {row['strict_successes']} / {row['scheduled']} | {row['complete']} | {row['native_assessable']} | {_percent(row['strict_success_rate_all_attempts'])} |")
    lines += ['', 'Component failures are conditional on the stated prerequisite. Unknown counts are retained.', '',
        '| Stage / controller | Component | Prerequisite | Failed / conditional assessed | Eligible but unassessed | All-attempt unassessed |',
        '|---|---|---|---:|---:|---:|']
    for row in summary['controllers']:
        for name, component in row['components'].items():
            lines.append(f"| {_cell(row['stage'])} / {_cell(row['controller'])} | {_cell(name)} | {_cell(component['condition'])} | {component['conditional_failed']} / {component['conditional_assessed']} | {component['conditional_unassessed']} | {component['unassessed']} |")
    lines += ['', '| Stage / controller | Contact-affected / native assessed | Unassessed | Critical 1 ms ticks |',
        '|---|---:|---:|---:|']
    for row in summary['controllers']:
        contact=row['critical_contacts']
        lines.append(f"| {_cell(row['stage'])} / {_cell(row['controller'])} | {contact['attempts_with_contacts']} / {contact['assessed_attempts']} | {contact['unassessed_attempts']} | {contact['native_ticks']} |")
    lines += ['', 'Posthoc terminal predicates are measured separately from the frozen task gate.', '',
        '| Stage / controller | Predicate | False at final tick / assessed | Final dwell min / median | Required dwell |',
        '|---|---|---:|---:|---:|']
    for row in summary['controllers']:
        for name in ('release', 'settling', 'retreat'):
            component=row['components'][name];final=component['final_predicate'];dwell=component['final_consecutive_dwell_seconds']
            observed='Not reported' if dwell['p50'] is None else f"{dwell['minimum']:.3f} / {dwell['p50']:.3f} s"
            required=', '.join(f'{value:.3f} s' for value in component['required_dwell_seconds']) or 'Not reported'
            lines.append(f"| {_cell(row['stage'])} / {_cell(row['controller'])} | {name} | {final['false']} / {final['assessed']} | {observed} | {required} |")
    lines += ['', '| Stage / controller | Visual decisions available | Maximum missing span | Inference p95 | Full boundary p95 |',
        '|---|---:|---:|---:|---:|']
    for row in summary['controllers']:
        perception=row['perception']; timing=row['timing']
        def ms(value):return 'Not reported' if value is None else f'{value*1000:.3f} ms'
        span='Not reported' if perception['maximum_missing_seconds'] is None else f"{perception['maximum_missing_seconds']:.3f} s"
        lines.append(f"| {_cell(row['stage'])} / {_cell(row['controller'])} | {perception['available_decisions']} / {perception['assessed_decisions']} | {span} | {ms(timing['policy_inference_wall_seconds']['p95'])} | {ms(timing['full_boundary_wall_seconds']['p95'])} |")
    lines += ['', '| Fit | Model bytes | Fit/save seconds | Shared export seconds | Process peak RSS bytes |',
        '|---|---:|---:|---:|---:|']
    for row in summary['training']:
        lines.append('| '+' | '.join(_cell(v) for v in (row['stage']+' / '+row['variant'],row['model_bytes'],
            row['training_wall_seconds'],row['shared_export_wall_seconds'],row['process_peak_rss_bytes']))+' |')
    lines += ['', 'Recorded execution environments:', '']
    if summary['hardware']:
        for row in summary['hardware']:
            lines.append('- '+_cell(json.dumps(row['recorded_runtime'], sort_keys=True))+
                f"; {row['episode_receipts']} episode receipts. CPU/GPU device identities were not recorded.")
    else:
        lines.append('- No bound runtime receipt is available.')
    lines += ['', *('- '+note for note in summary['notes']), '']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    root=Path(args.study).resolve();output=Path(args.output).absolute()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(root):
        parser.error('Choose a fresh output directory outside the study.')
    summary=summarize(root)
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (output/'summary.md').write_text(markdown(summary))
    print(json.dumps(dict(status='complete',output=str(output),controller_rows=len(summary['controllers']),
        physical_robot_ready=False,posthoc=True)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
