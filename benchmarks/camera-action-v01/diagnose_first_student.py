"""Posthoc diagnosis of the first registered initial student and matching teacher.

Only sealed completed episodes are read. No policy inference, native physics,
training, teacher queries, experiment selection or qualification is performed.
The development attempt is chosen by protocol order, never by its outcome.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np

SCHEMA = 'dvidia.camera-action-first-student-diagnosis/v2'


def identity(raw):
    return dict(bytes=len(raw), sha256=sha256(raw).hexdigest())


def read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024 ** 2:
        raise ValueError('Expected bounded ordinary evidence.')
    return json.loads(path.read_bytes())


def bound(root, name, inventory, used):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or any(p.is_symlink() for p in (root / relative, *(root / relative).parents)):
        raise ValueError('Only plain study-relative evidence is accepted.')
    path = root / relative
    if path.stat().st_size > 128 * 1024 ** 2:
        raise ValueError('Evidence exceeds the bounded diagnostic format.')
    raw = path.read_bytes()
    actual = identity(raw)
    if actual != inventory[name]:
        raise ValueError('Sealed evidence changed: ' + name)
    used[name] = actual
    return raw


def first_time(mask, times):
    indices = np.flatnonzero(mask)
    return float(times[indices[0]]) if len(indices) else None


def intervals(mask, times):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [dict(first_tick_seconds=float(times[g[0]]), last_tick_seconds=float(times[g[-1]]),
        duration_seconds=len(g) * .001, ticks=len(g)) for g in groups]


def control_clock(steps):
    times = np.asarray([step['before']['timestamp_seconds'] for step in steps], dtype=np.float64)
    if (times.shape != (1000,) or not np.isfinite(times).all()
            or not np.allclose(times, np.arange(1000) * .020, atol=1e-8, rtol=0)):
        raise ValueError('Decision timestamps must follow the exact 50 Hz clock.')
    return times


def stage_binding(root, name, attempt, protocol, protocol_hash):
    """Bind the chosen local episode to its sealed, registered stage row."""
    path = root / 'stages' / (name + '.json')
    receipt = read(path); receipt_identity = identity(path.read_bytes())
    seal_path = path.with_name(name + '.integrity.json')
    started_path = path.with_name(name + '.started.json')
    if read(seal_path) != dict(sha256=receipt_identity['sha256'], protocol_sha256=protocol_hash):
        raise ValueError('Selected stage is not sealed: ' + name)
    started = read(started_path)
    if (receipt.get('schema') != 'dvidia.camera-action-study/v1/stage'
            or receipt.get('stage') != name or receipt.get('status') != 'complete'
            or receipt.get('protocol_sha256') != protocol_hash
            or started.get('stage') != name or started.get('protocol_sha256') != protocol_hash
            or started.get('source_sha256') != protocol['source_sha256']
            or started.get('model_bindings') != receipt.get('model_bindings')):
        raise ValueError('Stage registration or started bindings differ: ' + name)
    registered = protocol['rosters'][name]
    rows = receipt['rows']
    if ([r['id'] for r in rows] != [r['id'] for r in registered]
            or len({r['id'] for r in rows}) != len(rows)
            or receipt.get('scheduled') != len(registered)):
        raise ValueError('Stage does not contain the exact registered roster: ' + name)
    row = next(r for r in rows if r['id'] == attempt['id'])
    relative = 'episodes/' + name + '/' + attempt['id']
    if (any(row.get(key) != attempt.get(key) for key in ('case', 'controller', 'negative_control'))
            or row.get('path') != relative or row.get('status') != 'complete'
            or row.get('complete') is not True or row.get('recorded_complete_intervals') != 1000
            or row.get('policy_artifact_sha256') != receipt['model_bindings'][attempt['controller']]):
        raise ValueError('Chosen stage row differs from the complete registered attempt.')
    for filename, key in (('integrity.json', 'integrity_sha256'), ('rollout.json', 'receipt_sha256')):
        episode_path = root / relative / filename
        read(episode_path)  # Reject nonordinary or unbounded files before hashing.
        if identity(episode_path.read_bytes())['sha256'] != row[key]:
            raise ValueError('Chosen episode differs from its sealed stage row: ' + filename)
    rollout = read(root / relative / 'rollout.json')
    if (rollout.get('case') != {**attempt['case'], 'policy_artifact_sha256': row['policy_artifact_sha256']}
            or rollout.get('score') != row['score']
            or rollout.get('negative_control') != attempt['negative_control']
            or row['strict_success'] is not (rollout['score'].get('success') is True)):
        raise ValueError('Chosen rollout identity or outcome differs from its sealed stage row.')
    sources = {str(p.relative_to(root)): identity(p.read_bytes()) for p in (path, seal_path, started_path)}
    return row, sources


def interpretation(student, teacher, divergence):
    """Every observation depends on the measured case, including counterexamples."""
    acquired = student['bilateral_object_contact_ticks'] > 0
    observed = [f"Student strict success: {student['strict_success']}; teacher strict success: {teacher['strict_success']}."]
    if not acquired:
        observed.append('No bilateral student object contact was recorded; this case provides no acquired bilateral grip from which to assess retention.')
    else:
        observed.append(f"Student bilateral object contact occurred on {student['bilateral_object_contact_ticks']} native ticks; acquisition failure alone cannot explain this case.")
    for name, result in (('Student', student), ('Teacher', teacher)):
        event = result['closure_events']['jaw_target_first_below_20mm']
        if event is None:
            observed.append(name + ' never commanded a single-finger jaw target below 20 mm.')
        else:
            observed.append(f"{name} first commanded a single-finger jaw target below 20 mm at {event['time_seconds']:.3f} s, "
                f"with TCP-object XY distance {event['tcp_object_xy_distance_m'] * 1000:.2f} mm. This command does not imply attained finger position.")
    observed.append(f"Rate-limited intervals: student {student['rate_limited_intervals']}/{student['complete_intervals']}, "
        f"teacher {teacher['rate_limited_intervals']}/{teacher['complete_intervals']}.")
    observed.append(f"Critical native contact ticks: student {student['native_score']['unexpected_contact_native_ticks']}, "
        f"teacher {teacher['native_score']['unexpected_contact_native_ticks']}.")
    onset = divergence['first_more_than_10mm_seconds']; missing = student['first_missing_estimate_seconds']
    precedes_missing = onset is not None and (missing is None or onset < missing)
    if onset is None:
        observed.append('The same-clock tool trajectories never separated by more than 10 mm.')
    elif precedes_missing:
        observed.append('Tool separation exceeded 10 mm before any missing student estimate; missing visual tracking alone cannot explain that onset.')
    else:
        observed.append('Missing student estimates began no later than 10 mm tool separation; this comparison does not establish divergence preceding missing vision.')
    predicates = dict(student_failed=student['strict_success'] is False, no_bilateral_contact=not acquired,
        student_more_rate_limited=student['rate_limited_intervals'] > teacher['rate_limited_intervals'],
        divergence_precedes_missing=precedes_missing)
    hypothesis = ('Small errors on teacher-state absolute targets may compound through learned feedback into a different measured state distribution and poor approach/closure timing. '
        'The measured pattern is consistent with covariate shift, not a proven causal diagnosis.' if all(predicates.values())
        else 'The measured case does not meet the specific acquisition/rate-limit/early-divergence pattern; no mechanism is inferred automatically.')
    return dict(observed=observed, hypothesis= hypothesis, hypothesis_predicates=predicates,
        unresolved=['No counterfactual policy intervention was run.',
            'Normalization-unit excursions are unclipped descriptive distances, not training-support bounds; the frozen model clips standardized inputs to +/-10.',
            'One development case cannot establish failure frequency or the final corrected-model outcome.'],
        constraint='Keep the registered initial/correction/confirmation protocol unchanged; use this finding only for posthoc reporting.')


def episode(root, relative, attempt, model):
    directory = root / relative
    manifest_path = directory / 'integrity.json'
    manifest = read(manifest_path)
    if manifest.get('schema') != 'dvidia.camera-action-rollout/v2/integrity':
        raise ValueError('A sealed native v2 episode is required.')
    inventory = manifest['files']; used = {'integrity.json': identity(manifest_path.read_bytes())}
    read_bound = lambda name: json.loads(bound(directory, name, inventory, used))
    recording = read_bound('recording.json'); rollout = read_bound('rollout.json')
    native = read_bound('native_episode.json'); metadata = read_bound('native_metadata.json')
    raw_native = bound(directory, 'native.npz', inventory, used)
    with np.load(io.BytesIO(raw_native), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    steps = recording['steps']
    if (rollout.get('status') != 'complete' or rollout.get('recorded_complete_intervals') != 1000
            or len(steps) != 1000 or metadata.get('rows') != 20000
            or rollout['score'].get('valid') is not True
            or any(rollout['case'].get(key) != value for key, value in attempt['case'].items())):
        raise ValueError('Use only the matching sealed complete registered episode.')
    controller = attempt['controller']
    if rollout['mode'] != controller or rollout['teacher_queries'] != (1000 if controller == 'teacher' else 0):
        raise ValueError('Controller authority or query count differs.')
    times = arrays['t_seconds']
    if (times.shape != (20000,) or not np.allclose(times, np.arange(1, 20001) * .001, atol=1e-8, rtol=0)
            or any(not np.isfinite(value).all() for value in arrays.values())):
        raise ValueError('Native numeric evidence must be finite with exact 1 ms cadence.')
    issued = np.array([s['command']['issued']['joint_targets_rad'] for s in steps])
    jaw = np.array([s['command']['issued']['gripper_position_m'] for s in steps])
    measured = np.array([s['before']['signals']['joint_position_rad']['value'] for s in steps])
    velocity = np.array([s['before']['signals']['joint_velocity_rad_s']['value'] for s in steps])
    applied = np.array([s['applied']['ramped_targets']['joint_targets_rad'] for s in steps])
    control_times = control_clock(steps)
    available = []; phases = []; previous_phase = None
    for step in steps:
        if step['command']['authority'] != controller:
            raise ValueError('Episode has mixed authority.')
        estimate = read_bound(step['perception']['path'])['estimate']
        available.append(estimate['status'] == 'available')
        if controller == 'teacher':
            phase = read_bound(step['teacher_query']['path'])['phase']
            if phase != previous_phase:
                phases.append(dict(time_seconds=step['before']['timestamp_seconds'], phase=phase))
            previous_phase = phase
        elif step['teacher_query'] is not None:
            raise ValueError('A student episode cannot query the teacher.')
    available = np.asarray(available)
    both = np.all(arrays['grasp_contact_counts'] > 0, axis=1)
    any_contact = np.any(arrays['grasp_contact_counts'] > 0, axis=1)
    critical = arrays['unexpected_contact_count'] > 0
    score = rollout['score']['native_recomputed']
    max_lift = float(arrays['object_position_m'][:, 2].max() - native['initial_object_position_m'][2])
    if int(critical.sum()) != score['unexpected_contact_native_ticks'] or not np.isclose(max_lift, score['max_object_lift_m'], atol=1e-10, rtol=0):
        raise ValueError('Native scalar metrics differ from saved independent score.')
    velocity_indices = [model['feature_names'].index(f'lag0:joint_velocity_rad_s_{i}/3') for i in range(6)]
    mean = np.asarray(model['feature_mean'])[velocity_indices]
    scale = np.asarray(model['feature_scale'])[velocity_indices]
    normalized_velocity = (velocity / 3. - mean) / scale
    events = {}
    for label, threshold in (('jaw_target_first_below_20mm', .020), ('jaw_target_first_below_5mm', .005)):
        indices = np.flatnonzero(jaw < threshold)
        if not len(indices):
            events[label] = None
            continue
        index = int(indices[0]); time = float(control_times[index]); native_index = round(time * 1000) - 1
        if native_index < 0:
            raise ValueError('This event has no preceding saved native tick.')
        delta = arrays['tcp_position_m'][native_index] - arrays['object_position_m'][native_index]
        events[label] = dict(time_seconds=time, requested_jaw_m=float(jaw[index]),
            actual_finger_joint_positions_m=arrays['gripper_joint_position_m'][native_index].tolist(),
            tcp_object_xy_distance_m=float(np.linalg.norm(delta[:2])), tcp_minus_object_z_m=float(delta[2]),
            native_grasp_contacts=arrays['grasp_contact_counts'][native_index].tolist())
    pair_counts = native['metrics']['contact_pair_native_ticks']
    table = metadata['contact_names']['table_geom_id']; pads = metadata['contact_names']['pad_geom_ids']
    pad_table = {side: {pair: ticks for pair, ticks in pair_counts.items()
        if table in [int(v) for v in pair.split(':')] and any(pad in [int(v) for v in pair.split(':')] for pad in ids)}
        for side, ids in pads.items()}
    result = dict(relative_episode=relative, case=attempt['case'], controller=controller,
        complete_intervals=1000, teacher_queries=rollout['teacher_queries'], strict_success=rollout['score']['success'],
        initial_visual_error_m=rollout['initial_anchor_error_m'], native_score=score,
        tcp_path_length_m=float(np.linalg.norm(np.diff(arrays['tcp_position_m'], axis=0), axis=1).sum()),
        tcp_path_scope='Euclidean travel between all saved 1 ms distal-site positions, not link6 feature positions.',
        rate_limited_intervals=sum(step['applied']['rate_limited'] for step in steps),
        issued_to_measured_max_absolute_joint_difference_rad=float(np.abs(issued - measured).max()),
        issued_to_final_applied_joint_rms_difference_rad=float(np.sqrt(np.mean((issued - applied) ** 2))),
        issued_joint_total_variation_rad=np.abs(np.diff(issued, axis=0)).sum(axis=0).tolist(),
        issued_jaw_total_variation_m=float(np.abs(np.diff(jaw)).sum()),
        bilateral_object_contact_ticks=int(both.sum()), bilateral_object_contact_intervals=intervals(both, times),
        any_object_contact_intervals=intervals(any_contact, times), critical_contact_intervals=intervals(critical, times),
        recorded_pad_table_contact_pair_ticks=pad_table,
        pair_counter_scope='Simulator-recorded 1 ms geom-pair counters; sides may overlap. This analysis does not rerun a contact solver.',
        first_missing_estimate_seconds=first_time(~available, control_times),
        visual_available_decisions=int(available.sum()), closure_events=events, teacher_phase_starts=phases,
        maximum_absolute_joint_velocity_rad_s=np.abs(velocity).max(axis=0).tolist(),
        velocity_normalization=dict(maximum_absolute_units=float(np.abs(normalized_velocity).max()),
            first_time_above_5_units_seconds=first_time(np.any(np.abs(normalized_velocity) > 5, axis=1), control_times),
            decisions_above_5_units=int(np.any(np.abs(normalized_velocity) > 5, axis=1).sum()),
            scope='Unclipped descriptive posthoc distance from training feature mean divided by stored feature scale. The frozen model clips standardized inputs to +/-10 before inference. Five units is not a registered gate or proof of being outside the full training support.'),
        source_files=used)
    return result, arrays


def diagnose(study):
    root = Path(study).absolute()
    protocol = read(root / 'protocol.json'); registration = read(root / 'registration.json')
    protocol_identity = identity((root / 'protocol.json').read_bytes())
    if registration['protocol_sha256'] != protocol_identity['sha256']:
        raise ValueError('Protocol binding differs.')
    student_attempt = next(row for row in protocol['rosters']['development-initial'] if row['controller'] == 'student')
    teacher_attempt = next(row for row in protocol['rosters']['demonstrations'] if row['case'] == student_attempt['case'])
    train_path = root / 'stages/train-initial.json'; train = read(train_path)
    train_identity = identity(train_path.read_bytes())
    if read(root / 'stages/train-initial.integrity.json') != dict(sha256=train_identity['sha256'], protocol_sha256=protocol_identity['sha256']):
        raise ValueError('Initial fit is not sealed.')
    model_path = root / train['models']['full']['path']; model_identity = identity(model_path.read_bytes())
    if train['models']['full']['model_sha256'] != model_identity['sha256']:
        raise ValueError('Initial model changed.')
    model = read(model_path)
    student_relative = 'episodes/development-initial/' + student_attempt['id']
    teacher_relative = 'episodes/demonstrations/' + teacher_attempt['id']
    student_row, student_stage_sources = stage_binding(root, 'development-initial', student_attempt, protocol, protocol_identity['sha256'])
    teacher_row, teacher_stage_sources = stage_binding(root, 'demonstrations', teacher_attempt, protocol, protocol_identity['sha256'])
    student, student_arrays = episode(root, student_relative, student_attempt, model)
    teacher, teacher_arrays = episode(root, teacher_relative, teacher_attempt, model)
    if student_row['policy_artifact_sha256'] != model_identity['sha256']:
        raise ValueError('The first student did not run the sealed initial full model.')
    distance = np.linalg.norm(student_arrays['tcp_position_m'] - teacher_arrays['tcp_position_m'], axis=1)
    divergence = dict(first_more_than_10mm_seconds=first_time(distance > .010, student_arrays['t_seconds']),
        first_more_than_50mm_seconds=first_time(distance > .050, student_arrays['t_seconds']),
        scope='Same simulated time, same registered start/goal; this is not phase-aligned prediction error.')
    return dict(schema=SCHEMA, posthoc=True, scope='First registered initial development student versus exact matching synthetic teacher; not aggregate confirmation and not model tuning.',
        audit_status='sealed_local_diagnostic_pending_full_study_audit',
        audit_scope='Selected episode and stage byte bindings plus the numeric checks stated here; not a full-study audit, source attestation or hardware verification.',
        physics_executed=False, model_executed=False, teacher_queried=False, qualification_changed=False,
        selection='First protocol-ordered full-student attempt, regardless of outcome.',
        source=dict(protocol=protocol_identity, initial_training_stage=train_identity, full_model=model_identity,
            registration=identity((root / 'registration.json').read_bytes()),
            initial_training_seal=identity((root / 'stages/train-initial.integrity.json').read_bytes()),
            stage_files={**student_stage_sources, **teacher_stage_sources},
            analysis_source=identity(Path(__file__).read_bytes())),
        validation_on_teacher_observations=model['training']['validation'], student=student, teacher=teacher,
        same_clock_trajectory_divergence=divergence, interpretation=interpretation(student, teacher, divergence))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).absolute()
    if output.exists() or output.resolve().is_relative_to(Path(args.study).resolve()):
        parser.error('Choose a fresh proof file outside the study.')
    result = diagnose(args.study)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), sha256=identity(output.read_bytes())['sha256'],
        student_success=result['student']['strict_success'], teacher_success=result['teacher']['strict_success'], physics_executed=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
