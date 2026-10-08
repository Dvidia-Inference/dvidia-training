"""Frozen sample-count comparisons and exact-scene native qualification.

This experiment learns from existing aligned numerical labels, not actions
recovered from human video. It never broadens an installed capsule's scene gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from .footage_pipeline import _canonical, _fresh, _json, _prepare, _prepared_copy

DEFAULT_COUNTS = (2, 4, 7)
SOURCE_FILES = ('footage_benchmark.py', 'footage_pipeline.py', 'footage_data.py', 'footage_train.py')
EVIDENCE_FILE_MAX_BYTES = 8 * 1024 * 1024
EVIDENCE_TOTAL_MAX_BYTES = 128 * 1024 * 1024


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b'\n')


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer in [{low}, {high}].')
    return value


def normalize_counts(counts, available):
    """Counts measure whole training groups, not frames or sidecar samples."""
    _integer(available, 1, 64, 'Available training groups')
    if counts is None:
        return tuple(value for value in DEFAULT_COUNTS if value <= available) or (available,)
    if not isinstance(counts, (list, tuple)) or not counts:
        raise ValueError('Supply a nonempty list of training group counts.')
    if len(counts) > 8:
        raise ValueError('Compare at most eight training group counts in one benchmark.')
    for value in counts:
        _integer(value, 1, available, 'Training group count')
    if len(set(counts)) != len(counts):
        raise ValueError('Training group counts must be distinct.')
    return tuple(sorted(counts))


def nested_subsets(dataset, counts=None, seed=17):
    """Select nested connected groups; retain all development/test episodes."""
    from .footage_data import _groups
    _integer(seed, 0, 999999, 'Subset seed')
    episodes = dataset.get('episodes')
    if type(episodes) is not list or not episodes:
        raise ValueError('A prepared dataset with episodes is required.')
    checked = deepcopy(episodes)
    groups = _groups(checked)
    assignments = {}
    for original, recalculated in zip(episodes, checked):
        split = original.get('split')
        if split not in ('train', 'dev', 'test'):
            raise ValueError('Each source episode needs a train, dev or test split.')
        group = recalculated['group_id']
        if group in assignments and assignments[group] != split:
            raise ValueError('Connected source groups cross training/development/test splits.')
        assignments[group] = split
        if original.get('group_id') != group:
            raise ValueError('Declared source groups differ from connected recording lineage.')
    fixed = {name: [row['id'] for row in episodes if row['split'] == name]
             for name in ('train', 'dev', 'test')}
    if not all(fixed.values()):
        raise ValueError('Keep nonempty training, development and test groups.')
    if dataset.get('splits') is not None and dataset['splits'] != fixed:
        raise ValueError('Prepared split metadata differs from episode assignments.')
    training = [group['id'] for group in groups if assignments[group['id']] == 'train']
    ordered = sorted(training, key=lambda group: _digest([seed, group]))
    result = []
    for count in normalize_counts(counts, len(training)):
        selected = ordered[:count]
        result.append({'training_group_count': count, 'train_group_ids': selected,
                       'train_episode_ids': [row['id'] for row in episodes
                                             if row['split'] == 'train' and row['group_id'] in selected],
                       'dev_episode_ids': fixed['dev'], 'test_episode_ids': fixed['test']})
    return result


def _layout_digest(scene):
    # A new seed or material value cannot turn the same layout into new coverage.
    return _digest({key: scene[key] for key in ('object_position', 'target_position')})


def make_scenes(seed, nominal=8, stress=4, source_scenes=()):
    """Create distinct bounded box layouts without observing any outcome."""
    import numpy as np
    from .arm_runner import validate_scene
    from .skill_capsule import scene_sha256
    _integer(seed, 0, 999900, 'Scene seed')
    _integer(nominal, 0, 24, 'Nominal scene count')
    _integer(stress, 0, 24, 'Stress scene count')
    if not 1 <= nominal + stress <= 24:
        raise ValueError('Use one to 24 total evaluation scenes.')
    excluded = {_layout_digest(validate_scene(scene)) for scene in source_scenes}
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(nominal + stress):
        for attempt in range(10000):
            xy = np.round(rng.uniform([.27, -.17, .29, -.17], [.53, .17, .54, .17]), 6)
            if (np.linalg.norm(xy[:2]) > .54 or np.linalg.norm(xy[2:]) > .54
                    or np.linalg.norm(xy[:2] - xy[2:]) < .10):
                continue
            size = rng.choice([.03, .035, .04, .045, .05], 3).tolist()
            z = .29 + size[2] / 2
            stressed = index >= nominal
            scene = validate_scene({
                'object_position': [*xy[:2].tolist(), z], 'target_position': [*xy[2:].tolist(), z],
                'object_size': size, 'object_mass': float(rng.choice([.02, .04, .06, .08])),
                'object_friction': float(rng.choice([.4, .8, 1.2, 1.5])),
                'jaw_force_limit': .3 if stressed else 15.,
                'joint_torque_limit': 5. if stressed else 40., 'seed': seed + index})
            layout = _layout_digest(scene)
            if layout in excluded:
                continue
            excluded.add(layout)
            cases.append({'id': f'scene-{index:02}', 'group': 'actuator_stress' if stressed else 'nominal',
                          'scene': scene, 'scene_sha256': scene_sha256(scene), 'layout_sha256': layout})
            break
        else:
            raise ValueError('Could not create a distinct supported evaluation layout.')
    return cases


def freeze_protocol(output, protocol):
    """An exclusive plan is written before any fitting or physics evaluation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = _canonical(protocol) + b'\n'
    with (output / 'protocol.json').open('xb') as handle:
        handle.write(raw)
    with (output / 'protocol.sha256').open('x') as handle:
        handle.write(sha256(raw).hexdigest() + '\n')
    return sha256(raw).hexdigest()


def verify_protocol(output, protocol):
    output = Path(output)
    raw = (output / 'protocol.json').read_bytes()
    expected = _canonical(protocol) + b'\n'
    if raw != expected or (output / 'protocol.sha256').read_text().strip() != sha256(raw).hexdigest():
        raise ValueError('Frozen benchmark protocol changed. Choose a fresh experiment.')


def _source_hashes():
    return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _source_scenes(dataset, sidecars, directory):
    if directory is None:
        return [], {'status': 'unestablished', 'reason': 'No hash-bound native scene evidence was supplied.'}
    from .arm_runner import validate_scene
    from .footage_train import _json as parse_evidence
    directory = Path(directory)
    if not directory.is_dir() or any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError('Source evidence must be a local directory without symbolic links.')
    paths = list(directory.glob('*.json'))
    if len(paths) > 1200:
        raise ValueError('Source evidence directory exceeds 1200 JSON files.')
    index, total = {}, 0
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > EVIDENCE_FILE_MAX_BYTES:
            raise ValueError('Native evidence must use regular JSON files of at most 8 MiB.')
        raw = path.read_bytes()
        if len(raw) > EVIDENCE_FILE_MAX_BYTES:
            raise ValueError('Native evidence grew beyond its bounded size while being read.')
        total += len(raw)
        if total > EVIDENCE_TOTAL_MAX_BYTES:
            raise ValueError('Native evidence exceeds the 128 MiB benchmark limit.')
        index[sha256(raw).hexdigest()] = raw
    scenes, bindings = [], []
    for row in dataset['episodes']:
        provenance = sidecars[row['id']]['provenance']
        if provenance['source_kind'] != 'native-simulation-telemetry':
            raise ValueError('Native scene exclusion requires native telemetry evidence for every episode.')
        digest = provenance['validation']['evidence_sha256']
        if digest not in index:
            raise ValueError(f"Missing hash-bound native evidence for {row['id']}.")
        record = parse_evidence(index[digest], maximum=EVIDENCE_FILE_MAX_BYTES)
        if (record.get('kind') != 'dvidia.native-telemetry-evidence' or record.get('schema_version') != 1
                or record.get('scope') != 'simulation-only'):
            raise ValueError('Expected simulation-only native telemetry evidence.')
        config = record.get('final_info', {}).get('config', {})
        if config.get('scene_jitter') != 0 or config.get('table_height') != .29:
            raise ValueError('Scene exclusion supports declared zero-jitter scenes on the existing table.')
        try:
            scene = validate_scene({key: config[key] for key in (
                'object_position', 'target_position', 'object_mass', 'object_friction',
                'jaw_force_limit', 'joint_torque_limit')} | {
                'object_size': [2 * value for value in config['object_half_size']], 'seed': record['seed']})
        except (KeyError, TypeError) as exc:
            raise ValueError('Native evidence lacks a complete supported scene declaration.') from exc
        scenes.append(scene)
        bindings.append({'episode_id': row['id'], 'split': row['split'], 'evidence_sha256': digest,
                         'layout_sha256': _layout_digest(scene), 'scene': scene})
    return scenes, {'status': 'excluded_hash_bound_native_declarations', 'bindings': bindings,
                    'limits': 'Hashes and supplier scene declarations check consistency, not independent capture attestation.'}


def _materialize_subset(snapshot, spec, output):
    dataset = _json(snapshot)
    chosen = set(spec['train_episode_ids'] + spec['dev_episode_ids'] + spec['test_episode_ids'])
    dataset['episodes'] = [row for row in dataset['episodes'] if row['id'] in chosen]
    dataset['groups'] = [row for row in dataset['groups'] if any(identifier in chosen for identifier in row['episode_ids'])]
    dataset['splits'] = {name: [row['id'] for row in dataset['episodes'] if row['split'] == name]
                         for name in ('train', 'dev', 'test')}
    dataset['actual_media_episodes'] = len(dataset['episodes'])
    dataset['independent_groups'] = len(dataset['groups'])
    dataset['benchmark_parent_dataset_sha256'] = sha256(snapshot.read_bytes()).hexdigest()
    temporary = snapshot.parent / f"subset-{spec['training_group_count']:03}.json"
    with temporary.open('xb') as handle:
        handle.write(_canonical(dataset) + b'\n')
    try:
        _prepared_copy(temporary, output)
    finally:
        temporary.unlink()


def _assert_inputs(output, protocol):
    from .capsule_installer import capsule_runtime
    verify_protocol(output, protocol)
    if capsule_runtime() != protocol['runtime'] or _source_hashes() != protocol['benchmark_source_sha256']:
        raise ValueError('Frozen benchmark runtime or training source changed.')
    for path, expected in protocol['input_files_sha256'].items():
        if sha256((output / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen benchmark input changed.')


def _episode_metrics(episode, prefix):
    wall = episode['episode_wall_seconds']
    simulated = episode['simulated_seconds']
    return {f'{prefix}_final_target_error_m': episode['final_distance'],
            f'{prefix}_simulated_seconds': simulated,
            f'{prefix}_episode_wall_seconds': wall,
            f'{prefix}_simulated_seconds_per_wall_second': simulated / wall if wall else None}


def _measurement_summary(rows, prefix):
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    measured = [row for row in rows if finite(row.get(f'{prefix}_episode_wall_seconds'))
                and finite(row.get(f'{prefix}_simulated_seconds'))]
    wall = sum(row[f'{prefix}_episode_wall_seconds'] for row in measured)
    simulated = sum(row[f'{prefix}_simulated_seconds'] for row in measured)
    errors = [row[f'{prefix}_final_target_error_m'] for row in rows
              if finite(row.get(f'{prefix}_final_target_error_m'))]
    return {'measured_scenes': len(measured), 'timed_scenes': len(measured),
            'measured_error_scenes': len(errors), 'total_simulated_seconds': simulated,
            'total_episode_wall_seconds': wall,
            'simulated_seconds_per_wall_second': simulated / wall if wall else None,
            'final_target_error_mean_m': sum(errors) / len(errors) if errors else None,
            'final_target_error_max_m': max(errors) if errors else None}


def _aggregate(rows):
    groups = {}
    for group in sorted({row['group'] for row in rows}):
        values = [row for row in rows if row['group'] == group]
        groups[group] = {'planned_scenes': len(values),
                         'student_successes': sum(row.get('student_success') is True for row in values),
                         'qualified_scenes': sum(row.get('qualified') is True for row in values),
                         'control_successes': sum(row.get('control_success') is True for row in values),
                         'errors': sum(row.get('status') == 'error' for row in values),
                         'student_measurements': _measurement_summary(values, 'student'),
                         'control_measurements': _measurement_summary(values, 'control')}
    student_measurements = _measurement_summary(rows, 'student')
    control_measurements = _measurement_summary(rows, 'control')
    return {'planned_scenes': len(rows), 'student_successes': sum(row.get('student_success') is True for row in rows),
            'qualified_scenes': sum(row.get('qualified') is True for row in rows),
            'control_successes': sum(row.get('control_success') is True for row in rows),
            'errors': sum(row.get('status') == 'error' for row in rows), 'groups': groups,
            'student_episode_wall_seconds': student_measurements['total_episode_wall_seconds'],
            'control_episode_wall_seconds': control_measurements['total_episode_wall_seconds'],
            'student_measurements': student_measurements,
            'control_measurements': control_measurements}


def _paired_outcomes(teachers, rows):
    paired = {'both_success': 0, 'teacher_only': 0, 'student_only': 0, 'both_failure': 0, 'errors': 0}
    for teacher, row in zip(teachers, rows):
        if row['status'] == 'error' or teacher['status'] == 'error':
            paired['errors'] += 1
        else:
            key = ('both_success' if teacher['success'] else 'student_only') if row['student_success'] else (
                'teacher_only' if teacher['success'] else 'both_failure')
            paired[key] += 1
    return paired


def benchmark(dataset_path, output, *, counts=None, seed=19043, nominal=8, stress=4,
              task_source=None, grounding=None, source_evidence=None):
    """Fit every frozen count before evaluating any of its scene outcomes."""
    started = time.perf_counter()
    from .arm_entrypoints import require_arm
    require_arm()
    from .arm_runner import authored_grounding, run_skill, write_run
    from .capsule_installer import capsule_runtime, capsule_worker, install_capsule
    from .footage_pipeline import inspect_run, run
    from .footage_train import _load_dataset, RIDGE_GRID
    from .skillspace import load_skillspace
    _integer(seed, 0, 999900, 'Benchmark seed')
    dataset_path = Path(dataset_path)
    dataset, original_dataset_hash, _, sidecars = _load_dataset(dataset_path)
    if set(sidecars) != {row['id'] for row in dataset['episodes']}:
        raise ValueError('Every benchmark episode needs validated aligned movement supervision.')
    subsets = nested_subsets(dataset, counts, seed)
    original_scenes, exclusion = _source_scenes(dataset, sidecars, source_evidence)
    cases = make_scenes(seed, nominal, stress, original_scenes)
    task_source = Path(task_source) if task_source is not None else Path(__file__).with_name('place_cup.skill.json')
    task = _json(task_source, limit=128 * 1024)
    raw = task_source.read_bytes()
    resolved_grounding = _json(Path(grounding)) if grounding is not None else (
        None if 'simulation_grounding' in task else authored_grounding(raw))
    load_skillspace(raw, resolved_grounding)
    output = _fresh(output)
    _prepare(dataset_path, output / 'source', offline=True, seed=seed)
    if sha256(dataset_path.read_bytes()).hexdigest() != original_dataset_hash:
        raise ValueError('The prepared source changed while the benchmark snapshot was being made.')
    snapshot = output / 'source' / 'dataset.json'
    (output / 'task.skill.json').write_bytes(raw)
    input_hashes = {'source/dataset.json': sha256(snapshot.read_bytes()).hexdigest(),
                    'task.skill.json': sha256(raw).hexdigest()}
    if resolved_grounding is not None:
        _write(output / 'grounding.json', resolved_grounding)
        input_hashes['grounding.json'] = sha256((output / 'grounding.json').read_bytes()).hexdigest()
    for spec in subsets:
        relative = f"subsets/groups-{spec['training_group_count']:03}"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _materialize_subset(snapshot, spec, destination)
        spec['dataset_path'] = f'{relative}/dataset.json'
        input_hashes[spec['dataset_path']] = sha256((output / spec['dataset_path']).read_bytes()).hexdigest()
    protocol = {'schema_version': 1, 'kind': 'dvidia.footage-sample-count-benchmark',
                'scope': 'simulation-only', 'physical_robot_ready': False,
                'source_dataset_sha256': original_dataset_hash,
                'runtime': capsule_runtime(), 'benchmark_source_sha256': _source_hashes(),
                'input_files_sha256': input_hashes, 'seed': seed, 'subsets': subsets,
                'scenes': cases, 'source_scene_exclusion': exclusion,
                'training': {'frames_per_clip': 12, 'resolution': 16, 'latent_dim': 8,
                             'movement_centers': 8, 'regularization_grid': list(RIDGE_GRID),
                             'selection': 'The same fixed development episodes select regularization for each count.'},
                'rules': {'counts': 'Nested whole connected training groups; development and test episodes never change.',
                          'freeze': 'Freeze this protocol before fitting; freeze all candidates before any scene evaluation.',
                          'teacher': 'One authored baseline per exact scene, shared by all candidates.',
                          'control': 'Each candidate uses identical recorded joint targets with jaws forced open.',
                          'qualification': 'Existing native exact-scene predicate and installation gate are unchanged.',
                          'limits': 'No human-video action recovery, demonstration sufficiency threshold, hardware qualification or end-to-end speedup is implied.'}}
    protocol_hash = freeze_protocol(output, protocol)
    candidates = []
    for spec in subsets:
        _assert_inputs(output, protocol)
        directory = f"candidates/groups-{spec['training_group_count']:03}"
        result = run(output / spec['dataset_path'], output / directory, seed=seed,
                     task_source=output / 'task.skill.json',
                     grounding=output / 'grounding.json' if resolved_grounding is not None else None)
        inspect_run(output / directory)
        if (result['movement']['train_episode_ids'] != spec['train_episode_ids']
                or [row['episode_id'] for row in result['movement']['alignment_records'] if row['split'] == 'dev'] != spec['dev_episode_ids']
                or [row['episode_id'] for row in result['movement']['alignment_records'] if row['split'] == 'test'] != spec['test_episode_ids']):
            raise ValueError('Training changed the frozen source-group assignments.')
        capsule_path = output / directory / 'candidate.skill-capsule.json'
        installation = install_capsule(capsule_path.read_bytes(), output / 'installs')
        candidates.append({'training_group_count': spec['training_group_count'],
                           'training_episode_count': len(spec['train_episode_ids']),
                           'training_action_samples': result['movement']['training']['sample_count'],
                           'training_pipeline_seconds': result['elapsed_seconds'],
                           'training_pipeline_timing_scope': result['timing_scope'],
                           'visual_test': result['visual']['test'], 'movement_test': result['movement']['test'],
                           'installation_id': installation['installation_id'], 'capsule_path': f'{directory}/candidate.skill-capsule.json',
                           'capsule_file_sha256': sha256(capsule_path.read_bytes()).hexdigest()})
    models = {'protocol_sha256': protocol_hash, 'candidates': candidates}
    with (output / 'models-frozen.json').open('xb') as handle:
        handle.write(_canonical(models) + b'\n')
    models_hash = sha256((output / 'models-frozen.json').read_bytes()).hexdigest()
    def verify():
        _assert_inputs(output, protocol)
        if sha256((output / 'models-frozen.json').read_bytes()).hexdigest() != models_hash:
            raise ValueError('Frozen candidate set changed.')
        for candidate in candidates:
            if sha256((output / candidate['capsule_path']).read_bytes()).hexdigest() != candidate['capsule_file_sha256']:
                raise ValueError('Frozen candidate weights changed.')
    teachers = []
    for case in cases:
        verify()
        relative = f"teacher/{case['id']}"
        record = {'case_id': case['id'], 'group': case['group'], 'scene_sha256': case['scene_sha256'],
                  'status': 'completed', 'run_path': f'{relative}/run.json'}
        try:
            payload = run_skill(raw, resolved_grounding, case['scene'], offline=True)
            write_run(output / relative, payload, source=raw, grounding=resolved_grounding)
            episode = payload['episodes'][0]
            record.update(success=episode['success'], reason=episode['reason'],
                          episode_wall_seconds=episode['episode_wall_seconds'], **_episode_metrics(episode, 'teacher'))
        except (ValueError, RuntimeError) as exc:
            record.update(status='error', success=False, error=str(exc))
            _write(output / relative / 'benchmark-error.json', record)
        teachers.append(record)
    comparisons = []
    for candidate in candidates:
        rows = []
        for case in cases:
            verify()
            relative = f"qualification/groups-{candidate['training_group_count']:03}/{case['id']}"
            record = {'case_id': case['id'], 'group': case['group'], 'scene_sha256': case['scene_sha256'],
                      'status': 'completed', 'qualified': False, 'run_path': f'{relative}/run.json',
                      'controls_path': f'{relative}/controls.json', 'qualification_path': f'{relative}/qualification.json'}
            try:
                payload = capsule_worker(output / 'installs', candidate['installation_id'], case['scene'],
                                         output / relative, qualify=True)
                controls = _json(output / record['controls_path'])
                student, control = payload['episodes'][0], controls['episodes'][0]
                receipt = payload['local_qualification']
                record.update(student_success=student['success'], student_reason=student['reason'],
                              control_success=control['success'], control_reason=control['reason'],
                              qualified=receipt.get('status') == 'validated_simulation_scene',
                              qualification_status=receipt.get('status'),
                              **_episode_metrics(student, 'student'), **_episode_metrics(control, 'control'))
            except (ValueError, RuntimeError) as exc:
                record.update(status='error', error=str(exc))
                _write(output / relative / 'benchmark-error.json', record)
            rows.append(record)
        paired_groups = {group: _paired_outcomes([row for row in teachers if row['group'] == group],
                                                [row for row in rows if row['group'] == group])
                         for group in sorted({row['group'] for row in rows})}
        comparisons.append({**candidate, 'outcomes': rows, 'summary': _aggregate(rows),
                            'paired_with_teacher': _paired_outcomes(teachers, rows),
                            'paired_with_teacher_by_group': paired_groups})
    verify()
    report = {'schema_version': 1, 'kind': 'dvidia.footage-sample-count-result',
              'scope': 'simulation-only', 'physical_robot_ready': False,
              'protocol_sha256': protocol_hash, 'models_frozen_sha256': models_hash,
              'source_scene_exclusion': exclusion['status'],
              'declared_source_exact_layout_exclusion_established': exclusion['status'] == 'excluded_hash_bound_native_declarations',
              'analysis_scope': {'composition': 'Single composition-order pilot: count is confounded with which training groups are added.',
                                 'scene_independence': 'The same distinct benchmark scenes are reused for every count; condition-by-scene runs are not independent scenes.',
                                 'stratification': 'Nominal and actuator-stress outcomes are reported separately. Combined counts and clocks are execution totals, not a pooled performance estimate.',
                                 'novelty': 'Only exact source object/target layout exclusion is checked relative to supplied hash-bound native declarations; no out-of-distribution or human-transfer claim.',
                                 'modality': 'The movement candidate learns aligned numerical actions; the RGB predictor does not control the arm.'},
              'teacher': {'outcomes': teachers, 'planned_scenes': len(cases),
                          'successes': sum(row['success'] for row in teachers),
                          'errors': sum(row['status'] == 'error' for row in teachers),
                          'episode_wall_seconds': sum(row.get('episode_wall_seconds', 0.) for row in teachers),
                          'measurements': _measurement_summary(teachers, 'teacher'),
                          'groups': {group: {'planned_scenes': len(values),
                                             'successes': sum(row['success'] for row in values),
                                             'errors': sum(row['status'] == 'error' for row in values),
                                             'measurements': _measurement_summary(values, 'teacher')}
                                     for group in sorted({row['group'] for row in teachers})
                                     for values in [[row for row in teachers if row['group'] == group]]}},
              'comparisons': comparisons, 'benchmark_wall_seconds': time.perf_counter() - started,
              'timing_scope': 'Benchmark function wall time includes arm dependency imports, source validation, scene planning, input copies, frozen plans, fits, installation, native evaluations and artifact output before the final report. Process startup, top-level module imports and final report writing are excluded. Pipeline and native episode clocks retain their own narrower scopes.',
              'limitations': ['Training group counts and numerical action sample counts are different evidence units.',
                              'The same frozen source dev/test episodes and benchmark scenes are shared across counts; they are not independent repeated studies.',
                              'Scene exclusion is relative to hash-bound supplier declarations when supplied; otherwise source-scene disjointness is unestablished.',
                              'Actuator-stress failures can be shared physical/controller limits, not a demonstrated need for more labels.',
                              'Privileged native simulator state and authored phases, gripper/contact logic and recovery remain in use.',
                              'No raw human-video action bridge, minimum demonstration count, physical robot skill or guaranteed performance is established.',
                              'Aggregate results never authorize an untested scene; installed execution still needs its exact current-runtime qualification.']}
    _write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Actual prepared dataset.json with aligned action sidecars.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', help='Nested whole training-group counts; default eligible 2, 4, 7.')
    parser.add_argument('--seed', type=int, default=19043)
    parser.add_argument('--nominal', type=int, default=8)
    parser.add_argument('--stress', type=int, default=4)
    parser.add_argument('--task-source', type=Path)
    parser.add_argument('--grounding', type=Path)
    parser.add_argument('--source-evidence', type=Path, help='Hash-bound native evidence JSON directory for source-layout exclusion.')
    args = parser.parse_args()
    report = benchmark(args.dataset, args.output, counts=args.counts, seed=args.seed,
                       nominal=args.nominal, stress=args.stress, task_source=args.task_source,
                       grounding=args.grounding, source_evidence=args.source_evidence)
    print(json.dumps({'output': str(args.output), 'scope': report['scope'],
                      'source_scene_exclusion': report['source_scene_exclusion'],
                      'comparisons': [{'training_group_count': row['training_group_count'], **row['summary']}
                                      for row in report['comparisons']]}))


if __name__ == '__main__':
    main()
