"""Group-held-out laptop pilot for the authored 20 ms actuator transition.

This is an offline predictor experiment, not a shoe policy or physical dynamics
qualification. A group contains three fresh, related simulation resets; groups,
not individual transitions or sibling interventions, define the split.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
try:
    import resource
except ImportError:  # Windows has no process-lifetime getrusage interface.
    resource = None
import sys
from time import perf_counter, perf_counter_ns

from .episode_tape import CASES, MAX_BYTES, SOURCE_FILES, canonical, digest, record_episode, validate_tape, write_tape

CORPUS_KIND = 'dvidia.actuator-transition-corpus.v1'
REPORT_KIND = 'dvidia.actuator-transition-pilot.v1'
ALPHAS = (.0001, .01, 1., 100.)
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_GROUPS = 60
MAX_TOTAL_TAPE_BYTES = 512 * 1024 * 1024
MODEL_NAMES = ('action', 'state', 'persistence', 'constant_velocity', 'training_mean')
SPLITS = ('train', 'development', 'test')
BASE_CONFIG = {
    'timestep': .001, 'control_dt': .02, 'horizon': 18., 'settle_time': .4,
    'object_position': [.42, -.04, .31], 'target_position': [.54, .10, .31],
    'object_half_size': [.02, .02, .02], 'object_mass': .04, 'object_friction': .8,
    'table_height': .29, 'position_tolerance': .025, 'dwell_seconds': .25,
    'scene_jitter': .006, 'joint_torque_limit': 40., 'jaw_force_limit': 15.,
    'joint_target_rate_limit': 3., 'jaw_width_rate_limit': .16, 'gravity': 9.81,
}
LIMITATIONS = [
    'Authored rigid-box simulator with privileged joint telemetry; no shoe footage or shoe physics.',
    'Predicts the next 20 ms joint-position and jaw-gap deltas, not object motion, contact force or a control policy.',
    'Held-out results are a small grouped research test, not independent physical robot qualification.',
    'Models use before-state and optionally issued actuator commands; phases, cases and simulator configuration are not inputs.',
    'Compute-only inference timings exclude sensing, transport, actuator response and operating-system deadline guarantees.',
    'Hashes verify internal consistency, not provenance authenticity or measured physical accuracy.',
]


def _json(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('A bounded regular JSON file is required.')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('JSON file exceeds its byte limit.')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field.')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON value.')
    try:
        document = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        canonical(document)
    except (UnicodeError, RecursionError, json.JSONDecodeError, TypeError, OverflowError) as exc:
        raise ValueError('Finite UTF-8 JSON is required.') from exc
    return document, raw


def _write(path, document):
    raw = canonical(document) + b'\n'
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return {'file': Path(path).name, 'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}


def _fresh(output):
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise ValueError('Output already exists; use a fresh directory.')
    path.mkdir(parents=True)
    return path


def _id(value):
    return type(value) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}', value) is not None


def _hash(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _sources():
    return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sorted(SOURCE_FILES)}


def generate_corpus(output, seed_start=311000, groups=12):
    """Collect all predeclared attempts; failed recordings remain in the manifest."""
    if type(groups) is not int or not 6 <= groups <= MAX_GROUPS:
        raise ValueError(f'Use 6–{MAX_GROUPS} reset groups.')
    if type(seed_start) is not int or not 0 <= seed_start <= 2**31-groups:
        raise ValueError('Seed start must be a bounded nonnegative integer.')
    output = _fresh(output)
    (output/'tapes').mkdir()
    held = max(1, groups//6)
    train_groups = groups-2*held
    source_hashes = _sources()
    manifest = {
        'kind': CORPUS_KIND, 'schema_version': 1, 'scope': 'simulation-only',
        'physical_robot_ready': False, 'shoe_skill_ready': False, 'status': 'collecting',
        'protocol': {
            'revision': 'grouped-native-actuator-20ms-v1', 'cases': list(CASES),
            'split_unit': 'predeclared reset group; all sibling interventions stay together',
            'variation': 'seeded scene jitter plus predeclared bounded group mass/friction; neither is a model feature',
            'split_counts': {'train': train_groups, 'development': held, 'test': held},
            'source_files_sha256': source_hashes,
        },
        'groups': [],
    }
    for index in range(groups):
        group_id = f'reset-{index:03d}'
        split = 'train' if index < train_groups else 'development' if index < train_groups+held else 'test'
        config = deepcopy(BASE_CONFIG)
        # Deterministic and declared before simulation, not fitted to observed outcomes.
        config['object_mass'] = .035 + .010*index/max(1, groups-1)
        config['object_friction'] = .70 + .20*index/max(1, groups-1)
        seed = seed_start+index
        group = {'group_id': group_id, 'split': split, 'seed': seed,
                 'session_id': f'session-{seed}', 'reset_lineage_id': f'native-reset-{seed}',
                 'configuration': config, 'configuration_sha256': digest(config), 'attempts': []}
        for case in CASES:
            group['attempts'].append({'episode_id': f'{group_id}-{case}',
                'source_recording_id': f'recording-{seed}-{case}', 'case': case,
                'status': 'planned', 'path': None, 'sha256': None, 'bytes': None,
                'error': None, 'recording_elapsed_seconds': None, 'provenance': None, 'outcome': None})
        manifest['groups'].append(group)
    # This declaration is immutable evidence of the plan before any attempt starts.
    _write(output/'collection-plan.json', manifest)
    start = perf_counter()
    for group in manifest['groups']:
        for attempt in group['attempts']:
            attempt_start = perf_counter()
            try:
                from .arm_env import ArmConfig
                tape = record_episode(episode_id=attempt['episode_id'], source_recording_id=attempt['source_recording_id'],
                                      session_id=group['session_id'], seed=group['seed'], case=attempt['case'],
                                      config=ArmConfig(**group['configuration']))
                path = output/'tapes'/f'{attempt["episode_id"]}.json'
                receipt = write_tape(path, tape)
                attempt.update(status='completed', path=path.relative_to(output).as_posix(),
                               sha256=receipt['sha256'], bytes=receipt['bytes'],
                               provenance=deepcopy(tape['provenance']), outcome=deepcopy(tape['outcome']))
            except (OSError, ValueError, ImportError, RuntimeError) as exc:
                attempt.update(status='failed', error=f'{type(exc).__name__}: {str(exc)[:240]}')
            attempt['recording_elapsed_seconds'] = perf_counter()-attempt_start
    failures = sum(a['status'] != 'completed' for g in manifest['groups'] for a in g['attempts'])
    manifest['status'] = 'completed' if not failures else 'completed-with-failed-attempts'
    manifest['collection_elapsed_seconds'] = perf_counter()-start
    _write(output/'corpus.json', manifest)
    return manifest


def _manifest(path):
    manifest, raw = _json(path, MAX_MANIFEST_BYTES)
    expected = {'kind', 'schema_version', 'scope', 'physical_robot_ready', 'shoe_skill_ready',
                'status', 'protocol', 'groups', 'collection_elapsed_seconds'}
    if type(manifest) is not dict or set(manifest) != expected or manifest['kind'] != CORPUS_KIND or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        raise ValueError('Unsupported transition corpus manifest.')
    if manifest['scope'] != 'simulation-only' or manifest['physical_robot_ready'] is not False or manifest['shoe_skill_ready'] is not False:
        raise ValueError('Unsupported corpus qualification flags.')
    if manifest['status'] not in ('completed', 'completed-with-failed-attempts') or not _finite(manifest['collection_elapsed_seconds']) or manifest['collection_elapsed_seconds'] < 0:
        raise ValueError('Corpus must retain completed or failed recording attempts.')
    groups = manifest['groups']
    if type(groups) is not list or not 3 <= len(groups) <= MAX_GROUPS:
        raise ValueError('Corpus requires bounded train/development/test reset groups.')
    protocol = manifest['protocol']
    if type(protocol) is not dict or set(protocol) != {'revision', 'cases', 'split_unit', 'variation', 'split_counts', 'source_files_sha256'} or protocol['revision'] != 'grouped-native-actuator-20ms-v1' or protocol['cases'] != list(CASES):
        raise ValueError('Unsupported grouped corpus protocol.')
    if any(type(protocol[k]) is not str or not 1 <= len(protocol[k]) <= 200 for k in ('split_unit', 'variation')):
        raise ValueError('Bounded corpus protocol descriptions required.')
    hashes = protocol['source_files_sha256']
    if type(hashes) is not dict or set(hashes) != SOURCE_FILES or not all(_hash(v) for v in hashes.values()):
        raise ValueError('Source hashes are required.')
    counts = {split: 0 for split in SPLITS}
    lineage = {name: {} for name in ('group', 'seed', 'session', 'lineage', 'reset', 'recording', 'episode', 'tape')}
    total_bytes = 0
    paths = set()
    root = Path(path).resolve().parent
    def unique(name, value, group_id):
        previous = lineage[name].get(value)
        if previous is not None and previous != group_id:
            raise ValueError(f'Duplicate {name} lineage across reset groups; split independence is invalid.')
        lineage[name][value] = group_id
    for group in groups:
        if type(group) is not dict or set(group) != {'group_id', 'split', 'seed', 'session_id', 'reset_lineage_id', 'configuration', 'configuration_sha256', 'attempts'}:
            raise ValueError('Unsupported reset-group fields.')
        group_id, split = group['group_id'], group['split']
        if not all(_id(group[k]) for k in ('group_id', 'session_id', 'reset_lineage_id')) or split not in SPLITS or type(group['seed']) is not int or not 0 <= group['seed'] <= 2**31:
            raise ValueError('Bounded reset group identifiers and seed are required.')
        if group_id in lineage['group']:
            raise ValueError('Duplicate reset group.')
        unique('group', group_id, group_id)
        config = group['configuration']
        if type(config) is not dict or set(config) != set(BASE_CONFIG) or not _hash(group['configuration_sha256']) or digest(config) != group['configuration_sha256']:
            raise ValueError('Predeclared configuration hash mismatch.')
        for name, value in (('seed', group['seed']), ('session', group['session_id']), ('lineage', group['reset_lineage_id']),
                            ('reset', (group['seed'], group['configuration_sha256']))):
            unique(name, value, group_id)
        counts[split] += 1
        attempts = group['attempts']
        if type(attempts) is not list or len(attempts) != len(CASES) or {a.get('case') for a in attempts if type(a) is dict} != set(CASES):
            raise ValueError('Every group must retain all three intervention attempts.')
        for attempt in attempts:
            if type(attempt) is not dict or set(attempt) != {'episode_id', 'source_recording_id', 'case', 'status', 'path', 'sha256', 'bytes', 'error', 'recording_elapsed_seconds', 'provenance', 'outcome'}:
                raise ValueError('Unsupported recording-attempt fields.')
            if not _id(attempt['episode_id']) or not _id(attempt['source_recording_id']) or attempt['status'] not in ('completed', 'failed') or not _finite(attempt['recording_elapsed_seconds']) or attempt['recording_elapsed_seconds'] < 0:
                raise ValueError('Every attempt needs bounded identifiers, status and elapsed time.')
            for name, key in (('recording', 'source_recording_id'), ('episode', 'episode_id')):
                value = attempt[key]
                if value in lineage[name]:
                    raise ValueError(f'Duplicate {name} identifier or alias.')
                unique(name, value, group_id)
            if attempt['status'] == 'failed':
                if any(attempt[key] is not None for key in ('path', 'sha256', 'bytes', 'provenance', 'outcome')) or type(attempt['error']) is not str or not 1 <= len(attempt['error']) <= 300:
                    raise ValueError('Failed attempt must retain an error and no fabricated tape.')
                continue
            if attempt['error'] is not None or not _hash(attempt['sha256']) or type(attempt['bytes']) is not int or not 1 <= attempt['bytes'] <= MAX_BYTES:
                raise ValueError('Completed recording needs a bounded byte hash.')
            name = attempt['path']
            if type(name) is not str or not re.fullmatch(r'tapes/[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\.json', name) or name in paths:
                raise ValueError('Tape must have a unique, relative corpus path.')
            paths.add(name)
            tape_path = root/name
            # No symlink at any component and no escaping the corpus root.
            if any(p.is_symlink() for p in (root/'tapes', tape_path)) or not tape_path.is_file() or tape_path.resolve().parent != (root/'tapes').resolve():
                raise ValueError('Tape path must remain a regular file inside the corpus.')
            if tape_path.stat().st_size != attempt['bytes']:
                raise ValueError('Tape byte count differs from manifest.')
            total_bytes += attempt['bytes']
            if total_bytes > MAX_TOTAL_TAPE_BYTES:
                raise ValueError('Corpus exceeds the 512 MiB tape budget.')
            provenance = attempt['provenance']
            if type(provenance) is not dict or provenance.get('episode_id') != attempt['episode_id'] or provenance.get('source_recording_id') != attempt['source_recording_id'] or provenance.get('session_id') != group['session_id'] or provenance.get('seed') != group['seed'] or provenance.get('case') != attempt['case'] or provenance.get('configuration_sha256') != group['configuration_sha256'] or provenance.get('source_files_sha256') != hashes or not _hash(provenance.get('scene_sha256')):
                raise ValueError('Recording provenance differs from its declared reset group.')
            unique('tape', attempt['sha256'], group_id)
    if type(protocol['split_counts']) is not dict or protocol['split_counts'] != counts or any(count == 0 for count in counts.values()):
        raise ValueError('Split group counts must match predeclared groups.')
    failed = sum(a['status'] == 'failed' for g in groups for a in g['attempts'])
    if (manifest['status'] == 'completed') != (failed == 0):
        raise ValueError('Corpus status disagrees with failed attempt denominator.')
    return manifest, raw, root


def _load_attempt(root, group, attempt):
    path = root/attempt['path']
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES+1)
    if len(raw) != attempt['bytes'] or sha256(raw).hexdigest() != attempt['sha256']:
        raise ValueError('Tape source hash differs from manifest.')
    tape = validate_tape(raw)
    if tape['provenance'] != attempt['provenance'] or tape['configuration'] != group['configuration'] or tape['outcome'] != attempt['outcome']:
        raise ValueError('Tape content differs from declared corpus provenance or outcome.')
    return tape


def _metrics(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError('Predictions must retain every evaluation row.')
    if not rows:
        return {'rows': 0, 'joint_rmse_rad': None, 'jaw_rmse_m': None, 'joint_mae_rad': None,
                'jaw_mae_m': None, 'joint_max_abs_error_rad': None, 'jaw_max_abs_error_m': None}
    errors = [[p-t for p, t in zip(prediction, row['target_delta'])] for row, prediction in zip(rows, predictions)]
    if any(len(p) != 7 or not all(_finite(x) for x in p) for p in predictions):
        raise ValueError('Finite seven-output predictions required.')
    joint = [abs(x) for error in errors for x in error[:6]]
    jaw = [abs(error[6]) for error in errors]
    return {'rows': len(rows), 'joint_rmse_rad': math.sqrt(sum(x*x for x in joint)/len(joint)),
            'jaw_rmse_m': math.sqrt(sum(x*x for x in jaw)/len(jaw)),
            'joint_mae_rad': sum(joint)/len(joint), 'jaw_mae_m': sum(jaw)/len(jaw),
            'joint_max_abs_error_rad': max(joint), 'jaw_max_abs_error_m': max(jaw)}


def _score(metric):
    if not metric['rows']:
        raise ValueError('Development rows are required for fixed alpha selection.')
    return metric['joint_rmse_rad']/.01 + metric['jaw_rmse_m']/.001


def _weights(rows):
    counts = {}
    for row in rows:
        counts[row['episode_id']] = counts.get(row['episode_id'], 0)+1
    # Each retained episode contributes the same total, independent of its length.
    return [len(rows)/(len(counts)*counts[row['episode_id']]) for row in rows]


def _predictions(name, model, rows, mean):
    from .transition_model import predict_features
    if name in ('action', 'state'):
        return [predict_features(model, row['features']) for row in rows]
    if name == 'constant_velocity':
        return [[.02*x for x in row['features'][6:12]]+[0.] for row in rows]
    if name == 'training_mean':
        return [mean[:] for row in rows]
    return [[0.]*7 for row in rows]


def _quantile(values, q):
    values = sorted(values)
    coordinate = (len(values)-1)*q
    low = int(coordinate)
    high = min(low+1, len(values)-1)
    return values[low]+(values[high]-values[low])*(coordinate-low)


def _timing(model, rows):
    from .transition_model import predict_features
    for index in range(100):
        predict_features(model, rows[index % len(rows)]['features'])
    times = []
    for index in range(1000):
        features = rows[index % len(rows)]['features']
        start = perf_counter_ns()
        predict_features(model, features)
        times.append(perf_counter_ns()-start)
    return {'warmup_calls': 100, 'measured_calls': 1000, 'p50_microseconds': _quantile(times, .5)/1000,
            'p95_microseconds': _quantile(times, .95)/1000, 'worst_microseconds': max(times)/1000,
            'scope': 'compute-only predict_features; pre-extracted held-out input vectors; no sensing or transport'}


def _matched_branch(rows_by_case, models, mean):
    from .transition_model import predict_features
    nominal = { (row['group_id'], row['index']): row for row in rows_by_case.get('nominal', []) }
    for changed in rows_by_case.get('release-after-lift', []):
        reference = nominal.get((changed['group_id'], changed['index']))
        if reference is None or reference['features'][:13] != changed['features'][:13] or reference['features'][13:19] != changed['features'][13:19] or reference['features'][19] == changed['features'][19]:
            continue
        result = {'status': 'matched', 'group_id': changed['group_id'], 'index': changed['index'],
                  'before_time_seconds': changed['before_time_seconds'], 'before_state_identical': True,
                  'joint_command_identical': True, 'issued_jaw_gap_m': {'nominal': reference['features'][19], 'release': changed['features'][19]},
                  'observed_delta': {'nominal': reference['target_delta'], 'release': changed['target_delta']},
                  'prediction_delta': {}}
        for name, model in models.items():
            result['prediction_delta'][name] = {'nominal': predict_features(model, reference['features']), 'release': predict_features(model, changed['features'])}
        return result
    return {'status': 'unavailable', 'reason': 'No retained nominal/release transition has exactly identical before-state and joint command with a changed issued jaw command.'}


def _hardware():
    import numpy
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
    multiplier = 1 if sys.platform == 'darwin' else 1024
    return {'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor() or platform.machine(),
            'logical_cpu_count': os.cpu_count(), 'python': platform.python_version(), 'numpy': numpy.__version__,
            'gpu_used': False, 'process_lifetime_peak_rss_bytes': int(rss*multiplier) if rss is not None else None,
            'memory_scope': 'process-lifetime high-water mark; includes earlier work in this process; not isolated learner memory' if rss is not None else 'process-lifetime RSS unavailable on this platform'}


def run_pilot(manifest_path, output):
    """Fit on train, select on development, freeze artifacts, then parse test tapes."""
    output = _fresh(output)
    start = perf_counter()
    report = {'kind': REPORT_KIND, 'schema_version': 1, 'scope': 'simulation-only',
              'physical_robot_ready': False, 'shoe_skill_ready': False, 'status': 'running',
              'attempts': [], 'counts': {}, 'metrics': {}, 'model_artifacts': {},
              'freeze': None, 'hardware': None, 'timings': {}, 'limitations': LIMITATIONS[:],
              'qualification': 'unqualified research pilot; internal consistency only', 'stage': 'manifest-validation'}
    try:
        from .transition_model import extract_rows, fit_model, load_model, predict_features, write_model
        manifest, manifest_raw, corpus_root = _manifest(manifest_path)
        report['corpus_manifest_sha256'] = sha256(manifest_raw).hexdigest()
        with (output/'corpus-manifest.json').open('xb') as stream:
            stream.write(manifest_raw)
        manifest_receipt = {'file': 'corpus-manifest.json', 'sha256': sha256(manifest_raw).hexdigest(), 'bytes': len(manifest_raw)}
        report['source_corpus'] = {'snapshot': manifest_receipt, 'original_tapes_included': False,
                                 'original_manifest_sha256': report['corpus_manifest_sha256']}
        report['protocol'] = {'control_dt_seconds': .02, 'target': 'next actuator joint-position and jaw-gap deltas',
            'alpha_grid': list(ALPHAS), 'development_score': 'joint_rmse_rad/0.01 + jaw_rmse_m/0.001',
            'selection_tie_break': 'first alpha in ascending fixed grid', 'training_weighting': 'equal total weight per retained training episode',
            'normalization': 'training rows only; model core records means/scales', 'test_used_for_selection': False,
            'feature_modes': {'action': 'before q/qdot/jaw + issued joint/jaw targets', 'state': 'before q/qdot/jaw only'},
            'excluded_inputs': ['case', 'phase', 'simulation configuration', 'future state', 'applied end-of-interval controls', 'object state'],
            'slice_definitions': {'moving': 'any measured abs joint delta > 0.0001 rad', 'jaw_changing': 'abs measured jaw delta > 0.00001 m'}}
        report['counts'] = {'groups': {split: sum(g['split'] == split for g in manifest['groups']) for split in SPLITS},
                            'attempts': len(manifest['groups'])*len(CASES), 'completed_recordings': 0, 'failed_recordings': 0,
                            'rows': {split: 0 for split in SPLITS}, 'excluded_steps': {split: 0 for split in SPLITS},
                            'attempted_steps': {split: 0 for split in SPLITS}, 'simulation_successes': 0, 'simulation_unsuccessful': 0}
        by_split = {split: [] for split in SPLITS}
        load_seconds = {split: 0. for split in SPLITS}
        extraction_seconds = {split: 0. for split in SPLITS}
        receipt_by_episode = {}
        for group in manifest['groups']:
            for attempt in group['attempts']:
                receipt = {'group_id': group['group_id'], 'split': group['split'], 'episode_id': attempt['episode_id'],
                    'case': attempt['case'], 'recording_status': attempt['status'], 'source_sha256': attempt['sha256'],
                    'source_bytes': attempt['bytes'], 'recording_error': attempt['error'],
                    'simulation_outcome': deepcopy(attempt['outcome']), 'rows': 0, 'attempted_steps': 0, 'exclusions': [],
                    'validation_status': 'not_checked' if attempt['status'] == 'completed' else 'failed_recording'}
                report['attempts'].append(receipt)
                receipt_by_episode[attempt['episode_id']] = receipt
                report['counts']['completed_recordings' if attempt['status'] == 'completed' else 'failed_recordings'] += 1
                if attempt['outcome'] is not None:
                    report['counts']['simulation_successes' if attempt['outcome']['result']['success'] else 'simulation_unsuccessful'] += 1
        def consume(split):
            for group in manifest['groups']:
                if group['split'] != split:
                    continue
                for attempt in group['attempts']:
                    receipt = receipt_by_episode[attempt['episode_id']]
                    if attempt['status'] == 'failed':
                        continue
                    load_start = perf_counter()
                    try:
                        tape = _load_attempt(corpus_root, group, attempt)
                    except (OSError, ValueError) as exc:
                        receipt.update(validation_status='error', validation_error=str(exc)[:200])
                        raise
                    receipt['validation_status'] = 'validated'
                    load_seconds[split] += perf_counter()-load_start
                    extraction_start = perf_counter()
                    extracted = extract_rows(tape)
                    extraction_seconds[split] += perf_counter()-extraction_start
                    for row in extracted['rows']:
                        row['group_id'] = group['group_id']
                    by_split[split].extend(extracted['rows'])
                    receipt.update(rows=len(extracted['rows']), attempted_steps=extracted['attempted_steps'], exclusions=extracted['exclusions'])
                    report['counts']['rows'][split] += len(extracted['rows'])
                    report['counts']['excluded_steps'][split] += len(extracted['exclusions'])
                    report['counts']['attempted_steps'][split] += extracted['attempted_steps']
        report['stage'] = 'train-development-loading'
        consume('train')
        consume('development')
        train, development = by_split['train'], by_split['development']
        if not train or not development:
            raise ValueError('Both training and development require retained valid 20 ms rows.')
        weights = _weights(train)
        mean = [sum(w*row['target_delta'][i] for w, row in zip(weights, train))/sum(weights) for i in range(7)]
        report['training_mean_delta'] = mean
        models, selections = {}, {}
        fit_start = perf_counter()
        report['stage'] = 'training-selection'
        for mode in ('action', 'state'):
            trials = []
            candidates = []
            for alpha in ALPHAS:
                model = fit_model([{key: value for key, value in row.items() if key != 'group_id'} for row in train], alpha=alpha, feature_mode=mode, sample_weights=weights)
                metric = _metrics(development, [predict_features(model, row['features']) for row in development])
                score = _score(metric)
                trials.append({'alpha': alpha, 'development_score': score, 'metrics': metric})
                candidates.append(model)
            best = min(range(len(trials)), key=lambda i: trials[i]['development_score'])
            models[mode] = candidates[best]
            selections[mode] = {'alpha': trials[best]['alpha'], 'trials': trials}
        report['timings']['fit_and_development_selection_seconds'] = perf_counter()-fit_start
        report['selection'] = selections
        report['stage'] = 'freeze-before-test'
        for mode, model in models.items():
            path = output/f'{mode}.model.json'
            write_model(path, model)
            raw = path.read_bytes()
            report['model_artifacts'][mode] = {'file': path.name, 'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}
        freeze = {'kind': 'dvidia.actuator-transition-freeze.v1', 'schema_version': 1,
                  'corpus_manifest_sha256': report['corpus_manifest_sha256'], 'models': deepcopy(report['model_artifacts']),
                  'selection': deepcopy(selections), 'training_mean_delta': mean,
                  'held_out_transition_targets_read': False, 'elapsed_seconds_at_freeze': perf_counter()-start,
                  'training_sources': [a['sha256'] for g in manifest['groups'] if g['split'] == 'train' for a in g['attempts'] if a['status'] == 'completed'],
                  'development_sources': [a['sha256'] for g in manifest['groups'] if g['split'] == 'development' for a in g['attempts'] if a['status'] == 'completed'],
                  'test_sources': [a['sha256'] for g in manifest['groups'] if g['split'] == 'test' for a in g['attempts'] if a['status'] == 'completed']}
        report['freeze'] = _write(output/'freeze.json', freeze)
        # Only now may held-out transition targets be parsed. Manifest terminal outcomes were visible during preflight. Reload the frozen model files.
        frozen = {mode: load_model(output/artifact['file']) for mode, artifact in report['model_artifacts'].items()}
        report['stage'] = 'held-out-evaluation'
        report['timings']['held_out_loading_started_elapsed_seconds'] = perf_counter()-start
        consume('test')
        test = by_split['test']
        if not test:
            raise ValueError('Held-out split contains no valid 20 ms transition rows.')
        report['metrics']['test'] = {}
        report['metrics']['development'] = {}
        predictions = {}
        for name in MODEL_NAMES:
            model = frozen.get(name)
            predictions[name] = _predictions(name, model, test, mean)
            report['metrics']['test'][name] = _metrics(test, predictions[name])
            report['metrics']['development'][name] = _metrics(development, _predictions(name, model, development, mean))
        def grouped(key):
            result = {}
            for value in sorted({row[key] for row in test}):
                indices = [i for i, row in enumerate(test) if row[key] == value]
                result[value] = {name: _metrics([test[i] for i in indices], [predictions[name][i] for i in indices]) for name in MODEL_NAMES}
            return result
        report['metrics']['per_case'] = grouped('case')
        report['metrics']['per_group'] = grouped('group_id')
        report['metrics']['slices'] = {}
        masks = {'moving': lambda row: any(abs(x) > .0001 for x in row['target_delta'][:6]),
                 'jaw_changing': lambda row: abs(row['target_delta'][6]) > .00001}
        for slice_name, predicate in masks.items():
            indices = [i for i, row in enumerate(test) if predicate(row)]
            report['metrics']['slices'][slice_name] = {name: _metrics([test[i] for i in indices], [predictions[name][i] for i in indices]) for name in MODEL_NAMES}
        by_case = {case: [row for row in test if row['case'] == case] for case in CASES}
        report['matched_branch'] = _matched_branch(by_case, frozen, mean)
        report['timings']['inference'] = {mode: _timing(model, test) for mode, model in frozen.items()}
        report['timings']['source_load_and_validation_seconds'] = load_seconds
        report['timings']['feature_and_target_extraction_seconds'] = extraction_seconds
        report['timings']['source_collection_seconds'] = manifest['collection_elapsed_seconds']
        report['counts']['case_attempts'] = {case: sum(a['case'] == case for a in report['attempts']) for case in CASES}
        report['counts']['case_rows'] = {case: len(by_case[case]) for case in CASES}
        report['artifact_bytes'] = sum(a['bytes'] for a in report['model_artifacts'].values())
        report['hardware'] = _hardware()
        report['status'], report['stage'] = 'completed', 'complete'
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        report['status'] = 'error'
        report['error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
    report['timings']['pipeline_elapsed_seconds_before_final_receipt'] = perf_counter()-start
    report_receipt = _write(output/'report.json', report)
    artifacts = {report_receipt['file']: report_receipt}
    for name in ('corpus-manifest.json', 'freeze.json', 'action.model.json', 'state.model.json'):
        path = output/name
        if path.is_file():
            raw = path.read_bytes()
            artifacts[name] = {'file': name, 'sha256': sha256(raw).hexdigest(), 'bytes': len(raw)}
    _write(output/'integrity.json', {'kind': 'dvidia.actuator-transition-integrity.v1', 'schema_version': 1,
                                    'artifacts': artifacts, 'authenticity_verified': False, 'accuracy_verified': False})
    return report


def _inspect_pilot(output_dir):
    """Check local artifact consistency without upgrading it to accuracy evidence."""
    from .transition_model import load_model
    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Pilot output must be a regular directory.')
    integrity, _ = _json(root/'integrity.json', MAX_MANIFEST_BYTES)
    if type(integrity) is not dict or set(integrity) != {'kind', 'schema_version', 'artifacts', 'authenticity_verified', 'accuracy_verified'} or integrity['kind'] != 'dvidia.actuator-transition-integrity.v1' or integrity['schema_version'] != 1 or integrity['authenticity_verified'] is not False or integrity['accuracy_verified'] is not False:
        raise ValueError('Unsupported pilot integrity receipt.')
    allowed = {'report.json', 'corpus-manifest.json', 'freeze.json', 'action.model.json', 'state.model.json'}
    artifacts = integrity['artifacts']
    if type(artifacts) is not dict or not set(artifacts) <= allowed or 'report.json' not in artifacts:
        raise ValueError('Unsupported pilot artifact paths.')
    for name, artifact in artifacts.items():
        if type(artifact) is not dict or set(artifact) != {'file', 'sha256', 'bytes'} or artifact['file'] != name or not _hash(artifact['sha256']) or type(artifact['bytes']) is not int or not 0 < artifact['bytes'] <= MAX_REPORT_BYTES:
            raise ValueError('Bounded artifact receipts are required.')
        _, raw = _json(root/name, MAX_REPORT_BYTES)
        if len(raw) != artifact['bytes'] or sha256(raw).hexdigest() != artifact['sha256']:
            raise ValueError('Pilot artifact hash mismatch.')
    report, _ = _json(root/'report.json', MAX_REPORT_BYTES)
    if type(report) is not dict or report.get('kind') != REPORT_KIND or report.get('schema_version') != 1 or report.get('physical_robot_ready') is not False or report.get('shoe_skill_ready') is not False or report.get('status') not in ('completed', 'error'):
        raise ValueError('Unsupported pilot report.')
    if report['status'] == 'completed':
        if set(artifacts) != allowed or set(report.get('model_artifacts', {})) != {'action', 'state'}:
            raise ValueError('Completed pilot must include both frozen models and corpus snapshot.')
        freeze, _ = _json(root/'freeze.json', MAX_REPORT_BYTES)
        if freeze.get('kind') != 'dvidia.actuator-transition-freeze.v1' or freeze.get('held_out_transition_targets_read') is not False or freeze.get('models') != report['model_artifacts'] or freeze.get('selection') != report.get('selection') or freeze.get('training_mean_delta') != report.get('training_mean_delta') or freeze.get('corpus_manifest_sha256') != report.get('corpus_manifest_sha256'):
            raise ValueError('Frozen selection differs from the evaluation report.')
        if report.get('freeze') != artifacts['freeze.json'] or report['source_corpus']['snapshot'] != artifacts['corpus-manifest.json'] or report['source_corpus']['original_tapes_included'] is not False:
            raise ValueError('Pilot frozen or corpus receipt differs from retained artifacts.')
        for mode in ('action', 'state'):
            artifact = report['model_artifacts'][mode]
            if artifact != artifacts[f'{mode}.model.json']:
                raise ValueError('Model artifact differs from the freeze receipt.')
            model = load_model(root/artifact['file'])
            if model['feature_mode'] != mode or model['alpha'] != report['selection'][mode]['alpha'] or model['training']['rows'] != report['counts']['rows']['train']:
                raise ValueError('Frozen model disagrees with selected mode or alpha.')
        manifest, _ = _json(root/'corpus-manifest.json', MAX_MANIFEST_BYTES)
        expected_groups = {split: sum(g['split'] == split for g in manifest['groups']) for split in SPLITS}
        attempts = report['attempts']
        if len(attempts) != len(manifest['groups'])*len(CASES) or report['counts']['groups'] != expected_groups:
            raise ValueError('Pilot recording denominator differs from retained corpus.')
        if report['corpus_manifest_sha256'] != artifacts['corpus-manifest.json']['sha256'] or report['source_corpus']['original_manifest_sha256'] != report['corpus_manifest_sha256']:
            raise ValueError('Original corpus manifest hash differs from its retained byte snapshot.')
        declared = {a['episode_id']: (g, a) for g in manifest['groups'] for a in g['attempts']}
        if len(declared) != len(attempts) or {a['episode_id'] for a in attempts} != set(declared):
            raise ValueError('Recording attempt identifiers differ from the corpus.')
        for receipt in attempts:
            group, source = declared[receipt['episode_id']]
            expected = {'group_id': group['group_id'], 'split': group['split'], 'case': source['case'],
                        'recording_status': source['status'], 'source_sha256': source['sha256'],
                        'source_bytes': source['bytes'], 'recording_error': source['error'], 'simulation_outcome': source['outcome']}
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError('Attempt receipt differs from the retained corpus declaration.')
            if type(receipt.get('rows')) is not int or receipt['rows'] < 0 or type(receipt.get('attempted_steps')) is not int or receipt['attempted_steps'] < 0 or type(receipt.get('exclusions')) is not list or receipt['rows']+len(receipt['exclusions']) != receipt['attempted_steps']:
                raise ValueError('Attempt rows and exclusions must cover every attempted step.')
            expected_validation = 'validated' if source['status'] == 'completed' else 'failed_recording'
            if receipt.get('validation_status') != expected_validation:
                raise ValueError('Completed pilot contains an unchecked source recording.')
        for split, key in (('train', 'training_sources'), ('development', 'development_sources'), ('test', 'test_sources')):
            expected = [a['sha256'] for g in manifest['groups'] if g['split'] == split for a in g['attempts'] if a['status'] == 'completed']
            if freeze.get(key) != expected:
                raise ValueError('Freeze source hashes differ from the retained grouped corpus.')
        completed = sum(a['recording_status'] == 'completed' for a in attempts)
        if report['counts']['attempts'] != len(attempts) or report['counts']['completed_recordings'] != completed or report['counts']['failed_recordings'] != len(attempts)-completed:
            raise ValueError('Failed recording denominators are inconsistent.')
        for split in SPLITS:
            subset = [a for a in attempts if a['split'] == split]
            rows = sum(a['rows'] for a in subset)
            if report['counts']['rows'][split] != rows or report['counts']['excluded_steps'][split] != sum(len(a['exclusions']) for a in subset) or report['counts']['attempted_steps'][split] != sum(a['attempted_steps'] for a in subset):
                raise ValueError('Pilot row and exclusion counts are inconsistent.')
            if split in ('test', 'development') and any(metric['rows'] != rows for metric in report['metrics'][split].values()):
                raise ValueError('Evaluation metrics omit retained rows.')
        if freeze['elapsed_seconds_at_freeze'] > report['timings']['held_out_loading_started_elapsed_seconds']:
            raise ValueError('Held-out loading preceded the frozen selection.')
    return {'kind': 'dvidia.actuator-transition-pilot-inspection.v1', 'status': report['status'],
            'integrity': 'verified', 'verification_scope': 'internal artifact consistency only; original source tapes not included',
            'authenticity_verified': False, 'accuracy_verified': False, 'physical_robot_ready': False,
            'shoe_skill_ready': False, 'counts': report.get('counts', {}), 'artifact_count': len(artifacts)}


def inspect_pilot(output_dir):
    """Return bounded internal-consistency evidence; malformed data fails cleanly."""
    try:
        return _inspect_pilot(output_dir)
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('Malformed pilot artifact schema.') from exc
