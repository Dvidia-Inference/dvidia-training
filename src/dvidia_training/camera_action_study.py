"""Registered local camera-to-action study; failures never become qualifications.

Every native attempt has a literal registered identity. Repeating a stage may
finish its previously unattempted roster, but never overwrites or retries an
existing attempt. Study inspection imports no renderer and issues no commands.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from functools import wraps
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
import zipfile

import numpy as np

SCHEMA = 'dvidia.camera-action-study/v1'
SEED = 311401
BOOTSTRAP_SEED = 311402
GIB = 1024 ** 3
STUDY_MAX_BYTES = 120 * GIB
DISK_RESERVE_BYTES = 30 * GIB
ATTEMPT_HEADROOM_BYTES = 320 * 1024 ** 2
GOALS = ((.35, .08, .08), (.40, -.08, .08))
SOURCES = ('camera_action_study.py', 'camera_action_capture.py',
           'camera_action_model.py', 'camera_action_export.py',
           'piper_rgbd_tracker.py', 'piper_action_capture.py',
           'action_demonstration.py', 'piper_task.py', 'reference_task.py',
           'reference_sim.py', 'piper_imitation_score.py', 'piper_motor_primitive.py',
           'piper_feedback_env.py', 'piper_feedback_score.py',
           'piper_imitation.py', 'robot_profiles.py', 'robot_profiles.lock.json',
           'episode_tape.py', 'embodiment_calibration.py')
STAGES = ('diagnostic', 'demonstrations', 'train-initial',
          'development-initial', 'corrections', 'train-corrected',
          'development-corrected', 'confirmation', 'negative')
LIMITATIONS = [
    'Synthetic robot demonstrations from an authored teacher; not human-video motor recovery.',
    'One simulated PiPER, a known rigid block, two known goals and bounded starting-position interpolation.',
    'The frozen visual estimator can abstain under gripper occlusion or object tilt; historical anchors are explicitly historical.',
    'Synthetic camera time and host computation are recorded separately; no physical sensor delivery or real-time hardware claim.',
    'Contact, friction, depth accuracy and actuator models remain unqualified on physical hardware.',
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024 ** 2:
        raise ValueError('Expected a bounded ordinary JSON artifact.')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field.')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON number.')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=reject)


def write(path, value):
    _create(Path(path), canonical(value) + b'\n')


def _create(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.camera-action-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _root(root):
    root = Path(root).absolute()
    if any(p.is_symlink() and str(p) not in ('/tmp', '/var') for p in (root, *root.parents)):
        raise ValueError('Study paths cannot traverse symbolic links.')
    return root


def runtime_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in SOURCES}


def _historical_starts():
    # Previously published original imitation starts; random fresh positions
    # use micrometre precision and are explicitly checked against this roster.
    return {(x, y) for x in (.284, .294, .304, .314) for y in (-.045, -.035, -.025, -.015)} | {
        (x, y) for x in (.289, .309) for y in (-.04, -.02)} | {
        (.287, -.032), (.291, -.018), (.299, -.041), (.303, -.028), (.311, -.038), (.307, -.022)}


def registered_cases(seed=SEED):
    if type(seed) is not int or seed != SEED:
        raise ValueError('This v0.1 study uses the registered seed 311401.')
    rng = np.random.default_rng(seed)
    seen = _historical_starts()
    points = []
    while len(points) < 72:
        point = tuple(np.round(rng.uniform([.287, -.042], [.311, -.018]), 6).tolist())
        if point not in seen:
            seen.add(point)
            points.append(point)
    splits = {}
    offset = 0
    for split, count in (('train', 32), ('development', 8), ('confirmation', 32)):
        splits[split] = []
        for index, (x, y) in enumerate(points[offset:offset + count]):
            group = f'{split}-{index:02}'
            for goal, position in enumerate(GOALS):
                splits[split].append(dict(id=f'{group}-goal{goal}', group_id=group,
                    split=split, goal_id=goal, seed=seed + 2 * (offset + index) + goal,
                    object_position_m=[x, y, .082], goal_position_m=list(position)))
        offset += count
    return splits


def _attempt(case, controller, *, prefix='', negative=None):
    return dict(id=f'{prefix}{case["id"]}-{controller}' + (f'-{negative}' if negative else ''),
                case=deepcopy(case), controller=controller, negative_control=negative)


def prepare(root, *, fixed_model, seed=SEED):
    """Register every scene/controller before capture and pin the comparator."""
    from .piper_motor_primitive import validate_model
    root = _root(root)
    if root.exists():
        raise ValueError('Use a fresh study directory; registered runs are immutable.')
    model = validate_model(read(fixed_model))
    sources = runtime_hashes()
    cases = registered_cases(seed)
    dev_evaluation = cases['development']
    rosters = {
        'diagnostic': [_attempt(c, 'teacher', prefix='diagnostic-') for c in dev_evaluation[:4]],
        'demonstrations': [_attempt(c, 'teacher') for c in cases['train'] + cases['development']],
        'corrections': [_attempt(cases['train'][2 * i + i % 2], 'mixed') for i in range(32)],
    }
    for stage in ('initial', 'corrected'):
        rosters[f'development-{stage}'] = [_attempt(c, controller, prefix=stage + '-')
            for c in dev_evaluation for controller in ('student', 'blind')]
    rosters['confirmation'] = [_attempt(c, controller) for c in cases['confirmation']
                              for controller in ('student', 'blind', 'fixed', 'teacher')]
    rosters['negative'] = [_attempt(c, 'student', prefix='negative-', negative=negative)
        for c in cases['confirmation'][:4]
        for negative in ('initial_missing', 'initial_stale', 'altered_calibration', 'open_gripper')]
    protocol = dict(schema=SCHEMA, seed=seed, cases=cases, rosters=rosters,
        goals=[list(g) for g in GOALS], source_sha256=sources, fixed_model_sha256=digest(model),
        historical_start_exclusion=[list(p) for p in sorted(_historical_starts())],
        start_domain=dict(x_m=[.287, .311], y_m=[-.042, -.018],
            reason='Three millimetre inset from the original fixed primitive envelope buffers observed camera estimation error; estimates are never clipped or replaced by native truth.'),
        grouping='Both goals, all cameras, controls and corrections stay in their exact starting-position group.',
        capture=dict(duration_s=20., width=320, height=240, camera_hz=25, control_hz=50,
                     physics_timestep_s=.001, camera_pairs_per_complete_episode=501),
        fitting=dict(hidden_units=[64, 64], epochs=50, batch_size=512, learning_rate=.003, seed=17,
                     normalization='training-only', corrections_rounds=1, teacher_mix_probability=.5),
        data_quality=dict(minimum_successful_episodes_per_goal=28, all_training_groups_covered=True),
        diagnostic=dict(all_four_strict_successful=True, maximum_initial_camera_error_m=.010,
            error_source='Privileged native diagnostic only; never a learner feature.'),
        selection='full learner development strict success descending, critical contacts ascending, common physical-scale prediction loss ascending, then initial before corrected; matched blind from selected stage',
        acceptance=dict(student_successes_minimum=58, student_attempts=64, confidence=.975,
            bootstrap_resamples=10000, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_unit='starting-position group; both goals paired',
            comparisons=['student-minus-fixed', 'student-minus-blind'], no_critical_contact_increase=True,
            minimum_common_admitted_groups=24,
            common_admission='Both goals must have an available bound initial camera and at least one completed control interval for both compared controllers. Require a positive paired interval on this subset as well; initial comparator refusal alone cannot establish improvement.',
            all_negative_controls_correct=True, no_privileged_features_or_teacher_intervention=True),
        storage=dict(maximum_study_bytes=STUDY_MAX_BYTES, disk_reserve_bytes=DISK_RESERVE_BYTES,
                     next_attempt_headroom_bytes=ATTEMPT_HEADROOM_BYTES),
        physical_robot_ready=False, limitations=LIMITATIONS)
    root.mkdir(parents=True)
    write(root / 'fixed-model.json', model)
    write(root / 'protocol.json', protocol)
    write(root / 'registration.json', dict(schema=SCHEMA + '/registration',
        protocol_sha256=file_hash(root / 'protocol.json'), fixed_model_file_sha256=file_hash(root / 'fixed-model.json'),
        source_sha256=sources, registered_before_any_capture=True))
    return inspect(root)


def _protocol(root, *, runtime=False):
    root = _root(root)
    registration = read(root / 'registration.json')
    if registration.get('schema') != SCHEMA + '/registration':
        raise ValueError('Unknown registration schema.')
    if file_hash(root / 'protocol.json') != registration['protocol_sha256']:
        raise ValueError('Registered protocol bytes changed.')
    if file_hash(root / 'fixed-model.json') != registration['fixed_model_file_sha256']:
        raise ValueError('Pinned fixed comparator changed.')
    protocol = read(root / 'protocol.json')
    if protocol.get('schema') != SCHEMA or protocol.get('source_sha256') != registration['source_sha256']:
        raise ValueError('Invalid or inconsistent protocol.')
    if runtime and runtime_hashes() != registration['source_sha256']:
        raise ValueError('Runtime changed after registration; preserve this study and register a new one.')
    return protocol


def _tree_bytes(root):
    size = 0
    for directory, subdirs, files in os.walk(root):
        if any((Path(directory) / name).is_symlink() for name in subdirs + files):
            raise ValueError('Study artifacts cannot contain symbolic links.')
        size += sum((Path(directory) / name).stat().st_size for name in files)
    return size


def storage_preflight(root, *, known_study_bytes=None):
    root = _root(root)
    size = _tree_bytes(root) if known_study_bytes is None else known_study_bytes
    free = shutil.disk_usage(root).free
    if size + ATTEMPT_HEADROOM_BYTES > STUDY_MAX_BYTES:
        raise ValueError('Study storage budget would exceed 120 GiB; existing evidence retained.')
    if free - ATTEMPT_HEADROOM_BYTES < DISK_RESERVE_BYTES:
        raise ValueError('Next attempt would consume the protected 30 GiB disk reserve.')
    return dict(study_bytes=size, free_bytes=free, maximum_study_bytes=STUDY_MAX_BYTES,
                disk_reserve_bytes=DISK_RESERVE_BYTES)


def _single_worker(function):
    @wraps(function)
    def wrapped(root, *args, **kwargs):
        _protocol(root, runtime=True)
        lock = _root(root) / 'worker.lock'
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ValueError('This study already has an active or interrupted worker; no concurrent execution or automatic lock removal.') from exc
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(canonical(dict(pid=os.getpid(), operation=function.__name__)))
                stream.flush()
                os.fsync(stream.fileno())
            return function(root, *args, **kwargs)
        finally:
            lock.unlink(missing_ok=True)
    return wrapped


def _stage(root, name):
    path = Path(root) / 'stages' / (name + '.json')
    if not path.exists():
        return None
    seal = read(path.with_name(name + '.integrity.json'))
    if seal != dict(sha256=file_hash(path), protocol_sha256=file_hash(Path(root) / 'protocol.json')):
        raise ValueError('Stage receipt identity changed.')
    return read(path)


def _seal_stage(root, name, receipt):
    path = Path(root) / 'stages' / (name + '.json')
    write(path, receipt)
    write(path.with_name(name + '.integrity.json'), dict(sha256=file_hash(path),
        protocol_sha256=file_hash(Path(root) / 'protocol.json')))


def _require(root, stage, status='complete'):
    result = _stage(root, stage)
    if result is None or result.get('status') != status:
        raise ValueError(f'Stage {stage} must be {status} before continuing.')
    return result


def _begin(root, name, protocol, *, model_bindings=None):
    target = Path(root) / 'stages' / (name + '.json')
    if target.exists():
        raise ValueError('Stage already sealed; existing attempts cannot be retried.')
    started = target.with_name(name + '.started.json')
    if not started.exists():
        write(started, dict(schema=SCHEMA + '/stage-start', stage=name,
            protocol_sha256=file_hash(Path(root) / 'protocol.json'), source_sha256=protocol['source_sha256'],
            model_bindings=model_bindings))
    elif read(started)['protocol_sha256'] != file_hash(Path(root) / 'protocol.json'):
        raise ValueError('Stage start is bound to a different protocol.')
    elif read(started).get('model_bindings') != model_bindings:
        raise ValueError('Model identity differs from the started stage.')


def _critical(summary):
    # The capture receipt reports the native per-millisecond count explicitly.
    value = summary.get('critical_contact_ticks')
    if value is None:
        value = summary.get('native_status', {}).get('native_info', {}).get('metrics', {}).get('critical_contact_ticks')
    if value is None:
        value = summary.get('score', {}).get('critical_contact_ticks')
    return value if type(value) is int and value >= 0 else None


def _row(root, attempt, summary, directory, *, model_sha256=None):
    expected_mode = 'student' if attempt['controller'] == 'blind' else attempt['controller']
    expected_case = {**attempt['case'], 'policy_artifact_sha256': model_sha256} if model_sha256 else attempt['case']
    if (summary.get('case') != expected_case or summary.get('mode') != expected_mode
            or summary.get('negative_control') != attempt['negative_control']):
        raise ValueError('Rollout identity differs from its registered case and control.')
    score = summary.get('score') or {}
    complete = summary.get('status') == 'complete' and summary.get('recorded_complete_intervals') == 1000
    return dict(id=attempt['id'], case=attempt['case'], controller=attempt['controller'],
        negative_control=attempt['negative_control'], path=str(directory.relative_to(root)),
        policy_artifact_sha256=model_sha256,
        status=summary.get('status', 'unknown'), complete=complete, strict_success=complete and score.get('success') is True,
        score=score, critical_contact_ticks=_critical(summary),
        initial_anchor_available=summary.get('initial_anchor_available') is True,
        initial_anchor_binding_valid=summary.get('initial_anchor_binding_valid') is True,
        initial_anchor_error_m=summary.get('initial_anchor_error_m'),
        recorded_complete_intervals=summary.get('recorded_complete_intervals', 0),
        provenance_ok=summary.get('no_privileged_features_or_teacher_intervention',
            summary.get('provenance', {}).get('learner_input_whitelist_verified', False)) is True,
        perception=summary.get('perception'), timing=_timing_summary(summary.get('timing') or {}),
        receipt_sha256=file_hash(directory / 'rollout.json') if (directory / 'rollout.json').exists() else None,
        integrity_sha256=file_hash(directory / 'integrity.json') if (directory / 'integrity.json').exists() else None)


def _timing_summary(timing):
    result = {name: timing.get(name) for name in ('wall_seconds', 'scope')}
    boundaries = timing.get('boundaries') or []
    for metric in ('estimator_wall_seconds', 'policy_inference_wall_seconds', 'full_boundary_wall_seconds',
                   'render_encode_wall_seconds'):
        values = [row[metric] for row in boundaries if type(row.get(metric)) in (int, float)
                  and math.isfinite(row[metric]) and row[metric] >= 0.]
        result[metric] = (dict(samples=len(values), p50=float(np.quantile(values, .5)),
            p95=float(np.quantile(values, .95)), maximum=max(values)) if values else None)
    return result


def _peak_rss():
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if platform.system() == 'Darwin' else 1024)
    except ImportError:
        return None


class _FixedPolicy:
    """Adapt the pinned open-loop comparator using only allowed camera context."""
    def __init__(self, model):
        from .piper_motor_primitive import MotorPrimitivePolicy
        self.policy = MotorPrimitivePolicy(model)
        self.anchor = None

    def __call__(self, context):
        snapshot = context['snapshot']
        signals = snapshot['signals']
        if self.anchor is None:
            estimate = context['estimate']
            candidate = estimate.get('object_position_estimate_m')
            if candidate is None:
                candidate = estimate.get('signals', {}).get('object_position_estimate_m')
            if isinstance(candidate, dict):
                candidate = candidate.get('value') if candidate.get('status') == 'available' else None
            if candidate is None:
                raise ValueError('Fixed comparator requires its initial camera position.')
            self.anchor = list(candidate)
        return self.policy(dict(t_seconds=snapshot['timestamp_seconds'],
            joint_position_rad=signals['joint_position_rad']['value'],
            object_position_m=self.anchor, target_position_m=context['goal_position_m']))


def _model_path(root, stage, controller):
    return Path(root) / 'models' / stage / ('blind' if controller == 'blind' else 'full') / 'model.json'


def _model_document(root, stage, controller):
    receipt = _require(root, 'train-' + stage)
    variant = 'blind' if controller == 'blind' else 'full'
    path = _model_path(root, stage, controller)
    if receipt['models'][variant]['model_sha256'] != file_hash(path):
        raise ValueError('Model bytes differ from the sealed training receipt.')
    return read(path)


def _model_bindings(root, name, model_stage):
    protocol = _protocol(root)
    result = {}
    for controller in {a['controller'] for a in protocol['rosters'][name]}:
        if controller == 'teacher':
            result[controller] = digest(protocol['source_sha256'])
        elif controller == 'fixed':
            result[controller] = file_hash(Path(root) / 'fixed-model.json')
        else:
            _model_document(root, model_stage, controller)
            result[controller] = file_hash(_model_path(root, model_stage, controller))
    return result


def _policy(root, controller, model_stage):
    if controller == 'teacher':
        return None
    if controller == 'fixed':
        return _FixedPolicy(read(Path(root) / 'fixed-model.json'))
    from .camera_action_model import ModelPolicy
    return ModelPolicy(_model_document(root, model_stage, controller))


def _run_stage(root, assets, name, *, model_stage=None):
    from .camera_action_capture import run_rollout, inspect_rollout
    root = _root(root)
    protocol = _protocol(root, runtime=True)
    model_bindings = _model_bindings(root, name, model_stage)
    _begin(root, name, protocol, model_bindings=model_bindings)
    rows = []
    blocked = None
    study_bytes = _tree_bytes(root)
    for attempt in protocol['rosters'][name]:
        directory = root / 'episodes' / name / attempt['id']
        if blocked is not None:
            rows.append(dict(**attempt, status='unattempted', complete=False, strict_success=False,
                             critical_contact_ticks=None, reason=blocked))
            continue
        if directory.exists():
            if not (directory / 'rollout.json').exists() or not (directory / 'integrity.json').exists():
                rows.append(dict(**attempt, status='unknown', complete=False, strict_success=False,
                    critical_contact_ticks=None, path=str(directory.relative_to(root)), reason='Unsealed prior attempt retained; never retried.'))
                continue
            inspect_rollout(directory)
            summary = read(directory / 'rollout.json')
        else:
            try:
                storage_preflight(root, known_study_bytes=study_bytes)
            except ValueError as exc:
                blocked = str(exc)
                rows.append(dict(**attempt, status='unattempted', complete=False, strict_success=False,
                    critical_contact_ticks=None, reason=blocked))
                continue
            controller = attempt['controller']
            policy = _policy(root, controller, model_stage) if controller != 'teacher' else None
            print(json.dumps(dict(stage=name, attempt=attempt['id'], status='starting')), flush=True)
            bound_case = {**attempt['case'], 'policy_artifact_sha256': model_bindings[controller]}
            summary = run_rollout(assets, directory, case=bound_case,
                mode='student' if controller == 'blind' else controller, policy=policy,
                negative_control=attempt['negative_control'], duration_s=20.)
            inspect_rollout(directory)
            study_bytes += _tree_bytes(directory)
        rows.append(_row(root, attempt, summary, directory, model_sha256=model_bindings[attempt['controller']]))
        print(json.dumps(dict(stage=name, attempt=attempt['id'], status=rows[-1]['status'],
                              strict_success=rows[-1]['strict_success'])), flush=True)
    receipt = dict(schema=SCHEMA + '/stage', stage=name,
        protocol_sha256=file_hash(root / 'protocol.json'), rows=rows, model_bindings=model_bindings,
        scheduled=len(protocol['rosters'][name]), strict_successes=sum(r['strict_success'] for r in rows),
        status='blocked' if blocked or any(r['status'] in ('unknown', 'unattempted') for r in rows) else 'complete',
        reason=blocked, physical_robot_ready=False)
    return receipt


def data_quality(rows, protocol):
    training = [r for r in rows if r['case']['split'] == 'train']
    successes = [r for r in training if r.get('strict_success') and r.get('initial_anchor_available')
                 and r.get('initial_anchor_binding_valid')]
    required = {c['group_id'] for c in protocol['cases']['train']}
    covered = {r['case']['group_id'] for r in successes}
    goals = {str(g): sum(r['case']['goal_id'] == g for r in successes) for g in range(2)}
    return dict(passed=covered == required and all(n >= 28 for n in goals.values()),
        covered_training_groups=len(covered), required_training_groups=32,
        missing_training_groups=sorted(required - covered), successful_episodes_per_goal=goals,
        minimum_successful_episodes_per_goal=28,
        successful_with_available_bound_initial_camera=len(successes))


@_single_worker
def collect(root, assets, *, stage='diagnostic'):
    root = _root(root)
    if stage not in ('diagnostic', 'demonstrations', 'corrections'):
        raise ValueError('Unknown collection stage.')
    if stage != 'diagnostic':
        _require(root, 'diagnostic')
    if stage == 'corrections':
        _require(root, 'train-initial')
        _require(root, 'development-initial')
    receipt = _run_stage(root, assets, stage, model_stage='initial' if stage == 'corrections' else None)
    if stage == 'diagnostic':
        passed = all(r.get('strict_success') and r.get('initial_anchor_available')
                     and r.get('initial_anchor_binding_valid')
                     and type(r.get('initial_anchor_error_m')) in (int, float)
                     and math.isfinite(r['initial_anchor_error_m'])
                     and 0. <= r['initial_anchor_error_m'] <= .010 for r in receipt['rows'])
        receipt['diagnostic_passed'] = passed
        if not passed:
            receipt.update(status='blocked', reason='Diagnostic did not demonstrate complete teacher success and a bound initial visual anchor within 10 mm of native diagnostic truth in all four cases.')
    elif stage == 'demonstrations':
        quality = data_quality(receipt['rows'], _protocol(root))
        receipt['data_quality'] = quality
        if not quality['passed']:
            receipt.update(status='blocked', reason='Teacher demonstration quality gate failed; learning and confirmation are blocked.')
    _seal_stage(root, stage, receipt)
    if receipt['status'] == 'blocked' and not (root / 'report.json').exists():
        write(root / 'report.json', report_study(root))
    return receipt


def _dataset(root, rows, *, blind, label_source):
    from .camera_action_export import export_episode, combine_datasets
    selected = [r for r in rows if r.get('complete') and r.get('initial_anchor_available')
                and r.get('initial_anchor_binding_valid')
                and (label_source == 'teacher_query' or r.get('strict_success'))]
    if not selected:
        raise ValueError('No complete, camera-bound episodes are eligible for learning.')
    return combine_datasets([export_episode(Path(root) / r['path'], blind=blind,
        label_source=label_source) for r in selected]), [r['id'] for r in selected]


@_single_worker
def train(root, *, stage='initial'):
    from .camera_action_export import combine_datasets, blind_dataset
    from .camera_action_model import train as fit
    root = _root(root)
    if stage not in ('initial', 'corrected'):
        raise ValueError('Unknown model fitting stage.')
    protocol = _protocol(root, runtime=True)
    demos = _require(root, 'demonstrations')
    corrections = _require(root, 'corrections') if stage == 'corrected' else None
    name = 'train-' + stage
    _begin(root, name, protocol)
    training_rows = [r for r in demos['rows'] if r['case']['split'] == 'train']
    dev_rows = [r for r in demos['rows'] if r['case']['split'] == 'development']
    export_began = time.perf_counter()
    full_dataset, selected = _dataset(root, training_rows, blind=False, label_source='issued')
    if corrections:
        correction_data, correction_ids = _dataset(root, corrections['rows'], blind=False, label_source='teacher_query')
        full_dataset = combine_datasets([full_dataset, correction_data])
        selected += correction_ids
    full_development, development_ids = _dataset(root, dev_rows, blind=False, label_source='issued')
    export_seconds = time.perf_counter() - export_began
    models = {}
    for variant in ('full', 'blind'):
        target = root / 'models' / stage / variant
        if target.exists():
            raise ValueError('Model fitting attempt already exists; never silently refit it.')
        target.mkdir(parents=True)
        write(target / 'started.json', dict(stage=stage, variant=variant,
            protocol_sha256=file_hash(root / 'protocol.json')))
        began = time.perf_counter()
        dataset = blind_dataset(full_dataset) if variant == 'blind' else full_dataset
        development = blind_dataset(full_development) if variant == 'blind' else full_development
        model = fit(dataset, blind=variant == 'blind', stage=stage, seed=17, epochs=50,
                    batch_size=512, learning_rate=.003, validation_dataset=development)
        write(target / 'model.json', model)
        models[variant] = dict(path=str((target / 'model.json').relative_to(root)),
            model_sha256=file_hash(target / 'model.json'), model_bytes=(target / 'model.json').stat().st_size,
            training_wall_seconds=time.perf_counter() - began, training_episodes=selected,
            shared_export_wall_seconds=export_seconds, process_peak_rss_bytes=_peak_rss(),
            memory_scope='Process high-water RSS including shared dataset export and any preceding fit.',
            development_episodes=development_ids, training=model.get('training'),
            dataset_provenance=dataset.get('provenance'), physical_robot_ready=False)
        write(target / 'receipt.json', models[variant])
    receipt = dict(schema=SCHEMA + '/training', stage=name, status='complete', models=models,
                   protocol_sha256=file_hash(root / 'protocol.json'), physical_robot_ready=False)
    _seal_stage(root, name, receipt)
    return receipt


def _loss(model):
    training = model.get('training') or {}
    validation = training.get('validation') or {}
    value = validation.get('action_scale_mse')
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    raise ValueError('Model must report a finite development prediction loss for registered selection.')


def freeze(root):
    root = _root(root)
    protocol = _protocol(root, runtime=True)
    existing = root / 'freeze.json'
    if existing.exists():
        _verify_freeze(root)
        return read(existing)
    candidates = []
    for index, stage in enumerate(('initial', 'corrected')):
        receipt = _require(root, 'development-' + stage)
        if receipt.get('model_bindings') != _model_bindings(root, 'development-' + stage, stage):
            raise ValueError('Development outcomes do not match the trained model identities.')
        rows = [r for r in receipt['rows'] if r['controller'] == 'student']
        if len(rows) != 16 or any(r.get('critical_contact_ticks') is None for r in rows):
            raise ValueError('Development comparison requires all 16 learner outcomes and contact counts.')
        candidates.append(dict(stage=stage, strict_successes=sum(r['strict_success'] for r in rows),
            critical_contact_ticks=sum(r['critical_contact_ticks'] for r in rows),
            prediction_loss=_loss(_model_document(root, stage, 'student')), order=index))
    chosen = min(candidates, key=lambda c: (-c['strict_successes'], c['critical_contact_ticks'], c['prediction_loss'], c['order']))
    identities = {}
    for stage in ('initial', 'corrected'):
        for controller in ('student', 'blind'):
            _model_document(root, stage, controller)
            path = _model_path(root, stage, controller)
            identities[str(path.relative_to(root))] = file_hash(path)
    for stage in STAGES[:7]:
        path = root / 'stages' / (stage + '.json')
        identities[str(path.relative_to(root))] = file_hash(path)
    result = dict(schema=SCHEMA + '/freeze', selected_stage=chosen['stage'], candidates=candidates,
        protocol_sha256=file_hash(root / 'protocol.json'), source_sha256=protocol['source_sha256'],
        artifacts=identities, fixed_model_sha256=file_hash(root / 'fixed-model.json'),
        frozen_before_confirmation=True, physical_robot_ready=False)
    write(existing, result)
    return result


def _verify_freeze(root, *, runtime=True):
    frozen = read(Path(root) / 'freeze.json')
    protocol = _protocol(root, runtime=runtime)
    if frozen.get('schema') != SCHEMA + '/freeze' or frozen['source_sha256'] != protocol['source_sha256']:
        raise ValueError('Frozen runtime identity changed.')
    if frozen['protocol_sha256'] != file_hash(Path(root) / 'protocol.json'):
        raise ValueError('Frozen protocol identity changed.')
    if frozen['fixed_model_sha256'] != file_hash(Path(root) / 'fixed-model.json'):
        raise ValueError('Frozen comparator identity changed.')
    for name, expected in frozen['artifacts'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or file_hash(Path(root) / path) != expected:
            raise ValueError('Frozen artifact changed.')
    return frozen


@_single_worker
def evaluate(root, assets, *, stage='development-initial'):
    root = _root(root)
    if stage in ('development-initial', 'development-corrected'):
        model_stage = stage.split('-', 1)[1]
        _require(root, 'train-' + model_stage)
    elif stage in ('confirmation', 'negative'):
        frozen = freeze(root)
        model_stage = frozen['selected_stage']
        if stage == 'negative':
            _require(root, 'confirmation')
    else:
        raise ValueError('Unknown evaluation stage.')
    receipt = _run_stage(root, assets, stage, model_stage=model_stage)
    if stage in ('confirmation', 'negative'):
        receipt['freeze_sha256'] = file_hash(root / 'freeze.json')
    _seal_stage(root, stage, receipt)
    if stage == 'negative':
        report = report_study(root)
        write(root / 'report.json', report)
    return receipt


def _admitted(row):
    return (row.get('initial_anchor_available') is True and row.get('initial_anchor_binding_valid') is True
            and type(row.get('recorded_complete_intervals')) is int and row['recorded_complete_intervals'] > 0)


def paired_bootstrap(rows, comparator, *, admitted_only=False):
    grouped = {}
    for row in rows:
        if row['controller'] not in ('student', comparator):
            continue
        group = row['case']['group_id']
        key = (row['controller'], row['case']['goal_id'])
        if key in grouped.setdefault(group, {}):
            raise ValueError('Duplicate paired outcome.')
        grouped[group][key] = row
    keys = {(c, g) for c in ('student', comparator) for g in (0, 1)}
    if len(grouped) != 32 or any(set(g) != keys for g in grouped.values()):
        raise ValueError('Paired interval requires exactly 32 complete starting-position groups.')
    retained = {key: value for key, value in grouped.items()
                if not admitted_only or all(_admitted(row) for row in value.values())}
    differences = np.asarray([sum(float(g[('student', goal)].get('strict_success') is True)
                                 - float(g[(comparator, goal)].get('strict_success') is True)
                                 for goal in (0, 1)) / 2. for _, g in sorted(retained.items())])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(differences)
    sampled = differences[rng.integers(0, count, size=(10000, count))].mean(axis=1) if count else None
    bounds = np.quantile(sampled, [.0125, .9875]).tolist() if count else [None, None]
    return dict(comparison='student-minus-' + comparator, difference=float(differences.mean()) if count else None,
        confidence=.975, lower=bounds[0], upper=bounds[1], resamples=10000, seed=BOOTSTRAP_SEED,
        groups=count, excluded_groups=32-count, admitted_only=admitted_only,
        pairing='both goals within exact starting-position group', positive=bounds[0] is not None and bounds[0] > 0.)


def _negative_correct(row):
    if row.get('negative_control') == 'open_gripper':
        score = row.get('score') or {}
        return (row.get('complete') is True and row.get('strict_success') is False
                and score.get('valid') is True
                and (score.get('strict_v01') or {}).get('trajectory_complete') is True)
    return (row.get('status') == 'refused' and row.get('recorded_complete_intervals') == 0
            and row.get('strict_success') is False)


def report_study(root):
    root = _root(root)
    protocol = _protocol(root)
    confirmation = _stage(root, 'confirmation')
    negatives = _stage(root, 'negative')
    report = dict(schema=SCHEMA + '/report', physical_robot_ready=False, qualified=False,
        qualification='research_candidate', protocol_sha256=file_hash(root / 'protocol.json'),
        limitations=LIMITATIONS, status='incomplete', controllers={}, comparisons=[], admitted_comparisons=[], gates={})
    if confirmation is None or negatives is None:
        blocked = [name for name in STAGES if (_stage(root, name) or {}).get('status') == 'blocked']
        report.update(status='blocked' if blocked else 'incomplete', blocked_stages=blocked)
        return report
    frozen = _verify_freeze(root, runtime=False)
    if any(stage.get('freeze_sha256') != file_hash(root / 'freeze.json') for stage in (confirmation, negatives)):
        raise ValueError('Evaluation stage is not bound to the frozen selection.')
    rows = confirmation['rows']
    for stage, receipt in (('confirmation', confirmation), ('negative', negatives)):
        expected = {a['id']: a for a in protocol['rosters'][stage]}
        for row in receipt['rows']:
            original = expected.get(row['id'])
            if original is None or any(row.get(key) != original[key] for key in ('case', 'controller', 'negative_control')):
                raise ValueError('Reported outcome identity differs from the registered roster.')
    complete_roster = (len(rows) == 256 and {r['id'] for r in rows} == {r['id'] for r in protocol['rosters']['confirmation']}
                       and len(negatives['rows']) == 16
                       and {r['id'] for r in negatives['rows']} == {r['id'] for r in protocol['rosters']['negative']}
                       and confirmation['status'] == 'complete' and negatives['status'] == 'complete')
    for controller in ('student', 'blind', 'fixed', 'teacher'):
        selected = [r for r in rows if r['controller'] == controller]
        contacts = [r.get('critical_contact_ticks') for r in selected]
        report['controllers'][controller] = dict(scheduled=len(selected),
            strict_successes=sum(r.get('strict_success') is True for r in selected),
            complete=sum(r.get('complete') is True for r in selected),
            admitted_cases=sum(_admitted(r) for r in selected),
            initial_refusals=sum(r.get('status') in ('refused', 'interrupted')
                and r.get('recorded_complete_intervals') == 0 for r in selected),
            critical_contact_episodes=sum(v > 0 for v in contacts if v is not None),
            critical_contact_ticks=sum(v for v in contacts if v is not None),
            contact_counts_complete=all(v is not None for v in contacts))
    if complete_roster:
        report['comparisons'] = [paired_bootstrap(rows, comparator) for comparator in ('fixed', 'blind')]
        report['admitted_comparisons'] = [paired_bootstrap(rows, comparator, admitted_only=True) for comparator in ('fixed', 'blind')]
    student = report['controllers']['student']
    fixed = report['controllers']['fixed']
    report['gates'] = dict(
        all_registered_attempts_accounted=complete_roster,
        at_least_58_of_64=student['scheduled'] == 64 and student['strict_successes'] >= 58,
        positive_paired_comparisons=len(report['comparisons']) == 2 and all(c['positive'] for c in report['comparisons']),
        positive_common_admitted_comparisons=len(report['admitted_comparisons']) == 2
            and all(c['groups'] >= 24 and c['positive'] for c in report['admitted_comparisons']),
        no_critical_contact_increase=student['contact_counts_complete'] and fixed['contact_counts_complete']
            and student['critical_contact_episodes'] <= fixed['critical_contact_episodes'],
        all_negative_controls_correct=len(negatives['rows']) == 16 and all(_negative_correct(r) for r in negatives['rows']),
        no_privileged_features_or_teacher_intervention=all(r.get('provenance_ok') is True for r in rows if r['controller'] in ('student', 'blind', 'fixed')))
    report['qualified'] = all(report['gates'].values())
    report['qualification'] = 'bounded_simulation_skill' if report['qualified'] else 'research_candidate'
    report['status'] = 'complete' if complete_roster else 'incomplete'
    report['selected_stage'] = frozen['selected_stage']
    return report


def inspect(root):
    """Read-only UI/CLI state; does not import native simulation or run models."""
    root = _root(root)
    protocol = _protocol(root)
    stages = {name: _stage(root, name) for name in STAGES}
    blocked = [name for name, value in stages.items() if value and value.get('status') == 'blocked']
    current = next((name for name, value in stages.items() if value is None), None)
    report = report_study(root)
    command = ('collect' if current in ('diagnostic', 'demonstrations', 'corrections') else
               'train' if current and current.startswith('train-') else 'evaluate')
    actions = [] if blocked or current is None else [dict(command=command,
        stage=current.removeprefix('train-') if command == 'train' else current,
        requires_assets=command != 'train')]
    return dict(schema=SCHEMA + '/inspection', status='blocked' if blocked else ('complete' if current is None else 'ready'),
        study=str(root), protocol_sha256=file_hash(root / 'protocol.json'),
        stage_summaries={name: None if value is None else {key: value.get(key) for key in
            ('status', 'scheduled', 'strict_successes', 'reason', 'diagnostic_passed', 'data_quality')} for name, value in stages.items()},
        next_stage=None if blocked else current, next_actions=actions, blocked_stages=blocked, report=report,
        qualification=report['qualification'], qualified=report['qualified'],
        registered_attempts={name: len(rows) for name, rows in protocol['rosters'].items()},
        scope='Simulation-only camera-to-action research; no physical commands or human-video motor recovery.',
        physical_robot_ready=False, physics_executed=False, robot_commands_sent=False)


def package(root, output):
    """Create an evidence package, including blockers; installer bundles are separate."""
    root, output = _root(root), _root(output)
    if output.exists() or output.is_relative_to(root):
        raise ValueError('Choose a fresh package directory outside the study.')
    state = inspect(root)
    output.mkdir(parents=True)
    files = [root / 'protocol.json', root / 'registration.json', root / 'fixed-model.json']
    files += sorted((root / 'stages').glob('*.json')) if (root / 'stages').exists() else []
    files += sorted((root / 'models').rglob('*.json')) if (root / 'models').exists() else []
    if (root / 'freeze.json').exists():
        files.append(root / 'freeze.json')
    inventory = {}
    for source in files:
        relative = source.relative_to(root)
        raw = source.read_bytes()
        _create(output / 'evidence' / relative, raw)
        inventory[str(Path('evidence') / relative)] = dict(bytes=len(raw), sha256=sha256(raw).hexdigest())
    write(output / 'report.json', state['report'])
    inventory['report.json'] = dict(bytes=(output / 'report.json').stat().st_size, sha256=file_hash(output / 'report.json'))
    write(output / 'manifest.json', dict(schema=SCHEMA + '/evidence-package', files=inventory,
        qualification=state['report']['qualification'], qualified=state['report']['qualified'],
        raw_episode_data_included=False, model_runtime_installation_included=False,
        source_study_protocol_sha256=file_hash(root / 'protocol.json'), physical_robot_ready=False))
    with zipfile.ZipFile(output / 'camera-action-evidence.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in [*inventory, 'manifest.json']:
            archive.write(output / name, name)
    return dict(status='complete', output=str(output), qualification=state['report']['qualification'],
                qualified=state['report']['qualified'], evidence_sha256=file_hash(output / 'camera-action-evidence.zip'),
                physical_robot_ready=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('prepare')
    create.add_argument('--study', required=True)
    create.add_argument('--fixed-model', required=True)
    create.add_argument('--seed', type=int, default=SEED)
    for name, choices, default in (
            ('collect', ('diagnostic', 'demonstrations', 'corrections'), 'diagnostic'),
            ('train', ('initial', 'corrected'), 'initial'),
            ('evaluate', ('development-initial', 'development-corrected', 'confirmation', 'negative'), 'development-initial')):
        cmd = sub.add_parser(name)
        cmd.add_argument('--study', required=True)
        cmd.add_argument('--stage', choices=choices, default=default)
        if name != 'train':
            cmd.add_argument('--assets', required=True)
    check = sub.add_parser('inspect')
    check.add_argument('--study', required=True)
    packed = sub.add_parser('package')
    packed.add_argument('--study', required=True)
    packed.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.study, fixed_model=args.fixed_model, seed=args.seed)
        elif args.command in ('collect', 'evaluate'):
            result = globals()[args.command](args.study, args.assets, stage=args.stage)
        elif args.command == 'train':
            result = train(args.study, stage=args.stage)
        elif args.command == 'package':
            result = package(args.study, args.output)
        else:
            result = inspect(args.study)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get('status') not in ('blocked', 'refused', 'incomplete') else 1
    except (ValueError, OSError, RuntimeError, TypeError) as exc:
        print(json.dumps(dict(status='refused', reason=str(exc), physical_robot_ready=False), allow_nan=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
