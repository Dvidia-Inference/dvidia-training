"""Bound, immutable native attempts for the learned PiPER hold/advance clock.

This runner neither prepares a study nor trains or selects a candidate. Every
attempt retains the unchanged strict task scorer, complete native telemetry,
independent intervention audit and a request-bound receipt. Only literal
training teacher cases may emit progress labels.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np

from . import piper_learned_grip_benchmark as prior
from .piper_imitation_benchmark import read, write, sha, save_episode, load_episode, peak_rss
from .piper_imitation import digest


SCHEMA = 'dvidia.piper-grip-clock-study/v0.3'
EPISODE_SCHEMA = 'dvidia.piper-grip-clock-episode/v0.3'
BASE_SHA = prior.BASE_SHA
JAW_SHA = 'cc2ffa5ab78a7501a175d385aa4ace65df284b5e9eae5ce9536f7b124065c9fe'
VARIANTS = ('fixed', 'teacher', 'clock', 'blind_clock')
SOURCE_FILES = tuple(dict.fromkeys((
    'piper_grip_clock_benchmark.py', 'piper_grip_clock.py',
    'piper_grip_clock_model.py', 'piper_grip_clock_env.py',
    'piper_grip_clock_cli.py', *prior.SOURCE_FILES)))
_canonical = prior._canonical
_save_npz = prior._save_npz
_load_npz = prior._load_npz
config = prior.config


def runtime_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def jaw_contract(base, jaw):
    from .piper_grip_clock import validate_jaw_document
    return validate_jaw_document(base, jaw)


def _case(value):
    from . import piper_imitation as core
    if type(value) is not dict or set(value) != {'id', 'start_xy', 'goal_xy', 'seed', 'condition'}:
        raise ValueError('A case needs its literal ID, start, goal, seed and condition only.')
    core._case_ids([value['id']])
    if value['condition'] not in ('nominal', 'post_grip'):
        raise ValueError('Unknown clock study intervention condition.')
    core._integer(value['seed'], 'case seed', 0, 999999)
    for name in ('start_xy', 'goal_xy'):
        if type(value[name]) is not list:
            raise ValueError('Case coordinates must be literal two-number lists.')
        core._array(value[name], (2,), name, 1.)
    config(value)  # The unchanged bounded task workspace remains authoritative.
    return _canonical(value)


def _options(teacher_config=None, intervention_parameters=None):
    from .piper_grip_clock import TeacherConfig
    from .piper_grip_clock_env import PostGripParameters
    if teacher_config is None:
        teacher = TeacherConfig()
    elif type(teacher_config) is TeacherConfig:
        teacher = teacher_config
    elif type(teacher_config) is dict:
        teacher = TeacherConfig(**teacher_config)
    else:
        raise ValueError('Teacher configuration must be its frozen value or a parameter mapping.')
    if intervention_parameters is None:
        intervention_parameters = {}
    if type(intervention_parameters) is not dict or 'interruption' in intervention_parameters:
        raise ValueError('Intervention parameters cannot change the assigned condition.')
    intervention = asdict(PostGripParameters(interruption=False, **intervention_parameters))
    del intervention['interruption']
    return teacher, intervention


def _documents(base, jaw, clock_document, variant):
    from . import piper_motor_primitive as primitive
    from . import piper_grip_clock_model as clock_model
    if variant not in VARIANTS:
        raise ValueError('Unknown fixed clock-study variant.')
    needs_clock = variant in ('clock', 'blind_clock')
    if needs_clock != (clock_document is not None):
        raise ValueError('This variant has the wrong clock-model requirement.')
    base = primitive.validate_model(base)
    jaw_mode, jaw = jaw_contract(base, jaw)
    model = clock_model.validate_model(clock_document) if needs_clock else None
    if model is not None and (model['blind'] is not (variant == 'blind_clock')
            or model['base_sha256'] != digest(base) or model['jaw_sha256'] != digest(jaw)):
        raise ValueError('Clock/base/jaw or full/blind feature identity differs.')
    if model is not None:
        expected_schema = 'dvidia.piper-grip-progress-clock/v2' if jaw_mode == 'coordinated_reference' else 'dvidia.piper-grip-progress-clock/v1'
        if model['schema'] != expected_schema:
            raise ValueError('Clock-model schema differs from the explicit jaw architecture.')
    return base, jaw, model


def _sample_arrays(samples, case_id):
    """Validate finite, data-only prefixes without any outcome-based selection."""
    from . import piper_grip_clock_model as model
    if type(samples) is not dict or set(samples) != {'features', 'blind_features', 'targets', 'groups', 'weights'}:
        raise ValueError('Clock labels have an unknown or missing sample field.')
    groups = list(samples['groups'])
    if len(groups) > 1000 or any(group != case_id for group in groups):
        raise ValueError('Training labels must retain their complete original episode group.')
    count = len(groups)
    arrays = {}
    for name, shape in (('features', (count, model.FEATURE_COUNT)),
                        ('blind_features', (count, model.FEATURE_COUNT)),
                        ('targets', (count,)), ('weights', (count,))):
        value = np.asarray(samples[name], dtype=np.float64)
        if count == 0 and name in ('features', 'blind_features') and value.shape == (0,):
            value = value.reshape((0, model.FEATURE_COUNT))
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Clock sample dimensions or numeric values differ: '+name)
        arrays[name] = value
    if (not np.isin(arrays['targets'], (0., .02)).all()
            or not np.array_equal(arrays['weights'], np.where(arrays['targets'] == 0., 20., 1.))):
        raise ValueError('Clock targets must be exact binary increments with 20/1 hold/advance weights.')
    blind = arrays['features'].copy()
    blind[:, model.LIVE_START:model.LIVE_STOP] = 0.
    if not np.array_equal(arrays['blind_features'], blind):
        raise ValueError('Blind training features may change only the live/history slots.')
    model._features(arrays['features'], (count, model.FEATURE_COUNT), blind=False)
    model._features(arrays['blind_features'], (count, model.FEATURE_COUNT), blind=True)
    arrays['groups'] = np.asarray(groups, dtype='<U100')
    return arrays


def _samples_from_episode(episode, case_id):
    """Rebuild every pre-issue teacher label from the retained action record."""
    from . import piper_grip_clock_model as model
    samples = dict(features=[], blind_features=[], targets=[], groups=[], weights=[])
    progress = 0.
    for index, action in enumerate(episode['actions']):
        decision = action.get('clock_decision')
        if type(decision) is not dict or decision.get('decision_origin') != 'authored_causal_grip_teacher':
            raise ValueError('Training action lacks its causal authored progress decision.')
        target = decision.get('increment_s')
        if type(target) not in (int, float) or target not in (0., .02):
            raise ValueError('Teacher label is not an exact hold/advance progress increment.')
        feature = np.asarray(decision.get('features'), dtype=np.float64)
        if feature.shape != (model.FEATURE_COUNT,) or not np.isfinite(feature).all():
            raise ValueError('Teacher decision is missing its finite causal 58-feature row.')
        values = [action.get('t_seconds'), decision.get('physical_time_s'),
                  decision.get('progress_before_s'), decision.get('progress_after_s')]
        if any(type(value) not in (int, float) or not np.isfinite(value) for value in values):
            raise ValueError('Teacher decision lacks finite physical/pre-issue/post-issue clock values.')
        elapsed, physical, before, after = values
        if (abs(elapsed-index*.02) > 1e-8 or abs(physical-elapsed) > 1e-8
                or abs(before-progress) > 1e-8 or abs(after-before-target) > 1e-8
                or before < -1e-8 or before > elapsed+1e-8 or after > 20.+1e-8
                or not np.allclose(feature[[model.ELAPSED_INDEX, -2, -1]],
                    [elapsed/20., before/20., max(0., elapsed-before)/20.], atol=1e-9, rtol=0)):
            raise ValueError('Teacher progress label is not bound to its causal pre-issue clock features.')
        progress = after
        blind = feature.copy()
        blind[model.LIVE_START:model.LIVE_STOP] = 0.
        samples['features'].append(feature)
        samples['blind_features'].append(blind)
        samples['targets'].append(float(target))
        samples['groups'].append(case_id)
        samples['weights'].append(20. if target == 0. else 1.)
    return _sample_arrays(samples, case_id)


def rollout(assets, case, base, jaw, *, variant, clock_document=None,
            teacher_config=None, intervention_parameters=None, collect_labels=False):
    """One fresh native episode; learner variants construct no teacher object."""
    from .piper_grip_clock_env import PostGripPiperTaskEnv
    from . import piper_grip_clock as clock
    from .robot_profiles import verify_assets
    from . import reference_sim as sim
    case = _case(case)
    base, jaw, clock_document = _documents(base, jaw, clock_document, variant)
    teacher, intervention = _options(teacher_config, intervention_parameters)
    if type(collect_labels) is not bool or collect_labels and variant != 'teacher':
        raise ValueError('Only authored teacher attempts may collect progress labels.')
    c = config(case)
    implementation = runtime_hashes()
    env = PostGripPiperTaskEnv(assets, c, interruption=case['condition'] == 'post_grip', **intervention)
    jaw_mode, _ = jaw_contract(base, jaw)
    result = dict(schema=EPISODE_SCHEMA, config=_canonical(asdict(c)),
        case_id=case['id'], seed=case['seed'], condition=case['condition'],
        policy_kind=variant, base_model_sha256=digest(base), jaw_model_sha256=digest(jaw),
        model_sha256=digest(clock_document) if clock_document is not None else None, jaw_mode=jaw_mode,
        teacher_config=asdict(teacher), post_grip_parameters=intervention,
        physical_robot_ready=False, observation_origin='privileged_native_simulator_state',
        trajectory=[], actions=[], valid=False, success=False, reason='not_started',
        simulated_seconds=0., error=None, metrics={}, implementation_sha256=implementation)
    inference, steps = [], []
    began = time.perf_counter()
    try:
        obs, info = env.reset(case['seed'])
        mj, _ = sim._native_imports()
        result.update(initial_object_position_m=env.initial_object_position.tolist(),
            compiled_model_sha256=sim.model_digest(env.model), asset_provenance=verify_assets('piper', assets),
            native_servo_gains=env.model.actuator_gainprm[:, 0].tolist(),
            native_force_limits=env.model.actuator_forcerange.tolist(),
            platform=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version(),
                          numpy=np.__version__, mujoco=mj.__version__))
        result['trajectory'].append(obs)
        if variant == 'fixed':
            policy = clock.FixedClock(base, jaw)
        elif variant == 'teacher':
            policy = clock.TeacherClock(base, jaw, config=teacher)
        else:
            policy = clock.LearnedClock(base, jaw, clock_document, blind=variant == 'blind_clock')
        for _ in range(1000):
            started = time.perf_counter()
            inference_started = time.perf_counter()
            action = policy(obs)
            inference.append(time.perf_counter()-inference_started)
            # command must be the complete same record, including all clock
            # diagnostics, for the unchanged strict action/state consistency.
            record = dict(t_seconds=obs['t_seconds'], **action,
                          clock_decision=_canonical(policy.last_decision))
            result['actions'].append(record)
            obs.update(issued_joint_targets_rad=list(action['joint_targets_rad']),
                       issued_gripper_position_m=action['gripper_position_m'], command=record)
            obs, info = env.step(action)
            steps.append(time.perf_counter()-started)
            if obs is not None:
                # Newly measured states describe no issued action yet. The
                # terminal state retains these nulls; earlier states are filled
                # only when that exact state supplies the next causal command.
                obs.update(issued_joint_targets_rad=None, issued_gripper_position_m=None, command=None)
                result['trajectory'].append(obs)
            if env.done:
                break
        result.update(valid=info['valid'], success=info['success'], reason=info['reason'],
            simulated_seconds=info['simulation_time'], metrics={**info, 'contact_pair_native_ticks': env.contact_pair_native_ticks},
            policy_diagnostics=policy.diagnostics())
    except Exception as exc:
        result.update(valid=False, success=False, reason='exception',
                      error=dict(type=type(exc).__name__, message=str(exc)[:500]))
    result['timing'] = dict(wall_seconds=time.perf_counter()-began,
        inference=prior.prior._summary(inference), control_step=prior.prior._summary(steps),
        inference_samples_s=inference, control_step_samples_s=steps,
        missed_20ms_boundaries=sum(value > .02 for value in steps), process_peak_rss_bytes=peak_rss(),
        scope='CPU computation including dense native telemetry; excludes camera, communication and real actuation.')
    if runtime_hashes() != implementation:
        raise ValueError('Runtime source changed during the physical attempt; retained started marker forbids rerun.')
    samples = _samples_from_episode(result, case['id']) if collect_labels else {}
    return result, env.native_arrays(), env.native_metadata(), samples


def _request_binding(root, case, variant, model_path=None, collect_labels=False):
    root = Path(root)
    case = _case(case)
    if type(collect_labels) is not bool or collect_labels and variant != 'teacher':
        raise ValueError('Training progress labels require the authored teacher variant.')
    protocol = read(root/'protocol.json')
    cases = protocol.get('cases')
    if type(cases) is not dict or set(cases) != {'train', 'development', 'confirmation'}:
        raise ValueError('Protocol must declare the exact three clock-study splits.')
    matches, ids, starts = [], set(), {}
    for split, rows in cases.items():
        if type(rows) is not list:
            raise ValueError('Every split must be a literal case list.')
        for row in rows:
            checked = _case(row)
            if checked['id'] in ids:
                raise ValueError('Case IDs must be unique across the full protocol.')
            ids.add(checked['id'])
            point = tuple(checked['start_xy'])
            if point in starts and starts[point] != split:
                raise ValueError('Whole initial-XY groups cannot cross study splits.')
            starts[point] = split
            if checked == case:
                matches.append(split)
    if len(matches) != 1:
        raise ValueError('Attempt differs from the literal scheduled case.')
    split = matches[0]
    if collect_labels and split != 'train':
        raise ValueError('Progress labels are restricted to the teacher training split.')
    base_path, jaw_path = root/'base-model.json', root/'jaw-model.json'
    if sha(base_path) != BASE_SHA:
        raise ValueError('Frozen base bytes changed.')
    base, jaw, model = _documents(read(base_path), read(jaw_path),
                                  read(model_path) if model_path is not None else None, variant)
    jaw_mode, _ = jaw_contract(base, jaw)
    declared_jaw = protocol.get('jaw_model_raw_sha256')
    if jaw_mode == 'physical_adapter':
        expected_jaw = JAW_SHA
        if declared_jaw is not None and declared_jaw != expected_jaw:
            raise ValueError('Physical-adapter protocol must retain the frozen jaw4 bytes.')
    else:
        from .piper_motor_primitive import _hash
        _hash(declared_jaw, 'coordinated jaw manifest raw SHA256')
        expected_jaw = declared_jaw
    if sha(jaw_path) != expected_jaw or protocol.get('jaw_mode', jaw_mode) != jaw_mode:
        raise ValueError('Frozen jaw bytes or explicit architecture changed.')
    teacher, intervention = _options(protocol.get('teacher_config'), protocol.get('post_grip_parameters'))
    implementation = runtime_hashes()
    if split == 'confirmation':
        if not (root/'frozen.json').is_file():
            raise ValueError('Confirmation cannot begin before candidate/runtime freeze.')
        frozen = read(root/'frozen.json')
        if frozen.get('protocol_sha256') != sha(root/'protocol.json') or frozen.get('runtime_sha256') != implementation:
            raise ValueError('Frozen confirmation protocol or runtime changed.')
        for key, expected in (('base_model_raw_sha256', BASE_SHA), ('jaw_model_raw_sha256', expected_jaw)):
            if key in frozen and frozen[key] != expected:
                raise ValueError('Frozen confirmation base or jaw identity changed.')
        selected_key = 'blind_sha256' if variant == 'blind_clock' else 'selected_sha256'
        if model is not None and selected_key in frozen and frozen[selected_key] != sha(model_path):
            raise ValueError('Confirmation clock differs from the frozen selected candidate.')
    return dict(schema=SCHEMA, case=case, split=split, variant=variant, collect_labels=collect_labels,
        protocol_sha256=sha(root/'protocol.json'), runtime_sha256=implementation,
        base_model_raw_sha256=BASE_SHA, base_model_sha256=digest(base),
        jaw_model_raw_sha256=expected_jaw, jaw_model_sha256=digest(jaw), jaw_mode=jaw_mode,
        model_raw_sha256=sha(model_path) if model_path is not None else None,
        model_sha256=digest(model) if model is not None else None,
        teacher_config=asdict(teacher), post_grip_parameters=intervention)


def _score(episode, arrays, metadata, goal):
    from .piper_feedback_score import score_feedback_episode
    from .piper_grip_clock_env import analyze_post_grip
    score = score_feedback_episode(episode, arrays, metadata, goal)
    try:
        analysis = analyze_post_grip(arrays, metadata)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        analysis = dict(schema='dvidia.piper-post-grip-analysis/v1', valid=False,
                        reason=str(exc)[:180], interruption_condition=episode.get('condition') == 'post_grip')
    result = dict(score, post_grip_analysis=analysis)
    if not analysis['valid']:
        result.update(valid=False, success=False, native_score_reason=score['reason'],
                      reason='post_grip_intervention_mismatch')
    return result, analysis


def _completed_attempt(folder, binding):
    folder = Path(folder)
    names = {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'score.json', 'post-grip.json'}
    if binding['collect_labels']:
        names.add('samples.npz')
    if (not folder.is_dir() or folder.is_symlink() or set(path.name for path in folder.iterdir()) != names|{'receipt.json'}
            or any(not (folder/name).is_file() or (folder/name).is_symlink() for name in names|{'receipt.json'})):
        raise ValueError('Completed attempt is missing its exact evidence/receipt roster.')
    receipt = read(folder/'receipt.json')
    if (receipt.get('binding') != binding or read(folder/'started.json') != binding
            or receipt.get('case_id') != binding['case']['id'] or receipt.get('variant') != binding['variant']
            or type(receipt.get('files')) is not dict or set(receipt['files']) != names):
        raise ValueError('Completed attempt identity, model, protocol or runtime differs from the request.')
    for name, expected in receipt['files'].items():
        if type(expected) is not str or sha(folder/name) != expected:
            raise ValueError('A completed attempt changed; refusing reuse.')
    episode = load_episode(folder/'episode.json.gz')
    case = binding['case']
    checks = {'schema': EPISODE_SCHEMA, 'case_id': case['id'], 'seed': case['seed'],
              'condition': case['condition'], 'policy_kind': binding['variant'],
              'config': _canonical(asdict(config(case))),
              'base_model_sha256': binding['base_model_sha256'],
              'jaw_model_sha256': binding['jaw_model_sha256'], 'model_sha256': binding['model_sha256'],
              'jaw_mode': binding['jaw_mode'],
              'teacher_config': binding['teacher_config'], 'post_grip_parameters': binding['post_grip_parameters'],
              'implementation_sha256': binding['runtime_sha256']}
    if any(episode.get(key) != expected for key, expected in checks.items()):
        raise ValueError('Completed episode differs from its bound case, configuration or model/runtime.')
    arrays, metadata = _load_npz(folder/'native.npz'), read(folder/'native.json')
    declaration = metadata.get('post_grip_intervention', {})
    expected_parameters = dict(interruption=case['condition'] == 'post_grip', **binding['post_grip_parameters'])
    if declaration.get('parameters') != expected_parameters:
        raise ValueError('Completed native intervention differs from its assigned condition or parameters.')
    score, analysis = _score(episode, arrays, metadata, case['goal_xy'])
    if score != read(folder/'score.json') or analysis != read(folder/'post-grip.json'):
        raise ValueError('Completed attempt differs from independent score/intervention reconstruction.')
    if binding['collect_labels']:
        expected = _samples_from_episode(episode, case['id'])
        actual = _load_npz(folder/'samples.npz')
        if set(actual) != set(expected) or any(not np.array_equal(actual[name], value) for name, value in expected.items()):
            raise ValueError('Completed training samples differ from the complete causal teacher action log.')
    return score


def execute(root, assets, case, variant, folder, *, model_path=None, collect_labels=False):
    """Execute once or reuse an exact complete attempt; never retry physics."""
    root, folder = Path(root), Path(folder)
    binding = _request_binding(root, case, variant, model_path, collect_labels)
    if folder.is_symlink():
        raise ValueError('Attempt evidence must not be a symlink.')
    if (folder/'score.json').exists():
        return _completed_attempt(folder, binding)
    if (folder/'started.json').exists() or folder.exists() and any(folder.iterdir()):
        raise ValueError('Interrupted or unbound attempt retained; automatic rerun refused.')
    write(folder/'started.json', binding)
    episode, arrays, metadata, samples = rollout(assets, case, read(root/'base-model.json'), read(root/'jaw-model.json'),
        variant=variant, clock_document=read(model_path) if model_path is not None else None,
        teacher_config=binding['teacher_config'], intervention_parameters=binding['post_grip_parameters'],
        collect_labels=collect_labels)
    if _request_binding(root, case, variant, model_path, collect_labels) != binding:
        raise ValueError('Attempt input or runtime changed during execution; automatic rerun refused.')
    save_episode(folder/'episode.json.gz', episode)
    _save_npz(folder/'native.npz', arrays)
    write(folder/'native.json', metadata)
    score, analysis = _score(episode, arrays, metadata, case['goal_xy'])
    write(folder/'post-grip.json', analysis)
    write(folder/'score.json', score)
    if collect_labels:
        expected = _samples_from_episode(episode, case['id'])
        supplied = _sample_arrays(samples, case['id'])
        if set(supplied) != set(expected) or any(not np.array_equal(supplied[name], value) for name, value in expected.items()):
            raise ValueError('Training labels differ from their complete causal action log.')
        _save_npz(folder/'samples.npz', supplied)
    names = sorted(path.name for path in folder.iterdir() if path.is_file())
    write(folder/'receipt.json', dict(schema=SCHEMA, files={name: sha(folder/name) for name in names},
        case_id=case['id'], variant=variant, binding=binding))
    print(json.dumps(dict(stage=variant, case=case['id'], valid=score['valid'], success=score['success'],
                         interruption=case['condition'], activated=analysis.get('activated'))), flush=True)
    return score
