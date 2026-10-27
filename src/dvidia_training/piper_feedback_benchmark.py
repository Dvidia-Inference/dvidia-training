"""Frozen CPU study of learned PiPER feedback around the v0.1 primitive.

Synthetic expert queries are available only in collection. Confirmation has
no teacher access in student variants and preserves every scheduled attempt.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from .piper_imitation_benchmark import write, read, sha, save_episode, load_episode, peak_rss, _atomic_create
from .piper_imitation import digest

SCHEMA = 'dvidia.piper-feedback-study/v0.2'
BASE_SHA = '2976a605575d9edefb5db366e72c5e006c0ce85a5531ad3d61512436864c8ded'
VARIANTS = ('feedback', 'blind', 'feedforward', 'teacher', 'open_jaw')
SOURCE_FILES = ('piper_feedback_benchmark.py', 'piper_feedback_model.py', 'piper_feedback_env.py',
                'piper_feedback_score.py', 'piper_motor_primitive.py', 'piper_imitation.py',
                'piper_imitation_score.py', 'piper_task.py', 'reference_task.py',
                'reference_sim.py', 'robot_profiles.py', 'robot_profiles.lock.json')
LIMITS = [
    'Synthetic authored-expert action labels and privileged simulator sensing, not human video or hardware.',
    'Same rigid 40 g block, two trained destinations and narrow starting region as v0.1.',
    'Jaw-actuator pulses target acquisition; no airborne regrasp or general slip recovery is established.',
    'Native servos, joint/jaw caps and gravity compensation remain authored; no task gate or IK in student inference.',
    'Full-step timing includes native telemetry; sensing, transport and physical actuator latency are excluded.',
]
V01_STARTS = frozenset(
    [(x, y) for x in (.284, .294, .304, .314) for y in (-.045, -.035, -.025, -.015)]
    + [(x, y) for x in (.289, .309) for y in (-.04, -.02)]
    + [(.287, -.032), (.291, -.018), (.299, -.041), (.303, -.028), (.311, -.038), (.307, -.022)])


def runtime_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def _canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _pulse_anchor(base, start, target):
    """Schedule from the already frozen reference, never a test rollout."""
    from . import piper_motor_primitive as primitive
    high = base['schema'] == primitive.HIGH_RESOLUTION_SCHEMA
    goal = primitive._goal([*target, .08])
    coefficient = np.einsum('s,stj->tj', primitive._spatial(np.asarray(start), high),
                            np.asarray(base['weights'])[goal])
    for tick in range(350, 701):
        jaw = .035*float(np.clip(primitive._time_basis(tick*.02, high)@coefficient[:, 6], 0., 1.))
        if jaw < .033:
            return tick*.02
    raise ValueError('Frozen reference has no closing transition in the bounded scheduling window.')


def prepare(root, base_model):
    root, base_model = Path(root), Path(base_model)
    if root.exists() or sha(base_model) != BASE_SHA:
        raise ValueError('Use a fresh output and the exact frozen v0.1 model.')
    points = {
        'train': [(x, y) for x in (.285, .295, .305, .313) for y in (-.044, -.034, -.024, -.016)],
        'development': [(x, y) for x in (.290, .310) for y in (-.039, -.019)],
        'confirmation': [(.286, -.031), (.292, -.017), (.298, -.042), (.302, -.027), (.312, -.037), (.306, -.021)],
    }
    all_points = [p for rows in points.values() for p in rows]
    if len(set(all_points)) != len(all_points) or set(all_points)&V01_STARTS:
        raise ValueError('New split starts must be disjoint from each other and every v0.1 start.')
    goals = ((.35, .08), (.40, -.08))
    base = read(base_model)
    cases = {}
    for split, starts in points.items():
        rows = []
        for i, start in enumerate(starts):
            for goal, target in enumerate(goals):
                for condition in ('nominal', 'jaw_interruption'):
                    pulse = []
                    if condition != 'nominal':
                        anchor = _pulse_anchor(base, start, target)
                        if split == 'train':
                            offset, duration, opening = ((.38, .32, .029) if (i+goal)%2 == 0 else (.46, .40, .031))
                        elif split == 'development':
                            offset, duration, opening = (.42, .36, .030)
                        else:
                            offset, duration, opening = (.44, .38, .0305)
                        onset = round(anchor+offset, 3)
                        pulse = [dict(start_s=onset, end_s=round(onset+duration, 3), minimum_opening_m=opening)]
                    rows.append(dict(id=f'{split}-{i:02}-goal{goal}-{condition}', start_xy=list(start),
                                     goal_xy=list(target), seed=271000+i*2+goal, condition=condition, pulses=pulse))
        cases[split] = rows
    root.mkdir(parents=True)
    shutil.copyfile(base_model, root/'base-model.json')
    protocol = dict(schema=SCHEMA, base_model_raw_sha256=BASE_SHA, cases=cases,
        variants=list(VARIANTS), control_dt_s=.02, native_dt_s=.001, timeout_s=20,
        grouping='Both destinations and conditions for each exact initial XY stay in one split.',
        novelty='All starts differ from every v0.1 train/development/confirmation start; goals remain trained.',
        intervention='Explicit jaw-actuator minimum-opening pulse and rate-limited closure tail; no state/force writes. Literal schedule derives only from the frozen v0.1 reference closing transition (<33 mm at 20 ms samples from 7 to 14 s), plus split-specific offsets; no confirmation physics is used to schedule it.',
        primary_endpoint='Unchanged strict v0.1 full native task success. Native reconstruction and terminal-only outcomes are separate.',
        selection='Development only; all candidates and attempts retained. Confirmation starts only after freeze.',
        training='All finite causal expert labels retained, including prefixes of unsuccessful attempts. No success-based frame selection.',
        dagger='At most one 32-case training-only mixed-policy round; expert action probability 0.5 per control tick, seed 2717.',
        physical_robot_ready=False, limitations=LIMITS)
    write(root/'protocol.json', protocol)
    return protocol


def config(case):
    from .piper_task import PiperTaskConfig
    return replace(PiperTaskConfig(), object_position_m=(*case['start_xy'], .082),
                   target_position_m=(*case['goal_xy'], .08))


def _summary(values):
    return ({'samples': len(values), 'p50_s': float(np.percentile(values, 50)),
             'p95_s': float(np.percentile(values, 95)), 'p99_s': float(np.percentile(values, 99)),
             'maximum_s': float(max(values))} if values else None)


def rollout(assets, case, base_document, *, variant, feedback_document=None, collect_labels=False, expert_probability=1.):
    from .piper_feedback_env import FeedbackPiperTaskEnv, JawPulse
    from .piper_feedback_model import FeedbackPolicy, FeedbackFeatures, NominalReference, residual_target
    from .piper_motor_primitive import MotorPrimitivePolicy
    from .piper_task import PiperTaskPolicy
    from .robot_profiles import verify_assets
    from . import reference_sim as sim
    if variant not in (*VARIANTS, 'mixture'):
        raise ValueError('Unknown fixed feedback variant.')
    if collect_labels and variant not in ('teacher', 'mixture') or variant == 'mixture' and not collect_labels:
        raise ValueError('Expert labels are restricted to training collection.')
    if not collect_labels and expert_probability != 1.:
        raise ValueError('Evaluation may not request expert blending.')
    c = config(case)
    env = FeedbackPiperTaskEnv(assets, c, pulses=tuple(JawPulse(**p) for p in case['pulses']))
    result = dict(schema='dvidia.piper-feedback-episode/v0.2', config=_canonical(asdict(c)),
                  case_id=case['id'], seed=case['seed'], condition=case['condition'],
                  pulses=case['pulses'], policy_kind=variant, base_model_sha256=digest(base_document),
                  model_sha256=digest(feedback_document) if feedback_document else None,
                  physical_robot_ready=False, observation_origin='privileged_native_simulator_state',
                  trajectory=[], actions=[], valid=False, success=False, reason='not_started',
                  simulated_seconds=0., error=None, metrics={}, implementation_sha256=runtime_hashes())
    samples = {'features': [], 'blind_features': [], 'targets': [], 'groups': [], 'weights': []}
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
        if variant == 'teacher':
            policy = PiperTaskPolicy(env)
        elif variant in ('feedback', 'blind', 'mixture', 'open_jaw'):
            policy = FeedbackPolicy(base_document, feedback_document, blind=variant == 'blind')
        else:
            policy = MotorPrimitivePolicy(base_document)
        expert = (policy if variant == 'teacher' else PiperTaskPolicy(env)) if collect_labels else None
        nominal, full, blind = NominalReference(base_document), FeedbackFeatures(), FeedbackFeatures(blind=True)
        rng = np.random.default_rng(2717+case['seed'])
        for index in range(1000):
            started = time.perf_counter()
            reference = nominal(obs) if collect_labels else None
            expert_action = expert(obs) if collect_labels else None
            p = time.perf_counter()
            action = expert_action if variant == 'teacher' and collect_labels else policy(obs)
            inference.append(time.perf_counter()-p)
            executed_expert = bool(variant == 'teacher' or variant == 'mixture' and rng.random() < expert_probability)
            if executed_expert and collect_labels:
                action = expert_action
            if variant == 'open_jaw':
                action = dict(action, gripper_position_m=.035)
            record = dict(t_seconds=obs['t_seconds'], **action)
            if collect_labels:
                samples['features'].append(full.features(obs, reference))
                samples['blind_features'].append(blind.features(obs, reference))
                samples['targets'].append(residual_target(expert_action, reference))
                samples['groups'].append(case['id'])
                weight = 5. if expert.issued_phase in ('close', 'lift', 'transfer', 'place', 'release') else 1.
                samples['weights'].append(weight)
                record.update(expert_action=expert_action, executed_expert=executed_expert, expert_phase=expert.issued_phase)
            result['actions'].append(record)
            obs.update(issued_joint_targets_rad=list(action['joint_targets_rad']),
                       issued_gripper_position_m=action['gripper_position_m'], command=record)
            obs, info = env.step(action)
            steps.append(time.perf_counter()-started)
            if obs is not None:
                result['trajectory'].append(obs)
            if env.done:
                break
        result.update(valid=info['valid'], success=info['success'], reason=info['reason'],
                      simulated_seconds=info['simulation_time'], metrics={**info, 'contact_pair_native_ticks': env.contact_pair_native_ticks})
        if hasattr(policy, 'diagnostics'):
            result['policy_diagnostics'] = policy.diagnostics()
    except Exception as exc:
        result.update(reason='exception', error=dict(type=type(exc).__name__, message=str(exc)[:500]))
    result['timing'] = dict(wall_seconds=time.perf_counter()-began, inference=_summary(inference),
                           control_step=_summary(steps), inference_samples_s=inference,
                           control_step_samples_s=steps, missed_20ms_boundaries=sum(v > .02 for v in steps),
                           process_peak_rss_bytes=peak_rss(), scope='CPU computation including dense native telemetry; excludes camera, communication and real actuation.')
    return result, env.native_arrays(), env.native_metadata(), samples


def _save_npz(path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_create(path, buffer.getvalue())


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _request_binding(root, case, variant, model_path, collect_labels, expert_probability):
    """Bind a physical attempt to the complete literal request, before any reuse."""
    from .piper_feedback_model import validate_model
    from .piper_motor_primitive import validate_model as validate_base
    if variant not in (*VARIANTS, 'mixture') or type(collect_labels) is not bool:
        raise ValueError('Unknown variant or invalid collection mode.')
    if (type(expert_probability) not in (int, float) or not np.isfinite(expert_probability)
            or not 0. <= expert_probability <= 1.):
        raise ValueError('Expert probability must be a finite probability.')
    if collect_labels and variant not in ('teacher', 'mixture') or variant == 'mixture' and not collect_labels:
        raise ValueError('Expert labels are restricted to training collection.')
    if not collect_labels and expert_probability != 1.:
        raise ValueError('Evaluation may not request expert blending.')
    protocol = read(root/'protocol.json')
    matches = [split for split, rows in protocol['cases'].items() if _canonical(case) in rows]
    if len(matches) != 1:
        raise ValueError('Attempt case differs from the literal protocol schedule.')
    if collect_labels and matches != ['train']:
        raise ValueError('Expert label collection is restricted to the training split.')
    if matches == ['confirmation'] and not (root/'frozen.json').exists():
        raise ValueError('Confirmation cannot begin before candidate/runtime freeze.')
    needs_model = variant in ('feedback', 'blind', 'open_jaw', 'mixture')
    if needs_model != (model_path is not None):
        raise ValueError('This attempt variant has the wrong model requirement.')
    base = validate_base(read(root/'base-model.json'))
    if sha(root/'base-model.json') != BASE_SHA:
        raise ValueError('Frozen v0.1 base bytes changed.')
    model = validate_model(read(model_path)) if model_path is not None else None
    if model is not None and (model['base_sha256'] != digest(base) or model['blind'] != (variant == 'blind')):
        raise ValueError('Feedback model/base/feature mode differs from the requested variant.')
    return dict(case=_canonical(case), variant=variant, collect_labels=collect_labels,
                expert_probability=float(expert_probability), protocol_sha256=sha(root/'protocol.json'),
                base_model_raw_sha256=sha(root/'base-model.json'), base_model_sha256=digest(base),
                model_raw_sha256=sha(model_path) if model_path is not None else None,
                model_sha256=digest(model) if model is not None else None,
                runtime_sha256=runtime_hashes())


def _completed_attempt(folder, binding):
    """Reuse only a complete, identically bound attempt, independently rescored."""
    from .piper_feedback_score import score_feedback_episode
    folder = Path(folder)
    names = {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'score.json'}
    if binding['collect_labels']:
        names.add('samples.npz')
    if not (folder/'receipt.json').is_file() or any(not (folder/name).is_file() for name in names):
        raise ValueError('Completed attempt is missing its exact evidence/receipt roster.')
    receipt = read(folder/'receipt.json')
    if (receipt.get('binding') != binding or read(folder/'started.json') != binding
            or receipt.get('case_id') != binding['case']['id'] or receipt.get('variant') != binding['variant']
            or type(receipt.get('files')) is not dict or set(receipt['files']) != names):
        raise ValueError('Completed attempt identity, model, protocol or runtime differs from the request.')
    for name, expected in receipt['files'].items():
        if not isinstance(expected, str) or sha(folder/name) != expected:
            raise ValueError('A completed attempt changed; refusing reuse.')
    episode = load_episode(folder/'episode.json.gz')
    case = binding['case']
    if (episode.get('case_id') != case['id'] or episode.get('seed') != case['seed']
            or episode.get('condition') != case['condition'] or episode.get('pulses') != case['pulses']
            or episode.get('policy_kind') != binding['variant']
            or episode.get('config') != _canonical(asdict(config(case)))
            or episode.get('base_model_sha256') != binding['base_model_sha256']
            or episode.get('model_sha256') != binding['model_sha256']
            or episode.get('implementation_sha256') != binding['runtime_sha256']):
        raise ValueError('Completed episode identity/configuration/model/runtime differs from its bound request.')
    arrays, metadata = _load_npz(folder/'native.npz'), read(folder/'native.json')
    if metadata.get('pulses') != case['pulses']:
        raise ValueError('Completed native fault schedule differs from the requested case.')
    saved_score = read(folder/'score.json')
    if score_feedback_episode(episode, arrays, metadata, case['goal_xy']) != saved_score:
        raise ValueError('Completed attempt score differs from independent reconstruction.')
    return saved_score


def execute(root, assets, case, variant, folder, *, model_path=None, collect_labels=False, expert_probability=1.):
    from .piper_feedback_score import score_feedback_episode
    root, folder = Path(root), Path(folder)
    tape, native, meta, score_path = (folder/name for name in ('episode.json.gz', 'native.npz', 'native.json', 'score.json'))
    binding = _request_binding(root, case, variant, model_path, collect_labels, expert_probability)
    if score_path.exists():
        return _completed_attempt(folder, binding)
    marker = folder/'started.json'
    if marker.exists():
        raise ValueError('Unfinished physical attempt exists; do not automatically rerun it.')
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Attempt directory contains unbound artifacts; use a fresh directory.')
    write(marker, binding)
    episode, arrays, metadata, samples = rollout(assets, case, read(root/'base-model.json'), variant=variant,
        feedback_document=read(model_path) if model_path else None, collect_labels=collect_labels,
        expert_probability=expert_probability)
    save_episode(tape, episode)
    _save_npz(native, arrays)
    write(meta, metadata)
    score = score_feedback_episode(episode, arrays, metadata, case['goal_xy'])
    write(score_path, score)
    if collect_labels:
        _save_npz(folder/'samples.npz', {k: np.asarray(v, dtype='<U100' if k == 'groups' else np.float64) for k, v in samples.items()})
    files = [p for p in folder.iterdir() if p.is_file() and p.name != 'receipt.json']
    write(folder/'receipt.json', dict(files={p.name: sha(p) for p in files}, case_id=case['id'], variant=variant,
                                      binding=binding))
    print(json.dumps(dict(stage=variant, case=case['id'], valid=score['valid'], success=score['success'])), flush=True)
    return score


def collect(root, assets, *, candidate=None):
    root = Path(root)
    protocol = read(root/'protocol.json')
    rows = protocol['cases']['train']
    folder = root/'demonstrations' if candidate is None else root/'dagger'
    if candidate is not None:
        rows = [c for c in rows if c['condition'] == 'jaw_interruption']
    for case in rows:
        execute(root, assets, case, 'teacher' if candidate is None else 'mixture', folder/case['id'],
                model_path=root/'candidates'/candidate/'model.json' if candidate else None,
                collect_labels=True, expert_probability=.5 if candidate else 1.)
    return {'attempts': len(rows), 'folder': folder.name}


def dataset(root, *, dagger=False, blind=False):
    files = sorted((Path(root)/'demonstrations').glob('*/samples.npz'))
    if dagger:
        files += sorted((Path(root)/'dagger').glob('*/samples.npz'))
    if not files:
        raise ValueError('Collect demonstrations first.')
    batches = [_load_npz(p) for p in files]
    return dict(features=np.concatenate([b['blind_features' if blind else 'features'] for b in batches]),
                targets=np.concatenate([b['targets'] for b in batches]),
                groups=np.concatenate([b['groups'] for b in batches]).tolist(),
                weights=np.concatenate([b['weights'] for b in batches]))


def fit(root, name, *, blind=False, dagger=False, basis=256, ridge=.001, lengthscale=1.):
    from .piper_feedback_model import train
    root = Path(root)
    if not name.isalnum() or len(name) > 32:
        raise ValueError('Use a short alphanumeric candidate name.')
    directory = root/'candidates'/name
    if directory.exists() or (root/'frozen.json').exists():
        raise ValueError('Candidate exists or confirmation is already frozen.')
    data = dataset(root, dagger=dagger, blind=blind)
    began = time.perf_counter()
    model = train({k: data[k] for k in ('features', 'targets', 'groups')},
                  base_sha256=digest(read(root/'base-model.json')), blind=blind,
                  basis=basis, ridge=ridge, lengthscale=lengthscale, sample_weights=data['weights'])
    write(directory/'model.json', model)
    receipt = dict(model_sha256=sha(directory/'model.json'), protocol_sha256=sha(root/'protocol.json'),
                   source_sha256=runtime_hashes(), blind=blind, dagger=dagger,
                   training_seconds=time.perf_counter()-began, model_bytes=(directory/'model.json').stat().st_size,
                   training=model['training'])
    write(directory/'receipt.json', receipt)
    return receipt


def development(root, assets, name):
    root = Path(root)
    model = root/'candidates'/name/'model.json'
    blind = read(model)['blind']
    rows = []
    for case in read(root/'protocol.json')['cases']['development']:
        score = execute(root, assets, case, 'blind' if blind else 'feedback', root/'candidates'/name/'development'/case['id'], model_path=model)
        rows.append(dict(case_id=case['id'], condition=case['condition'], score=score))
    result = dict(model_sha256=sha(model), protocol_sha256=sha(root/'protocol.json'),
                  runtime_sha256=runtime_hashes(), rows=rows,
                  successes=sum(r['score']['success'] for r in rows), scheduled=len(rows))
    if not (root/'candidates'/name/'development.json').exists():
        write(root/'candidates'/name/'development.json', result)
    elif read(root/'candidates'/name/'development.json') != result:
        raise ValueError('Saved development aggregate differs from its exact bound attempts.')
    return result


def _validated_candidate(root, name, expected_blind, protocol, runtime):
    from .piper_feedback_model import validate_model, _dataset_hash
    if type(name) is not str or not name.isalnum() or len(name) > 32:
        raise ValueError('Use a short alphanumeric candidate name.')
    directory = root/'candidates'/name
    model_path = directory/'model.json'
    model = validate_model(read(model_path))
    receipt = read(directory/'receipt.json')
    if (model['blind'] is not expected_blind or receipt.get('blind') is not expected_blind
            or type(receipt.get('dagger')) is not bool
            or model['base_sha256'] != digest(read(root/'base-model.json'))
            or sha(model_path) != receipt.get('model_sha256')
            or receipt.get('protocol_sha256') != sha(root/'protocol.json')
            or receipt.get('source_sha256') != runtime
            or receipt.get('training') != model['training']):
        raise ValueError('Candidate feature mode, base, fitted model, training, protocol or runtime binding changed.')
    data = dataset(root, dagger=receipt['dagger'], blind=expected_blind)
    expected_groups = list(dict.fromkeys(data['groups']))
    expected_roster = {row['id'] for row in protocol['cases']['train']}
    training = model['training']
    raw_hash = _dataset_hash(data['features'], data['targets'], data['groups'], data['weights'])
    if (set(expected_groups) != expected_roster or training['train_groups'] != expected_groups
            or training['validation_groups'] != [] or training['sample_count'] != len(data['groups'])
            or training['source_sample_count'] != len(data['groups'])
            or training['dataset_sha256'] != raw_hash or training['source_dataset_sha256'] != raw_hash):
        raise ValueError('Candidate does not bind exactly the complete collected training episodes.')
    evaluation = read(directory/'development.json')
    if (evaluation.get('model_sha256') != sha(model_path)
            or evaluation.get('protocol_sha256') != sha(root/'protocol.json')
            or evaluation.get('runtime_sha256') != runtime):
        raise ValueError('Development evaluation model/protocol/runtime binding changed.')
    rows = []
    variant = 'blind' if expected_blind else 'feedback'
    for case in protocol['cases']['development']:
        binding = _request_binding(root, case, variant, model_path, False, 1.)
        score = _completed_attempt(directory/'development'/case['id'], binding)
        rows.append(dict(case_id=case['id'], condition=case['condition'], score=score))
    if (evaluation.get('rows') != rows or evaluation.get('scheduled') != len(rows)
            or evaluation.get('successes') != sum(row['score']['success'] for row in rows)):
        raise ValueError('Development aggregate is incomplete or differs from the scheduled bound attempts.')
    return model, receipt


def freeze(root, selected, blind):
    root = Path(root)
    if (root/'frozen.json').exists():
        raise ValueError('Study is already frozen.')
    protocol, runtime = read(root/'protocol.json'), runtime_hashes()
    if sha(root/'base-model.json') != BASE_SHA or protocol.get('base_model_raw_sha256') != BASE_SHA:
        raise ValueError('Frozen v0.1 base or its protocol binding changed.')
    full_model, full_receipt = _validated_candidate(root, selected, False, protocol, runtime)
    blind_model, blind_receipt = _validated_candidate(root, blind, True, protocol, runtime)
    matched = ('algorithm', 'seed', 'basis', 'lengthscale', 'ridge', 'sample_count',
               'source_sample_count', 'train_groups', 'validation_groups', 'weight_sum', 'random_basis_sha256')
    if (any(full_model['training'][key] != blind_model['training'][key] for key in matched)
            or full_receipt['dagger'] != blind_receipt['dagger']
            or full_model['frequencies'] != blind_model['frequencies']
            or full_model['phases'] != blind_model['phases']):
        raise ValueError('Full and blind controls must share architecture, training groups and sample/weight budget.')
    frozen = dict(schema=SCHEMA, protocol_sha256=sha(root/'protocol.json'), base_model_sha256=BASE_SHA,
                  selected=selected, blind=blind, selected_sha256=sha(root/'candidates'/selected/'model.json'),
                  blind_sha256=sha(root/'candidates'/blind/'model.json'), runtime_sha256=runtime,
                  selected_receipt_sha256=sha(root/'candidates'/selected/'receipt.json'),
                  blind_receipt_sha256=sha(root/'candidates'/blind/'receipt.json'),
                  selected_development_sha256=sha(root/'candidates'/selected/'development.json'),
                  blind_development_sha256=sha(root/'candidates'/blind/'development.json'))
    write(root/'frozen.json', frozen)
    return frozen


def confirm(root, assets):
    root = Path(root)
    frozen, protocol = read(root/'frozen.json'), read(root/'protocol.json')
    if (frozen['runtime_sha256'] != runtime_hashes() or frozen['protocol_sha256'] != sha(root/'protocol.json')
            or sha(root/'base-model.json') != frozen['base_model_sha256']):
        raise ValueError('Frozen source, protocol or base model changed.')
    models = {kind: root/'candidates'/frozen['blind' if kind == 'blind' else 'selected']/'model.json'
              for kind in ('feedback', 'blind', 'open_jaw')}
    if sha(models['feedback']) != frozen['selected_sha256'] or sha(models['blind']) != frozen['blind_sha256']:
        raise ValueError('A frozen feedback model changed.')
    for name_key, prefix in (('selected', 'selected'), ('blind', 'blind')):
        directory = root/'candidates'/frozen[name_key]
        for filename, suffix in (('receipt.json', 'receipt'), ('development.json', 'development')):
            if sha(directory/filename) != frozen[f'{prefix}_{suffix}_sha256']:
                raise ValueError('A frozen candidate training/development receipt changed.')
    rows = []
    # Cases interleave all five variants under the same literal fault schedule.
    for case in protocol['cases']['confirmation']:
        for variant in VARIANTS:
            score = execute(root, assets, case, variant, root/'confirmation'/case['id']/variant,
                            model_path=models.get(variant))
            rows.append(dict(case_id=case['id'], condition=case['condition'], variant=variant, score=score))
    report = dict(schema=SCHEMA, frozen_sha256=sha(root/'frozen.json'), protocol_sha256=sha(root/'protocol.json'),
                  rows=rows, variants={v: dict(scheduled=sum(r['variant'] == v for r in rows),
                      successes=sum(r['variant'] == v and r['score']['success'] for r in rows),
                      valid=sum(r['variant'] == v and r['score']['valid'] for r in rows)) for v in VARIANTS},
                  physical_robot_ready=False, limitations=LIMITS)
    if not (root/'report.json').exists():
        write(root/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'collect', 'dagger', 'train', 'development', 'freeze', 'confirm'))
    parser.add_argument('run', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--base-model', type=Path)
    parser.add_argument('--name', default='rff256')
    parser.add_argument('--blind-name', default='blind256')
    parser.add_argument('--blind', action='store_true')
    parser.add_argument('--with-dagger', action='store_true')
    parser.add_argument('--basis', type=int, default=256)
    parser.add_argument('--ridge', type=float, default=.001)
    parser.add_argument('--lengthscale', type=float, default=1.)
    args = parser.parse_args(argv)
    if args.command in ('collect', 'dagger', 'development', 'confirm') and args.assets is None:
        parser.error('Native execution requires --assets.')
    if args.command == 'prepare':
        if args.base_model is None:
            parser.error('prepare requires --base-model.')
        result = prepare(args.run, args.base_model)
    elif args.command in ('collect', 'dagger'):
        result = collect(args.run, args.assets, candidate=args.name if args.command == 'dagger' else None)
    elif args.command == 'train':
        result = fit(args.run, args.name, blind=args.blind, dagger=args.with_dagger,
                     basis=args.basis, ridge=args.ridge, lengthscale=args.lengthscale)
    elif args.command == 'development':
        result = development(args.run, args.assets, args.name)
    elif args.command == 'freeze':
        result = freeze(args.run, args.name, args.blind_name)
    else:
        result = confirm(args.run, args.assets)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
