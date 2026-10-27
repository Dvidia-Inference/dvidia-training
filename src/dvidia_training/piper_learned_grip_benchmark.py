"""A lineage-bound jaw-only study after the unsuccessful full residual study.

The imported training evidence is preserved under its original protocol and
runtime. New attempts have their own protocol, sources, models and receipts.
Development was reused adaptively; confirmation is inaccessible until freeze.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import piper_feedback_benchmark as prior
from .piper_imitation_benchmark import write, read, sha, save_episode, load_episode
from .piper_imitation import digest

SCHEMA = prior.SCHEMA
BASE_SHA = prior.BASE_SHA
VARIANTS = prior.VARIANTS
config = prior.config
_canonical = prior._canonical
_save_npz = prior._save_npz
_load_npz = prior._load_npz
_completed_attempt = prior._completed_attempt
SOURCE_FILES = ('piper_learned_grip_benchmark.py', 'piper_grip_adapter.py',
                *prior.SOURCE_FILES)
LIMITS = [
    'Synthetic authored-expert jaw labels and privileged simulator sensing; not human video or hardware.',
    'Only the jaw is controlled by the new learned adapter; all six arm commands remain the frozen v0.1 primitive.',
    'Same rigid 40 g block, two trained destinations and narrow starting region as v0.1.',
    'Development starts were reused after five full-residual candidates each failed all 16 development cases.',
    'Interventions affect acquisition; airborne regrasp and general slip recovery are not established.',
    'Native servos, joint/jaw caps and gravity compensation remain authored.',
    'Timing excludes sensing, network transport and physical actuators; no hard real-time guarantee.',
]


def runtime_hashes():
    return {name: sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def _files(root):
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Study evidence must not contain symlinks.')
        if p.is_file():
            result[p.relative_to(root).as_posix()] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    return result


def prepare(root, parent):
    root, parent = Path(root), Path(parent)
    if root.exists() or root.is_symlink() or not parent.is_dir() or (parent/'frozen.json').exists():
        raise ValueError('Use a fresh output and the unfrozen development parent study.')
    protocol = read(parent/'protocol.json')
    if (protocol['schema'] != SCHEMA or sha(parent/'base-model.json') != BASE_SHA
            or (parent/'confirmation').exists()):
        raise ValueError('Parent must have the exact frozen base and untouched confirmation.')
    names = sorted(p.name for p in (parent/'candidates').iterdir() if p.is_dir())
    if names != ['dag256', 'dag512', 'dagstrong128', 'full256', 'full512']:
        raise ValueError('Parent must preserve the exact five unsuccessful development candidates.')
    for name in names:
        evaluation = read(parent/'candidates'/name/'development.json')
        if evaluation['scheduled'] != 16 or evaluation['successes'] != 0:
            raise ValueError('Parent development lineage differs from the declared five negative fits.')
    source = parent/'demonstrations'
    if {p.parent.name for p in source.glob('*/samples.npz')} != {c['id'] for c in protocol['cases']['train']}:
        raise ValueError('The complete 64 teacher episodes are required.')
    # Validate original runtime while it is still available, before copying.
    for case in protocol['cases']['train']:
        binding = prior._request_binding(parent, case, 'teacher', None, True, 1.)
        prior._completed_attempt(source/case['id'], binding)
    root.mkdir(parents=True)
    history = root/'prior-studies'/'study01'
    shutil.copytree(parent, history)
    for name, expected in prior.runtime_hashes().items():
        target = history/'sources'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path(prior.__file__).with_name(name), target)
        if sha(target) != expected:
            raise ValueError('Parent runtime changed during evidence capture.')
    manifest = dict(schema='dvidia.piper-feedback-training-parent/v1',
        path='prior-studies/study01', protocol_sha256=sha(history/'protocol.json'),
        runtime_sha256=prior.runtime_hashes(), files=_files(history),
        teacher_sample_files={p.relative_to(history).as_posix(): sha(p)
                              for p in sorted((history/'demonstrations').glob('*/samples.npz'))},
        negative_candidates=names, confirmation_attempts=0,
        use='All 64 original teacher episodes; DAgger and failed development records retained for transparency, excluded from jaw-only fitting.')
    write(root/'training-parent.json', manifest)
    shutil.copyfile(parent/'base-model.json', root/'base-model.json')
    protocol.update(architecture='jaw-only binary CART over a frozen six-joint learned primitive',
        training_parent_manifest_sha256=sha(root/'training-parent.json'),
        training='Reuse all 64 original teacher episodes and original causal labels, without physical recollection. Original protocol/runtime bindings remain intact in prior-studies/study01.',
        dagger='No further correction collection; parent mixed-policy data is retained but excluded from this model family.',
        selection='Adaptively reused development only; retain every new candidate and attempt. Confirmation untouched until model/source freeze.',
        limitations=LIMITS)
    write(root/'protocol.json', protocol)
    return protocol


def _parent(root):
    manifest = read(root/'training-parent.json')
    if (manifest.get('schema') != 'dvidia.piper-feedback-training-parent/v1'
            or manifest.get('path') != 'prior-studies/study01'
            or sha(root/'training-parent.json') != read(root/'protocol.json')['training_parent_manifest_sha256']):
        raise ValueError('Training parent lineage changed.')
    parent = root/'prior-studies'/'study01'
    if _files(parent) != manifest['files']:
        raise ValueError('An imported parent evidence file changed or disappeared.')
    return parent


def rollout(assets, case, base_document, *, variant, feedback_document=None,
            collect_labels=False, expert_probability=1.):
    """Fresh full-duration native attempt; student inference has no expert handle."""
    from .piper_feedback_env import FeedbackPiperTaskEnv, JawPulse
    from .piper_grip_adapter import GripPolicy
    from .piper_motor_primitive import MotorPrimitivePolicy
    from .piper_task import PiperTaskPolicy
    from .robot_profiles import verify_assets
    from . import reference_sim as sim
    if variant not in VARIANTS or collect_labels or expert_probability != 1.:
        raise ValueError('This study permits evaluation only, without expert labels or blending.')
    c = config(case)
    env = FeedbackPiperTaskEnv(assets, c, pulses=tuple(JawPulse(**p) for p in case['pulses']))
    result = dict(schema='dvidia.piper-feedback-episode/v0.2', config=_canonical(asdict(c)),
        case_id=case['id'], seed=case['seed'], condition=case['condition'], pulses=case['pulses'],
        policy_kind=variant, base_model_sha256=digest(base_document),
        model_sha256=digest(feedback_document) if feedback_document else None,
        physical_robot_ready=False, observation_origin='privileged_native_simulator_state',
        trajectory=[], actions=[], valid=False, success=False, reason='not_started',
        simulated_seconds=0., error=None, metrics={}, implementation_sha256=runtime_hashes())
    inference, steps = [], []
    began = time.perf_counter()
    try:
        obs, info = env.reset(case['seed'])
        mj, _ = sim._native_imports()
        import platform
        result.update(initial_object_position_m=env.initial_object_position.tolist(),
            compiled_model_sha256=sim.model_digest(env.model), asset_provenance=verify_assets('piper', assets),
            native_servo_gains=env.model.actuator_gainprm[:, 0].tolist(),
            native_force_limits=env.model.actuator_forcerange.tolist(),
            platform=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version(),
                          numpy=np.__version__, mujoco=mj.__version__))
        result['trajectory'].append(obs)
        if variant == 'teacher':
            policy = PiperTaskPolicy(env)
        elif variant == 'feedforward':
            policy = MotorPrimitivePolicy(base_document)
        else:
            policy = GripPolicy(base_document, feedback_document, blind=variant == 'blind')
        for _ in range(1000):
            started = time.perf_counter()
            p = time.perf_counter()
            action = policy(obs)
            inference.append(time.perf_counter()-p)
            if variant == 'open_jaw':
                action = dict(action, gripper_position_m=.035)
            record = dict(t_seconds=obs['t_seconds'], **action)
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
    result['timing'] = dict(wall_seconds=time.perf_counter()-began,
        inference=prior._summary(inference), control_step=prior._summary(steps),
        inference_samples_s=inference, control_step_samples_s=steps,
        missed_20ms_boundaries=sum(v > .02 for v in steps), process_peak_rss_bytes=prior.peak_rss(),
        scope='CPU computation including dense native telemetry; excludes camera, communication and real actuation.')
    return result, env.native_arrays(), env.native_metadata(), {}


def _request_binding(root, case, variant, model_path, collect_labels=False, expert_probability=1.):
    from .piper_grip_adapter import validate_model
    from .piper_motor_primitive import validate_model as validate_base
    if variant not in VARIANTS or collect_labels or expert_probability != 1.:
        raise ValueError('Only evaluation without labels/blending is supported.')
    protocol = read(root/'protocol.json')
    matches = [split for split, rows in protocol['cases'].items() if _canonical(case) in rows]
    if len(matches) != 1 or matches[0] not in ('development', 'confirmation'):
        raise ValueError('Only a literal scheduled development or confirmation case may execute.')
    if matches == ['confirmation'] and not (root/'frozen.json').exists():
        raise ValueError('Confirmation cannot begin before candidate/runtime freeze.')
    if (variant in ('feedback', 'blind', 'open_jaw')) != (model_path is not None):
        raise ValueError('Variant model requirement differs.')
    base = validate_base(read(root/'base-model.json'))
    if sha(root/'base-model.json') != BASE_SHA:
        raise ValueError('Frozen base bytes changed.')
    model = validate_model(read(model_path)) if model_path is not None else None
    if model and (model['base_sha256'] != digest(base) or model['blind'] != (variant == 'blind')):
        raise ValueError('Model/base/feature mode mismatch.')
    return dict(case=_canonical(case), variant=variant, collect_labels=False, expert_probability=1.,
        protocol_sha256=sha(root/'protocol.json'), base_model_raw_sha256=BASE_SHA,
        base_model_sha256=digest(base), model_raw_sha256=sha(model_path) if model_path else None,
        model_sha256=digest(model) if model else None, runtime_sha256=runtime_hashes())


def execute(root, assets, case, variant, folder, *, model_path=None):
    from .piper_feedback_score import score_feedback_episode
    root, folder = Path(root), Path(folder)
    binding = _request_binding(root, case, variant, model_path)
    if (folder/'score.json').exists():
        return _completed_attempt(folder, binding)
    if (folder/'started.json').exists() or folder.exists() and any(folder.iterdir()):
        raise ValueError('Interrupted or unbound attempt retained; automatic rerun refused.')
    write(folder/'started.json', binding)
    episode, arrays, metadata, _ = rollout(assets, case, read(root/'base-model.json'),
        variant=variant, feedback_document=read(model_path) if model_path else None)
    save_episode(folder/'episode.json.gz', episode)
    _save_npz(folder/'native.npz', arrays)
    write(folder/'native.json', metadata)
    score = score_feedback_episode(episode, arrays, metadata, case['goal_xy'])
    write(folder/'score.json', score)
    write(folder/'receipt.json', dict(files={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
        case_id=case['id'], variant=variant, binding=binding))
    print(json.dumps(dict(stage=variant, case=case['id'], valid=score['valid'], success=score['success'])), flush=True)
    return score


def fit(root, name, *, blind=False, max_depth=6, min_leaf=64, max_bins=31):
    from .piper_grip_adapter import train_parent_data
    root = Path(root)
    directory = root/'candidates'/name
    if not name.isalnum() or len(name) > 32 or directory.exists() or (root/'frozen.json').exists():
        raise ValueError('Use a fresh short candidate name before confirmation freeze.')
    parent = _parent(root)
    began = time.perf_counter()
    model = train_parent_data(parent, blind=blind, max_depth=max_depth, min_leaf=min_leaf, max_bins=max_bins)
    write(directory/'model.json', model)
    receipt = dict(model_sha256=sha(directory/'model.json'), protocol_sha256=sha(root/'protocol.json'),
        source_sha256=runtime_hashes(), blind=blind, dagger=False,
        training_parent_sha256=sha(root/'training-parent.json'),
        training_seconds=time.perf_counter()-began, model_bytes=(directory/'model.json').stat().st_size,
        training=model['training'])
    write(directory/'receipt.json', receipt)
    return receipt


def development(root, assets, name):
    root = Path(root)
    if (root/'frozen.json').exists():
        raise ValueError('No development tuning after freeze.')
    model = root/'candidates'/name/'model.json'
    variant = 'blind' if read(model)['blind'] else 'feedback'
    rows = [dict(case_id=c['id'], condition=c['condition'],
                 score=execute(root, assets, c, variant, root/'candidates'/name/'development'/c['id'], model_path=model))
            for c in read(root/'protocol.json')['cases']['development']]
    result = dict(model_sha256=sha(model), protocol_sha256=sha(root/'protocol.json'),
        runtime_sha256=runtime_hashes(), rows=rows, successes=sum(r['score']['success'] for r in rows), scheduled=len(rows))
    path = root/'candidates'/name/'development.json'
    if path.exists() and read(path) != result:
        raise ValueError('Saved development aggregate differs.')
    if not path.exists():
        write(path, result)
    return result


def freeze(root, selected, blind):
    from .piper_grip_adapter import validate_model
    root = Path(root)
    if (root/'frozen.json').exists():
        raise ValueError('Study already frozen.')
    parent = _parent(root)
    parent_protocol = read(parent/'protocol.json')
    expected_groups = [c['id'] for c in parent_protocol['cases']['train']]
    models = []
    for name, expected_blind in ((selected, False), (blind, True)):
        directory = root/'candidates'/name
        model = validate_model(read(directory/'model.json'))
        receipt, evaluation = read(directory/'receipt.json'), read(directory/'development.json')
        if (model['blind'] is not expected_blind or receipt['blind'] is not expected_blind
                or model['base_sha256'] != digest(read(root/'base-model.json'))
                or receipt['model_sha256'] != sha(directory/'model.json')
                or receipt['source_sha256'] != runtime_hashes() or receipt['training'] != model['training']
                or receipt['training_parent_sha256'] != sha(root/'training-parent.json')
                or receipt['protocol_sha256'] != sha(root/'protocol.json')
                or evaluation['model_sha256'] != sha(directory/'model.json')
                or evaluation['protocol_sha256'] != sha(root/'protocol.json')
                or evaluation['runtime_sha256'] != runtime_hashes()):
            raise ValueError('Candidate model/training/development/source lineage changed.')
        training = model['training']
        lineage = training.get('parent_artifacts')
        if (training['train_groups'] != expected_groups or training['sample_count'] != 64000
                or not isinstance(lineage, dict)
                or lineage.get('protocol_sha256') != sha(parent/'protocol.json')
                or lineage.get('base_model_raw_sha256') != BASE_SHA
                or lineage.get('episodes') != [dict(case_id=name,
                    episode_sha256=sha(parent/'demonstrations'/name/'episode.json.gz'), samples=1000)
                    for name in expected_groups]):
            raise ValueError('Model must bind exactly all 64 original teacher episodes and their causal labels.')
        rows = []
        for c in read(root/'protocol.json')['cases']['development']:
            variant = 'blind' if expected_blind else 'feedback'
            score = _completed_attempt(directory/'development'/c['id'], _request_binding(root, c, variant, directory/'model.json'))
            rows.append(dict(case_id=c['id'], condition=c['condition'], score=score))
        if evaluation['rows'] != rows or evaluation['scheduled'] != 16 or evaluation['successes'] != sum(r['score']['success'] for r in rows):
            raise ValueError('Incomplete or changed development aggregate.')
        models.append(model)
    # Input masking is the sole difference between the matched learned models.
    recipe_keys = ('algorithm', 'max_depth', 'min_leaf', 'max_bins', 'sample_count',
                   'scene_count', 'train_groups', 'weight_sum')
    if any(models[0]['training'][k] != models[1]['training'][k] for k in recipe_keys):
        raise ValueError('Full and blind models must share the complete training budget and tree recipe.')
    frozen = dict(schema=SCHEMA, protocol_sha256=sha(root/'protocol.json'), base_model_sha256=BASE_SHA,
        selected=selected, blind=blind, selected_sha256=sha(root/'candidates'/selected/'model.json'),
        blind_sha256=sha(root/'candidates'/blind/'model.json'), runtime_sha256=runtime_hashes(),
        training_parent_sha256=sha(root/'training-parent.json'))
    for prefix, name in (('selected', selected), ('blind', blind)):
        for key, file in (('receipt', 'receipt.json'), ('development', 'development.json')):
            frozen[f'{prefix}_{key}_sha256'] = sha(root/'candidates'/name/file)
    write(root/'frozen.json', frozen)
    return frozen


def confirm(root, assets):
    root = Path(root)
    _parent(root)
    frozen, protocol = read(root/'frozen.json'), read(root/'protocol.json')
    if (frozen['runtime_sha256'] != runtime_hashes() or frozen['protocol_sha256'] != sha(root/'protocol.json')
            or frozen.get('training_parent_sha256') != sha(root/'training-parent.json')
            or sha(root/'base-model.json') != frozen['base_model_sha256']):
        raise ValueError('Frozen runtime/protocol/base changed.')
    for prefix, name in (('selected', frozen['selected']), ('blind', frozen['blind'])):
        for key, file in (('', 'model.json'), ('_receipt', 'receipt.json'), ('_development', 'development.json')):
            if sha(root/'candidates'/name/file) != frozen[f'{prefix}{key}_sha256']:
                raise ValueError('A frozen candidate artifact changed.')
    models = {v: root/'candidates'/frozen['blind' if v == 'blind' else 'selected']/'model.json'
              for v in ('feedback', 'blind', 'open_jaw')}
    rows = []
    for c in protocol['cases']['confirmation']:
        for variant in VARIANTS:
            score = execute(root, assets, c, variant, root/'confirmation'/c['id']/variant, model_path=models.get(variant))
            rows.append(dict(case_id=c['id'], condition=c['condition'], variant=variant, score=score))
    report = dict(schema=SCHEMA, frozen_sha256=sha(root/'frozen.json'), protocol_sha256=sha(root/'protocol.json'),
        rows=rows, variants={v: dict(scheduled=sum(r['variant'] == v for r in rows),
            successes=sum(r['variant'] == v and r['score']['success'] for r in rows),
            valid=sum(r['variant'] == v and r['score']['valid'] for r in rows)) for v in VARIANTS},
        physical_robot_ready=False, limitations=LIMITS)
    if (root/'report.json').exists() and read(root/'report.json') != report:
        raise ValueError('Saved confirmation aggregate changed.')
    if not (root/'report.json').exists():
        write(root/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'development', 'freeze', 'confirm'))
    parser.add_argument('run', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--name', default='jaw6')
    parser.add_argument('--blind-name', default='blind6')
    parser.add_argument('--blind', action='store_true')
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--min-leaf', type=int, default=64)
    parser.add_argument('--max-bins', type=int, default=31)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.run, args.parent)
    elif args.command == 'train':
        result = fit(args.run, args.name, blind=args.blind, max_depth=args.max_depth,
                     min_leaf=args.min_leaf, max_bins=args.max_bins)
    elif args.command == 'development':
        result = development(args.run, args.assets, args.name)
    elif args.command == 'freeze':
        result = freeze(args.run, args.name, args.blind_name)
    else:
        result = confirm(args.run, args.assets)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'files', 'cases', 'training')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
