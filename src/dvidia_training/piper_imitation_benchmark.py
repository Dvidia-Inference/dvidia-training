"""Reproducible CPU imitation study: synthetic demonstrations to PiPER actions.

Training/development and confirmation are separate commands. Confirmation refuses
to start before a model is frozen and never overwrites a previous attempt.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import time
import tempfile

import numpy as np

SCHEMA = 'dvidia.piper-imitation-study/v1'
VARIANTS = ('learned', 'teacher', 'open_jaw', 'hold', 'shuffled_labels')
LIMITS = [
    'Synthetic state/action demonstrations from an authored PiPER controller, not human video or teleoperation.',
    'Learned inference uses privileged simulator joint, TCP, object and contact measurements; no vision pipeline is trained.',
    'Only starting-position interpolation for a 40 g rigid block and two trained destinations is evaluated.',
    'The pinned native servo and rate limiter remain authored low-level control; no teacher, phase machine or IK is called by the student.',
    'Uncalibrated simulated friction, contacts and actuators; no physical robot qualification or other-arm transfer.',
    'Compute latency excludes sensing, camera processing, communication and physical actuation; CPU/GPU minima are not certified.',
]
RUNTIME_FILES = ('piper_imitation.py', 'piper_imitation_score.py', 'piper_imitation_benchmark.py',
                 'piper_motor_primitive.py', 'piper_task.py', 'reference_task.py', 'reference_sim.py', 'robot_profiles.py', 'robot_profiles.lock.json')
TEACHER_FILES = ('piper_task.py','robot_profiles.py','robot_profiles.lock.json','reference_task.py','reference_sim.py')


def runtime_hashes():
    return {name:sha(Path(__file__).with_name(name)) for name in RUNTIME_FILES}


def peak_rss():
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if platform.system()=='Darwin' else 1024)
    except ImportError: return None


def write(path, value):
    _atomic_create(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def _atomic_create(path, raw):
    # Link a fully flushed temporary inode into place exclusively. A crash may
    # leave an orphan .writing file, but never a partially visible final tape.
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.writing-',delete=False) as stream:
            temporary=Path(stream.name); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def sha(path): return sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def save_episode(path, episode):
    raw = json.dumps(episode, separators=(',', ':'), allow_nan=False).encode()
    _atomic_create(path,gzip.compress(raw,mtime=0))


def load_episode(path): return json.loads(gzip.decompress(Path(path).read_bytes()))


def prepare(root):
    root = Path(root)
    if root.exists(): raise ValueError('Use a fresh study directory.')
    targets = [(0.35, 0.08), (0.40, -0.08)]
    starts = {'train': [(x, y) for x in (.284, .294, .304, .314) for y in (-.045, -.035, -.025, -.015)],
              'development': [(x, y) for x in (.289, .309) for y in (-.04, -.02)],
              'confirmation': [(.287, -.032), (.291, -.018), (.299, -.041), (.303, -.028), (.311, -.038), (.307, -.022)]}
    cases = {split: [dict(id=f'{split}-{i:02}-goal{g}', start_xy=list(start), goal_xy=list(target), seed=261000+i*2+g)
                     for i, start in enumerate(points) for g, target in enumerate(targets)] for split, points in starts.items()}
    all_starts = [p for rows in starts.values() for p in rows]
    assert len(set(all_starts)) == len(all_starts)
    root.mkdir(parents=True)
    write(root/'protocol.json', dict(schema=SCHEMA, cases=cases, variants=list(VARIANTS),
        data_grouping='Complete episodes grouped by exact initial XY; both destinations share the same split.',
        training_selection='All teacher attempts retained; only independently successful complete training episodes supply labels.',
        development='Checkpoint choices and diagnostics use development cases only; all development trials remain.',
        confirmation='Model and protocol hash frozen before these rollouts. No retries or result-based tuning.',
        timeout_s=20, control_dt_s=.02, timestep_s=.001, mass_kg=.04,
        target_novelty=False, start_novelty=True, physical_robot_ready=False, limitations=LIMITS))
    return read(root/'protocol.json')


def config(case):
    from .piper_task import PiperTaskConfig
    return replace(PiperTaskConfig(), object_position_m=(*case['start_xy'], .082), target_position_m=(*case['goal_xy'], .08))


def collect(root, assets):
    from .piper_task import run_episode
    from .piper_imitation import training_sample
    from .piper_imitation_score import score_episode
    root, assets = Path(root), Path(assets)
    protocol = read(root/'protocol.json')
    from .robot_profiles import verify_assets
    provenance = verify_assets('piper', assets)
    if (root/'training'/'samples.npz').exists(): raise ValueError('Training dataset already collected.')
    samples, groups, rows = [], [], []
    for index, case in enumerate(protocol['cases']['train']):
        path = root/'demonstrations'/(case['id']+'.json.gz')
        if path.exists(): episode = load_episode(path)
        else:
            print(json.dumps(dict(stage='demonstration', case=case['id'])), flush=True)
            episode = run_episode(assets, config(case), seed=case['seed'])
            save_episode(path, episode)
        if (episode.get('seed') != case['seed'] or episode.get('config') != json.loads(json.dumps(asdict(config(case))))
                or episode.get('control') != 'placement' or episode.get('policy_mode') != 'baseline'
                or episode.get('asset_provenance') != provenance
                or episode.get('implementation_sha256') != {n:sha(Path(__file__).with_name(n)) for n in TEACHER_FILES}):
            raise ValueError('Cached teacher identity or runtime differs from this protocol: '+case['id'])
        score = score_episode(episode, case['goal_xy'])
        usable = score['success'] and len(episode['actions']) == 1000
        rows.append(dict(**case, episode=str(path.relative_to(root)), sha256=sha(path), score=score, used_for_training=usable))
        if usable:
            groups.append(case['id'])
            for obs, action in zip(episode['trajectory'][:-1], episode['actions']):
                command = {k: action[k] for k in ('joint_targets_rad','gripper_position_m')}
                samples.append(training_sample(obs, command, case['id']))
        print(json.dumps(dict(stage='demonstration_scored', case=case['id'], success=score['success'], included=usable)), flush=True)
    if not samples: raise ValueError('No successful complete teacher demonstrations; collection receipts retained.')
    training = root/'training'; training.mkdir(exist_ok=True)
    ids = [groups.index(row['case_id']) for row in samples]
    np.savez_compressed(training/'samples.npz', features=np.asarray([r['features'] for r in samples], dtype=np.float32),
                        targets=np.asarray([r['targets'] for r in samples], dtype=np.float32), group=np.asarray(ids, dtype=np.int32))
    write(training/'dataset.json', dict(schema=SCHEMA, protocol_sha256=sha(root/'protocol.json'),
        rows=rows, scene_ids=groups, sample_count=len(samples), samples_sha256=sha(training/'samples.npz'),
        feature_source='causal pre-action observations only; no phase/outcome labels in input whitelist',
        supervision_origin='synthetic_authored_controller', physical_robot_ready=False))
    return read(training/'dataset.json')


def dataset(root, shuffle=False):
    root = Path(root); receipt = read(root/'training'/'dataset.json')
    if sha(root/'training'/'samples.npz') != receipt['samples_sha256']: raise ValueError('Training data changed.')
    with np.load(root/'training'/'samples.npz', allow_pickle=False) as data:
        x, y, group = data['features'], data['targets'], data['group']
    if shuffle: y = y[np.random.default_rng(19217).permutation(len(y))]
    return dict(scene_ids=receipt['scene_ids'], samples=[dict(features=a.tolist(), targets=b.tolist(), case_id=receipt['scene_ids'][int(i)])
                                                       for a, b, i in zip(x,y,group)])


def fit(root, name, epochs=160, hidden=64, shuffle=False, absolute=False, primitive=False, high_resolution=False):
    from .piper_imitation import train, save_model
    root = Path(root)
    if not name.isalnum() or len(name)>32: raise ValueError('Use a short alphanumeric development model name.')
    target = root/'development_models'/name
    if target.exists(): raise ValueError('Model name already used; preserve previous development attempts.')
    target.mkdir(parents=True)
    began = time.perf_counter()
    if primitive:
        from .piper_motor_primitive import train as fit_primitive
        model=fit_primitive(dataset(root),shuffled=shuffle,high_resolution=high_resolution)
    else:
        method = dict(action_representation='absolute',label_shuffle_seed=19217 if shuffle else None) if absolute else {}
        model = train(dataset(root, shuffle and not absolute), hidden_units=hidden, epochs=epochs, batch_size=512, seed=17, **method)
    save_model(model, target/'model.json')
    write(target/'receipt.json', dict(name=name, model_sha256=sha(target/'model.json'), protocol_sha256=sha(root/'protocol.json'),
        training_dataset_sha256=sha(root/'training'/'dataset.json'), shuffled_labels=shuffle, absolute_targets=absolute, primitive=primitive, high_resolution=high_resolution,
        training_seconds=time.perf_counter()-began, training=model['training'], model_bytes=(target/'model.json').stat().st_size))
    return read(target/'receipt.json')


def rollout(assets, case, model_path, variant='learned'):
    from .piper_task import PiperTaskEnv, task_contract
    from .piper_imitation import LearnedPiperPolicy, load_model
    from . import reference_sim as sim
    c = config(case); env = PiperTaskEnv(assets, c)
    result = dict(schema='dvidia.piper-learned-episode/v1', config=asdict(c), profile='piper', seed=case['seed'],
        valid=False, success=False, reason='not_started', error=None, trajectory=[], actions=[], metrics={},
        policy_kind=variant, policy_learned=variant in ('learned','shuffled_labels','open_jaw'),
        physical_robot_ready=False, observation_origin='privileged_native_simulator_state',
        task_contract=task_contract(), model_sha256=sha(model_path) if model_path else None,
        timing={}, simulated_seconds=0.)
    began = time.perf_counter(); inference, steps = [], []
    try:
        policy = LearnedPiperPolicy(load_model(model_path)) if model_path else None
        obs, info = env.reset(case['seed']); initial = list(obs['joint_position_rad']); result['trajectory'].append(obs)
        result.update(initial_object_position_m=env.initial_object_position.tolist(),
            compiled_model_sha256=sim.model_digest(env.model),
            native_servo_gains=env.model.actuator_gainprm[:,0].tolist(),
            native_servo_bias=env.model.actuator_biasprm[:,:3].tolist(),
            native_force_limits=env.model.actuator_forcerange.tolist(),
            native_control_limits=env.model.actuator_ctrlrange.tolist(),
            native_model_settings=dict(gravity_m_s2=env.model.opt.gravity.tolist(),body_gravcomp=env.model.body_gravcomp.tolist()),
            implementation_sha256=runtime_hashes())
        from .robot_profiles import verify_assets
        result['asset_provenance'] = verify_assets('piper',assets)
        mj, _ = sim._native_imports()
        result['platform'] = dict(system=platform.system(), machine=platform.machine(), python=platform.python_version(), numpy=np.__version__, mujoco=mj.__version__)
        for _ in range(1000):
            start = time.perf_counter()
            if variant == 'hold': action = dict(joint_targets_rad=initial, gripper_position_m=.035)
            else:
                p = time.perf_counter(); action = policy(obs); inference.append(time.perf_counter()-p)
                if variant == 'open_jaw': action['gripper_position_m'] = .035
            record = dict(t_seconds=obs['t_seconds'], **action)
            result['actions'].append(record)
            obs.update(issued_joint_targets_rad=list(action['joint_targets_rad']),
                       issued_gripper_position_m=action['gripper_position_m'], command=record)
            obs, info = env.step(action); steps.append(time.perf_counter()-start)
            if obs is not None: result['trajectory'].append(obs)
            if env.done: break
        result.update(valid=info['valid'], success=info['success'], reason=info['reason'], simulated_seconds=info['simulation_time'],
                      metrics={**info, 'contact_pair_native_ticks':env.contact_pair_native_ticks})
        if policy is not None: result['policy_diagnostics'] = policy.diagnostics()
    except Exception as exc:
        result.update(reason='exception', error=dict(type=type(exc).__name__, message=str(exc)[:1000]))
    def summary(values):
        return dict(samples=len(values), p50_s=float(np.percentile(values,50)), p95_s=float(np.percentile(values,95)), p99_s=float(np.percentile(values,99)), maximum_s=max(values)) if values else None
    result['timing'] = dict(wall_seconds=time.perf_counter()-began, inference=summary(inference), control_step=summary(steps),
        missed_20ms_boundaries=sum(t>.02 for t in steps), measurement_scope='CPU compute only, excludes sensing, transport and real actuation')
    result['timing'].update(inference_samples_s=inference, control_step_samples_s=steps,
        process_peak_rss_bytes=peak_rss())
    return result


def evaluate_development(root, assets, name):
    from .piper_imitation_score import score_episode
    root = Path(root); model = root/'development_models'/name/'model.json'
    if (root/'development_models'/name/'evaluation.json').exists(): raise ValueError('Development evaluation already exists.')
    rows = []; model_digest = sha(model); source_hashes = runtime_hashes()
    for case in read(root/'protocol.json')['cases']['development']:
        print(json.dumps(dict(stage='development', model=name, case=case['id'])), flush=True)
        episode = rollout(assets, case, model)
        if sha(model)!=model_digest or runtime_hashes()!=source_hashes: raise ValueError('Runtime/model changed during development.')
        path = root/'development_models'/name/'attempts'/(case['id']+'.json.gz'); save_episode(path, episode)
        rows.append(dict(**case, score=score_episode(episode,case['goal_xy']), episode=str(path.relative_to(root)), sha256=sha(path), timing=episode['timing']))
    write(root/'development_models'/name/'evaluation.json', dict(model_sha256=sha(model), runtime_sha256=source_hashes, rows=rows,
        scheduled=len(rows), successes=sum(r['score']['success'] for r in rows)))
    return read(root/'development_models'/name/'evaluation.json')


def freeze(root, name, shuffled_name):
    from .piper_imitation_score import score_episode
    root = Path(root)
    if (root/'frozen.json').exists(): raise ValueError('Study already frozen.')
    receipt = read(root/'development_models'/name/'evaluation.json')
    selected = root/'development_models'/name/'model.json'; shuffled = root/'development_models'/shuffled_name/'model.json'
    selected_receipt, shuffled_receipt = read(selected.parent/'receipt.json'), read(shuffled.parent/'receipt.json')
    dataset_digest, protocol_digest = sha(root/'training'/'dataset.json'), sha(root/'protocol.json')
    training_data = read(root/'training'/'dataset.json')
    if sha(root/'training'/'samples.npz')!=training_data['samples_sha256']: raise ValueError('Training arrays changed before freeze.')
    for row in training_data['rows']:
        if sha(root/row['episode'])!=row['sha256']: raise ValueError('Teacher demonstration changed before freeze.')
    for model, training_receipt in ((selected,selected_receipt),(shuffled,shuffled_receipt)):
        if (training_receipt['model_sha256']!=sha(model) or training_receipt['training_dataset_sha256']!=dataset_digest
                or training_receipt['protocol_sha256']!=protocol_digest): raise ValueError('Training model/data/protocol binding changed.')
    if selected_receipt['shuffled_labels'] or not shuffled_receipt['shuffled_labels']: raise ValueError('Require a genuine student and separately trained shuffled control.')
    selected_model, shuffled_model=read(selected),read(shuffled)
    if selected_model['schema']!=shuffled_model['schema']: raise ValueError('Control model family differs.')
    from .piper_imitation import digest
    expected_source=digest(dataset(root))
    for candidate,candidate_receipt in ((selected_model,selected_receipt),(shuffled_model,shuffled_receipt)):
        if candidate['train_cases']!=training_data['scene_ids']:
            raise ValueError('Model train roster differs from the collected training split.')
        if candidate_receipt.get('primitive',False): source=candidate['source_dataset_sha256']
        elif candidate_receipt.get('absolute_targets',False): source=candidate['training']['source_dataset_sha256']
        else: source=candidate['dataset_sha256']
        expected=(digest(dataset(root,True)) if candidate_receipt['shuffled_labels'] and not candidate_receipt.get('primitive',False)
                  and not candidate_receipt.get('absolute_targets',False) else expected_source)
        if source!=expected: raise ValueError('Model source arrays differ from the collected training split.')
    if selected_receipt.get('primitive',False):
        fields=('algorithm','sample_count','scene_count','spatial_ridge','time_ridge')
        architecture=('feature_contract','reactive_contact_feedback','known_goals_xy_m','start_bounds_xy_m',
                      'spatial_terms','time_centers_s','time_width_s','joint_limits_rad','joint_delta_limit_rad','jaw_limit_m','output_encoding')
        if not shuffled_receipt.get('primitive',False) or any(selected_model[k]!=shuffled_model[k] for k in architecture):
            raise ValueError('Control primitive architecture differs.')
        if (selected_model['source_dataset_sha256']!=shuffled_model['source_dataset_sha256']
                or selected_model['supervision_sha256']==shuffled_model['supervision_sha256']
                or selected_model['training']['shuffled_labels'] or not shuffled_model['training']['shuffled_labels']
                or shuffled_model['training']['label_shuffle_seed']!=19217):
            raise ValueError('Primitive control must permute the same complete action labels.')
    else:
        fields=('algorithm','sample_count','epochs','batch_size','learning_rate','seed')
        if selected_model['hidden_units']!=shuffled_model['hidden_units']: raise ValueError('Control architecture differs.')
    for field in fields:
        if selected_receipt['training'][field]!=shuffled_receipt['training'][field]: raise ValueError('Control training budget differs.')
    if selected_receipt.get('absolute_targets',False)!=shuffled_receipt.get('absolute_targets',False):
        raise ValueError('Control action representation differs.')
    if selected_receipt.get('absolute_targets',False) and (selected_receipt['training']['label_shuffle_seed'] is not None
            or shuffled_receipt['training']['label_shuffle_seed']!=19217):
        raise ValueError('Require complete absolute target permutation for the V2 control.')
    if receipt['model_sha256']!=sha(selected) or receipt['runtime_sha256']!=runtime_hashes(): raise ValueError('Development model/runtime binding changed.')
    planned = read(root/'protocol.json')['cases']['development']
    rows = receipt.get('rows',[])
    if (len(rows)!=len(planned) or receipt.get('scheduled')!=len(planned)
            or receipt.get('successes')!=sum(r.get('score',{}).get('success') is True for r in rows)):
        raise ValueError('Development counts differ from the complete schedule.')
    for case,row in zip(planned,rows):
        expected='development_models/'+name+'/attempts/'+case['id']+'.json.gz'
        if any(row.get(k)!=case[k] for k in ('id','start_xy','goal_xy','seed')) or row.get('episode')!=expected:
            raise ValueError('Development attempt identity differs from its schedule.')
        episode=load_episode(root/expected)
        if (row.get('sha256')!=sha(root/expected) or row.get('score')!=score_episode(episode,case['goal_xy'])
                or episode.get('model_sha256')!=sha(selected) or episode.get('implementation_sha256')!=runtime_hashes()):
            raise ValueError('Development tape/model/runtime or recomputed score changed.')
    (root/'model').mkdir()
    shutil.copyfile(selected,root/'model'/'model.json'); shutil.copyfile(shuffled,root/'model'/'shuffled-model.json')
    write(root/'frozen.json', dict(schema=SCHEMA, model_sha256=sha(selected), shuffled_model_sha256=sha(shuffled),
        dataset_sha256=sha(root/'training'/'dataset.json'), samples_sha256=sha(root/'training'/'samples.npz'), protocol_sha256=sha(root/'protocol.json'),
        selected_name=name, development_successes=receipt['successes'], development_scheduled=receipt['scheduled'],
        selected_model_schema=selected_model['schema'], selected_feature_contract=selected_model['feature_contract'],
        evaluated_contact_recovery=False,
        development_evaluation_sha256=sha(root/'development_models'/name/'evaluation.json'),
        runtime_sha256=runtime_hashes(),
        frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), confirmation_opened=False))
    return read(root/'frozen.json')


def confirm(root, assets, resume=False):
    from .piper_task import run_episode
    from .piper_imitation_score import score_episode
    root = Path(root); frozen = read(root/'frozen.json'); protocol = read(root/'protocol.json')
    for name, key in (('model/model.json','model_sha256'),('model/shuffled-model.json','shuffled_model_sha256'),('training/dataset.json','dataset_sha256'),('training/samples.npz','samples_sha256'),('protocol.json','protocol_sha256')):
        if sha(root/name)!=frozen[key]: raise ValueError('Frozen experiment changed: '+name)
    if runtime_hashes()!=frozen['runtime_sha256']: raise ValueError('Frozen runtime changed before confirmation.')
    schedule = [dict(id=c['id'],variant=v,episode='attempts/'+c['id']+'-'+v+'.json.gz') for c in protocol['cases']['confirmation'] for v in VARIANTS]
    opened = root/'confirmation-opened.json'
    if opened.exists():
        if not resume or read(opened)['frozen_sha256']!=sha(root/'frozen.json') or read(opened)['planned_attempts']!=schedule:
            raise ValueError('Confirmation already opened. Use --resume with the same frozen protocol; no completed trial is rerun.')
    else:
        if resume: raise ValueError('No opened confirmation to resume.')
        write(opened,dict(frozen_sha256=sha(root/'frozen.json'),planned_attempts=schedule,scheduled=len(schedule),opened_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if (root/'report.json').exists(): raise ValueError('Confirmation already complete; inspect its immutable report.')
    rows = []
    for case in protocol['cases']['confirmation']:
        for variant in VARIANTS:
            if runtime_hashes()!=frozen['runtime_sha256']: raise ValueError('Frozen runtime changed during confirmation.')
            print(json.dumps(dict(stage='confirmation', variant=variant, case=case['id'])), flush=True)
            path = root/'attempts'/(case['id']+'-'+variant+'.json.gz')
            score_path = root/'attempt-scores'/(case['id']+'-'+variant+'.json')
            started = root/'attempt-starts'/(case['id']+'-'+variant+'.json')
            if (path.exists() or score_path.exists()) and (not started.exists() or read(started).get('frozen_sha256')!=sha(root/'frozen.json')):
                raise ValueError('Existing confirmation tape has no matching frozen start marker.')
            if score_path.exists():
                row = read(score_path)
                if (any(row.get(k)!=case[k] for k in ('id','start_xy','goal_xy','seed'))
                        or row.get('variant')!=variant or row.get('episode')!=str(path.relative_to(root))
                        or not started.exists() or read(started).get('frozen_sha256')!=sha(root/'frozen.json')
                        or row.get('sha256')!=sha(path) or row.get('score')!=score_episode(load_episode(path),case['goal_xy'])):
                    raise ValueError('Recorded confirmation identity or score changed.')
                rows.append(row); continue
            if path.exists(): episode = load_episode(path)  # Recover a finished tape; never reexecute it.
            elif started.exists():
                episode = dict(valid=False,success=False,reason='interrupted',error={'type':'interrupted','message':'Started attempt interrupted; retained as failure without retry.'},
                               config=asdict(config(case)),actions=[],trajectory=[],metrics={},timing={})
                save_episode(path,episode)
            else:
                write(started,dict(frozen_sha256=sha(root/'frozen.json'),started_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                if variant=='teacher': episode = run_episode(assets, config(case), seed=case['seed'])
                else: episode = rollout(assets,case,root/'model'/('shuffled-model.json' if variant=='shuffled_labels' else 'model.json'),variant)
                save_episode(path,episode)
            if episode.get('reason')!='interrupted':
                model_key='shuffled_model_sha256' if variant=='shuffled_labels' else 'model_sha256'
                expected_runtime = ({n:sha(Path(__file__).with_name(n)) for n in TEACHER_FILES}
                                    if variant=='teacher' else frozen['runtime_sha256'])
                if (episode.get('config')!=json.loads(json.dumps(asdict(config(case))))
                        or episode.get('seed')!=case['seed'] or episode.get('implementation_sha256')!=expected_runtime
                        or (variant=='teacher' and (episode.get('control')!='placement' or episode.get('policy_mode')!='baseline'))
                        or (variant!='teacher' and (episode.get('policy_kind')!=variant or episode.get('model_sha256')!=frozen[model_key]))):
                    raise ValueError('Confirmation tape identity, model or runtime differs from the schedule.')
            rows.append(dict(**case,variant=variant,score=score_episode(episode,case['goal_xy']),episode=str(path.relative_to(root)),
                             sha256=sha(path),error=episode.get('error'),timing=episode.get('timing',{})))
            write(score_path,rows[-1])
    write(root/'report.json', dict(schema=SCHEMA, status='completed', protocol_sha256=sha(root/'protocol.json'), frozen_sha256=sha(root/'frozen.json'),
        attempts=rows, variants={v:dict(scheduled=sum(r['variant']==v for r in rows),successes=sum(r['variant']==v and r['score']['success'] for r in rows)) for v in VARIANTS},
        training=read(root/'training'/'dataset.json'), selected_training=read(root/'development_models'/frozen['selected_name']/'receipt.json'),
        physical_robot_ready=False,motor_policy_learned=True,selected_model_schema=frozen['selected_model_schema'],
        evaluated_contact_recovery=False,limitations=LIMITS+([read(root/'model/model.json')['scope']] if 'motor-primitive' in frozen['selected_model_schema'] else [])))
    return read(root/'report.json')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','collect','train','development','freeze','confirm'))
    p.add_argument('--run',type=Path,required=True); p.add_argument('--assets',type=Path); p.add_argument('--name',default='mlp160')
    p.add_argument('--shuffled-name',default='shuffled160'); p.add_argument('--epochs',type=int,default=160); p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--shuffle',action='store_true'); p.add_argument('--absolute',action='store_true'); p.add_argument('--primitive',action='store_true'); p.add_argument('--high-resolution',action='store_true'); p.add_argument('--resume',action='store_true'); a=p.parse_args(argv)
    if a.absolute and a.primitive: p.error('Choose one action model family.')
    if a.high_resolution and not a.primitive: p.error('--high-resolution requires --primitive')
    if a.command in ('collect','development','confirm') and a.assets is None: p.error('--assets is required')
    result = (prepare(a.run) if a.command=='prepare' else collect(a.run,a.assets) if a.command=='collect'
        else fit(a.run,a.name,a.epochs,a.hidden,a.shuffle,a.absolute,a.primitive,a.high_resolution) if a.command=='train' else evaluate_development(a.run,a.assets,a.name) if a.command=='development'
        else freeze(a.run,a.name,a.shuffled_name) if a.command=='freeze' else confirm(a.run,a.assets,a.resume))
    print(json.dumps({k:v for k,v in result.items() if k in ('schema','variants','sample_count','successes','scheduled','training','model_bytes','training_seconds')},allow_nan=False),flush=True)


if __name__=='__main__': main()
