"""Frozen, paired evaluation of a bounded acquisition supervisor.

The v0.3 runtime and scorer are imported unchanged. This is an authored timing
supervisor around a frozen policy, not a new learned manipulation model.
Use ``python -m dvidia_training.piper_grip_wait_v04_benchmark --help``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import math
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np

from . import piper_grip_clock as clock
from . import piper_grip_clock_benchmark as old
from . import piper_grip_clock_cli as old_cli
from . import piper_motor_primitive as primitive
from .piper_imitation_benchmark import read, write, sha, save_episode, load_episode, peak_rss

SCHEMA = 'dvidia.piper-grip-wait-study/v0.4'
VARIANTS = ('guarded', 'clock', 'fixed', 'teacher')
CANDIDATES = {'wait200': .20, 'wait360': .36, 'wait500': .50}
STUDY_SEED = 284031


def sources():
    names = (*old.SOURCE_FILES, 'piper_imitation_benchmark.py', 'piper_grip_wait_v04.py', Path(__file__).name)
    return {name: sha(Path(__file__).with_name(name)) for name in dict.fromkeys(names)}


def make_cases():
    excluded = old_cli.historical_starts()
    excluded += [c['start_xy'] for split in old_cli.make_cases().values() for c in split]
    rng = np.random.default_rng(STUDY_SEED)
    accepted, result = [], {}
    for split, count in (('development', 4), ('confirmation', 32)):
        points = []
        for _ in range(100000):
            point = np.round(rng.uniform((.284, -.045), (.314, -.015)), 6).tolist()
            if any(math.dist(point, other) < .00075-1e-12 for other in excluded+accepted):
                continue
            points.append(point); accepted.append(point)
            if len(points) == count: break
        if len(points) != count: raise ValueError('Finite fresh roster generation failed.')
        result[split] = [dict(id=f'{split}-xy{i:02}-g{g}-{condition}', start_xy=point,
            goal_xy=list(goal), seed=284100+len(result)*100+i*2+g, condition=condition)
            for i, point in enumerate(points) for g, goal in enumerate(primitive.GOALS)
            for condition in ('nominal', 'post_grip')]
    return result


def prepare(output, previous):
    from .piper_grip_wait_v04 import GripWaitConfig
    root, previous = Path(output), Path(previous)
    if root.exists(): raise ValueError('Use a fresh study directory.')
    prior = read(previous/'report.json')
    if (prior.get('protocol_sha256') != sha(previous/'protocol.json')
            or prior.get('frozen_sha256') != sha(previous/'frozen.json')
            or read(previous/'frozen.json')['runtime_sha256'] != old.runtime_hashes()):
        raise ValueError('The prior frozen runtime/study differs.')
    documents = {name: (previous/name).read_bytes() for name in (
        'base-model.json', 'jaw-model.json', 'selected.json')}
    old_frozen=read(previous/'frozen.json')
    for name,key in (('base-model.json','base_model_raw_sha256'),('jaw-model.json','jaw_model_raw_sha256'),('selected.json','selected_sha256')):
        if sha256(documents[name]).hexdigest()!=old_frozen[key]:
            raise ValueError('Prior policy bytes differ from the frozen v0.3 selection.')
    old._documents(read(previous/'base-model.json'),read(previous/'jaw-model.json'),read(previous/'selected.json'),'clock')
    configs = {name: asdict(GripWaitConfig(maximum_wait_s=seconds)) for name, seconds in CANDIDATES.items()}
    protocol = dict(schema=SCHEMA, cases=make_cases(), candidates=configs,
        source_sha256=sources(), input_sha256={n: sha256(b).hexdigest() for n,b in documents.items()},
        previous_protocol_sha256=prior['protocol_sha256'], previous_frozen_sha256=prior['frozen_sha256'],
        previous_report_sha256=sha(previous/'report.json'), variants=list(VARIANTS), workers=4,
        selection='Development strict successes descending, aborted ascending, critical-contact attempts ascending, total acquisition wait ticks ascending, wait budget ascending. All 16 cases retained per candidate.',
        confirmation=dict(groups=32, cases_per_variant=128, scheduled=512,
            primary_contrasts=['guarded-minus-clock', 'guarded-minus-fixed'],
            interval='Paired whole-XY differences, four goal/condition cases per group; two-sided marginal97.5% t intervals df31, Bonferroni simultaneous95% for two contrasts.',
            no_optional_stopping=True, no_confirmation_retuning=True),
        success='Unchanged native strict scorer AND no supervisor abort. Raw strict score also retained. All scheduled attempts remain in denominator.',
        runtime_scope='Authored acquisition supervisor; frozen learned predictions advisory only. 20ms observations, 1ms native MuJoCo. Abort preserves measured arm pose with native caps for remainder of20s evidence.',
        limitations=['Privileged native simulator contacts/state, not camera or human-video learning.',
            'One rigid40g block, two known goals and narrow30mm-square start envelope, not arbitrary skills or arms.',
            'Supervisor checks sampled readiness; one-way latch does not certify continuous retention or physical safety.',
            'Support-loss guard can false-abort on contact bounce; one control interval of sensing delay remains.',
            'Timing includes concurrent simulator load; excludes camera, rendering, network and physical actuation.'],
        qualified_robot_skill=False, physical_robot_ready=False)
    root.mkdir(parents=True)
    for name, raw in documents.items(): (root/name).write_bytes(raw)
    for name in protocol['source_sha256']:
        path=root/'source'/name; path.parent.mkdir(exist_ok=True)
        path.write_bytes(Path(__file__).with_name(name).read_bytes())
    write(root/'protocol.json', protocol)
    write(root/'prepared.json',dict(schema=SCHEMA,protocol_sha256=sha(root/'protocol.json'),source_sha256=protocol['source_sha256']))
    return protocol


def checked(root):
    root=Path(root); protocol=read(root/'protocol.json'); prepared=read(root/'prepared.json')
    if (protocol.get('schema') != SCHEMA or protocol['source_sha256'] != sources()
            or protocol['cases'] != make_cases() or prepared != dict(schema=SCHEMA,
                protocol_sha256=sha(root/'protocol.json'),source_sha256=sources())):
        raise ValueError('Study source/protocol/roster changed.')
    for name, expected in protocol['input_sha256'].items():
        if sha(root/name) != expected: raise ValueError('Frozen policy input changed.')
    for name, expected in protocol['source_sha256'].items():
        if sha(root/'source'/name) != expected: raise ValueError('Bound source snapshot changed.')
    return protocol


def rollout(assets, case, base, jaw, full, variant, config):
    from .piper_grip_wait_v04 import GuardedLearnedClock, GripWaitConfig
    from .piper_grip_clock_env import PostGripPiperTaskEnv
    from .robot_profiles import verify_assets
    from . import reference_sim as sim
    case=old._case(case)
    c=old.config(case)
    env=PostGripPiperTaskEnv(assets,c,interruption=case['condition']=='post_grip')
    teacher, intervention=old._options()
    result=dict(schema=old.EPISODE_SCHEMA, study_schema=SCHEMA, config=old._canonical(asdict(c)),
        case_id=case['id'], seed=case['seed'], condition=case['condition'], policy_kind=variant,
        base_model_sha256=old.digest(base), jaw_model_sha256=old.digest(jaw),
        model_sha256=old.digest(full) if variant in ('guarded','clock') else None,
        jaw_mode='coordinated_reference', teacher_config=asdict(teacher), post_grip_parameters=intervention,
        supervisor_config=config if variant=='guarded' else None,
        physical_robot_ready=False, observation_origin='privileged_native_simulator_state',
        trajectory=[], actions=[], valid=False,success=False,reason='not_started',error=None,
        simulated_seconds=0.,metrics={},implementation_sha256=sources())
    inference,steps=[],[]; began=time.perf_counter()
    try:
        obs,info=env.reset(case['seed']); mj,_=sim._native_imports()
        result.update(initial_object_position_m=env.initial_object_position.tolist(),
            compiled_model_sha256=sim.model_digest(env.model),asset_provenance=verify_assets('piper',assets),
            native_servo_gains=env.model.actuator_gainprm[:,0].tolist(),native_force_limits=env.model.actuator_forcerange.tolist(),
            platform=dict(system=platform.system(),machine=platform.machine(),python=platform.python_version(),numpy=np.__version__,mujoco=mj.__version__))
        if variant=='guarded': policy=GuardedLearnedClock(base,jaw,full,config=GripWaitConfig(**config))
        elif variant=='clock': policy=clock.LearnedClock(base,jaw,full)
        elif variant=='fixed': policy=clock.FixedClock(base,jaw)
        elif variant=='teacher': policy=clock.TeacherClock(base,jaw)
        else: raise ValueError('Unknown controller.')
        result['trajectory'].append(obs)
        for _ in range(1000):
            began_step=time.perf_counter(); began_policy=time.perf_counter(); action=policy(obs)
            inference.append(time.perf_counter()-began_policy)
            record=dict(t_seconds=obs['t_seconds'],**action,clock_decision=old._canonical(policy.last_decision))
            result['actions'].append(record)
            obs.update(issued_joint_targets_rad=list(action['joint_targets_rad']),issued_gripper_position_m=action['gripper_position_m'],command=record)
            obs,info=env.step(action); steps.append(time.perf_counter()-began_step)
            if obs is not None:
                obs.update(issued_joint_targets_rad=None,issued_gripper_position_m=None,command=None)
                result['trajectory'].append(obs)
            if env.done: break
        result.update(valid=info['valid'],success=info['success'],reason=info['reason'],
            simulated_seconds=info['simulation_time'], metrics={**info,'contact_pair_native_ticks':env.contact_pair_native_ticks},
            policy_diagnostics=policy.diagnostics())
    except Exception as error:
        result.update(valid=False,success=False,reason='exception',error=dict(type=type(error).__name__,message=str(error)[:500]))
    result['timing']=dict(wall_seconds=time.perf_counter()-began,inference=old.prior.prior._summary(inference),
        control_step=old.prior.prior._summary(steps),inference_samples_s=inference,control_step_samples_s=steps,
        missed_20ms_boundaries=sum(x>.02 for x in steps),process_peak_rss_bytes=peak_rss(),
        scope='Shared-host CPU simulator timing; no camera/transport/rendering/physical actuation.')
    return result,env.native_arrays(),env.native_metadata()


def execute(root, assets, case, variant, folder, config):
    root,folder=Path(root),Path(folder); protocol=checked(root)
    if variant not in VARIANTS: raise ValueError('Unknown controller.')
    phase=next((name for name,roster in protocol['cases'].items() if case in roster),None)
    if phase is None: raise ValueError('Case is outside the literal study roster.')
    if phase=='development':
        candidates=[name for name,value in protocol['candidates'].items() if value==config]
        if variant!='guarded' or len(candidates)!=1:
            raise ValueError('Development requires one declared guarded candidate.')
        expected_folder=root/phase/candidates[0]/case['id']
    else: expected_folder=root/phase/case['id']/variant
    if folder.resolve()!=expected_folder.resolve() or folder.is_symlink():
        raise ValueError('Attempt folder differs from the literal roster/candidate.')
    binding=dict(schema=SCHEMA,protocol_sha256=sha(root/'protocol.json'),runtime_sha256=sources(),
        case=case,variant=variant,config=config if variant=='guarded' else None,
        frozen_sha256=sha(root/'frozen.json') if phase=='confirmation' else None)
    if phase=='confirmation':
        frozen=read(root/'frozen.json')
        if (frozen['protocol_sha256'] != binding['protocol_sha256'] or frozen['source_sha256'] != sources()
                or config != protocol['candidates'][frozen['selected']]): raise ValueError('Confirmation selection changed.')
    if folder.exists() and any(folder.iterdir()):
        if not(folder/'receipt.json').is_file(): raise ValueError('Interrupted attempt retained; no automatic physics rerun.')
        receipt=read(folder/'receipt.json')
        names={'started.json','episode.json.gz','native.npz','native.json','score.json','post-grip.json'}
        if (set(p.name for p in folder.iterdir())!=names|{'receipt.json'} or set(receipt.get('files',{}))!=names
                or any(p.is_symlink() or not p.is_file() for p in folder.iterdir())
                or receipt['binding'] != binding or read(folder/'started.json')!=binding):
            raise ValueError('Attempt binding/roster differs.')
        for name,expected in receipt['files'].items():
            if sha(folder/name)!=expected: raise ValueError('Retained attempt changed.')
        episode=load_episode(folder/'episode.json.gz')
        teacher, intervention=old._options()
        expected=dict(schema=old.EPISODE_SCHEMA,study_schema=SCHEMA,case_id=case['id'],seed=case['seed'],
            condition=case['condition'],policy_kind=variant,config=old._canonical(asdict(old.config(case))),
            base_model_sha256=old.digest(read(root/'base-model.json')),jaw_model_sha256=old.digest(read(root/'jaw-model.json')),
            model_sha256=old.digest(read(root/'selected.json')) if variant in ('guarded','clock') else None,
            jaw_mode='coordinated_reference',supervisor_config=config if variant=='guarded' else None,
            teacher_config=asdict(teacher),post_grip_parameters=intervention,
            implementation_sha256=sources())
        if any(episode.get(k)!=v for k,v in expected.items()):raise ValueError('Retained episode metadata differs from its binding.')
        metadata=read(folder/'native.json')
        if metadata.get('post_grip_intervention',{}).get('parameters') != dict(interruption=case['condition']=='post_grip',**intervention):
            raise ValueError('Retained native intervention differs from the assigned condition.')
        score,analysis=old._score(episode,old._load_npz(folder/'native.npz'),metadata,case['goal_xy'])
        if score!=read(folder/'score.json') or analysis!=read(folder/'post-grip.json'):
            raise ValueError('Independent strict/intervention rescoring differs.')
        return summary(case,variant,score,episode,sha(folder/'receipt.json'))
    write(folder/'started.json',binding)
    episode,arrays,metadata=rollout(assets,case,read(root/'base-model.json'),read(root/'jaw-model.json'),read(root/'selected.json'),variant,config)
    if checked(root)!=protocol: raise ValueError('Source changed during an attempt.')
    save_episode(folder/'episode.json.gz',episode); old._save_npz(folder/'native.npz',arrays); write(folder/'native.json',metadata)
    score,analysis=old._score(episode,arrays,metadata,case['goal_xy'])
    write(folder/'score.json',score); write(folder/'post-grip.json',analysis)
    write(folder/'receipt.json',dict(schema=SCHEMA,binding=binding,
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    row=summary(case,variant,score,episode,sha(folder/'receipt.json'))
    print(json.dumps({k:row[k] for k in ('case_id','variant','valid','success','aborted')}),flush=True)
    return row


def summary(case,variant,score,episode,receipt_sha):
    d=episode.get('policy_diagnostics',{}); native=score.get('native_recomputed',{})
    abort=bool(d.get('aborted',False))
    return dict(case_id=case['id'],start_xy=case['start_xy'],goal_xy=case['goal_xy'],condition=case['condition'],variant=variant,
        valid=score['valid'],raw_strict_success=score['success'],success=bool(score['valid'] and score['success'] and not abort),
        aborted=abort,abort_reason=d.get('abort_reason'),acquisition_released=d.get('acquisition_released'),
        acquisition_fail=not native.get('grasp_seen',False),retention_fail=bool(native.get('retention_lost')),
        critical_contact=native.get('unexpected_contact_native_ticks',0)>0,post_grip=score.get('post_grip_analysis',{}),
        policy_diagnostics=d,receipt_sha256=receipt_sha,timing=episode.get('timing',{}),score_reason=score.get('reason'))


def _job(args): return execute(*args)


def run(root,assets,phase):
    root=Path(root); protocol=checked(root)
    with old_cli._locked(root):
        jobs=[]
        if phase=='development':
            if (root/'frozen.json').exists(): raise ValueError('Development is closed after freeze.')
            for name,config in protocol['candidates'].items():
                jobs += [(root,assets,case,'guarded',root/phase/name/case['id'],config) for case in protocol['cases'][phase]]
        elif phase=='confirmation':
            frozen=read(root/'frozen.json'); config=protocol['candidates'][frozen['selected']]
            jobs=[(root,assets,case,v,root/phase/case['id']/v,config) for case in protocol['cases'][phase] for v in VARIANTS]
        else: raise ValueError('Unknown phase.')
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows=list(pool.map(_job,jobs,chunksize=1))
        value=dict(schema=SCHEMA,phase=phase,protocol_sha256=sha(root/'protocol.json'),rows=rows)
        path=root/(phase+'-results.json')
        if path.exists():
            if read(path)!=value: raise ValueError('Existing phase results differ.')
        else: write(path,value)
        return value


def freeze(root):
    root=Path(root); protocol=checked(root)
    with old_cli._locked(root):
        if (root/'frozen.json').exists(): raise ValueError('Selection already frozen.')
        development=read(root/'development-results.json'); reconstructed=[];ranking=[]
        for name,config in protocol['candidates'].items():
            own=[]
            for case in protocol['cases']['development']:
                if not (root/'development'/name/case['id']/'receipt.json').is_file():
                    raise ValueError('Complete retained development evidence is required; selection never starts physics.')
                own.append(execute(root,None,case,'guarded',root/'development'/name/case['id'],config))
            if len(own)!=16 or any(not r['valid'] for r in own): raise ValueError('All16 development attempts must be valid before selection.')
            reconstructed.extend(own)
            ranking.append(dict(name=name,strict_successes=sum(r['success'] for r in own),aborts=sum(r['aborted'] for r in own),
                critical=sum(r['critical_contact'] for r in own),wait_ticks=sum(r['policy_diagnostics']['wait_ticks'] for r in own),budget=config['maximum_wait_s']))
        if development!=dict(schema=SCHEMA,phase='development',protocol_sha256=sha(root/'protocol.json'),rows=reconstructed):
            raise ValueError('Development aggregate differs from independently retained attempts.')
        ranking.sort(key=lambda r:(-r['strict_successes'],r['aborts'],r['critical'],r['wait_ticks'],r['budget']))
        value=dict(schema=SCHEMA,selected=ranking[0]['name'],ranking=ranking,protocol_sha256=sha(root/'protocol.json'),
            source_sha256=sources(),development_sha256=sha(root/'development-results.json'),
            development_evidence={str(p.relative_to(root)):sha(p) for p in sorted((root/'development').rglob('*')) if p.is_file()},
            selection_rule=protocol['selection'],confirmation_started=False)
        write(root/'frozen.json',value); return value


def report(root):
    root=Path(root); protocol=checked(root); frozen=read(root/'frozen.json')
    for name,expected in frozen['development_evidence'].items():
        if sha(root/name)!=expected: raise ValueError('Frozen development evidence changed.')
    rows=[];config=protocol['candidates'][frozen['selected']]
    for case in protocol['cases']['confirmation']:
        for variant in VARIANTS:
            if not (root/'confirmation'/case['id']/variant/'receipt.json').is_file():
                raise ValueError('Complete retained confirmation evidence is required; reporting never starts physics.')
            rows.append(execute(root,None,case,variant,root/'confirmation'/case['id']/variant,config))
    totals={}
    for v in VARIANTS:
        own=[r for r in rows if r['variant']==v]
        totals[v]={k:sum(bool(r[k]) for r in own) for k in ('valid','success','raw_strict_success','aborted','acquisition_fail','retention_fail','critical_contact')}
        totals[v]['scheduled']=len(own)
        totals[v]['by_condition']={c:sum(r['success'] for r in own if r['condition']==c) for c in ('nominal','post_grip')}
        totals[v]['timing']={target:old_cli._timing([x for r in own for x in r['timing'][field]]) for target,field in (('inference','inference_samples_s'),('full_step','control_step_samples_s'))}
        totals[v]['timing']['episode_wall']=old_cli._timing([r['timing']['wall_seconds'] for r in own])
    groups={}
    for r in rows: groups.setdefault(tuple(r['start_xy']),{v:[] for v in VARIANTS})[r['variant']].append(int(r['success']))
    if len(groups)!=32 or any(len(a)!=4 for g in groups.values() for a in g.values()): raise ValueError('Incomplete paired group roster.')
    contrasts=[old_cli._paired(list(groups.values()),'guarded',other) for other in ('clock','fixed')]
    value=dict(schema=SCHEMA,protocol_sha256=sha(root/'protocol.json'),frozen_sha256=sha(root/'frozen.json'),
        variants=totals,contrasts=contrasts,rows=rows,selected=frozen['selected'],development=frozen['ranking'],
        qualified_robot_skill=False,physical_robot_ready=False,limitations=protocol['limitations'])
    path=root/'report.json'
    if path.exists():
        if read(path)!=value:raise ValueError('Frozen report differs.')
    else:write(path,value)
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','development','freeze','confirmation','report'))
    parser.add_argument('--run',required=True);parser.add_argument('--previous');parser.add_argument('--assets')
    args=parser.parse_args(argv)
    if args.command=='prepare':
        if not args.previous:parser.error('prepare requires --previous')
        result=prepare(args.run,args.previous)
    elif args.command in ('development','confirmation'):
        if not args.assets:parser.error('native execution requires --assets')
        result=run(args.run,args.assets,args.command)
    elif args.command=='freeze': result=freeze(args.run)
    else:result=report(args.run)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256','development_evidence','cases')},indent=2))

if __name__=='__main__': main()
