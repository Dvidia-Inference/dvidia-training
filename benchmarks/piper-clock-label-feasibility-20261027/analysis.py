"""TRAINING-only, offline diagnostic; no fitting, reset, rollout, or mj_step.

The analysis deliberately does not produce a usable training-label dataset.
All thresholds below are sensitivity probes, not prospective acceptance gates.
Run from repository root with PYTHONPATH=src and the existing native Python.
"""
from __future__ import annotations
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import mujoco as mj
from dvidia_training import piper_motor_primitive as primitive
from dvidia_training import piper_grip_adapter as grip
from dvidia_training import robot_profiles
from dvidia_training.piper_task import build_task_model


def hashed(path):
    content = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(content), sha256=sha256(content).hexdigest())


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(('min', 'p50', 'p90', 'p95', 'p99', 'max'), map(float, np.quantile(a, [0,.5,.9,.95,.99,1]))))


def distances(a, b, divisor=1.):
    return np.maximum(0., (a*a).sum(1)[:,None] + (b*b).sum(1)[None,:] - 2.*a@b.T)/divisor


def monotone(cost, max_steps=3):
    """Global offline DP: start at zero, hold/advance <=60ms per20ms row.

    No terminal progress is imposed. Lowest predecessor/index breaks ties.
    This uses future rows OFFLINE; it is not a causal inference procedure.
    """
    n,m=cost.shape
    previous=np.full(m,np.inf); previous[0]=cost[0,0]
    back=np.zeros((n,m),dtype=np.uint8)
    indices=np.arange(m)
    for i in range(1,n):
        candidates=[]
        # Larger offset means earlier predecessor and wins exact ties.
        for offset in range(max_steps,-1,-1):
            v=np.full(m,np.inf)
            if offset: v[offset:]=previous[:-offset]
            else: v=previous.copy()
            candidates.append(v)
        stacked=np.asarray(candidates)
        choices=np.argmin(stacked,axis=0)
        offsets=max_steps-choices
        back[i]=offsets
        previous=cost[i]+stacked[choices,indices]
    path=np.zeros(n,dtype=int);path[-1]=int(np.argmin(previous))
    for i in range(n-1,0,-1):path[i-1]=path[i]-int(back[i,path[i]])
    return path


def projection_summary(cost, tolerance, metrics):
    nearest=np.argmin(cost,axis=1); rows=np.arange(len(cost))
    best=np.sqrt(cost[rows,nearest])
    near=cost <= (best[:,None]+tolerance)**2
    first=np.argmax(near,axis=1)
    last=near.shape[1]-1-np.argmax(near[:,::-1],axis=1)
    span=(last-first)*.02
    path=monotone(cost)
    constrained=np.sqrt(cost[rows,path])
    delta=np.diff(nearest)*.02
    metrics.update(nearest_error=quantiles(best), physical_clock_error=quantiles(np.sqrt(np.diag(cost))),
        nearoptimal_progress_span_s=quantiles(span),
        ambiguous_ge_500ms_count=int(np.count_nonzero(span>=.5-1e-9)),
        nearest_backtrack_gt_20ms_count=int(np.count_nonzero(delta<-.0200001)),
        nearest_forward_jump_gt_60ms_count=int(np.count_nonzero(delta>.0600001)),
        nearest_biggest_backtrack_s=float(max(0.,-delta.min())),
        nearest_biggest_forward_jump_s=float(max(0.,delta.max())),
        bounded_monotonic_error=quantiles(constrained),
        bounded_monotonic_error_increase=quantiles(constrained-best),
        bounded_monotonic_terminal_reference_s=float(path[-1]*.02),
        bounded_monotonic_hold_fraction=float(np.mean(np.diff(path)==0)),
        bounded_monotonic_maxrate_fraction=float(np.mean(np.diff(path)==3)))
    return nearest,path,span,best,constrained


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--study',default='runs/piper-feedback-v02-20261027-study02')
    parser.add_argument('--assets',default='runs/clock-label-assets')
    args=parser.parse_args(); out=Path(args.output)
    if out.exists():raise ValueError('Refuse to overwrite diagnostic output.')
    study=Path(args.study);parent=study/'prior-studies/study01'
    protocol_path=parent/'protocol.json';protocol=json.loads(protocol_path.read_text())
    roster=protocol['cases']['train']
    if len(roster)!=64 or any(not c['id'].startswith('train-') for c in roster):raise ValueError('Training roster differs.')
    base_path=study/'base-model.json';base=primitive.validate_model(base_path.read_bytes())
    jaw_path=study/'candidates/jaw4/model.json'; jaw=grip.validate_model(jaw_path.read_bytes())
    asset_receipt=robot_profiles.verify_assets('piper',args.assets)
    model=build_task_model(args.assets);data=mj.MjData(model)
    tcp_id=mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,'piper_task_tcp')
    def fk(q):
        result=[]
        for row in q:
            data.qpos[:6]=row
            mj.mj_kinematics(model,data)
            result.append(data.site_xpos[tcp_id].copy())
        return np.asarray(result)
    times=np.arange(1001)*.02
    basis=primitive._time_basis(times,True)
    weights=np.asarray(base['weights']);limits=np.asarray(primitive.JOINT_LIMITS)
    rows=[]; inputs=[]; pooled={k:[] for k in ('tcp_best','tcp_dp','tcp_clock','tcp_span','q_best','q_dp','q_clock','q_span','label_delta','q_delta','tcp_delta')}
    jaw_sensitivity={str(offset):dict(rows=0,flipped=0) for offset in (.5,1.,2.)}
    maximum_fk_error=0.;start=time.monotonic()
    for index,case in enumerate(roster):
        tape_path=parent/'demonstrations'/case['id']/'episode.json.gz'
        inputs.append(hashed(tape_path))
        episode=json.loads(gzip.decompress(tape_path.read_bytes()))
        trajectory=episode['trajectory']; actions=episode['actions']
        if len(trajectory)!=1001 or len(actions)!=1000 or episode['case_id']!=case['id']:raise ValueError('Incomplete training tape.')
        actual_t=np.asarray([s['t_seconds'] for s in trajectory])
        if np.max(abs(actual_t-times))>1e-8:raise ValueError('Nonuniform training grid.')
        xy=np.asarray(trajectory[0]['object_position_m'][:2])
        goal=primitive._goal(trajectory[0]['target_position_m'])
        latent=basis@np.einsum('s,stj->tj',primitive._spatial(xy,True),weights[goal])
        reference_q=latent[:,:6]*3.
        nominal_jaw=.035*np.clip(latent[:,6],0.,1.)
        reference_tcp=fk(reference_q)
        measured_q=np.asarray([s['joint_position_rad'] for s in trajectory])
        measured_tcp=np.asarray([s['tcp_position_m'] for s in trajectory])
        # FK consistency validates static coordinates only, never dynamics.
        check=np.arange(0,1001,50)
        fk_error=np.linalg.norm(fk(measured_q[check])-measured_tcp[check],axis=1)
        maximum_fk_error=max(maximum_fk_error,float(fk_error.max()))
        qcost=distances(measured_q,reference_q,6.);tcost=distances(measured_tcp,reference_tcp)
        qstats={};tstats={}
        qnear,qdp,qspan,qbest,qerr=projection_summary(qcost,.01,qstats)
        tnear,tdp,tspan,tbest,terr=projection_summary(tcost,.001,tstats)
        label_delta=abs(qdp-tdp)*.02
        self_tcp=distances(reference_tcp,reference_tcp)
        self_q=distances(reference_q,reference_q,6.)
        separated=abs(np.arange(1001)[:,None]-np.arange(1001)[None,:])>=25
        repeated=separated&(self_tcp<=.001**2)&(self_q<=.01**2)
        slow=(np.linalg.norm(np.diff(reference_tcp,axis=0),axis=1)/.02<.001)&(np.sqrt(np.mean(np.diff(reference_q,axis=0)**2,axis=1))/.02<.01)
        command_q=np.asarray([s['joint_targets_rad'] for s in actions])
        command_cost=distances(command_q,reference_q,6.)
        command_nearest=np.argmin(command_cost,axis=1)
        bound_error=np.maximum(limits[:,0]+.02-reference_q,reference_q-(limits[:,1]-.02))
        row=dict(case_id=case['id'],condition=case['condition'],goal_index=goal,frames=1001,
            q_rms_rad=qstats,tcp_m=tstats,
            q_vs_tcp_monotonic_progress_difference_s=quantiles(label_delta),
            q_vs_tcp_difference_ge_500ms_count=int(np.count_nonzero(label_delta>=.5-1e-9)),
            reference_nonlocal_repeated_pose_fraction=float(np.mean(np.any(repeated,axis=1))),
            reference_stationary_interval_fraction=float(np.mean(slow)),
            raw_reference_limit_exceedance_rad=float(max(0.,bound_error.max())),
            teacher_command_nearest_q_error_rad=quantiles(np.sqrt(command_cost[np.arange(1000),command_nearest])),
            teacher_command_vs_measured_nearest_progress_difference_s=quantiles(abs(command_nearest-qnear[:1000])*.02))
        rows.append(row)
        for key,value in [('tcp_best',tbest),('tcp_dp',terr),('tcp_clock',np.sqrt(np.diag(tcost))),('tcp_span',tspan),('q_best',qbest),('q_dp',qerr),('q_clock',np.sqrt(np.diag(qcost))),('q_span',qspan),('label_delta',label_delta),('q_delta',np.diff(qnear)*.02),('tcp_delta',np.diff(tnear)*.02)]:pooled[key].append(value)
        encoder=grip.GripFeatures()
        for i,state in enumerate(trajectory[:-1]):
            feature=encoder.features(state,dict(gripper_position_m=float(nominal_jaw[i])))
            original=grip._probability(jaw['nodes'],feature)>.5
            for offset in (.5,1.,2.):
                if times[i]+offset>20.:continue
                # Input sensitivity only: same causal measurements, changed physical context.
                altered=feature.copy();altered[4]=(times[i]+offset)/20.
                altered[5]=float(np.interp(times[i]+offset,times,nominal_jaw))/.035
                counter=jaw_sensitivity[str(offset)];counter['rows']+=1
                counter['flipped']+=int((grip._probability(jaw['nodes'],altered)>.5)!=original)
        print(f'{index+1}/64 {case["id"]}',flush=True)
    pooled={k:np.concatenate(v) for k,v in pooled.items()}
    result=dict(schema='dvidia.piper-clock-label-feasibility-draft/v1',status='diagnostic_only_not_validated_labels',
        scope='Only 64 inherited synthetic TRAINING teachers; no development/confirmation trajectory reads; no fitting or native stepping.',
        decisions='Do not export progress labels or start clock training from this diagnostic.',
        method=dict(reference='Frozen raw six-joint reference at 20ms grid, FK TCP point, per-reset measured XY and fixed known goal.',
            parameterization='p = frozen reference seconds /20, including stationary pauses.',
            q_metric='six-joint RMS radians; TCP metric Euclidean metres, orientation omitted',
            ambiguity_tolerance=dict(q_rms_rad=.01,tcp_m=.001,nonlocal_separation_s=.5),
            nearest='Global pointwise argmin; earliest index breaks ties.',
            monotonic='Global offline DP, start reference0, no forced end, increments0..3 ticks per20ms, earliest predecessor ties. Uses future rows only offline.',
            thresholds='Diagnostic sensitivity probes chosen before running, not acceptance or qualification gates.',
            native_execution='build_task_model + mj_kinematics only; no reset/mj_step/mj_forward/rollout.',
            jaw_sensitivity='Same teacher causal grip features; perturb physical time+physical nominal jaw by +0.5/+1/+2s. Decision sensitivity only, no physical retiming claim.'),
        input_bindings=[hashed(Path(__file__)),hashed(Path('runs/piper-feedback-v03-protocol-draft.md')),hashed(base_path),hashed(jaw_path),hashed(protocol_path),hashed(study/'training-parent.json'),hashed(robot_profiles.LOCK_PATH)]+inputs,
        asset_receipt=asset_receipt,kinematics_teacher_tcp_max_error_m=maximum_fk_error,
        totals=dict(episodes=64,frames=64064,training_action_rows=64000),
        pooled_quantiles={k:quantiles(v) for k,v in pooled.items()},
        pooled_counts=dict(tcp_ambiguous_ge_500ms=int(np.count_nonzero(pooled['tcp_span']>=.5-1e-9)),q_ambiguous_ge_500ms=int(np.count_nonzero(pooled['q_span']>=.5-1e-9)),
            q_vs_tcp_progress_disagree_ge_500ms=int(np.count_nonzero(pooled['label_delta']>=.5-1e-9)),
            tcp_backtrack_gt_20ms=int(np.count_nonzero(pooled['tcp_delta']<-.0200001)),tcp_jump_gt_60ms=int(np.count_nonzero(pooled['tcp_delta']>.0600001)),
            q_backtrack_gt_20ms=int(np.count_nonzero(pooled['q_delta']<-.0200001)),q_jump_gt_60ms=int(np.count_nonzero(pooled['q_delta']>.0600001))),
        jaw_context_sensitivity={k:dict(v,flip_fraction=v['flipped']/v['rows']) for k,v in jaw_sensitivity.items()},
        per_episode=rows,wall_seconds=time.monotonic()-start,physical_robot_ready=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
