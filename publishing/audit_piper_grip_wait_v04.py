"""Read-only reconstruction of v0.4 bounded acquisition traces and native scores.

No v0.4 runtime or lifecycle implementation is imported, and no simulator or
policy is constructed. Logged readiness and abort flags are observations to
check, not input to this independent state machine. Reuses only the previous
independent raw-basis/feature reconstruction and frozen numeric strict scorer.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np

try:
    from . import audit_piper_grip_clock_v03 as previous
except ImportError:
    import audit_piper_grip_clock_v03 as previous

SCHEMA = 'dvidia.piper-grip-wait-independent-audit/v1'
STUDY_SCHEMA = 'dvidia.piper-grip-wait-study/v0.4'
VARIANTS = ('guarded', 'clock', 'fixed', 'teacher')
NAMES = {'started.json', 'episode.json.gz', 'native.npz', 'native.json', 'score.json', 'post-grip.json'}
TOL = 2e-10
_read, _hash = previous._read, previous._hash


def _same(actual, expected):
    if isinstance(expected, (float, np.floating)):
        return (type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, float(expected), abs_tol=TOL, rel_tol=0.))
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


class AcquisitionReconstruction:
    """Independent sampled acquisition state, with total wait budget and abort."""
    def __init__(self, config, gate, release):
        self.config = config
        self.gate, self.release = gate, release
        self.history = deque()
        self.progress = self.wait_ticks = 0
        self.started = self.released = self.aborted = False
        self.started_at = self.released_at = self.abort_at = self.abort_reason = None
        self.abort_pose = None
        self.advisory_holds = self.overrides = 0

    def observe(self, state, raw_jaw, probability):
        t = state['t_seconds']
        local = previous.local_offset(state['object_position_m'], state['tcp_position_m'],
                                      state['tcp_quaternion_wxyz'])
        force = np.asarray([state['pad_normal_forces_n'][s] for s in ('left','right')])
        count = np.asarray([state['grasp_contact_counts'][s] for s in ('left','right')])
        supported = state['table_object_contact_count'] > 0
        bilateral = bool(np.all(count > 0) and np.all(force >= self.config['eligible_minimum_pad_force_n']))
        good = bool(np.all(count > 0) and np.all(force >= self.config['secure_minimum_pad_force_n']))
        if not good: self.history.clear()
        self.history.append((t, local, good))
        cutoff = t-self.config['secure_dwell_s']+1e-8
        while len(self.history)>1 and self.history[1][0] <= cutoff: self.history.popleft()
        span = t-self.history[0][0]
        drift = max(float(np.linalg.norm(local-row[1])) for row in self.history)
        near = bool(np.linalg.norm(local) <= self.config['maximum_object_tcp_distance_m'])
        secure = bool(near and good and all(row[2] for row in self.history)
            and span+1e-8 >= self.config['secure_dwell_s']
            and drift <= self.config['maximum_relative_drift_m'])
        in_window = self.gate <= self.progress < self.release
        closed = bool(raw_jaw < self.config['closed_jaw_reference_m'])
        eligible = bool(in_window and closed and bilateral and supported and near)
        learned_advance = bool(probability > .5)
        self.advisory_holds += int(not learned_advance)
        abort_reason = None
        if self.aborted:
            advance, reason = False, 'aborted_pose_hold'
        elif self.released:
            advance, reason = True, 'secure_grip_latched'
        elif in_window and secure and (eligible or self.started):
            self.released, self.released_at = True, t
            advance, reason = True, 'secure_grip_confirmed'
        elif in_window and closed and not supported:
            advance, reason, abort_reason = False, 'support_lost_before_secure', 'support_lost_before_secure'
        else:
            if eligible and not self.started:
                self.started, self.started_at = True, t
            if self.started:
                if not supported:
                    advance, reason, abort_reason = False, 'support_lost_before_secure', 'support_lost_before_secure'
                elif self.wait_ticks >= round(self.config['maximum_wait_s']/.02):
                    advance, reason, abort_reason = False, 'wait_budget_exhausted', 'wait_budget_exhausted'
                else:
                    self.wait_ticks += 1
                    advance, reason = False, 'supervised_acquisition_wait'
            else:
                advance = True
                reason = 'centering_without_bilateral_support' if in_window else 'outside_acquisition_window'
        if abort_reason:
            self.aborted, self.abort_reason, self.abort_at = True, abort_reason, t
            self.abort_pose = np.asarray(state['joint_position_rad'], dtype=float).copy()
        overridden = advance != learned_advance
        self.overrides += int(overridden)
        before = self.progress*.02
        self.progress = min(1000, self.progress+int(advance))
        return dict(decision_origin='authored_bounded_grip_supervisor',reason=reason,
            learner_advance=learned_advance,advance_probability=probability,learned_prediction_advisory=True,
            supervisor_override=overridden,in_grasp_window=in_window,raw_jaw_closed=closed,
            eligible_for_wait=eligible,gate_progress_s=self.gate*.02,release_progress_s=self.release*.02,
            acquisition_started=self.started,acquisition_released=self.released,
            acquisition_started_at_s=self.started_at,secure_release_time_s=self.released_at,
            wait_ticks=self.wait_ticks,wait_seconds=self.wait_ticks*.02,
            maximum_wait_s=self.config['maximum_wait_s'],aborted=self.aborted,
            abort_reason=self.abort_reason,abort_time_s=self.abort_at,abort_pose_hold=self.aborted,
            secure_grip=secure,grip_dwell_s=span,relative_drift_m=drift,bilateral_contact=bilateral,
            support_observed=bool(supported),near_tool=near,object_tool_relative_m=local.tolist(),
            physical_time_s=t,progress_before_s=before,progress_after_s=self.progress*.02,
            increment_s=(self.progress*.02-before))


def inspect_guarded_trace(episode, base, model, config):
    states, actions = episode['trajectory'], episode['actions']
    if len(states) != len(actions)+1 or not states:
        raise ValueError('Guarded trace lacks the complete state/action prefix.')
    refs = previous.References(base,states[0]['object_position_m'],states[0]['target_position_m'])
    gate,release = refs.window(config['gate_offset_s'])
    machine = AcquisitionReconstruction(config,gate,release)
    measurements = previous.CausalMeasurements(states[0]['object_position_m'],states[0]['target_position_m'],
        dict(minimum_pad_force_n=.20,secure_dwell_s=.10,maximum_relative_drift_m=.002,acquisition_only=True))
    counts=dict(action_count=len(actions),feature_mismatches=0,progress_mismatches=0,
        supervisor_metadata_mismatches=0,command_mismatches=0,diagnostic_mismatches=0,
        state_action_binding_mismatches=0,maximum_joint_command_error_rad=0.,maximum_jaw_command_error_m=0.,
        mismatch_examples=[])
    for i,(state,action) in enumerate(zip(states,actions)):
        issues=[]; decision=action['clock_decision']; before=machine.progress*.02
        nominal_jaw=refs.action(state,i*.02)[1]
        feature,_=measurements.observe(state,nominal_jaw,before,gate,release)
        probability=previous.tree_probability(model,feature)
        raw_jaw=refs.action(state,before)[1]
        expected=machine.observe(state,raw_jaw,probability)
        if not np.allclose(np.asarray(decision.get('features')),feature,atol=TOL,rtol=0):
            counts['feature_mismatches']+=1;issues.append('causal_features')
        progress_fields=('physical_time_s','progress_before_s','progress_after_s','increment_s')
        if (not math.isclose(state['t_seconds'],i*.02,abs_tol=1e-8,rel_tol=0)
            or not math.isclose(action['t_seconds'],i*.02,abs_tol=1e-8,rel_tol=0)
            or any(not _same(decision.get(k),expected[k]) for k in progress_fields)):
            counts['progress_mismatches']+=1;issues.append('binary_causal_progress')
        if any(not _same(decision.get(k),v) for k,v in expected.items() if k not in progress_fields):
            counts['supervisor_metadata_mismatches']+=1;issues.append('sampled_supervisor_state')
        target,jaw=refs.action(state,before)
        if machine.aborted:
            q=np.asarray(state['joint_position_rad']);limits=np.asarray(base['joint_limits_rad'])
            target=np.clip(q+np.clip(machine.abort_pose-q,-base['joint_delta_limit_rad'],base['joint_delta_limit_rad']),
                           limits[:,0]+.02,limits[:,1]-.02)
        joint_error=float(np.max(np.abs(np.asarray(action['joint_targets_rad'])-target)))
        jaw_error=abs(action['gripper_position_m']-jaw)
        counts['maximum_joint_command_error_rad']=max(counts['maximum_joint_command_error_rad'],joint_error)
        counts['maximum_jaw_command_error_m']=max(counts['maximum_jaw_command_error_m'],jaw_error)
        if (joint_error>TOL or jaw_error>TOL or not _same(decision.get('nominal_physical_jaw_m'),nominal_jaw)
            or not _same(decision.get('frozen_jaw_target_m'),jaw) or decision.get('jaw_mode')!='coordinated_reference'):
            counts['command_mismatches']+=1;issues.append('native_capped_reference_or_abort_pose')
        if (state.get('command')!=action or not _same(state.get('issued_joint_targets_rad'),action['joint_targets_rad'])
                or not _same(state.get('issued_gripper_position_m'),action['gripper_position_m'])):
            counts['state_action_binding_mismatches']+=1;issues.append('state_action_binding')
        if issues and len(counts['mismatch_examples'])<25:
            counts['mismatch_examples'].append(dict(action_index=i,physical_time_s=state['t_seconds'],mismatches=issues))
    diagnostic=episode.get('policy_diagnostics',{})
    expected_diag=dict(schema='dvidia.piper-bounded-grip-wait/v0.4',authored_acquisition_supervisor=True,
        learned_prediction_advisory=True,newly_trained_model=False,config=config,acquisition_started=machine.started,
        acquisition_released=machine.released,acquisition_started_at_s=machine.started_at,
        secure_release_time_s=machine.released_at,wait_ticks=machine.wait_ticks,wait_seconds=machine.wait_ticks*.02,
        advisory_hold_count=machine.advisory_holds,supervisor_override_count=machine.overrides,
        aborted=machine.aborted,abort_reason=machine.abort_reason,abort_time_s=machine.abort_at,
        original_arm_path_preserved=not machine.aborted,abort_changes_arm_reference=machine.aborted,
        native_command_caps_preserved=True,physical_robot_ready=False,physical_action_calls=len(actions),
        progress_seconds=machine.progress*.02)
    for key,expected in expected_diag.items():
        if isinstance(expected,dict):same=diagnostic.get(key)==expected
        else:same=_same(diagnostic.get(key),expected)
        counts['diagnostic_mismatches']+=int(not same)
    counts.update(reconstructed_aborted=machine.aborted,reconstructed_abort_reason=machine.abort_reason,
        reconstructed_wait_ticks=machine.wait_ticks,reconstructed_acquisition_released=machine.released,
        first_acquisition_release_s=machine.released_at,final_progress_s=machine.progress*.02,
        complete_20s_action_trace=len(actions)==1000,mismatch_example_limit=25)
    return counts


def _requests(root,protocol,phase,frozen):
    if phase in ('all','development'):
        for name,config in protocol['candidates'].items():
            for case in protocol['cases']['development']:
                yield ('development',name,case,'guarded',root/'development'/name/case['id'],config)
    if phase in ('all','confirmation'):
        if frozen is None:raise ValueError('Confirmation audit requires the frozen selection.')
        config=protocol['candidates'][frozen['selected']]
        for case in protocol['cases']['confirmation']:
            for variant in VARIANTS:
                yield ('confirmation',frozen['selected'],case,variant,root/'confirmation'/case['id']/variant,config)


def _attempt(request,context):
    phase,candidate,case,variant,folder,config=request
    root,protocol=context['root'],context['protocol']
    expected_binding=dict(schema=STUDY_SCHEMA,protocol_sha256=context['protocol_hash'],runtime_sha256=protocol['source_sha256'],
        case=case,variant=variant,config=config if variant=='guarded' else None,
        frozen_sha256=context['frozen_hash'] if phase=='confirmation' else None)
    if folder.is_symlink() or not folder.is_dir() or {p.name for p in folder.iterdir()} != NAMES|{'receipt.json'}:
        raise ValueError('Attempt evidence inventory differs from the scheduled request.')
    receipt=_read(folder/'receipt.json')
    if receipt.get('binding')!=expected_binding or _read(folder/'started.json')!=expected_binding or set(receipt.get('files',{}))!=NAMES:
        raise ValueError('Attempt receipt/started identity differs from the frozen request.')
    for name,expected in receipt['files'].items():
        if _hash(folder/name)!=expected:raise ValueError('Attempt evidence bytes changed: '+name)
    episode=previous._episode(folder/'episode.json.gz');arrays=previous._npz(folder/'native.npz');metadata=_read(folder/'native.json')
    base,jaw,model=context['base'],context['jaw'],context['model']
    expected=dict(schema='dvidia.piper-grip-clock-episode/v0.3',study_schema=STUDY_SCHEMA,
        case_id=case['id'],seed=case['seed'],condition=case['condition'],policy_kind=variant,
        base_model_sha256=previous._canonical_hash(base),jaw_model_sha256=previous._canonical_hash(jaw),
        model_sha256=previous._canonical_hash(model) if variant in ('guarded','clock') else None,
        jaw_mode='coordinated_reference',supervisor_config=config if variant=='guarded' else None,
        implementation_sha256=protocol['source_sha256'],teacher_config=context['teacher'],post_grip_parameters=context['intervention'])
    if any(episode.get(k)!=v for k,v in expected.items()):raise ValueError('Episode identity differs from its frozen request.')
    for name,value in (('object_position_m',[*case['start_xy'],.082]),('target_position_m',[*case['goal_xy'],.08])):
        if not np.allclose(episode['config'][name],value,atol=1e-12,rtol=0):raise ValueError('Episode workspace differs.')
    if metadata.get('post_grip_intervention',{}).get('parameters')!=dict(interruption=case['condition']=='post_grip',**context['intervention']):
        raise ValueError('Native intervention differs from its assigned condition.')
    numeric=context['strict'].score_feedback_episode(episode,arrays,metadata,case['goal_xy'])
    try:post=context['post'].analyze_post_grip(arrays,metadata)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError) as exc:
        post=dict(schema='dvidia.piper-post-grip-analysis/v1',valid=False,reason=str(exc)[:180],interruption_condition=case['condition']=='post_grip')
    saved=_read(folder/'score.json')
    expected_score=dict(numeric,post_grip_analysis=post)
    if not post['valid']:expected_score.update(valid=False,success=False,native_score_reason=numeric['reason'],reason='post_grip_intervention_mismatch')
    numeric_matches=expected_score==saved
    intervention_matches=post==_read(folder/'post-grip.json')
    if variant=='guarded':trace=inspect_guarded_trace(episode,base,model,config)
    else:trace=previous.inspect_trace(episode,arrays,base,jaw,model if variant=='clock' else None,context['teacher'],'coordinated_reference')
    mismatches=sum(value for key,value in trace.items() if key.endswith('_mismatches'))+int(not numeric_matches)+int(not intervention_matches)
    aborted=trace.get('reconstructed_aborted',False)
    success=bool(expected_score['valid'] and expected_score['success'] and not aborted)
    return dict(phase=phase,candidate=candidate,case_id=case['id'],variant=variant,relative_path=folder.relative_to(root).as_posix(),
        receipt_sha256=_hash(folder/'receipt.json'),numeric_score_matches=numeric_matches,
        intervention_matches=intervention_matches,audit_mismatches=mismatches,valid=expected_score['valid'],
        raw_strict_success=expected_score['success'],success=success,aborted=aborted,
        abort_reason=trace.get('reconstructed_abort_reason'),trace=trace)


_CONTEXT=None

def _initialize(context):
    global _CONTEXT
    from dvidia_training import piper_feedback_score as strict
    from dvidia_training import piper_grip_clock_env as post
    for module in (strict,post):
        if _hash(module.__file__)!=context['protocol']['source_sha256'][Path(module.__file__).name]:
            raise ValueError('Loaded strict/intervention scorer source differs from the frozen protocol.')
    _CONTEXT=dict(context,strict=strict,post=post)


def _worker(request):
    try:return _attempt(request,_CONTEXT)
    except (ValueError,KeyError,TypeError,IndexError,OverflowError,OSError) as exc:
        phase,candidate,case,variant,folder,config=request
        return dict(phase=phase,candidate=candidate,case_id=case['id'],variant=variant,
            relative_path=folder.relative_to(_CONTEXT['root']).as_posix(),audit_mismatches=1,
            error=dict(type=type(exc).__name__,message=str(exc)[:250]),valid=False,raw_strict_success=False,
            success=False,aborted=None,abort_reason=None)


def audit(study,output,*,phase='all',source_root=None,workers=1):
    if phase not in ('all','development','confirmation'):raise ValueError('Unknown audit phase.')
    if type(workers) is not int or not 1<=workers<=4:raise ValueError('Use one to four audit workers.')
    root=Path(study);source_root=Path(source_root) if source_root else Path(__file__).resolve().parents[1]/'src'/'dvidia_training'
    protocol=_read(root/'protocol.json');prepared=_read(root/'prepared.json');protocol_hash=_hash(root/'protocol.json')
    if protocol.get('schema')!=STUDY_SCHEMA or prepared!=dict(schema=STUDY_SCHEMA,protocol_sha256=protocol_hash,source_sha256=protocol['source_sha256']):
        raise ValueError('Prepared protocol seal differs.')
    for name,digest in protocol['source_sha256'].items():
        if _hash(root/'source'/name)!=digest or _hash(source_root/name)!=digest:raise ValueError('Frozen source identity differs: '+name)
    for name,digest in protocol['input_sha256'].items():
        if _hash(root/name)!=digest:raise ValueError('Frozen policy bytes differ: '+name)
    frozen=_read(root/'frozen.json') if (root/'frozen.json').is_file() else None
    frozen_hash=_hash(root/'frozen.json') if frozen else None
    if frozen:
        if frozen['protocol_sha256']!=protocol_hash or frozen['source_sha256']!=protocol['source_sha256']:
            raise ValueError('Frozen confirmation selection differs from the protocol.')
        for name,digest in frozen['development_evidence'].items():
            if _hash(root/name)!=digest:raise ValueError('Frozen development evidence differs.')
        if _hash(root/'development-results.json')!=frozen['development_sha256']:
            raise ValueError('Frozen development aggregate differs.')
    teacher=dict(gate_offset_s=.16,minimum_pad_force_n=.20,secure_dwell_s=.10,maximum_relative_drift_m=.002,acquisition_only=True)
    intervention=dict(force_threshold_n=.10,lift_threshold_m=.065,eligibility_dwell_s=.100,delay_s=.100,pulse_duration_s=.040,minimum_opening_m=.020)
    context=dict(root=root,protocol=protocol,protocol_hash=protocol_hash,frozen_hash=frozen_hash,
        base=_read(root/'base-model.json'),jaw=_read(root/'jaw-model.json'),model=_read(root/'selected.json'),teacher=teacher,intervention=intervention)
    requests=list(_requests(root,protocol,phase,frozen))
    required={'development':48,'confirmation':512,'all':560}[phase]
    if len(requests)!=required:raise ValueError('Scheduled audit roster size differs.')
    if workers==1:
        _initialize(context);rows=[_worker(request) for request in requests]
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(context,)) as pool:
            rows=list(pool.map(_worker,requests,chunksize=1))
    if _hash(root/'protocol.json')!=protocol_hash or frozen_hash is not None and _hash(root/'frozen.json')!=frozen_hash:
        raise ValueError('Protocol/selection changed during the audit.')
    summary={}
    for variant in VARIANTS:
        own=[row for row in rows if row['variant']==variant]
        summary[variant]={key:sum(bool(row.get(key)) for row in own) for key in ('valid','raw_strict_success','success','aborted')}
        summary[variant]['attempts']=len(own)
    value=dict(schema=SCHEMA,study_schema=STUDY_SCHEMA,phase=phase,protocol_sha256=protocol_hash,frozen_sha256=frozen_hash,
        source_sha256=protocol['source_sha256'],inputs_sha256=protocol['input_sha256'],
        auditor_sha256=_hash(__file__),dependency_auditor_sha256=_hash(previous.__file__),
        report_sha256=_hash(root/'report.json') if (root/'report.json').is_file() else None,
        scheduled_attempts=len(requests),audited_attempts=len(rows),audit_mismatches=sum(row['audit_mismatches'] for row in rows),
        rows=rows,summary=summary,qualified_robot_skill=False,physical_robot_ready=False,
        scope='Read-only independent reconstruction of recorded sampled supervisor state, frozen commands and numeric native strict scores; no new physics, sensor calibration, continuous-readiness or hardware safety qualification.')
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    # Existing flushed/exclusive writer is only an output helper; no lifecycle
    # source or v0.4 runtime is imported to decide any audit result.
    from dvidia_training.piper_imitation_benchmark import _atomic_create
    _atomic_create(output,raw)
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=('all','development','confirmation'),default='all')
    parser.add_argument('--source-root');parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args(argv)
    result=audit(args.study,args.output,phase=args.phase,source_root=args.source_root,workers=args.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256','inputs_sha256')},indent=2))

if __name__=='__main__':main()
