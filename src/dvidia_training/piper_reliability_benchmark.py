"""Paired PiPER v0.5 reliability receipts, with frozen v0.4 as reference.

Inspectors check recorded consistency, not physical authenticity or sensor accuracy.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import gzip
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

from . import piper_task as task, piper_task_benchmark as base, piper_grip_benchmark as prior
from . import piper_contact_benchmark as contact, piper_recovery_benchmark as recovery, robot_profiles
from .reference_replay import _decode, _json_bytes, _read

SCHEMA = 'dvidia.piper-reliability-benchmark/v0.5'
PROTOCOL_SCHEMA = 'dvidia.piper-reliability-protocol/v0.5'
INTEGRITY_SCHEMA = 'dvidia.piper-reliability-integrity/v0.5'
MODES = ('recovery', 'reacquire', 'reacquire_grasp')
VARIANTS = tuple(v for mode in MODES for v in (mode+'_placement', mode+'_open_jaw'))+('hold',)
MASSES, FRICTIONS, STRATA = contact.MASSES, contact.FRICTIONS, contact.STRATA
SOURCE_FILES = recovery.SOURCE_FILES + ('piper_reacquisition.py', 'piper_grasp_strategy.py', 'piper_reliability_benchmark.py')
MAX_FILE_BYTES, MAX_BUNDLE_BYTES, MAX_DECODED_BYTES = contact.MAX_FILE_BYTES, contact.MAX_BUNDLE_BYTES, contact.MAX_DECODED_BYTES
SCOPE = ('Paired authored PiPER reliability pilot, privileged native state and synthetic rigid blocks. '
         'No learned policy, sensor pipeline, calibrated contact, shoe competence or physical robot qualification.')
EVALUATION_CONTRACT = {'schema':'dvidia.piper-reliability-sampled-evaluation/v0.5',
    'contact':contact.EVALUATION_CONTRACT,
    'comparison':'Completion and valid denominator first; completion time deltas only in paired common successes. '
                 'Unsuccessful acquisitions retain bounded terminal decisions and the full original 20-second episode. '
                 'Recovery duration includes unresolved holds; sampled motion is not a physical speed guarantee.',
    'scope':'Recorded 20ms boundaries, not true-slip labels, independent physical truth or real-time sensor latency.'}

decode_attempt = prior.decode_attempt
_summary = base._summary
_near = contact._near


def implementation_hashes():
    root = Path(__file__).parent
    return {name:sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _memory_contract():
    return contact._memory_contract()


def _recovery_contract():
    from .piper_contact_recovery import recovery_contract
    return recovery_contract()


def _variant(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown recovery variant')
    if variant == 'hold':
        return 'hold','baseline'
    mode = next(m for m in sorted(MODES,key=len,reverse=True) if variant.startswith(m+'_'))
    return ('replay_open_jaw' if variant.endswith('open_jaw') else 'placement'),mode


def make_protocol(*, seed=950031, cohort='development'):
    protocol = contact.make_protocol(seed=seed,cohort=cohort)
    protocol.update(schema=PROTOCOL_SCHEMA, scope=SCOPE, variants=list(VARIANTS),
        implementation_sha256=implementation_hashes(), recovery_contract=_recovery_contract(), reliability_contract=_reliability_contract(),
        planned_attempts=12*len(VARIANTS), policy_inputs_contract=POLICY_INPUTS_CONTRACT,
        evaluation_contract=EVALUATION_CONTRACT,
        comparison='Identical native mechanics, initial state and full twenty-second duration per scene. '
            'Frozen v0.4 recovery versus bounded second lift and optional fixed alternate-grasp plus second lift. '
            'Each placement has its own complete arm-tape/open-jaw control, plus a shared initial-arm hold. '
            'The second-lift ablation separates carry recovery from alternate-grasp planning. '
            'Every attempted failure, invalid result and unresolved recovery remains in the denominator.')
    return protocol


def evaluate_episode(raw):
    observed = contact.evaluate_episode(raw)
    events = raw.get('grip_guard',{}).get('events',[])
    holds = []
    for event in events:
        if event['type']!='reassessment_started':
            continue
        ending = next((e for e in events if e['t_seconds']>=event['t_seconds'] and e['type'] in ('carry_resumed','abort')),None)
        end = ending['t_seconds'] if ending else raw.get('simulated_seconds',event['t_seconds'])
        holds.append({'start_s':event['t_seconds'],'end_s':end,'duration_s':end-event['t_seconds'],
            'outcome':ending['type'] if ending else 'unresolved',
            'actual_motion':contact._motion_window(raw.get('trajectory',[]),event['t_seconds'],end)})
    abort = next((e for e in events if e['type']=='abort'),None)
    success = raw.get('valid') is True and raw.get('success') is True
    completion = raw.get('metrics',{}).get('first_success_time_seconds')
    return {'schema':EVALUATION_CONTRACT['schema'],'contact':observed,'holds':holds,
        'completion_time_s':completion if success else None,
        'abort_time_s':abort['t_seconds'] if abort else None,
        'abort_reason':abort['reason'] if abort else None,
        'second_probe_started':sum(e['type']=='second_probe_started' for e in events),
        'second_probe_qualified':sum(e['type']=='second_probe_qualified' for e in events),
        'restoration_started':sum(e['type']=='speed_restoration_started' for e in events),
        'full_speed_restored':sum(e['type']=='full_speed_restored' for e in events),
        'restoration_motion':[{'type':e['type'],'t_seconds':e['t_seconds'],
             'before':contact._motion_window(raw.get('trajectory',[]),max(0.,e['t_seconds']-.1),e['t_seconds']),
             'after':contact._motion_window(raw.get('trajectory',[]),e['t_seconds'],e['t_seconds']+.1)}
             for e in events if e['type'] in ('speed_restoration_started','full_speed_restored')]}


def aggregate(attempts, scenes):
    rows=[]
    for friction,stratum in zip(FRICTIONS,STRATA):
        for mass in MASSES:
            for variant in VARIANTS:
                group=[r for r in attempts if r['variant']==variant and r['config']['friction']==friction and r['config']['mass_kg']==mass]
                success=sum(r['valid'] is True and r['success'] is True for r in group)
                evaluations=[r['recovery_evaluation'] for r in group]
                rows.append({'friction':friction,'mass_kg':mass,'stratum':stratum,'variant':variant,
                    'attempted':len(group),'valid':sum(r['valid'] is True for r in group),'success':success,
                    'failed_or_invalid':len(group)-success,'success_fraction':success/len(group) if group else None,
                    'retries':sum(r['grip_guard']['retry_count'] for r in group),
                    'aborted_attempts':sum(r['grip_guard']['aborted'] for r in group),
                    'resumed_completed':sum(e['contact']['resumed_completed'] for e in evaluations),
                    'resumed_not_completed':sum(e['contact']['resumed_not_completed'] for e in evaluations),
                    'holds':sum(len(e['holds']) for e in evaluations),
                    'unresolved_holds':sum(h['outcome']!='carry_resumed' for e in evaluations for h in e['holds']),
                    'restoration_started':sum(e['restoration_started'] for e in evaluations),
                    'full_speed_restored':sum(e['full_speed_restored'] for e in evaluations),
                    'second_probe_started':sum(e['second_probe_started'] for e in evaluations),
                    'second_probe_qualified':sum(e['second_probe_qualified'] for e in evaluations)})
    return rows


def paired_comparisons(attempts,scenes):
    lookup={(r['scene_id'],r['variant']):r for r in attempts}
    comparisons=[]
    for reference,candidate in ((MODES[a],MODES[b]) for a in range(len(MODES)) for b in range(a+1,len(MODES))):
        pairs=[]
        for scene in scenes:
            a,b=(lookup[scene['id'],m+'_placement'] for m in (reference,candidate))
            sa,sb=(r['valid'] is True and r['success'] is True for r in (a,b))
            ta,tb=(r['recovery_evaluation']['completion_time_s'] for r in (a,b))
            aa,ab=(r['recovery_evaluation']['abort_time_s'] for r in (a,b))
            pairs.append({'scene_id':scene['id'],'reference_valid':a['valid'],'candidate_valid':b['valid'],
                'reference_success':sa,'candidate_success':sb,
                'completion_delta_s':tb-ta if sa and sb and ta is not None and tb is not None else None,
                'common_abort_delta_s':ab-aa if not sa and not sb and aa is not None and ab is not None else None})
        common=[p['completion_delta_s'] for p in pairs if p['completion_delta_s'] is not None]
        comparisons.append({'reference':reference,'candidate':candidate,'attempted_pairs':len(pairs),
            'common_successes':sum(p['reference_success'] and p['candidate_success'] for p in pairs),
            'gained_successes':sum(not p['reference_success'] and p['candidate_success'] for p in pairs),
            'lost_successes':sum(p['reference_success'] and not p['candidate_success'] for p in pairs),
            'neither_succeeded':sum(not p['reference_success'] and not p['candidate_success'] for p in pairs),
            'common_success_mean_delta_s':sum(common)/len(common) if common else None,'pairs':pairs})
    return comparisons


def quality_gates(attempts, scenes):
    reference_pairs=[p for p in paired_comparisons(attempts,scenes) if p['reference']==MODES[0]]
    comparison=next(p for p in reference_pairs if p['candidate']==MODES[-1])
    controls=[r for r in attempts if r['control']!='placement']
    clean=bool(attempts) and all(r['valid'] and not r.get('warnings') and r.get('metrics',{}).get('unexpected_contact_native_ticks')==0 for r in attempts)
    improved=comparison['gained_successes']>0 and comparison['lost_successes']==0
    return {'complete_planned_denominator':len(attempts)==len(scenes)*len(VARIANTS),
        'all_native_attempts_valid':all(r['valid'] for r in attempts),
        'all_controls_fail':bool(controls) and not any(r['valid'] and r['success'] for r in controls),
        'native_integrity_clean':clean,'gained_without_lost_completions':improved,
        'selected_candidate':MODES[-1],
        'reference_comparisons':[{'candidate':p['candidate'],'gained_successes':p['gained_successes'],
            'lost_successes':p['lost_successes'],'gained_without_lost_completions':p['gained_successes']>0 and p['lost_successes']==0}
            for p in reference_pairs],
        'scope':'Descriptive pilot gates; a valid negative result is retained. Native policy audit and prior-success regression are separate release checks, not inferred from these flags.'}


def execution_metrics(attempts):
    result=[]
    for variant in VARIANTS:
        rows=[r for r in attempts if r['variant']==variant]
        samples=sorted(v for r in rows for v in r.get('timing',{}).get('control_step_samples_seconds',[]))
        quantile=lambda q:samples[max(0,math.ceil(len(samples)*q)-1)] if samples else None
        wall=sum(r.get('timing',{}).get('episode_wall_seconds',0.) for r in rows)
        simulated=sum(r.get('simulated_seconds',0.) for r in rows)
        result.append({'variant':variant,'attempted':len(rows),'control_windows':len(samples),
            'p50_seconds':quantile(.5),'p95_seconds':quantile(.95),'p99_seconds':quantile(.99),
            'max_seconds':samples[-1] if samples else None,
            'missed_20ms_windows':sum(v>.02 for v in samples),
            'missed_20ms_fraction':sum(v>.02 for v in samples)/len(samples) if samples else None,
            'episode_wall_seconds':wall,'simulated_seconds':simulated,
            'simulated_seconds_per_episode_wall_second':simulated/wall if wall else None,
            'scope':'Observed host computation including native integration/state collection; no sensor-to-actuator or hard-real-time guarantee.'})
    return result


def _html(report):
    rows=''.join('<tr>'+''.join('<td>'+escape(str(row[k]))+'</td>' for k in ('friction','mass_kg','variant','success','attempted','valid','retries','holds','unresolved_holds'))+'</tr>' for row in report['summary'])
    attempts=''.join('<li><a href="'+escape(r['result_file'],quote=True)+'">'+escape(r['id'])+'</a>: '+('completed' if r['valid'] and r['success'] else 'failed' if r['valid'] else 'invalid')+'</li>' for r in report['attempts'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>DVIDIA · PiPER reliability v0.5</title><style>body{max-width:1100px;margin:auto;padding:24px;font:18px/1.6 Georgia;background:#fafaf7;color:#292d27}'
        'table{font:14px/1.5 system-ui;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}.scroll{overflow:auto}a{color:#396751}</style>'
        '<main><p>DVIDIA RESEARCH</p><h1>Measured placement reliability · v0.5</h1><p>'+escape(SCOPE)+'</p><p>All '+str(report['attempted'])+' planned attempts retained.</p>'
        '<p><a href="report.json">Complete report and paired timing</a> · <a href="protocol.json">Frozen protocol</a> · <a href="integrity.json">Integrity</a></p>'
        '<div class="scroll"><table><tr><th>Friction</th><th>Mass kg</th><th>Variant</th><th>Completed</th><th>Attempted</th><th>Valid</th><th>Retries</th><th>Holds</th><th>Unresolved</th></tr>'+rows+'</table></div>'
        '<p>'+escape(EVALUATION_CONTRACT['comparison'])+'</p><p>'+escape(base.TIMING_BOUNDARY)+'</p><h2>Every attempt</h2><ul>'+attempts+'</ul>'
        '<footer><a href="https://research.dvidia.org/">DVIDIA research</a> · <a href="https://x.com/iammrriver">@iammrriver</a> · <a href="mailto:hello@dvidia.org">hello@dvidia.org</a></footer></main></html>')


def run_benchmark(assets, output, *, seed=950031, cohort='development', progress=None):
    protocol = make_protocol(seed=seed, cohort=cohort)
    asset_root = task._asset_root(assets)
    provenance = robot_profiles.verify_assets('piper', asset_root)
    _, blobs = robot_profiles._payload('piper', asset_root)
    destination = Path(output)
    if destination.is_symlink():
        raise ValueError('Output must be a fresh regular directory')
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'attempts').mkdir()
    (destination/'licenses/agilex_piper').mkdir(parents=True)
    for source, name in (('agilex_piper/LICENSE',base.LICENSE_PATH),('agilex_piper/README.md',base.MODEL_NOTE_PATH)):
        (destination/name).write_bytes(blobs[source])
    base._write(destination/'protocol.json', protocol)
    digest, attempts, records = sha256((destination/'protocol.json').read_bytes()).hexdigest(), [], {}
    for scene in protocol['scenes']:
        tapes = {}
        for variant in VARIANTS:
            if implementation_hashes() != protocol['implementation_sha256']:
                raise ValueError('Implementation changed after protocol freeze; incomplete run preserved')
            control, mode = _variant(variant)
            began, identity = time.perf_counter(), scene['id']+'--'+variant
            try:
                if control == 'replay_open_jaw' and tapes.get(mode) is None:
                    raise ValueError('Valid complete '+mode+' placement arm tape unavailable')
                raw = task.run_episode(asset_root, task.PiperTaskConfig(**scene['config']), seed=scene['seed'],
                                       control=control, policy_mode=mode, record_step_timings=True, record_policy_inputs=True, arm_tape=tapes.get(mode) if control == 'replay_open_jaw' else None)
                if type(raw.get('valid')) is not bool or type(raw.get('success')) is not bool:
                    raise ValueError('Task runner omitted Boolean outcomes')
                if control == 'placement':
                    tape = raw.get('arm_tape')
                    ticks = round(scene['config']['timeout_s']/scene['config']['control_dt'])
                    tapes[mode] = tape if raw['valid'] and type(tape) is list and len(tape) == ticks else None
            except Exception as exc:
                inactive = _inactive(mode)
                policy_sources = _policy_sources(mode)
                raw = {'schema':task.SCHEMA,'scope':task.SCOPE,'valid':False,'success':False,'reason':'exception',
                       'error':{'type':type(exc).__name__,'message':str(exc).replace(str(asset_root.resolve()),'[assets]')[:500]},
                       'trajectory':[],'actions':[],'arm_tape':[],'metrics':{},'timing':{'control_step_samples_seconds':[]},'warnings':[],'simulated_seconds':0.,
                       'contract':protocol['contract'],
                       'implementation_sha256':{name:protocol['implementation_sha256'][name] for name in base.EPISODE_SOURCE_FILES},
                       'policy_implementation_sha256':{name:protocol['implementation_sha256'][name] for name in policy_sources},
                       'grip_guard':inactive}
            raw.update(id=identity,scene_id=scene['id'],variant=variant,control=control,policy_mode=mode,profile=task.PROFILE,
                       physical_robot_ready=False,learned_policy=False,seed=scene['seed'],config=scene['config'],
                       attempt_wall_seconds=time.perf_counter()-began)
            raw['recovery_evaluation'] = evaluate_episode(raw)
            decoded, name = _json_bytes(raw), 'attempts/'+identity+'.json.gz'
            encoded = gzip.compress(decoded, mtime=0)
            if len(decoded)>MAX_FILE_BYTES or len(encoded)>MAX_FILE_BYTES:
                raise ValueError('Attempt exceeds bounded gzip/raw size; incomplete run preserved')
            with (destination/name).open('xb') as stream:
                stream.write(encoded)
            records[name] = {'bytes':len(encoded),'sha256':sha256(encoded).hexdigest(),'encoding':'gzip',
                             'decoded_bytes':len(decoded),'decoded_sha256':sha256(decoded).hexdigest()}
            if implementation_hashes() != protocol['implementation_sha256']:
                raise ValueError('Implementation changed during attempt; incomplete evidence preserved')
            attempts.append({**_summary(raw),'result_file':name})
            if progress:
                progress({'id':identity,'valid':raw['valid'],'success':raw['success']})
    runtimes = [row['platform'] for row in attempts if type(row.get('platform')) is dict]
    report = {'schema':SCHEMA,'profile':task.PROFILE,'scope':SCOPE,'physical_robot_ready':False,'learned_policy':False,
              'cohort':cohort,'protocol':protocol,'protocol_sha256':digest,'implementation_sha256':protocol['implementation_sha256'],
              'asset_provenance':provenance,'simulation_asset_license':'MIT',
              'asset_license':{'path':base.LICENSE_PATH,'sha256':sha256(blobs['agilex_piper/LICENSE']).hexdigest()},
              'hardware':{'platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),
                          'logical_cpu_count':os.cpu_count(),'gpu_used':False,'process_lifetime_peak_rss_bytes':base._peak_rss_bytes(),
                          'episode_runtimes':list({json.dumps(x,sort_keys=True):x for x in runtimes}.values()),
                          'memory_scope':'Process lifetime high-water RSS before inspection; not per-episode or minimum RAM.'},
              'scenes':protocol['scenes'],'attempted':len(attempts),'valid_attempts':sum(row['valid'] for row in attempts),
              'summary':aggregate(attempts,protocol['scenes']),'paired_comparisons':paired_comparisons(attempts,protocol['scenes']),
              'quality_gates':quality_gates(attempts,protocol['scenes']),'execution_metrics':execution_metrics(attempts),'attempts':attempts}
    base._write(destination/'report.json',report)
    (destination/'report.html').write_text(_html(report),encoding='utf-8',newline='\n')
    for name in ('protocol.json','report.json','report.html',base.LICENSE_PATH,base.MODEL_NOTE_PATH):
        data = (destination/name).read_bytes()
        records[name] = {'bytes':len(data),'sha256':sha256(data).hexdigest()}
    base._write(destination/'integrity.json',{'schema':INTEGRITY_SCHEMA,'files':records})
    inspect_reliability_bundle(destination, assets=asset_root)
    return report



POLICY_INPUTS_CONTRACT = {
    'native_lift_seen': 'Boolean native 1ms lift latch recorded at each 20ms state, including terminal; monotonic and corroborated by sampled lift evidence. The original teacher reads this recorded flag during policy audit.',
    'audit': 'Portable checks validate stored native/task/pair evidence and explicit second-probe prerequisites/bounds. With pinned assets, replay the frozen authored policy and reconstruct IK at recorded states. No native trajectory reintegration or authenticity claim.'}
INSPECTION_SCOPE = ('Recorded consistency and bounded second-probe evidence. Optional native audit reconstructs '
    'policy decisions and IK using recorded native states and lift latch, not physics reintegration, authenticity or hardware safety.')


def _reliability_contract():
    from . import piper_reacquisition, piper_grasp_strategy
    contracts = {'recovery': _recovery_contract(), 'reacquire': piper_reacquisition.reacquisition_contract(),
                 'reacquire_grasp': piper_grasp_strategy.combined_contract()}
    return {mode: contracts[mode] for mode in MODES}


def _policy_sources(mode):
    names = ('piper_grip_guard.py',)
    if mode != 'baseline':
        names += ('piper_contact_memory.py', 'piper_contact_recovery.py')
    if mode in ('reacquire', 'reacquire_grasp'):
        names += ('piper_reacquisition.py',)
    if mode == 'reacquire_grasp':
        names += ('piper_grasp_strategy.py',)
    return names


def _inactive(mode):
    from . import piper_grip_guard, piper_contact_recovery, piper_reacquisition, piper_grasp_strategy
    owner = {'baseline': piper_grip_guard, 'recovery': piper_contact_recovery,
             'reacquire': piper_reacquisition, 'reacquire_grasp': piper_grasp_strategy}[mode]
    return owner.inactive_summary(mode)


def _policy_inputs(raw):
    states = raw['trajectory']
    flags = [s.get('native_lift_seen') for s in states]
    if any(type(value) is not bool for value in flags) or flags and flags[0]:
        raise ValueError('Missing/invalid initial recorded native lift input')
    if any(a and not b for a,b in zip(flags,flags[1:])):
        raise ValueError('Recorded native lift input is not latched')
    if raw['valid'] and flags[-1] is not raw['metrics'].get('lift_seen'):
        raise ValueError('Recorded native lift input disagrees with terminal metric')
    if not any(flags):
        return
    index = flags.index(True)
    config = raw['config']
    origin = states[0]['object_position_m'][2]
    # A native qualifying interval can start/end between snapshots. Corroborate
    # its interior boundaries, without pretending to recreate the 1ms latch.
    minimum = max(1, math.floor(config['min_lift_dwell_s']/config['control_dt'])-1)
    start = states[index]['t_seconds']-config['min_lift_dwell_s']-config['control_dt']
    streak = longest = 0
    for s in states[:index+1]:
        if s['t_seconds'] < start-1e-9:
            continue
        eligible = (all(s['grasp_contact_counts'].values()) and not s['table_object_contact_count']
                    and s['object_position_m'][2]-origin+1e-9 >= config['min_lift_m'])
        streak = streak+1 if eligible else 0
        longest = max(longest,streak)
    if longest < minimum:
        raise ValueError('Native lift input lacks sampled qualification corroboration')


def _same_tree(actual, expected):
    if type(expected) is dict:
        return type(actual) is dict and set(actual)==set(expected) and all(_same_tree(actual[k],v) for k,v in expected.items())
    if type(expected) is list:
        return type(actual) is list and len(actual)==len(expected) and all(_same_tree(a,b) for a,b in zip(actual,expected))
    return _near(actual, expected)


def audit_policy(raw, assets, protocol):
    """Reconstruct decisions/IK at recorded states; deliberately no mj_step.

    qpos/qvel are loaded only into an independent inspection model. This does
    not produce or modify a rollout and cannot validate physical authenticity.
    """
    if implementation_hashes() != protocol['implementation_sha256']:
        raise ValueError('Native policy audit requires the exact frozen source')
    if raw['control'] != 'placement' or not raw['actions']:
        return
    from .piper_contact_recovery import ContactRecoveryPiperTaskPolicy
    from .piper_reacquisition import ReacquisitionPiperTaskPolicy
    from .piper_grasp_strategy import GraspReacquisitionPiperTaskPolicy
    env = task.PiperTaskEnv(assets,task.PiperTaskConfig(**raw['config']))
    env.reset(raw['seed'])
    if task.sim.model_digest(env.model) != raw['compiled_model_sha256']:
        raise ValueError('Native policy audit model differs from recorded model')
    policy = {'recovery': ContactRecoveryPiperTaskPolicy, 'reacquire': ReacquisitionPiperTaskPolicy,
              'reacquire_grasp': GraspReacquisitionPiperTaskPolicy}[raw['policy_mode']](env,raw['policy_mode'])
    mj,np = task.sim._native_imports()
    for state, recorded in zip(raw['trajectory'],raw['actions']):
        env.data.qpos[:] = state['qpos']
        env.data.qvel[:] = state['qvel']
        env.data.ctrl[:] = state['native_actuator_controls']
        env.lift_seen, env.elapsed, env.done = state['native_lift_seen'],state['t_seconds'],False
        mj.mj_forward(env.model,env.data)
        quaternion=np.empty(4)
        mj.mju_mat2Quat(quaternion,env.data.site_xmat[env.tcp])
        if sum(a*b for a,b in zip(quaternion,state['tcp_quaternion_wxyz']))<0:
            quaternion=-quaternion
        if (not _near(state['tcp_position_m'],env.data.site_xpos[env.tcp].tolist())
                or not _near(state['tcp_quaternion_wxyz'],quaternion.tolist())):
            raise ValueError('Recorded TCP pose differs from native forward kinematics')
        observation = deepcopy(state)
        for key in ('command','guard_diagnostic','issued_joint_targets_rad','issued_gripper_position_m'):
            observation.pop(key,None)
        action = policy(observation)
        expected = dict(t_seconds=state['t_seconds'],**action,phase=policy.issued_phase)
        if not _same_tree(recorded,expected):
            raise ValueError('Recorded command differs from causal policy/IK replay at '+str(state['t_seconds']))
        if not _same_tree(state.get('guard_diagnostic'),policy.diagnostic):
            raise ValueError('Recorded diagnostic differs from causal policy replay at '+str(state['t_seconds']))
    if not _same_tree(raw['grip_guard'],policy.summary()):
        raise ValueError('Recorded events/summary differ from causal policy replay')


def _reliability(raw, protocol):
    """Portable probe evidence audit; full policy/IK audit is a separate layer."""
    if raw.get('recovery_evaluation') != evaluate_episode(raw):
        raise ValueError('Reliability evaluation differs from recorded evidence')
    mode, guard = raw['policy_mode'],raw['grip_guard']
    if mode in ('baseline','recovery'):
        recovery._recovery(dict(raw,recovery_evaluation=recovery.evaluate_episode(raw)),protocol)
        return
    expected = _inactive(mode)
    if (mode not in MODES or type(guard) is not dict or guard.get('schema')!=expected['schema']
            or guard.get('mode')!=mode or guard.get('contract')!=expected['contract']
            or raw.get('policy_implementation_sha256')!={n:protocol['implementation_sha256'][n] for n in _policy_sources(mode)}):
        raise ValueError('Reliability controller identity/contract/source mismatch')
    if raw['control']!='placement' or not raw['actions']:
        if guard != expected:
            raise ValueError('Inactive reliability mode contains policy decisions')
        return
    if guard.get('active') is not True or guard.get('diagnostic_count') != len(raw['actions']):
        raise ValueError('Active reliability diagnostic count mismatch')
    events = guard.get('events')
    allowed = {'probe_started','acquisition_qualified','alarm','retry_started','retry_completed','abort',
               'slowdown_started','reassessment_started','carry_resumed','unchanged_retry_refused',
               'speed_restoration_started','full_speed_restored','second_probe_started','second_probe_qualified'}
    if type(events) is not list or len(events)>100 or any(type(e) is not dict or e.get('type') not in allowed for e in events):
        raise ValueError('Invalid bounded reliability events')
    by_time, prior_time = {},-1.
    for event in events:
        clock = base._numeric(event.get('t_seconds'),'policy event time',0.,raw['simulated_seconds'])
        if clock < prior_time:
            raise ValueError('Unordered reliability events')
        prior_time = clock
        by_time.setdefault(round(clock,8),[]).append(event)
    from .piper_reacquisition import reacquisition_contract
    cfg = reacquisition_contract()['parameters']
    memory = protocol['contact_contract']['parameters']
    acquisition = protocol['acquisition_contract']['parameters']
    history, origin, second, hold_started, low_since, qualify_since = [],None,None,None,None,None
    active_probe, qualified, probe_count, resume_count, observed_events = False,False,0,0,0
    for state in raw['trajectory']:
        command = state.get('command')
        current = by_time.get(round(state['t_seconds'],8),[])
        if command is None:
            if current:
                raise ValueError('Reliability event lacks issued action')
            continue
        d, now = state.get('guard_diagnostic'),state['t_seconds']
        if type(d) is not dict or d.get('t_seconds')!=now:
            raise ValueError('Missing aligned reliability diagnostic')
        if d.get('phase')=='probe' or d.get('qualified') is True:
            history.append(state)
            history=[s for s in history if s['t_seconds']>=now-memory['history_window_s']-1e-9]
        else:
            history=[]
        h=contact._history_values(history,memory)
        for key,value in h.items():
            if not _near(d.get(key),value):
                raise ValueError('Reliability history differs from causal native observations at '+str(now)+': '+key)
        if (d.get('bilateral_contact') is not all(state['grasp_contact_counts'].values())
                or d.get('table_supported') is not bool(state['table_object_contact_count'])):
            raise ValueError('Reliability current contact diagnostic differs from native state')
        types=[e['type'] for e in current]
        for event in current:
            if (type(event.get('diagnostic')) is not dict or event['diagnostic'].get('t_seconds')!=now
                    or type(event.get('reason')) is not str or type(event.get('retry_index')) is not int):
                raise ValueError('Malformed aligned reliability event')
        observed_events+=len(current)
        if 'retry_completed' in types:
            origin=None
        if 'probe_started' in types:
            origin=state
        if origin is not None and not _near(d.get('object_height_delta_m'),state['object_position_m'][2]-origin['object_position_m'][2]):
            raise ValueError('Reliability object height differs from original probe origin')
        if 'acquisition_qualified' in types:
            qualified=True
        if 'retry_started' in types:
            qualified=False
        if 'reassessment_started' in types:
            if not qualified:
                raise ValueError('Reliability hold precedes qualified acquisition')
            hold_started,low_since=now,None
        stable=(h['history_known'] and h['history_displacement_m']<=memory['settled_displacement_m']
            and h['history_excursion_m']<=memory['settled_displacement_m'] and h['history_rate_m_s']<=memory['settled_rate_m_s']
            and h['history_loaded_bilateral_all'] and h['history_unsupported_all'])
        if not active_probe:
            low=(hold_started is not None and now-hold_started+1e-9>=memory['hold_minimum_s']
                and stable and not state['unexpected_contact_count'] and origin is not None
                and cfg['low_hold_minimum_rise_m']<=d['object_height_delta_m']<acquisition['probe_min_object_rise_m']
                and (command['phase']=='memory_hold' or 'second_probe_started' in types))
            low_since=(now if low_since is None else low_since) if low else None
        if 'second_probe_started' in types:
            if (probe_count or not qualified or low_since is None or now-low_since+1e-9<cfg['low_hold_dwell_s']
                    or types.count('second_probe_started')!=1):
                raise ValueError('Second probe lacks bounded stable-low causal dwell')
            second,active_probe,qualify_since=state,True,None
            probe_count+=1
        if active_probe:
            if second is None or command['gripper_position_m']!=0.:
                raise ValueError('Second probe must retain a closed jaw')
            measured=state['tcp_position_m'][2]-second['tcp_position_m'][2]
            anchor=contact._local_offset(second)
            drift=math.dist(contact._local_offset(state),anchor)
            expected_fields={'second_probe_count':probe_count,'second_probe_started_s':second['t_seconds'],
                'second_probe_origin_tcp_m':second['tcp_position_m'],'second_probe_orientation_wxyz':second['tcp_quaternion_wxyz'],
                'second_probe_anchor_local_m':anchor,'second_probe_relative_drift_m':drift,'second_probe_measured_tcp_rise_m':measured}
            if any(not _near(d.get(k),v) for k,v in expected_fields.items()):
                raise ValueError('Second probe origin/motion differs from causal native state')
            unsafe=(not h['history_known'] or state['unexpected_contact_count'] or state['table_object_contact_count']
                    or not all(state['grasp_contact_counts'].values()) or min(state['pad_normal_forces_n'].values())<=.01
                    or drift>acquisition['relative_drift_limit_m'])
            eligible=(stable and d['object_height_delta_m']>=acquisition['probe_min_object_rise_m']
                      and measured>=cfg['probe_minimum_tcp_rise_m'])
            qualify_since=(now if qualify_since is None else qualify_since) if eligible else None
            dwell=0. if qualify_since is None else now-qualify_since
            if 'second_probe_qualified' in types:
                if (unsafe or now-second['t_seconds']>cfg['probe_timeout_s']+1e-9
                        or dwell+1e-9<memory['settled_dwell_s'] or 'carry_resumed' not in types):
                    raise ValueError('Second probe resume lacks original qualification and measured travel')
                if (not _near(d.get('second_probe_qualification_dwell_s'),dwell)
                        or not _near(command['joint_targets_rad'],state['joint_position_rad'])
                        or not _near(d.get('resume_local_offset_m'),contact._local_offset(state))):
                    raise ValueError('Second probe qualification must hold measured pose at transition')
                resume_count+=1
                active_probe=False
            elif 'abort' in types:
                if not _near(command['joint_targets_rad'],state['joint_position_rad']):
                    raise ValueError('Second probe refusal must hold measured joints closed')
                active_probe=False
            else:
                if unsafe or now-second['t_seconds']>cfg['probe_timeout_s']+1e-9 or command['phase']!='memory_second_probe':
                    raise ValueError('Second probe continues without contact or beyond its budget')
                rise=min(cfg['probe_height_m'],cfg['probe_speed_m_s']*(now-second['t_seconds']+raw['config']['control_dt']))
                goal=list(second['tcp_position_m']);goal[2]+=rise
                if (not _near(d.get('second_probe_requested_rise_m'),rise) or not _near(d.get('second_probe_goal_m'),goal)
                        or not _near(d.get('second_probe_qualification_dwell_s'),dwell)):
                    raise ValueError('Second probe target/time/dwell differs from bounded vertical request')
        elif 'second_probe_qualified' in types:
            raise ValueError('Second probe qualification lacks preceding probe')
        if 'carry_resumed' in types or 'abort' in types:
            hold_started,low_since=None,None
    if (observed_events!=len(events) or type(guard.get('second_probe_count')) is not int
            or guard['second_probe_count']!=probe_count or type(guard.get('second_probe_resume_count')) is not int
            or guard['second_probe_resume_count']!=resume_count or probe_count>cfg['maximum_probes']):
        raise ValueError('Second probe event/count mismatch')


def load_reliability_bundle(directory, *, assets=None):
    """Return report and exact sealed compressed bytes after bounded inspection."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Bundle must be regular non-symlink directory')
    integrity = _read(root/'integrity.json',65536)
    seal = _decode(integrity)
    if type(seal) is not dict or set(seal)!={'schema','files'} or seal['schema']!=INTEGRITY_SCHEMA:
        raise ValueError('Invalid grip integrity schema')
    declarations = seal['files']
    if type(declarations) is not dict or len(declarations)!=12*len(VARIANTS)+5:
        raise ValueError('Incomplete paired artifact count')
    static = {'protocol.json','report.json','report.html',base.LICENSE_PATH,base.MODEL_NOTE_PATH}
    files, total, raw_total = {}, 0, 0
    for name, record in declarations.items():
        compressed = type(name) is str and re.fullmatch(r'attempts/[a-z0-9_-]{1,100}\.json\.gz',name) is not None
        if name not in static and not compressed:
            raise ValueError('Artifact path outside allowlist')
        path = root/name
        if any(parent.is_symlink() for parent in path.parents if parent!=root and root in parent.parents):
            raise ValueError('Artifact parent symlink')
        data = _read(path,MAX_FILE_BYTES)
        total += len(data)
        keys = {'bytes','sha256','encoding','decoded_bytes','decoded_sha256'} if compressed else {'bytes','sha256'}
        if (type(record) is not dict or set(record)!=keys or type(record['bytes']) is not int
                or record['bytes']!=len(data) or record['sha256']!=sha256(data).hexdigest()):
            raise ValueError('Artifact compressed digest/size mismatch')
        if total>MAX_BUNDLE_BYTES:
            raise ValueError('Compressed bundle byte bound exceeded')
        files[name] = data
        checked = decode_attempt(data,record) if compressed else data
        raw_total += len(checked)
        if raw_total>MAX_DECODED_BYTES:
            raise ValueError('Decoded bundle byte bound exceeded')
    paths = list(root.rglob('*'))
    if any(p.is_symlink() for p in paths) or {p.relative_to(root).as_posix() for p in paths if p.is_file()}!=set(files)|{'integrity.json'}:
        raise ValueError('Unsealed or symlink bundle inventory')
    if not static<=files.keys():
        raise ValueError('Missing static evidence or license')
    report, protocol = _decode(files['report.json']),_decode(files['protocol.json'])
    if type(report) is not dict or report.get('schema')!=SCHEMA or type(protocol) is not dict or protocol.get('schema')!=PROTOCOL_SCHEMA:
        raise ValueError('Invalid paired report/protocol')
    for doc in (report,protocol):
        if doc.get('profile')!=task.PROFILE or doc.get('physical_robot_ready') is not False or doc.get('learned_policy') is not False:
            raise ValueError('Paired profile/qualification scope mismatch')
    hashes = protocol.get('implementation_sha256')
    if type(hashes) is not dict or set(hashes)!=set(SOURCE_FILES) or any(type(x) is not str or re.fullmatch('[0-9a-f]{64}',x) is None for x in hashes.values()):
        raise ValueError('Malformed frozen source identity')
    if (report.get('implementation_sha256')!=hashes or report.get('protocol')!=protocol
            or report.get('protocol_sha256')!=sha256(files['protocol.json']).hexdigest()
            or report.get('scenes')!=protocol.get('scenes') or report.get('cohort') not in ('development','confirmation')
            or report['cohort']!=protocol.get('cohort') or protocol.get('variants')!=list(VARIANTS)
            or type(protocol.get('planned_attempts')) is not int or protocol['planned_attempts']!=12*len(VARIANTS)
            or protocol.get('native_command_units')!=['rad']*6+['m'] or protocol.get('evaluation_contract')!=EVALUATION_CONTRACT
            or protocol.get('reliability_contract')!=_reliability_contract() or protocol.get('policy_inputs_contract')!=POLICY_INPUTS_CONTRACT
            or protocol.get('recovery_contract')!=_recovery_contract() or protocol.get('contact_contract')!=_memory_contract() or type(protocol.get('contract')) is not dict
            or protocol.get('acquisition_contract')!=_memory_contract()['acquisition']
            or protocol['contract'].get('profile')!=task.PROFILE or protocol['contract'].get('physical_robot_ready') is not False):
        raise ValueError('Frozen paired protocol/units/scope mismatch')
    pinned, notices = base._pinned_assets()
    if (report.get('asset_provenance')!=pinned or pinned['lock_sha256']!=hashes['robot_profiles.lock.json']
            or report.get('simulation_asset_license')!='MIT' or protocol.get('simulation_asset_license')!='MIT'
            or report.get('asset_license')!={'path':base.LICENSE_PATH,'sha256':sha256(files[base.LICENSE_PATH]).hexdigest()}
            or sha256(files[base.LICENSE_PATH]).hexdigest()!=notices['agilex_piper/LICENSE']
            or sha256(files[base.MODEL_NOTE_PATH]).hexdigest()!=notices['agilex_piper/README.md']):
        raise ValueError('Pinned original asset/license mismatch')
    scenes = report['scenes']
    expected_plan=make_protocol(seed=protocol.get('seed'),cohort=protocol['cohort'])
    if scenes!=json.loads(_json_bytes(expected_plan['scenes'])) or protocol.get('layout_offset_limit_m')!=expected_plan['layout_offset_limit_m']:
        raise ValueError('Scenes differ from seeded fixed benchmark plan')
    if type(scenes) is not list or len(scenes)!=12:
        raise ValueError('Contact protocol requires twelve scenes')
    lookup, layouts = {}, {}
    for scene in scenes:
        if (type(scene) is not dict or type(scene.get('id')) is not str or scene['id'] in lookup
                or type(scene.get('layout_index')) is not int or scene['layout_index'] not in (0,1)
                or scene.get('friction') not in FRICTIONS or scene.get('stratum')!=STRATA[FRICTIONS.index(scene['friction'])]
                or scene.get('mass_kg') not in MASSES
                or scene['id']!=f"friction{round(scene['friction']*100):03}-mass{round(scene['mass_kg']*1000):03}-layout{scene['layout_index']:02}"
                or type(scene.get('seed')) is not int or not 0<=scene['seed']<=999999 or type(scene.get('config')) is not dict):
            raise ValueError('Invalid paired scene identity')
        config = task.PiperTaskConfig(**scene['config'])
        if config.friction!=scene['friction'] or config.mass_kg!=scene['mass_kg']:
            raise ValueError('Scene friction disagrees with native config')
        layout = {k:v for k,v in scene['config'].items() if k not in ('friction','mass_kg')}
        if scene['layout_index'] in layouts and layouts[scene['layout_index']]!=layout:
            raise ValueError('Friction strata do not share identical layouts/mechanics')
        layouts[scene['layout_index']] = layout
        lookup[scene['id']] = scene
    if {(s['friction'],s['mass_kg'],s['layout_index']) for s in scenes}!={(f,m,i) for f in FRICTIONS for m in MASSES for i in (0,1)}:
        raise ValueError('Missing planned friction/layout pair')
    expected = {(scene,variant) for scene in lookup for variant in VARIANTS}
    attempts, seen, raws = report.get('attempts'),set(),{}
    if type(attempts) is not list or len(attempts)!=12*len(VARIANTS) or type(report.get('attempted')) is not int or report['attempted']!=12*len(VARIANTS):
        raise ValueError('Missing planned paired attempts')
    for row in attempts:
        if type(row) is not dict or type(row.get('scene_id')) is not str or type(row.get('variant')) is not str:
            raise ValueError('Invalid paired attempt identity')
        identity = row['scene_id'],row['variant']
        if identity not in expected or identity in seen or type(row.get('valid')) is not bool or type(row.get('success')) is not bool or row['success'] and not row['valid']:
            raise ValueError('Duplicate/inconsistent paired attempt')
        scene, (control,mode) = lookup[row['scene_id']],_variant(row['variant'])
        name = row.get('result_file')
        if (row.get('id')!=row['scene_id']+'--'+row['variant'] or name!='attempts/'+row['id']+'.json.gz'
                or name not in files or row.get('control')!=control or row.get('policy_mode')!=mode
                or row.get('profile')!=task.PROFILE or row.get('physical_robot_ready') is not False or row.get('learned_policy') is not False
                or type(row.get('seed')) is not int or row['seed']!=scene['seed'] or row.get('config')!=scene['config']):
            raise ValueError('Attempt differs from paired scene/variant')
        raw = _decode(decode_attempt(files[name],declarations[name]))
        if type(raw) is not dict or _summary(raw)!={k:v for k,v in row.items() if k!='result_file'}:
            raise ValueError('Attempt summary differs from decoded trace')
        if raw['valid'] and raw.get('asset_provenance')!=pinned:
            raise ValueError('Episode asset identity mismatch')
        base._episode(raw,scene,protocol)
        recovery._timing_samples(raw)
        recovery._completion_timestamp(raw)
        _policy_inputs(raw)
        _reliability(raw,protocol)
        if assets is not None:
            audit_policy(raw,assets,protocol)
        seen.add(identity)
        # Retain only paired-comparison fields, not all decoded trajectories.
        comparison = {key:raw.get(key) for key in ('valid','arm_tape','compiled_model_sha256','native_model_settings',
                      'native_control_limits','native_servo_gains','native_servo_bias','native_force_limits','contact_geometry')}
        comparison['has_actions'] = bool(raw['actions'])
        comparison['start'] = raw['trajectory'][0] if raw['trajectory'] else None
        raws[identity] = comparison
    for scene in scenes:
        rows = [raws[scene['id'],variant] for variant in VARIANTS]
        compiled = [r for r in rows if r.get('compiled_model_sha256') is not None]
        if any(type(r['compiled_model_sha256']) is not str or re.fullmatch('[0-9a-f]{64}',r['compiled_model_sha256']) is None for r in compiled):
            raise ValueError('Malformed recorded compiled model identity')
        models = [{key:r.get(key) for key in ('compiled_model_sha256','native_model_settings','native_control_limits',
                   'native_servo_gains','native_servo_bias','native_force_limits','contact_geometry')} for r in compiled]
        if any(model!=models[0] for model in models[1:]):
            raise ValueError('Paired controls use different native mechanics')
        starts = [{k:r['start'][k] for k in ('qpos','qvel','native_actuator_controls')} for r in rows if r['valid']]
        if any(start!=starts[0] for start in starts[1:]):
            raise ValueError('Paired controls start from different native state')
        for placement,opened in ((rows[i],rows[i+1]) for i in range(0,len(rows)-1,2)):
            if opened['has_actions'] and (not placement['valid'] or opened['arm_tape']!=placement['arm_tape'][:len(opened['arm_tape'])]):
                raise ValueError('Open-jaw replay differs from its own placement tape')
        held = rows[-1]
        if held['has_actions'] and (held['start'] is None or any(command!=held['start']['joint_position_rad'] for command in held['arm_tape'])):
            raise ValueError('Common hold moves its initial arm targets')
    if (set(files)!=static|{r['result_file'] for r in attempts} or report.get('summary')!=aggregate(attempts,scenes)
            or report.get('paired_comparisons')!=paired_comparisons(attempts,scenes)
            or report.get('quality_gates')!=quality_gates(attempts,scenes) or report.get('execution_metrics')!=execution_metrics(attempts)
            or type(report.get('valid_attempts')) is not int or report['valid_attempts']!=sum(r['valid'] for r in attempts)):
        raise ValueError('Paired inventory/aggregate denominator mismatch')
    files['integrity.json'] = integrity
    return report,files


def inspect_reliability_bundle(directory, *, assets=None):
    report, files = load_reliability_bundle(directory, assets=assets)
    return {'schema':'dvidia.piper-reliability-inspection/v0.5','scope':INSPECTION_SCOPE, 'native_policy_audit':assets is not None,
            'files_verified':len(files)-1,'attempted':report['attempted'],'valid_attempts':report['valid_attempts'],
            'summary':report['summary'],'physical_robot_ready':False,'learned_policy':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    run = commands.add_parser('benchmark')
    run.add_argument('--assets',required=True,type=Path)
    run.add_argument('--output',required=True,type=Path)
    run.add_argument('--seed',default=950031,type=int)
    run.add_argument('--cohort',choices=('development','confirmation'),default='development')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('bundle',type=Path)
    inspect.add_argument('--assets',type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command=='benchmark':
            report = run_benchmark(args.assets,args.output,seed=args.seed,cohort=args.cohort,
                                   progress=lambda row:print(json.dumps(row),file=sys.stderr,flush=True))
            result = {'output':str(args.output),'attempted':report['attempted'],'summary':report['summary']}
        else:
            result = inspect_reliability_bundle(args.bundle,assets=args.assets)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError,RuntimeError,TypeError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':
    main()
