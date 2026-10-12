"""Paired PiPER v0.4 recovery receipts, with original v0.3 history as control.

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
from . import piper_contact_benchmark as contact, robot_profiles
from .reference_replay import _decode, _json_bytes, _read

SCHEMA = 'dvidia.piper-recovery-benchmark/v0.4'
PROTOCOL_SCHEMA = 'dvidia.piper-recovery-protocol/v0.4'
INTEGRITY_SCHEMA = 'dvidia.piper-recovery-integrity/v0.4'
VARIANTS = ('history_placement', 'history_open_jaw', 'recovery_carry_placement', 'recovery_carry_open_jaw', 'recovery_placement', 'recovery_open_jaw', 'hold')
MODES = ('history', 'recovery_carry', 'recovery')
MASSES, FRICTIONS, STRATA = contact.MASSES, contact.FRICTIONS, contact.STRATA
SOURCE_FILES = contact.SOURCE_FILES + ('piper_contact_recovery.py', 'piper_recovery_benchmark.py')
MAX_FILE_BYTES, MAX_BUNDLE_BYTES, MAX_DECODED_BYTES = contact.MAX_FILE_BYTES, contact.MAX_BUNDLE_BYTES, contact.MAX_DECODED_BYTES
SCOPE = ('Paired authored PiPER recovery pilot, privileged native state and synthetic rigid blocks. '
         'No learned policy, sensor pipeline, calibrated contact, shoe competence or physical robot qualification.')
EVALUATION_CONTRACT = {'schema':'dvidia.piper-recovery-sampled-evaluation/v0.4',
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
    mode = next(m for m in ('recovery_carry','history','recovery') if variant.startswith(m+'_'))
    return ('replay_open_jaw' if variant.endswith('open_jaw') else 'placement'),mode


def make_protocol(*, seed=914031, cohort='development'):
    protocol = contact.make_protocol(seed=seed,cohort=cohort)
    protocol.update(schema=PROTOCOL_SCHEMA, scope=SCOPE, variants=list(VARIANTS),
        implementation_sha256=implementation_hashes(), recovery_contract=_recovery_contract(),
        evaluation_contract=EVALUATION_CONTRACT,
        comparison='Identical native mechanics, initial state and full twenty-second duration per scene. '
            'Frozen v0.3 history versus v0.4 carry-only with old acquisition versus full v0.4 recovery. '
            'Each placement has its own complete arm-tape/open-jaw control, plus a shared initial-arm hold. '
            'The carry-only ablation isolates carry changes from the bounded acquisition decision. '
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
                    'full_speed_restored':sum(e['full_speed_restored'] for e in evaluations)})
    return rows


def paired_comparisons(attempts,scenes):
    lookup={(r['scene_id'],r['variant']):r for r in attempts}
    comparisons=[]
    for reference,candidate in (('history','recovery_carry'),('history','recovery'),('recovery_carry','recovery')):
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


def _html(report):
    rows=''.join('<tr>'+''.join('<td>'+escape(str(row[k]))+'</td>' for k in ('friction','mass_kg','variant','success','attempted','valid','retries','holds','unresolved_holds'))+'</tr>' for row in report['summary'])
    attempts=''.join('<li><a href="'+escape(r['result_file'],quote=True)+'">'+escape(r['id'])+'</a>: '+('completed' if r['valid'] and r['success'] else 'failed' if r['valid'] else 'invalid')+'</li>' for r in report['attempts'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>DVIDIA · PiPER recovery v0.4</title><style>body{max-width:1100px;margin:auto;padding:24px;font:18px/1.6 Georgia;background:#fafaf7;color:#292d27}'
        'table{font:14px/1.5 system-ui;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}.scroll{overflow:auto}a{color:#396751}</style>'
        '<main><p>DVIDIA RESEARCH</p><h1>Measured grip recovery · v0.4</h1><p>'+escape(SCOPE)+'</p><p>All 84 planned attempts retained.</p>'
        '<p><a href="report.json">Complete report and paired timing</a> · <a href="protocol.json">Frozen protocol</a> · <a href="integrity.json">Integrity</a></p>'
        '<div class="scroll"><table><tr><th>Friction</th><th>Mass kg</th><th>Variant</th><th>Completed</th><th>Attempted</th><th>Valid</th><th>Retries</th><th>Holds</th><th>Unresolved</th></tr>'+rows+'</table></div>'
        '<p>'+escape(EVALUATION_CONTRACT['comparison'])+'</p><p>'+escape(base.TIMING_BOUNDARY)+'</p><h2>Every attempt</h2><ul>'+attempts+'</ul>'
        '<footer><a href="https://research.dvidia.org/">DVIDIA research</a> · <a href="https://x.com/iammrriver">@iammrriver</a> · <a href="mailto:hello@dvidia.org">hello@dvidia.org</a></footer></main></html>')


def run_benchmark(assets, output, *, seed=914031, cohort='development', progress=None):
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
                                       control=control, policy_mode=mode, record_step_timings=True, arm_tape=tapes.get(mode) if control == 'replay_open_jaw' else None)
                if type(raw.get('valid')) is not bool or type(raw.get('success')) is not bool:
                    raise ValueError('Task runner omitted Boolean outcomes')
                if control == 'placement':
                    tape = raw.get('arm_tape')
                    ticks = round(scene['config']['timeout_s']/scene['config']['control_dt'])
                    tapes[mode] = tape if raw['valid'] and type(tape) is list and len(tape) == ticks else None
            except Exception as exc:
                from . import piper_grip_guard, piper_contact_memory, piper_contact_recovery
                inactive = (piper_grip_guard.inactive_summary(mode) if mode=='baseline'
                            else piper_contact_memory.inactive_summary(mode) if mode=='history' else piper_contact_recovery.inactive_summary(mode))
                policy_sources = ('piper_grip_guard.py',) if mode=='baseline' else ('piper_grip_guard.py','piper_contact_memory.py') + (() if mode=='history' else ('piper_contact_recovery.py',))
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
              'summary':aggregate(attempts,protocol['scenes']),'paired_comparisons':paired_comparisons(attempts,protocol['scenes']),'attempts':attempts}
    base._write(destination/'report.json',report)
    (destination/'report.html').write_text(_html(report),encoding='utf-8',newline='\n')
    for name in ('protocol.json','report.json','report.html',base.LICENSE_PATH,base.MODEL_NOTE_PATH):
        data = (destination/name).read_bytes()
        records[name] = {'bytes':len(data),'sha256':sha256(data).hexdigest()}
    base._write(destination/'integrity.json',{'schema':INTEGRITY_SCHEMA,'files':records})
    inspect_recovery_bundle(destination)
    return report



def _acquisition_values(rows, parameters):
    span=rows[-1]['t_seconds']-rows[0]['t_seconds'] if rows else 0.
    known=span+1e-9>=parameters['acquisition_minimum_span_s']
    jaw=max((max(s['gripper_joint_position_m'][i] for s in rows)-min(s['gripper_joint_position_m'][i] for s in rows) for i in range(2)),default=0.) if rows else None
    forces={side:max(s['pad_normal_forces_n'][side] for s in rows)-min(s['pad_normal_forces_n'][side] for s in rows) for side in ('left','right')} if rows else None
    stable=(known and all(all(s['grasp_contact_counts'].values()) and min(s['pad_normal_forces_n'].values())>.01
        and s['table_object_contact_count']>0 and s['_previous_jaw_request']==0. for s in rows)
        and jaw<=parameters['acquisition_jaw_excursion_m'] and max(forces.values())<=parameters['acquisition_normal_excursion_n'])
    return {'acquisition_window_count':len(rows),'acquisition_window_span_s':span,'acquisition_window_known':known,
        'acquisition_jaw_excursion_m':jaw,'acquisition_normal_excursion_n':forces,
        'acquisition_stable_closed_supported':stable}


def _recovery(raw,protocol):
    """Validate new causal decisions, then delegate unchanged gates to v0.3.

    The compatibility view is created only after original requested targets and
    every variable scale have been checked. It removes v0.4-only events and maps
    verified scale back to v0.3's fixed carry scale for that frozen inspector.
    It never changes the receipt or substitutes derived states for native data.
    """
    mode,guard=raw['policy_mode'],raw.get('grip_guard')
    if mode in ('history','baseline'):
        view=dict(raw,contact_evaluation=contact.evaluate_episode(raw))
        contact._memory(view,protocol)
    else:
        from .piper_contact_recovery import SCHEMA as policy_schema
        names=('piper_grip_guard.py','piper_contact_memory.py','piper_contact_recovery.py')
        if (mode not in ('recovery_carry','recovery') or type(guard) is not dict
                or guard.get('schema')!=policy_schema or guard.get('mode')!=mode
                or guard.get('contract')!=protocol['recovery_contract']
                or raw.get('policy_implementation_sha256')!={n:protocol['implementation_sha256'][n] for n in names}):
            raise ValueError('Recovery mode/contract/source mismatch')
        events=guard.get('events')
        if type(events) is not list or len(events)>100 or any(type(e) is not dict for e in events):
            raise ValueError('Invalid bounded recovery event list')
        new_types={'unchanged_retry_refused','speed_restoration_started','full_speed_restored'}
        by_time={}
        for e in events:
            clock=base._numeric(e.get('t_seconds'),'recovery event time',0.,raw['simulated_seconds']+1e-8)
            by_time.setdefault(round(clock,8),[]).append(e)
        for field,kind in (('restoration_started_count','speed_restoration_started'),('full_speed_restored_count','full_speed_restored')):
            if type(guard.get(field)) is not int or not 0<=guard[field]<=2 or guard[field]!=sum(e['type']==kind for e in events):
                raise ValueError('Restoration count mismatch')
        refused=sum(e['type']=='unchanged_retry_refused' for e in events)
        if type(guard.get('unchanged_retry_refused')) is not bool or refused>1 or guard['unchanged_retry_refused']!=(refused==1) or mode=='recovery_carry' and refused:
            raise ValueError('Unsupported retry refusal or mode mismatch')
        if not guard.get('active') and (refused or guard['restoration_started_count'] or guard['full_speed_restored_count']):
            raise ValueError('Inactive recovery contains decisions')
        cfg,memory=protocol['recovery_contract']['parameters'],protocol['contact_contract']['parameters']
        rows,previous_jaw,origin=[],None,None
        resumed,started,complete=False,False,False
        stable_since,last_time,progress=None,None,0.
        seen_new=0
        view=deepcopy(raw)
        view['policy_mode']='history'
        view['policy_implementation_sha256']={n:protocol['implementation_sha256'][n] for n in names[:2]}
        view['grip_guard'].update(schema='dvidia.piper-contact-memory/v0.3',mode='history',contract=protocol['contact_contract'],events=[deepcopy(e) for e in events if e['type'] not in new_types])
        for index,state in enumerate(raw['trajectory']):
            command=state.get('command')
            current=by_time.get(round(state['t_seconds'],8),[])
            if command is None:
                if any(e['type'] in new_types for e in current):
                    raise ValueError('Unissued terminal recovery decision')
                continue
            if not guard['active']:
                continue
            d,clock=state.get('guard_diagnostic'),state['t_seconds']
            if type(d) is not dict:
                raise ValueError('Missing recovery diagnostic')
            if d.get('phase') in ('probe','close'):
                rows.append(dict(state,_previous_jaw_request=previous_jaw))
                rows=[s for s in rows if s['t_seconds']>=clock-cfg['acquisition_window_s']-1e-9]
            else:
                rows=[]
            acquisition=_acquisition_values(rows,cfg)
            for key,value in acquisition.items():
                if not _near(d.get(key),value) or any(not _near(e.get('diagnostic',{}).get(key),value) for e in current):
                    raise ValueError('Acquisition window differs from causal native observations: '+key)
            if any(e['type']=='probe_started' for e in current):
                origin=state
            if any(e['type']=='reassessment_started' for e in current) or any(e['type']=='abort' and str(e.get('reason','')).startswith('reassessment_budget_exhausted:') for e in current):
                stable_since,last_time,progress,started,complete=None,None,0.,False,False
            if any(e['type']=='carry_resumed' for e in current):
                resumed=True
            refusal=[e for e in current if e['type']=='unchanged_retry_refused']
            if refusal:
                if (mode!='recovery' or d.get('phase')!='probe' or not acquisition['acquisition_stable_closed_supported']
                        or state['unexpected_contact_count'] or origin is None or index<1
                        or d.get('retry_index')!=0
                        or command['joint_targets_rad']!=state['joint_position_rad'] or command['gripper_position_m']!=0.
                        or not any(e['type']=='alarm' and e['reason']=='object_not_following' for e in current)
                        or not any(e['type']=='abort' and e['reason']=='unchanged_grasp_not_following' for e in current)
                        or any(e['type']=='retry_started' for e in current)):
                    raise ValueError('Retry refusal lacks supported unchanged-grasp evidence')
                for s in raw['trajectory'][index-1:index+1]:
                    tcp_rise=s['tcp_position_m'][2]-origin['tcp_position_m'][2]
                    object_rise=s['object_position_m'][2]-origin['object_position_m'][2]
                    if (s['t_seconds']<origin['t_seconds'] or not s['table_object_contact_count']
                            or tcp_rise<=protocol['acquisition_contract']['parameters']['monitor_min_probe_motion_m'] or object_rise>=.25*tcp_rise):
                        raise ValueError('Retry refusal lacks two causal no-following observations')
            expected_events=set()
            unscaled=d.get('unscaled_joint_targets_rad')
            restoring=unscaled is not None and resumed and command['phase']!='memory_slowdown'
            if restoring:
                stable=(d['history_known'] and d['history_displacement_m']<=memory['settled_displacement_m']
                    and d['history_excursion_m']<=memory['settled_displacement_m'] and d['history_rate_m_s']<=memory['settled_rate_m_s']
                    and d['history_loaded_bilateral_all'] and d['history_unsupported_all']
                    and d['object_height_delta_m']>=protocol['acquisition_contract']['parameters']['probe_min_object_rise_m'])
                elapsed=0. if last_time is None else max(0.,clock-last_time)
                last_time=clock
                dwell=0.
                if not started:
                    stable_since=(clock if stable_since is None else stable_since) if stable else None
                    dwell=0. if stable_since is None else clock-stable_since
                    if dwell+1e-9>=cfg['restoration_dwell_s']:
                        started=True
                        expected_events.add('speed_restoration_started')
                elif stable and not complete:
                    progress=min(cfg['restoration_ramp_s'],progress+elapsed)
                    if progress+1e-9>=cfg['restoration_ramp_s']:
                        complete=True
                        expected_events.add('full_speed_restored')
                scale=memory['resumed_joint_delta_scale']+(1-memory['resumed_joint_delta_scale'])*progress/cfg['restoration_ramp_s']
                expected={'restoration_stable_dwell_seconds':dwell,'restoration_progress_seconds':progress,
                    'restoration_started':started,'restoration_complete':complete,'recovery_scale':scale,'joint_delta_scale':scale}
                if any(not _near(d.get(k),v) for k,v in expected.items()):
                    raise ValueError('Restoration scale/dwell differs from causal stable history')
                base._vector(unscaled,6,'unscaled recovery teacher targets')
                target=[q+scale*(u-q) for q,u in zip(state['joint_position_rad'],unscaled)]
                if not _near(command['joint_targets_rad'],target):
                    raise ValueError('Recovery command differs from verified scaled teacher target')
                # Unchanged gate inspection uses a view of the verified command.
                projected=view['trajectory'][index]
                projected['guard_diagnostic']['joint_delta_scale']=memory['resumed_joint_delta_scale']
                projected['command']['joint_targets_rad']=[q+memory['resumed_joint_delta_scale']*(u-q) for q,u in zip(state['joint_position_rad'],unscaled)]
            observed={e['type'] for e in current if e['type'] in ('speed_restoration_started','full_speed_restored')}
            if observed!=expected_events or len(observed)!=sum(e['type'] in observed for e in current):
                raise ValueError('Restoration event lacks its causal transition')
            for event in current:
                if event['type'] not in new_types:
                    continue
                seen_new+=1
                if (type(event.get('diagnostic')) is not dict or event['diagnostic'].get('t_seconds')!=clock
                        or event['diagnostic'].get('bilateral_contact')!=all(state['grasp_contact_counts'].values())
                        or event['diagnostic'].get('table_supported')!=(state['table_object_contact_count']>0)
                        or type(event.get('retry_index')) is not int or event['retry_index']!=d.get('retry_index')
                        or type(event.get('reason')) is not str
                        or event.get('phase')!=('guard_stop' if event['type']=='unchanged_retry_refused' else command['phase'])):
                    raise ValueError('New recovery event differs from causal issued observation')
            previous_jaw=command['gripper_position_m']
        if seen_new!=sum(e['type'] in new_types for e in events):
            raise ValueError('Recovery event is outside recorded causal boundaries')
        view['contact_evaluation']=contact.evaluate_episode(view)
        contact._memory(view,protocol)
    if raw.get('recovery_evaluation')!=evaluate_episode(raw):
        raise ValueError('Recovery evaluation differs from native recorded outcomes/motion')


def _timing_samples(raw):
    values=raw.get('timing',{}).get('control_step_samples_seconds')
    if type(values) is not list or len(values)>len(raw['actions']) or raw['valid'] and len(values)!=len(raw['actions']):
        raise ValueError('Missing or mismatched raw control computation windows')
    for value in values:
        base._numeric(value,'control computation window')
    if values and (raw['timing'].get('control_step_seconds')!=task.sim._summary(values)
            or raw['timing'].get('missed_control_boundaries')!=sum(v>raw['config']['control_dt'] for v in values)):
        raise ValueError('Control computation summary differs from raw windows')


def load_recovery_bundle(directory):
    """Return report and exact sealed compressed bytes after bounded inspection."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Bundle must be regular non-symlink directory')
    integrity = _read(root/'integrity.json',65536)
    seal = _decode(integrity)
    if type(seal) is not dict or set(seal)!={'schema','files'} or seal['schema']!=INTEGRITY_SCHEMA:
        raise ValueError('Invalid grip integrity schema')
    declarations = seal['files']
    if type(declarations) is not dict or len(declarations)!=89:
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
            or type(protocol.get('planned_attempts')) is not int or protocol['planned_attempts']!=84
            or protocol.get('native_command_units')!=['rad']*6+['m'] or protocol.get('evaluation_contract')!=EVALUATION_CONTRACT
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
    if scenes!=expected_plan['scenes'] or protocol.get('layout_offset_limit_m')!=expected_plan['layout_offset_limit_m']:
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
    if type(attempts) is not list or len(attempts)!=84 or type(report.get('attempted')) is not int or report['attempted']!=84:
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
        _timing_samples(raw)
        _recovery(raw,protocol)
        seen.add(identity)
        # Retain only paired-comparison fields, not all 84 decoded trajectories.
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
        for placement,opened in ((rows[0],rows[1]),(rows[2],rows[3]),(rows[4],rows[5])):
            if opened['has_actions'] and (not placement['valid'] or opened['arm_tape']!=placement['arm_tape'][:len(opened['arm_tape'])]):
                raise ValueError('Open-jaw replay differs from its own placement tape')
        held = rows[6]
        if held['has_actions'] and (held['start'] is None or any(command!=held['start']['joint_position_rad'] for command in held['arm_tape'])):
            raise ValueError('Common hold moves its initial arm targets')
    if (set(files)!=static|{r['result_file'] for r in attempts} or report.get('summary')!=aggregate(attempts,scenes)
            or report.get('paired_comparisons')!=paired_comparisons(attempts,scenes)
            or type(report.get('valid_attempts')) is not int or report['valid_attempts']!=sum(r['valid'] for r in attempts)):
        raise ValueError('Paired inventory/aggregate denominator mismatch')
    files['integrity.json'] = integrity
    return report,files


def inspect_recovery_bundle(directory):
    report, files = load_recovery_bundle(directory)
    return {'schema':'dvidia.piper-recovery-inspection/v0.4','scope':'Recorded consistency, not authenticity or physics recomputation',
            'files_verified':len(files)-1,'attempted':report['attempted'],'valid_attempts':report['valid_attempts'],
            'summary':report['summary'],'physical_robot_ready':False,'learned_policy':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    run = commands.add_parser('benchmark')
    run.add_argument('--assets',required=True,type=Path)
    run.add_argument('--output',required=True,type=Path)
    run.add_argument('--seed',default=914031,type=int)
    run.add_argument('--cohort',choices=('development','confirmation'),default='development')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('bundle',type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command=='benchmark':
            report = run_benchmark(args.assets,args.output,seed=args.seed,cohort=args.cohort,
                                   progress=lambda row:print(json.dumps(row),file=sys.stderr,flush=True))
            result = {'output':str(args.output),'attempted':report['attempted'],'summary':report['summary']}
        else:
            result = inspect_recovery_bundle(args.bundle)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError,RuntimeError,TypeError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':
    main()
