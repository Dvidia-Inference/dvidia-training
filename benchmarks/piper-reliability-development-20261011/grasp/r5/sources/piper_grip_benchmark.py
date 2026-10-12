"""Frozen paired PiPER grip-guard experiments and bounded offline inspection.

Recorded consistency and independent sampled outcome proxies are inspected;
neither a file seal nor these synthetic trials authenticate physical competence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
import zlib

from . import piper_task as task, piper_task_benchmark as base, robot_profiles
from .reference_replay import _decode, _json_bytes, _read

SCHEMA = 'dvidia.piper-grip-benchmark/v1'
PROTOCOL_SCHEMA = 'dvidia.piper-grip-protocol/v1'
INTEGRITY_SCHEMA = 'dvidia.piper-grip-integrity/v1'
VARIANTS = ('baseline_placement', 'baseline_open_jaw', 'guarded_placement', 'guarded_open_jaw', 'hold')
FRICTIONS = (.8, .55, .15)
STRATA = ('nominal', 'moderate_friction', 'low_friction')
SOURCE_FILES = base.SOURCE_FILES + ('piper_grip_guard.py', 'piper_grip_benchmark.py')
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_DECODED_BYTES = 512 * 1024 * 1024
SCOPE = ('Paired authored PiPER grip-guard pilot, privileged native state and synthetic rigid blocks. '
         'No learned policy, sensor pipeline, calibrated contact, shoe competence or physical robot qualification.')
EVALUATION_CONTRACT = {
    'schema': 'dvidia.piper-grip-sampled-evaluation/v1',
    'following_rise_m': .010, 'following_dwell_s': .100,
    'qualification_lookahead_s': .600, 'local_offset_drift_limit_m': .008,
    'post_alarm_window_s': .100,
    'scope': 'Independent recorded 20ms boundary proxies, not recomputed native physics or classifier accuracy. '
             'Following requires table-free object rise and stable TCP-local object offset, without controller predicates. '
             'Unresolved decisions and absent opportunities remain explicit. Alarm delays use simulator timestamps; '
             'wall computation is reported separately. Post-alarm path includes native servo motion while holding.',
}


def implementation_hashes():
    root = Path(__file__).parent
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _guard_contract():
    from .piper_grip_guard import guard_contract
    return guard_contract()


def _variant(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown paired variant')
    return ('hold', 'baseline') if variant == 'hold' else (
        'replay_open_jaw' if variant.endswith('open_jaw') else 'placement',
        'guarded' if variant.startswith('guarded') else 'baseline')


def make_protocol(*, seed=712047, cohort='development'):
    if type(seed) is not int or not 0 <= seed <= 999994:
        raise ValueError('Seed must be an integer in [0,999994]')
    if cohort not in ('development', 'confirmation'):
        raise ValueError('Cohort must be development or confirmation')
    default, rng, layouts = task.PiperTaskConfig(), random.Random(seed), []
    for index in range(2):
        start, target = list(default.object_position_m), list(default.target_position_m)
        for axis in (0, 1):
            start[axis] += rng.uniform(-.005, .005)
            target[axis] += rng.uniform(-.005, .005)
        layouts.append((start, target))
    scenes = []
    for friction, stratum in zip(FRICTIONS, STRATA):
        for index, (start, target) in enumerate(layouts):
            config = replace(default, object_position_m=tuple(start), target_position_m=tuple(target), friction=friction)
            scenes.append({'id': f'friction{round(friction*100):03}-layout{index:02}', 'layout_index': index,
                           'stratum': stratum, 'friction': friction, 'seed': seed+len(scenes), 'config': asdict(config)})
    return {'schema': PROTOCOL_SCHEMA, 'profile': task.PROFILE, 'scope': SCOPE, 'physical_robot_ready': False,
            'learned_policy': False, 'cohort': cohort, 'seed': seed, 'scenes': scenes, 'variants': list(VARIANTS),
            'planned_attempts': 30, 'contract': task.task_contract(), 'guard_contract': _guard_contract(),
            'evaluation_contract': EVALUATION_CONTRACT, 'implementation_sha256': implementation_hashes(),
            'native_command_units': ['rad']*6+['m'], 'simulation_asset_license': 'MIT', 'layout_offset_limit_m': .005,
            'timing_boundary': base.TIMING_BOUNDARY,
            'comparison': 'Two shared XY layouts at each friction; identical native model per scene. Each placement has '
                          'a separately matched open-jaw replay of its own complete valid requested-arm tape; common hold '
                          'uses initial arm targets. Aborts retain the full timeout unless native physics becomes invalid. '
                          'Invalid/incomplete placement leaves its dependent replay invalid; no tape extension.',
            'split_note': 'Synthetic layouts; confirmation is a declaration, not proof of unseen scenes. '
                          'Controller and evaluation thresholds are frozen before running, not accuracy estimates.'}


def decode_attempt(data, record=None):
    """Bounded single-member gzip decoding; optionally verify its raw-byte seal."""
    if type(data) is not bytes or len(data) > MAX_FILE_BYTES:
        raise ValueError('Compressed attempt exceeds byte bound')
    decoder = zlib.decompressobj(16+zlib.MAX_WBITS)
    try:
        decoded = decoder.decompress(data, MAX_FILE_BYTES+1)
        if len(decoded) > MAX_FILE_BYTES or decoder.unconsumed_tail:
            raise ValueError('Decoded attempt exceeds byte bound')
        decoded += decoder.flush(MAX_FILE_BYTES+1-len(decoded))
    except zlib.error as exc:
        raise ValueError('Invalid gzip attempt') from exc
    if len(decoded) > MAX_FILE_BYTES or not decoder.eof or decoder.unused_data:
        raise ValueError('Truncated, concatenated or oversized gzip attempt')
    if record is not None and (record.get('encoding') != 'gzip' or type(record.get('decoded_bytes')) is not int
                               or record['decoded_bytes'] != len(decoded)
                               or record.get('decoded_sha256') != sha256(decoded).hexdigest()):
        raise ValueError('Decoded attempt digest/size mismatch')
    return decoded


def _local_offset(state):
    # Inverse unit quaternion rotates the object-to-TCP vector into the TCP frame.
    w, x, y, z = state['tcp_quaternion_wxyz']
    v = [a-b for a, b in zip(state['object_position_m'], state['tcp_position_m'])]
    u = (-x, -y, -z)
    cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    second = (u[1]*cross[2]-u[2]*cross[1], u[2]*cross[0]-u[0]*cross[2], u[0]*cross[1]-u[1]*cross[0])
    return [v[i]+2*w*cross[i]+2*second[i] for i in range(3)]


def evaluate_episode(raw):
    """Derive future following and postalarm motion independently from guard code."""
    states, config = raw['trajectory'], raw['config']
    events = raw.get('grip_guard', {}).get('events', [])
    dt, spec = config['control_dt'], EVALUATION_CONTRACT
    if not states:
        return {'schema': spec['schema'], 'qualification_decisions': [], 'alarms': [],
                'following_opportunity': None, 'first_following_time_s': None,
                'sampled_drop_events': 0, 'aborted': raw.get('grip_guard', {}).get('aborted', False),
                'retry_count': raw.get('grip_guard', {}).get('retry_count', 0), 'no_recorded_observation': True}
    initial_z = states[0]['object_position_m'][2]
    qualifies, streak, first = [], [], None
    for state in states:
        good = state['table_object_contact_count'] == 0 and state['object_position_m'][2]-initial_z >= spec['following_rise_m']
        if good and streak and math.dist(_local_offset(state), _local_offset(streak[0])) > spec['local_offset_drift_limit_m']:
            streak = []
        streak = streak+[state] if good else []
        ok = bool(streak and state['t_seconds']-streak[0]['t_seconds']+1e-9 >= spec['following_dwell_s'])
        qualifies.append(ok)
        if ok and first is None:
            first = state['t_seconds']
    decisions = []
    for event in events:
        if event['type'] != 'acquisition_qualified':
            continue
        at, end = event['t_seconds'], event['t_seconds']+spec['qualification_lookahead_s']
        opportunity = any(ok and at-1e-9 <= state['t_seconds'] <= end+1e-9 for state, ok in zip(states, qualifies))
        complete = states[-1]['t_seconds']+1e-9 >= end
        decisions.append({'t_seconds': at, 'following_observed': True if opportunity else False if complete else None,
                          'lookahead_complete': complete})
    # Loss proxy after an independently observed following window, before intentional placement.
    losses, was_following, since = [], False, None
    for state, ok in zip(states, qualifies):
        phase = (state.get('command') or {}).get('phase')
        if phase in ('place', 'release', 'retreat', 'verify'):
            was_following, since = False, None
            continue
        if ok:
            was_following, since = True, None
        elif was_following:
            since = state['t_seconds'] if since is None else since
            if state['t_seconds']-since+1e-9 >= spec['following_dwell_s']:
                losses.append(since)
                was_following, since = False, None
    alarms = []
    for event in events:
        if event['type'] != 'alarm':
            continue
        at, end = event['t_seconds'], event['t_seconds']+spec['post_alarm_window_s']
        window = [state for state in states if at-1e-9 <= state['t_seconds'] <= end+1e-9]
        complete = bool(window and window[0]['t_seconds'] <= at+dt/2 and window[-1]['t_seconds']+1e-9 >= end)
        earlier = [loss for loss in losses if loss <= at+1e-9]
        alarms.append({'t_seconds': at, 'reason': event.get('reason'),
                       'sampled_loss_before_alarm_s': earlier[-1] if earlier else None,
                       'sampled_loss_to_alarm_delay_s': at-earlier[-1] if earlier else None,
                       'no_preceding_sampled_loss_opportunity': not bool(earlier),
                       'post_alarm_window_complete': complete,
                       'tcp_path_m': sum(math.dist(a['tcp_position_m'], b['tcp_position_m']) for a,b in zip(window,window[1:])),
                       'tcp_rise_m': max((s['tcp_position_m'][2]-window[0]['tcp_position_m'][2] for s in window), default=0.),
                       'object_rise_m': max((s['object_position_m'][2]-window[0]['object_position_m'][2] for s in window), default=0.)})
    return {'schema': spec['schema'], 'qualification_decisions': decisions, 'alarms': alarms,
            'following_opportunity': first is not None, 'first_following_time_s': first,
            'sampled_drop_events': len(losses), 'aborted': raw.get('grip_guard', {}).get('aborted', False),
            'retry_count': raw.get('grip_guard', {}).get('retry_count', 0), 'no_recorded_observation': False}


def aggregate(attempts, scenes):
    result = []
    for friction, stratum in zip(FRICTIONS, STRATA):
        for variant in VARIANTS:
            rows = [row for row in attempts if row['variant'] == variant and row['config']['friction'] == friction]
            success = sum(row['valid'] is True and row['success'] is True for row in rows)
            decisions = [decision for row in rows for decision in row['grip_evaluation']['qualification_decisions']]
            alarms = [alarm for row in rows for alarm in row['grip_evaluation']['alarms']]
            result.append({'friction': friction, 'stratum': stratum, 'variant': variant, 'attempted': len(rows),
                           'valid': sum(row['valid'] is True for row in rows), 'success': success,
                           'failed_or_invalid': len(rows)-success, 'success_fraction': success/len(rows) if rows else None,
                           'qualification_decisions': len(decisions),
                           'following_observed_after_decision': sum(x['following_observed'] is True for x in decisions),
                           'following_not_observed_after_decision': sum(x['following_observed'] is False for x in decisions),
                           'unresolved_qualification_decisions': sum(x['following_observed'] is None for x in decisions),
                           'no_qualification_decision_attempts': sum(not row['grip_evaluation']['qualification_decisions'] for row in rows),
                           'no_following_opportunity_attempts': sum(row['grip_evaluation']['following_opportunity'] is False for row in rows),
                           'unresolved_following_opportunity_attempts': sum(row['grip_evaluation']['following_opportunity'] is None for row in rows),
                           'aborted_attempts': sum(row['grip_evaluation']['aborted'] for row in rows),
                           'retries': sum(row['grip_evaluation']['retry_count'] for row in rows), 'alarms': len(alarms),
                           'alarms_without_preceding_sampled_loss': sum(x['no_preceding_sampled_loss_opportunity'] for x in alarms),
                           'incomplete_post_alarm_windows': sum(not x['post_alarm_window_complete'] for x in alarms),
                           'sampled_drop_events': sum(row['grip_evaluation']['sampled_drop_events'] for row in rows)})
    return result


def _summary(raw):
    return base._summary(raw)


def _html(report):
    summary = ''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in
                     (r['friction'],r['variant'],r['success'],r['attempted'],r['valid'],r['aborted_attempts'],r['retries']))+'</tr>'
                      for r in report['summary'])
    attempts = []
    for row in report['attempts']:
        guard, evaluation = row['grip_guard'], row['grip_evaluation']
        decisions = evaluation['qualification_decisions']
        decision_text = ', '.join(f"{d['t_seconds']:.2f}s: "+('following observed' if d['following_observed'] is True else
                                  'following not observed' if d['following_observed'] is False else 'unresolved') for d in decisions) or 'no qualification decision'
        event_text = ', '.join(f"{e['t_seconds']:.2f}s {e['type'].replace('_',' ')} ({e['reason']})" for e in guard['events']) or 'no guard events'
        motion = ', '.join(f"{a['t_seconds']:.2f}s alarm: TCP path {a['tcp_path_m']*1000:.2f}mm, rise {a['tcp_rise_m']*1000:.2f}mm; "+
                           ('complete 100ms window' if a['post_alarm_window_complete'] else 'incomplete window') for a in evaluation['alarms'])
        attempts.append('<li><strong>'+escape(row['id'])+'</strong>: '+('completed' if row['success'] else 'failed' if row['valid'] else 'invalid')+
                        ' · '+escape(decision_text)+'. Retries: '+str(guard['retry_count'])+'. '+
                        ('Abort: '+escape(str(guard['abort_reason']))+'. ' if guard['aborted'] else '')+
                        '<br>'+escape(event_text)+(('<br>'+escape(motion)) if motion else '')+
                        '<br><a href="'+escape(row['result_file'],quote=True)+'">complete JSON gzip</a></li>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>DVIDIA · PiPER paired grip pilot</title><style>body{max-width:1100px;margin:auto;padding:24px;font:18px/1.6 Georgia;'
            'background:#fafaf7;color:#292d27}table{font:14px/1.5 system-ui;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}'
            '.scroll{overflow:auto}a{color:#396751}</style><main><p>DVIDIA RESEARCH</p><h1>PiPER paired grip pilot</h1><p>'+escape(SCOPE)+
            '</p><p>'+escape(report['cohort'])+' · all 30 planned attempts retained.</p><p><a href="report.json">Report</a> · '
            '<a href="protocol.json">Frozen protocol</a> · <a href="integrity.json">Integrity and decoded hashes</a> · '
            '<a href="'+base.LICENSE_PATH+'">Original MIT model notice</a></p><div class="scroll"><table><tr><th>Friction</th><th>Variant</th>'
            '<th>Completed</th><th>Attempted</th><th>Valid</th><th>Aborted</th><th>Retries</th></tr>'+summary+'</table></div><p>'+
            escape(EVALUATION_CONTRACT['scope'])+'</p><p>'+escape(base.TIMING_BOUNDARY)+'</p><h2>All complete traces</h2><ul>'+''.join(attempts)+
            '</ul><footer><a href="https://research.dvidia.org/">DVIDIA research</a> · <a href="https://x.com/iammrriver">@iammrriver</a> · '
            '<a href="mailto:hello@dvidia.org">hello@dvidia.org</a></footer></main></html>')


def run_benchmark(assets, output, *, seed=712047, cohort='development', progress=None):
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
                                       control=control, policy_mode=mode, arm_tape=tapes.get(mode) if control == 'replay_open_jaw' else None)
                if type(raw.get('valid')) is not bool or type(raw.get('success')) is not bool:
                    raise ValueError('Task runner omitted Boolean outcomes')
                if control == 'placement':
                    tape = raw.get('arm_tape')
                    ticks = round(scene['config']['timeout_s']/scene['config']['control_dt'])
                    tapes[mode] = tape if raw['valid'] and type(tape) is list and len(tape) == ticks else None
            except Exception as exc:
                raw = {'schema':task.SCHEMA,'scope':task.SCOPE,'valid':False,'success':False,'reason':'exception',
                       'error':{'type':type(exc).__name__,'message':str(exc).replace(str(asset_root.resolve()),'[assets]')[:500]},
                       'trajectory':[],'actions':[],'arm_tape':[],'metrics':{},'timing':{},'warnings':[],'simulated_seconds':0.,
                       'contract':protocol['contract'],
                       'implementation_sha256':{name:protocol['implementation_sha256'][name] for name in base.EPISODE_SOURCE_FILES},
                       'policy_implementation_sha256':{'piper_grip_guard.py':protocol['implementation_sha256']['piper_grip_guard.py']},
                       'grip_guard':{'schema':'dvidia.piper-grip-guard/v1','mode':mode,'active':False,'contract':protocol['guard_contract'],
                                     'events':[],'retry_count':0,'acquisition_qualified':False,'acquisition_qualification_count':0,
                                     'abort_reason':None,'aborted':False,'diagnostic_count':0}}
            raw.update(id=identity,scene_id=scene['id'],variant=variant,control=control,policy_mode=mode,profile=task.PROFILE,
                       physical_robot_ready=False,learned_policy=False,seed=scene['seed'],config=scene['config'],
                       attempt_wall_seconds=time.perf_counter()-began)
            raw['grip_evaluation'] = evaluate_episode(raw)
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
              'summary':aggregate(attempts,protocol['scenes']),'attempts':attempts}
    base._write(destination/'report.json',report)
    (destination/'report.html').write_text(_html(report),encoding='utf-8',newline='\n')
    for name in ('protocol.json','report.json','report.html',base.LICENSE_PATH,base.MODEL_NOTE_PATH):
        data = (destination/name).read_bytes()
        records[name] = {'bytes':len(data),'sha256':sha256(data).hexdigest()}
    base._write(destination/'integrity.json',{'schema':INTEGRITY_SCHEMA,'files':records})
    inspect_grip_bundle(destination)
    return report


def _guard(raw, protocol):
    guard = raw.get('grip_guard')
    if (type(guard) is not dict or guard.get('schema') != 'dvidia.piper-grip-guard/v1'
            or guard.get('mode') != raw['policy_mode'] or guard.get('contract') != protocol['guard_contract']
            or raw.get('policy_implementation_sha256') != {'piper_grip_guard.py':protocol['implementation_sha256']['piper_grip_guard.py']}):
        raise ValueError('Guard mode/contract/source mismatch')
    for name in ('active','acquisition_qualified','aborted'):
        if type(guard.get(name)) is not bool:
            raise ValueError('Invalid guard Boolean')
    for name, maximum in (('retry_count',1),('acquisition_qualification_count',2),('diagnostic_count',1500)):
        if type(guard.get(name)) is not int or not 0<=guard[name]<=maximum:
            raise ValueError('Invalid guard count')
    events = guard.get('events')
    if type(events) is not list or len(events)>100:
        raise ValueError('Invalid bounded guard events')
    allowed = {'probe_started','acquisition_qualified','alarm','retry_started','retry_completed','abort'}
    previous = -1.
    states = {round(state['t_seconds'],8):state for state in raw['trajectory']}
    probe_origin = None
    parameters = protocol['guard_contract']['parameters']
    for event in events:
        if type(event) is not dict or event.get('type') not in allowed:
            raise ValueError('Invalid guard event identity')
        clock = base._numeric(event.get('t_seconds'),'guard event time',0.,raw['simulated_seconds']+1e-8)
        if clock<previous or type(event.get('retry_index')) is not int or not 0<=event['retry_index']<=1:
            raise ValueError('Invalid guard event order/retry')
        if type(event.get('phase')) is not str or type(event.get('reason')) is not str or type(event.get('diagnostic')) is not dict:
            raise ValueError('Invalid guard event evidence')
        state = states.get(round(clock,8))
        diagnostic = event['diagnostic']
        if (state is None or raw['valid'] and state.get('command') is None or diagnostic.get('t_seconds')!=state['t_seconds']
                or diagnostic.get('bilateral_contact')!=all(state['grasp_contact_counts'].values())
                or diagnostic.get('table_supported')!=(state['table_object_contact_count']>0)):
            raise ValueError('Guard event is not aligned to its causal recorded observation')
        if event['type']=='alarm' and state.get('command') is not None and (state['command']['joint_targets_rad']!=state['joint_position_rad']
                                      or state['command']['gripper_position_m']!=0.):
            raise ValueError('Alarm must request measured arm hold and closed jaws')
        if event['type']=='retry_started' and (state['table_object_contact_count']<1 or state['unexpected_contact_count']):
            raise ValueError('Recovery may begin only with a supported clear object')
        if event['type']=='probe_started':
            probe_origin = state
        if event['type']=='acquisition_qualified':
            dwell = parameters['qualification_dwell_s']
            window = [s for s in raw['trajectory'] if clock-dwell-1e-8<=s['t_seconds']<=clock+1e-8]
            if (probe_origin is None or not window or window[0]['t_seconds']>clock-dwell+1e-8
                    or window[0]['t_seconds']<probe_origin['t_seconds']-1e-8
                    or any(s['table_object_contact_count'] or not all(s['grasp_contact_counts'].values())
                           or min(s['pad_normal_forces_n'].values())<=.01
                           or s['object_position_m'][2]-probe_origin['object_position_m'][2]<parameters['probe_min_object_rise_m']
                           or math.dist(_local_offset(s),_local_offset(probe_origin))>parameters['relative_drift_limit_m']
                           for s in window)):
                raise ValueError('Qualification lacks recorded sustained table-free probe evidence')
        previous = clock
    diagnostics = [state.get('guard_diagnostic') for state in raw['trajectory'] if state.get('guard_diagnostic') is not None]
    if (any(type(x) is not dict for x in diagnostics) or guard['diagnostic_count']<len(diagnostics)
            or guard['diagnostic_count']>len(diagnostics)+(0 if raw['valid'] else 1)):
        raise ValueError('Guard diagnostic count mismatch')
    if guard['acquisition_qualification_count']!=sum(e['type']=='acquisition_qualified' for e in events):
        raise ValueError('Guard qualification decision count mismatch')
    if guard['retry_count']!=sum(e['type']=='retry_started' for e in events) or guard['aborted']!=any(e['type']=='abort' for e in events):
        raise ValueError('Guard recovery event count mismatch')
    if ((guard['aborted'] and (type(guard.get('abort_reason')) is not str or not guard['abort_reason']))
            or (not guard['aborted'] and guard.get('abort_reason') is not None)):
        raise ValueError('Guard abort reason mismatch')
    final_qualified = False
    for event in events:
        if event['type']=='acquisition_qualified':
            final_qualified = True
        elif event['type']=='retry_started':
            final_qualified = False
    aborts = [event for event in events if event['type']=='abort']
    if (guard['acquisition_qualified']!=final_qualified or len(aborts)>1
            or aborts and guard['abort_reason']!=aborts[-1]['reason']):
        raise ValueError('Guard final decision state differs from recorded events')
    if raw['valid'] and guard['active'] != (raw['variant']=='guarded_placement'):
        raise ValueError('Guard activation differs from paired variant')
    if not guard['active'] and (events or diagnostics or guard['retry_count'] or guard['acquisition_qualification_count'] or guard['aborted']):
        raise ValueError('Inactive guard contains decisions')
    if guard['active']:
        qualified, retry_open, abort_targets = False, False, None
        by_time = {}
        for event in events:
            by_time.setdefault(round(event['t_seconds'],8),[]).append(event)
        for state in raw['trajectory']:
            command = state.get('command')
            if command is None:
                if state.get('guard_diagnostic') is not None:
                    raise ValueError('Terminal state cannot carry an unexecuted guard decision')
                continue
            diagnostic = state.get('guard_diagnostic')
            if (type(diagnostic) is not dict or diagnostic.get('t_seconds')!=state['t_seconds']
                    or diagnostic.get('bilateral_contact')!=all(state['grasp_contact_counts'].values())
                    or diagnostic.get('table_supported')!=(state['table_object_contact_count']>0)):
                raise ValueError('Guard diagnostic differs from its causal observation')
            for event in by_time.get(round(state['t_seconds'],8),[]):
                if event['type']=='acquisition_qualified':
                    qualified = True
                elif event['type']=='retry_started':
                    qualified, retry_open = False, True
                elif event['type']=='retry_completed':
                    retry_open = False
                elif event['type']=='abort':
                    abort_targets = state['joint_position_rad']
            phase = command.get('phase')
            if phase in ('lift','transfer','place','release','retreat','verify') and not qualified:
                raise ValueError('Guarded carry requires preceding probe qualification')
            if phase in ('guard_retry_open','guard_retry_recenter') and (
                    not retry_open and not any(e['type']=='retry_completed' for e in by_time.get(round(state['t_seconds'],8),[]))
                    or state['table_object_contact_count']<1 or state['unexpected_contact_count']
                    or command['gripper_position_m']!=.035):
                raise ValueError('Recovery opening/motion requires current supported clear object')
            if abort_targets is not None and (command['joint_targets_rad']!=abort_targets or command['gripper_position_m']!=0.):
                raise ValueError('Aborted guard must retain its measured closed holding command')
    if raw.get('grip_evaluation') != evaluate_episode(raw):
        raise ValueError('Independent sampled evaluation differs from recorded states')


def load_grip_bundle(directory):
    """Return report and exact sealed compressed bytes after bounded inspection."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Bundle must be regular non-symlink directory')
    integrity = _read(root/'integrity.json',65536)
    seal = _decode(integrity)
    if type(seal) is not dict or set(seal)!={'schema','files'} or seal['schema']!=INTEGRITY_SCHEMA:
        raise ValueError('Invalid grip integrity schema')
    declarations = seal['files']
    if type(declarations) is not dict or len(declarations)!=35:
        raise ValueError('Incomplete paired artifact count')
    static = {'protocol.json','report.json','report.html',base.LICENSE_PATH,base.MODEL_NOTE_PATH}
    files, decoded, total, raw_total = {}, {}, 0, 0
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
        decoded[name] = decode_attempt(data,record) if compressed else data
        raw_total += len(decoded[name])
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
            or type(protocol.get('planned_attempts')) is not int or protocol['planned_attempts']!=30
            or protocol.get('native_command_units')!=['rad']*6+['m'] or protocol.get('evaluation_contract')!=EVALUATION_CONTRACT
            or protocol.get('guard_contract')!=_guard_contract() or type(protocol.get('contract')) is not dict
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
    if type(scenes) is not list or len(scenes)!=6:
        raise ValueError('Paired protocol requires six scenes')
    lookup, layouts = {}, {}
    for scene in scenes:
        if (type(scene) is not dict or type(scene.get('id')) is not str or scene['id'] in lookup
                or type(scene.get('layout_index')) is not int or scene['layout_index'] not in (0,1)
                or scene.get('friction') not in FRICTIONS or scene.get('stratum')!=STRATA[FRICTIONS.index(scene['friction'])]
                or scene['id']!=f"friction{round(scene['friction']*100):03}-layout{scene['layout_index']:02}"
                or type(scene.get('seed')) is not int or not 0<=scene['seed']<=999999 or type(scene.get('config')) is not dict):
            raise ValueError('Invalid paired scene identity')
        config = task.PiperTaskConfig(**scene['config'])
        if config.friction!=scene['friction']:
            raise ValueError('Scene friction disagrees with native config')
        layout = {k:v for k,v in scene['config'].items() if k!='friction'}
        if scene['layout_index'] in layouts and layouts[scene['layout_index']]!=layout:
            raise ValueError('Friction strata do not share identical layouts/mechanics')
        layouts[scene['layout_index']] = layout
        lookup[scene['id']] = scene
    if {(s['friction'],s['layout_index']) for s in scenes}!={(f,i) for f in FRICTIONS for i in (0,1)}:
        raise ValueError('Missing planned friction/layout pair')
    expected = {(scene,variant) for scene in lookup for variant in VARIANTS}
    attempts, seen, raws = report.get('attempts'),set(),{}
    if type(attempts) is not list or len(attempts)!=30 or type(report.get('attempted')) is not int or report['attempted']!=30:
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
        raw = _decode(decoded[name])
        if type(raw) is not dict or _summary(raw)!={k:v for k,v in row.items() if k!='result_file'}:
            raise ValueError('Attempt summary differs from decoded trace')
        if raw['valid'] and raw.get('asset_provenance')!=pinned:
            raise ValueError('Episode asset identity mismatch')
        base._episode(raw,scene,protocol)
        _guard(raw,protocol)
        seen.add(identity)
        raws[identity] = raw
    for scene in scenes:
        rows = [raws[scene['id'],variant] for variant in VARIANTS]
        compiled = [r for r in rows if r.get('compiled_model_sha256') is not None]
        if any(type(r['compiled_model_sha256']) is not str or re.fullmatch('[0-9a-f]{64}',r['compiled_model_sha256']) is None for r in compiled):
            raise ValueError('Malformed recorded compiled model identity')
        models = [{key:r.get(key) for key in ('compiled_model_sha256','native_model_settings','native_control_limits',
                   'native_servo_gains','native_servo_bias','native_force_limits','contact_geometry')} for r in compiled]
        if any(model!=models[0] for model in models[1:]):
            raise ValueError('Paired controls use different native mechanics')
        starts = [r['trajectory'][0]['qpos'] for r in rows if r['valid']]
        if any(start!=starts[0] for start in starts[1:]):
            raise ValueError('Paired controls start from different native state')
        for placement,opened in ((rows[0],rows[1]),(rows[2],rows[3])):
            if opened['actions'] and (not placement['valid'] or opened['arm_tape']!=placement['arm_tape'][:len(opened['arm_tape'])]):
                raise ValueError('Open-jaw replay differs from its own placement tape')
        held = rows[4]
        if held['actions'] and (not held['trajectory'] or any(command!=held['trajectory'][0]['joint_position_rad'] for command in held['arm_tape'])):
            raise ValueError('Common hold moves its initial arm targets')
    if (set(files)!=static|{r['result_file'] for r in attempts} or report.get('summary')!=aggregate(attempts,scenes)
            or type(report.get('valid_attempts')) is not int or report['valid_attempts']!=sum(r['valid'] for r in attempts)):
        raise ValueError('Paired inventory/aggregate denominator mismatch')
    files['integrity.json'] = integrity
    return report,files


def inspect_grip_bundle(directory):
    report, files = load_grip_bundle(directory)
    return {'schema':'dvidia.piper-grip-inspection/v1','scope':'Recorded consistency, not authenticity or physics recomputation',
            'files_verified':len(files)-1,'attempted':report['attempted'],'valid_attempts':report['valid_attempts'],
            'summary':report['summary'],'physical_robot_ready':False,'learned_policy':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    run = commands.add_parser('benchmark')
    run.add_argument('--assets',required=True,type=Path)
    run.add_argument('--output',required=True,type=Path)
    run.add_argument('--seed',default=712047,type=int)
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
            result = inspect_grip_bundle(args.bundle)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError,RuntimeError,TypeError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':
    main()
