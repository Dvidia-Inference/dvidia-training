"""Frozen paired PiPER probe/contact-history experiments and bounded offline inspection.

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

from . import piper_task as task, piper_task_benchmark as base, piper_grip_benchmark as prior, robot_profiles
from .reference_replay import _decode, _json_bytes, _read

SCHEMA = 'dvidia.piper-contact-benchmark/v0.3'
PROTOCOL_SCHEMA = 'dvidia.piper-contact-protocol/v0.3'
INTEGRITY_SCHEMA = 'dvidia.piper-contact-integrity/v0.3'
VARIANTS = ('baseline_placement', 'baseline_open_jaw', 'probe_only_placement', 'probe_only_open_jaw', 'history_placement', 'history_open_jaw', 'hold')
MASSES = (.04, .06)
FRICTIONS = (.8, .55, .15)
STRATA = ('nominal', 'moderate_friction', 'low_friction')
SOURCE_FILES = prior.SOURCE_FILES + ('piper_contact_memory.py', 'piper_contact_benchmark.py')
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_DECODED_BYTES = 512 * 1024 * 1024
SCOPE = ('Paired authored PiPER contact-history pilot, privileged native state and synthetic rigid blocks. '
         'No learned policy, sensor pipeline, calibrated contact, shoe competence or physical robot qualification.')
EVALUATION_CONTRACT = {
    'schema': 'dvidia.piper-contact-sampled-evaluation/v0.3',
    'following': dict(prior.EVALUATION_CONTRACT),
    'motion_window_s': .100, 'settled_window_s': .100,
    'settled_tcp_speed_limit_m_s': .010,
    'minimum_settled_object_rise_m': .006,
    'scope': 'Independent recorded 20ms boundaries, not detector accuracy or physics recomputation. '
             'Motion windows retain actual native movement after commands, not instantaneous arrest. '
             'Settled, resumed and completed are separate; no observed true-slip opportunity is not a perfect score.',
}



def implementation_hashes():
    root = Path(__file__).parent
    return {name: sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _memory_contract():
    from .piper_contact_memory import contact_memory_contract
    return contact_memory_contract()


def _variant(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown contact-history variant')
    if variant == 'hold':
        return 'hold', 'baseline'
    mode = next(name for name in ('baseline', 'probe_only', 'history') if variant.startswith(name+'_'))
    return ('replay_open_jaw' if variant.endswith('open_jaw') else 'placement'), mode


def make_protocol(*, seed=817031, cohort='development'):
    if type(seed) is not int or not 0 <= seed <= 999988:
        raise ValueError('Seed must be an integer in [0,999988]')
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
        for mass in MASSES:
            for index, (start, target) in enumerate(layouts):
                config = replace(default, object_position_m=tuple(start), target_position_m=tuple(target),
                                 friction=friction, mass_kg=mass)
                scenes.append({'id': f'friction{round(friction*100):03}-mass{round(mass*1000):03}-layout{index:02}',
                               'layout_index': index, 'stratum': stratum, 'friction': friction, 'mass_kg': mass,
                               'seed': seed+len(scenes), 'config': asdict(config)})
    return {'schema': PROTOCOL_SCHEMA, 'profile': task.PROFILE, 'scope': SCOPE, 'physical_robot_ready': False,
            'learned_policy': False, 'cohort': cohort, 'seed': seed, 'scenes': scenes, 'variants': list(VARIANTS),
            'planned_attempts': 84, 'contract': task.task_contract(), 'contact_contract': _memory_contract(),
            'acquisition_contract': _memory_contract()['acquisition'],
            'evaluation_contract': EVALUATION_CONTRACT, 'implementation_sha256': implementation_hashes(),
            'native_command_units': ['rad']*6+['m'], 'simulation_asset_license': 'MIT', 'layout_offset_limit_m': .005,
            'timing_boundary': base.TIMING_BOUNDARY,
            'comparison': 'Two shared XY layouts at each friction and mass; identical native mechanics and starting '
                          'state within each scene. Baseline, acquisition probe-only and contact-history variants '
                          'each have separately matched complete arm-tape/open-jaw controls, plus common hold. '
                          'Probe-only and history share acquisition; only history adapts after qualification. '
                          'All failures and full-duration aborts remain; invalid/incomplete tapes are never extended.',
            'split_note': 'Synthetic declared confirmation does not prove unseen scenes or physical reliability. '
                          'Controller/evaluation parameters and sources are frozen before any native attempt.'}


decode_attempt = prior.decode_attempt


def _local_offset(state):
    # Inverse unit quaternion rotates the object-to-TCP vector into the TCP frame.
    w, x, y, z = state['tcp_quaternion_wxyz']
    v = [a-b for a, b in zip(state['object_position_m'], state['tcp_position_m'])]
    u = (-x, -y, -z)
    cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    second = (u[1]*cross[2]-u[2]*cross[1], u[2]*cross[0]-u[0]*cross[2], u[0]*cross[1]-u[1]*cross[0])
    return [v[i]+2*w*cross[i]+2*second[i] for i in range(3)]


def _motion_window(states, start, end):
    window = [s for s in states if start-1e-9 <= s['t_seconds'] <= end+1e-9]
    complete = bool(window and window[0]['t_seconds'] <= start+1e-9 and window[-1]['t_seconds'] >= end-1e-9)
    spans = [b['t_seconds']-a['t_seconds'] for a,b in zip(window,window[1:])]
    tcp_segments = [math.dist(a['tcp_position_m'],b['tcp_position_m']) for a,b in zip(window,window[1:])]
    object_segments = [math.dist(a['object_position_m'],b['object_position_m']) for a,b in zip(window,window[1:])]
    span = window[-1]['t_seconds']-window[0]['t_seconds'] if window else 0.
    return {'start_s':start,'end_s':end,'complete':complete,'sample_count':len(window),'observed_span_s':span,
            'tcp_path_m':sum(tcp_segments),'object_path_m':sum(object_segments),
            'tcp_mean_speed_m_s':sum(tcp_segments)/span if span>0 else None,
            'object_mean_speed_m_s':sum(object_segments)/span if span>0 else None,
            'tcp_peak_sampled_speed_m_s':max((d/t for d,t in zip(tcp_segments,spans) if t>0),default=None),
            'object_peak_sampled_speed_m_s':max((d/t for d,t in zip(object_segments,spans) if t>0),default=None),
            'tcp_rise_m':max((s['tcp_position_m'][2]-window[0]['tcp_position_m'][2] for s in window),default=None),
            'object_rise_m':max((s['object_position_m'][2]-window[0]['object_position_m'][2] for s in window),default=None)}


def evaluate_episode(raw):
    """Measure actual recorded motion/outcomes, independently of controller labels."""
    states, guard = raw.get('trajectory',[]), raw.get('grip_guard',{})
    events = guard.get('events',[])
    loss_view = dict(raw)
    loss_view['grip_guard'] = dict(guard,events=[dict(e,type='alarm') for e in events if e['type']=='reassessment_started'])
    sampled_holds = {round(e['t_seconds'],8):e for e in prior.evaluate_episode(loss_view)['alarms']}
    interventions = []
    width = EVALUATION_CONTRACT['motion_window_s']
    for event in events:
        if event['type'] not in ('slowdown_started','reassessment_started','carry_resumed'):
            continue
        clock = event['t_seconds']
        entry = {'type':event['type'],'t_seconds':clock,'reason':event['reason'],
                 'before':_motion_window(states,max(0.,clock-width),clock),
                 'after':_motion_window(states,clock,clock+width)}
        later = next((e['t_seconds'] for e in events if e['t_seconds']>clock and e['type'] in
                      ('reassessment_started','carry_resumed','abort')),None)
        if event['type'] in ('slowdown_started','reassessment_started'):
            terminal = later if later is not None else raw.get('simulated_seconds',clock)
            entry['intervention'] = _motion_window(states,clock,terminal)
            entry['ended_by_observed_event'] = later is not None
        if event['type']=='reassessment_started':
            entry['sampled_following_loss_comparison'] = sampled_holds.get(round(clock,8))
        if event['type']=='carry_resumed':
            settled = [s for s in states if clock-EVALUATION_CONTRACT['settled_window_s']-1e-9<=s['t_seconds']<=clock+1e-9]
            complete = bool(settled and settled[0]['t_seconds']<=clock-EVALUATION_CONTRACT['settled_window_s']+1e-9)
            loaded = bool(complete and all(not s['table_object_contact_count'] and all(s['grasp_contact_counts'].values())
                          and min(s['pad_normal_forces_n'].values())>.01 and s['unexpected_contact_count']==0
                          and s['object_position_m'][2]-states[0]['object_position_m'][2]>=EVALUATION_CONTRACT['minimum_settled_object_rise_m']
                          for s in settled))
            # Actual absolute motion is separate from the controller's relative history predicate.
            speed = entry['before']['tcp_peak_sampled_speed_m_s']
            entry['recorded_loaded_unsupported_window'] = loaded if complete else None
            entry['recorded_low_tcp_motion_window'] = (loaded and speed is not None and
                    speed<=EVALUATION_CONTRACT['settled_tcp_speed_limit_m_s']) if complete else None
            entry['task_completed'] = bool(raw.get('valid') and raw.get('success'))
        interventions.append(entry)
    resumes = [e for e in interventions if e['type']=='carry_resumed']
    return {'schema':EVALUATION_CONTRACT['schema'],'following':prior.evaluate_episode(raw),
            'interventions':interventions,'resume_decisions':len(resumes),
            'resumed_completed':bool(resumes and raw.get('valid') and raw.get('success')),
            'resumed_not_completed':bool(resumes and not (raw.get('valid') and raw.get('success'))),
            'loaded_settling_observed_at_resume':sum(e['recorded_loaded_unsupported_window'] is True for e in resumes),
            'low_tcp_motion_observed_at_resume':sum(e['recorded_low_tcp_motion_window'] is True for e in resumes),
            'unresolved_resume_windows':sum(e['recorded_loaded_unsupported_window'] is None for e in resumes)}


def aggregate(attempts, scenes):
    result = []
    for friction, stratum in zip(FRICTIONS, STRATA):
        for mass in MASSES:
            for variant in VARIANTS:
                rows = [r for r in attempts if r['variant']==variant and r['config']['friction']==friction and r['config']['mass_kg']==mass]
                success = sum(r['valid'] is True and r['success'] is True for r in rows)
                follow = [r['contact_evaluation']['following'] for r in rows]
                decisions = [d for f in follow for d in f['qualification_decisions']]
                result.append({'friction':friction,'mass_kg':mass,'stratum':stratum,'variant':variant,
                               'attempted':len(rows),'valid':sum(r['valid'] is True for r in rows),'success':success,
                               'failed_or_invalid':len(rows)-success,'success_fraction':success/len(rows) if rows else None,
                               'qualification_decisions':len(decisions),
                               'following_observed_after_decision':sum(d['following_observed'] is True for d in decisions),
                               'following_not_observed_after_decision':sum(d['following_observed'] is False for d in decisions),
                               'unresolved_qualification_decisions':sum(d['following_observed'] is None for d in decisions),
                               'no_qualification_decision_attempts':sum(not f['qualification_decisions'] for f in follow),
                               'no_following_opportunity_attempts':sum(f['following_opportunity'] is False for f in follow),
                               'unresolved_following_opportunity_attempts':sum(f['following_opportunity'] is None for f in follow),
                               'aborted_attempts':sum(f['aborted'] for f in follow),'retries':sum(f['retry_count'] for f in follow),
                               'sampled_following_loss_events':sum(f['sampled_drop_events'] for f in follow),
                               'slowdown_decisions':sum(r['grip_guard'].get('slowdown_count',0) for r in rows),
                               'reassessment_decisions':sum(r['grip_guard'].get('reassessment_count',0) for r in rows),
                               'resume_decisions':sum(r['contact_evaluation']['resume_decisions'] for r in rows),
                               'loaded_settling_observed_at_resume':sum(r['contact_evaluation']['loaded_settling_observed_at_resume'] for r in rows),
                               'low_tcp_motion_observed_at_resume':sum(r['contact_evaluation']['low_tcp_motion_observed_at_resume'] for r in rows),
                               'unresolved_resume_windows':sum(r['contact_evaluation']['unresolved_resume_windows'] for r in rows),
                               'holds_without_preceding_sampled_loss':sum(e['sampled_following_loss_comparison']['no_preceding_sampled_loss_opportunity']
                                   for r in rows for e in r['contact_evaluation']['interventions'] if e['type']=='reassessment_started' and e['sampled_following_loss_comparison'] is not None),
                               'incomplete_post_intervention_motion_windows':sum(not e['after']['complete'] for r in rows for e in r['contact_evaluation']['interventions']),
                               'resumed_completed':sum(r['contact_evaluation']['resumed_completed'] for r in rows),
                               'resumed_not_completed':sum(r['contact_evaluation']['resumed_not_completed'] for r in rows)})
    return result


def _summary(raw):
    return base._summary(raw)


def _html(report):
    rows = ''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in
                    (r['friction'],r['mass_kg'],r['variant'],r['success'],r['attempted'],r['valid'],r['resumed_completed'],r['resumed_not_completed']))+'</tr>' for r in report['summary'])
    attempts = []
    for row in report['attempts']:
        guard,evaluation = row['grip_guard'],row['contact_evaluation']
        observations = ', '.join(f"{d['t_seconds']:.2f}s "+('following observed' if d['following_observed'] is True else 'following not observed' if d['following_observed'] is False else 'unresolved') for d in evaluation['following']['qualification_decisions']) or 'no qualification decision'
        events = ', '.join(f"{e['t_seconds']:.2f}s {e['type'].replace('_',' ')} ({e['reason']})" for e in guard['events']) or 'no controller events'
        motion = []
        for intervention in evaluation['interventions']:
            def speed(window):
                value=window['tcp_mean_speed_m_s']
                return 'unobserved' if value is None else f'{value*1000:.2f}mm/s'
            motion.append(f"{intervention['t_seconds']:.2f}s {intervention['type'].replace('_',' ')}: actual TCP mean speed {speed(intervention['before'])} before → {speed(intervention['after'])} after; "+
                          f"TCP path {intervention['after']['tcp_path_m']*1000:.2f}mm / object path {intervention['after']['object_path_m']*1000:.2f}mm over "+
                          ('complete 100ms window' if intervention['after']['complete'] else 'incomplete window'))
        attempts.append('<li><strong>'+escape(row['id'])+'</strong>: '+('completed' if row['success'] else 'failed' if row['valid'] else 'invalid')+
                        ' · '+escape(observations)+'. Retries '+str(guard['retry_count'])+'. '+
                        ('Abort: '+escape(str(guard['abort_reason']))+'. ' if guard['aborted'] else '')+'<br>'+escape(events)+
                        (('<br>'+escape('; '.join(motion))) if motion else '')+' · <a href="'+escape(row['result_file'],quote=True)+'">complete JSON gzip</a></li>')
    attempts = ''.join(attempts)
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>DVIDIA · PiPER contact history comparison</title><style>body{max-width:1100px;margin:auto;padding:24px;font:18px/1.6 Georgia;'
            'background:#fafaf7;color:#292d27}table{font:14px/1.5 system-ui;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd}'
            '.scroll{overflow:auto}a{color:#396751}</style><main><p>DVIDIA RESEARCH</p><h1>Probe, adapt, complete</h1><p>'+escape(SCOPE)+
            '</p><p>'+escape(report['cohort'])+' · all 84 planned attempts retained.</p><p><a href="report.json">Full report</a> · '
            '<a href="protocol.json">Frozen protocol</a> · <a href="integrity.json">Stored and decoded integrity</a></p><div class="scroll"><table>'
            '<tr><th>Friction</th><th>Mass kg</th><th>Variant</th><th>Completed</th><th>Attempted</th><th>Valid</th><th>Resumed and completed</th>'
            '<th>Resumed, incomplete</th></tr>'+rows+'</table></div><p>'+escape(EVALUATION_CONTRACT['scope'])+'</p><p>'+escape(base.TIMING_BOUNDARY)+
            '</p><h2>Every attempt</h2><ul>'+attempts+'</ul><footer><a href="https://research.dvidia.org/">DVIDIA research</a> · '
            '<a href="https://x.com/iammrriver">@iammrriver</a> · <a href="mailto:hello@dvidia.org">hello@dvidia.org</a></footer></main></html>')


def run_benchmark(assets, output, *, seed=817031, cohort='development', progress=None):
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
                from . import piper_grip_guard, piper_contact_memory
                inactive = (piper_grip_guard.inactive_summary(mode) if mode=='baseline'
                            else piper_contact_memory.inactive_summary(mode))
                policy_sources = ('piper_grip_guard.py',) if mode=='baseline' else ('piper_grip_guard.py','piper_contact_memory.py')
                raw = {'schema':task.SCHEMA,'scope':task.SCOPE,'valid':False,'success':False,'reason':'exception',
                       'error':{'type':type(exc).__name__,'message':str(exc).replace(str(asset_root.resolve()),'[assets]')[:500]},
                       'trajectory':[],'actions':[],'arm_tape':[],'metrics':{},'timing':{},'warnings':[],'simulated_seconds':0.,
                       'contract':protocol['contract'],
                       'implementation_sha256':{name:protocol['implementation_sha256'][name] for name in base.EPISODE_SOURCE_FILES},
                       'policy_implementation_sha256':{name:protocol['implementation_sha256'][name] for name in policy_sources},
                       'grip_guard':inactive}
            raw.update(id=identity,scene_id=scene['id'],variant=variant,control=control,policy_mode=mode,profile=task.PROFILE,
                       physical_robot_ready=False,learned_policy=False,seed=scene['seed'],config=scene['config'],
                       attempt_wall_seconds=time.perf_counter()-began)
            raw['contact_evaluation'] = evaluate_episode(raw)
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
    inspect_contact_bundle(destination)
    return report


def _near(actual, expected):
    if expected is None or type(expected) is bool:
        return actual is expected
    if type(expected) is int:
        return type(actual) is int and actual==expected
    if isinstance(expected,float):
        try:
            return type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-7,abs_tol=1e-9)
        except (OverflowError,TypeError,ValueError):
            return False
    if type(expected) is list:
        return type(actual) is list and len(actual)==len(expected) and all(_near(a,b) for a,b in zip(actual,expected))
    return actual==expected


def _history_values(history, parameters):
    span = history[-1]['t_seconds']-history[0]['t_seconds'] if history else 0.
    known = span+1e-9>=parameters['minimum_history_s']
    offsets = [_local_offset(s) for s in history]
    displacement = math.dist(offsets[0],offsets[-1]) if offsets else None
    path = sum(math.dist(a,b) for a,b in zip(offsets,offsets[1:]))
    excursion = max((math.dist(a,b) for a in offsets for b in offsets),default=0.)
    slope = None
    if known:
        mean = sum(s['t_seconds'] for s in history)/len(history)
        denominator = sum((s['t_seconds']-mean)**2 for s in history)
        slope = [sum((s['t_seconds']-mean)*o[axis] for s,o in zip(history,offsets))/denominator for axis in range(3)]
    return {'history_count':len(history),'history_known':known,'history_span_seconds':span,
            'history_displacement_m':displacement,'history_rate_m_s':displacement/span if known else None,
            'history_path_m':path,'history_excursion_m':excursion,'history_slope_m_s':slope,
            'history_straightness':displacement/path if known and path>1e-12 else (0. if known else None),
            'history_bilateral_fraction':sum(all(s['grasp_contact_counts'].values()) for s in history)/len(history) if history else None,
            'history_loaded_bilateral_all':known and all(all(s['grasp_contact_counts'].values()) and min(s['pad_normal_forces_n'].values())>.01 for s in history),
            'history_unsupported_all':known and all(not s['table_object_contact_count'] for s in history)}


def _memory(raw, protocol):
    """Recorded decision consistency, without rerunning dynamics or trusting labels."""
    guard, mode = raw.get('grip_guard'), raw['policy_mode']
    source_names = ('piper_grip_guard.py',) if mode=='baseline' else ('piper_grip_guard.py','piper_contact_memory.py')
    expected_sources = {name:protocol['implementation_sha256'][name] for name in source_names}
    if (type(guard) is not dict or guard.get('mode')!=mode or raw.get('policy_implementation_sha256')!=expected_sources
            or guard.get('contract')!=(protocol['acquisition_contract'] if mode=='baseline' else protocol['contact_contract'])
            or guard.get('schema')!=('dvidia.piper-grip-guard/v1' if mode=='baseline' else 'dvidia.piper-contact-memory/v0.3')):
        raise ValueError('Contact mode/contract/source mismatch')
    old_types = {'probe_started','acquisition_qualified','alarm','retry_started','retry_completed','abort'}
    new_types = {'slowdown_started','reassessment_started','carry_resumed'}
    events = guard.get('events')
    if type(events) is not list or len(events)>100 or any(type(e) is not dict or e.get('type') not in old_types|new_types for e in events):
        raise ValueError('Invalid bounded contact events')
    if mode!='history' and any(e['type'] in new_types for e in events):
        raise ValueError('Only history mode may intervene after acquisition')
    # Reuse the frozen acquisition validator on a view, not a second controller.
    view, compatibility = dict(raw),dict(protocol)
    view['grip_guard'] = dict(guard)
    view['grip_guard'].update(schema='dvidia.piper-grip-guard/v1',
        mode='baseline' if mode=='baseline' else 'guarded',contract=protocol['acquisition_contract'],
        events=[e for e in events if e['type'] in old_types])
    view['policy_mode'] = 'baseline' if mode=='baseline' else 'guarded'
    view['variant'] = 'guarded_placement' if mode!='baseline' and raw['control']=='placement' else 'baseline_placement'
    view['policy_implementation_sha256'] = {'piper_grip_guard.py':protocol['implementation_sha256']['piper_grip_guard.py']}
    view['grip_evaluation'] = prior.evaluate_episode(view)
    compatibility['guard_contract'] = protocol['acquisition_contract']
    prior._guard(view,compatibility)
    if mode!='baseline':
        for name,kind,limit in (('slowdown_count','slowdown_started',3),('reassessment_count','reassessment_started',2),('resume_count','carry_resumed',2)):
            if type(guard.get(name)) is not int or not 0<=guard[name]<=limit or guard[name]!=sum(e['type']==kind for e in events):
                raise ValueError('Contact intervention count mismatch')
        if guard['resume_count']>guard['reassessment_count'] or not guard['active'] and any(guard[name] for name in ('slowdown_count','reassessment_count','resume_count')):
            raise ValueError('Inactive or unmatched contact intervention')
    states = {round(s['t_seconds'],8):s for s in raw['trajectory']}
    previous, by_time = -1.,{}
    for event in events:
        clock = base._numeric(event.get('t_seconds'),'contact event time',0.,raw['simulated_seconds']+1e-8)
        state, diag = states.get(round(clock,8)),event.get('diagnostic')
        if (clock<previous or state is None or type(diag) is not dict or type(event.get('retry_index')) is not int
                or not 0<=event['retry_index']<=1 or type(event.get('phase')) is not str or type(event.get('reason')) is not str
                or diag.get('t_seconds')!=state['t_seconds'] or diag.get('bilateral_contact')!=all(state['grasp_contact_counts'].values())
                or diag.get('table_supported')!=(state['table_object_contact_count']>0)):
            raise ValueError('Contact event is not aligned to causal recorded observation')
        if event['type'] in new_types and (state.get('command') is None or not guard['active']):
            raise ValueError('Contact intervention lacks issued action')
        event_phases = {'probe_started':'guard_probe','acquisition_qualified':'guard_probe',
                        'alarm':'guard_stop','retry_started':'guard_stop','retry_completed':'guard_retry_recenter',
                        'abort':'guard_abort','slowdown_started':'memory_slowdown',
                        'reassessment_started':'memory_hold','carry_resumed':'memory_hold'}
        if event['phase']!=event_phases[event['type']]:
            raise ValueError('Contact event phase differs from its declared decision')
        by_time.setdefault(round(clock,8),[]).append(event)
        previous = clock
    if mode!='baseline' and guard['active']:
        cfg = protocol['contact_contract']['parameters']
        qualified = probing = aborted = destination_supported = resumed = False
        history, origin, stage, started, targets, settled_since = [],None,'observe',None,None,None
        trend_streak = loss_streak = 0
        for state in raw['trajectory']:
            command, clock = state.get('command'),state['t_seconds']
            if command is None:
                continue
            diag = state['guard_diagnostic']
            current_events = by_time.get(round(clock,8),[])
            phase = command.get('phase')
            diagnostic_phase = diag.get('phase')
            if (diagnostic_phase in ('lift','transfer','place') and phase not in (diagnostic_phase,'memory_slowdown','memory_hold','guard_abort')
                    or phase in ('lift','transfer','place','memory_slowdown','memory_hold') and diagnostic_phase not in ('lift','transfer','place')
                    or diagnostic_phase=='probe' and phase not in ('guard_probe','guard_stop','guard_abort')):
                raise ValueError('Contact diagnostic phase differs from issued carry/probe phase')
            if any(e['type']=='probe_started' for e in current_events):
                probing, origin = True,state
            if probing or qualified:
                history.append(state)
                history = [s for s in history if s['t_seconds']>=clock-cfg['history_window_s']-1e-9]
            else:
                history = []
            calculated = _history_values(history,cfg)
            for key,value in calculated.items():
                if not _near(diag.get(key),value):
                    raise ValueError('Causal contact history differs from recorded past states: '+key)
                for event in current_events:
                    if not _near(event['diagnostic'].get(key),value):
                        raise ValueError('Contact event history differs from recorded past states: '+key)
            if origin is not None and not _near(diag.get('object_height_delta_m'),state['object_position_m'][2]-origin['object_position_m'][2]):
                raise ValueError('Contact height differs from recorded probe origin')
            if diag.get('memory_stage') not in ('observe','slowdown','hold','destination_supported'):
                raise ValueError('Unknown contact history stage')
            arrived = (qualified and diag.get('phase') in ('lift','transfer','place') and
                       state['table_object_contact_count']>0 and all(state['grasp_contact_counts'].values()) and
                       math.dist(state['object_position_m'],state['target_position_m'])<=raw['config']['goal_position_tolerance_m'])
            monitored = mode=='history' and qualified and not aborted and not destination_supported and not arrived and diag.get('phase') in ('lift','transfer','place')
            if monitored and not state['unexpected_contact_count']:
                trend = (calculated['history_known'] and calculated['history_rate_m_s']>=cfg['trend_rate_m_s']
                         and calculated['history_displacement_m']>=cfg['trend_displacement_m'] and calculated['history_straightness']>=cfg['minimum_straightness'])
                trend_streak = trend_streak+1 if trend else 0
                loss_streak = loss_streak+1 if not all(state['grasp_contact_counts'].values()) else 0
            if not _near(diag.get('slip_trend_streak'),trend_streak) or not _near(diag.get('persistent_loss_streak'),loss_streak):
                raise ValueError('Contact trend/loss streak differs from causal snapshots')
            if stage=='hold' and monitored and not state['unexpected_contact_count']:
                stable = (clock-started+1e-9>=cfg['hold_minimum_s'] and calculated['history_known']
                          and calculated['history_displacement_m']<=cfg['settled_displacement_m']
                          and calculated['history_excursion_m']<=cfg['settled_displacement_m']
                          and calculated['history_rate_m_s']<=cfg['settled_rate_m_s']
                          and calculated['history_loaded_bilateral_all'] and calculated['history_unsupported_all']
                          and origin is not None and state['object_position_m'][2]-origin['object_position_m'][2]>=protocol['acquisition_contract']['parameters']['probe_min_object_rise_m'])
                settled_since = (clock if settled_since is None else settled_since) if stable else None
                dwell = 0. if settled_since is None else clock-settled_since
                if not _near(diag.get('settled_dwell_seconds'),dwell):
                    raise ValueError('Settling dwell differs from recorded loaded unsupported history')
            for event in current_events:
                kind = event['type']
                if kind=='acquisition_qualified':
                    qualified,probing = True,False
                elif kind=='retry_started':
                    qualified,probing = False,False
                elif kind=='retry_completed':
                    origin = None
                elif kind=='abort':
                    aborted,probing = True,False
                elif kind in new_types:
                    if not qualified or not monitored:
                        raise ValueError('Contact intervention precedes qualified monitored carry')
                    if kind=='slowdown_started':
                        if stage!='observe' or trend_streak<cfg['trend_samples'] or loss_streak>=cfg['loss_samples']:
                            raise ValueError('Slowdown lacks sustained causal relative trend')
                        stage,started = 'slowdown',clock
                    elif kind=='reassessment_started':
                        if not (loss_streak>=cfg['loss_samples'] or stage=='slowdown' and clock-started+1e-9>=cfg['slowdown_duration_s']):
                            raise ValueError('Reassessment lacks bounded slowdown or contact loss')
                        stage,started,targets,settled_since = 'hold',clock,state['joint_position_rad'],None
                    else:
                        if stage!='hold' or settled_since is None or clock-settled_since+1e-9<cfg['settled_dwell_s']:
                            raise ValueError('Resume lacks sustained loaded unsupported settling')
                        if not _near(diag.get('resume_local_offset_m'),_local_offset(state)):
                            raise ValueError('Resume offset differs from current recorded object pose')
                        stage,resumed = 'observe',True
                        trend_streak = loss_streak = 0
            expected_stage = 'destination_supported' if destination_supported else stage
            if diag['memory_stage']!=expected_stage:
                raise ValueError('Contact memory stage differs from preceding events and supported arrival')
            if phase in ('memory_slowdown','memory_hold') and not qualified:
                raise ValueError('Contact carry requires preceding probe qualification')
            if phase=='memory_hold' or any(e['type']=='carry_resumed' for e in current_events):
                if targets is None or command['joint_targets_rad']!=targets or command['gripper_position_m']!=0.:
                    raise ValueError('Reassessment must retain measured closed hold through resume boundary')
                if stage=='hold' and clock-started>cfg['hold_timeout_s']+raw['config']['control_dt']+1e-9 and not aborted:
                    raise ValueError('Reassessment exceeded its fixed hold budget')
            scale, unscaled = diag.get('joint_delta_scale'),diag.get('unscaled_joint_targets_rad')
            if phase=='memory_slowdown':
                if stage!='slowdown' or clock-started>=cfg['slowdown_duration_s']+1e-9 or scale!=cfg['slowdown_joint_delta_scale']:
                    raise ValueError('Slowdown phase/scale/duration mismatch')
            if unscaled is not None:
                base._vector(unscaled,6,'unscaled teacher targets')
                expected_scale = cfg['slowdown_joint_delta_scale'] if phase=='memory_slowdown' else cfg['resumed_joint_delta_scale']
                if (type(unscaled) is not list or len(unscaled)!=6 or not _near(scale,expected_scale)
                        or phase!='memory_slowdown' and not resumed or diag.get('motion_phase') not in ('lift','transfer','place')
                        or not _near(command['joint_targets_rad'],[q+scale*(u-q) for q,u in zip(state['joint_position_rad'],unscaled)])):
                    raise ValueError('Requested reduced joint delta differs from recorded teacher targets')
            elif scale!=1.:
                raise ValueError('Requested scale lacks original teacher targets')
            if arrived:
                destination_supported = True
    if raw.get('contact_evaluation')!=evaluate_episode(raw):
        raise ValueError('Independent contact evaluation differs from recorded states')


def load_contact_bundle(directory):
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
            or protocol.get('contact_contract')!=_memory_contract() or type(protocol.get('contract')) is not dict
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
        _memory(raw,protocol)
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
        starts = [r['start']['qpos'] for r in rows if r['valid']]
        if any(start!=starts[0] for start in starts[1:]):
            raise ValueError('Paired controls start from different native state')
        for placement,opened in ((rows[0],rows[1]),(rows[2],rows[3]),(rows[4],rows[5])):
            if opened['has_actions'] and (not placement['valid'] or opened['arm_tape']!=placement['arm_tape'][:len(opened['arm_tape'])]):
                raise ValueError('Open-jaw replay differs from its own placement tape')
        held = rows[6]
        if held['has_actions'] and (held['start'] is None or any(command!=held['start']['joint_position_rad'] for command in held['arm_tape'])):
            raise ValueError('Common hold moves its initial arm targets')
    if (set(files)!=static|{r['result_file'] for r in attempts} or report.get('summary')!=aggregate(attempts,scenes)
            or type(report.get('valid_attempts')) is not int or report['valid_attempts']!=sum(r['valid'] for r in attempts)):
        raise ValueError('Paired inventory/aggregate denominator mismatch')
    files['integrity.json'] = integrity
    return report,files


def inspect_contact_bundle(directory):
    report, files = load_contact_bundle(directory)
    return {'schema':'dvidia.piper-contact-inspection/v0.3','scope':'Recorded consistency, not authenticity or physics recomputation',
            'files_verified':len(files)-1,'attempted':report['attempted'],'valid_attempts':report['valid_attempts'],
            'summary':report['summary'],'physical_robot_ready':False,'learned_policy':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    run = commands.add_parser('benchmark')
    run.add_argument('--assets',required=True,type=Path)
    run.add_argument('--output',required=True,type=Path)
    run.add_argument('--seed',default=817031,type=int)
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
            result = inspect_contact_bundle(args.bundle)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError,RuntimeError,TypeError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':
    main()
