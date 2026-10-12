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
                    'full_speed_restored':sum(e['full_speed_restored'] for e in evaluations)})
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
              'summary':aggregate(attempts,protocol['scenes']),'paired_comparisons':paired_comparisons(attempts,protocol['scenes']),'attempts':attempts}
    base._write(destination/'report.json',report)
    (destination/'report.html').write_text(_html(report),encoding='utf-8',newline='\n')
    for name in ('protocol.json','report.json','report.html',base.LICENSE_PATH,base.MODEL_NOTE_PATH):
        data = (destination/name).read_bytes()
        records[name] = {'bytes':len(data),'sha256':sha256(data).hexdigest()}
    base._write(destination/'integrity.json',{'schema':INTEGRITY_SCHEMA,'files':records})
    inspect_reliability_bundle(destination, assets=asset_root)
    return report



__NEW_AUDIT__

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
        for placement,opened in ((rows[i],rows[i+1]) for i in range(0,len(rows)-1,2)):
            if opened['has_actions'] and (not placement['valid'] or opened['arm_tape']!=placement['arm_tape'][:len(opened['arm_tape'])]):
                raise ValueError('Open-jaw replay differs from its own placement tape')
        held = rows[-1]
        if held['has_actions'] and (held['start'] is None or any(command!=held['start']['joint_position_rad'] for command in held['arm_tape'])):
            raise ValueError('Common hold moves its initial arm targets')
    if (set(files)!=static|{r['result_file'] for r in attempts} or report.get('summary')!=aggregate(attempts,scenes)
            or report.get('paired_comparisons')!=paired_comparisons(attempts,scenes)
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
