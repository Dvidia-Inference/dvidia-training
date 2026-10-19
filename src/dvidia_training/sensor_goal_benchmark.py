"""Paired RGB/drive corroboration benchmark. No provider calls or robot control."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import time
import zlib
import numpy as np

from . import local_goal_benchmark as old
from .local_goal_verifier import GoalContract,verify_goal
from .sensor_goal_verifier import specification,camera_window,actuator_window,fuse

read_json,write_json,sha,relative_file=old.read_json,old.write_json,old.sha,old.relative_file
COHORTS=('native','primary_forgery','sensor_fault','cross_channel')
NEGATIVES=('wrong_position','wrong_orientation','missing_target','wrong_object','unreleased',
           'sustained_motion','instant_alias','yaw_flip180','lookalike_wrongobject','elevated_support')
FAULTS=('camera_missing','camera_stale','camera_gap','camera_bad_calibration_id',
        'actuator_missing','actuator_stale','actuator_gap','actuator_wrong_units')
TIMING_STAGES=('input_json_io','png_io_decode','baseline','camera_observation','actuator_observation','fusion','processed_window')
TIMING_SCOPE=('perf_counter_ns stages for JSON/file reads and PNG decoding, v03 baseline, RGB observation, '
    'drive observation, fusion, and their complete processed-window path. No inference cache; repeats read the same local '
    'sealed files, so OS file cache may be warm. Excludes rendering/acquisition, the200ms history wait, '
    'networking and robot actuation. Native scene/render generation wall time reported separately. Not hard real-time latency.')


def decode_png(path):
    """Decode only this fixture encoder's bounded RGB8/filter0 PNGs, checking CRC."""
    raw=Path(path).read_bytes()
    if len(raw)>8_000_000 or raw[:8]!=b'\x89PNG\r\n\x1a\n': raise ValueError('Unsupported PNG')
    offset=8; compressed=b''; width=height=None; ended=False
    while offset<len(raw):
        if offset+12>len(raw): raise ValueError('Truncated PNG')
        length=struct.unpack('>I',raw[offset:offset+4])[0]
        kind=raw[offset+4:offset+8]; data=raw[offset+8:offset+8+length]
        if offset+12+length>len(raw): raise ValueError('Truncated PNG chunk')
        crc=struct.unpack('>I',raw[offset+8+length:offset+12+length])[0]
        if zlib.crc32(kind+data)&0xffffffff!=crc: raise ValueError('PNG CRC mismatch')
        if kind==b'IHDR':
            if width is not None or offset!=8 or length!=13: raise ValueError('Invalid PNG header')
            width,height,depth,color,compress,filtering,interlace=struct.unpack('>IIBBBBB',data)
            if (width,height,depth,color,compress,filtering,interlace)!=(640,480,8,2,0,0,0):
                raise ValueError('Expected trusted camera RGB8 dimensions')
        elif kind==b'IDAT': compressed+=data
        elif kind==b'IEND':
            if length or offset+12!=len(raw): raise ValueError('Invalid PNG terminator')
            ended=True; break
        else: raise ValueError('Unexpected fixture PNG chunk')
        offset+=length+12
    if width is None or not ended: raise ValueError('Incomplete PNG')
    expected=height*(1+width*3)
    decoder=zlib.decompressobj(); pixels=decoder.decompress(compressed,expected+1)
    if len(pixels)!=expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Invalid compressed pixel length')
    rows=np.frombuffer(pixels,dtype=np.uint8).reshape(height,1+width*3)
    if np.any(rows[:,0]): raise ValueError('Only fixture PNG filter0 accepted')
    return rows[:,1:].reshape(height,width,3).copy()


def evaluate_case(root,case,*,timings=False):
    """Recompute from packets/pixels only. Case truth/scenario fields are never consumed."""
    root=Path(root); marks={}; started=time.perf_counter_ns(); tick=started
    primary=read_json(relative_file(root,case['primary_packet_file']))
    cp=read_json(relative_file(root,case['camera_packet_file']))
    ap=read_json(relative_file(root,case['actuator_packet_file']))
    contract=GoalContract(**case['contract']); now=case['now_s']
    end=time.perf_counter_ns(); marks['input_json_io']=(end-tick)/1e6; tick=end
    decoded=[]
    try:
        if type(cp) is not dict: raise ValueError('Expected camera packet object')
        frames=cp.get('frames',[])
        if type(frames) is not list or len(frames)>11: raise ValueError('Bounded camera list required')
        for f in frames: decoded.append(decode_png(relative_file(root,f['image'])))
    except (ValueError,KeyError,TypeError,zlib.error):
        decoded=None
    end=time.perf_counter_ns(); marks['png_io_decode']=(end-tick)/1e6; tick=end
    baseline=verify_goal(primary,contract=contract,now_s=now)
    end=time.perf_counter_ns(); marks['baseline']=(end-tick)/1e6; tick=end
    camera=camera_window(cp,decoded,now_s=now)
    end=time.perf_counter_ns(); marks['camera_observation']=(end-tick)/1e6; tick=end
    actuator=actuator_window(ap,now_s=now)
    end=time.perf_counter_ns(); marks['actuator_observation']=(end-tick)/1e6; tick=end
    fused=fuse(primary,baseline,camera,actuator,contract=contract)
    end=time.perf_counter_ns(); marks['fusion']=(end-tick)/1e6; marks['processed_window']=(end-started)/1e6
    result=dict(baseline=baseline,camera=camera,actuator=actuator,fused=fused)
    if timings: result['timing_ms']=marks
    return result


def _shift(packet,delta):
    p=deepcopy(packet)
    if 'history_start_s' in p: p['history_start_s']+=delta
    for item in p.get('samples',p.get('frames',[])): item['t_s']+=delta
    return p


def add_controls(root,native):
    root=Path(root); cases=list(native)
    def save(case,key,value):
        path='packets/'+case['id']+'-'+key+'.json'; write_json(root/path,value); case[key]=path
    for complete in [c for c in native if c['scenario']=='complete']:
        layout=complete['layout_id']; group={c['scenario']:c for c in native if c['layout_id']==layout}
        primary=read_json(relative_file(root,complete['primary_packet_file']))
        camera=read_json(relative_file(root,complete['camera_packet_file']))
        actuator=read_json(relative_file(root,complete['actuator_packet_file']))
        for name in NEGATIVES:
            case=deepcopy(group[name]); case.update(id=layout+'-forgery-'+name,cohort='primary_forgery',
                prescribed_intervention='Replace only primary observations with complete-scene readings aligned to the same decision time. Keep actual RGB/drive streams and physical truth.')
            save(case,'primary_packet_file',_shift(primary,case['now_s']-complete['now_s']))
            cases.append(case)
        for name in FAULTS:
            case=deepcopy(complete); case.update(id=layout+'-fault-'+name,cohort='sensor_fault',scenario=name,
                prescribed_intervention='Alter only '+name+'; physical state and primary packet unchanged.')
            is_camera=name.startswith('camera'); p=deepcopy(camera if is_camera else actuator)
            key='frames' if is_camera else 'samples'
            if name.endswith('_missing'): p[key]=[]
            elif name.endswith('_stale'): p=_shift(p,-.5)
            elif name.endswith('_gap'): del p[key][len(p[key])//2]
            elif name.endswith('_bad_calibration_id'): p['calibration_id']='untrusted-calibration'
            elif name.endswith('_wrong_units'): p['samples'][-1]['units']['gripper_effort']='kg'
            save(case,'camera_packet_file' if is_camera else 'actuator_packet_file',p)
            cases.append(case)
        for name in ('refreshed_camera_replay','forged_encoder_only','forged_effort_only','all_channels_forged'):
            source=group['unreleased' if name.startswith('forged_') else 'sustained_motion']
            case=deepcopy(source); case.update(id=layout+'-'+name,cohort='cross_channel',scenario=name,
                prescribed_intervention='Primary complete forgery plus '+name+'; underlying physical truth unchanged.')
            delta=case['now_s']-complete['now_s']; save(case,'primary_packet_file',_shift(primary,delta))
            if name in ('refreshed_camera_replay','all_channels_forged'):
                save(case,'camera_packet_file',_shift(camera,delta))
            if name=='all_channels_forged': save(case,'actuator_packet_file',_shift(actuator,delta))
            elif name.startswith('forged_'):
                p=read_json(relative_file(root,source['actuator_packet_file']))
                for real,good in zip(p['samples'],actuator['samples']):
                    if name=='forged_encoder_only':
                        real['finger_position_m']=good['finger_position_m']
                    else: real['gripper_effort_n']=good['gripper_effort_n']
                save(case,'actuator_packet_file',p)
            cases.append(case)
    return cases


def source_identity():
    identity=old.source_identity(); repo=Path(__file__).resolve().parents[2]
    names=['src/dvidia_training/'+n+'.py' for n in ('sensor_goal_observers','sensor_goal_verifier','sensor_goal_scenes','sensor_goal_benchmark')]
    names+=['pyproject.toml']
    for name in names:
        committed=subprocess.check_output(['git','show',identity['commit']+':'+name],cwd=repo)
        digest=sha(repo/name)
        if hashlib.sha256(committed).hexdigest()!=digest: raise ValueError('Commit frozen source first: '+name)
        identity['files'][name]=digest
    return identity


def _seal(root):
    write_json(root/'integrity.json',dict(schema='dvidia.sensor-goal-integrity/v1',files={
        str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='integrity.json'}))


def verify_integrity(root):
    root=Path(root); seal=read_json(root/'integrity.json')
    if seal.get('schema')!='dvidia.sensor-goal-integrity/v1' or not seal.get('files'): raise ValueError('Invalid input seal')
    for name,entry in seal['files'].items():
        p=relative_file(root,name)
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']: raise ValueError('Input integrity mismatch: '+name)
    if any(name not in seal['files'] for name in ('protocol.json','cases.json','generation.json')): raise ValueError('Unsealed run metadata')
    catalog=read_json(root/'cases.json')
    if catalog.get('status')=='generation_failed': raise ValueError('Cannot score failed native generation')
    for case in catalog['cases']:
        for key in ('primary_packet_file','camera_packet_file','actuator_packet_file','state_file','current_image','goal_image'):
            if case[key] not in seal['files']: raise ValueError('Unsealed case reference: '+str(case[key]))
        camera=read_json(relative_file(root,case['camera_packet_file']))
        for f in camera['frames']:
            if f['image'] not in seal['files']: raise ValueError('Unsealed camera frame')
    return seal


def generate(assets,output,*,seed=2026102007,count=8,repeats=3,warmups=1):
    from .sensor_goal_scenes import generate_native_cases
    from .local_goal_scenes import make_layouts
    output=Path(output)
    if output.exists(): raise ValueError('Use a fresh immutable output directory')
    if not 1<=count<=32 or not 1<=repeats<=20 or not 0<=warmups<=5: raise ValueError('Invalid bounded run size')
    protocol=dict(schema='dvidia.sensor-goal-protocol/v1',source=source_identity(),seed=seed,layouts=make_layouts(seed,count),
        layout_count=count,repeats=repeats,warmups=warmups,planned_cases=count*34,
        native_scenarios=['complete','occluded',*NEGATIVES],cohorts=list(COHORTS),faults=list(FAULTS),
        primary_contract=old.contract_description(),corroboration=specification(),timing_scope=TIMING_SCOPE,
        oracle='Frozen native _measure and observe across100ticks2ms; no observer features enter physical truth.',
        selection='All scheduled cases count; generation mismatches remain. Hard generation failure seals diagnostics and stops scoring.',
        development_exposure='Development uses seed2026101904. Final seed2026102007 was not inspected before source/protocol freeze.',
        scope='Synthetic 30g rigid shoe proxy, fixed PiPER and orthographic calibrated color observer. No learned detector, hardware validation, genuine sensor authentication or policy training.',provider_calls=0)
    write_json(output/'protocol.json',protocol); started=time.perf_counter()
    try:
        native=generate_native_cases(output,Path(assets),seed=seed,count=count)
        if len(native)!=count*12 or any(c.get('physical_truth') is None or c.get('primary_packet_file') is None for c in native):
            raise ValueError('Incomplete native case inventory')
        cases=add_controls(output,native)
        if len(cases)!=protocol['planned_cases']: raise ValueError('Wrong scheduled case inventory')
    except Exception as exc:
        write_json(output/'cases.json',dict(schema='dvidia.sensor-goal-cases/v1',status='generation_failed',cases=locals().get('native',[])))
        write_json(output/'generation.json',dict(status='generation_failed',error=str(exc),wall_seconds=time.perf_counter()-started))
        _seal(output); raise RuntimeError('Native generation failed; sealed diagnostics retained') from exc
    write_json(output/'cases.json',dict(schema='dvidia.sensor-goal-cases/v1',cases=cases))
    write_json(output/'generation.json',dict(status='completed',wall_seconds=time.perf_counter()-started,
        case_count=len(cases),invalid_cases=[c['id'] for c in cases if not c['generation_valid']]))
    _seal(output)
    return dict(run=str(output),cases=len(cases),generation_invalid=sum(not c['generation_valid'] for c in cases))


def score(rows,cases,*,repeats):
    if type(repeats) is not int or not 1<=repeats<=20: raise ValueError('Invalid repeat count')
    catalog={c['id']:c for c in cases}
    if len(catalog)!=len(cases) or len({r['id'] for r in rows})!=len(rows) or {r['id'] for r in rows}!=set(catalog):
        raise ValueError('Exactly one result required for every scheduled case')
    for r in rows:
        c=catalog[r['id']]
        if any(r.get(key)!=value for key,value in c.items()): raise ValueError('Result case differs from frozen inventory')
        if r['cohort'] not in COHORTS or type(r['physical_truth']['complete']) is not bool or type(r['generation_valid']) is not bool:
            raise ValueError('Invalid truth/cohort')
        if any(r[k]['verdict'] not in old.VERDICTS for k in ('baseline','fused')): raise ValueError('Invalid verdict')
        if r['baseline']['verdict']!='complete' and r['baseline']['verdict']!=r['fused']['verdict']:
            raise ValueError('Fusion cannot upgrade or change a noncomplete primary verdict')
        timings=r['timing_ms']
        if type(timings) is not list or len(timings)!=repeats: raise ValueError('Every case requires all timing repeats')
        for t in timings:
            if (type(t) is not dict or set(t)!=set(TIMING_STAGES)
                or any(type(v) not in (int,float) or not 0<=v<=1e9 or not math.isfinite(v) for v in t.values())):
                raise ValueError('Invalid timing stages')
    def group(sub):
        result=dict(planned=len(sub),generation_invalid=sum(not r['generation_valid'] for r in sub))
        for key in ('baseline','fused'):
            positive=[r for r in sub if r['physical_truth']['complete']]; negative=[r for r in sub if not r['physical_truth']['complete']]
            passed=[r for r in sub if r[key]['verdict']=='complete']
            result[key]=dict(verdict_counts={v:sum(r[key]['verdict']==v for r in sub) for v in old.VERDICTS},
                physical_true_completion=old.fraction(sum(r[key]['verdict']=='complete' for r in positive),len(positive)),
                physical_false_completion=old.fraction(sum(r[key]['verdict']=='complete' for r in negative),len(negative)),
                completion_precision=old.fraction(sum(r['physical_truth']['complete'] for r in passed),len(passed)),
                abstention=old.fraction(sum(r[key]['verdict']=='unknown' for r in sub),len(sub)))
        bad=[r for r in sub if not r['physical_truth']['complete'] and r['baseline']['verdict']=='complete']
        good=[r for r in sub if r['physical_truth']['complete'] and r['baseline']['verdict']=='complete']
        result['paired']=dict(baseline_false_completions=len(bad),baseline_true_completions=len(good),
            prevented_false_completions={v:sum(r['fused']['verdict']==v for r in bad) for v in ('unknown','incomplete')},
            lost_true_completions={v:sum(r['fused']['verdict']==v for r in good) for v in ('unknown','incomplete')})
        return result
    stages=rows[0]['timing_ms'][0] if rows else {}
    return dict(overall=group(rows),cohorts={c:group([r for r in rows if r['cohort']==c]) for c in COHORTS},
        scenarios={s:group([r for r in rows if r['scenario']==s]) for s in sorted({r['scenario'] for r in rows})},
        latency_ms={k:old.quantiles([t[k] for r in rows for t in r['timing_ms']]) for k in stages})


def evaluate(root):
    root=Path(root)
    if (root/'benchmark.json').exists(): raise ValueError('Results already exist')
    verify_integrity(root); protocol=read_json(root/'protocol.json'); cases=read_json(root/'cases.json')['cases']
    if source_identity()!=protocol['source']: raise ValueError('Source changed after freeze')
    if len(cases)!=protocol['planned_cases'] or len({c['id'] for c in cases})!=len(cases): raise ValueError('Inventory mismatch')
    rows=[]
    for case in cases:
        for _ in range(protocol['warmups']): evaluate_case(root,case)
        prior=None; durations=[]
        for _ in range(protocol['repeats']):
            result=evaluate_case(root,case,timings=True); durations.append(result.pop('timing_ms'))
            if prior is not None and prior!=result: raise ValueError('Nondeterministic result')
            prior=result
        rows.append(dict(**case,**prior,timing_ms=durations))
    result=dict(schema='dvidia.sensor-goal-results/v1',status='completed',source=protocol['source'],
        protocol_sha256=sha(root/'protocol.json'),input_integrity_sha256=sha(root/'integrity.json'),
        cases=rows,metrics=score(rows,cases,repeats=protocol['repeats']),timing_scope=TIMING_SCOPE,provider_calls=0,
        environment=dict(python=platform.python_version(),numpy=np.__version__,system=platform.system(),
            release=platform.release(),machine=platform.machine(),logical_cpus=os.cpu_count()))
    verify_integrity(root); write_json(root/'benchmark.json',result)
    return result['metrics']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); commands=p.add_subparsers(dest='command',required=True)
    g=commands.add_parser('generate');g.add_argument('--assets',required=True);g.add_argument('--output',required=True)
    g.add_argument('--seed',type=int,default=2026102007);g.add_argument('--count',type=int,default=8)
    g.add_argument('--repeats',type=int,default=3);g.add_argument('--warmups',type=int,default=1)
    e=commands.add_parser('evaluate');e.add_argument('--run',required=True)
    args=p.parse_args(argv)
    value=generate(args.assets,args.output,seed=args.seed,count=args.count,repeats=args.repeats,warmups=args.warmups) if args.command=='generate' else evaluate(args.run)
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__': main()
