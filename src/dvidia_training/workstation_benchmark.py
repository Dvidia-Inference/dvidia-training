"""Frozen native tracking and contact diagnostics for the live workstation.

Every planned attempt is retained. This protocol evaluates authored target
tracking and free-object mechanics, not autonomous shoe grasping or real transfer.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import time

from . import workstation as ws, reference_sim as sim


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def protocol(stability_seconds=600):
    if type(stability_seconds) is not int or not 10 <= stability_seconds <= 600:
        raise ValueError('Stability duration must be an integer from 10 through 600 seconds')
    targets = [('home', [.36,-.06,.27]), ('left', [.32,-.16,.24]),
               ('right', [.36,.10,.25]), ('near', [.27,-.04,.22]),
               ('far', [.47,-.04,.24]), ('high', [.36,-.06,.32]),
               ('low', [.36,-.06,.12]), ('shoe_approach', [.32,-.145,.16])]
    attempts = [{'id':name+'-'+str(repeat), 'kind':'tracking', 'target':position,
                 'seconds':4, 'jaw':.035} for repeat in range(2) for name,position in targets]
    attempts += [
        {'id':'jaw_close_open','kind':'jaw','target':list(ws.HOME_TARGET),'seconds':4,'jaw':.035},
        {'id':'shoe_nudge','kind':'push','target':list(ws.HOME_TARGET),'seconds':4,'jaw':.035},
        {'id':'table_disabled','kind':'gravity_control','target':list(ws.HOME_TARGET),'seconds':2,'jaw':.035},
        {'id':'sustained_hold','kind':'stability','target':list(ws.HOME_TARGET),'seconds':stability_seconds,'jaw':.035}]
    return {'schema':'dvidia.workstation-benchmark-protocol/v1','attempts':attempts,
        'scope':'Authored target tracking, jaw travel, free-object support and numerical stability; no learned grasp policy',
        'physics_dt_s':ws.TIMESTEP,'control_dt_s':ws.CONTROL_DT,
        'tracking_position_tolerance_m':.005,'tracking_orientation_tolerance_deg':3.,
        'jaw_endpoint_tolerance_m':.002,'push_impulse_ns':[0,.12,0],
        'tracking_deadline_s':4,'sample_interval_s':.02,'stability_sample_interval_s':.1,
        'timing_scope':'Native Python control interval including ten physics steps, forward kinematics and diagnostic extraction; rendering and network excluded',
        'repeats_scope':'Two deterministic resets per target; not independent real-world reliability trials'}


def hashes():
    root=Path(__file__).parent
    return {name:sha256((root/name).read_bytes()).hexdigest() for name in
            ['workstation.py','workstation_benchmark.py','asset_library.py','asset_catalog.json','robot_profiles.py']}


def peak_rss():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    except ImportError:
        return None


def benchmark(assets, output, *, stability_seconds=600, source_commit=None):
    destination=Path(output)
    if destination.exists() or destination.is_symlink():raise ValueError('Benchmark output must be fresh')
    if source_commit is not None and (len(source_commit)!=40 or any(c not in '0123456789abcdef' for c in source_commit)):
        raise ValueError('Source commit must be a complete Git SHA')
    mj,np=sim._native_imports()
    plan=protocol(stability_seconds)
    plan.update(source_commit=source_commit, implementation_sha256=hashes())
    destination.mkdir(parents=True)
    (destination/'protocol.json').write_bytes(encode(plan))
    records=[]; started=time.perf_counter()
    all_durations=[]
    for case in plan['attempts']:
        begin=time.perf_counter(); warnings=[]
        rows=[]; durations=[]; errors=[]; valid=True; controls_within_limits=True
        arm_environment_contact_samples=0; maximum_arm_environment_force_n=0.
        try:
            with sim.native_warnings(warnings):
                model=ws.build_model(assets); data=ws.initialize(model)
                tcp,shoe=ws._ids(model)
                if case['kind']=='gravity_control':
                    table=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'workstation_table_top')
                    model.geom_contype[table]=0;model.geom_conaffinity[table]=0
                    mj.mj_forward(model,data)
                initial=ws.state(model,data,case['target'])
                rows.append({'step':0,**initial})
                jaw_closed=None; peak_shoe_displacement=0.; push_origin=None
                for step in range(round(case['seconds']/ws.CONTROL_DT)):
                    jaw=0. if case['kind']=='jaw' and step<100 else case['jaw']
                    impulse=plan['push_impulse_ns'] if case['kind']=='push' and step==50 else None
                    if impulse is not None:push_origin=data.xpos[shoe].copy()
                    start=time.perf_counter_ns()
                    state=ws.step(model,data,case['target'],jaw,impulse_xyz=impulse)
                    durations.append((time.perf_counter_ns()-start)/1e6)
                    controls_within_limits &= bool(np.all(data.ctrl>=model.actuator_ctrlrange[:,0]) and np.all(data.ctrl<=model.actuator_ctrlrange[:,1]))
                    for contact in state['contacts']:
                        body1,body2=(int(model.geom_bodyid[contact[name]]) for name in ('geom1','geom2'))
                        if ((body1==0 and body2 not in (0,1,shoe)) or (body2==0 and body1 not in (0,1,shoe))):
                            arm_environment_contact_samples+=1
                            maximum_arm_environment_force_n=max(maximum_arm_environment_force_n,contact['normal_force_n'])
                    if case['kind']=='jaw' and step==99:jaw_closed=float(data.qpos[6])
                    if push_origin is not None:peak_shoe_displacement=max(peak_shoe_displacement,float(np.linalg.norm(data.xpos[shoe]-push_origin)))
                    if case['kind']!='stability' or (step+1)%5==0:
                        rows.append({'step':step+1,**state})
                final=ws.state(model,data,case['target'])
                dot=abs(float(np.dot(final['tcp_quaternion'],ws.ORIENTATION)))
                angle=math.degrees(2*math.acos(min(1.,dot)))
                result={'position_error_m':final['position_error_m'],'orientation_error_deg':angle,
                        'final_shoe_position_m':final['shoe_position'],'final_jaw_m':float(data.qpos[6]),
                        'closed_jaw_m':jaw_closed,'peak_post_push_displacement_m':peak_shoe_displacement,
                        'controls_within_limits':bool(controls_within_limits),
                        'simulated_seconds':float(data.time),
                        'arm_environment_loaded_contact_samples':arm_environment_contact_samples,
                        'maximum_arm_environment_normal_force_n':maximum_arm_environment_force_n}
                if case['kind']=='tracking':passed=final['position_error_m']<=.005 and angle<=3
                elif case['kind']=='jaw':passed=abs(jaw_closed)<=.002 and abs(float(data.qpos[6])-.035)<=.002
                elif case['kind']=='push':passed=peak_shoe_displacement>.001
                elif case['kind']=='gravity_control':passed=final['shoe_position'][2]<-.3
                else:passed=controls_within_limits and final['position_error_m']<=.005
                valid=(not warnings and not any(w.number for w in data.warning)
                       and all(np.isfinite(value).all() for value in (data.qpos,data.qvel,data.qacc,data.ctrl))
                       and abs(float(data.time)-case['seconds'])<1e-6)
                passed=passed and controls_within_limits
        except Exception as exc:
            valid=False;passed=False;result={};errors.append(type(exc).__name__+': '+str(exc))
        records.append({'case':case,'valid':valid,'passed':bool(passed and valid),
                        'warnings':warnings,'errors':errors,'result':result,
                        'wall_seconds_including_model':time.perf_counter()-begin,
                        'control_interval_compute_ms':durations,'trajectory':rows})
        all_durations.extend(durations)
        # Preserve each completed attempt even if a later attempt is interrupted.
        (destination/'attempts.json').write_bytes((json.dumps(records, separators=(',', ':'), allow_nan=False)+'\n').encode())
    array=np.asarray(all_durations)
    tracking=[item for item in records if item['case']['kind']=='tracking']
    measured_tracking=[r['result']['position_error_m'] for r in tracking if 'position_error_m' in r['result']]
    report={'schema':'dvidia.workstation-benchmark/v1','source_commit':source_commit,
        'protocol_sha256':sha256((destination/'protocol.json').read_bytes()).hexdigest(),
        'native_version':mj.__version__,'physical_robot_ready':False,'learned_policy':False,
        'attempts':len(records),'valid_attempts':sum(r['valid'] for r in records),
        'tracking':{'passed':sum(r['passed'] for r in tracking),'attempts':len(tracking),
                    'worst_position_error_m':max(measured_tracking,default=None),
                    'unmeasured_attempts':len(tracking)-len(measured_tracking)},
        'diagnostics':{r['case']['kind']:{'passed':r['passed'],**r['result']} for r in records if r['case']['kind']!='tracking'},
        'timing':{'wall_seconds':time.perf_counter()-started,
            'compute_ms_p50':float(np.percentile(array,50)) if len(array) else None,
            'compute_ms_p95':float(np.percentile(array,95)) if len(array) else None,
            'compute_ms_p99':float(np.percentile(array,99)) if len(array) else None,
            'compute_ms_max':float(array.max()) if len(array) else None,
            'intervals_over_20ms':int(np.sum(array>20)),'intervals':len(array)},
        'host':{'os':platform.system(),'os_release':platform.release(),'machine':platform.machine(),'logical_cpus':os.cpu_count(),
                'peak_process_rss_bytes':peak_rss()},
        'scope':plan['scope'],'timing_scope':plan['timing_scope'],
        'sampling_scope':'Full qpos, ctrl, body poses and loaded contact diagnostics at 20 ms, or 100 ms for sustained hold; not an exhaustive every-physics-tick contact record'}
    (destination/'report.json').write_bytes(encode(report))
    files={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p.read_bytes()).hexdigest()} for p in sorted(destination.iterdir()) if p.is_file()}
    (destination/'integrity.json').write_bytes(encode({'schema':'dvidia.workstation-benchmark-integrity/v1','files':files}))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stability-seconds',type=int,default=600);parser.add_argument('--source-commit')
    args=parser.parse_args()
    print(json.dumps(benchmark(args.assets,args.output,stability_seconds=args.stability_seconds,source_commit=args.source_commit),indent=2))


if __name__=='__main__':main()
