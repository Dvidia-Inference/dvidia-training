"""Reproduce direct-grasp shape, load and mounted-tool controls on CPU."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
import time

from . import prerequisite_benchmark as evidence
from . import asset_preview, reference_replay, reference_sim as sim

SCHEMA = 'dvidia.graspability/v3'
SCOPE = ('Authored direct in-reach pickup with pinned PiPER drives, rigid object proxies and '
         'privileged native observations. Shape, explicit load and mounted-tool conditions are '
         'diagnostic controls. No learned skill, tool detachment, retrieval or hardware qualification.')


def implementation_hashes():
    root=Path(__file__).parent
    files=('graspability.py','graspability_protocol.py','graspability_protocol_v03.json',
           'graspability_benchmark.py','workstation.py','asset_library.py','asset_catalog.json',
           'robot_profiles.py','robot_profiles.lock.json','prerequisite_benchmark.py','asset_preview.py',
           'reference_sim.py','graspability_lab.html','graspability_lab.js','graspability_lab.bundle.js')
    return {name:sha256((root/name).read_bytes()).hexdigest() for name in files}


def protocol(split='development'):
    from . import graspability_protocol
    return graspability_protocol.protocol(split)


def robot_actuator_fingerprint(model):
    """Hash the robot joint/drive contract, excluding object/tool morphology.

    The pinned PiPER has eight robot joint coordinates and seven actuator inputs.
    Body inertia is intentionally outside this fingerprint: adding the rake
    changes physical loading. Full compiled model hashes identify each condition.
    """
    mj,_=sim._native_imports()
    names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,i) for i in range(8)]
    if model.nu!=7 or names!=['joint'+str(i) for i in range(1,9)]:
        raise ValueError('Expected pinned eight-coordinate/seven-input PiPER')
    fields={name:getattr(model,name).tolist() for name in (
        'actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange',
        'actuator_gear','actuator_trnid','actuator_trntype','actuator_dyntype',
        'actuator_gaintype','actuator_biastype','actuator_ctrllimited','actuator_forcelimited',
        'actuator_dynprm','eq_type','eq_obj1id','eq_obj2id','eq_data','eq_solref','eq_solimp')}
    fields.update({name:getattr(model,name)[:8].tolist() for name in (
        'jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_stiffness',
        'jnt_solref','jnt_solimp','dof_damping','dof_frictionloss','dof_armature')})
    fields.update(schema='dvidia.piper-joint-actuator-fingerprint/v1',joint_names=names,
                  gravity=model.opt.gravity.tolist(),timestep=float(model.opt.timestep))
    return sha256(evidence.encode(fields)).hexdigest()


def summarize(records):
    keys=sorted({(r['case']['object_shape'],r['case']['mass'],r['case']['tool_condition']) for r in records})
    groups=[]
    for shape,mass,tool in keys:
        rows=[r for r in records if (r['case']['object_shape'],r['case']['mass'],r['case']['tool_condition'])==(shape,mass,tool)]
        outcomes={}
        for row in rows:
            outcome=row.get('scored_outcome','failed') if row['valid'] else 'invalid_attempt'
            outcomes[outcome]=outcomes.get(outcome,0)+1
        groups.append(dict(task='in-reach-graspability',policy='shared-v03',object_shape=shape,mass_kg=mass,
            tool_condition=tool,attempts=len(rows),valid=sum(r['valid'] for r in rows),
            successes=sum(r['success'] and r['valid'] for r in rows),outcomes=outcomes,
            simulation_seconds=sum(r['metrics'].get('simulation_seconds',0.) for r in rows),
            wall_seconds=sum(r['wall_seconds'] for r in rows)))
    return groups


def paired_comparisons(records):
    def key(row):
        return evidence.encode({k:v for k,v in row['case'].items() if k!='tool_condition'})
    absent={key(r):r for r in records if r['case']['tool_condition']=='absent'}
    mounted=[r for r in records if r['case']['tool_condition']=='mounted']
    if len(absent)!=len(mounted) or len(records)!=2*len(mounted):
        raise ValueError('Each declared object/load case requires one matched tool pair')
    pairs=[]
    seen=set()
    for a in mounted:
        identity=key(a)
        if identity not in absent or identity in seen:
            raise ValueError('Missing or duplicated tool-condition pair')
        seen.add(identity);b=absent[identity]
        fingerprint=a.get('robot_actuator_sha256')
        same=bool(fingerprint and fingerprint==b.get('robot_actuator_sha256'))
        valid=bool(a['valid'] and b['valid'] and same)
        pairs.append(dict(task='in-reach-graspability',case_id=a['case']['id'],object_shape=a['case']['object_shape'],
            mass_kg=a['case']['mass'],mounted_id=a['id'],absent_id=b['id'],valid_pair=valid,
            mounted_success=a['success'],absent_success=b['success'],same_robot_actuators=same,
            absent_gain=bool(valid and not a['success'] and b['success']),
            absent_loss=bool(valid and a['success'] and not b['success']),
            comparison_scope='Same declared object, load, pose and controller. Tool is present or absent from reset; its collision geometry and loading differ.'))
    return pairs


def runtime_files():
    root=Path(__file__).parent
    return {'index.html':(root/'graspability_lab.html').read_bytes(),
            'graspability_lab.bundle.js':(root/'graspability_lab.bundle.js').read_bytes(),
            'reference_replay.third-party.txt':(root/'reference_replay.third-party.txt').read_bytes()}


def benchmark(assets,output,*,split='development',source_commit=None):
    from . import graspability,graspability_protocol
    destination=Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Benchmark output must be a fresh directory')
    if source_commit is not None and re.fullmatch(r'[a-f0-9]{40}',source_commit) is None:
        raise ValueError('Source commit must be a full Git SHA')
    plan=protocol(split)
    attempts=plan['attempts']
    ids=[row.get('id') for row in attempts]
    if not 1<=len(attempts)<=100 or any(not isinstance(v,str) or not re.fullmatch(r'[a-z0-9_-]{1,160}',v) for v in ids) or len(set(ids))!=len(ids):
        raise ValueError('Protocol attempts require distinct bounded portable IDs')
    plan.update(schema=SCHEMA,source_commit=source_commit,implementation_sha256=implementation_hashes())
    runtime=runtime_files();mj,np=sim._native_imports()
    provenance,notices=asset_preview._licenses('piper',assets)
    destination.mkdir(parents=True)
    evidence._write(destination,'protocol.json',evidence.encode(plan))
    source=dict(commit=source_commit,protocol_sha256=sha256(evidence.encode(plan)).hexdigest(),
                mujoco=mj.__version__,numpy=np.__version__,robot_assets=provenance)
    records=[]
    for attempt in attempts:
        start=time.perf_counter()
        row=dict(id=attempt['id'],task=attempt['task'],policy=attempt['policy'],case=attempt['case'],
                 valid=False,success=False,metrics={},events=[],errors=[])
        try:
            result=graspability.run_case(assets,attempt['case'],tool=attempt['case']['tool_condition'],record=True)
            row['wall_seconds']=time.perf_counter()-start
            if type(result['valid']) is not bool or type(result['success']) is not bool:
                raise ValueError('Attempt validity and success must be explicit booleans')
            row.update(valid=result['valid'],metrics=result['metrics'],events=result['events'],
                       controller_reported_success=result['success'])
            for name in ('native_trace','native_trace_columns','control_compute_ms','warnings'):
                if name in result:row[name]=result[name]
            row['robot_actuator_sha256']=robot_actuator_fingerprint(result['model'])
            score=graspability_protocol.audit_result(result)
            score['compiled_model_sha256']=sim.model_digest(result['model'])
            row['independent_score']=score
            row['success']=bool(score['task_success'] and row['valid'])
            outcome=row['metrics'].get('outcome','incomplete')
            row['scored_outcome']='completed' if row['success'] else 'retention_rejected' if outcome=='completed' else outcome
            row['replay']=evidence.export_trial(destination,attempt['id'],result,source)
            evidence.encode(row)
        except Exception as exc:
            row.update(valid=False,success=False)
            row['errors'].append(type(exc).__name__+': attempt, independent audit or export failed')
            try:evidence.encode(row)
            except (TypeError,ValueError):
                row.update(metrics={},events=[])
                for name in ('native_trace','native_trace_columns','control_compute_ms','warnings','independent_score'):
                    row.pop(name,None)
        row.setdefault('wall_seconds',time.perf_counter()-start)
        row['export_seconds']=max(0.,time.perf_counter()-start-row['wall_seconds'])
        evidence._write(destination,'trials/'+attempt['id']+'.json',evidence.encode(row))
        records.append({k:v for k,v in row.items() if k!='native_trace'})
        print(json.dumps(dict(attempt=row['id'],valid=row['valid'],success=row['success'],
            outcome=row.get('scored_outcome','invalid_attempt') if row['valid'] else 'invalid_attempt',errors=row['errors'])),flush=True)
    report=dict(schema=SCHEMA,scope=SCOPE,source=source,split=split,
        host=dict(system=platform.system(),release=platform.release(),machine=platform.machine(),python=platform.python_version(),numpy=np.__version__,mujoco=mj.__version__,
                  compute='CPU native simulation; browser replay needs WebGL2. No GPU required for native benchmark.'),
        learned_policy=False,hardware_validated=False,attempts=records,groups=summarize(records),
        paired_comparisons=paired_comparisons(records),
        timing_scope='Native wall time includes model setup and recording. Export and independent auditing are separate. Rendering, downloads and hardware sensing/actuation are excluded.',
        success_scope='Every result is independently rescored from dense native traces; author-reported completion cannot override it.',
        robot_fingerprint_scope='Compiled robot joints, actuator settings and coupling; excludes tool and object inertia. Full model hashes are condition-specific.')
    evidence._write(destination,'benchmark.json',evidence.encode(report))
    for name,content in runtime.items():evidence._write(destination,name,content)
    for name,content in notices.items():evidence._write(destination,name,content)
    evidence._write(destination,'LICENSE',reference_replay.MIT_LICENSE.encode())
    evidence._write(destination,'README.md',('# DVIDIA graspability v0.3\n\n'+SCOPE+'\n\n'
        'Three rigid shapes, explicit 30 g and 300 g loads, and mounted versus absent rake conditions. '
        'Tool-free means absent from reset: no in-flight detachment or retrieval is simulated. '
        'A 30 g shoe-shaped control is not a physical shoe mass measurement.\n\n'
        'All attempts and dense contact traces are retained, including failures. The independent score '
        'requires 500 ms continuous bilateral finger contact, object-origin rise >=35 mm from the initial '
        'verified settled state, and no world, rake or non-finger robot support.\n\n'
        'Serve this folder using `python -m http.server 8000`. Browser visualization replays recorded poses; '
        'fresh physics runs through Python.\n\n'
        'Reproduce from the exact source commit, install `.[arm]`, fetch assets using '
        '`dvidia-robot-sim fetch piper --output robot-assets`, then run '
        '`dvidia-graspability-benchmark --assets robot-assets --output fresh-results --split '+split+'`.\n\n'
        'Pinned model gains are not measured hardware calibration. Collision geometry is a rigid approximation. '
        'CPU-only native benchmark; WebGL2 replay. Minimum RAM and real robot transfer are not established.\n\n'
        'DVIDIA code and original synthetic object primitives: MIT. Robot, Three.js and font notices retained. '
        'No trademark rights are granted.\n').encode())
    evidence.seal(destination)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=('development','heldout'),default='development')
    parser.add_argument('--source-commit');args=parser.parse_args()
    report=benchmark(args.assets,args.output,split=args.split,source_commit=args.source_commit)
    print(json.dumps({'groups':report['groups']}))
    return 0 if all(row['valid'] for row in report['attempts']) else 1


if __name__=='__main__':raise SystemExit(main())
