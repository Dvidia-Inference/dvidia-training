"""Passive CPU physics checks for the deliberately bounded USD workstation.

USD variants change authored mass, friction and drop height. Negative controls
disable support, gravity or friction explicitly. No arm skill is trained here.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import gzip
import json
import math
from pathlib import Path
import platform
import time

from . import usd_scene as bridge
from . import reference_sim as sim
from . import asset_preview

SCHEMA='dvidia.usd-workstation-benchmark/v1'
DT=.001
SOURCE_FILES=('usd_scene.py','usd_scene_benchmark.py','asset_library.py','asset_catalog.json',
              'robot_profiles.py','robot_profiles.lock.json','reference_sim.py','reference_assets.lock.json',
              'asset_preview.py','reference_replay.py')


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def identity(raw):return dict(bytes=len(raw),sha256=sha256(raw).hexdigest())


def write(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(encoded(raw) if isinstance(raw,dict) else raw)


def roster():
    cases=[]
    for mass in (.04,.08,.16):
        for friction in (.1,.5,1.):
            for height in (.05,.10,.20):
                cases.append(dict(kind='drop',mass_kg=mass,friction=friction,height_m=height,control='normal'))
            cases.append(dict(kind='slide',mass_kg=mass,friction=friction,height_m=0.,control='normal'))
    for height in (.05,.10,.20):
        for control in ('table_disabled','gravity_disabled'):
            cases.append(dict(kind='drop',mass_kg=.04,friction=.5,height_m=height,control=control))
    for mass in (.04,.08,.16):
        cases.append(dict(kind='slide',mass_kg=mass,friction=0.,height_m=0.,control='friction_disabled'))
    return [dict(id=f'case-{i:02}',seed=301729,**case) for i,case in enumerate(cases)]


def variant(example,output,case):
    """Author each variation through the real USD API; retain exact input bytes."""
    from pxr import Sdf,Usd,UsdPhysics,Gf
    source=Path(example);recipe=bridge.load_recipe(source)
    raw=(source.parent/recipe['stage']).read_bytes()
    if sha256(raw).hexdigest()!=recipe['stage_sha256']:raise ValueError('Example stage changed.')
    layer=Sdf.Layer.CreateAnonymous('workstation.usda');layer.ImportFromString(raw.decode())
    stage=Usd.Stage.Open(layer,load=Usd.Stage.LoadNone)
    obj=stage.GetPrimAtPath('/World/Object');half=.02
    mass=UsdPhysics.MassAPI(obj);mass.GetMassAttr().Set(case['mass_kg'])
    inertia=case['mass_kg']*(.04**2+.04**2)/12
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(inertia,inertia,inertia))
    center=(.55,.15,half+case['height_m']) if case['kind']=='drop' else (.35,.25,half)
    obj.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*center))
    for name in ('TableMaterial','ObjectMaterial'):
        material=UsdPhysics.MaterialAPI(stage.GetPrimAtPath('/World/'+name))
        material.GetStaticFrictionAttr().Set(case['friction']);material.GetDynamicFrictionAttr().Set(case['friction'])
    raw=layer.ExportToString().encode()
    recipe['stage_sha256']=sha256(raw).hexdigest()
    recipe['reset']=dict(seed=case['seed'],object_xy_jitter_m=[0.,0.],object_yaw_jitter_rad=0.)
    recipe['goal']['object_center_m']=[center[0],center[1],half]
    output=Path(output);write(output/'workstation.usda',raw);write(output/'recipe.json',recipe)
    return output/'recipe.json'


def sources():
    return {name:identity(Path(__file__).with_name(name).read_bytes()) for name in SOURCE_FILES}


def prepare(example,output):
    root=Path(output)
    if root.exists():raise ValueError('Use a fresh benchmark directory; attempts are never overwritten.')
    recipe=bridge.load_recipe(example);stage=bridge.inspect_stage(Path(example).parent/recipe['stage'])
    if stage['stage_sha256']!=recipe['stage_sha256']:raise ValueError('Example stage SHA differs.')
    if (stage['object']['half_size_m']!=[.02,.02,.02] or stage['object']['quaternion_wxyz']!=[1,0,0,0]
            or stage['table']['half_size_m']!=[.6,.4,.04] or stage['table']['center_m']!=[.45,0,-.04]):
        raise ValueError('This analytic benchmark requires the canonical40mm cube and workstation table.')
    protocol=dict(schema=SCHEMA,cases=roster(),source=sources(),timestep_s=DT,
        example_recipe=identity(Path(example).read_bytes()),example_stage=identity((Path(example).parent/recipe['stage']).read_bytes()),
        scope='Passive rigid-box scene conversion, drop/support and Coulomb sliding. Arm holds native home controls; no policy, shoe grasp or video learning.',
        thresholds=dict(drop_early_position_error_m=.001,drop_final_height_error_m=.001,
            drop_final_speed_m_s=.02,normal_slide_final_speed_m_s=.02,
            sliding_distance_relative_error=.35,sliding_distance_absolute_floor_m=.002,
            initial_slide_speed_tolerance_m_s=.005,negative_slide_final_speed_m_s=.30),
        duration_s=dict(drop=2.,slide=1.),slide_initial_speed_m_s=.4,slide_pre_settle_s=.5,
        negative_controls='Disable compiled tabletop collision or gravity after verified construction; friction-disabled is authored in USD for both bodies. All controls are declared and recorded.',
        goal_semantics='Independent geometry, velocities, table contact and no robot contact over250ms; this verifies passive settling, not a manipulation skill.',
        timing_scope='One CPU worker; shared-host load may overlap frozen grip confirmation and publication. Includes model loading/validation; excludes downloads, camera and physical actuators.',
        physics_calibrated=False,learned_skill=False,physical_robot_ready=False)
    root.mkdir(parents=True)
    for name in SOURCE_FILES:write(root/'source'/name,Path(__file__).with_name(name).read_bytes())
    write(root/'protocol.json',protocol)
    for case in protocol['cases']:variant(example,root/'cases'/case['id']/'input',case)
    write(root/'input-inventory.json',dict(schema=SCHEMA,files={str(p.relative_to(root)):identity(p.read_bytes()) for p in sorted((root/'cases').rglob('*')) if p.is_file()}))
    write(root/'preparation-seal.json',dict(schema=SCHEMA,protocol=identity((root/'protocol.json').read_bytes()),
        inputs=identity((root/'input-inventory.json').read_bytes())))
    return protocol


def _row(model,data,body,obj,table,mj,np):
    velocity=np.zeros(6);mj.mj_objectVelocity(model,data,mj.mjtObj.mjOBJ_BODY,body,velocity,0)
    pairs=[(int(c.geom1),int(c.geom2)) for c in data.contact[:data.ncon]]
    return dict(t_s=float(data.time),position_m=data.xpos[body].tolist(),linear_velocity_m_s=velocity[3:].tolist(),
        angular_velocity_rad_s=velocity[:3].tolist(),table_contact=any(set(p)=={obj,table} for p in pairs),
        robot_contact=any(obj in p and table not in p for p in pairs))


def score(case,rows,protocol):
    if not rows:raise ValueError('A complete native trajectory is required.')
    threshold=protocol['thresholds'];first,last=rows[0],rows[-1]
    half=.02;g=9.81
    finite=all(math.isfinite(x) for row in rows for name in ('position_m','linear_velocity_m_s','angular_velocity_rad_s') for x in row[name])
    duration=protocol['duration_s'][case['kind']]
    cadence=(len(rows)==round(duration/.01)+1 and all(abs(r['t_s']-i*.01)<1e-8 for i,r in enumerate(rows)))
    common=dict(finite=finite,complete_cadence=cadence,robot_contact_absent=not any(r['robot_contact'] for r in rows))
    metrics={}
    if case['kind']=='drop':
        early=min(rows,key=lambda r:abs(r['t_s']-.05))
        analytic=first['position_m'][2]-.5*g*early['t_s']**2
        metrics=dict(early_position_error_m=abs(early['position_m'][2]-analytic),
            final_height_m=last['position_m'][2],final_speed_m_s=math.dist(last['linear_velocity_m_s'],[0,0,0]))
        if case['control']=='normal':
            checks=dict(**common,freefall_accuracy=metrics['early_position_error_m']<=threshold['drop_early_position_error_m'],
                resting_height=abs(last['position_m'][2]-half)<=threshold['drop_final_height_error_m'],
                stopped=metrics['final_speed_m_s']<=threshold['drop_final_speed_m_s'],supported=last['table_contact'])
        elif case['control']=='table_disabled':
            checks=dict(**common,missing_support_detected=last['position_m'][2]<-.1 and not last['table_contact'])
        else:checks=dict(**common,missing_gravity_detected=abs(last['position_m'][2]-first['position_m'][2])<.001 and not last['table_contact'])
    else:
        displacement=last['position_m'][0]-first['position_m'][0]
        ideal=None if case['friction']==0 else protocol['slide_initial_speed_m_s']**2/(2*g*case['friction'])
        metrics=dict(displacement_m=displacement,ideal_stopping_distance_m=ideal,
            stopping_distance_error_m=None if ideal is None else abs(displacement-ideal),final_speed_m_s=math.dist(last['linear_velocity_m_s'],[0,0,0]))
        initial_velocity=math.dist(first['linear_velocity_m_s'],[protocol['slide_initial_speed_m_s'],0,0])<=threshold['initial_slide_speed_tolerance_m_s']
        if case['control']=='normal':checks=dict(**common,initial_velocity=initial_velocity,supported=last['table_contact'],stopped=metrics['final_speed_m_s']<threshold['normal_slide_final_speed_m_s'],
            sliding_accuracy=metrics['stopping_distance_error_m']<=max(threshold['sliding_distance_absolute_floor_m'],threshold['sliding_distance_relative_error']*ideal),
            moved=displacement>=.25*ideal)
        else:checks=dict(**common,initial_velocity=initial_velocity,missing_friction_detected=metrics['final_speed_m_s']>=threshold['negative_slide_final_speed_m_s'] and displacement>.30)
    return dict(valid=finite and cadence,checks=checks,success=all(checks.values()),metrics=metrics)


def run(output,assets):
    root=Path(output);protocol=json.loads((root/'protocol.json').read_text())
    seal=json.loads((root/'preparation-seal.json').read_text())
    if seal!=dict(schema=SCHEMA,protocol=identity((root/'protocol.json').read_bytes()),inputs=identity((root/'input-inventory.json').read_bytes())):
        raise ValueError('Frozen protocol or input inventory changed.')
    if protocol['source']!=sources() or protocol['cases']!=roster():raise ValueError('Frozen source/roster changed.')
    if ({p.name for p in (root/'source').iterdir()}!=set(SOURCE_FILES)
            or any(p.is_symlink() for p in (root/'source').iterdir())
            or any(identity((root/'source'/name).read_bytes())!=want for name,want in protocol['source'].items())):
        raise ValueError('Frozen source snapshot changed.')
    inventory=json.loads((root/'input-inventory.json').read_text())
    if ({p.name for p in (root/'cases').iterdir()}!={case['id'] for case in protocol['cases']}
            or any({p.name for p in (root/'cases'/case['id']/'input').iterdir()}!={'recipe.json','workstation.usda'}
                   for case in protocol['cases'])):
        raise ValueError('Frozen case/input directory membership differs.')
    expected_inputs={str(p.relative_to(root)) for p in (root/'cases').glob('*/input/*') if p.is_file()}
    if (set(inventory['files'])!=expected_inputs or len(expected_inputs)!=len(protocol['cases'])*2
            or any(p.is_symlink() for p in (root/'cases').rglob('*'))):
        raise ValueError('Frozen input membership differs or contains symlinks.')
    for name,want in inventory['files'].items():
        if identity((root/name).read_bytes())!=want:raise ValueError('Frozen USD input changed.')
    if any({p.name for p in (root/'cases'/case['id']).iterdir()}!={'input'} for case in protocol['cases']):
        raise ValueError('A benchmark attempt already started; automatic rerun refused before any native work.')
    write(root/'execution-intent.json',dict(schema=SCHEMA,protocol=seal['protocol'],inputs=seal['inputs'],scheduled=len(protocol['cases']),worker_count=1))
    mj,np=sim._native_imports();results=[];geometry=None;scene=None
    for case in protocol['cases']:
        folder=root/'cases'/case['id']
        if (folder/'started.json').exists():raise ValueError('An attempt started already; automatic rerun refused.')
        write(folder/'started.json',dict(case=case,protocol_sha256=sha256((root/'protocol.json').read_bytes()).hexdigest(),source=sources()))
        began=time.perf_counter();model,receipt=bridge.build_scene(folder/'input'/'recipe.json',assets)
        data=bridge.initialize(model,receipt['recipe']);body=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'usd_object')
        obj=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'usd_object_geom');table=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'usd_table')
        joint=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'usd_object_free');dof=int(model.jnt_dofadr[joint])
        if case['control']=='table_disabled':model.geom_contype[table]=model.geom_conaffinity[table]=0
        if case['control']=='gravity_disabled':model.opt.gravity[:]=0
        warnings=[]
        with sim.native_warnings(warnings):
            if case['kind']=='slide':
                for _ in range(round(protocol['slide_pre_settle_s']/DT)):mj.mj_step(model,data)
                data.qvel[dof:dof+6]=0;data.qvel[dof]=protocol['slide_initial_speed_m_s'];data.time=0;mj.mj_forward(model,data)
            frames=[];rows=[];monitor=bridge.GoalMonitor(receipt['recipe']);goal=None
            steps=round(protocol['duration_s'][case['kind']]/DT)
            for tick in range(steps+1):
                goal=monitor.update(model,data)
                if tick%10==0:
                    rows.append(_row(model,data,body,obj,table,mj,np))
                    frames.append(np.concatenate((data.xpos,data.xquat),axis=1).astype('<f4'))
                if tick<steps:
                    mj.mj_step(model,data);mj.mj_forward(model,data)
        evaluated=score(case,rows,protocol)
        evaluated['checks']['no_native_warning']=not warnings and not any(int(w.number) for w in data.warning)
        evaluated['success']=all(evaluated['checks'].values())
        evaluated.update(case=case,native_goal=goal,wall_seconds=time.perf_counter()-began,native_warnings=warnings,
            compiled_property_checks=receipt['compiled_checks'],native_timestep_s=float(model.opt.timestep),
            control=case['control'],native_model_sha256=sim.model_digest(model))
        write(folder/'receipt.json',receipt);write(folder/'trajectory.json',dict(schema=SCHEMA,rows=rows))
        packed=gzip.compress(np.asarray(frames,dtype='<f4').tobytes(),mtime=0);write(folder/'poses.bin.gz',packed)
        write(folder/'result.json',evaluated);results.append(evaluated)
        if scene is None:
            scene,geometry=asset_preview._scene(model,dict(recipe_sha256=receipt['recipe_sha256'],scope=protocol['scope']))
            scene.update(schema='dvidia.usd-native-replay/v1',scope=protocol['scope'])
        print(json.dumps(dict(case=case['id'],kind=case['kind'],control=case['control'],valid=evaluated['valid'],passed=evaluated['success'])),flush=True)
    if sources()!=protocol['source']:raise ValueError('Benchmark source changed during execution.')
    if (identity((root/'protocol.json').read_bytes())!=seal['protocol'] or identity((root/'input-inventory.json').read_bytes())!=seal['inputs']
            or any(identity((root/name).read_bytes())!=want for name,want in inventory['files'].items())):
        raise ValueError('Frozen protocol or input changed during execution.')
    summary=dict(schema=SCHEMA,scheduled=len(results),valid=sum(r['valid'] for r in results),passed=sum(r['success'] for r in results),
        normal_scheduled=sum(r['case']['control']=='normal' for r in results),normal_passed=sum(r['success'] for r in results if r['case']['control']=='normal'),
        negative_scheduled=sum(r['case']['control']!='normal' for r in results),negative_passed=sum(r['success'] for r in results if r['case']['control']!='normal'),
        wall_seconds=sum(r['wall_seconds'] for r in results),results=results,protocol_sha256=sha256((root/'protocol.json').read_bytes()).hexdigest(),
        runtime=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,mujoco=mj.__version__),
        scope=protocol['scope'],physics_calibrated=False,learned_skill=False,physical_robot_ready=False)
    from pxr import Usd
    summary['runtime']['openusd']=list(Usd.GetVersion())
    summary['independent_native_goal']={kind+'_'+control:dict(scheduled=sum(r['case']['kind']==kind and r['case']['control']==control for r in results),
        settled_at_recipe_goal=sum(r['case']['kind']==kind and r['case']['control']==control and bool(r['native_goal']['success']) for r in results))
        for kind,control in sorted({(r['case']['kind'],r['case']['control']) for r in results})}
    summary['independent_native_goal_scope']='Separate from passive physics checks. The recipe goal remains at the initial XY; successful sliding can correctly end outside that goal.'
    write(root/'report.json',summary)
    scene['geometry_binary']=dict(href='geometry.bin.gz',compression='gzip',**identity(geometry))
    write(root/'model.json',scene);write(root/'geometry.bin.gz',gzip.compress(geometry,mtime=0))
    motion=dict(schema=SCHEMA,body_count=len(scene['bodies']),cases=[])
    for result in results:
        folder=root/'cases'/result['case']['id'];rows=json.loads((folder/'trajectory.json').read_text())['rows'];packed=(folder/'poses.bin.gz').read_bytes();raw=gzip.decompress(packed)
        motion['cases'].append(dict(case=result['case'],result=result,rows=rows,pose_file='cases/'+result['case']['id']+'/poses.bin.gz',pose_identity=identity(raw)))
    write(root/'motion.json',motion)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run'))
    p.add_argument('--output',required=True);p.add_argument('--example');p.add_argument('--assets');a=p.parse_args(argv)
    if a.command=='prepare':
        if not a.example:p.error('prepare requires --example')
        result=prepare(a.example,a.output)
    else:
        if not a.assets:p.error('run requires --assets')
        result=run(a.output,a.assets)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','results','source')},indent=2))

if __name__=='__main__':main()
