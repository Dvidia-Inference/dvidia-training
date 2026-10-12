from dataclasses import replace
from hashlib import sha256
import gzip,json,math,pathlib,shutil
from dvidia_training import piper_task as task

ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path(task.__file__).parent

def yaw(q,degrees):
    a=math.radians(degrees)/2;c,s=math.cos(a),math.sin(a)
    w,x,y,z=q
    return(c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w)

protocol=json.loads(pathlib.Path('benchmarks/piper-recovery-v04-20261011/receipt/protocol.json').read_text())
scenes=[s for s in protocol['scenes'] if (s['friction'],s['mass_kg']) in ((.55,.06),(.8,.04))]
plan={'purpose':'Development exploration on already-published v0.4 scenes; fixed world yaw tool-only grasp candidates; carry/initial approach keeps the original pose; unchanged native physics/scorer and v0.4 controller.', 'source_protocol_sha256':sha256(pathlib.Path('benchmarks/piper-recovery-v04-20261011/receipt/protocol.json').read_bytes()).hexdigest(),'scene_ids':[s['id'] for s in scenes],'candidate_yaw_degrees':[45,-45],'planned_attempts':len(scenes)*3,'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('piper_*.py')}}
(ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(ROOT/'sources').mkdir()
for p in SOURCE.glob('piper_*.py'): shutil.copyfile(p, ROOT/'sources'/p.name)
summary=[]
for yaw_degrees in plan['candidate_yaw_degrees']:
  for scene in scenes:
    base=task.PiperTaskConfig(**scene['config'])
    cfg=replace(base,tool_orientation_wxyz=yaw(base.tool_orientation_wxyz,yaw_degrees),carry_orientation_wxyz=base.carry_orientation_wxyz)
    result=task.run_episode('runs/robot-profile-upstream/assets',cfg,seed=scene['seed'],policy_mode='recovery',record_step_timings=True)
    name=scene['id']+'-yaw'+str(yaw_degrees)+'.json.gz'
    with gzip.open(ROOT/name,'wt') as f:json.dump(result,f,separators=(',',':'),allow_nan=False)
    summary.append({'file':name,'scene_id':scene['id'],'yaw_degrees':yaw_degrees,'valid':result['valid'],'success':result['success'],'reason':result['reason'],'metrics':result['metrics'],'grip_guard':result['grip_guard'],'error':result['error'],'maximum_object_rise_m':max(o['object_position_m'][2]-result['initial_object_position_m'][2] for o in result['trajectory'])})
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(name,result['valid'],result['success'],result['grip_guard']['abort_reason'],round(summary[-1]['maximum_object_rise_m'],5),flush=True)
