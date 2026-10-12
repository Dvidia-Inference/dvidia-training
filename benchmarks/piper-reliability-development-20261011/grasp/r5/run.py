from hashlib import sha256
from unittest.mock import patch
import gzip,json,pathlib,shutil
from dvidia_training import piper_task as task, piper_grasp_strategy as grasp

ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path(task.__file__).parent
protocol=json.loads(pathlib.Path('benchmarks/piper-recovery-v04-20261011/receipt/protocol.json').read_text())
scenes=[s for s in protocol['scenes'] if (s['friction'],s['mass_kg']) in ((.55,.06),(.8,.04))]
plan={'purpose':'Development on already-published v0.4 scenes. Fixed -45deg tool yaw; preserve original approach then retain yaw through carry. Full reacquisition controller; candidate-only exact grasp-target orientation-error clamps .05/.10 rad; all other poses original .05. Native world, task scorer and20s budget unchanged.', 'source_protocol_sha256':sha256(pathlib.Path('benchmarks/piper-recovery-v04-20261011/receipt/protocol.json').read_bytes()).hexdigest(),'scene_ids':[s['id'] for s in scenes],'candidate_orientation_limits':[.05,.10],'planned_attempts':len(scenes)*2,'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('piper_*.py')}}
(ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(ROOT/'sources').mkdir()
for p in SOURCE.glob('piper_*.py'):shutil.copyfile(p,ROOT/'sources'/p.name)
original=grasp.GraspReacquisitionPiperTaskPolicy
summary=[]
for clamp in plan['candidate_orientation_limits']:
 strategy=grasp.GraspStrategyConfig(yaw_degrees=-45,orientation_error_limit_rad=clamp,retain_carry_yaw=True)
 for scene in scenes:
    def factory(env,mode):return original(env,mode,strategy)
    with patch.object(grasp,'GraspReacquisitionPiperTaskPolicy',factory):
      result=task.run_episode('runs/robot-profile-upstream/assets',task.PiperTaskConfig(**scene['config']),seed=scene['seed'],policy_mode='reacquire_grasp',record_step_timings=True,record_policy_inputs=True)
    result['experimental_grasp_strategy']=grasp.grasp_strategy_contract(strategy)
    result['loaded_source_sha256']=plan['source_sha256']
    name=scene['id']+'-clamp'+str(clamp)+'.json.gz'
    with gzip.open(ROOT/name,'wt') as f:json.dump(result,f,separators=(',',':'),allow_nan=False)
    summary.append({'file':name,'scene_id':scene['id'],'orientation_error_limit_rad':clamp,'valid':result['valid'],'success':result['success'],'reason':result['reason'],'metrics':result['metrics'],'grip_guard':result['grip_guard'],'error':result['error'],'maximum_object_rise_m':max(o['object_position_m'][2]-result['initial_object_position_m'][2] for o in result['trajectory']) if result['trajectory'] else None})
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(name,result['valid'],result['success'],result['grip_guard']['abort_reason'],summary[-1]['maximum_object_rise_m'],flush=True)
