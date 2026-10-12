from hashlib import sha256
from unittest.mock import patch
import gzip,json,pathlib,shutil
from dvidia_training import piper_task as task, piper_contact_recovery as recovery, piper_grasp_strategy as grasp

ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path(task.__file__).parent
protocol=json.loads(pathlib.Path('benchmarks/piper-recovery-v04-20261011/receipt/protocol.json').read_text())
scenes=[s for s in protocol['scenes'] if (s['friction'],s['mass_kg']) in ((.55,.06),(.8,.04))]
plan={'purpose':'Development on already-published v0.4 scenes. Fixed tool-only world yaw candidates, original carry/initial approach. Candidate-only orientation-error clamp .20 rad. Read-through policy view preserves world config/model, native mechanics and scorer.', 'source_protocol_sha256':sha256(pathlib.Path('benchmarks/piper-recovery-v04-20261011/receipt/protocol.json').read_bytes()).hexdigest(),'scene_ids':[s['id'] for s in scenes],'candidate_yaw_degrees':[0,-45],'planned_attempts':len(scenes)*2,'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('piper_*.py')}}
(ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(ROOT/'sources').mkdir()
for p in SOURCE.glob('piper_*.py'):shutil.copyfile(p,ROOT/'sources'/p.name)
original=recovery.ContactRecoveryPiperTaskPolicy
summary=[]
for yaw in plan['candidate_yaw_degrees']:
 strategy=grasp.GraspStrategyConfig(yaw_degrees=yaw)
 for scene in scenes:
    def factory(env,mode):return original(grasp.GraspPlanningEnvironment(env,strategy),mode)
    with patch.object(recovery,'ContactRecoveryPiperTaskPolicy',factory):
      result=task.run_episode('runs/robot-profile-upstream/assets',task.PiperTaskConfig(**scene['config']),seed=scene['seed'],policy_mode='recovery',record_step_timings=True)
    result['experimental_grasp_strategy']=grasp.grasp_strategy_contract(strategy)
    result['policy_implementation_sha256']['piper_grasp_strategy.py']=plan['source_sha256']['piper_grasp_strategy.py']
    name=scene['id']+'-yaw'+str(yaw)+'.json.gz'
    with gzip.open(ROOT/name,'wt') as f:json.dump(result,f,separators=(',',':'),allow_nan=False)
    summary.append({'file':name,'scene_id':scene['id'],'yaw_degrees':yaw,'valid':result['valid'],'success':result['success'],'reason':result['reason'],'metrics':result['metrics'],'grip_guard':result['grip_guard'],'error':result['error'],'maximum_object_rise_m':max(o['object_position_m'][2]-result['initial_object_position_m'][2] for o in result['trajectory'])})
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(name,result['valid'],result['success'],result['grip_guard']['abort_reason'],round(summary[-1]['maximum_object_rise_m'],5),flush=True)
