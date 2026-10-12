from hashlib import sha256
import gzip,json,pathlib,shutil
from dvidia_training import piper_task as task, piper_grasp_strategy as grasp

ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path(task.__file__).parent
protocol=json.loads(pathlib.Path('benchmarks/piper-recovery-development-20261011/paired-benchmark/protocol.json').read_text())
scenes=protocol['scenes']
plan={'purpose':'Frozen-parameter historical regression across all12 seed813407 scenes after selection on published941827 development scenes, integrated reacquire_grasp. Experimental selected fixed -45deg tool yaw/.10rad exact-grasp IK cap/original carry yaw, no hidden material or scoring input. Full20s native episodes.', 'source_protocol_sha256':sha256(pathlib.Path('benchmarks/piper-recovery-development-20261011/paired-benchmark/protocol.json').read_bytes()).hexdigest(),'scene_ids':[s['id'] for s in scenes],'strategy_contract':grasp.combined_contract(),'planned_attempts':len(scenes),'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('piper_*.py')}}
(ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(ROOT/'sources').mkdir()
for p in SOURCE.glob('piper_*.py'):shutil.copyfile(p,ROOT/'sources'/p.name)
summary=[]
for scene in scenes:
    result=task.run_episode('runs/robot-profile-upstream/assets',task.PiperTaskConfig(**scene['config']),seed=scene['seed'],policy_mode='reacquire_grasp',record_step_timings=True,record_policy_inputs=True)
    result['loaded_source_sha256']=plan['source_sha256']
    name=scene['id']+'.json.gz'
    with gzip.open(ROOT/name,'wt') as f:json.dump(result,f,separators=(',',':'),allow_nan=False)
    summary.append({'file':name,'scene_id':scene['id'],'valid':result['valid'],'success':result['success'],'reason':result['reason'],'metrics':result['metrics'],'grip_guard':result['grip_guard'],'error':result['error'],'maximum_object_rise_m':max(o['object_position_m'][2]-result['initial_object_position_m'][2] for o in result['trajectory']) if result['trajectory'] else None})
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(name,result['valid'],result['success'],result['grip_guard']['abort_reason'],summary[-1]['maximum_object_rise_m'],flush=True)
