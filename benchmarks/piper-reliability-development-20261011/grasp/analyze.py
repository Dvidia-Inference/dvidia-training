"""Summarize every recorded grasp exploration and paired published regressions."""
from hashlib import sha256
import gzip,json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parent
REPO=ROOT.parents[2]
rounds=[]
for folder in sorted(ROOT.glob('r[0-9]*')):
 if not (folder/'summary.json').exists():continue
 rows=json.loads((folder/'summary.json').read_text())
 rounds.append({'round':folder.name,'attempts':len(rows),'valid':sum(r['valid'] for r in rows),'successes':sum(r['success'] for r in rows),'prohibited_contact_attempts':sum(bool(r['metrics'].get('unexpected_contact_native_ticks')) for r in rows),'warning_attempts':sum(bool(r['metrics'].get('warnings')) for r in rows),'planned_attempts':json.loads((folder/'plan.json').read_text())['planned_attempts']})
comparisons=[]
for round_name,baseline_relative in [('r6','benchmarks/piper-recovery-v04-20261011/receipt'),('r7','benchmarks/piper-recovery-development-20261011/paired-benchmark')]:
 folder=ROOT/round_name
 if not (folder/'summary.json').exists():continue
 samples=[]
 for entry in json.loads((folder/'summary.json').read_text()):
  scene=entry['scene_id']
  new=json.load(gzip.open(folder/entry['file']))
  old=json.load(gzip.open(REPO/baseline_relative/'attempts'/(scene+'--recovery_placement.json.gz')))
  equal={k:new[k]==old[k] for k in ('config','compiled_model_sha256','native_servo_gains','native_servo_bias','native_force_limits','native_control_limits','native_model_settings','initial_object_position_m')}
  equal['initial_qpos']=new['trajectory'][0]['qpos']==old['trajectory'][0]['qpos']
  equal['initial_qvel']=new['trajectory'][0]['qvel']==old['trajectory'][0]['qvel']
  common=new['success'] and old['success']
  probes=[]
  for kind,r in [('baseline',old),('candidate',new)]:
   events=[e for e in r['grip_guard']['events'] if e['type']=='probe_started']
   if events:
    t=events[0]['t_seconds']; window=[o for o in r['trajectory'] if t-.10-1e-9<=o['t_seconds']<=t+1e-9]
    probes.append({'mode':kind,'first_probe_time_s':t,'mean_single_jaw_position_m':statistics.mean(o['gripper_joint_position_m'][0] for o in window),'mean_sum_pad_normal_force_n':statistics.mean(sum(o['pad_normal_forces_n'].values()) for o in window),'mean_min_pad_normal_force_n':statistics.mean(min(o['pad_normal_forces_n'].values()) for o in window)})
  samples.append({'scene_id':scene,'baseline_success':old['success'],'candidate_success':new['success'],'gained':new['success'] and not old['success'],'lost':old['success'] and not new['success'],'equal_world_and_start':equal,'candidate_prohibited_native_ticks':new['metrics']['unexpected_contact_native_ticks'],'candidate_warning_count':len(new['warnings']),'baseline_completion_s':old['metrics']['first_success_time_seconds'],'candidate_completion_s':new['metrics']['first_success_time_seconds'],'paired_success_completion_delta_s':new['metrics']['first_success_time_seconds']-old['metrics']['first_success_time_seconds'] if common else None,'candidate_second_probe_count':new['grip_guard'].get('second_probe_count'),'candidate_abort_reason':new['grip_guard']['abort_reason'],'probe_force_windows':probes})
 comparisons.append({'round':round_name,'scenes':samples,'baseline_successes':sum(s['baseline_success'] for s in samples),'candidate_successes':sum(s['candidate_success'] for s in samples),'gains':sum(s['gained'] for s in samples),'losses':sum(s['lost'] for s in samples),'all_world_and_initial_state_fields_equal':all(all(s['equal_world_and_start'].values()) for s in samples),'mean_common_success_delta_s':statistics.mean([s['paired_success_completion_delta_s'] for s in samples if s['paired_success_completion_delta_s'] is not None])})
mismatches=[]
for directory in sorted(ROOT.glob('r[0-9]*')):
 for p in sorted(directory.glob('*.json.gz')):
  r=json.load(gzip.open(p))
  for field in ('implementation_sha256','policy_implementation_sha256'):
   for name,digest in r.get(field,{}).items():
    source=directory/'sources'/name
    if source.exists() and sha256(source.read_bytes()).hexdigest()!=digest:
     mismatches.append({'round':directory.name,'trace':p.name,'file':name,'recorded_disk_sha256':digest,'startup_snapshot_sha256':sha256(source.read_bytes()).hexdigest()})
analysis={'scope':'Development only, including failed orientation/IK alternatives. r6 selected on already-published seed941827 scenes; r7 frozen-parameter regression on historical813407. Not independent fresh confirmation. No parameters tuned after r7. Prior scenes may not be called unseen.', 'rounds':rounds,'attempts':sum(r['attempts'] for r in rounds),'valid':sum(r['valid'] for r in rounds),'successes':sum(r['successes'] for r in rounds),'comparisons':comparisons,'source_hash_mismatches':mismatches,'r1_plan_count_correction':'r1 declared2candidates x4scenes=8 before any attempt but its planned_attempts formula incorrectly retained12; see r1/plan-count-correction.json. No rows omitted.'}
(ROOT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
print(json.dumps({'rounds':rounds,'comparisons':[{k:v for k,v in x.items() if k!='scenes'} for x in comparisons],'source_hash_mismatches':len(mismatches)},indent=2))
